#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <numeric>
// 目的: 为茶叶病害检测任务提供可复现、可配置且更健壮的数据增强流水线。
// 描述: 包含配置结构、边界框表示、标注解析、图像增强策略（几何与颜色）、
//       边界框保护/过滤、并行处理、以及增强结果保存与统计。

using namespace std;
namespace fs = std::filesystem;
using namespace std::chrono;

// ==================== 1. 优化的配置结构 ====================
// 说明:
// - `AugmentationConfig` 汇总所有增强相关参数，便于集中管理和从文件加载。
// - 将常用概率、范围、边界框过滤等参数拆分，方便针对茶叶病害调优。
// - 该配置旨在在保持多样性的同时，避免生成极端或无效的边界框。
struct AugmentationConfig {
    // 基础参数
    int augmentations_per_image = 5;
    int num_workers = 4;
    int target_size = 640;
    bool save_original = true;
    
    // 边界框过滤参数 - 针对茶叶病害优化
    float min_bbox_area = 0.0008f;    // 最小边界框面积
    float min_bbox_width = 0.015f;    // 最小宽度
    float min_bbox_height = 0.015f;    // 最小高度
    bool remove_small_bboxes = true;  // 移除小边界框
    
    // 几何变换概率
    struct {
        float rotation = 0.6f;
        float scale = 0.7f;
        float translation = 0.3f;
        float flip = 0.5f;
        float shear = 0.2f;
        float perspective = 0.1f;
    } probs;
    
    // 几何变换范围
    struct {
        float rotation_range = 25.0f;
        float min_scale = 0.8f;      // 提高最小值，避免过度缩小
        float max_scale = 1.3f;
        float translate_range = 0.08f; // 减少平移范围
        float shear_range = 8.0f;
        float perspective_range = 0.001f;
    } ranges;
    
    // 颜色变换概率
    struct {
        float hsv = 0.8f;
        float blur = 0.3f;
        float noise = 0.2f;
        float contrast = 0.4f;
        float brightness = 0.4f;
    } color_probs;
    
    // 高级增强
    struct {
        float mosaic_prob = 0.3f;
        float mixup_prob = 0.2f;
        bool enable_advanced_aug = true;
    } advanced;
    
    bool loadFromFile(const string& path) {
        ifstream file(path);
        if (!file.is_open()) return false;
        
        string line;
        // 逐行解析键值对格式的简单配置文件，忽略注释行和空行
        while (getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            size_t eq_pos = line.find('=');
            if (eq_pos == string::npos) continue;
            
            string key = line.substr(0, eq_pos);
            key.erase(remove_if(key.begin(), key.end(), ::isspace), key.end());
            string value_str = line.substr(eq_pos + 1);
            value_str.erase(remove_if(value_str.begin(), value_str.end(), ::isspace), value_str.end());
            
            try {
                float value = stof(value_str);
                
                // 基础参数
                if (key == "augmentations_per_image") augmentations_per_image = value;
                else if (key == "num_workers") num_workers = value;
                else if (key == "target_size") target_size = value;
                
                // 边界框过滤参数
                else if (key == "min_bbox_area") min_bbox_area = value;
                else if (key == "min_bbox_width") min_bbox_width = value;
                else if (key == "min_bbox_height") min_bbox_height = value;
                
                // 几何变换概率
                else if (key == "rotation_prob") probs.rotation = value;
                else if (key == "scale_prob") probs.scale = value;
                else if (key == "translation_prob") probs.translation = value;
                else if (key == "flip_prob") probs.flip = value;
                else if (key == "shear_prob") probs.shear = value;
                else if (key == "perspective_prob") probs.perspective = value;
                
                // 几何变换范围
                else if (key == "rotation_range") ranges.rotation_range = value;
                else if (key == "min_scale") ranges.min_scale = value;
                else if (key == "max_scale") ranges.max_scale = value;
                else if (key == "translate_range") ranges.translate_range = value;
                
            } catch (...) {}
        }
        return true;
    }
};

// ==================== 2. 优化的边界框类 ====================
// 说明:
// - `BoundingBox` 使用YOLO风格的中心点归一化坐标 (x_center, y_center, width, height)，
//   并提供一系列工具函数用于验证、修复及坐标变换（例如几何变换后的角点更新）。
class BoundingBox {
public:
    int class_id = 0;
    float x_center = 0.5f;
    float y_center = 0.5f;
    float width = 0.1f;
    float height = 0.1f;
    
    BoundingBox() = default;
    BoundingBox(int cls, float x, float y, float w, float h) 
        : class_id(cls), x_center(clamp(x)), y_center(clamp(y)), width(clamp(w)), height(clamp(h)) {}
    
    // 将归一化坐标约束到 [0,1]，防止数值越界
    static float clamp(float val) {
        if (val < 0.0f) return 0.0f;
        if (val > 1.0f) return 1.0f;
        return val;
    }
    
    // 边界框尺寸验证方法
    // 用于判断边界框是否达到最小宽/高/面积阈值，针对茶叶病害通常较小目标需要更敏感的阈值
    bool isValidSize(float min_width = 0.015f, float min_height = 0.015f, float min_area = 0.0008f) const {
        float area = width * height;
        return (width >= min_width && height >= min_height && area >= min_area);
    }
    
    // 边界框修复方法
    // 当边界框小于最小尺寸时，将宽/高扩展到最小值并保持中心点不变（归一化后）
    void adjustToMinSize(float min_width = 0.015f, float min_height = 0.015f) {
        if (width < min_width) {
            float expand = (min_width - width) / 2;
            x_center = clamp(x_center);
            width = min_width;
        }
        if (height < min_height) {
            float expand = (min_height - height) / 2;
            y_center = clamp(y_center);
            height = min_height;
        }
    }
    
    // 转换为YOLO文件行格式: [class x_center y_center width height]
    vector<float> toYOLOFormat() const {
        return {static_cast<float>(class_id), x_center, y_center, width, height};
    }
    
    // 计算边界框的四个角点（像素坐标），用于在仿射/透视变换后重计算归一化的中心及尺寸
    vector<cv::Point2f> getCorners(const cv::Size& img_size) const {
        float x_min = (x_center - width/2) * img_size.width;
        float y_min = (y_center - height/2) * img_size.height;
        float x_max = (x_center + width/2) * img_size.width;
        float y_max = (y_center + height/2) * img_size.height;
        
        return {
            cv::Point2f(x_min, y_min),
            cv::Point2f(x_max, y_min),
            cv::Point2f(x_max, y_max),
            cv::Point2f(x_min, y_max)
        };
    }
    
    // 从角点更新边界框
    // 接收变换后的角点（像素坐标），并重新计算归一化的中心、宽高，最后执行范围裁剪
    void updateFromCorners(const vector<cv::Point2f>& corners, const cv::Size& img_size) {
        if (corners.size() < 4) return;
        
        float x_min = min({corners[0].x, corners[1].x, corners[2].x, corners[3].x});
        float x_max = max({corners[0].x, corners[1].x, corners[2].x, corners[3].x});
        float y_min = min({corners[0].y, corners[1].y, corners[2].y, corners[3].y});
        float y_max = max({corners[0].y, corners[1].y, corners[2].y, corners[3].y});
        
        x_center = ((x_min + x_max) / 2) / img_size.width;
        y_center = ((y_min + y_max) / 2) / img_size.height;
        width = (x_max - x_min) / img_size.width;
        height = (y_max - y_min) / img_size.height;
        
        // 确保坐标在有效范围内
        x_center = clamp(x_center);
        y_center = clamp(y_center);
        width = clamp(width);
        height = clamp(height);
    }
};

// ==================== 3. 标注类 ====================
struct Annotation {
    string id;
    string image_path;
    cv::Mat image;
    vector<BoundingBox> bboxes;
    cv::Size original_size;
    
    // 检查有效性
    bool isValid() const {
        if (image.empty()) return false;
        if (bboxes.empty()) return false;
        
        for (const auto& bbox : bboxes) {
            if (!bbox.isValidSize()) {
                return false;
            }
        }
        return true;
    }
    
    // 过滤小边界框
    void filterSmallBBoxes(float min_width = 0.015f, float min_height = 0.015f, float min_area = 0.0008f) {
        vector<BoundingBox> filtered;
        for (const auto& bbox : bboxes) {
            if (bbox.isValidSize(min_width, min_height, min_area)) {
                filtered.push_back(bbox);
            }
        }
        bboxes = filtered;
    }
};

// ==================== 4. 边界框处理器 ====================
class BBoxProcessor {
public:
    static vector<BoundingBox> filterBBoxes(const vector<BoundingBox>& bboxes, 
                                           float min_area = 0.0008f,
                                           float min_width = 0.015f, 
                                           float min_height = 0.015f) {
        vector<BoundingBox> filtered;
        for (const auto& bbox : bboxes) {
            if (bbox.isValidSize(min_width, min_height, min_area)) {
                filtered.push_back(bbox);
            }
        }
        return filtered;
    }
    
    static void clampBBoxCoordinates(vector<BoundingBox>& bboxes) {
        for (auto& bbox : bboxes) {
            bbox.x_center = BoundingBox::clamp(bbox.x_center);
            bbox.y_center = BoundingBox::clamp(bbox.y_center);
            bbox.width = BoundingBox::clamp(bbox.width);
            bbox.height = BoundingBox::clamp(bbox.height);
        }
    }
};

// ==================== 5. 标注解析器 ====================
class AnnotationParser {
public:
    static bool parseYOLO(const string& txt_path, Annotation& ann) {
        ifstream file(txt_path);
        if (!file.is_open()) {
            cerr << "无法打开标注文件: " << txt_path << endl;
            return false;
        }
        
        string line;
        vector<BoundingBox> bboxes;
        
        while (getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            istringstream iss(line);
            vector<string> tokens;
            string token;
            
            while (iss >> token) tokens.push_back(token);
            
            if (tokens.size() < 5) {
                cerr << "标注格式错误: " << txt_path << endl;
                continue;
            }
            
            try {
                int class_id = stoi(tokens[0]);
                float x_center = stof(tokens[1]);
                float y_center = stof(tokens[2]);
                float width = stof(tokens[3]);
                float height = stof(tokens[4]);
                
                // 确保坐标在有效范围内
                x_center = BoundingBox::clamp(x_center);
                y_center = BoundingBox::clamp(y_center);
                width = BoundingBox::clamp(width);
                height = BoundingBox::clamp(height);
                
                bboxes.push_back(BoundingBox(class_id, x_center, y_center, width, height));
            } catch (const exception& e) {
                cerr << "解析错误: " << e.what() << " in " << txt_path << endl;
                continue;
            }
        }
        
        if (bboxes.empty()) {
            cerr << "没有有效的边界框: " << txt_path << endl;
            return false;
        }
        
        ann.bboxes = bboxes;
        return true;
    }
};

// ==================== 6. 优化的增强器类 ====================
// 说明:
// - `AdvancedAugmentor` 使用 `AugmentationConfig` 中的概率/范围来对图像进行多种
//   随机增强（比例、旋转、平移、错切、透视、翻转，及 HSV/模糊/噪声/对比度/亮度等）。
// - 对每种几何变换，在变换后通过角点或中心变换来更新边界框，尽量保证标注保持一致。
// - 最后会运行边界框保护逻辑：根据配置决定是移除过小bbox还是调整到最小尺寸。
class AdvancedAugmentor {
    AugmentationConfig config_;
    mt19937 rng_;
    cv::RNG cv_rng_;
    
public:
    // 构造函数: 支持传入随机种子以便复现
    AdvancedAugmentor(const AugmentationConfig& config, int seed = 0) : config_(config) {
        rng_ = (seed == 0) ? mt19937(random_device{}()) : mt19937(seed);
        cv_rng_ = cv::RNG(seed);
    }
    
    // augment: 对单张样本进行一系列随机增强，并返回增强后的样本
    // 实现细节: 先做几何变换（按概率），然后做颜色变换（按概率），最后保护/过滤bbox
    Annotation augment(const Annotation& input) {
        Annotation result = input;
        input.image.copyTo(result.image);
        
        uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        // 应用几何变换（顺序可影响结果）
        if (dist(rng_) < config_.probs.scale) applyScale(result);
        if (dist(rng_) < config_.probs.rotation) applyRotation(result);
        if (dist(rng_) < config_.probs.translation) applyTranslation(result);
        if (dist(rng_) < config_.probs.shear) applyShear(result);
        if (dist(rng_) < config_.probs.perspective) applyPerspective(result);
        if (dist(rng_) < config_.probs.flip) applyFlip(result);
        
        // 应用颜色变换
        if (dist(rng_) < config_.color_probs.hsv) applyHSV(result);
        if (dist(rng_) < config_.color_probs.blur) applyBlur(result);
        if (dist(rng_) < config_.color_probs.noise) applyNoise(result);
        if (dist(rng_) < config_.color_probs.contrast) applyContrast(result);
        if (dist(rng_) < config_.color_probs.brightness) applyBrightness(result);
        
        // 保护边界框（过滤或调整）
        protectBoundingBoxes(result);
        
        return result;
    }
    
private:
    // 根据配置移除或调整过小的边界框，并做坐标裁剪
    void protectBoundingBoxes(Annotation& data) {
        if (config_.remove_small_bboxes) {
            data.filterSmallBBoxes(config_.min_bbox_width, config_.min_bbox_height, config_.min_bbox_area);
        } else {
            // 调整过小的边界框到最小尺寸（保留样本，但避免无效小框）
            for (auto& bbox : data.bboxes) {
                if (!bbox.isValidSize(config_.min_bbox_width, config_.min_bbox_height, config_.min_bbox_area)) {
                    bbox.adjustToMinSize(config_.min_bbox_width, config_.min_bbox_height);
                }
            }
        }
        
        // 最终把所有坐标约束到 [0,1]
        BBoxProcessor::clampBBoxCoordinates(data.bboxes);
    }
    
    // 以下为各类增强函数：它们在修改图像后，尽量同步更新 bbox 信息（通过中心变换或角点变换）
    void applyScale(Annotation& data) {
        uniform_real_distribution<float> scale_dist(config_.ranges.min_scale, 
                                                   config_.ranges.max_scale);
        float scale = scale_dist(rng_);
        
        cv::Mat scaled;
        cv::resize(data.image, scaled, cv::Size(), scale, scale);
        
        // 将缩放后的图像居中放回原尺寸画布，避免尺寸变化导致后续代码复杂化
        cv::Mat new_img = cv::Mat::zeros(data.image.size(), data.image.type());
        int x_offset = max(0, (data.image.cols - scaled.cols) / 2);
        int y_offset = max(0, (data.image.rows - scaled.rows) / 2);
        
        if (x_offset + scaled.cols <= new_img.cols && y_offset + scaled.rows <= new_img.rows) {
            scaled.copyTo(new_img(cv::Rect(x_offset, y_offset, scaled.cols, scaled.rows)));
        }
        
        data.image = new_img;
        
        // 更新边界框中心与尺寸（归一化坐标系）
        for (auto& bbox : data.bboxes) {
            bbox.x_center = (bbox.x_center - 0.5f) * scale + 0.5f;
            bbox.y_center = (bbox.y_center - 0.5f) * scale + 0.5f;
            bbox.width *= scale;
            bbox.height *= scale;
        }
    }

    void applyRotation(Annotation& data) {
        uniform_real_distribution<float> angle_dist(-config_.ranges.rotation_range, 
                                                   config_.ranges.rotation_range);
        float angle = angle_dist(rng_);

        cv::Point2f center(data.image.cols / 2.0f, data.image.rows / 2.0f);
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);

        cv::Mat rotated;
        cv::warpAffine(data.image, rotated, rot_mat, data.image.size(),
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
        data.image = rotated;

        // 使用角点变换来更新每个 bbox，能更准确地应对旋转后的外接矩形
        for (auto& bbox : data.bboxes) {
            auto corners = bbox.getCorners(data.original_size);
            vector<cv::Point2f> rotated_corners;
            cv::transform(corners, rotated_corners, rot_mat);
            bbox.updateFromCorners(rotated_corners, data.original_size);
        }
    }

    void applyTranslation(Annotation& data) {
        uniform_real_distribution<float> trans_dist(-config_.ranges.translate_range, 
                                                   config_.ranges.translate_range);
        float dx = trans_dist(rng_);
        float dy = trans_dist(rng_);

        cv::Mat trans_mat = (cv::Mat_<double>(2, 3) << 
            1, 0, dx * data.image.cols,
            0, 1, dy * data.image.rows
        );

        cv::Mat translated;
        cv::warpAffine(data.image, translated, trans_mat, data.image.size(),
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
        data.image = translated;

        // 平移直接作用于归一化中心坐标
        for (auto& bbox : data.bboxes) {
            bbox.x_center += dx;
            bbox.y_center += dy;
        }
    }

    void applyShear(Annotation& data) {
        uniform_real_distribution<float> shear_dist(-config_.ranges.shear_range, 
                                                   config_.ranges.shear_range);
        float shear_x = shear_dist(rng_);
        float shear_y = shear_dist(rng_);

        cv::Mat shear_mat = (cv::Mat_<double>(2, 3) << 
            1, shear_x, 0,
            shear_y, 1, 0
        );

        cv::Mat sheared;
        cv::warpAffine(data.image, sheared, shear_mat, data.image.size(),
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
        data.image = sheared;

        for (auto& bbox : data.bboxes) {
            auto corners = bbox.getCorners(data.original_size);
            vector<cv::Point2f> sheared_corners;
            cv::transform(corners, sheared_corners, shear_mat);
            bbox.updateFromCorners(sheared_corners, data.original_size);
        }
    }

    void applyPerspective(Annotation& data) {
        uniform_real_distribution<float> persp_dist(0.0f, config_.ranges.perspective_range);
        float persp_value = persp_dist(rng_);

        vector<cv::Point2f> src_points = {
            cv::Point2f(0, 0),
            cv::Point2f(data.image.cols, 0),
            cv::Point2f(data.image.cols, data.image.rows),
            cv::Point2f(0, data.image.rows)
        };

        vector<cv::Point2f> dst_points = {
            cv::Point2f(persp_value * data.image.cols, persp_value * data.image.rows),
            cv::Point2f((1 - persp_value) * data.image.cols, persp_value * data.image.rows),
            cv::Point2f((1 - persp_value) * data.image.cols, (1 - persp_value) * data.image.rows),
            cv::Point2f(persp_value * data.image.cols, (1 - persp_value) * data.image.rows)
        };

        cv::Mat persp_mat = cv::getPerspectiveTransform(src_points, dst_points);

        cv::Mat perspective;
        cv::warpPerspective(data.image, perspective, persp_mat, data.image.size(),
                           cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
        data.image = perspective;

        for (auto& bbox : data.bboxes) {
            auto corners = bbox.getCorners(data.original_size);
            vector<cv::Point2f> persp_corners;
            cv::perspectiveTransform(corners, persp_corners, persp_mat);
            bbox.updateFromCorners(persp_corners, data.original_size);
        }
    }

    void applyFlip(Annotation& data) {
        // 水平翻转: 仅需更新 x_center
        cv::flip(data.image, data.image, 1);

        for (auto& bbox : data.bboxes) {
            bbox.x_center = 1.0f - bbox.x_center;
        }
    }

    // 颜色空间与图像质量变换
    void applyHSV(Annotation& data) {
        cv::Mat hsv;
        cv::cvtColor(data.image, hsv, cv::COLOR_BGR2HSV);

        float h_gain = cv_rng_.uniform(-0.02f, 0.02f) * 180;
        float s_gain = cv_rng_.uniform(-0.5f, 0.5f);
        float v_gain = cv_rng_.uniform(-0.3f, 0.3f);

        vector<cv::Mat> channels;
        cv::split(hsv, channels);

        channels[0] = channels[0] + h_gain;
        channels[0].setTo(0, channels[0] < 0);
        channels[0].setTo(180, channels[0] > 180);

        channels[1] = channels[1] * (1.0f + s_gain);
        cv::threshold(channels[1], channels[1], 255, 255, cv::THRESH_TRUNC);
        channels[1].setTo(0, channels[1] < 0);

        channels[2] = channels[2] * (1.0f + v_gain);
        cv::threshold(channels[2], channels[2], 255, 255, cv::THRESH_TRUNC);
        channels[2].setTo(0, channels[2] < 0);

        cv::merge(channels, hsv);
        cv::cvtColor(hsv, data.image, cv::COLOR_HSV2BGR);
    }

    void applyBlur(Annotation& data) {
        int kernel_size = cv_rng_.uniform(1, 3) * 2 + 1;
        cv::GaussianBlur(data.image, data.image, cv::Size(kernel_size, kernel_size), 0);
    }

    void applyNoise(Annotation& data) {
        cv::Mat noise = cv::Mat::zeros(data.image.size(), data.image.type());
        cv::randn(noise, 0, 10);
        data.image += noise;
    }

    void applyContrast(Annotation& data) {
        float alpha = 1.0f + cv_rng_.uniform(-0.3f, 0.3f);
        data.image.convertTo(data.image, -1, alpha, 0);
    }

    void applyBrightness(Annotation& data) {
        float beta = cv_rng_.uniform(-30, 30);
        data.image.convertTo(data.image, -1, 1.0, beta);
    }
};

// ==================== 7. 数据集加载器 ====================
class TeaDiseaseDataset {
    string root_path_;
    vector<Annotation> annotations_;
    
public:
    TeaDiseaseDataset(const string& root_path) : root_path_(root_path) {}
    
    bool load() {
        string image_dir = root_path_ + "/images";
        string label_dir = root_path_ + "/labels";
        
        if (!fs::exists(image_dir) || !fs::exists(label_dir)) {
            cerr << "错误: 目录结构不正确，期望 images/ 和 labels/ 目录" << endl;
            return false;
        }
        
        cout << "加载茶叶病害数据集: " << root_path_ << endl;
        
        vector<string> image_files;
        for (const auto& entry : fs::directory_iterator(image_dir)) {
            if (isImageFile(entry.path())) {
                image_files.push_back(entry.path().string());
            }
        }
        
        if (image_files.empty()) {
            cerr << "错误: 没有图片文件" << endl;
            return false;
        }
        
        cout << "找到 " << image_files.size() << " 张图片" << endl;
        
        // 逐张图片尝试加载及对应标注文件，过滤无标注或无法解析的样本
        int loaded = 0;
        int skipped = 0;
        for (const auto& img_path : image_files) {
            fs::path path_obj(img_path);
            string stem = path_obj.stem().string();
            string txt_path = label_dir + "/" + stem + ".txt";
            
            if (!fs::exists(txt_path)) {
                cout << "跳过: " << stem << " (无标注)" << endl;
                skipped++;
                continue;
            }
            
            Annotation ann;
            ann.id = stem;
            ann.image_path = img_path;
            ann.image = cv::imread(img_path);
            
            if (ann.image.empty()) {
                cerr << "无法加载图片: " << img_path << endl;
                skipped++;
                continue;
            }
            
            ann.original_size = ann.image.size();

            if (AnnotationParser::parseYOLO(txt_path, ann)) {
                // 过滤小边界框以减少噪声样本
                ann.filterSmallBBoxes(0.015f, 0.015f, 0.0008f);

                if (!ann.bboxes.empty()) {
                    annotations_.push_back(ann);
                    loaded++;
                } else {
                    // 如果所有bbox都被过滤掉，则跳过该样本
                    cout << "样本无有效边界框: " << stem << endl;
                    skipped++;
                }

                if (loaded % 100 == 0) {
                    cout << "已加载 " << loaded << " 个有效样本" << endl;
                }
            } else {
                cout << "无效样本: " << stem << endl;
                skipped++;
            }
        }
        
        cout << "加载完成: " << loaded << " 个有效样本, " << skipped << " 个跳过" << endl;
        return loaded > 0;
    }
    
    const vector<Annotation>& getAnnotations() const { return annotations_; }
    size_t size() const { return annotations_.size(); }
    
private:
    bool isImageFile(const fs::path& path) {
        static const vector<string> exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
        string ext = path.extension().string();
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return find(exts.begin(), exts.end(), ext) != exts.end();
    }
};

// ==================== 8. 标注保存器 ====================
class AnnotationSaver {
public:
    static bool saveYOLOAnnotation(const string& path, const Annotation& ann) {
        ofstream file(path);
        if (!file.is_open()) {
            cerr << "无法保存标注: " << path << endl;
            return false;
        }
        
        file << fixed << setprecision(6);
        for (const auto& bbox : ann.bboxes) {
            auto yolo_format = bbox.toYOLOFormat();
            file << static_cast<int>(yolo_format[0]) << " " 
                 << yolo_format[1] << " " << yolo_format[2] << " " 
                 << yolo_format[3] << " " << yolo_format[4] << endl;
        }
        
        file.close();
        return true;
    }
};

// ==================== 9. 并行处理器 ====================
class ParallelProcessor {
    AugmentationConfig config_;
    string output_dir_;
    
    vector<thread> workers_;
    queue<pair<int, int>> tasks_;
    mutex task_mutex_, log_mutex_;
    condition_variable task_cv_;
    atomic<bool> stop_{false};
    atomic<int> processed_{0}, failed_{0}, total_{0};
    ofstream log_file_;
    ofstream stats_file_;
    
public:
    ParallelProcessor(const AugmentationConfig& config, const string& output_dir) 
        : config_(config), output_dir_(output_dir) {
        fs::create_directories(output_dir + "/images");
        fs::create_directories(output_dir + "/labels");
        fs::create_directories(output_dir + "/logs");
        log_file_.open(output_dir + "/logs/process.log");
        stats_file_.open(output_dir + "/logs/bbox_stats.csv");
        stats_file_ << "image_id,operation,bbox_count,avg_width,avg_height,avg_area\n";
    }
    
    ~ParallelProcessor() {
        stop_ = true;
        task_cv_.notify_all();
        for (auto& w : workers_) if (w.joinable()) w.join();
        if (log_file_.is_open()) log_file_.close();
        if (stats_file_.is_open()) stats_file_.close();
    }
    
    bool process(const string& dataset_path) {
        auto start = high_resolution_clock::now();
        
        TeaDiseaseDataset dataset(dataset_path);
        if (!dataset.load()) {
            cerr << "加载数据集失败" << endl;
            return false;
        }
        
        const auto& annotations = dataset.getAnnotations();
        if (annotations.empty()) {
            cerr << "数据集为空" << endl;
            return false;
        }
        
        // 准备任务
        int tasks_per_img = config_.augmentations_per_image;
        if (config_.save_original) tasks_per_img++;
        total_ = annotations.size() * tasks_per_img;
        
        for (size_t i = 0; i < annotations.size(); i++) {
            for (int j = 0; j < tasks_per_img; j++) {
                tasks_.push({i, j});
            }
        }
        
        cout << "总任务: " << total_ << endl;
        
        // 启动工作线程
        int num_workers = min(config_.num_workers, (int)thread::hardware_concurrency());
        cout << "启动 " << num_workers << " 个工作线程" << endl;
        
        for (int i = 0; i < num_workers; i++) {
            workers_.emplace_back([this, i, &annotations]() {
                work(i, annotations);
            });
        }
        
        // 等待完成
        wait();
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<seconds>(end - start);
        saveStats(annotations.size(), duration);
        
        return true;
    }
    
private:
    void work(int worker_id, const vector<Annotation>& annotations) {
        AdvancedAugmentor augmentor(config_, worker_id);
        
        while (!stop_) {
            pair<int, int> task = {-1, -1};
            
            {
                unique_lock<mutex> lock(task_mutex_);
                task_cv_.wait(lock, [this] { return !tasks_.empty() || stop_; });
                
                if (stop_ && tasks_.empty()) return;
                if (tasks_.empty()) continue;
                
                task = tasks_.front();
                tasks_.pop();
            }
            
            int img_idx = task.first;
            int aug_id = task.second;
            
            if (img_idx >= annotations.size()) continue;
            
            bool success = processTask(worker_id, annotations[img_idx], aug_id, augmentor);
            if (success) processed_++;
            else failed_++;
            
            if ((processed_ + failed_) % 10 == 0) {
                int total_done = processed_ + failed_;
                float progress = static_cast<float>(total_done) / total_ * 100.0f;
                lock_guard<mutex> lock(log_mutex_);
                cout << "\r进度: " << fixed << setprecision(1) << progress << "% (" 
                     << total_done << "/" << total_ << ")";
                cout.flush();
            }
        }
    }
    
    bool processTask(int worker_id, const Annotation& original, int aug_id, AdvancedAugmentor& augmentor) {
        try {
            Annotation result = original;
            string operation = "augment";
            
            if (aug_id == 0 && config_.save_original) {
                // 原始版本，不进行增强
                operation = "original";
            } else {
                int actual_aug_id = aug_id;
                if (config_.save_original) actual_aug_id--;
                result = augmentor.augment(original);
                operation = "augment_" + to_string(actual_aug_id);
            }
            
            if (!result.isValid()) {
                throw runtime_error("无效的标注");
            }
            
            // 生成文件名
            string suffix = (aug_id == 0 && config_.save_original) 
                          ? "_original" 
                          : "_aug" + to_string(aug_id);
            
            string img_name = "tea_disease_" + result.id + suffix + ".jpg";
            string label_name = "tea_disease_" + result.id + suffix + ".txt";
            
            string img_path = output_dir_ + "/images/" + img_name;
            string label_path = output_dir_ + "/labels/" + label_name;
            
            // 保存图片（JPEG，质量 95）
            vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY, 95};
            if (!cv::imwrite(img_path, result.image, compression_params)) {
                throw runtime_error("无法保存图片");
            }
            
            // 保存标注
            if (!AnnotationSaver::saveYOLOAnnotation(label_path, result)) {
                throw runtime_error("无法保存标注");
            }
            
            // 记录边界框统计信息
            logBBoxStatistics(result.id, operation, result.bboxes);
            log(worker_id, original.id, aug_id, result.bboxes.size(), true, "");
            return true;
            
        } catch (const exception& e) {
            log(worker_id, original.id, aug_id, 0, false, e.what());
            return false;
        }
    }
    
    void logBBoxStatistics(const string& img_id, const string& operation, const vector<BoundingBox>& bboxes) {
        lock_guard<mutex> lock(log_mutex_);
        if (stats_file_.is_open() && !bboxes.empty()) {
            float avg_width = 0, avg_height = 0, avg_area = 0;
            for (const auto& bbox : bboxes) {
                avg_width += bbox.width;
                avg_height += bbox.height;
                avg_area += bbox.width * bbox.height;
            }
            avg_width /= bboxes.size();
            avg_height /= bboxes.size();
            avg_area /= bboxes.size();
            
            stats_file_ << img_id << "," << operation << "," << bboxes.size() << ","
                       << avg_width << "," << avg_height << "," << avg_area << endl;
        }
    }
    
    void log(int worker_id, const string& img_id, int aug_id, 
             int bbox_count, bool success, const string& error) {
        lock_guard<mutex> lock(log_mutex_);
        if (log_file_.is_open()) {
            auto now = system_clock::now();
            auto time = system_clock::to_time_t(now);
            log_file_ << put_time(localtime(&time), "%Y-%m-%d %H:%M:%S") << ","
                     << worker_id << "," << img_id << "," << aug_id << ","
                     << bbox_count << "," << (success ? "success" : "failed");
            if (!error.empty()) log_file_ << "," << error;
            log_file_ << endl;
        }
    }
    
    void wait() {
        while (true) {
            this_thread::sleep_for(milliseconds(100));
            unique_lock<mutex> lock(task_mutex_);
            if (tasks_.empty()) {
                int done = processed_ + failed_;
                if (done >= total_) break;
            }
        }
        this_thread::sleep_for(seconds(1));
        cout << "\n处理完成!" << endl;
    }
    
    void saveStats(int dataset_size, const seconds& duration) {
        string stats_path = output_dir_ + "/logs/process_stats.txt";
        ofstream file(stats_path);
        if (!file.is_open()) return;
        
        file << "=== 茶叶病害数据集增强统计 ===" << endl << endl;
        file << "处理时间: " << duration.count() << "秒" << endl;
        file << "原始图片数量: " << dataset_size << endl;
        file << "每张图片增强数量: " << config_.augmentations_per_image << endl;
        file << "总任务数量: " << total_ << endl;
        file << "成功任务: " << processed_ << endl;
        file << "失败任务: " << failed_ << endl;
        file << "成功率: " << fixed << setprecision(1) 
             << (static_cast<float>(processed_) / total_ * 100.0f) << "%" << endl;
        file << endl << "增强配置:" << endl;
        file << "- 旋转概率: " << config_.probs.rotation << endl;
        file << "- 缩放概率: " << config_.probs.scale << " (范围: " << config_.ranges.min_scale << " - " << config_.ranges.max_scale << ")" << endl;
        file << "- 翻转概率: " << config_.probs.flip << endl;
        file << "- HSV变换概率: " << config_.color_probs.hsv << endl;
        file << "- 边界框过滤: " << (config_.remove_small_bboxes ? "移除" : "调整") << "过小边界框" << endl;
        file << "- 最小边界框尺寸: " << config_.min_bbox_width << " x " << config_.min_bbox_height << endl;
        file.close();
    }
};

// ==================== 10. 主程序 ====================
int main(int argc, char* argv[]) {
    cout << "==========================================" << endl;
    cout << "    茶叶病害数据集增强工具" << endl;
    cout << "==========================================" << endl;
    
    string input_path, output_path, config_path;
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--input" && i+1 < argc) input_path = argv[++i];
        else if (arg == "--output" && i+1 < argc) output_path = argv[++i];
        else if (arg == "--config" && i+1 < argc) config_path = argv[++i];
        else if (arg == "--help") {
            cout << "用法: ./tea_augmentor [选项]" << endl;
            cout << "选项:" << endl;
            cout << "  --input <路径>   茶叶病害数据集目录" << endl;
            cout << "  --output <路径>  输出目录" << endl;
            cout << "  --config <文件>  配置文件" << endl;
            return 0;
        }
    }
    
    // 获取输入路径
    if (input_path.empty()) {
        cout << "输入茶叶病害数据集路径: ";
        cin >> input_path;
    }
    
    // 设置输出路径
    if (output_path.empty()) {
        output_path = input_path + "_augmented";
        cout << "输出路径: " << output_path << endl;
    }
    
    // 加载配置
    AugmentationConfig config;
    if (!config_path.empty()) {
        if (!config.loadFromFile(config_path)) {
            cerr << "警告: 无法加载配置文件，使用默认配置" << endl;
        }
    } else {
        // 检查默认配置文件
        string default_config = input_path + "/tea_aug_config.txt";
        if (fs::exists(default_config)) {
            config.loadFromFile(default_config);
        } else {
            // 创建默认配置文件
            ofstream cfg(default_config);
            cfg << "# 茶叶病害数据集增强配置" << endl;
            cfg << "augmentations_per_image = 5" << endl;
            cfg << "num_workers = 4" << endl;
            cfg << "target_size = 640" << endl;
            cfg << endl << "# 边界框过滤参数" << endl;
            cfg << "min_bbox_area = 0.0008" << endl;
            cfg << "min_bbox_width = 0.015" << endl;
            cfg << "min_bbox_height = 0.015" << endl;
            cfg << endl << "# 几何变换概率" << endl;
            cfg << "rotation_prob = 0.6" << endl;
            cfg << "scale_prob = 0.7" << endl;
            cfg << "translation_prob = 0.3" << endl;
            cfg << "flip_prob = 0.5" << endl;
            cfg << endl << "# 几何变换范围" << endl;
            cfg << "rotation_range = 25.0" << endl;
            cfg << "min_scale = 0.8" << endl;
            cfg << "max_scale = 1.3" << endl;
            cfg << "translate_range = 0.08" << endl;
            cfg.close();
            cout << "已创建配置文件: " << default_config << endl;
            cout << "请根据需要调整参数后重新运行程序。" << endl;
            return 0;
        }
    }
    
    // 说明: 程序接受命令行参数以指定输入/输出路径与配置文件，若未提供将提示用户或使用默认。
    // 运行流程: 读取配置 -> 加载数据集 -> 启动并行处理器进行增强 -> 保存增强结果与统计信息
    cout << "\n开始处理茶叶病害数据集..." << endl;
    cout << "输入目录: " << input_path << endl;
    cout << "输出目录: " << output_path << endl;
    cout << "配置参数:" << endl;
    cout << "- 每张图片增强数量: " << config.augmentations_per_image << endl;
    cout << "- 边界框过滤: " << (config.remove_small_bboxes ? "移除" : "调整") << "尺寸小于 " 
         << config.min_bbox_width << "x" << config.min_bbox_height << " 的边界框" << endl;
    
    try {
        ParallelProcessor processor(config, output_path);
        if (processor.process(input_path)) {
            cout << "\n处理完成! 增强后的数据集保存到: " << output_path << endl;

            // 创建一个简单的 YAML 配置文件，方便直接在 YOLOv8 等工具中加载增强后数据集
            string yaml_path = output_path + "/tea_disease.yaml";
            ofstream yaml_file(yaml_path);
            if (yaml_file.is_open()) {
                yaml_file << "# 茶叶病害数据集配置" << endl;
                yaml_file << "path: " << output_path << endl;
                yaml_file << "train: images" << endl;
                yaml_file << "val: images" << endl;
                yaml_file << "test: " << endl << endl;
                yaml_file << "# 病害类别 (请根据实际数据集调整)" << endl;
                yaml_file << "names:" << endl;
                yaml_file << "  0: algal_spot" << endl;
                yaml_file << "  1: brown_blight" << endl;
                yaml_file << "  2: gray_blight" << endl;
                yaml_file << "  3: healthy_leaf" << endl;
                yaml_file << "  4: helopeltis" << endl;
                yaml_file << "  5: red_spot" << endl;
                yaml_file.close();
                cout << "创建数据集配置文件: " << yaml_path << endl;
            }

            cout << "\n增强统计信息已保存到: " << output_path << "/logs/" << endl;
            cout << "包含边界框尺寸分布和增强效果分析" << endl;

        } else {
            cerr << "处理失败!" << endl;
            return 1;
        }
    } catch (const exception& e) {
        cerr << "错误: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}