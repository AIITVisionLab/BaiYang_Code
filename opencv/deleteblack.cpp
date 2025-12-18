#/*************************************************************************
#  说明: 数据集清理工具 — 检测并处理全黑、暗、低质量图片及问题标注
#  用法: ./enhanced_cleaner --dataset <路径> [--dry-run] [--black-threshold <值>] ...
#  该程序会在数据集目录下生成备份和报告文件（backup_..., cleanup_*.log/csv/txt）。
#*************************************************************************/

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <map>
#include <set>
#include <cmath>

using namespace std;
namespace fs = std::filesystem;

// ==================== 1. 配置结构 ====================
/*
 * CleanupConfig 说明:
 *  - image: 与图片质量检测相关的阈值集合。
 *      - black_threshold: 判定全黑图片的灰度最大值阈值（max < threshold 即视为全黑）。
 *      - dark_threshold: 判定暗图的平均亮度阈值（mean < threshold 视为暗图）。
 *      - min_brightness / max_brightness: 用于质量评分或过滤极端亮度图像。
 *      - min_contrast: 基于像素标准差判断低对比度的阈值。
 *  - annotation: 标注有效性相关阈值（基于归一化坐标/面积/宽高）。
 *      - min_bbox_count: 每张图片应至少包含的边界框数量。
 *      - min_bbox_area/min_bbox_width/min_bbox_height: 单个 bbox 的最小尺寸限制。
 *      - max_bbox_area: 防止过大的 bbox（可能是错误标注）。
 *  - cleanup: 清理动作控制开关。
 *      - delete_images/delete_labels: 是否删除对应文件。
 *      - backup_first: 删除前是否先备份到 backup_ 目录。
 *      - move_to_trash: 是否移动到回收站而非直接删除。
 *      - dry_run: 模拟运行（不实际删除），适合先预览要删除的文件。
 *  - format: 图片基本格式与尺寸检查，用于快速过滤损坏或过小的图片。
 */
struct CleanupConfig {
    // 图片质量阈值
    struct {
        float black_threshold = 10.0f;    // 全黑图片阈值（像素最大值<10）
        float dark_threshold = 30.0f;     // 暗图阈值（像素最大值<30）
        float min_brightness = 40.0f;     // 最小平均亮度
        float min_contrast = 15.0f;       // 最小对比度
        float max_brightness = 240.0f;    // 最大平均亮度（避免过曝）
    } image;
    
    // 标注质量阈值
    struct {
        int min_bbox_count = 1;           // 最小边界框数量
        float min_bbox_area = 0.0008f;    // 最小边界框面积
        float min_bbox_width = 0.015f;    // 最小边界框宽度
        float min_bbox_height = 0.015f;   // 最小边界框高度
        float max_bbox_area = 0.8f;       // 最大边界框面积
    } annotation;
    
    // 清理选项
    struct {
        bool delete_images = true;        // 删除图片
        bool delete_labels = true;        // 删除标注
        bool backup_first = true;         // 先备份
        bool move_to_trash = false;       // 移动到回收站
        bool dry_run = false;             // 模拟运行（不实际删除）
    } cleanup;
    
    // 图片格式检查
    struct {
        bool check_empty = true;          // 检查空图片
        bool check_format = true;         // 检查格式
        bool check_size = true;           // 检查尺寸
        int min_width = 100;              // 最小宽度
        int min_height = 100;             // 最小高度
    } format;
};

// ==================== 2. 增强的图片质量检测器 ====================
/*
 * EnhancedImageDetector:
 *  - 提供一组静态方法用于检测图片常见质量问题（全黑、暗、低对比度、过曝、单色、格式异常）。
 *  - 所有检测函数均接收 `cv::Mat`，返回 bool（是否存在该问题）。
 *  - `calculateQualityScore` 给出 0-100 的质量评分，结合平均亮度与对比度用于排序/阈值判断。
 *  - 私有方法 `getAverageBrightness` 和 `getContrast` 分别计算平均灰度与像素标准差（对比度）。
 *  - 设计要点：尽量使用灰度图统计信息避免颜色通道差异带来的干扰；使用阈值参数允许外部调整灵敏度。
 */
class EnhancedImageDetector {
public:
    // 检测全黑图片（最严格检测）
    static bool isCompletelyBlack(const cv::Mat& image, float threshold = 10.0f) {
        if (image.empty()) return true;
        
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image;
        }
        
        double minVal, maxVal;
        cv::minMaxLoc(gray, &minVal, &maxVal);
        
        // 全黑：最大像素值小于阈值
        bool is_black = (maxVal < threshold);
        
        if (is_black) {
            cout << "检测到全黑图片: 像素范围 [" << minVal << ", " << maxVal << "]" << endl;
        }
        
        return is_black;
    }
    
    // 检测暗图
    static bool isDarkImage(const cv::Mat& image, float threshold = 30.0f) {
        if (image.empty()) return true;
        
        float brightness = getAverageBrightness(image);
        float contrast = getContrast(image);
        
        bool is_dark = (brightness < threshold);
        
        if (is_dark) {
            cout << "检测到暗图: 亮度=" << brightness << ", 对比度=" << contrast << endl;
        }
        
        return is_dark;
    }
    
    // 检测低对比度图片
    static bool isLowContrastImage(const cv::Mat& image, float threshold = 15.0f) {
        if (image.empty()) return true;
        
        float contrast = getContrast(image);
        bool is_low_contrast = (contrast < threshold);
        
        if (is_low_contrast) {
            cout << "检测到低对比度图片: 对比度=" << contrast << endl;
        }
        
        return is_low_contrast;
    }
    
    // 检测过曝图片
    static bool isOverExposedImage(const cv::Mat& image, float threshold = 240.0f) {
        if (image.empty()) return false;
        
        float brightness = getAverageBrightness(image);
        bool is_overexposed = (brightness > threshold);
        
        if (is_overexposed) {
            cout << "检测到过曝图片: 亮度=" << brightness << endl;
        }
        
        return is_overexposed;
    }
    
    // 检测单色图片（全灰/全白）
    static bool isMonochromeImage(const cv::Mat& image, float threshold = 5.0f) {
        if (image.empty()) return true;
        
        if (image.channels() == 3) {
            vector<cv::Mat> channels;
            cv::split(image, channels);
            
            // 计算三通道之间的差异
            cv::Mat diff1, diff2, diff3;
            cv::absdiff(channels[0], channels[1], diff1);
            cv::absdiff(channels[1], channels[2], diff2);
            cv::absdiff(channels[0], channels[2], diff3);
            
            double max_diff1, max_diff2, max_diff3;
            cv::minMaxLoc(diff1, nullptr, &max_diff1);
            cv::minMaxLoc(diff2, nullptr, &max_diff2);
            cv::minMaxLoc(diff3, nullptr, &max_diff3);
            
            double max_diff = max({max_diff1, max_diff2, max_diff3});
            bool is_monochrome = (max_diff < threshold);
            
            if (is_monochrome) {
                cout << "检测到单色图片: 最大通道差异=" << max_diff << endl;
            }
            
            return is_monochrome;
        }
        
        return false;
    }
    
    // 检测无效图片格式
    static bool isInvalidImage(const cv::Mat& image, int min_width = 100, int min_height = 100) {
        if (image.empty()) {
            cout << "图片为空" << endl;
            return true;
        }
        
        // 检查尺寸
        if (image.cols < min_width || image.rows < min_height) {
            cout << "图片尺寸过小: " << image.cols << "x" << image.rows << endl;
            return true;
        }
        
        // 检查数据类型
        if (image.depth() != CV_8U) {
            cout << "无效图片数据类型: " << image.depth() << endl;
            return true;
        }
        
        // 检查通道数
        if (image.channels() != 1 && image.channels() != 3) {
            cout << "无效通道数: " << image.channels() << endl;
            return true;
        }
        
        return false;
    }
    
    // 计算图片质量得分（0-100）
    static float calculateQualityScore(const cv::Mat& image) {
        if (image.empty()) return 0.0f;
        
        float brightness = getAverageBrightness(image);
        float contrast = getContrast(image);
        
        // 亮度得分 (40-180 最佳)
        float brightness_score = 0.0f;
        if (brightness >= 40.0f && brightness <= 180.0f) {
            brightness_score = 100.0f;
        } else if (brightness < 40.0f) {
            brightness_score = (brightness / 40.0f) * 100.0f;
        } else {
            brightness_score = ((255.0f - brightness) / 75.0f) * 100.0f;
        }
        
        // 对比度得分 (>20 最佳)
        float contrast_score = min(contrast / 20.0f * 100.0f, 100.0f);
        
        // 综合得分
        float quality_score = brightness_score * 0.4f + contrast_score * 0.6f;
        
        return quality_score;
    }
    
private:
    // 获取平均亮度
    static float getAverageBrightness(const cv::Mat& image) {
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image;
        }
        
        return cv::mean(gray)[0];
    }
    
    // 获取对比度（标准差）
    static float getContrast(const cv::Mat& image) {
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image;
        }
        
        cv::Scalar mean, stddev;
        cv::meanStdDev(gray, mean, stddev);
        return stddev[0];
    }
};

// ==================== 3. 增强的标注检测器 ====================
class EnhancedAnnotationDetector {
public:
    struct BBox {
        int class_id = 0;
        float x_center = 0.5f;
        float y_center = 0.5f;
        float width = 0.1f;
        float height = 0.1f;
        
        float area() const { return width * height; }
        
        bool isValid(float min_width = 0.015f, 
                    float min_height = 0.015f,
                    float min_area = 0.0008f,
                    float max_area = 0.8f) const {
            float bbox_area = area();
            return (x_center >= 0.0f && x_center <= 1.0f &&
                    y_center >= 0.0f && y_center <= 1.0f &&
                    width >= min_width && width <= 1.0f &&
                    height >= min_height && height <= 1.0f &&
                    bbox_area >= min_area && bbox_area <= max_area);
        }
    };
    
    // 加载标注
    static vector<BBox> loadAnnotations(const string& filepath) {
        vector<BBox> bboxes;
        ifstream file(filepath);
        
        if (!file.is_open()) {
            cout << "无法打开标注文件: " << filepath << endl;
            return bboxes;
        }
        
        string line;
        int line_num = 0;
        
        while (getline(file, line)) {
            line_num++;
            if (line.empty()) continue;
            
            try {
                BBox bbox = parseLine(line);
                bboxes.push_back(bbox);
                
                if (!bbox.isValid()) {
                    cout << "无效边界框 (行 " << line_num << "): " << line << endl;
                }
            } catch (...) {
                cout << "解析错误 (行 " << line_num << "): " << line << endl;
            }
        }
        
        return bboxes;
    }
    
    // 检查标注问题
    static bool hasProblems(const vector<BBox>& bboxes,
                           float min_width = 0.015f,
                           float min_height = 0.015f,
                           float min_area = 0.0008f,
                           float max_area = 0.8f,
                           int min_count = 1) {
        if (bboxes.empty()) {
            cout << "标注文件为空" << endl;
            return true;
        }
        
        int invalid_count = 0;
        for (const auto& bbox : bboxes) {
            if (!bbox.isValid(min_width, min_height, min_area, max_area)) {
                invalid_count++;
            }
        }
        
        if (invalid_count > 0) {
            cout << "包含 " << invalid_count << " 个无效边界框" << endl;
        }
        
        if (bboxes.size() < min_count) {
            cout << "边界框数量不足: " << bboxes.size() << " (需要 " << min_count << ")" << endl;
            return true;
        }
        
        return (invalid_count > 0);
    }
    
    // 检查重复边界框
    static bool hasDuplicates(const vector<BBox>& bboxes, float iou_threshold = 0.7f) {
        for (size_t i = 0; i < bboxes.size(); i++) {
            for (size_t j = i + 1; j < bboxes.size(); j++) {
                float iou = calculateIoU(bboxes[i], bboxes[j]);
                if (iou > iou_threshold) {
                    cout << "检测到重复边界框: IoU = " << iou << endl;
                    return true;
                }
            }
        }
        return false;
    }
    
private:
    static BBox parseLine(const string& line) {
        stringstream ss(line);
        float cls, x, y, w, h;
        ss >> cls >> x >> y >> w >> h;
        
        BBox bbox;
        bbox.class_id = static_cast<int>(cls);
        bbox.x_center = x;
        bbox.y_center = y;
        bbox.width = w;
        bbox.height = h;
        
        return bbox;
    }
    
    static float calculateIoU(const BBox& b1, const BBox& b2) {
        float x1_min = b1.x_center - b1.width/2;
        float x1_max = b1.x_center + b1.width/2;
        float y1_min = b1.y_center - b1.height/2;
        float y1_max = b1.y_center + b1.height/2;
        
        float x2_min = b2.x_center - b2.width/2;
        float x2_max = b2.x_center + b2.width/2;
        float y2_min = b2.y_center - b2.height/2;
        float y2_max = b2.y_center + b2.height/2;
        
        float inter_x1 = max(x1_min, x2_min);
        float inter_y1 = max(y1_min, y2_min);
        float inter_x2 = min(x1_max, x2_max);
        float inter_y2 = min(y1_max, y2_max);
        
        float inter_width = max(0.0f, inter_x2 - inter_x1);
        float inter_height = max(0.0f, inter_y2 - inter_y1);
        float inter_area = inter_width * inter_height;
        
        float b1_area = b1.width * b1.height;
        float b2_area = b2.width * b2.height;
        float union_area = b1_area + b2_area - inter_area;
        
        if (union_area == 0) return 0;
        return inter_area / union_area;
    }
};

// ==================== 4. 文件处理器 ====================
class FileProcessor {
public:
    // 检查文件是否存在
    static bool exists(const string& path) {
        return fs::exists(path);
    }
    
    // 删除文件
    static bool deleteFile(const string& path, bool move_to_trash = false) {
        try {
            if (!exists(path)) {
                cout << "文件不存在: " << path << endl;
                return false;
            }
            
            if (move_to_trash) {
                return moveToTrash(path);
            } else {
                fs::remove(path);
                cout << "已删除: " << path << endl;
                return true;
            }
        } catch (const exception& e) {
            cerr << "删除失败: " << path << " - " << e.what() << endl;
            return false;
        }
    }
    
    // 备份文件
    static bool backupFile(const string& src_path, const string& backup_dir) {
        try {
            if (!exists(src_path)) {
                return false;
            }
            
            fs::create_directories(backup_dir);
            
            string filename = fs::path(src_path).filename().string();
            string backup_path = backup_dir + "/" + filename;
            
            // 添加时间戳避免重名
            if (exists(backup_path)) {
                time_t now = time(nullptr);
                char time_str[20];
                strftime(time_str, sizeof(time_str), "%Y%m%d_%H%M%S", localtime(&now));
                backup_path = backup_dir + "/" + string(time_str) + "_" + filename;
            }
            
            fs::copy(src_path, backup_path);
            cout << "已备份: " << src_path << " -> " << backup_path << endl;
            return true;
        } catch (const exception& e) {
            cerr << "备份失败: " << src_path << " - " << e.what() << endl;
            return false;
        }
    }
    
    // 获取图片文件列表
    static vector<string> getImageFiles(const string& dir) {
        vector<string> files;
        
        if (!exists(dir)) {
            cerr << "目录不存在: " << dir << endl;
            return files;
        }
        
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (isImageFile(entry.path())) {
                files.push_back(entry.path().string());
            }
        }
        
        return files;
    }
    
    // 获取目录大小
    static size_t getDirectorySize(const string& dir) {
        size_t total_size = 0;
        
        if (!exists(dir)) return 0;
        
        for (const auto& entry : fs::recursive_directory_iterator(dir)) {
            if (fs::is_regular_file(entry.path())) {
                total_size += fs::file_size(entry.path());
            }
        }
        
        return total_size;
    }
    
private:
    static bool isImageFile(const fs::path& path) {
        static const vector<string> exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
        string ext = path.extension().string();
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return find(exts.begin(), exts.end(), ext) != exts.end();
    }
    
    static bool moveToTrash(const string& path) {
        // 创建回收站目录
        string trash_dir = "./dataset_trash";
        fs::create_directories(trash_dir);
        
        // 移动文件
        string filename = fs::path(path).filename().string();
        string trash_path = trash_dir + "/" + filename;
        
        if (exists(trash_path)) {
            time_t now = time(nullptr);
            char time_str[20];
            strftime(time_str, sizeof(time_str), "%Y%m%d_%H%M%S", localtime(&now));
            trash_path = trash_dir + "/" + string(time_str) + "_" + filename;
        }
        
        fs::rename(path, trash_path);
        cout << "已移动到回收站: " << path << endl;
        return true;
    }
};

// ==================== 5. 增强的数据集清理器 ====================
class EnhancedDatasetCleaner {
private:
    CleanupConfig config_;
    string backup_dir_;
    
    // 统计数据
    struct Stats {
        int total_images = 0;
        int processed_images = 0;
        int black_images = 0;
        int dark_images = 0;
        int low_contrast_images = 0;
        int overexposed_images = 0;
        int monochrome_images = 0;
        int invalid_images = 0;
        int empty_annotations = 0;
        int invalid_annotations = 0;
        int duplicate_bboxes = 0;
        int deleted_images = 0;
        int deleted_labels = 0;
        map<string, int> problem_reasons;
    } stats_;
    
public:
    EnhancedDatasetCleaner(const CleanupConfig& config, const string& dataset_path) 
        : config_(config) {
        backup_dir_ = dataset_path + "/backup_" + getCurrentTime();
    }
    
    // 主清理函数
    bool cleanDataset(const string& dataset_path) {
        string image_dir = dataset_path + "/images";
        string label_dir = dataset_path + "/labels";
        
        cout << "==========================================" << endl;
        cout << "    增强版数据集清理工具" << endl;
        cout << "==========================================" << endl;
        
        cout << "图片目录: " << image_dir << endl;
        cout << "标注目录: " << label_dir << endl;
        cout << "备份目录: " << backup_dir_ << endl;
        cout << "==========================================" << endl;
        
        // 检查目录
        if (!FileProcessor::exists(image_dir)) {
            cerr << "错误: 图片目录不存在" << endl;
            return false;
        }
        
        if (!FileProcessor::exists(label_dir)) {
            cout << "警告: 标注目录不存在" << endl;
        }
        
        // 获取图片文件
        vector<string> image_files = FileProcessor::getImageFiles(image_dir);
        stats_.total_images = image_files.size();
        
        if (stats_.total_images == 0) {
            cerr << "错误: 没有找到图片文件" << endl;
            return false;
        }
        
        cout << "找到 " << stats_.total_images << " 张图片" << endl;
        
        // 创建日志文件
        ofstream log_file(dataset_path + "/cleanup_detailed.log");
        ofstream csv_file(dataset_path + "/cleanup_report.csv");
        
        // CSV头部
        csv_file << "image,status,brightness,contrast,quality_score,bbox_count,"
                 << "is_black,is_dark,is_low_contrast,is_overexposed,is_monochrome,"
                 << "has_annotation_issues,reason,action\n";
        
        // 处理每张图片
        for (size_t i = 0; i < image_files.size(); i++) {
            string image_path = image_files[i];
            string base_name = fs::path(image_path).stem().string();
            string label_path = label_dir + "/" + base_name + ".txt";
            
            stats_.processed_images++;
            
            // 记录处理
            log_file << "\n处理文件: " << base_name << endl;
            
            try {
                // 检查图片
                bool should_delete = checkImage(image_path, base_name, log_file, csv_file);
                
                // 检查标注
                if (!should_delete && FileProcessor::exists(label_path)) {
                    should_delete = should_delete || checkAnnotation(label_path, base_name, log_file);
                }
                
                // 处理文件
                if (should_delete) {
                    handleDeletion(image_path, label_path, base_name, log_file);
                }
                
            } catch (const exception& e) {
                cerr << "处理失败: " << base_name << " - " << e.what() << endl;
                log_file << "错误: " << e.what() << endl;
            }
            
            // 进度显示
            if ((i + 1) % 10 == 0) {
                cout << "进度: " << (i + 1) << "/" << image_files.size() 
                     << " (" << fixed << setprecision(1) 
                     << (float)(i + 1) / image_files.size() * 100.0f << "%)" << endl;
            }
        }
        
        // 保存统计和报告
        saveReport(dataset_path, log_file, csv_file);
        
        log_file.close();
        csv_file.close();
        
        return true;
    }
    
    // 显示统计信息
    void printStatistics() {
        cout << "\n==========================================" << endl;
        cout << "            清理统计报告" << endl;
        cout << "==========================================" << endl;
        cout << "总图片数: " << stats_.total_images << endl;
        cout << "已处理: " << stats_.processed_images << endl;
        cout << endl;
        cout << "图片质量问题:" << endl;
        cout << "  - 全黑图片: " << stats_.black_images << endl;
        cout << "  - 暗图: " << stats_.dark_images << endl;
        cout << "  - 低对比度: " << stats_.low_contrast_images << endl;
        cout << "  - 过曝图片: " << stats_.overexposed_images << endl;
        cout << "  - 单色图片: " << stats_.monochrome_images << endl;
        cout << "  - 无效格式: " << stats_.invalid_images << endl;
        cout << endl;
        cout << "标注问题:" << endl;
        cout << "  - 空标注: " << stats_.empty_annotations << endl;
        cout << "  - 无效标注: " << stats_.invalid_annotations << endl;
        cout << "  - 重复边界框: " << stats_.duplicate_bboxes << endl;
        cout << endl;
        cout << "清理结果:" << endl;
        cout << "  - 已删除图片: " << stats_.deleted_images << endl;
        cout << "  - 已删除标注: " << stats_.deleted_labels << endl;
        cout << endl;
        
        cout << "问题分布:" << endl;
        for (const auto& [reason, count] : stats_.problem_reasons) {
            cout << "  - " << reason << ": " << count << endl;
        }
        cout << "==========================================" << endl;
    }
    
private:
    // 检查图片质量
    bool checkImage(const string& image_path, const string& base_name, 
                   ofstream& log_file, ofstream& csv_file) {
        cv::Mat image = cv::imread(image_path);
        vector<string> problems;
        
        // 基本检查
        if (EnhancedImageDetector::isInvalidImage(image, config_.format.min_width, 
                                                 config_.format.min_height)) {
            stats_.invalid_images++;
            problems.push_back("invalid_format");
            log_file << "  - 无效图片格式" << endl;
        }
        
        // 质量检查
        if (EnhancedImageDetector::isCompletelyBlack(image, config_.image.black_threshold)) {
            stats_.black_images++;
            problems.push_back("completely_black");
            log_file << "  - 全黑图片" << endl;
        }
        
        if (EnhancedImageDetector::isDarkImage(image, config_.image.dark_threshold)) {
            stats_.dark_images++;
            problems.push_back("dark_image");
            log_file << "  - 暗图" << endl;
        }
        
        if (EnhancedImageDetector::isLowContrastImage(image, config_.image.min_contrast)) {
            stats_.low_contrast_images++;
            problems.push_back("low_contrast");
            log_file << "  - 低对比度" << endl;
        }
        
        if (EnhancedImageDetector::isOverExposedImage(image, config_.image.max_brightness)) {
            stats_.overexposed_images++;
            problems.push_back("overexposed");
            log_file << "  - 过曝图片" << endl;
        }
        
        if (EnhancedImageDetector::isMonochromeImage(image)) {
            stats_.monochrome_images++;
            problems.push_back("monochrome");
            log_file << "  - 单色图片" << endl;
        }
        
        // 计算质量信息
        float brightness = 0, contrast = 0, quality_score = 0;
        if (!image.empty()) {
            cv::Mat gray;
            if (image.channels() == 3) {
                cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = image;
            }
            
            cv::Scalar mean, stddev;
            cv::meanStdDev(gray, mean, stddev);
            brightness = mean[0];
            contrast = stddev[0];
            quality_score = EnhancedImageDetector::calculateQualityScore(image);
        }
        
        // 记录到CSV
        csv_file << base_name << "," 
                 << (problems.empty() ? "good" : "bad") << ","
                 << brightness << "," << contrast << "," << quality_score << ",0,"
                 << (problems.empty() ? "0" : "1") << ","  // 简化表示
                 << "N/A,N/A,N/A,N/A,N/A,";  // 占位符
        
        bool should_delete = !problems.empty();
        if (should_delete) {
            string reason = problems[0];  // 主要问题
            csv_file << "1," << reason << ",delete";
            
            // 统计问题原因
            stats_.problem_reasons[reason]++;
        } else {
            csv_file << "0,ok,keep";
        }
        
        csv_file << endl;
        
        return should_delete;
    }
    
    // 检查标注
    bool checkAnnotation(const string& label_path, const string& base_name, ofstream& log_file) {
        auto bboxes = EnhancedAnnotationDetector::loadAnnotations(label_path);
        vector<string> problems;
        
        if (bboxes.empty()) {
            stats_.empty_annotations++;
            problems.push_back("empty_annotation");
            log_file << "  - 空标注文件" << endl;
        } else {
            if (EnhancedAnnotationDetector::hasProblems(bboxes, 
                                                       config_.annotation.min_bbox_width,
                                                       config_.annotation.min_bbox_height,
                                                       config_.annotation.min_bbox_area,
                                                       config_.annotation.max_bbox_area,
                                                       config_.annotation.min_bbox_count)) {
                stats_.invalid_annotations++;
                problems.push_back("invalid_annotation");
                log_file << "  - 无效标注" << endl;
            }
            
            if (EnhancedAnnotationDetector::hasDuplicates(bboxes)) {
                stats_.duplicate_bboxes++;
                problems.push_back("duplicate_bboxes");
                log_file << "  - 重复边界框" << endl;
            }
        }
        
        return !problems.empty();
    }
    
    // 处理删除
    void handleDeletion(const string& image_path, const string& label_path,
                       const string& base_name, ofstream& log_file) {
        if (config_.cleanup.dry_run) {
            cout << "[模拟] 删除: " << base_name << endl;
            log_file << "  - [模拟] 标记为删除" << endl;
            return;
        }
        
        // 备份
        if (config_.cleanup.backup_first) {
            FileProcessor::backupFile(image_path, backup_dir_);
            if (FileProcessor::exists(label_path)) {
                FileProcessor::backupFile(label_path, backup_dir_);
            }
        }
        
        // 删除图片
        if (config_.cleanup.delete_images) {
            if (FileProcessor::deleteFile(image_path, config_.cleanup.move_to_trash)) {
                stats_.deleted_images++;
                log_file << "  - 已删除图片" << endl;
            }
        }
        
        // 删除标注
        if (config_.cleanup.delete_labels && FileProcessor::exists(label_path)) {
            if (FileProcessor::deleteFile(label_path, config_.cleanup.move_to_trash)) {
                stats_.deleted_labels++;
                log_file << "  - 已删除标注" << endl;
            }
        }
    }
    
    // 保存报告
    void saveReport(const string& dataset_path, ofstream& log_file, ofstream& csv_file) {
        string report_path = dataset_path + "/cleanup_summary.txt";
        ofstream report_file(report_path);
        
        if (report_file.is_open()) {
            report_file << "==========================================" << endl;
            report_file << "        数据集清理总结报告" << endl;
            report_file << "==========================================" << endl;
            report_file << "清理时间: " << getCurrentTime() << endl;
            report_file << "数据集路径: " << dataset_path << endl;
            report_file << "备份目录: " << backup_dir_ << endl;
            report_file << endl;
            
            report_file << "配置参数:" << endl;
            report_file << "  图片质量:" << endl;
            report_file << "    - 全黑阈值: " << config_.image.black_threshold << endl;
            report_file << "    - 暗图阈值: " << config_.image.dark_threshold << endl;
            report_file << "    - 最小亮度: " << config_.image.min_brightness << endl;
            report_file << "    - 最小对比度: " << config_.image.min_contrast << endl;
            report_file << "    - 最大亮度: " << config_.image.max_brightness << endl;
            report_file << "  标注质量:" << endl;
            report_file << "    - 最小边界框数: " << config_.annotation.min_bbox_count << endl;
            report_file << "    - 最小边界框面积: " << config_.annotation.min_bbox_area << endl;
            report_file << "  清理选项:" << endl;
            report_file << "    - 模拟运行: " << (config_.cleanup.dry_run ? "是" : "否") << endl;
            report_file << "    - 备份文件: " << (config_.cleanup.backup_first ? "是" : "否") << endl;
            report_file << endl;
            
            report_file << "统计结果:" << endl;
            report_file << "  总计: " << stats_.total_images << " 张图片" << endl;
            report_file << "  全黑图片: " << stats_.black_images << endl;
            report_file << "  暗图: " << stats_.dark_images << endl;
            report_file << "  低对比度: " << stats_.low_contrast_images << endl;
            report_file << "  过曝图片: " << stats_.overexposed_images << endl;
            report_file << "  单色图片: " << stats_.monochrome_images << endl;
            report_file << "  无效格式: " << stats_.invalid_images << endl;
            report_file << "  空标注: " << stats_.empty_annotations << endl;
            report_file << "  无效标注: " << stats_.invalid_annotations << endl;
            report_file << "  重复边界框: " << stats_.duplicate_bboxes << endl;
            report_file << "  已删除图片: " << stats_.deleted_images << endl;
            report_file << "  已删除标注: " << stats_.deleted_labels << endl;
            report_file << endl;
            
            report_file << "建议:" << endl;
            if (stats_.black_images > 0) {
                report_file << "  - 发现 " << stats_.black_images << " 张全黑图片，建议检查数据采集过程" << endl;
            }
            if (stats_.invalid_annotations > 0) {
                report_file << "  - 发现 " << stats_.invalid_annotations << " 个无效标注，建议重新标注" << endl;
            }
            if (stats_.duplicate_bboxes > 0) {
                report_file << "  - 发现 " << stats_.duplicate_bboxes << " 个重复标注，建议清理" << endl;
            }
            
            report_file << "==========================================" << endl;
            report_file.close();
        }
    }
    
    // 获取当前时间
    string getCurrentTime() {
        time_t now = time(nullptr);
        char time_str[20];
        strftime(time_str, sizeof(time_str), "%Y%m%d_%H%M%S", localtime(&now));
        return string(time_str);
    }
};

// ==================== 6. 主函数 ====================
// 程序入口: 解析命令行参数 -> 初始化配置 -> 运行清理流程 -> 输出报告
// 主要流程由 `EnhancedDatasetCleaner::cleanDataset` 实现，最后打印统计信息。
int main(int argc, char* argv[]) {
    CleanupConfig config;
    string dataset_path;
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--dataset" && i+1 < argc) {
            dataset_path = argv[++i];
        } else if (arg == "--black-threshold" && i+1 < argc) {
            config.image.black_threshold = stof(argv[++i]);
        } else if (arg == "--dark-threshold" && i+1 < argc) {
            config.image.dark_threshold = stof(argv[++i]);
        } else if (arg == "--min-contrast" && i+1 < argc) {
            config.image.min_contrast = stof(argv[++i]);
        } else if (arg == "--dry-run") {
            config.cleanup.dry_run = true;
        } else if (arg == "--backup-only") {
            config.cleanup.delete_images = false;
            config.cleanup.delete_labels = false;
            config.cleanup.backup_first = true;
        } else if (arg == "--move-to-trash") {
            config.cleanup.move_to_trash = true;
        } else if (arg == "--help") {
            cout << "用法: ./enhanced_cleaner [选项]" << endl;
            cout << "选项:" << endl;
            cout << "  --dataset <路径>          数据集目录" << endl;
            cout << "  --black-threshold <数值>  全黑阈值 (默认: 10.0)" << endl;
            cout << "  --dark-threshold <数值>   暗图阈值 (默认: 30.0)" << endl;
            cout << "  --min-contrast <数值>     最小对比度 (默认: 15.0)" << endl;
            cout << "  --dry-run                 模拟运行，不实际删除" << endl;
            cout << "  --backup-only             只备份，不删除" << endl;
            cout << "  --move-to-trash           移动到回收站" << endl;
            cout << "  --help                    显示帮助" << endl;
            return 0;
        }
    }
    
    if (dataset_path.empty()) {
        cout << "输入数据集路径: ";
        cin >> dataset_path;
    }
    
    if (!fs::exists(dataset_path)) {
        cerr << "错误: 数据集目录不存在" << endl;
        return 1;
    }
    
    // 显示配置
    cout << "\n配置参数:" << endl;
    cout << "  - 全黑阈值: " << config.image.black_threshold << endl;
    cout << "  - 暗图阈值: " << config.image.dark_threshold << endl;
    cout << "  - 最小对比度: " << config.image.min_contrast << endl;
    cout << "  - 模拟运行: " << (config.cleanup.dry_run ? "是" : "否") << endl;
    cout << "  - 备份文件: " << (config.cleanup.backup_first ? "是" : "否") << endl;
    cout << endl;
    
    if (!config.cleanup.dry_run) {
        cout << "警告: 此操作将删除检测到的低质量文件!" << endl;
        cout << "输入 'yes' 继续: ";
        
        string confirm;
        cin >> confirm;
        
        if (confirm != "yes" && confirm != "YES" && confirm != "y") {
            cout << "操作取消" << endl;
            return 0;
        }
    }
    
    try {
        EnhancedDatasetCleaner cleaner(config, dataset_path);
        
        if (cleaner.cleanDataset(dataset_path)) {
            cleaner.printStatistics();
            cout << "\n清理完成!" << endl;
            
            if (!config.cleanup.dry_run) {
                cout << "日志文件: " << dataset_path << "/cleanup_detailed.log" << endl;
                cout << "CSV报告: " << dataset_path << "/cleanup_report.csv" << endl;
                cout << "总结报告: " << dataset_path << "/cleanup_summary.txt" << endl;
            }
        } else {
            cerr << "清理失败!" << endl;
            return 1;
        }
    } catch (const exception& e) {
        cerr << "错误: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}