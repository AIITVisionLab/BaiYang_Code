#include "studentmanager.h"

/**
 * @brief 获取 StudentManager 的单例实例
 * 
 * 详细说明：
 * 使用 C++11 的静态局部变量特性实现单例模式。
 * 静态局部变量 instance 只会在第一次调用此函数时初始化，
 * 并且是线程安全的。
 * 
 * @return StudentManager& 单例引用
 */
StudentManager& StudentManager::instance() {
    static StudentManager instance;
    return instance;
}

/**
 * @brief 构造函数
 * 
 * 详细说明：
 * 在创建 StudentManager 实例时，调用 C API 初始化底层的学生链表。
 * 这是一个私有构造函数，只能通过 instance() 方法调用。
 */
StudentManager::StudentManager() {
    apiInitList();
}

/**
 * @brief 析构函数
 * 
 * 详细说明：
 * 在 StudentManager 销毁时，调用 C API 销毁底层的学生链表，释放内存。
 */
StudentManager::~StudentManager() {
    apiDestroyList();
}

/**
 * @brief 添加学生
 * 
 * 详细说明：
 * 将 C++ 的 Student 对象数据传递给 C API 的 apiAddStudent 函数。
 * 
 * @param stu 学生对象
 * @return true 添加成功
 * @return false 添加失败
 */
bool StudentManager::addStudent(const Student& stu) {
    return apiAddStudent(stu.number, stu.name, stu.GaoShu, stu.XianDai, stu.DaoLun, 
                           stu.XinLi, stu.YingYu, stu.CYuYan, stu.SiZheng, stu.TiYu, stu.ChuangYi) == 0;
}

/**
 * @brief 删除学生
 * 
 * 详细说明：
 * 调用 C API 的 apiDeleteStudent 函数，根据学号删除学生。
 * 
 * @param number 学号
 * @return true 删除成功
 * @return false 删除失败
 */
bool StudentManager::deleteStudent(unsigned long long number) {
    return apiDeleteStudent(number) == 0;
}

/**
 * @brief 更新学生信息
 * 
 * 详细说明：
 * 调用 C API 的 apiUpdateStudent 函数，更新学生信息。
 * 
 * @param stu 包含更新后信息的学生对象
 * @return true 更新成功
 * @return false 更新失败
 */
bool StudentManager::updateStudent(const Student& stu) {
    return apiUpdateStudent(stu.number, stu.name, stu.GaoShu, stu.XianDai, stu.DaoLun, 
                              stu.XinLi, stu.YingYu, stu.CYuYan, stu.SiZheng, stu.TiYu, stu.ChuangYi) == 0;
}

/**
 * @brief 获取所有学生列表
 * 
 * 详细说明：
 * 1. 获取学生总数。
 * 2. 遍历索引 0 到 count-1。
 * 3. 对每个索引调用 apiGetStudentAt 获取学生数据。
 * 4. 将获取到的学生数据添加到 std::vector 中。
 * 5. 返回包含所有学生的 vector。
 * 
 * @return std::vector<Student> 学生列表
 */
std::vector<Student> StudentManager::getAllStudents() {
    std::vector<Student> students;
    int count = apiGetStudentCount();
    for (int i = 0; i < count; ++i) {
        Student stu;
        if (apiGetStudentAt(i, &stu) == 0) {
            students.push_back(stu);
        }
    }
    return students;
}

/**
 * @brief 获取学生总数
 * 
 * @return int 学生数量
 */
int StudentManager::getStudentCount() {
    return apiGetStudentCount();
}

/**
 * @brief 获取平均分统计信息
 * 
 * 详细说明：
 * 调用 C API 的 apiCalculateAverage 获取格式化的平均分字符串。
 * 
 * @return std::string 格式化的平均分统计信息
 */
std::string StudentManager::getAverageStats() {
    char buf[8192] = {0};
    if (apiCalculateAverage(buf, sizeof(buf)) == 0) {
        return std::string(buf);
    }
    return "";
}
