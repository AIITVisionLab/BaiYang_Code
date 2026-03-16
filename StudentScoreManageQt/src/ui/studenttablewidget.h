#ifndef STUDENTTABLEWIDGET_H
#define STUDENTTABLEWIDGET_H

#include <QTableWidget>
#include <vector>
#include "student_c_api.h"

/**
 * @brief 学生表格控件
 * 
 * 封装了学生信息的展示逻辑，继承自 QTableWidget。
 */
class StudentTableWidget : public QTableWidget
{
    Q_OBJECT

public:
    explicit StudentTableWidget(QWidget *parent = nullptr);

    /**
     * @brief 刷新表格数据
     * @param students 学生列表
     */
    void refresh(const std::vector<Student>& students);

    /**
     * @brief 获取当前选中行的学生学号
     * @return 学号，如果未选中则返回 0
     */
    unsigned long long getSelectedStudentNumber() const;

    /**
     * @brief 获取当前选中行的学生信息
     * @param stu 输出参数，存储学生信息
     * @return 是否成功获取
     */
    bool getSelectedStudent(Student& stu) const;

    /**
     * @brief 选中指定学号的学生
     * @param number 学号
     * @return 是否找到并选中
     */
    bool selectStudent(unsigned long long number);

private:
    void setupHeader();
};

#endif // STUDENTTABLEWIDGET_H
