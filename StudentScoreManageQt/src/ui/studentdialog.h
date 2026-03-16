#ifndef STUDENTDIALOG_H
#define STUDENTDIALOG_H

#include <QDialog>
#include <QLineEdit>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QDialogButtonBox>
#include "student.h"

/**
 * @brief 学生信息编辑对话框类
 * 
 * 用于添加新学生或修改现有学生信息的模态对话框。
 * 包含学号、姓名和各科成绩的输入控件。
 * 
 * Modal dialog for adding new students or editing existing student information.
 * Contains input fields for student ID, name, and scores.
 */
class StudentDialog : public QDialog
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数
     * 
     * @param parent 父窗口指针，默认为 nullptr
     */
    explicit StudentDialog(QWidget *parent = nullptr);

    /**
     * @brief 设置对话框中的学生数据
     * 
     * 用于在修改学生信息时回显数据。
     * 
     * @param stu 包含要显示的学生信息的结构体
     */
    void setStudent(const Student& stu);

    /**
     * @brief 获取对话框中输入的学生数据
     * 
     * 当用户点击“确定”后，调用此函数获取输入的数据。
     * 
     * @return Student 填充了用户输入数据的学生结构体
     */
    Student getStudent() const;

    /**
     * @brief 设置学号输入框是否只读
     * 
     * 修改学生信息时，学号通常作为主键不可修改。
     * 
     * @param readOnly true 为只读，false 为可编辑
     */
    void setReadOnlyNumber(bool readOnly);

private:
    QLineEdit *numberEdit;          ///< 学号输入框 (Student ID input field)
    QLineEdit *nameEdit;            ///< 姓名输入框 (Name input field)
    QDoubleSpinBox *scoreEdits[9];  ///< 各科成绩输入框数组 (Array of score input fields)
    QDialogButtonBox *buttonBox;    ///< 确定/取消按钮盒 (OK/Cancel button box)
};

#endif // STUDENTDIALOG_H
