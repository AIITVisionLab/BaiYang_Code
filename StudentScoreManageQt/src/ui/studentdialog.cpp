#include "studentdialog.h"
#include <QVBoxLayout>
#include <QLabel>

/**
 * @brief 构造函数
 * 
 * 初始化对话框界面，创建表单布局和输入控件。
 * 包含学号、姓名输入框以及9门课程成绩的数字微调框。
 * 
 * @param parent 父窗口指针，默认为 nullptr
 * 
 * Constructor: Initializes the dialog UI with form layout and input fields.
 */
StudentDialog::StudentDialog(QWidget *parent) : QDialog(parent)
{
    setWindowTitle("学生信息");
    
    // 主布局：垂直布局
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    
    // 表单布局：用于排列标签和输入框
    QFormLayout *formLayout = new QFormLayout();

    // 学号输入框
    numberEdit = new QLineEdit();
    numberEdit->setPlaceholderText("请输入学号");
    formLayout->addRow("学号:", numberEdit);

    // 姓名输入框
    nameEdit = new QLineEdit();
    nameEdit->setPlaceholderText("请输入姓名");
    formLayout->addRow("姓名:", nameEdit);

    // 课程名称数组
    const char* subjects[9] = {"高数", "现代", "导论", "心理", "英语", "C语言", "思政", "体育", "创意"};
    
    // 循环创建9个成绩输入框
    for (int i = 0; i < 9; ++i) {
        scoreEdits[i] = new QDoubleSpinBox();
        scoreEdits[i]->setRange(0, 100); // 成绩范围 0-100
        scoreEdits[i]->setSingleStep(1.0); // 步长 1.0
        formLayout->addRow(QString("%1:").arg(subjects[i]), scoreEdits[i]);
    }

    // 将表单布局添加到主布局
    mainLayout->addLayout(formLayout);

    // 创建标准按钮盒（确定/取消）
    buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    // 连接信号槽
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    
    // 将按钮盒添加到主布局
    mainLayout->addWidget(buttonBox);
}

/**
 * @brief 设置对话框中的学生数据
 * 
 * 将传入的学生信息填充到对应的输入控件中。
 * 通常用于“修改学生”功能，回显当前学生的信息。
 * 
 * @param stu 包含学生信息的结构体
 * 
 * Sets the student data in the dialog fields (for editing).
 */
void StudentDialog::setStudent(const Student& stu)
{
    numberEdit->setText(QString::number(stu.number));
    nameEdit->setText(QString::fromUtf8(stu.name));
    scoreEdits[0]->setValue(stu.GaoShu);
    scoreEdits[1]->setValue(stu.XianDai);
    scoreEdits[2]->setValue(stu.DaoLun);
    scoreEdits[3]->setValue(stu.XinLi);
    scoreEdits[4]->setValue(stu.YingYu);
    scoreEdits[5]->setValue(stu.CYuYan);
    scoreEdits[6]->setValue(stu.SiZheng);
    scoreEdits[7]->setValue(stu.TiYu);
    scoreEdits[8]->setValue(stu.ChuangYi);
}

/**
 * @brief 获取对话框中输入的学生数据
 * 
 * 从各个输入控件中读取数据并封装成 Student 结构体返回。
 * 
 * @return Student 包含用户输入数据的学生结构体
 * 
 * Retrieves the student data entered in the dialog fields.
 */
Student StudentDialog::getStudent() const
{
    Student stu;
    stu.number = numberEdit->text().toULongLong();
    // 将 QString 转换为 C 风格字符串，并确保安全复制
    strncpy(stu.name, nameEdit->text().toUtf8().data(), sizeof(stu.name)-1);
    stu.name[sizeof(stu.name)-1] = '\0'; // 确保 null 结尾
    
    // 获取各科成绩
    stu.GaoShu = scoreEdits[0]->value();
    stu.XianDai = scoreEdits[1]->value();
    stu.DaoLun = scoreEdits[2]->value();
    stu.XinLi = scoreEdits[3]->value();
    stu.YingYu = scoreEdits[4]->value();
    stu.CYuYan = scoreEdits[5]->value();
    stu.SiZheng = scoreEdits[6]->value();
    stu.TiYu = scoreEdits[7]->value();
    stu.ChuangYi = scoreEdits[8]->value();
    return stu;
}

/**
 * @brief 设置学号输入框是否只读
 * 
 * 在修改学生信息时，学号通常作为唯一标识符不允许修改。
 * 
 * @param readOnly true 为只读，false 为可编辑
 * 
 * Sets the read-only state of the student number field.
 */
void StudentDialog::setReadOnlyNumber(bool readOnly)
{
    numberEdit->setReadOnly(readOnly);
}
