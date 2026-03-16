#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "studenttablewidget.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

/**
 * @brief 主窗口类
 * 
 * 负责显示学生管理系统的主界面，包括功能按钮和学生信息表格。
 * 继承自 QMainWindow。
 * 
 * Main window class responsible for displaying the main interface,
 * including function buttons and the student information table.
 */
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数
     * 
     * @param parent 父窗口指针，默认为 nullptr
     */
    MainWindow(QWidget *parent = nullptr);

    /**
     * @brief 析构函数
     * 
     * 释放 UI 资源和后端数据结构。
     */
    ~MainWindow();

private slots:
    /**
     * @brief 处理添加学生按钮点击事件
     * 
     * 弹出对话框以添加新学生。
     */
    void on_addStudentButton_clicked();

    /**
     * @brief 处理删除学生按钮点击事件
     * 
     * 删除选中的学生或提示输入学号进行删除。
     */
    void on_deleteStudentButton_clicked();

protected:
    /**
     * @brief 绘图事件处理
     * 
     * 用于绘制背景图片，支持自适应缩放且保持纵横比。
     */
    void paintEvent(QPaintEvent *event) override;

private slots:
    /**
     * @brief 处理查询学生按钮点击事件
     * 
     * 提示输入学号并在表格中高亮显示匹配的学生。
     */
    void on_queryStudentButton_clicked();

    /**
     * @brief 处理修改学生按钮点击事件
     * 
     * 弹出对话框以修改选中学生的信息。
     */
    void on_updateStudentButton_clicked();

    /**
     * @brief 处理显示全部按钮点击事件（刷新列表）
     * 
     * 重新加载并显示所有学生数据。
     */
    void on_showAllButton_clicked();

    /**
     * @brief 处理平均分统计按钮点击事件
     * 
     * 计算并显示所有学生的平均分统计信息。
     */
    void on_averageButton_clicked();

    /**
     * @brief 处理导出文件按钮点击事件
     * 
     * 将当前学生数据导出为 CSV 文件。
     */
    void on_exportButton_clicked();

private:
    Ui::MainWindow *ui;         ///< UI 指针，管理 Qt Designer 生成的界面元素
    StudentTableWidget *tableWidget;  ///< 学生信息表格控件，用于展示学生列表

    /**
     * @brief 初始化自定义 UI 布局
     * 
     * 创建并排列左侧按钮和右侧表格。
     */
    void setupCustomUI();

    /**
     * @brief 刷新表格数据
     * 
     * 从后端获取最新学生数据并更新到表格中。
     */
    void refreshTable();
};

#endif // MAINWINDOW_H
