#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "studentdialog.h"
#include "../manager/studentmanager.h"
#include <QMessageBox>
#include <QInputDialog>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include <QHeaderView>
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QPainter>
#include <QPaintEvent>

/**
 * @brief 构造函数
 * 
 * 详细说明：
 * 1. 调用父类 QMainWindow 的构造函数。
 * 2. 初始化 ui 指针（Qt Designer 生成的 UI 类）。
 * 3. 调用 setupUi 初始化基本界面。
 * 4. 调用 setupCustomUI 设置自定义的布局和控件（因为本项目主要使用代码构建 UI 而非 Designer）。
 * 5. 调用 refreshTable 加载并显示初始数据。
 * 
 * @param parent 父窗口指针，默认为 nullptr
 */
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    
    // 后端链表初始化由 StudentManager 构造函数处理
    
    // 设置自定义的按钮和表格布局
    setupCustomUI();

    // 刷新表格显示数据
    refreshTable();
}

/**
 * @brief 析构函数
 * 
 * 详细说明：
 * 释放 ui 指针占用的内存。
 * 注意：StudentManager 是单例，其析构函数会在程序退出时自动调用，负责销毁后端链表。
 */
MainWindow::~MainWindow()
{
    delete ui;
}

/**
 * @brief 初始化自定义 UI 布局
 * 
 * 详细说明：
 * 1. 创建中心部件 (centralWidget) 并设置为透明背景，以便显示背景图片。
 * 2. 创建主水平布局 (mainLayout)，将窗口分为左右两部分。
 * 3. 创建左侧垂直布局 (buttonLayout) 用于放置功能按钮。
 * 4. 创建一系列功能按钮（添加、删除、修改、查询等）。
 * 5. 设置按钮样式（半透明背景、圆角、字体等）。
 * 6. 将按钮添加到左侧布局中，并添加弹簧 (addStretch) 使按钮顶部对齐。
 * 7. 连接每个按钮的 clicked 信号到对应的槽函数。
 * 8. 创建右侧表格控件 (StudentTableWidget)。
 * 9. 将左侧按钮布局和右侧表格控件添加到主布局中，设置比例为 1:4。
 * 10. 设置窗口初始大小为 1000x600。
 */
void MainWindow::setupCustomUI()
{
    // 创建中心部件
    QWidget *centralWidget = new QWidget(this);
    centralWidget->setObjectName("centralwidget");
    // 设置背景透明，以便 MainWindow 的 paintEvent 绘制的背景可见
    centralWidget->setAttribute(Qt::WA_TranslucentBackground);
    this->setCentralWidget(centralWidget);

    // 主布局：水平布局 (Left: Buttons, Right: Table)
    QHBoxLayout *mainLayout = new QHBoxLayout(centralWidget);

    // 左侧按钮布局：垂直布局
    QVBoxLayout *buttonLayout = new QVBoxLayout();
    
    // 创建功能按钮
    QPushButton *btnAdd = new QPushButton("添加学生");
    QPushButton *btnDel = new QPushButton("删除学生");
    QPushButton *btnUpdate = new QPushButton("修改学生");
    QPushButton *btnQuery = new QPushButton("查询学生");
    QPushButton *btnRefresh = new QPushButton("刷新列表");
    QPushButton *btnAvg = new QPushButton("平均分统计");
    QPushButton *btnExport = new QPushButton("导出文件");

    // 设置按钮样式 (Padding and Font size)
    // 增加半透明背景，使背景图片可见，并强制设置字体颜色为黑色
    QString btnStyle = "QPushButton { padding: 10px; font-size: 14px; color: #000000; font-weight: bold; background-color: rgba(255, 255, 255, 200); border-radius: 5px; } QPushButton:hover { background-color: rgba(255, 255, 255, 230); }";
    btnAdd->setStyleSheet(btnStyle);
    btnDel->setStyleSheet(btnStyle);
    btnUpdate->setStyleSheet(btnStyle);
    btnQuery->setStyleSheet(btnStyle);
    btnRefresh->setStyleSheet(btnStyle);
    btnAvg->setStyleSheet(btnStyle);
    btnExport->setStyleSheet(btnStyle);

    // 将按钮添加到垂直布局中
    buttonLayout->addWidget(btnAdd);
    buttonLayout->addWidget(btnDel);
    buttonLayout->addWidget(btnUpdate);
    buttonLayout->addWidget(btnQuery);
    buttonLayout->addWidget(btnRefresh);
    buttonLayout->addWidget(btnAvg);
    buttonLayout->addWidget(btnExport);
    buttonLayout->addStretch(); // 底部添加弹簧，使按钮靠上对齐

    // 连接按钮点击信号到对应的槽函数
    connect(btnAdd, &QPushButton::clicked, this, &MainWindow::on_addStudentButton_clicked);
    connect(btnDel, &QPushButton::clicked, this, &MainWindow::on_deleteStudentButton_clicked);
    connect(btnUpdate, &QPushButton::clicked, this, &MainWindow::on_updateStudentButton_clicked);
    connect(btnQuery, &QPushButton::clicked, this, &MainWindow::on_queryStudentButton_clicked);
    connect(btnRefresh, &QPushButton::clicked, this, &MainWindow::on_showAllButton_clicked);
    connect(btnAvg, &QPushButton::clicked, this, &MainWindow::on_averageButton_clicked);
    connect(btnExport, &QPushButton::clicked, this, &MainWindow::on_exportButton_clicked);

    // 右侧表格配置
    tableWidget = new StudentTableWidget();

    // 将布局添加到主布局
    // 按钮区域占比 1，表格区域占比 4
    mainLayout->addLayout(buttonLayout, 1);
    mainLayout->addWidget(tableWidget, 4);
    
    // 设置窗口初始大小
    resize(1000, 600);
}

/**
 * @brief 绘制事件处理函数
 * 
 * 详细说明：
 * 1. 当窗口需要重绘时调用此函数。
 * 2. 创建 QPainter 对象。
 * 3. 加载背景图片 (:/background.jpg)。
 * 4. 如果图片加载成功：
 *    - 使用 KeepAspectRatioByExpanding 模式缩放图片，使其填满窗口且保持比例（可能会裁剪）。
 *    - 计算居中位置。
 *    - 绘制图片。
 * 5. 如果图片加载失败，填充白色背景。
 * 
 * @param event 绘制事件对象
 */
void MainWindow::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);
    QPainter painter(this);
    QPixmap pixmap(":/background.jpg");
    
    if (!pixmap.isNull()) {
        // 使用 KeepAspectRatioByExpanding 模式缩放图片，填满窗口且不失真
        QPixmap scaled = pixmap.scaled(this->size(), Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);
        
        // 居中绘制
        int x = (this->width() - scaled.width()) / 2;
        int y = (this->height() - scaled.height()) / 2;
        
        painter.drawPixmap(x, y, scaled);
    } else {
        // 如果图片加载失败，填充默认颜色
        painter.fillRect(this->rect(), Qt::white);
    }
}

/**
 * @brief 刷新表格数据
 * 
 * 详细说明：
 * 1. 调用 StudentManager::instance().getAllStudents() 获取最新的学生列表。
 * 2. 调用 tableWidget->refresh(students) 更新表格显示。
 *    - tableWidget 会清空旧数据并重新填充。
 */
void MainWindow::refreshTable()
{
    // 获取所有学生
    std::vector<Student> students = StudentManager::instance().getAllStudents();
    
    // 刷新表格
    tableWidget->refresh(students);
}

/**
 * @brief 处理添加学生按钮点击事件
 * 
 * 详细说明：
 * 1. 创建 StudentDialog 对话框实例。
 * 2. 调用 dialog.exec() 以模态方式显示对话框，等待用户操作。
 * 3. 如果用户点击了"确定" (QDialog::Accepted)：
 *    - 调用 dialog.getStudent() 获取用户输入的学生信息。
 *    - 调用 StudentManager::instance().addStudent(stu) 尝试添加学生。
 *    - 如果添加成功，弹出提示框并刷新表格。
 *    - 如果添加失败（如内存不足），弹出警告框。
 */
void MainWindow::on_addStudentButton_clicked()
{
    StudentDialog dialog(this);
    // 显示模态对话框
    if (dialog.exec() == QDialog::Accepted) {
        // 获取用户输入的学生信息
        Student stu = dialog.getStudent();
        // 调用 Manager 添加学生
        if (StudentManager::instance().addStudent(stu)) {
            QMessageBox::information(this, "成功", "添加成功");
            refreshTable(); // 刷新显示
        } else {
            QMessageBox::warning(this, "失败", "添加失败");
        }
    }
}

/**
 * @brief 处理删除学生按钮点击事件
 * 
 * 详细说明：
 * 1. 尝试从表格中获取当前选中行的学生学号。
 * 2. 如果有选中行 (number != 0)：
 *    - 弹出确认对话框。
 *    - 如果用户确认，调用 StudentManager 删除学生并刷新表格。
 * 3. 如果没有选中行：
 *    - 弹出输入框 (QInputDialog) 让用户输入学号。
 *    - 如果用户输入了学号并确认，调用 StudentManager 删除学生。
 *    - 根据删除结果弹出成功或失败提示。
 */
void MainWindow::on_deleteStudentButton_clicked()
{
    // 获取当前选中的学号
    unsigned long long number = tableWidget->getSelectedStudentNumber();
    
    if (number != 0) {
        // 弹出确认对话框
        if (QMessageBox::question(this, "确认", "确定要删除该学生吗？") == QMessageBox::Yes) {
            StudentManager::instance().deleteStudent(number);
            refreshTable();
        }
    } else {
        // 如果没有选中行，弹窗输入学号
        bool ok;
        number = QInputDialog::getText(this, "删除学生", "请输入要删除的学号:", QLineEdit::Normal, "", &ok).toULongLong();
        if (ok) {
            if (StudentManager::instance().deleteStudent(number)) {
                QMessageBox::information(this, "成功", "删除成功");
                refreshTable();
            } else {
                QMessageBox::warning(this, "失败", "未找到该学生");
            }
        }
    }
}

/**
 * @brief 处理查询学生按钮点击事件
 * 
 * 详细说明：
 * 1. 弹出输入框让用户输入学号。
 * 2. 如果用户确认输入：
 *    - 调用 tableWidget->selectStudent(number) 在表格中查找并高亮显示该学生。
 *    - 如果未找到，弹出警告提示。
 */
void MainWindow::on_queryStudentButton_clicked()
{
    bool ok;
    // 获取用户输入的学号
    unsigned long long number = QInputDialog::getText(this, "查询学生", "请输入学号:", QLineEdit::Normal, "", &ok).toULongLong();
    if (!ok) return;
    
    // 在表格中查找并高亮
    if (!tableWidget->selectStudent(number)) {
        QMessageBox::warning(this, "查询结果", "未找到该学生");
    }
}

/**
 * @brief 处理修改学生按钮点击事件
 * 
 * 详细说明：
 * 1. 检查是否有选中行。如果没有，提示用户先选择。
 * 2. 获取选中行的学生信息。
 * 3. 创建 StudentDialog 对话框。
 * 4. 将学生信息填充到对话框中 (setStudent)。
 * 5. 设置学号为只读 (setReadOnlyNumber)，因为学号通常作为主键不可修改。
 * 6. 显示对话框。
 * 7. 如果用户点击确定：
 *    - 获取修改后的学生信息。
 *    - 调用 StudentManager 更新学生信息。
 *    - 根据结果提示并刷新表格。
 */
void MainWindow::on_updateStudentButton_clicked()
{
    Student stu;
    if (!tableWidget->getSelectedStudent(stu)) {
        QMessageBox::warning(this, "提示", "请先选择要修改的学生");
        return;
    }
    
    // 创建对话框并设置初始值
    StudentDialog dialog(this);
    dialog.setStudent(stu);
    dialog.setReadOnlyNumber(true); // 学号作为主键不可修改
    
    if (dialog.exec() == QDialog::Accepted) {
        Student newStu = dialog.getStudent();
        // 调用 Manager 更新学生信息
        if (StudentManager::instance().updateStudent(newStu)) {
            QMessageBox::information(this, "成功", "修改成功");
            refreshTable();
        } else {
            QMessageBox::warning(this, "失败", "修改失败");
        }
    }
}

/**
 * @brief 处理显示全部按钮点击事件
 * 
 * 详细说明：
 * 直接调用 refreshTable() 重新加载所有数据。
 */
void MainWindow::on_showAllButton_clicked()
{
    refreshTable();
}

/**
 * @brief 处理平均分统计按钮点击事件
 * 
 * 详细说明：
 * 1. 调用 StudentManager 获取格式化的平均分统计字符串。
 * 2. 如果返回空字符串（无数据），弹出警告。
 * 3. 创建 QMessageBox 显示结果。
 *    - 使用 setDetailedText 显示详细的统计表格，适合大量文本。
 */
void MainWindow::on_averageButton_clicked()
{
    std::string stats = StudentManager::instance().getAverageStats();
    if (stats.empty()) {
        QMessageBox::warning(this, "平均分统计", "当前没有学生信息或未初始化");
        return;
    }

    // 显示结果
    QMessageBox msg(this);
    msg.setWindowTitle("平均分统计");
    msg.setText("平均分表格（详情见下方）");
    msg.setDetailedText(QString::fromUtf8(stats.c_str())); // 使用 DetailedText 显示大量文本
    msg.exec();
}

/**
 * @brief 处理导出文件按钮点击事件
 * 
 * 详细说明：
 * 1. 弹出文件保存对话框 (QFileDialog::getSaveFileName)，让用户选择保存路径和文件名。
 * 2. 如果用户取消，直接返回。
 * 3. 尝试以写入模式打开文件。如果失败，提示错误。
 * 4. 使用 QTextStream 进行文本写入。
 * 5. 写入 CSV 表头。
 * 6. 获取所有学生数据，遍历并按 CSV 格式（逗号分隔）写入每一行。
 * 7. 关闭文件并提示成功。
 */
void MainWindow::on_exportButton_clicked()
{
    // 获取保存路径
    QString fileName = QFileDialog::getSaveFileName(this, "导出学生信息", "", "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)");
    if (fileName.isEmpty())
        return;

    QFile file(fileName);
    // 以文本写入模式打开文件
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, "导出失败", "无法打开文件进行写入");
        return;
    }

    QTextStream out(&file);
    // 写入 CSV 表头
    out << "学号,姓名,高数,现代,导论,心理,英语,C语言,思政,体育,创意\n";

    // 获取所有学生并写入文件
    std::vector<Student> students = StudentManager::instance().getAllStudents();
    for (const auto& stu : students) {
        out << stu.number << ","
            << QString::fromUtf8(stu.name) << ","
            << stu.GaoShu << ","
            << stu.XianDai << ","
            << stu.DaoLun << ","
            << stu.XinLi << ","
            << stu.YingYu << ","
            << stu.CYuYan << ","
            << stu.SiZheng << ","
            << stu.TiYu << ","
            << stu.ChuangYi << "\n";
    }

    file.close();
    QMessageBox::information(this, "导出成功", "学生信息已成功导出到文件");
}
