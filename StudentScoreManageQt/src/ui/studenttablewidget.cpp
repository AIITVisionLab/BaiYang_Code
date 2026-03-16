#include "studenttablewidget.h"
#include <QHeaderView>
#include <QStringList>

/**
 * @brief 构造函数
 * 
 * 详细说明：
 * 1. 调用父类 QTableWidget 构造函数。
 * 2. 调用 setupHeader 初始化表头。
 * 3. 设置选择行为为"整行选择" (SelectRows)。
 * 4. 设置编辑触发器为"不可编辑" (NoEditTriggers)，防止用户直接在表格中修改数据。
 * 5. 设置样式表：
 *    - 背景半透明白色。
 *    - 交替行背景色。
 *    - 网格线颜色。
 *    - 字体颜色为黑色。
 *    - 表头样式。
 * 
 * @param parent 父窗口指针
 */
StudentTableWidget::StudentTableWidget(QWidget *parent)
    : QTableWidget(parent)
{
    setupHeader();
    
    // 整行选择模式
    setSelectionBehavior(QAbstractItemView::SelectRows);
    // 禁止直接在表格中编辑
    setEditTriggers(QAbstractItemView::NoEditTriggers);
    
    // 设置表格半透明背景，并强制设置字体颜色为黑色
    setStyleSheet("QTableWidget { background-color: rgba(255, 255, 255, 180); alternate-background-color: rgba(240, 240, 240, 180); gridline-color: #ccc; color: #000000; } QHeaderView::section { background-color: rgba(200, 200, 200, 200); color: #000000; font-weight: bold; }");
}

/**
 * @brief 初始化表头
 * 
 * 详细说明：
 * 1. 定义包含所有列名的 QStringList。
 * 2. 设置表格列数。
 * 3. 设置水平表头标签。
 * 4. 设置表头自适应宽度模式 (Stretch)，使列宽自动填满表格宽度。
 */
void StudentTableWidget::setupHeader()
{
    QStringList headers;
    headers << "学号" << "姓名" << "高数" << "现代" << "导论" << "心理" << "英语" << "C语言" << "思政" << "体育" << "创意";
    setColumnCount(headers.size());
    setHorizontalHeaderLabels(headers);
    // 表头自适应宽度
    horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
}

/**
 * @brief 刷新表格数据
 * 
 * 详细说明：
 * 1. 调用 setRowCount(0) 清空表格所有现有行。
 * 2. 遍历传入的学生列表 (students)。
 * 3. 对每个学生：
 *    - 获取当前行号 (rowCount)。
 *    - 插入新行 (insertRow)。
 *    - 创建 QTableWidgetItem 并设置对应的数据（学号、姓名、成绩）。
 *    - 注意：姓名需要处理 UTF-8 编码，成绩保留一位小数。
 *    - 将 Item 设置到对应的单元格 (setItem)。
 * 
 * @param students 学生数据列表
 */
void StudentTableWidget::refresh(const std::vector<Student>& students)
{
    setRowCount(0); // 清空表格所有行
    
    // 遍历所有学生
    for (const auto& stu : students) {
        int row = rowCount();
        insertRow(row); // 插入新行
        
        // 设置各列数据
        setItem(row, 0, new QTableWidgetItem(QString::number(stu.number)));
        setItem(row, 1, new QTableWidgetItem(QString::fromUtf8(stu.name))); // 处理中文编码
        setItem(row, 2, new QTableWidgetItem(QString::number(stu.GaoShu, 'f', 1)));
        setItem(row, 3, new QTableWidgetItem(QString::number(stu.XianDai, 'f', 1)));
        setItem(row, 4, new QTableWidgetItem(QString::number(stu.DaoLun, 'f', 1)));
        setItem(row, 5, new QTableWidgetItem(QString::number(stu.XinLi, 'f', 1)));
        setItem(row, 6, new QTableWidgetItem(QString::number(stu.YingYu, 'f', 1)));
        setItem(row, 7, new QTableWidgetItem(QString::number(stu.CYuYan, 'f', 1)));
        setItem(row, 8, new QTableWidgetItem(QString::number(stu.SiZheng, 'f', 1)));
        setItem(row, 9, new QTableWidgetItem(QString::number(stu.TiYu, 'f', 1)));
        setItem(row, 10, new QTableWidgetItem(QString::number(stu.ChuangYi, 'f', 1)));
    }
}

/**
 * @brief 获取当前选中行的学生学号
 * 
 * 详细说明：
 * 1. 获取当前选中行的索引 (currentRow)。
 * 2. 如果有选中行 (row >= 0)：
 *    - 获取第 0 列（学号列）的文本。
 *    - 转换为 unsigned long long 类型并返回。
 * 3. 如果没有选中行，返回 0。
 * 
 * @return unsigned long long 学号，未选中返回 0
 */
unsigned long long StudentTableWidget::getSelectedStudentNumber() const
{
    int row = currentRow();
    if (row >= 0) {
        return item(row, 0)->text().toULongLong();
    }
    return 0;
}

/**
 * @brief 获取当前选中行的完整学生信息
 * 
 * 详细说明：
 * 1. 获取当前选中行的索引。
 * 2. 如果没有选中行，返回 false。
 * 3. 从各列获取文本并解析为对应的数据类型，填充到 stu 结构体中。
 *    - 姓名需要从 QString 转换为 C 字符串。
 * 4. 返回 true。
 * 
 * @param stu 用于存储学生信息的引用
 * @return true 获取成功
 * @return false 未选中行
 */
bool StudentTableWidget::getSelectedStudent(Student& stu) const
{
    int row = currentRow();
    if (row < 0) return false;

    stu.number = item(row, 0)->text().toULongLong();
    strncpy(stu.name, item(row, 1)->text().toUtf8().data(), sizeof(stu.name)-1);
    stu.GaoShu = item(row, 2)->text().toFloat();
    stu.XianDai = item(row, 3)->text().toFloat();
    stu.DaoLun = item(row, 4)->text().toFloat();
    stu.XinLi = item(row, 5)->text().toFloat();
    stu.YingYu = item(row, 6)->text().toFloat();
    stu.CYuYan = item(row, 7)->text().toFloat();
    stu.SiZheng = item(row, 8)->text().toFloat();
    stu.TiYu = item(row, 9)->text().toFloat();
    stu.ChuangYi = item(row, 10)->text().toFloat();
    
    return true;
}

/**
 * @brief 根据学号选中并高亮显示学生
 * 
 * 详细说明：
 * 1. 遍历表格的所有行。
 * 2. 比较每一行第 0 列（学号）的文本与目标 number。
 * 3. 如果找到匹配行：
 *    - 调用 selectRow(i) 选中该行。
 *    - 调用 scrollToItem 滚动到该行，确保可见。
 *    - 返回 true。
 * 4. 如果遍历结束未找到，返回 false。
 * 
 * @param number 学号
 * @return true 找到并选中
 * @return false 未找到
 */
bool StudentTableWidget::selectStudent(unsigned long long number)
{
    for(int i=0; i<rowCount(); ++i) {
        if(item(i, 0)->text().toULongLong() == number) {
            selectRow(i);
            scrollToItem(item(i, 0));
            return true;
        }
    }
    return false;
}
