#include <QApplication>
#include "mainwindow.h"

/**
 * @brief 应用程序入口点
 * 
 * 初始化 Qt 应用程序对象，创建并显示主窗口，然后进入事件循环。
 * 
 * @param argc 命令行参数数量
 * @param argv 命令行参数数组
 * @return int 应用程序退出代码
 * 
 * Main entry point of the application.
 * Initializes the Qt application, shows the main window, and starts the event loop.
 */
int main(int argc, char *argv[])
{
    // 创建 Qt 应用程序实例
    QApplication a(argc, argv);
    
    // 创建主窗口实例
    MainWindow w;
    
    // 显示主窗口
    w.show();
    
    // 进入应用程序事件循环
    return a.exec();
}
