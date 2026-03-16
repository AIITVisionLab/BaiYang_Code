#ifndef UTILS_H
#define UTILS_H

/**
 * @brief 清屏函数
 * 
 * 清除终端屏幕上的所有内容。
 * 适配 Windows (cls) 和 Linux (clear) 操作系统。
 * 
 * Clears the terminal screen, supporting both Windows and Linux.
 */
void clearScreen();

/**
 * @brief 暂停程序，等待用户按键
 * 
 * 暂停程序的执行，提示用户按 Enter 键继续。
 * 用于在控制台程序中让用户有时间查看输出结果。
 * 
 * Pauses the program execution and waits for the user to press Enter.
 */
void pause();

#endif // UTILS_H
