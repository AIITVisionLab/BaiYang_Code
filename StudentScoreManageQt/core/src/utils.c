#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief 清屏函数
 * 
 * 根据操作系统类型调用相应的系统命令来清除终端屏幕。
 * Windows 使用 "cls"，Linux/Unix 使用 "clear"。
 * 
 * This function clears the terminal screen, adapting to Windows or Linux.
 */
void clearScreen() {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

/**
 * @brief 暂停程序执行
 * 
 * 提示用户按 Enter 键继续，并等待用户输入。
 * 
 * This function pauses the program and waits for the user to press Enter.
 */
void pause() {
    printf("\n按Enter键继续...");
    // 第一次 getchar() 可能会读取到上一次输入遗留的换行符
    getchar();
    // 第二次 getchar() 等待用户实际按键
    getchar();
}
