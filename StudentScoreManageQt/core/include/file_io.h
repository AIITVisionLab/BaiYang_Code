#ifndef FILEIO_H
#define FILEIO_H

#include "student.h"

/**
 * @brief 将学生数据保存到二进制文件 (students.dat)
 * 
 * 该函数将链表中的所有学生记录以二进制格式写入到名为 "students.dat" 的文件中。
 * 如果文件打开失败，会在控制台输出错误信息。
 * 
 * @param list 指向包含学生数据的链表的指针
 * 
 * This function saves all student records in the list to a binary file named "students.dat".
 */
void saveToBinaryFile(List* list);

/**
 * @brief 将学生数据导出为文本文件 (students.txt)
 * 
 * 该函数将链表中的所有学生记录以制表符分隔的文本格式写入到名为 "students.txt" 的文件中。
 * 这种格式便于人类阅读和导入到 Excel 等工具中。
 * 
 * @param list 指向包含学生数据的链表的指针
 * 
 * This function exports all student records to a text file named "students.txt" in a readable format.
 */
void saveToTextFile(List* list);

/**
 * @brief 从二进制文件加载学生数据
 * 
 * 该函数从 "students.dat" 文件中读取学生记录，并将其添加到链表中。
 * 在加载之前，会先清空链表中现有的数据。
 * 如果文件不存在，函数会提示并将创建一个新文件（在下次保存时）。
 * 
 * @param list 指向将要存储学生数据的链表的指针
 * 
 * This function loads student records from "students.dat" into the linked list.
 */
void loadFromBinaryFile(List* list);

#endif // FILEIO_H