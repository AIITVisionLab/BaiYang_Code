#include "student.h"
#include "file_io.h"
#include "list.h"
#include <stdio.h>

/**
 * @brief 将学生链表保存到二进制文件
 * 
 * 将链表中的所有学生数据以二进制格式写入到 "students.dat" 文件中。
 * 
 * @param list 指向学生链表的指针
 * 
 * This function saves the student list to a binary file named "students.dat".
 */
void saveToBinaryFile(List* list) {
    // 以二进制写模式打开文件
    FILE* fp = fopen("students.dat", "wb");
    if (!fp) {
        printf("文件打开失败\n");
        return;
    }

    Node* current = list->front;
    int count = 0;
    // 遍历链表
    while (current != NULL) {
        // 将当前节点的学生数据写入文件
        fwrite(&current->stu, sizeof(Student), 1, fp);
        current = current->next;
        count++;
    }

    // 关闭文件
    fclose(fp);
    printf("成功保存 %d 名学生记录到 students.dat\n", count);
}

/**
 * @brief 将学生链表保存到文本文件
 * 
 * 将链表中的所有学生数据以可读的文本格式写入到 "students.txt" 文件中。
 * 
 * @param list 指向学生链表的指针
 * 
 * This function saves the student list to a text file named "students.txt".
 */
void saveToTextFile(List* list) {
    // 以文本写模式打开文件
    FILE* fp = fopen("students.txt", "w");
    if (!fp) {
        printf("文件打开失败\n");
        return;
    }

    // 写入表头
    fprintf(fp, "学号\t姓名\t高数\t现代\t导论\t心理\t英语\tC语言\t道法\t体育\t创意\n");

    Node* current = list->front;
    int count = 0;
    // 遍历链表
    while (current != NULL) {
        // 将当前节点的学生数据格式化写入文件
        fprintf(fp, "%llu\t%s\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\n",
                current->stu.number, current->stu.name,
                current->stu.GaoShu, current->stu.XianDai,
                current->stu.DaoLun, current->stu.XinLi,
                current->stu.YingYu, current->stu.CYuYan,
                current->stu.SiZheng, current->stu.TiYu,
                current->stu.ChuangYi);
        current = current->next;
        count++;
    }

    // 关闭文件
    fclose(fp);
    printf("成功保存 %d 名学生记录到 students.txt\n", count);
}

/**
 * @brief 从二进制文件加载学生数据到链表
 * 
 * 从 "students.dat" 文件中读取学生数据，并重建链表。
 * 加载前会清空现有链表。
 * 
 * @param list 指向学生链表的指针
 * 
 * This function loads student data from "students.dat" into the linked list.
 */
void loadFromBinaryFile(List* list) {
    // 以二进制读模式打开文件
    FILE* fp = fopen("students.dat", "rb");
    if (!fp) {
        printf("未找到数据文件，将创建新文件\n");
        return;
    }

    // 加载前先销毁现有链表，防止内存泄漏或数据重复
    destroyList(list);

    Student stu;
    int count = 0;
    // 循环读取文件中的学生数据
    while (fread(&stu, sizeof(Student), 1, fp) == 1) {
        // 创建新节点
        Node* node = createNode();
        if (!node) break;

        // 填充数据并插入链表头部
        node->stu = stu;
        node->next = list->front;
        list->front = node;
        list->size++;
        count++;
    }

    // 关闭文件
    fclose(fp);
    printf("成功从 students.dat 加载 %d 名学生记录\n", count);
}
