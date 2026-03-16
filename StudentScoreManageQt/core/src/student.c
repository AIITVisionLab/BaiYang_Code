#include "student.h"
#include "list.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/**
 * @brief 输入学生信息
 * 
 * 从标准输入读取学生的各项信息，并填充到 Student 结构体中。
 * 
 * @param stu 指向 Student 结构体的指针，用于存储输入的信息
 * 
 * This function reads student information from standard input.
 */
void inputStudentInfo(Student* stu) {
    printf("请输入学号: ");
    scanf("%llu", &stu->number);

    printf("请输入姓名: ");
    scanf("%s", stu->name);

    printf("请输入高数成绩: ");
    scanf("%f", &stu->GaoShu);

    printf("请输入现代成绩: ");
    scanf("%f", &stu->XianDai);

    printf("请输入导论成绩: ");
    scanf("%f", &stu->DaoLun);

    printf("请输入心理成绩: ");
    scanf("%f", &stu->XinLi);

    printf("请输入英语成绩: ");
    scanf("%f", &stu->YingYu);

    printf("请输入C语言成绩: ");
    scanf("%f", &stu->CYuYan);

    printf("请输入思政成绩: ");
    scanf("%f", &stu->SiZheng);

    printf("请输入体育成绩: ");
    scanf("%f", &stu->TiYu);

    printf("请输入创意成绩: ");
    scanf("%f", &stu->ChuangYi);
}

/**
 * @brief 添加学生
 * 
 * 创建新节点，输入学生信息，并将新节点添加到链表头部。
 * 
 * @param list 指向学生链表的指针
 * 
 * This function adds a new student to the linked list.
 */
void addStudent(List* list) {
    // 创建新节点
    Node* node = createNode();
    if (!node) return;

    printf("\n=== 录入学生信息 ===\n");
    // 获取用户输入
    inputStudentInfo(&node->stu);

    // 头插法插入链表
    node->next = list->front;
    list->front = node;
    list->size++;

    printf("学生信息添加成功\n");
}

/**
 * @brief 显示单个学生信息（表格行格式）
 * 
 * 以表格行的形式打印单个学生的详细信息。
 * 
 * @param stu 指向要显示的学生结构体的指针
 * 
 * This function displays a single student's info in a table row format.
 */
void displayStudent(const Student* stu) {
    printf("| %-10llu | %-8s | %-6.1f | %-6.1f | %-6.1f | %-6.1f | %-6.1f | %-7.1f | %-6.1f | %-6.1f | %-7.1f |\n",
           stu->number, stu->name,
           stu->GaoShu, stu->XianDai,
           stu->DaoLun, stu->XinLi,
           stu->YingYu, stu->CYuYan,
           stu->SiZheng, stu->TiYu,
           stu->ChuangYi);
}

/**
 * @brief 显示所有学生信息
 * 
 * 遍历链表，以表格形式打印所有学生的信息。
 * 
 * @param list 指向学生链表的指针
 * 
 * This function displays all students in the list.
 */
void displayAllStudents(List* list) {
    if (list->size == 0) {
        printf("\n当前没有学生信息\n");
        return;
    }

    // 打印表头
    printf("\n==========================================================================================================================\n");
    printf("|   学号    |   姓名  |  高数  |  现代  |  导论  |  心理  |  英语  |  C语言  |  思政  |  体育  |  创意  |\n");
    printf("==========================================================================================================================\n");

    Node* current = list->front;
    // 遍历链表
    while (current != NULL) {
        displayStudent(&current->stu);
        current = current->next;
    }

    printf("==========================================================================================================================\n");
    printf("共 %d 名学生\n", list->size);
}

/**
 * @brief 查找学生
 * 
 * 提供按学号或按姓名查找学生的功能。
 * 
 * @param list 指向学生链表的指针
 * @return Node* 找到的学生节点指针，未找到返回 NULL
 * 
 * This function searches for a student by number or name.
 */
Node* findStudent(List* list) {
    if (list->size == 0) {
        printf("当前没有学生信息\n");
        return NULL;
    }

    unsigned long long number;
    char name[50];
    int choice;

    printf("\n=== 查找学生 ===\n");
    printf("1. 按学号查找\n");
    printf("2. 按姓名查找\n");
    printf("请选择查找方式: ");
    scanf("%d", &choice);

    Node* current = list->front;

    if (choice == 1) {
        printf("请输入学号: ");
        scanf("%llu", &number);

        // 按学号遍历查找
        while (current != NULL) {
            if (current->stu.number == number) {
                return current;
            }
            current = current->next;
        }
    } else if (choice == 2) {
        printf("请输入姓名: ");
        scanf("%s", name);

        // 按姓名遍历查找
        while (current != NULL) {
            if (strcmp(current->stu.name, name) == 0) {
                return current;
            }
            current = current->next;
        }
    } else {
        printf("无效选择\n");
        return NULL;
    }

    printf("未找到该学生\n");
    return NULL;
}

/**
 * @brief 显示单个学生详细信息（列表格式）
 * 
 * 以列表形式打印单个学生的详细信息，用于修改前的确认等场景。
 * 
 * @param node 指向学生节点的指针
 * 
 * This function displays detailed info of a student.
 */
void showStudentInfo(Node* node) {
    if (!node) return;

    printf("\n=== 学生信息 ===\n");
    printf("学号: %llu\n", node->stu.number);
    printf("姓名: %s\n", node->stu.name);
    printf("高数成绩: %.1f\n", node->stu.GaoShu);
    printf("现代成绩: %.1f\n", node->stu.XianDai);
    printf("导论成绩: %.1f\n", node->stu.DaoLun);
    printf("心理成绩: %.1f\n", node->stu.XinLi);
    printf("英语成绩: %.1f\n", node->stu.YingYu);
    printf("C语言成绩: %.1f\n", node->stu.CYuYan);
    printf("思政成绩: %.1f\n", node->stu.SiZheng);
    printf("体育成绩: %.1f\n", node->stu.TiYu);
    printf("创意成绩: %.1f\n", node->stu.ChuangYi);
}

/**
 * @brief 修改学生信息
 * 
 * 先查找学生，然后允许用户输入新的信息覆盖原有信息。
 * 
 * @param list 指向学生链表的指针
 * 
 * This function modifies an existing student's information.
 */
void modifyStudent(List* list) {
    // 查找学生
    Node* node = findStudent(list);
    if (!node) return;

    printf("\n=== 当前学生信息 ===\n");
    showStudentInfo(node);

    printf("\n=== 请输入新的信息 ===\n");
    // 输入新信息
    inputStudentInfo(&node->stu);

    printf("学生信息修改成功\n");
}

/**
 * @brief 删除学生信息
 * 
 * 先查找学生，确认后从链表中移除该节点并释放内存。
 * 
 * @param list 指向学生链表的指针
 * 
 * This function deletes a student from the list.
 */
void deleteStudent(List* list) {
    if (list->size == 0) {
        printf("当前没有学生信息\n");
        return;
    }

    // 查找学生
    Node* node = findStudent(list);
    if (!node) return;

    char confirm;
    printf("\n确认要删除该学生信息吗(y/n): ");
    scanf(" %c", &confirm);

    if (confirm != 'y' && confirm != 'Y') {
        printf("取消删除操作\n");
        return;
    }

    // 从链表中移除节点
    if (list->front == node) {
        // 如果是头节点
        list->front = node->next;
    } else {
        // 如果不是头节点，找到前驱节点
        Node* prev = list->front;
        while (prev != NULL && prev->next != node) {
            prev = prev->next;
        }
        if (prev) {
            prev->next = node->next;
        }
    }

    // 释放内存
    free(node);
    list->size--;
    printf("学生信息删除成功\n");
}

/**
 * @brief 计算学生平均分
 * 
 * 遍历链表，计算每个学生的平均分并显示。
 * 
 * @param list 指向学生链表的指针
 * 
 * This function calculates and displays the average score for each student.
 */
void calculateAverage(List* list) {
    if (list->size == 0) {
        printf("当前没有学生信息\n");
        return;
    }

    printf("\n=== 学生平均分 ===\n");
    printf("===========================================\n");
    printf("|   学号    |   姓名  |   平均分  |\n");
    printf("===========================================\n");

    Node* current = list->front;
    // 遍历链表
    while (current != NULL) {
        // 计算总分
        float sum = current->stu.GaoShu + current->stu.XianDai +
                   current->stu.DaoLun + current->stu.XinLi +
                   current->stu.YingYu + current->stu.CYuYan +
                   current->stu.SiZheng + current->stu.TiYu +
                   current->stu.ChuangYi;
        // 计算平均分
        float average = sum / 9.0;

        // 显示结果
        printf("| %-10llu | %-8s |   %-8.2f  |\n",
               current->stu.number, current->stu.name, average);
        current = current->next;
    }
    printf("===========================================\n");
}
