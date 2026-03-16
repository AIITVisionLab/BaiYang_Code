#ifndef STUDENT_H
#define STUDENT_H

#include <stdio.h>

/**
 * @brief 学生信息结构体
 * 
 * 存储单个学生的详细信息，包括学号、姓名以及各科成绩。
 * 
 * Structure to store detailed information of a single student.
 */
typedef struct {
    unsigned long long number;  // 学号 (Student ID)
    char name[50];              // 姓名 (Name)
    float GaoShu;               // 高等数学成绩 (Advanced Mathematics Score)
    float XianDai;              // 线性代数成绩 (Linear Algebra Score)
    float DaoLun;               // 专业导论成绩 (Introduction to Major Score)
    float XinLi;                // 心理健康成绩 (Mental Health Score)
    float YingYu;               // 大学英语成绩 (College English Score)
    float CYuYan;               // C语言程序设计成绩 (C Programming Score)
    float SiZheng;              // 思想政治成绩 (Ideological and Political Education Score)
    float TiYu;                 // 体育成绩 (Physical Education Score)
    float ChuangYi;             // 创新创意成绩 (Innovation and Creativity Score)
} Student;

/**
 * @brief 链表节点结构体
 * 
 * 单向链表的节点，包含一个学生数据结构和一个指向下一个节点的指针。
 * 
 * Linked list node structure containing student data and a pointer to the next node.
 */
typedef struct Node {
    Student stu;        // 学生数据 (Student data)
    struct Node* next;  // 指向下一个节点的指针 (Pointer to the next node)
} Node;

/**
 * @brief 链表结构体
 * 
 * 包含链表的头指针和链表的大小（节点数量）。
 * 
 * List structure containing the head pointer and the size of the list.
 */
typedef struct {
    Node* front;    // 链表头指针 (Head pointer of the list)
    int size;       // 链表节点数量 (Number of nodes in the list)
} List;

// --- 学生信息管理函数 (CLI Only - Not used in Qt) ---

/*
void addStudent(List* list);
void displayAllStudents(List* list);
Node* findStudent(List* list);
void showStudentInfo(Node* node);
void modifyStudent(List* list);
void deleteStudent(List* list);
void calculateAverage(List* list);
*/

#endif // STUDENT_H
