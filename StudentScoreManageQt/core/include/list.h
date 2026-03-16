#ifndef LIST_H
#define LIST_H

#include "student.h"

/**
 * @brief 创建一个新的链表节点
 * 
 * 动态分配内存给一个新的 Node 结构体，并将其 next 指针初始化为 NULL。
 * 调用者负责在不再需要时释放该节点的内存（通常通过 destroyList 或 free）。
 * 
 * @return Node* 指向新创建节点的指针。如果内存分配失败，返回 NULL。
 * 
 * This function allocates memory for a new list node and initializes it.
 */
Node* createNode();

/**
 * @brief 初始化链表
 * 
 * 将链表的头指针 (front) 设置为 NULL，并将大小 (size) 重置为 0。
 * 通常在链表首次使用前调用。
 * 
 * @param list 指向需要初始化的 List 结构体的指针
 * 
 * This function initializes a list structure, setting front to NULL and size to 0.
 */
void initList(List* list);

/**
 * @brief 销毁链表，释放内存
 * 
 * 遍历链表中的所有节点，释放每个节点占用的内存，并将链表恢复到初始化状态。
 * 防止内存泄漏。
 * 
 * @param list 指向需要销毁的 List 结构体的指针
 * 
 * This function frees all memory associated with the list nodes and resets the list.
 */
void destroyList(List* list);

#endif // LIST_H
