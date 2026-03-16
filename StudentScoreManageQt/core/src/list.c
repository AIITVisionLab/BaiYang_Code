#include "student.h"
#include "list.h"
#include <stdlib.h>
#include <stdio.h>

/**
 * @brief 创建一个新的链表节点
 * 
 * 详细说明：
 * 1. 使用 malloc 分配内存给一个新的 Node 结构体。
 * 2. 检查内存分配是否成功。如果失败，打印错误信息并返回 NULL。
 * 3. 将新节点的 next 指针初始化为 NULL，表示它目前没有后续节点。
 * 4. 返回指向新节点的指针。
 * 
 * @return Node* 指向新创建节点的指针，如果内存分配失败则返回 NULL
 */
Node* createNode() {
    // 1. 动态分配内存，大小为一个 Node 结构体
    Node* node = (Node*)malloc(sizeof(Node));
    
    // 2. 检查内存分配结果
    if (!node) {
        printf("内存分配失败\n");
        return NULL;
    }
    
    // 3. 初始化 next 指针为 NULL，防止野指针
    node->next = NULL;
    
    // 4. 返回新节点指针
    return node;
}

/**
 * @brief 初始化链表
 * 
 * 详细说明：
 * 1. 接收一个 List 结构体指针。
 * 2. 将链表的头指针 (front) 设置为 NULL，表示链表初始为空。
 * 3. 将链表的大小 (size) 设置为 0。
 * 
 * @param list 指向要初始化的链表的指针
 */
void initList(List* list) {
    // 将头指针置空
    list->front = NULL;
    // 将节点数量置零
    list->size = 0;
}

/**
 * @brief 销毁链表
 * 
 * 详细说明：
 * 1. 从链表头开始遍历整个链表。
 * 2. 使用临时指针 temp 保存当前节点，以便释放内存。
 * 3. 将 current 指针移动到下一个节点。
 * 4. 释放 temp 指向的节点内存。
 * 5. 重复步骤 2-4 直到链表末尾。
 * 6. 最后将链表头指针置空，大小置零，恢复到初始化状态。
 * 
 * @param list 指向要销毁的链表的指针
 */
void destroyList(List* list) {
    Node* current = list->front;
    // 遍历链表中的每一个节点
    while (current != NULL) {
        Node* temp = current;
        // 移动到下一个节点，保存当前链表结构
        current = current->next;
        // 释放当前节点的内存
        free(temp);
    }
    // 重置链表头指针和大小，防止悬空指针
    list->front = NULL;
    list->size = 0;
}
