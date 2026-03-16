
#include "student_c_api.h"
#include "list.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// 全局链表实例，用于在整个程序生命周期中存储所有学生数据
// 这是一个静态变量，仅在当前文件中可见，通过 API 函数进行访问
static List g_list;

// 初始化标志，用于防止重复初始化或在未初始化时进行操作
// 0: 未初始化, 1: 已初始化
static int g_inited = 0;

/**
 * @brief 初始化全局学生链表
 * 
 * 详细说明：
 * 1. 检查全局标志 g_inited。
 * 2. 如果未初始化 (!g_inited)，调用 initList(&g_list) 初始化全局链表结构。
 * 3. 将 g_inited 设置为 1，标记系统已准备就绪。
 */
void apiInitList() {
    // 只有在未初始化的情况下才执行初始化操作
    if (!g_inited) {
        // 调用底层 list.c 的初始化函数
        initList(&g_list);
        // 设置标志位
        g_inited = 1;
    }
}

/**
 * @brief 销毁全局学生链表
 * 
 * 详细说明：
 * 1. 检查全局标志 g_inited。
 * 2. 如果已初始化 (g_inited)，调用 destroyList(&g_list) 释放链表中所有节点的内存。
 * 3. 将 g_inited 重置为 0，防止后续非法访问。
 */
void apiDestroyList() {
    // 只有在已初始化的情况下才执行销毁操作
    if (g_inited) {
        // 调用底层 list.c 的销毁函数，释放所有内存
        destroyList(&g_list);
        // 重置标志位
        g_inited = 0;
    }
}

/**
 * @brief 添加新学生到全局链表
 * 
 * 详细说明：
 * 1. 检查系统是否已初始化，若未初始化则自动调用 apiInitList()。
 * 2. 调用 createNode() 创建一个新的节点。如果内存不足，返回 -1。
 * 3. 将传入的学生信息（学号、姓名、各科成绩）填充到新节点的 student 结构体中。
 *    - 使用 strncpy 安全复制姓名，防止缓冲区溢出。
 * 4. 使用"头插法"将新节点插入到链表头部：
 *    - 新节点的 next 指向当前的头节点 (g_list.front)。
 *    - 更新头节点 (g_list.front) 指向新节点。
 * 5. 增加链表大小计数 (g_list.size)。
 * 
 * @param number 学号
 * @param name 姓名
 * @param GaoShu ... (各科成绩)
 * @return int 成功返回 0，失败返回 -1
 */
int apiAddStudent(unsigned long long number, const char* name, float GaoShu, float XianDai, float DaoLun, float XinLi, float YingYu, float CYuYan, float SiZheng, float TiYu, float ChuangYi) {
    // 1. 自动初始化检查
    if (!g_inited) apiInitList();
    
    // 2. 创建新节点
    Node* node = createNode();
    // 内存分配失败检查
    if (!node) return -1;
    
    // 3. 填充学生数据
    node->stu.number = number;
    // 安全字符串复制：最多复制 sizeof(name)-1 个字符
    strncpy(node->stu.name, name, sizeof(node->stu.name)-1);
    // 确保字符串末尾有结束符
    node->stu.name[sizeof(node->stu.name)-1] = '\0'; 
    
    // 填充成绩数据
    node->stu.GaoShu = GaoShu;
    node->stu.XianDai = XianDai;
    node->stu.DaoLun = DaoLun;
    node->stu.XinLi = XinLi;
    node->stu.YingYu = YingYu;
    node->stu.CYuYan = CYuYan;
    node->stu.SiZheng = SiZheng;
    node->stu.TiYu = TiYu;
    node->stu.ChuangYi = ChuangYi;
    
    // 4. 执行头插法插入操作
    // 新节点的 next 指向原头节点
    node->next = g_list.front;
    // 更新头指针指向新节点
    g_list.front = node;
    
    // 5. 更新链表计数
    g_list.size++;
    
    return 0;
}

/**
 * @brief 从全局链表中删除学生
 * 
 * 详细说明：
 * 1. 检查初始化状态。
 * 2. 使用两个指针 prev 和 curr 遍历链表。
 *    - curr 指向当前检查的节点。
 *    - prev 指向 curr 的前一个节点（用于删除操作）。
 * 3. 遍历链表，比较 curr->stu.number 与目标 number。
 * 4. 如果找到匹配节点：
 *    - 如果 prev 不为 NULL（即删除的不是头节点），将 prev->next 指向 curr->next，跳过 curr。
 *    - 如果 prev 为 NULL（即删除的是头节点），将 g_list.front 更新为 curr->next。
 *    - 使用 free(curr) 释放节点内存。
 *    - 减少链表大小计数 (g_list.size)。
 *    - 返回 0 表示成功。
 * 5. 如果遍历结束仍未找到，返回 -1。
 * 
 * @param number 要删除的学生的学号
 * @return int 成功返回 0，失败（未找到或未初始化）返回 -1
 */
int apiDeleteStudent(unsigned long long number) {
    // 1. 检查初始化
    if (!g_inited) return -1;
    
    Node* prev = NULL;
    Node* curr = g_list.front;
    
    // 2. & 3. 遍历链表
    while (curr) {
        // 4. 找到匹配的学号
        if (curr->stu.number == number) {
            // 执行删除逻辑
            if (prev) {
                // 中间或尾部节点：前驱节点指向后继节点
                prev->next = curr->next;
            } else {
                // 头节点：头指针指向第二个节点
                g_list.front = curr->next; 
            }
            
            // 释放内存
            free(curr);
            // 更新计数
            g_list.size--;
            return 0;
        }
        // 移动指针继续查找
        prev = curr;
        curr = curr->next;
    }
    // 5. 未找到指定学号的学生
    return -1;
}

/**
 * @brief 更新全局链表中的学生信息
 * 
 * 详细说明：
 * 1. 检查初始化状态。
 * 2. 遍历链表查找与给定学号 (number) 匹配的节点。
 * 3. 如果找到匹配节点：
 *    - 更新姓名：使用 strncpy 安全复制，并强制添加 null 结束符。
 *    - 更新所有科目的成绩。
 *    - 返回 0 表示成功。
 * 4. 如果遍历结束未找到，返回 -1。
 * 
 * @param number 学号（查找键）
 * @param name 新姓名
 * @param GaoShu ... (新成绩)
 * @return int 成功返回 0，失败（未找到或未初始化）返回 -1
 */
int apiUpdateStudent(unsigned long long number, const char* name, float GaoShu, float XianDai, float DaoLun, float XinLi, float YingYu, float CYuYan, float SiZheng, float TiYu, float ChuangYi) {
    // 1. 检查初始化
    if (!g_inited) return -1;
    
    Node* curr = g_list.front;
    // 2. 遍历链表
    while (curr) {
        // 3. 找到匹配节点
        if (curr->stu.number == number) {
            // 更新姓名
            strncpy(curr->stu.name, name, sizeof(curr->stu.name)-1);
            curr->stu.name[sizeof(curr->stu.name)-1] = '\0';
            
            // 更新成绩
            curr->stu.GaoShu = GaoShu;
            curr->stu.XianDai = XianDai;
            curr->stu.DaoLun = DaoLun;
            curr->stu.XinLi = XinLi;
            curr->stu.YingYu = YingYu;
            curr->stu.CYuYan = CYuYan;
            curr->stu.SiZheng = SiZheng;
            curr->stu.TiYu = TiYu;
            curr->stu.ChuangYi = ChuangYi;
            return 0;
        }
        curr = curr->next;
    }
    // 4. 未找到
    return -1;
}

/**
 * @brief 查询单个学生信息并格式化输出
 * 
 * 详细说明：
 * 1. 检查初始化状态。
 * 2. 遍历链表查找指定学号的学生。
 * 3. 如果找到：
 *    - 使用 snprintf 将学生详细信息格式化为多行字符串。
 *    - 写入到提供的输出缓冲区 out_buf 中。
 *    - 返回 0。
 * 4. 如果未找到：
 *    - 将"未找到该学生"写入缓冲区。
 *    - 返回 -1。
 * 
 * @param number 要查询的学号
 * @param out_buf 输出缓冲区
 * @param buf_size 缓冲区大小
 * @return int 成功返回 0，失败返回 -1
 */
int apiQueryStudent(unsigned long long number, char* out_buf, int buf_size) {
    if (!g_inited) return -1;
    
    Node* curr = g_list.front;
    while (curr) {
        if (curr->stu.number == number) {
            // 格式化输出
            snprintf(out_buf, buf_size, "学号:%llu\n姓名:%s\n高数:%.1f\n现代:%.1f\n导论:%.1f\n心理:%.1f\n英语:%.1f\nC语言:%.1f\n思政:%.1f\n体育:%.1f\n创意:%.1f", curr->stu.number, curr->stu.name, curr->stu.GaoShu, curr->stu.XianDai, curr->stu.DaoLun, curr->stu.XinLi, curr->stu.YingYu, curr->stu.CYuYan, curr->stu.SiZheng, curr->stu.TiYu, curr->stu.ChuangYi);
            return 0;
        }
        curr = curr->next;
    }
    strncpy(out_buf, "未找到该学生", buf_size);
    return -1;
}

/**
 * @brief 查询所有学生信息并格式化为表格
 * 
 * 详细说明：
 * 1. 检查初始化状态。
 * 2. 初始化 used 变量记录缓冲区已使用的字节数。
 * 3. 写入表头（学号、姓名、各科成绩）。
 * 4. 遍历链表：
 *    - 检查缓冲区剩余空间是否足够。
 *    - 将每个学生的信息格式化为一行，追加到缓冲区。
 *    - 更新 used 计数。
 * 5. 返回 0。
 * 
 * @param out_buf 输出缓冲区
 * @param buf_size 缓冲区大小
 * @return int 成功返回 0，失败返回 -1
 */
int apiQueryAllStudents(char* out_buf, int buf_size) {
    if (!g_inited) return -1;
    
    Node* curr = g_list.front;
    int used = 0;
    
    // 写入表头
    used += snprintf(out_buf+used, buf_size-used, "学号\t姓名\t高数\t现代\t导论\t心理\t英语\tC语言\t思政\t体育\t创意\n");
    
    // 遍历并写入数据
    while (curr && used < buf_size-1) {
        used += snprintf(out_buf+used, buf_size-used, "%llu\t%s\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\n", curr->stu.number, curr->stu.name, curr->stu.GaoShu, curr->stu.XianDai, curr->stu.DaoLun, curr->stu.XinLi, curr->stu.YingYu, curr->stu.CYuYan, curr->stu.SiZheng, curr->stu.TiYu, curr->stu.ChuangYi);
        curr = curr->next;
    }
    return 0;
}

/**
 * @brief 计算每个学生的平均分并格式化输出
 * 
 * 详细说明：
 * 1. 检查初始化状态。
 * 2. 写入表头（学号、姓名、平均分）。
 * 3. 遍历链表：
 *    - 计算当前学生9门课程的总分。
 *    - 计算平均分 (总分 / 9.0)。
 *    - 将学号、姓名和平均分格式化为一行，追加到缓冲区。
 * 4. 返回 0。
 * 
 * @param out_buf 输出缓冲区
 * @param buf_size 缓冲区大小
 * @return int 成功返回 0，失败返回 -1
 */
int apiCalculateAverage(char* out_buf, int buf_size) {
    if (!g_inited) return -1;
    
    Node* current = g_list.front;
    int used = 0;
    
    // 写入表头
    used += snprintf(out_buf + used, buf_size - used,
                    "学号\t姓名\t平均分\n");

    // 遍历并计算
    while (current && used < buf_size - 1) {
        // 计算总分
        float sum = current->stu.GaoShu + current->stu.XianDai +
                    current->stu.DaoLun + current->stu.XinLi +
                    current->stu.YingYu + current->stu.CYuYan +
                    current->stu.SiZheng + current->stu.TiYu +
                    current->stu.ChuangYi;
        // 计算平均分
        float average = sum / 9.0f;

        // 格式化输出
        used += snprintf(out_buf + used, buf_size - used,
                        "%llu\t%s\t%.2f\n",
                        current->stu.number, current->stu.name, average);
        current = current->next;
    }
    return 0;
}

/**
 * @brief 获取学生总数
 * 
 * 详细说明：
 * 直接返回全局链表结构中的 size 字段。
 * 
 * @return int 学生总数，如果未初始化则返回 0
 */
int apiGetStudentCount() {
    if (!g_inited) return 0;
    return g_list.size;
}

/**
 * @brief 获取指定索引的学生信息
 * 
 * 详细说明：
 * 1. 检查初始化状态和索引有效性（0 <= index < size）。
 * 2. 遍历链表，使用计数器 i 记录当前节点索引。
 * 3. 当 i 等于目标 index 时：
 *    - 将当前节点的 student 数据复制到输出参数 out_stu 指向的内存。
 *    - 返回 0。
 * 4. 如果遍历结束未找到（理论上不应发生），返回 -1。
 * 
 * @param index 索引值
 * @param out_stu 用于存储获取到的学生信息的指针
 * @return int 成功返回 0，失败返回 -1
 */
int apiGetStudentAt(int index, Student* out_stu) {
    if (!g_inited || index < 0 || index >= g_list.size) return -1;
    
    Node* curr = g_list.front;
    int i = 0;
    
    while (curr) {
        if (i == index) {
            *out_stu = curr->stu;
            return 0;
        }
        curr = curr->next;
        i++;
    }
    return -1;
}
