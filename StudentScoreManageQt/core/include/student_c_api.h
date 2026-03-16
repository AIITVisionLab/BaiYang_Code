#ifndef STUDENT_C_API_H
#define STUDENT_C_API_H


#ifdef __cplusplus
extern "C" {
#endif

#include "student.h"

/**
 * @brief 初始化学生链表
 * 
 * 在使用任何其他 API 之前必须调用此函数。
 */
void apiInitList();

/**
 * @brief 销毁学生链表
 * 
 * 释放链表占用的所有内存。
 */
void apiDestroyList();

/**
 * @brief 添加新学生
 * 
 * @param number 学号
 * @param name 姓名
 * @param GaoShu 高数成绩
 * @param XianDai 现代成绩
 * @param DaoLun 导论成绩
 * @param XinLi 心理成绩
 * @param YingYu 英语成绩
 * @param CYuYan C语言成绩
 * @param SiZheng 思政成绩
 * @param TiYu 体育成绩
 * @param ChuangYi 创意成绩
 * @return int 成功返回 0，失败返回 -1
 */
int apiAddStudent(unsigned long long number, const char* name, float GaoShu, float XianDai, float DaoLun, float XinLi, float YingYu, float CYuYan, float SiZheng, float TiYu, float ChuangYi);

/**
 * @brief 删除学生
 * 
 * @param number 要删除的学生的学号
 * @return int 成功返回 0，失败返回 -1（未找到学生）
 */
int apiDeleteStudent(unsigned long long number);

/**
 * @brief 修改学生信息
 * 
 * @param number 要修改的学生的学号（作为查找依据）
 * @param name 新姓名
 * @param GaoShu 新高数成绩
 * @param XianDai 新现代成绩
 * @param DaoLun 新导论成绩
 * @param XinLi 新心理成绩
 * @param YingYu 新英语成绩
 * @param CYuYan 新C语言成绩
 * @param SiZheng 新思政成绩
 * @param TiYu 新体育成绩
 * @param ChuangYi 新创意成绩
 * @return int 成功返回 0，失败返回 -1（未找到学生）
 */
int apiUpdateStudent(unsigned long long number, const char* name, float GaoShu, float XianDai, float DaoLun, float XinLi, float YingYu, float CYuYan, float SiZheng, float TiYu, float ChuangYi);

/**
 * @brief 查询学生信息
 * 
 * @param number 学号
 * @param out_buf 输出缓冲区，用于存储格式化后的学生信息字符串
 * @param buf_size 缓冲区大小
 * @return int 成功返回 0，失败返回 -1（未找到学生）
 */
int apiQueryStudent(unsigned long long number, char* out_buf, int buf_size);

/**
 * @brief 查询所有学生信息（格式化字符串）
 * 
 * @param out_buf 输出缓冲区
 * @param buf_size 缓冲区大小
 * @return int 成功返回 0，失败返回 -1
 */
int apiQueryAllStudents(char* out_buf, int buf_size);

/**
 * @brief 获取学生总数
 * 
 * @return int 学生数量
 */
int apiGetStudentCount();

/**
 * @brief 获取指定索引的学生信息
 * 
 * 用于遍历链表。
 * @param index 索引（0 到 count-1）
 * @param out_stu 输出参数，用于存储学生结构体
 * @return int 成功返回 0，失败返回 -1（索引越界）
 */
int apiGetStudentAt(int index, Student* out_stu);

/**
 * @brief 计算平均分并输出表格字符串
 * 
 * @param out_buf 输出缓冲区
 * @param buf_size 缓冲区大小
 * @return int 成功返回 0，失败返回 -1
 */
int apiCalculateAverage(char* out_buf, int buf_size);

#ifdef __cplusplus
}
#endif

#endif // STUDENT_C_API_H
