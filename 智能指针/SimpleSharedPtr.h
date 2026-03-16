//
// Created by bai on 2026/3/16.
//

#ifndef SMARTPTRPROJECT_SIMPLESHAREDPTR_H
#define SMARTPTRPROJECT_SIMPLESHAREDPTR_H

#include <iostream>

/**
 * @brief 控制块
 *
 * 用于记录当前资源被多少个 SimpleSharedPtr 对象共享。
 * 当引用计数降为 0 时，说明没有任何智能指针再管理该资源，
 * 此时释放资源和控制块本身。
 */
struct ControlBlock {
    int ref_count;

    ControlBlock() : ref_count(1) {}
};

/**
 * @brief 简易版共享智能指针
 *
 * 功能：
 * 1. 支持资源共享
 * 2. 支持拷贝构造、拷贝赋值
 * 3. 支持移动构造、移动赋值
 * 4. 支持 use_count() 查看引用计数
 * 5. 支持 reset() 重新绑定资源
 *
 * 注意：
 * 不具备 std::shared_ptr 的线程安全、
 * 自定义删除器、weak_ptr 等高级功能。
 */
template <typename T>
class SimpleSharedPtr {
private:
    T* ptr;                  // 实际管理的堆对象指针
    ControlBlock* control;   // 指向控制块，控制引用计数

    /**
     * @brief 释放当前对象对资源的所有权
     *
     * 逻辑：
     * 1. 如果 control 不为空，说明当前对象正在参与共享
     * 2. 引用计数减 1
     * 3. 如果减到 0，说明当前对象是最后一个拥有者，需要释放资源
     */
    void release() {
        if (control) {
            --control->ref_count;

            if (control->ref_count == 0) {
                delete ptr;
                delete control;
            }

            // 无论计数是否为 0，当前对象都不应再持有旧资源
            ptr = nullptr;
            control = nullptr;
        }
    }

public:
    /**
     * @brief 默认构造
     *
     * 创建一个空智能指针，不管理任何资源。
     */
    SimpleSharedPtr() : ptr(nullptr), control(nullptr) {}

    /**
     * @brief 用原始指针构造智能指针
     * @param p 要托管的堆对象指针
     *
     * 如果 p 不为空，则创建控制块，引用计数初始化为 1。
     */
    explicit SimpleSharedPtr(T* p) : ptr(p) {
        if (p) {
            control = new ControlBlock();
        } else {
            control = nullptr;
        }
    }

    /**
     * @brief 析构函数
     *
     * 对当前资源执行一次 release。
     */
    ~SimpleSharedPtr() {
        release();
    }

    /**
     * @brief 拷贝构造函数
     * @param other 另一个智能指针
     *
     * 拷贝后两个智能指针共享同一块资源，
     * 所以引用计数需要加 1。
     */
    SimpleSharedPtr(const SimpleSharedPtr& other)
        : ptr(other.ptr), control(other.control) {
        if (control) {
            ++control->ref_count;
        }
    }

    /**
     * @brief 拷贝赋值运算符
     * @param other 另一个智能指针
     * @return 当前对象引用
     *
     * 步骤：
     * 1. 防止自赋值
     * 2. 先释放当前对象原来管理的资源
     * 3. 再共享 other 的资源
     * 4. 引用计数加 1
     */
    SimpleSharedPtr& operator=(const SimpleSharedPtr& other) {
        if (this != &other) {
            release();

            ptr = other.ptr;
            control = other.control;

            if (control) {
                ++control->ref_count;
            }
        }
        return *this;
    }

    /**
     * @brief 移动构造函数
     * @param other 被转移资源的对象
     *
     * 直接“偷走” other 的资源所有权，
     * 不增加引用计数。
     *
     * 移动后 other 变为空指针状态。
     */
    SimpleSharedPtr(SimpleSharedPtr&& other) noexcept
        : ptr(other.ptr), control(other.control) {
        other.ptr = nullptr;
        other.control = nullptr;
    }

    /**
     * @brief 移动赋值运算符
     * @param other 被转移资源的对象
     * @return 当前对象引用
     *
     * 步骤：
     * 1. 防止自赋值
     * 2. 先释放当前对象原有资源
     * 3. 接管 other 的资源
     * 4. 将 other 置空
     */
    SimpleSharedPtr& operator=(SimpleSharedPtr&& other) noexcept {
        if (this != &other) {
            release();

            ptr = other.ptr;
            control = other.control;

            other.ptr = nullptr;
            other.control = nullptr;
        }
        return *this;
    }

    /**
     * @brief 成员访问运算符
     * @return 被管理对象的原始指针
     *
     * 用法：
     * sp->name = "Tom";
     */
    T* operator->() const {
        return ptr;
    }

    /**
     * @brief 解引用运算符
     * @return 被管理对象的引用
     *
     * 用法：
     * (*sp).name = "Tom";
     */
    T& operator*() const {
        return *ptr;
    }

    /**
     * @brief 获取原始指针
     * @return 当前管理的原始指针
     *
     * 注意：返回原始指针不意味着转移所有权。
     */
    T* get() const {
        return ptr;
    }

    /**
     * @brief 获取当前引用计数
     * @return 当前共享资源的拥有者数量
     */
    int use_count() const {
        return control ? control->ref_count : 0;
    }

    /**
     * @brief 重置智能指针
     * @param p 新的原始指针，默认 nullptr
     *
     * 作用：
     * 1. 先释放当前资源
     * 2. 再托管新的资源
     */
    void reset(T* p = nullptr) {
        release();

        ptr = p;
        if (p) {
            control = new ControlBlock();
        } else {
            control = nullptr;
        }
    }

    /**
     * @brief 判断是否为空
     * @return true 表示非空，false 表示空
     */
    explicit operator bool() const {
        return ptr != nullptr;
    }
};

#endif // SMARTPTRPROJECT_SIMPLESHAREDPTR_H