#include <iostream>
#include "SimpleSharedPtr.h"
#include "Student.h"

int main() {
    std::cout << "===== 1. 基本构造测试 =====" << std::endl;
    SimpleSharedPtr<Student> sp1(new Student("Tom", 18));
    std::cout << "sp1 use_count = " << sp1.use_count() << std::endl;
    sp1->show();

    std::cout << "\n===== 2. 拷贝构造测试 =====" << std::endl;
    SimpleSharedPtr<Student> sp2(sp1);
    std::cout << "sp1 use_count = " << sp1.use_count() << std::endl;
    std::cout << "sp2 use_count = " << sp2.use_count() << std::endl;

    std::cout << "\n===== 3. 拷贝赋值测试 =====" << std::endl;
    SimpleSharedPtr<Student> sp3;
    sp3 = sp1;
    std::cout << "sp1 use_count = " << sp1.use_count() << std::endl;
    std::cout << "sp3 use_count = " << sp3.use_count() << std::endl;

    std::cout << "\n===== 4. 移动构造测试 =====" << std::endl;
    SimpleSharedPtr<Student> sp4(std::move(sp3));
    std::cout << "sp4 use_count = " << sp4.use_count() << std::endl;
    std::cout << "sp3 use_count = " << sp3.use_count() << std::endl;

    std::cout << "\n===== 5. reset 测试 =====" << std::endl;
    sp4.reset(new Student("Jerry", 20));
    std::cout << "sp4 use_count = " << sp4.use_count() << std::endl;
    sp4->show();

    std::cout << "\n===== 6. 作用域结束自动析构 =====" << std::endl;
    return 0;
}