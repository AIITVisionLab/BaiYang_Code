//
// Created by bai on 2026/3/16.
//

#ifndef SMARTPOINT_STUDENT_H
#define SMARTPOINT_STUDENT_H
#include <iostream>
#include <string>
    class Student {
    public:
        std::string name;
        int age;

        Student(const std::string& n = "", int a = 0) : name(n), age(a) {
            std::cout << "Student 构造: " << name << std::endl;
        }

        ~Student() {
            std::cout << "Student 析构: " << name << std::endl;
        }

        void show() const {
            std::cout << "name = " << name << ", age = " << age << std::endl;
        }
    };


#endif //SMARTPOINT_STUDENT_H