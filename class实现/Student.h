#pragma once
#include <string>
#include <iostream>
#include <thread>
class StudentFriend;
class Student{
public:
    Student();
    Student(const std::string& name,const int& age,const std::string& num);
    Student(const Student& student);
    Student(Student&& student);
    ~Student();


    void ChangeName(const std::string& name);
    void ChangeAge(const int& age);
    void ChangeNum(const std::string&num);

    int GetAge()const;
    std::string GetName()const;
    std::string GetNum()const;

    int GetMoney()const;
    void PayMoney(const int& money);

    Student& operator=(const Student& student);
    Student& operator=(Student&& student);
    Student& operator +(const Student& student);
    friend std::ostream& operator<<(std::ostream& os,const Student& student);

    static void TestStatic();

    friend void ChangeAge(Student& student,int age);

    friend class StudentFriend;
    
private:
    int _age;
    std::string _name;
    std::string _num;
    static int _money;
    std::thread _thread;
    int *_data;
};


class StudentFriend{
    public:
    StudentFriend()=default;
    void ChangeAge(Student&student,int age){
        student._age = age;
    }

};