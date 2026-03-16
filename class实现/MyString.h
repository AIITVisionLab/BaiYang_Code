#pragma once
#include <iostream>
#include<cstring>
class MyString{
public:
    MyString():_data(nullptr){}
    MyString(const char*str);
    MyString(const MyString&other);
    MyString& operator=(const MyString&other);

    MyString(MyString&& other);
    MyString& operator=(MyString&& other);
    ~MyString();

    MyString operator+(const MyString& other);
    bool operator==(const MyString&other); 
    friend std::ostream& operator<<(std::ostream&out,const MyString&str);
    
private:
    char* _data;
};