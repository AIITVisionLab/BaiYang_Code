#include <iostream>
#include "Student.h"
#include "MyString.h"
#include <windows.h>

int main()
{
    SetConsoleOutputCP(65001);
    std::cout << std::endl; // Flush/Ensure console is ready
    
    Student student1;
    student1.ChangeName("张三");
    student1.ChangeAge(18);
    student1.ChangeNum("20230001");
    student1.PayMoney(1000);
    std::cout<<"姓名："<<student1.GetName()<<std::endl;
    std::cout<<"年龄："<<student1.GetAge()<<std::endl;
    std::cout<<"学号："<<student1.GetNum()<<std::endl;
    std::cout<<"当前余额为："<<student1.GetMoney()<<std::endl;
    std::cout<<std::endl;

    Student student2;
    student2.ChangeName("李四");
    student2.ChangeAge(19);
    student2.ChangeNum("20230002");
    std::cout<<"姓名："<<student2.GetName()<<std::endl;
    std::cout<<"年龄："<<student2.GetAge()<<std::endl;
    std::cout<<"学号："<<student2.GetNum()<<std::endl;
    student2.PayMoney(10);
    std::cout<<"当前余额为："<<student2.GetMoney()<<std::endl;
    std::cout<<std::endl;

    Student student3("alex",20,"20230002");
    std::cout<<"姓名："<<student3.GetName()<<std::endl;
    std::cout<<"年龄："<<student3.GetAge()<<std::endl;
    std::cout<<"学号："<<student3.GetNum()<<std::endl;
    student3.PayMoney(100);
    std::cout<<"当前余额为："<<student3.GetMoney()<<std::endl;
    std::cout<<std::endl;

    Student student4(student1);
    std::cout<<"姓名："<<student4.GetName()<<std::endl;
    std::cout<<"年龄："<<student4.GetAge()<<std::endl;
    std::cout<<"学号："<<student4.GetNum()<<std::endl;
    std::cout<<"当前余额为："<<student4.GetMoney()<<std::endl;
    std::cout<<std::endl;
    
    Student student5(std::move(student1));
    std::cout<<"姓名："<<student5.GetName()<<std::endl;
    std::cout<<"年龄："<<student5.GetAge()<<std::endl;
    std::cout<<"学号："<<student5.GetNum()<<std::endl;
    std::cout<<"当前余额为："<<student5.GetMoney()<<std::endl;
    std::cout<<std::endl;


    Student student6;
    student6.GetMoney();
    std::cout<<"student6 val:"<<&student6<<std::endl;
    std::cout<<"student6 name:"<<student6.GetName()<<std::endl;
    std::cout<<"student6 age:"<<student6.GetAge()<<std::endl;
    std::cout<<"student6 num:"<<student6.GetNum()<<std::endl;
    std::cout<<"student6 money:"<<student6.GetMoney()<<std::endl;
    std::cout<<std::endl;
    student6=student5;
    std::cout<<"student6 name:"<<student6.GetName()<<std::endl;
    std::cout<<"student6 age:"<<student6.GetAge()<<std::endl;
    std::cout<<"student6 num:"<<student6.GetNum()<<std::endl;
    std::cout<<"student6 money:"<<student6.GetMoney()<<std::endl;
    std::cout<<std::endl;
    

    Student student7=std::move(student6);
    std::cout<<"name:"<<student7.GetName()<<std::endl;
    std::cout<<"age:"<<student7.GetAge()<<std::endl;
    std::cout<<"num:"<<student7.GetNum()<<std::endl;
    std::cout<<"money:"<<student7.GetMoney()<<std::endl;
    std::cout<<std::endl;

    ChangeAge(student7,20);
    std::cout<<"name:"<<student7.GetName()<<std::endl;
    std::cout<<"age:"<<student7.GetAge()<<std::endl;
    std::cout<<"num:"<<student7.GetNum()<<std::endl;
    std::cout<<"money:"<<student7.GetMoney()<<std::endl;
    std::cout<<std::endl;
    

    Student student8=student7;
    student8=student8+student7;
    std::cout<<"age:"<<student8.GetAge()<<std::endl;
    std::cout<<std::endl;
    std::cout<<student8<<std::endl;

    //my test
    MyString str1("hello");
    MyString str2("world");
    MyString str3=str1+str2;
    std::cout<<str3<<std::endl;
    if(str3=="helloworld"){
        std::cout<<"str3 is helloworld"<<std::endl;
    }
    return 0;
}
