#include "Student.h"
#include <iostream>
int Student::_money = 10000;
Student::Student() : _name("张三"), _age(18), _num("188888") {
  _data = new int();
}
Student::Student(const std::string &name, const int &age,
                 const std::string &num)
    : _name(name), _age(age), _num(num) {
  _data = new int();
}
Student::Student(const Student &student)
    : _name(student._name), _age(student._age), _num(student._num) {
  _data = new int();
  if (student._data) {
    *_data = *student._data;
  } else {
    *_data = 0;
  }
}
Student::Student(Student &&student)
    : _name(std::move(student._name)), _age(student._age),
      _num(std::move(student._num)), _thread(std::move(student._thread)),
      _data(student._data) {
  student._data = nullptr;
}

// change methods
void Student::ChangeAge(const int &age) { _age = age; }
void Student::ChangeNum(const std::string &num) { _num = num; }
void Student::ChangeName(const std::string &name) { _name = name; }

// get methods
int Student::GetAge() const { return _age; }
std::string Student::GetName() const { return _name; }
std::string Student::GetNum() const { return _num; }
int Student::GetMoney() const { return _money; }
void Student::PayMoney(const int &cost) { _money -= cost; }

Student::~Student() {
    std::cout << "析构函数被调用" << std::endl;
  if (_thread.joinable()) {
    _thread.join();
  }

  std::cout << "线程已被合并" << std::endl;
  delete _data;
}

Student &Student::operator=(const Student &student) {
  if (this == &student) {
    return *this;
  }
  this->_name = student._name;
  this->_age = student._age;
  this->_num = student._num;
  if (student._data) {
    if (!_data)
      _data = new int();
    *_data = *student._data;
  }
  return *this;
}

Student &Student::operator=(Student &&student) {
  if (this == &student) {
    return *this;
  }
  _name = std::move(student._name);
  _age = std::move(student._age);
  _num = std::move(student._num);

  if (_thread.joinable()) {
    _thread.join();
  }
  _thread = std::move(student._thread);

  delete _data;
  _data = student._data;
  student._data = nullptr;

  return *this;
}

Student& Student::operator+(const Student& student){
    this->_age += student._age;
    return *this;
}

std::ostream& operator<<(std::ostream& os,const Student& student){
    os<<"name:"<<student._name<<std::endl;
    os<<"age:"<<student._age<<std::endl;
    os<<"num:"<<student._num<<std::endl;
    os<<"money:"<<student._money<<std::endl;
    return os;
}

void Student::TestStatic() {
  std::cout << "static val:" << _money << std::endl;
}

void ChangeAge(Student& student,int age){
    student._age = age;
}
