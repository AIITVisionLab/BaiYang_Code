#include "MyString.h"

MyString::~MyString() {
    delete[] _data;
}

MyString::MyString(const char* str) {
    if (str == nullptr) {
        _data = nullptr;
        return;
    }
    _data = new char[strlen(str) + 1];
    strcpy(_data, str);
}

MyString::MyString(const MyString& other) {
    if (other._data == nullptr) {
        _data = nullptr;
        return;
    }
    _data = new char[strlen(other._data) + 1];
    strcpy(_data, other._data);
}

MyString& MyString::operator=(const MyString& other) {
    if (this == &other) {
        return *this;
    }
    delete[] _data;
    if (other._data == nullptr) {
        _data = nullptr;
        return *this;
    }
    _data = new char[strlen(other._data) + 1];
    strcpy(_data, other._data);
    return *this;
}

MyString::MyString(MyString&& other) {
    _data = other._data;
    other._data = nullptr;
}

MyString& MyString::operator=(MyString&& other) {
    if (this == &other) {
        return *this;
    }
    delete[] _data;
    _data = other._data;
    other._data = nullptr;
    return *this;
}

MyString MyString::operator+(const MyString& other) {
    size_t len1 = _data ? strlen(_data) : 0;
    size_t len2 = other._data ? strlen(other._data) : 0;
    char* temp = new char[len1 + len2 + 1];
    if (_data) strcpy(temp, _data);
    else temp[0] = '\0';
    if (other._data) strcat(temp, other._data);
    
    MyString res(temp);
    delete[] temp;
    return res;
}
bool MyString::operator==(const MyString&other){
    if(other._data==nullptr){
        return false;
    }
    if(strcmp(_data,other._data)==0){
        return true;
    }
    return false;
}
std::ostream& operator<<(std::ostream&out,const MyString&str){
    if(str._data==nullptr){
        out<<"nullptr";
        return out;
    }
    out<<str._data;
    return out;
}
