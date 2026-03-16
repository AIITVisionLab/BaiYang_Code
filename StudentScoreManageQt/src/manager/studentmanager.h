#ifndef STUDENTMANAGER_H
#define STUDENTMANAGER_H

#include "student_c_api.h"
#include <vector>
#include <string>

class StudentManager {
public:
    static StudentManager& instance();

    bool addStudent(const Student& stu);
    bool deleteStudent(unsigned long long number);
    bool updateStudent(const Student& stu);
    std::vector<Student> getAllStudents();
    int getStudentCount();
    std::string getAverageStats();

private:
    StudentManager();
    ~StudentManager();
    StudentManager(const StudentManager&) = delete;
    StudentManager& operator=(const StudentManager&) = delete;
};

#endif // STUDENTMANAGER_H
