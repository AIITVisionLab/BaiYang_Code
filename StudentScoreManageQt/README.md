# Student Score Management System (Qt)

这是一个基于 Qt5 开发的学生成绩管理系统。

比较特别的是，它采用了 **C/C++ 混合编程** 的方式：
*   **UI 层**：使用 Qt (C++) 构建图形界面。
*   **核心层**：使用纯 C 语言实现数据结构（链表）和文件操作。

这种设计模拟了在遗留 C 代码库上构建现代 GUI 的场景，实现了界面与业务逻辑的完全解耦。

##  快速开始

### 环境要求
*   Qt 5.x
*   CMake 3.10+
*   C++17 编译器 (GCC/Clang/MSVC)

### 编译运行

```bash
mkdir build && cd build
cmake ..
make
./StudentScoreManageQt
```

##  功能特性

*   **基础管理**：添加、修改、删除学生信息（支持批量删除）。
*   **成绩管理**：录入 9 门课程成绩，系统自动计算平均分。
*   **查询功能**：支持按学号快速查找并高亮显示。
*   **数据导出**：支持将数据导出为 CSV 表格，方便 Excel 处理。
*   **自动保存**：数据实时保存到本地二进制文件 (`students.dat`)，防止丢失。

##  项目结构

```text
.
├── core/                   # [C 语言] 核心业务层
│   ├── include/            # 头文件 (student.h, list.h, student_c_api.h)
│   └── src/                # 源文件
│       ├── list.c          # 通用链表数据结构实现
│       ├── file_io.c       # 二进制/文本文件读写
│       └── student_c_api.c # 对外暴露的 C API 接口 (CamelCase 命名)
├── src/                    # [C++ / Qt] 应用层
│   ├── manager/            # 业务逻辑适配层
│   │   └── studentmanager.cpp # 单例模式管理器，封装 C API 调用
│   ├── ui/                 # 图形界面层
│   │   ├── mainwindow.cpp     # 主窗口逻辑
│   │   └── studentdialog.cpp  # 学生信息编辑弹窗
│   └── main.cpp            # 程序入口
├── CMakeLists.txt          # CMake 构建脚本
└── README.md               # 项目说明文档
```

##  技术细节

*   **混合编程**：核心数据结构使用 C 语言编写，通过 `extern "C"` 暴露接口。
*   **非阻塞设计**：移除了底层 C 代码中所有 `scanf` 等交互式函数，确保 GUI 线程不卡死。
*   **资源管理**：C++ 层负责管理 C 内存的分配与释放，防止内存泄漏。

---

**贡献者:**
Jokerbai, Wang Luyao, Lin Ziyi, Wang Xin

