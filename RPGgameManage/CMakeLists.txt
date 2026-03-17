cmake_minimum_required(VERSION 4.1)
project(RPGgameManage)

set(CMAKE_CXX_STANDARD 20)

add_executable(RPGgameManage main.cpp
        src/Weapon.cpp
        include/Weapon.h
        src/Player.cpp
        include/Player.h
        src/Team.cpp
        include/Team.h
        src/GameWorld.cpp
        include/GameWorld.h
)
