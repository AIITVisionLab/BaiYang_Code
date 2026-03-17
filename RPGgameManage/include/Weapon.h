//
// Created by bai on 2026/3/16.
//

#ifndef RPGGAMEMANAGE_WEAPON_H
#define RPGGAMEMANAGE_WEAPON_H
#include<string>


class Weapon {
private:
    std::string name;
    int damage;
public:
    Weapon(const std::string& name, int damage);
    std::string getName()const;//名字
    int getDamage()const;      //损坏程度

    void show() const;  //展示

};


#endif //RPGGAMEMANAGE_WEAPON_H