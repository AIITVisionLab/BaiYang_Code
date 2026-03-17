#include "../include/Weapon.h"
#include <iostream>

Weapon::Weapon(const std::string& name, int damage)
    : name(name), damage(damage) {}

std::string Weapon::getName() const {
    return name;
}

int Weapon::getDamage() const {
    return damage;
}

void Weapon::show() const {
    std::cout << "武器: " << name << "，伤害: " << damage << std::endl;
}
