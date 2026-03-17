//
// Created by bai on 2026/3/16.
//

#include "../include/Player.h"
#include "../include/Team.h"
#include "../include/Weapon.h"
#include<iostream>
#include<utility>

Player::Player(const std::string& name, int level)
    :name(name), level(level){
    std::cout<<"Player created"<<std::endl;
}

Player::~Player() {
    std::cout<<"Player destroyed"<<std::endl;
}

std::string Player::getName() const {
    return name;
}

int Player::getLevel() const {
    return level;
}

void Player::equipWeapon(std::unique_ptr<Weapon> w) {
    weapon=std::move(w);
}

void Player::showWeapon() const {
    if (weapon) {
        std::cout<<name<<"的";
        weapon->show();
    }else{
    std::cout<<name<<"当前没有武器"<<std::endl;
    }
}

void Player::joinedTeam(const std::shared_ptr<Team>& t) {
    team=t;
}
void Player::leavedTeam() {
    team.reset();
}

void Player::showTeam() const {
    std::cout<<"玩家"<<name<<"等级"<<level<<std::endl;
}

void Player::show() const {
    std::cout<<"玩家："<<name<<"等级"<<level<<std::endl;
}

