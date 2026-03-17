//
// Created by bai on 2026/3/16.
//

#ifndef RPGGAMEMANAGE_PLAYER_H
#define RPGGAMEMANAGE_PLAYER_H
#include <string>
#include<memory>

//告诉编辑器后面会有这两个class
class Team;
class Weapon;
class Player {
private:
    std::string name;
    int level;
    std::unique_ptr<Weapon>weapon;
    std::weak_ptr<Team> team;
public:
    Player(const std::string& name, int level);
    ~Player();

    [[nodiscard]]std::string getName() const;
    [[nodiscard]]int getLevel() const;
    void equipWeapon(std::unique_ptr<Weapon> w);
    void showWeapon()const;

    void joinedTeam(const std::shared_ptr<Team> &t);
    void leavedTeam();
    void showTeam()const;

    void show()const;

};


#endif //RPGGAMEMANAGE_PLAYER_H