#include <iostream>
#include <iostream>
#include <memory>
#include "../RPGgameManage/include/Player.h"
#include "../RPGgameManage/include/Weapon.h"
#include "../RPGgameManage/include/Team.h"
#include "../RPGgameManage/include/GameWorld.h"

int main() {
    GameWorld world;

    world.createPlayers("Tom", 10);
    world.createPlayers("Jerry", 12);
    world.createPlayers("Alice", 15);

    world.createTeams("Alpha");
    world.createTeams("Beta");

    auto tom = world.findPlayer("Tom");
    auto jerry = world.findPlayer("Jerry");
    auto alice = world.findPlayer("Alice");

    auto alpha = world.findTeam("Alpha");
    auto beta = world.findTeam("Beta");

    std::cout << "\n基础信息" << std::endl;
    world.showAllPlayers();
    world.showAllTeams();

    std::cout << "\nunique_ptr：装备武器" << std::endl;
    tom->equipWeapon(std::make_unique<Weapon>("长剑", 50));
    jerry->equipWeapon(std::make_unique<Weapon>("法杖", 80));
    alice->equipWeapon(std::make_unique<Weapon>("匕首", 35));

    tom->showWeapon();
    jerry->showWeapon();
    alice->showWeapon();

    std::cout << "\nshared_ptr：加入队伍" << std::endl;
    alpha->addMember(tom);
    alpha->addMember(jerry);
    beta->addMember(alice);

    alpha->showMembers();
    beta->showMembers();

    std::cout << "\nweak_ptr：玩家查看自己队伍" << std::endl;
    tom->showTeam();
    jerry->showTeam();
    alice->showTeam();

    std::cout << "\nshared_ptr 引用计数观察" << std::endl;
    std::cout << "Tom use_count = " << tom.use_count() << std::endl;
    std::cout << "Jerry use_count = " << jerry.use_count() << std::endl;
    std::cout << "Alice use_count = " << alice.use_count() << std::endl;

    std::cout << "\n队伍移除成员" << std::endl;
    alpha->removeMember("Jerry");
    alpha->showMembers();
    jerry->showTeam();

    std::cout << "\n删除队伍，观察 weak_ptr" << std::endl;
    world.removeTeam("Beta");
    alice->showTeam();

    std::cout << "\n程序结束，观察析构顺序" << std::endl;
    return 0;
}