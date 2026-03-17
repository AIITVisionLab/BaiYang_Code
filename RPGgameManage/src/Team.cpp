#include "../include/Team.h"
#include "../include/Player.h"
#include <iostream>

Team::Team(const std::string& name) : teamName(name) {
    std::cout << "队伍创建: " << teamName << std::endl;
}

Team::~Team() {
    std::cout << "队伍销毁: " << teamName << std::endl;
}

std::string Team::getTeamName() const {
    return teamName;
}

void Team::addMember(const std::shared_ptr<Player>& p) {
    if (!p) return;

    for (const auto& member : members) {
        if (member && member->getName() == p->getName()) {
            std::cout << p->getName() << " 已经在队伍 " << teamName << " 中" << std::endl;
            return;
        }
    }

    members.push_back(p);
    p->joinedTeam(shared_from_this());
    std::cout << p->getName() << " 加入队伍 " << teamName << std::endl;
}

void Team::removeMember(const std::string& playerName) {
    for (auto it = members.begin(); it != members.end(); ++it) {
        if (*it && (*it)->getName() == playerName) {
            (*it)->leavedTeam();
            std::cout << playerName << " 离开队伍 " << teamName << std::endl;
            members.erase(it);
            return;
        }
    }

    std::cout << "队伍 " << teamName << " 中没有成员 " << playerName << std::endl;
}

void Team::showMembers() const {
    std::cout << "队伍 [" << teamName << "] 成员列表:" << std::endl;

    if (members.empty()) {
        std::cout << "  (空)" << std::endl;
        return;
    }

    for (const auto& member : members) {
        if (member) {
            std::cout << "  - " << member->getName()
                      << " (level " << member->getLevel() << ")"
                      << std::endl;
        }
    }
}