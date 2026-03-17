//
// Created by bai on 2026/3/16.
//

#include "../include/GameWorld.h"
#include "../include/Player.h"
#include "../include/Team.h"
#include <iostream>
#include <algorithm>
void GameWorld::createPlayers(const std::string& name, int level) {
    players.push_back(std::make_shared<Player>(name,level));
}
void GameWorld::createTeams(const std::string& teamName) {
    teams.push_back(std::make_shared<Team>(teamName));
}
std::shared_ptr<Player> GameWorld::findPlayer(const std::string& name) {
    for (auto player : players) {
        if (player&&player->getName() == name) {
            return player;
        }
    }
    return nullptr;
}
std::shared_ptr<Team> GameWorld::findTeam(const std::string& name) {
    for (auto &team:teams) {
        if (team&&team->getTeamName()==name) {
            return team;
        }
    }
    return nullptr;
}

void GameWorld::removeTeam(const std::string& name) {
    for (auto it =teams.begin(); it != teams.end(); ++it) {
        if (*it&&(*it)->getTeamName()==name) {
            std::cout<<"从世界移除队伍"<<std::endl;
            teams.erase(it);
            return;
        }
    }
    std::cout<<"没找到队伍"<<std::endl;
}

void GameWorld::showAllPlayers() const {
    std::cout<<"All players"<<std::endl;
    for (const auto& player : players) {
        if (player) {
            player->show();
        }
    }
}

void GameWorld::showAllTeams() const {
    std::cout<<"All teams"<<std::endl;
    for (const auto& team : teams) {
        if (team) {
            std::cout<<"队伍"<<team->getTeamName()<<std::endl;
        }
    }
}