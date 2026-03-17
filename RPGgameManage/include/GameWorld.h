//
// Created by bai on 2026/3/16.
//

#ifndef RPGGAMEMANAGE_GAMEWORLD_H
#define RPGGAMEMANAGE_GAMEWORLD_H
#include <string>
#include <memory>
#include<vector>

class Player;
class Team;
class GameWorld {
private:
    std::vector<std::shared_ptr<Player>> players;
    std::vector<std::shared_ptr<Team>> teams;
public:
    void createPlayers(const std::string& name,int level);
    void createTeams(const std::string& teamName);

    std::shared_ptr<Player> findPlayer(const std::string& name);
    std::shared_ptr<Team> findTeam(const std::string& name);

    void removeTeam(const std::string& name);
    void showAllPlayers()const;
    void showAllTeams()const;

};


#endif //RPGGAMEMANAGE_GAMEWORLD_H