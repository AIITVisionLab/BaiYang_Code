#pragma once
#include <memory>
#include <string>
#include <vector>

class Player;

class Team : public std::enable_shared_from_this<Team> {
private:
    std::string teamName;
    std::vector<std::shared_ptr<Player>> members;

public:
    Team(const std::string& name);
    ~Team();

    std::string getTeamName() const;

    void addMember(const std::shared_ptr<Player>& p);
    void removeMember(const std::string& playerName);
    void showMembers() const;
};