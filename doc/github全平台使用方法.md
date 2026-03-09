# GitHub + VS Code 联合教学文档（最基础功能版）

最后更新：2026-03-09

这份文档不是“GitHub 全平台大全”，而是专门给初学者用的 `GitHub + VS Code 联合教学文档`。  
目标很明确：只教最基础、最常用、最容易马上上手的功能。

你学完这份文档，应该至少会做下面这些事：

1. 注册并登录 GitHub
2. 在 GitHub 里创建仓库
3. 在电脑上安装 VS Code 和 Git
4. 在 VS Code 里登录 GitHub
5. 在 VS Code 里打开项目
6. 在 VS Code 里初始化 Git 仓库
7. 在 VS Code 里提交代码
8. 在 VS Code 里把项目发布到 GitHub
9. 在 VS Code 里克隆别人的 GitHub 仓库
10. 在 VS Code 里拉取、推送、切换分支
11. 在 VS Code 里创建 Pull Request

这份文档刻意不展开的内容：

1. GitHub Desktop
2. GitHub Mobile
3. 很复杂的 Git 命令
4. 高级权限模型
5. 高级冲突处理
6. GitHub Actions
7. Git rebase、cherry-pick、submodule 等高级主题

如果你是零基础，请按顺序做，不要跳。

---

## 目录

1. [先知道这几个词是什么意思](#先知道这几个词是什么意思)
2. [开始之前要准备什么](#开始之前要准备什么)
3. [第 1 部分：注册 GitHub 账号并完成最基本安全设置](#第-1-部分注册-github-账号并完成最基本安全设置)
4. [第 2 部分：安装 VS Code 和 Git](#第-2-部分安装-vs-code-和-git)
5. [第 3 部分：第一次配置 Git](#第-3-部分第一次配置-git)
6. [第 4 部分：认识 VS Code 里跟 GitHub 相关的几个位置](#第-4-部分认识-vs-code-里跟-github-相关的几个位置)
7. [第 5 部分：在 VS Code 里登录 GitHub](#第-5-部分在-vs-code-里登录-github)
8. [第 6 部分：最基础场景 A，本地项目发布到 GitHub](#第-6-部分最基础场景-a本地项目发布到-github)
9. [第 7 部分：最基础场景 B，从 GitHub 克隆仓库到 VS Code](#第-7-部分最基础场景-b从-github-克隆仓库到-vs-code)
10. [第 8 部分：日常最基础操作流程](#第-8-部分日常最基础操作流程)
11. [第 9 部分：分支的最基础用法](#第-9-部分分支的最基础用法)
12. [第 10 部分：在 VS Code 里创建 Pull Request](#第-10-部分在-vs-code-里创建-pull-request)
13. [第 11 部分：最常见问题](#第-11-部分最常见问题)
14. [第 12 部分：给老师和初学者的最短教学路线](#第-12-部分给老师和初学者的最短教学路线)
15. [官方参考链接](#官方参考链接)

---

## 先知道这几个词是什么意思

在开始操作之前，先把最常见的词记住。

### 1. GitHub

GitHub 是一个放代码和协作项目的平台。

### 2. VS Code

VS Code 是一个代码编辑器。  
你会在里面写代码、改文件、看差异、提交代码。

### 3. Git

Git 是版本管理工具。  
VS Code 里很多“提交、分支、同步”功能，本质上都是在调用 Git。

### 4. Repository

简称 `repo`，中文一般叫“仓库”。  
你可以把它先理解成“项目的版本库”。

### 5. Commit

一次提交。  
你可以理解成“给当前改动拍一张快照，并写一句说明”。

### 6. Push

把你本地电脑里的提交上传到 GitHub。

### 7. Pull

把 GitHub 上新的内容拉回本地电脑。

### 8. Clone

把 GitHub 上的仓库复制到你的电脑里。

### 9. Branch

分支。  
你可以在不影响主线代码的前提下单独开发。

### 10. Pull Request

简称 `PR`。  
意思是“我改好了，请你看一下，确认后合并”。

如果上面 10 个词你大致有概念，后面会轻松很多。

---

## 开始之前要准备什么

你至少需要下面这些东西：

1. 一个 GitHub 账号
2. 一台电脑
3. VS Code
4. Git
5. 网络环境正常

建议但不是必须的：

1. 一个能正常接收邮件的常用邮箱
2. 手机验证器 App，用来开 GitHub 的 2FA
3. VS Code 扩展：`GitHub Pull Requests and Issues`

这份文档默认你使用的是：

1. Windows
2. macOS
3. Linux

如果你是纯新手，我建议你用下面这套最简单组合：

1. 用 GitHub 网页注册账号
2. 用 VS Code 做日常操作
3. 用 VS Code 自带的 Source Control 功能做提交和同步
4. 用 VS Code 扩展做 Pull Request

也就是说，这份文档的核心目标不是“学命令行”，而是“在 VS Code 里把 GitHub 基本用起来”。

---

## 第 1 部分：注册 GitHub 账号并完成最基本安全设置

## 第 1 步：注册 GitHub 账号

1. 打开浏览器。
2. 进入 `https://github.com/`。
3. 点击右上角 `Sign up`。
4. 输入邮箱、密码、用户名。
5. 按页面提示完成注册。
6. 回到邮箱，找到 GitHub 的验证邮件。
7. 点击邮件里的验证链接。

注意：

1. 邮箱不验证，后面很多功能会受影响。
2. GitHub 官方文档说明，创建个人账号时也支持用 Google 或 Apple 登录。
3. 如果你是公司统一下发的企业托管账号，这个注册流程可能不适用。

## 第 2 步：第一次登录 GitHub

1. 打开 `https://github.com/`。
2. 点击 `Sign in`。
3. 输入用户名或邮箱。
4. 输入密码。
5. 按提示完成验证。

## 第 3 步：建议马上开启 2FA

虽然这份文档聚焦基础功能，但安全这一步不要跳。

推荐顺序：

1. `TOTP 验证器应用`
2. `安全密钥`
3. `短信` 只作为备选

最推荐的新手做法是：

1. 手机安装一个验证器 App
2. 在 GitHub 设置里开启 2FA
3. 扫描二维码
4. 保存恢复码

操作步骤：

1. 登录 GitHub 后，点击右上角头像。
2. 点击 `Settings`。
3. 进入 `Password and authentication`。
4. 找到 `Two-factor authentication`。
5. 点击开启。
6. 选择 TOTP 验证器方式。
7. 用手机验证器扫描二维码。
8. 输入 6 位验证码。
9. 下载恢复码并保存。

注意：

1. 官方更推荐 TOTP，而不是只用短信。
2. 恢复码一定要保存好。
3. 如果以后 VS Code 登录 GitHub 时触发二次验证，这一步就能派上用场。

---

## 第 2 部分：安装 VS Code 和 Git

这一部分做完后，你才能在 VS Code 里顺畅地使用 GitHub。

## 第 1 步：安装 VS Code

1. 打开 `https://code.visualstudio.com/`
2. 下载适合你系统的版本
3. 双击安装
4. 安装完成后启动 VS Code

第一次打开 VS Code 后，你至少先认识这几个位置：

1. 左侧活动栏
2. 文件资源管理器 `Explorer`
3. 源代码管理 `Source Control`
4. 扩展 `Extensions`
5. 左下角状态栏
6. 顶部命令面板入口

## 第 2 步：安装 Git

如果电脑没有 Git，VS Code 里的很多 Git 功能都不能正常工作。

### Windows

1. 打开 `https://git-scm.com/`
2. 下载 Windows 安装包
3. 双击安装
4. 大多数选项保持默认即可

### macOS

常见方法：

1. 执行 `xcode-select --install`
2. 或使用 Homebrew 安装 Git

### Linux

常见方法：

1. Ubuntu / Debian：`sudo apt install git`
2. Fedora：`sudo dnf install git`
3. Arch：`sudo pacman -S git`

## 第 3 步：确认 Git 是否安装成功

打开终端后执行：

```bash
git --version
```

如果看到版本号，说明 Git 已经安装好了。

你可以在以下位置打开终端：

1. 直接打开系统终端
2. 或在 VS Code 顶部菜单点击 `Terminal -> New Terminal`

---

## 第 3 部分：第一次配置 Git

这一步只需要做一次。

在终端里输入：

```bash
git config --global user.name "你的名字或GitHub用户名"
git config --global user.email "你的GitHub邮箱"
git config --global init.defaultBranch main
```

再检查一下：

```bash
git config --global --list
```

说明：

1. `user.name` 可以写你的名字，也可以写 GitHub 用户名。
2. `user.email` 最好写你 GitHub 已验证过的邮箱。
3. 如果邮箱不一致，GitHub 贡献记录可能不显示。

如果你是新手，最开始确实可以先用 `浏览器授权 + VS Code 内置认证`。  
但如果你准备长期用 GitHub 提交代码，尤其是经常 `push`、`pull`、`clone`，那把 SSH 配好会更稳定，也更省事。

## 第 3.1 节：先学会在 VS Code 里打开终端

虽然这份文档以图形界面为主，但你最好会一点点命令行。  
不是为了炫技，而是因为很多 Git 报错、状态检查、教学演示，用命令行会更直接。

在 VS Code 里打开终端的方法：

1. 顶部菜单点击 `Terminal -> New Terminal`
2. 或使用快捷键打开终端

打开后，终端一般会出现在 VS Code 底部。

你后面可以在这里输入 Git 命令。

## 第 3.2 节：一定要会的 8 个 Git 命令

这 8 个命令是最基础、最常用、最值得先记住的。

### 1. 查看当前状态

```bash
git status
```

这个命令会告诉你：

1. 你现在在哪个分支
2. 哪些文件改了
3. 哪些文件已暂存
4. 哪些文件还没提交

### 2. 查看改动内容

```bash
git diff
```

这个命令适合在提交前快速检查：

1. 你到底改了什么
2. 有没有误改内容

### 3. 暂存所有改动

```bash
git add .
```

意思是：

1. 把当前目录下的改动加入暂存区

### 4. 暂存单个文件

```bash
git add README.md
```

意思是：

1. 只把某一个文件加入暂存区

### 5. 提交

```bash
git commit -m "Update README"
```

意思是：

1. 把已经暂存的内容正式提交一次

### 6. 拉取远程最新代码

```bash
git pull
```

意思是：

1. 把 GitHub 上的新内容拉到本地

### 7. 推送本地提交到 GitHub

```bash
git push
```

意思是：

1. 把本地已经提交的内容传到 GitHub

### 8. 查看和切换分支

查看分支：

```bash
git branch
```

创建并切换到新分支：

```bash
git switch -c feature/demo
```

切回主分支：

```bash
git switch main
```

如果你的 Git 版本比较老，也可能还会看到有人用：

```bash
git checkout -b feature/demo
git checkout main
```

## 第 3.3 节：VS Code 按钮和命令行怎么对应

你不用死记两套操作，但最好知道它们大致对应关系。

1. VS Code 里的 `Initialize Repository`，大致对应 `git init`
2. VS Code 里的暂存按钮 `+`，大致对应 `git add`
3. VS Code 里的 `Commit`，大致对应 `git commit -m "..."`
4. VS Code 里的 `Push`，大致对应 `git push`
5. VS Code 里的 `Pull`，大致对应 `git pull`
6. VS Code 里的分支切换，常对应 `git switch`

## 第 3.4 节：一个最小命令行演示流程

假设你已经在 VS Code 里打开了某个项目，并且当前终端就在项目目录下。

你可以直接练习这组命令：

```bash
git status
git add .
git commit -m "First commit"
git push
```

这组命令的意思依次是：

1. 先看状态
2. 把改动加入暂存区
3. 做一次提交
4. 把提交推到 GitHub

## 第 3.5 节：把 SSH 提交代码彻底配清楚

这一节是给你解决这个问题的：

1. 我不想每次推送都走网页登录提示
2. 我想用 SSH 方式稳定提交代码
3. 我想把仓库地址从 HTTPS 改成 SSH

先记住一句话：

1. 配好 SSH 以后，真正提交代码用的命令还是 `git add`、`git commit`、`git push`
2. 变化的不是提交命令本身，而是“你连接 GitHub 的方式”

### 第 1 步：先理解 SSH 到底是干什么的

你可以把 SSH 理解成：

1. 你的电脑生成一对密钥
2. 私钥留在你电脑上
3. 公钥放到 GitHub 账号里
4. 以后 GitHub 通过这对密钥确认“这台电脑就是你”

所以 SSH 提交代码的核心不是“新命令很多”，而是先把认证关系配好。

### 第 2 步：查看你电脑里有没有现成 SSH key

在 VS Code 终端里输入：

```bash
ls -al ~/.ssh
```

如果你看到了这些类似文件：

1. `id_ed25519`
2. `id_ed25519.pub`
3. 或 `id_rsa`
4. 或 `id_rsa.pub`

说明你电脑上可能已经有 SSH key。

如果你完全没有看到这类文件，也没关系，继续下一步生成新的。

### 第 3 步：生成新的 SSH key

GitHub 官方现在优先推荐 `ed25519`。

在终端输入：

```bash
ssh-keygen -t ed25519 -C "你的GitHub邮箱"
```

例如：

```bash
ssh-keygen -t ed25519 -C "you@example.com"
```

执行后你会依次看到几个提示。

#### 提示 1：保存到哪里

一般会看到类似：

```text
Enter a file in which to save the key
```

最简单做法：

1. 直接按回车
2. 使用默认路径

默认通常是：

1. macOS / Linux：`~/.ssh/id_ed25519`
2. Windows：`C:\Users\你的用户名\.ssh\id_ed25519`

#### 提示 2：是否设置 passphrase

你会看到类似：

```text
Enter passphrase
```

建议：

1. 最好设置一个 passphrase
2. 这样即使别人拿到你的私钥文件，也不能直接使用

如果你暂时只想先跑通流程，也可以先留空，但从安全角度不推荐。

#### 如果你的系统太旧，不支持 ed25519

再改用：

```bash
ssh-keygen -t rsa -b 4096 -C "你的GitHub邮箱"
```

### 第 4 步：把私钥加入 ssh-agent

这一步的作用是：

1. 让系统记住你的私钥
2. 避免你每次都重新处理密钥

#### macOS / Linux

先启动 ssh-agent：

```bash
eval "$(ssh-agent -s)"
```

再添加私钥：

```bash
ssh-add ~/.ssh/id_ed25519
```

#### macOS 如果你想让系统钥匙串记住 passphrase

GitHub 官方文档给出的推荐做法是：

1. 打开或创建 `~/.ssh/config`
2. 写入下面内容

```text
Host github.com
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
```

3. 然后执行：

```bash
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
```

如果你的 key 没设置 passphrase，可以不带 `--apple-use-keychain`。

#### Windows PowerShell

先确保 `ssh-agent` 服务启动：

```powershell
Get-Service -Name ssh-agent | Set-Service -StartupType Manual
Start-Service ssh-agent
```

再把私钥加入 agent：

```powershell
ssh-add $env:USERPROFILE\.ssh\id_ed25519
```

#### Windows 一个很常见的坑

GitHub 官方文档专门提到，Windows 系统自带的 OpenSSH 和 Git for Windows 自带的 `ssh.exe` 可能打架。

如果你明明已经把 key 加进 agent，但 `git push` 还是老让你输或者直接失败，可以执行：

```powershell
git config --global core.sshCommand "C:/Windows/System32/OpenSSH/ssh.exe"
```

这个命令的作用是：

1. 强制 Git 使用 Windows 系统自己的 OpenSSH
2. 避免和 Git for Windows 自带的 SSH 程序冲突

### 第 5 步：把公钥复制出来

注意：

1. 你要复制的是公钥，也就是 `.pub` 结尾那个文件
2. 私钥不要上传，不要发给别人

#### macOS

```bash
pbcopy < ~/.ssh/id_ed25519.pub
```

#### Windows Git Bash

```bash
clip < ~/.ssh/id_ed25519.pub
```

#### Windows PowerShell

```powershell
cat ~/.ssh/id_ed25519.pub | clip
```

#### Linux

```bash
cat ~/.ssh/id_ed25519.pub
```

Linux 通常是把内容打印出来，然后你手动复制。

### 第 6 步：把公钥添加到 GitHub

1. 打开 GitHub 网页
2. 点击右上角头像
3. 点击 `Settings`
4. 进入 `SSH and GPG keys`
5. 点击 `New SSH key`
6. Title 里写一个容易识别的名字，比如：
   - `My Windows Laptop`
   - `MacBook Air`
   - `Office Linux PC`
7. Key type 选择认证用途
8. 把刚才复制的公钥粘贴进去
9. 点击添加

这里一定要确认：

1. 你粘贴的是公钥内容
2. 内容一般以 `ssh-ed25519` 或 `ssh-rsa` 开头
3. 不要多复制空格和换行

### 第 7 步：测试 SSH 是否真的通了

在终端输入：

```bash
ssh -T git@github.com
```

第一次连接时，通常会看到主机指纹确认提示。  
GitHub 官方文档建议你核对指纹是否和官方公布的一致，如果一致，再输入：

```bash
yes
```

如果配置正确，你通常会看到类似下面的成功信息：

```text
Hi USERNAME! You've successfully authenticated, but GitHub does not provide shell access.
```

只要看到这类成功认证提示，就说明：

1. 你的 SSH key 已经能被 GitHub 识别
2. 之后就可以用 SSH 地址操作仓库

### 第 8 步：把仓库远程地址改成 SSH

很多人前面已经用 HTTPS 克隆过仓库。  
这时不用重建仓库，直接改远程地址就行。

先看当前远程地址：

```bash
git remote -v
```

如果你现在看到的是这种：

```text
https://github.com/用户名/仓库名.git
```

那就执行：

```bash
git remote set-url origin git@github.com:用户名/仓库名.git
```

再检查一次：

```bash
git remote -v
```

如果输出已经变成这种格式，就说明改成功了：

```text
git@github.com:用户名/仓库名.git
```

### 第 9 步：以后怎么用 SSH 提交代码

重点来了：

1. 一旦远程地址已经是 SSH
2. 你平时提交代码的命令和以前几乎没有区别

还是这套：

```bash
git status
git add .
git commit -m "Describe your change"
git push
```

拉最新代码也还是：

```bash
git pull
```

也就是说：

1. SSH 不会改变你的日常 Git 操作习惯
2. SSH 只是把“连接 GitHub 的方式”改成了密钥认证

### 第 10 步：新仓库从一开始就用 SSH

如果你准备一开始就用 SSH 地址连仓库，可以这样写。

#### 克隆仓库时直接用 SSH

```bash
git clone git@github.com:用户名/仓库名.git
```

#### 本地项目第一次绑定远程仓库时直接用 SSH

```bash
git remote add origin git@github.com:用户名/仓库名.git
```

然后第一次推送：

```bash
git push -u origin main
```

### 第 11 步：SSH 提交代码最常见的 4 个报错

#### 报错 1：`Permission denied (publickey)`

最常见原因：

1. 公钥还没加到 GitHub
2. 私钥没加进 `ssh-agent`
3. 远程地址虽然是 SSH，但你电脑实际用的不是对应私钥

优先排查顺序：

1. `ssh -T git@github.com`
2. 检查 GitHub `SSH and GPG keys`
3. 检查 `git remote -v`

#### 报错 2：明明加了 key，Windows 还是不认

优先试这个：

```powershell
git config --global core.sshCommand "C:/Windows/System32/OpenSSH/ssh.exe"
```

这是 GitHub 官方文档里专门提到的 Windows 常见冲突处理方式。

#### 报错 3：`Repository not found`

常见原因：

1. 仓库地址写错
2. 仓库名写错
3. 你没有这个私有仓库权限

先检查：

```bash
git remote -v
```

#### 报错 4：每次都反复要求输入 passphrase

这通常不是 GitHub 账号密码问题，而是：

1. 你的私钥设置了 passphrase
2. 但 `ssh-agent` 没记住它

处理思路：

1. 重新执行 `ssh-add`
2. macOS 用钥匙串方式保存
3. Windows 检查 `ssh-agent` 服务是否真的在运行

### 第 12 步：给课堂演示用的一套最短 SSH 流程

如果你要给学生演示“SSH 提交代码”，最短可以只演示这 8 步：

1. `ssh-keygen -t ed25519 -C "邮箱"`
2. `ssh-add` 把私钥加入 agent
3. 复制 `~/.ssh/id_ed25519.pub`
4. GitHub `Settings -> SSH and GPG keys -> New SSH key`
5. `ssh -T git@github.com`
6. `git remote -v`
7. `git remote set-url origin git@github.com:用户名/仓库名.git`
8. `git push`

---

## 第 4 部分：认识 VS Code 里跟 GitHub 相关的几个位置

这一章非常重要，因为很多初学者不是不会操作，而是不知道按钮在哪。

## 1. 左侧活动栏的 Source Control

图标通常像一个分叉的线。  
这是 VS Code 里做 Git 操作的核心区域。

你会在这里做这些事：

1. 看哪些文件改了
2. 打开差异对比
3. 暂存文件
4. 提交
5. 推送
6. 拉取
7. 看分支状态

## 2. 左侧活动栏的 Extensions

你会在这里安装扩展。  
本教程最相关的扩展是：

1. `GitHub Pull Requests and Issues`

## 3. 左下角状态栏

这里很重要，初学者经常忽略。

你会在这里看到：

1. 当前分支名
2. 同步状态
3. 有时候还会显示问题状态或其他 Git 信息

## 4. 顶部命令面板

打开方式：

1. Windows / Linux：`Ctrl + Shift + P`
2. macOS：`Cmd + Shift + P`

你后面会经常用它执行这些命令：

1. `Git: Clone`
2. `Git: Initialize Repository`
3. `Publish to GitHub`
4. `GitHub Pull Requests: Create Pull Request`

## 5. Accounts 菜单

通常在 VS Code 右上角或底部账号入口。  
这里可以看到当前是否已登录 GitHub。

---

## 第 5 部分：在 VS Code 里登录 GitHub

这一部分分成两件事：

1. 安装扩展
2. 登录 GitHub

## 第 1 步：安装 `GitHub Pull Requests and Issues` 扩展

操作步骤：

1. 打开 VS Code。
2. 点击左侧 `Extensions`。
3. 在搜索框输入：`GitHub Pull Requests and Issues`
4. 找到 GitHub 官方扩展。
5. 点击 `Install`。

这个扩展主要负责：

1. 在 VS Code 里看 Pull Request
2. 在 VS Code 里创建 Pull Request
3. 在 VS Code 里看 Issue
4. 在 VS Code 里做基础 review

注意：

1. Git 的基础提交功能是 VS Code 自带的。
2. PR 和 Issues 的增强集成功能主要来自这个扩展。

## 第 2 步：在 VS Code 里登录 GitHub

有两种常见方式。

### 方式 A：通过 GitHub 视图登录

1. 安装扩展后，左侧会出现 GitHub 相关入口。
2. 点击 GitHub 图标。
3. 点击 `Sign In`。
4. VS Code 会弹出浏览器授权流程。
5. 浏览器会打开 GitHub 页面。
6. 登录 GitHub。
7. 点击确认授权。
8. 回到 VS Code。

### 方式 B：在需要的时候自动触发登录

官方文档说明，VS Code 对 GitHub 的认证本身是内置的。  
也就是说，当你执行下面这些需要 GitHub 身份的操作时，VS Code 也会自动提示你登录：

1. 克隆私有仓库
2. 推送到远程仓库
3. 访问某些 GitHub 资源

这时你只需要：

1. 按提示点击登录
2. 在浏览器完成授权
3. 再回到 VS Code

## 第 3 步：确认是否登录成功

你可以用下面几种方式确认：

1. GitHub 相关视图不再显示 `Sign In`
2. 右上角或账户菜单里能看到 GitHub 账号
3. 克隆仓库或推送时不再反复要求登录

如果这里总是跳登录页，先不要继续做复杂操作，先看文末的排错部分。

---

## 第 6 部分：最基础场景 A，本地项目发布到 GitHub

这个场景最适合教学演示。  
你电脑里已经有一个项目文件夹，现在想把它放到 GitHub。

## 可选前置：如果你想先在 GitHub 网页创建一个空仓库

虽然这份文档更推荐你直接在 VS Code 里使用 `Publish to GitHub`，但教学时也经常会先演示网页创建仓库。

最基础步骤如下：

1. 打开 GitHub 网页。
2. 点击右上角 `+`。
3. 选择 `New repository`。
4. 输入仓库名。
5. 选择 `Public` 或 `Private`。
6. 如果你准备稍后从 VS Code 本地项目推送上来，最简单的做法是先不要勾选自动创建 README、`.gitignore`、License。
7. 点击 `Create repository`。

这样做的好处是：

1. 你先在网页上明确看到仓库已经存在
2. 课堂演示时更容易理解“本地项目”和“远程仓库”的关系

这样做的代价是：

1. 后面你还需要把本地项目和这个远程仓库连起来

如果你是纯新手，又只是想最快完成一次成功上传，还是优先走下面的 `Publish to GitHub` 路线更省事。

## 第 1 步：在 VS Code 里打开项目文件夹

1. 打开 VS Code。
2. 点击 `File -> Open Folder`。
3. 选择你的项目文件夹。
4. 点击打开。

打开后，左侧 `Explorer` 应该能看到文件。

## 第 2 步：如果这个项目还不是 Git 仓库，就先初始化

打开 `Source Control` 视图。

如果这是一个普通文件夹，还没有 Git 仓库，通常你会看到：

1. `Initialize Repository`
2. 或类似初始化 Git 仓库的按钮

点击它。

这一步的意思是：

1. 让当前文件夹开始受 Git 管理
2. 后面 VS Code 才知道哪些文件被修改了

## 第 3 步：第一次查看改动

初始化后，VS Code 会开始识别当前文件夹里的文件状态。

你可以：

1. 进入 `Source Control`
2. 查看变更列表
3. 点击某个文件
4. 看左右对比差异

如果你还没做改动，也可以先新建一个最简单文件试一下，比如：

1. `README.md`
2. `main.py`
3. `index.html`

## 第 4 步：暂存文件

在 `Source Control` 里，每个文件旁边通常会有一个 `+`。

你可以：

1. 点某个文件旁边的 `+`，只暂存这个文件
2. 或点统一的暂存按钮，一次性暂存全部改动

“暂存”可以先理解成：

1. 这些改动我准备放进这次提交里

## 第 5 步：写提交说明并提交

1. 在 `Source Control` 顶部找到输入框。
2. 输入提交说明，例如：
   - `Initial commit`
   - `Add README`
   - `Create home page`
3. 点击 `Commit`
4. 如果 VS Code 弹出确认提示，按提示继续。

注意：

1. 提交说明不要写得太随便，比如 `1`、`test`、`aaa`。
2. 最简单也要写清楚“这次改了什么”。

## 第 6 步：发布到 GitHub

这是这份教程最关键的一步。

根据 VS Code 官方文档，你可以直接使用：

1. `Publish to GitHub`

它会帮你：

1. 在 GitHub 上创建仓库
2. 把本地提交推送上去

### 操作方式

方式一：

1. 在 `Source Control` 视图里找 `Publish to GitHub`
2. 点击它

方式二：

1. 打开命令面板
2. 输入 `Publish to GitHub`
3. 选择该命令

### 发布时你通常会遇到的选项

1. 选择发布到哪个 GitHub 账号
2. 选择仓库是否公开
3. 是否使用当前仓库名

### 对公开和私有的最简单理解

1. `Public`：别人能看到
2. `Private`：只有你和被授权的人能看到

如果你只是练习，建议：

1. 作品展示型项目用 `Public`
2. 学习草稿、私人练习用 `Private`

## 第 7 步：发布完成后到 GitHub 网页确认

1. 打开浏览器
2. 登录 GitHub
3. 进入你的个人主页
4. 找到新创建的仓库
5. 点进去
6. 确认文件都在

如果网页里能看到文件，说明你第一次把本地项目发布到 GitHub 成功了。

## 命令行补充：如果你想用终端把本地项目推到 GitHub

如果你前面已经在 GitHub 网页创建了一个空仓库，也可以在 VS Code 终端里这样做。

假设：

1. 你已经打开了项目文件夹
2. 终端当前就在项目根目录
3. 你的仓库地址已经准备好

最基础命令如下：

如果你使用 HTTPS：

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/你的用户名/仓库名.git
git push -u origin main
```

如果你使用 SSH：

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin git@github.com:你的用户名/仓库名.git
git push -u origin main
```

这些命令依次表示：

1. 初始化本地 Git 仓库
2. 暂存全部改动
3. 做第一次提交
4. 把主分支名设为 `main`
5. 添加 GitHub 远程仓库地址
6. 第一次推送到 GitHub

如果你已经执行过 `git init`，就不要重复执行。

---

## 第 7 部分：最基础场景 B，从 GitHub 克隆仓库到 VS Code

这个场景适合：

1. 你要继续写之前放在 GitHub 上的项目
2. 你要下载老师或同事给你的仓库
3. 你要参与团队项目

## 第 1 步：准备一个仓库地址

你可以直接从 GitHub 网页复制地址，也可以在 VS Code 里直接选仓库。

## 第 2 步：用 VS Code 的 `Git: Clone`

1. 打开 VS Code。
2. 打开命令面板。
3. 输入 `Git: Clone`。
4. 点击执行。

根据 VS Code 官方文档，你也可以在没有打开任何文件夹时，直接在 `Source Control` 视图里点：

1. `Clone Repository`

## 第 3 步：选择仓库来源

通常有两种情况：

### 情况 A：你已经复制了仓库 URL

1. 把 URL 粘贴进去
2. 按回车

### 情况 B：你已经登录 GitHub

1. VS Code 会弹出 GitHub 仓库列表
2. 你可以搜索仓库名
3. 选择目标仓库

## 第 4 步：选择本地保存位置

1. 选择你希望把项目放在哪个文件夹
2. 确认保存位置

## 第 5 步：克隆完成后打开仓库

克隆完成后，VS Code 通常会问你是否打开这个仓库。

1. 点击 `Open`
2. 进入项目目录

## 第 6 步：确认这是一个可工作的仓库

你可以看这几个地方：

1. `Explorer` 能看到项目文件
2. 左下角状态栏能看到当前分支名
3. `Source Control` 能正常打开

如果这些都正常，说明仓库已经克隆成功。

## 命令行补充：如果你想直接在 VS Code 终端里克隆

你也可以不用按钮，直接在终端里输入：

```bash
git clone https://github.com/用户名/仓库名.git
```

如果你已经配好了 SSH，更推荐直接这样克隆：

```bash
git clone git@github.com:用户名/仓库名.git
```

克隆完成后再进入目录：

```bash
cd 仓库名
code .
```

`code .` 的意思是：

1. 直接用 VS Code 打开当前项目目录

如果你的系统还没有配置 `code` 命令，也没关系，你可以手动在 VS Code 里打开这个文件夹。

---

## 第 8 部分：日常最基础操作流程

这一章是最重要的“每天都要做的操作”。

你平时最常做的流程，其实就 6 步：

1. 打开项目
2. 改文件
3. 查看改动
4. 提交
5. 推送
6. 拉取

下面拆开讲。

## 第 1 步：打开项目

1. 打开 VS Code
2. `File -> Open Folder`
3. 选择你的项目文件夹

## 第 2 步：修改文件

你可以直接在编辑器里改代码或文档。

例如：

1. 改 `README.md`
2. 改 `index.html`
3. 改 `app.js`

## 第 3 步：看 VS Code 是怎么显示改动的

根据 VS Code 官方文档，`Source Control` 视图是 Git 操作的中心。

你会在这里看到：

1. 哪些文件变了
2. 哪些文件已暂存
3. 哪些文件还没暂存
4. 同步状态

点击某个改动文件后，通常会看到差异对比视图。

## 第 4 步：暂存改动

你可以：

1. 暂存单个文件
2. 暂存所有文件

最基础做法：

1. 确认哪些文件是你真的想提交的
2. 给这些文件点 `+`

## 第 5 步：提交

1. 在 `Source Control` 输入框输入提交说明
2. 点击 `Commit`

推荐你把提交说明写成这种风格：

1. `Update README`
2. `Fix login button style`
3. `Add course notes`

对应的命令行写法是：

```bash
git add .
git commit -m "Update README"
```

## 第 6 步：推送到 GitHub

提交只是把改动保存到了你本地。  
如果你想让 GitHub 网页上也出现这些改动，你还要推送。

常见入口：

1. `Sync Changes`
2. `Push`
3. 状态栏里的同步按钮

VS Code 官方文档说明，当本地分支已经连接到远程分支后，状态栏会显示同步状态，也能看到 incoming / outgoing 的变化。

最简单理解：

1. `Push`：把本地发到 GitHub
2. `Pull`：把 GitHub 拉到本地
3. `Sync Changes`：通常会帮你一起处理拉取和推送

对应命令行：

```bash
git pull
git push
```

## 第 7 步：先拉再改，是个好习惯

如果这是团队项目，建议你每次开始前先：

1. 打开项目
2. 先点击 `Pull`
3. 再开始改文件

这样可以减少后面冲突的概率。

如果你想用命令行，最简单就是：

```bash
git pull
```

## 第 8 步：怎么看自己是不是已经同步成功

你可以检查：

1. 状态栏里的同步提示是否消失
2. `Source Control` 是否没有待推送数量
3. GitHub 网页里是否能看到最新提交

---

## 第 9 部分：分支的最基础用法

如果你只写个人练习项目，分支可以晚一点学。  
但只要涉及合作，分支就是基础中的基础。

## 先理解分支

你可以把分支理解成：

1. 主线代码外的一条独立工作线

常见主分支叫：

1. `main`

你在做新功能时，推荐不要直接改 `main`，而是新建一个分支。

## 第 1 步：在 VS Code 里看当前分支

看左下角状态栏。  
通常会显示当前分支名，比如：

1. `main`
2. `feature/homepage`

## 第 2 步：创建新分支

最常见方式：

1. 点击左下角当前分支名
2. 选择创建新分支
3. 输入分支名

分支名建议写清楚用途，例如：

1. `feature/login-page`
2. `fix/readme-error`
3. `docs/week1-notes`

## 第 3 步：切换分支

还是点击左下角分支名：

1. 选择要切换到的分支
2. 点击确认

切换后，VS Code 会把工作目录切到那个分支。

对应命令行：

创建并切换新分支：

```bash
git switch -c feature/login-page
```

切换到已有分支：

```bash
git switch main
```

## 第 4 步：在新分支上提交

流程和前面一样：

1. 改文件
2. 暂存
3. 提交
4. 推送

如果这是一个新分支，VS Code 可能会提示你：

1. `Publish Branch`

点击它即可把这个分支发到 GitHub。

第一次推送一个新分支时，对应命令行通常是：

```bash
git push -u origin feature/login-page
```

这里的 `-u` 可以先简单理解成：

1. 让本地分支和远程分支建立默认对应关系
2. 以后再推送时通常直接 `git push` 就够了

## 第 5 步：什么时候必须用分支

建议至少在这些情况使用分支：

1. 团队项目
2. 功能改动比较大
3. 你要发 PR
4. 你不想把半成品直接进主分支

---

## 第 10 部分：在 VS Code 里创建 Pull Request

这部分依赖你已经安装：

1. `GitHub Pull Requests and Issues`

并且你已经：

1. 登录 GitHub
2. 有一个已经推送到 GitHub 的分支

## 第 1 步：先确认你已经在分支上提交并推送

如果你还没推送分支，先做下面这件事：

1. 在 VS Code 里点击 `Publish Branch`
2. 或执行 `Push`

## 第 2 步：找到 PR 入口

根据 VS Code 官方文档，你可以这样创建 PR：

1. 打开命令面板
2. 输入 `GitHub Pull Requests: Create Pull Request`
3. 执行命令

或者：

1. 打开 `Pull Requests` 视图
2. 点击 `Create Pull Request`

## 第 3 步：填写 PR 信息

创建时通常要选这些内容：

1. base repository
2. base branch
3. 标题
4. 描述

最常见情况是：

1. base repository：当前项目仓库
2. base branch：`main`
3. compare branch：你刚刚工作的分支

## 第 4 步：标题和描述怎么写

标题最好写清楚动作：

1. `Add home page`
2. `Fix README typo`
3. `Update week 2 notes`

描述最简单至少写：

1. 改了什么
2. 为什么改
3. 有没有需要特别注意的地方

## 第 5 步：创建后你能做什么

根据 VS Code 官方文档，PR 创建后会进入一种 review 模式，你可以继续做这些事情：

1. 查看 PR 详情
2. 查看文件差异
3. 留评论
4. 合并 PR
5. 合并后删除分支

如果你觉得网页更直观，也完全可以：

1. 在 VS Code 里把分支推上去
2. 然后去 GitHub 网页上创建和合并 PR

对初学者来说，这也是完全合理的路径。

---

## 第 11 部分：最常见问题

## 问题 1：VS Code 里没有 Source Control 功能，或者显示 Git 不可用

常见原因：

1. Git 没安装
2. Git 装了，但系统还没识别
3. VS Code 启动得比 Git 安装更早，需要重启

处理办法：

1. 先执行 `git --version`
2. 如果没版本号，先装 Git
3. 装完 Git 后重启 VS Code

## 问题 2：为什么我在 VS Code 里一直登录 GitHub 失败

先检查：

1. 浏览器是否已正常登录 GitHub
2. 是否被浏览器拦截了授权跳转
3. 是否触发了 2FA 但没完成

最简单处理方法：

1. 退出 VS Code
2. 浏览器里先登录 GitHub
3. 再回到 VS Code 重新点 `Sign In`

## 问题 3：为什么我能提交，但不能推送

常见原因：

1. 还没登录 GitHub
2. 当前仓库还没有远程仓库
3. 远程仓库里已经有新内容，而你本地没先拉取
4. 你没有这个仓库的写权限

最基础排查方法：

1. 看是否已经登录 GitHub
2. 看仓库是不是已经 `Publish to GitHub`
3. 先执行一次 `Pull`
4. 如果是别人的仓库，确认你是不是协作者

## 问题 4：为什么 `Publish to GitHub` 按钮找不到

常见原因：

1. 当前文件夹还没初始化 Git 仓库
2. 你还没有任何提交
3. 你打开的不是一个真正的项目文件夹

处理方法：

1. 先点击 `Initialize Repository`
2. 先做一次提交
3. 再去找 `Publish to GitHub`

## 问题 5：为什么提交按钮是灰的

最常见原因：

1. 你还没写提交说明
2. 没有改动
3. 改动还没被识别

先检查：

1. `Source Control` 里有没有文件变更
2. 提交说明输入框里有没有文字

## 问题 6：为什么我一切换分支，文件内容变了

这是正常现象。  
因为不同分支本来就可能对应不同版本的文件。

你需要记住：

1. 切分支前最好先提交或处理好当前改动
2. 不要在有未处理改动时到处乱切

## 问题 7：为什么 Pull 后出现冲突

说明：

1. 你改的地方和远程改的地方撞上了

VS Code 官方文档说明，出现冲突时，`Source Control` 会标出冲突文件，你可以打开文件，查看冲突标记，或使用合并编辑器处理。

对初学者最稳妥的处理方式：

1. 不要慌
2. 打开冲突文件
3. 看清楚哪段是你的，哪段是远程的
4. 保留正确内容
5. 保存后重新提交

---

## 第 12 部分：给老师和初学者的最短教学路线

如果你是老师，或者你只是想按最短路径学会最基础操作，建议按下面顺序讲。

## 路线 A：一节课最基础版

1. 注册 GitHub
2. 开启 2FA
3. 安装 VS Code
4. 安装 Git
5. 配置 `user.name` 和 `user.email`
6. 在 VS Code 安装 `GitHub Pull Requests and Issues`
7. 在 VS Code 登录 GitHub
8. 打开一个本地文件夹
9. `Initialize Repository`
10. 新建一个 `README.md`
11. 提交
12. `Publish to GitHub`

如果学生能做到这里，第一课就算达标。

## 路线 B：第二节课建议讲什么

1. 用 `Git: Clone` 克隆仓库
2. 修改文件
3. 暂存
4. 提交
5. Push
6. Pull
7. 新建分支
8. `Publish Branch`
9. 创建 Pull Request

## 路线 C：初学者自己练习时每天重复的动作

1. 打开项目
2. 先 Pull
3. 改文件
4. 看 diff
5. 暂存
6. Commit
7. Push

如果你把上面这 7 步练熟，GitHub 和 VS Code 的基础协作就已经入门了。

---

## 官方参考链接

下面这些链接是我写这份文档时参考的官方资料，适合继续往下学。

1. VS Code Source Control 总览：`https://code.visualstudio.com/docs/sourcecontrol/overview`
2. VS Code 里使用 GitHub：`https://code.visualstudio.com/docs/sourcecontrol/github`
3. GitHub Pull Requests and Issues 扩展：`https://marketplace.visualstudio.com/items?itemName=GitHub.vscode-pull-request-github`
4. GitHub 注册账号：`https://docs.github.com/en/get-started/quickstart/creating-an-account-on-github`
5. GitHub 配置 2FA：`https://docs.github.com/en/authentication/securing-your-account-with-two-factor-authentication-2fa/configuring-two-factor-authentication`
6. GitHub Set up Git：`https://docs.github.com/en/get-started/git-basics/set-up-git`
7. 生成 SSH key 并加到 ssh-agent：`https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent`
8. 添加 SSH key 到 GitHub：`https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account`
9. 测试 SSH 连接：`https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection`
10. 管理远程仓库地址：`https://docs.github.com/en/get-started/git-basics/managing-remote-repositories`

---

## 命令行速记小抄

如果你只想记最少的一组命令，就先记这几条：

```bash
git status
git add .
git commit -m "Describe your change"
git pull
git push
git branch
git switch -c feature/demo
git switch main
ssh -T git@github.com
git remote -v
git remote set-url origin git@github.com:用户名/仓库名.git
```

前 8 条够大多数基础课堂练习。  
后 3 条是给 SSH 提交代码时最常用的检查和切换命令。

---

## 最后只记住这 10 句话就够了

1. GitHub 是远程仓库平台，VS Code 是你平时操作它的工作台。
2. VS Code 的 `Source Control` 是最常用入口。
3. 没装 Git，很多功能都用不了。
4. 第一次使用先配 `user.name` 和 `user.email`。
5. 不会命令行也没关系，基础流程用 VS Code 图形界面就够。
6. `Commit` 是本地保存版本，`Push` 才是传到 GitHub。
7. `Pull` 是把 GitHub 最新内容拿回来。
8. 团队协作尽量用分支，不要老是直接改 `main`。
9. `Publish to GitHub` 是新手最省事的第一条路。
10. 学会 `打开项目 -> 改文件 -> 暂存 -> 提交 -> 推送`，你就已经入门了。
