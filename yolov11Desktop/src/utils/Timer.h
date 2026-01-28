/**
 * @file Timer.h
 * @brief 高精度计时器与性能分析工具
 */

#ifndef YOLO_TIMER_H
#define YOLO_TIMER_H

#include <QString>
#include <QMap>
#include <QMutex>
#include <chrono>
#include <vector>
#include <memory>

namespace yolo {

/**
 * @brief 高精度计时器类
 *
 * 用于测量代码执行时间，支持：
 * - 毫秒/微秒精度
 * - 累计计时
 * - 平均值计算
 */
class Timer
{
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::nanoseconds;
    
    /**
     * @brief 构造函数
     * @param autoStart 是否自动开始计时
     */
    explicit Timer(bool autoStart = false);
    
    /**
     * @brief 开始计时
     */
    void start();
    
    /**
     * @brief 停止计时
     * @return 返回经过的时间（毫秒）
     */
    double stop();
    
    /**
     * @brief 重置计时器
     */
    void reset();
    
    /**
     * @brief 暂停计时
     */
    void pause();
    
    /**
     * @brief 恢复计时
     */
    void resume();
    
    /**
     * @brief 获取经过的时间（毫秒）
     */
    double elapsedMs() const;
    
    /**
     * @brief 获取经过的时间（微秒）
     */
    double elapsedUs() const;
    
    /**
     * @brief 获取经过的时间（秒）
     */
    double elapsedSec() const;
    
    /**
     * @brief 是否正在运行
     */
    bool isRunning() const { return m_running; }
    
    /**
     * @brief 记录一次计时（用于计算平均值）
     */
    void lap();
    
    /**
     * @brief 获取所有计次时间
     */
    std::vector<double> laps() const { return m_laps; }
    
    /**
     * @brief 获取平均计次时间（毫秒）
     */
    double averageLapMs() const;
    
    /**
     * @brief 获取最小计次时间（毫秒）
     */
    double minLapMs() const;
    
    /**
     * @brief 获取最大计次时间（毫秒）
     */
    double maxLapMs() const;
    
    /**
     * @brief 获取总计时次数
     */
    int lapCount() const { return static_cast<int>(m_laps.size()); }

private:
    TimePoint m_startTime;
    Duration m_accumulated;
    std::vector<double> m_laps;
    bool m_running;
    bool m_paused;
};

/**
 * @brief RAII 作用域计时器
 *
 * 构造时自动开始计时，析构时输出结果
 */
class ScopedTimer
{
public:
    /**
     * @brief 构造函数
     * @param name 计时器名称（用于输出）
     * @param printOnDestruct 析构时是否打印结果
     */
    explicit ScopedTimer(const QString& name, bool printOnDestruct = true);
    
    /**
     * @brief 析构函数
     */
    ~ScopedTimer();
    
    /**
     * @brief 获取经过的时间（毫秒）
     */
    double elapsedMs() const { return m_timer.elapsedMs(); }

private:
    QString m_name;
    Timer m_timer;
    bool m_printOnDestruct;
};

/**
 * @brief 性能分析器（单例）
 *
 * 用于收集与分析多个计时点的性能数据
 */
class Profiler
{
public:
    /**
     * @brief 性能统计数据
     */
    struct Stats {
        QString name;
        double totalMs;
        double avgMs;
        double minMs;
        double maxMs;
        int count;
    };
    
    /**
     * @brief 获取单例实例
     */
    static Profiler& instance();
    
    /**
     * @brief 开始一个计时段
     */
    void begin(const QString& name);
    
    /**
     * @brief 结束一个计时段
     */
    void end(const QString& name);
    
    /**
     * @brief 获取指定计时段的统计
     */
    Stats getStats(const QString& name) const;
    
    /**
     * @brief 获取所有计时段的统计
     */
    QMap<QString, Stats> getAllStats() const;
    
    /**
     * @brief 重置所有统计数据
     */
    void reset();
    
    /**
     * @brief 重置指定计时段
     */
    void reset(const QString& name);
    
    /**
     * @brief 启用/禁用性能分析
     */
    void setEnabled(bool enabled) { m_enabled = enabled; }
    
    /**
     * @brief 是否启用
     */
    bool isEnabled() const { return m_enabled; }
    
    /**
     * @brief 打印所有统计信息
     */
    void printReport() const;
    
    /**
     * @brief 导出报告到JSON
     */
    QString exportToJson() const;

private:
    Profiler();
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;
    
    struct TimerData {
        std::shared_ptr<Timer> timer;
        double totalMs = 0.0;
        double minMs = std::numeric_limits<double>::max();
        double maxMs = 0.0;
        int count = 0;
        
        TimerData() = default;
        TimerData(const TimerData&) = default;
        TimerData(TimerData&&) = default;
        TimerData& operator=(const TimerData&) = default;
        TimerData& operator=(TimerData&&) = default;
    };
    
    QMap<QString, TimerData> m_timers;
    bool m_enabled;
    mutable QMutex m_mutex;
};

/**
 * @brief RAII性能分析作用域
 */
class ProfileScope
{
public:
    explicit ProfileScope(const QString& name);
    ~ProfileScope();

private:
    QString m_name;
    bool m_enabled;
};

} // namespace yolo

// 便捷宏定义
#define PROFILE_SCOPE(name) yolo::ProfileScope _profile_##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)

#endif // YOLO_TIMER_H
