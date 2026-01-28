/**
 * @file Timer.cpp
 * @brief 高精度计时器与性能分析工具实现
 */

#include "Timer.h"
#include "Logger.h"
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <algorithm>
#include <numeric>

namespace yolo {

// ===== Timer =====

Timer::Timer(bool autoStart)
    : m_accumulated(Duration::zero())
    , m_running(false)
    , m_paused(false)
{
    if (autoStart) {
        start();
    }
}

void Timer::start()
{
    m_startTime = Clock::now();
    m_accumulated = Duration::zero();
    m_running = true;
    m_paused = false;
}

double Timer::stop()
{
    if (!m_running) return 0.0;
    
    double elapsed = elapsedMs();
    m_running = false;
    m_paused = false;
    return elapsed;
}

void Timer::reset()
{
    m_startTime = Clock::now();
    m_accumulated = Duration::zero();
    m_laps.clear();
    m_running = false;
    m_paused = false;
}

void Timer::pause()
{
    if (!m_running || m_paused) return;
    
    m_accumulated += Clock::now() - m_startTime;
    m_paused = true;
}

void Timer::resume()
{
    if (!m_running || !m_paused) return;
    
    m_startTime = Clock::now();
    m_paused = false;
}

double Timer::elapsedMs() const
{
    Duration elapsed = m_accumulated;
    if (m_running && !m_paused) {
        elapsed += Clock::now() - m_startTime;
    }
    return std::chrono::duration<double, std::milli>(elapsed).count();
}

double Timer::elapsedUs() const
{
    return elapsedMs() * 1000.0;
}

double Timer::elapsedSec() const
{
    return elapsedMs() / 1000.0;
}

void Timer::lap()
{
    double current = elapsedMs();
    m_laps.push_back(current);
    
    // 重新开始计时
    m_startTime = Clock::now();
    m_accumulated = Duration::zero();
}

double Timer::averageLapMs() const
{
    if (m_laps.empty()) return 0.0;
    double sum = std::accumulate(m_laps.begin(), m_laps.end(), 0.0);
    return sum / m_laps.size();
}

double Timer::minLapMs() const
{
    if (m_laps.empty()) return 0.0;
    return *std::min_element(m_laps.begin(), m_laps.end());
}

double Timer::maxLapMs() const
{
    if (m_laps.empty()) return 0.0;
    return *std::max_element(m_laps.begin(), m_laps.end());
}

// ===== ScopedTimer =====

ScopedTimer::ScopedTimer(const QString& name, bool printOnDestruct)
    : m_name(name)
    , m_timer(true)  // 自动开始
    , m_printOnDestruct(printOnDestruct)
{
}

ScopedTimer::~ScopedTimer()
{
    double elapsed = m_timer.elapsedMs();
    if (m_printOnDestruct) {
        LOG_DEBUG(QString("%1: %.3f ms").arg(m_name).arg(elapsed));
    }
}

// ===== Profiler =====

Profiler& Profiler::instance()
{
    static Profiler instance;
    return instance;
}

Profiler::Profiler()
    : m_enabled(false)
{
}

void Profiler::begin(const QString& name)
{
    if (!m_enabled) return;
    
    QMutexLocker locker(&m_mutex);
    
    auto it = m_timers.find(name);
    if (it == m_timers.end()) {
        TimerData data;
        data.timer = std::make_shared<Timer>();
        m_timers.insert(name, data);
        it = m_timers.find(name);
    }
    
    it->timer->start();
}

void Profiler::end(const QString& name)
{
    if (!m_enabled) return;
    
    QMutexLocker locker(&m_mutex);
    
    auto it = m_timers.find(name);
    if (it == m_timers.end()) return;
    
    double elapsed = it->timer->stop();
    it->totalMs += elapsed;
    it->minMs = std::min(it->minMs, elapsed);
    it->maxMs = std::max(it->maxMs, elapsed);
    it->count++;
}

Profiler::Stats Profiler::getStats(const QString& name) const
{
    QMutexLocker locker(&m_mutex);
    
    Stats stats;
    stats.name = name;
    
    auto it = m_timers.find(name);
    if (it != m_timers.end()) {
        stats.totalMs = it->totalMs;
        stats.avgMs = it->count > 0 ? it->totalMs / it->count : 0.0;
        stats.minMs = it->count > 0 ? it->minMs : 0.0;
        stats.maxMs = it->maxMs;
        stats.count = it->count;
    }
    
    return stats;
}

QMap<QString, Profiler::Stats> Profiler::getAllStats() const
{
    QMutexLocker locker(&m_mutex);
    
    QMap<QString, Stats> allStats;
    for (auto it = m_timers.begin(); it != m_timers.end(); ++it) {
        Stats stats;
        stats.name = it.key();
        stats.totalMs = it->totalMs;
        stats.avgMs = it->count > 0 ? it->totalMs / it->count : 0.0;
        stats.minMs = it->count > 0 ? it->minMs : 0.0;
        stats.maxMs = it->maxMs;
        stats.count = it->count;
        allStats.insert(it.key(), stats);
    }
    
    return allStats;
}

void Profiler::reset()
{
    QMutexLocker locker(&m_mutex);
    m_timers.clear();
}

void Profiler::reset(const QString& name)
{
    QMutexLocker locker(&m_mutex);
    m_timers.remove(name);
}

void Profiler::printReport() const
{
    QMutexLocker locker(&m_mutex);
    
    LOG_INFO("========== Performance Report ==========");
    
    for (auto it = m_timers.begin(); it != m_timers.end(); ++it) {
        double avgMs = it->count > 0 ? it->totalMs / it->count : 0.0;
        LOG_INFO(QString("%1: count=%2, total=%.3f ms, avg=%.3f ms, min=%.3f ms, max=%.3f ms")
                 .arg(it.key())
                 .arg(it->count)
                 .arg(it->totalMs)
                 .arg(avgMs)
                 .arg(it->count > 0 ? it->minMs : 0.0)
                 .arg(it->maxMs));
    }
    
    LOG_INFO("========================================");
}

QString Profiler::exportToJson() const
{
    QMutexLocker locker(&m_mutex);
    
    QJsonObject root;
    QJsonArray profiles;
    
    for (auto it = m_timers.begin(); it != m_timers.end(); ++it) {
        QJsonObject profile;
        profile["name"] = it.key();
        profile["count"] = it->count;
        profile["totalMs"] = it->totalMs;
        profile["avgMs"] = it->count > 0 ? it->totalMs / it->count : 0.0;
        profile["minMs"] = it->count > 0 ? it->minMs : 0.0;
        profile["maxMs"] = it->maxMs;
        profiles.append(profile);
    }
    
    root["profiles"] = profiles;
    
    QJsonDocument doc(root);
    return QString::fromUtf8(doc.toJson(QJsonDocument::Indented));
}

// ==================== ProfileScope ====================

ProfileScope::ProfileScope(const QString& name)
    : m_name(name)
    , m_enabled(Profiler::instance().isEnabled())
{
    if (m_enabled) {
        Profiler::instance().begin(m_name);
    }
}

ProfileScope::~ProfileScope()
{
    if (m_enabled) {
        Profiler::instance().end(m_name);
    }
}

} // namespace yolo
