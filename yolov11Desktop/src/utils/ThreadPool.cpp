/**
 * @file ThreadPool.cpp
 * @brief 线程池实现
 */

#include "ThreadPool.h"
#include "Logger.h"
#include <QThread>

namespace yolo {

// ===== WorkerThread =====

WorkerThread::WorkerThread(QObject* parent)
    : QThread(parent)
    , m_running(false)
    , m_busy(false)
    , m_completedTasks(0)
{
}

WorkerThread::~WorkerThread()
{
    stop();
}

void WorkerThread::submitTask(std::function<void()> task)
{
    QMutexLocker locker(&m_mutex);
    m_taskQueue.enqueue(task);
    m_condition.wakeOne();
}

void WorkerThread::stop()
{
    {
        QMutexLocker locker(&m_mutex);
        m_running = false;
        m_condition.wakeOne();
    }
    
    if (isRunning()) {
        wait(5000);  // 最多等待 5 秒
        if (isRunning()) {
            terminate();
            wait();
        }
    }
}

void WorkerThread::run()
{
    m_running = true;
    
    while (m_running) {
        std::function<void()> task;
        
        {
            QMutexLocker locker(&m_mutex);
            
            while (m_taskQueue.isEmpty() && m_running) {
                m_busy = false;
                m_condition.wait(&m_mutex);
            }
            
            if (!m_running) break;
            
            task = m_taskQueue.dequeue();
            m_busy = true;
        }
        
        if (task) {
            try {
                task();
            } catch (const std::exception& e) {
                LOG_ERROR(QString("Worker thread exception: %1").arg(e.what()));
            } catch (...) {
                LOG_ERROR("Worker thread unknown exception");
            }
            m_completedTasks++;
        }
    }
    
    m_busy = false;
}

// ===== ThreadPool =====

ThreadPool& ThreadPool::instance()
{
    static ThreadPool instance(0);
    return instance;
}

ThreadPool::ThreadPool(int threadCount, QObject* parent)
    : QObject(parent)
    , m_running(false)
    , m_nextWorker(0)
{
    int count = threadCount > 0 ? threadCount : QThread::idealThreadCount();
    count = qMax(1, qMin(count, 32));  // 限制 1-32 个线程
    
    for (int i = 0; i < count; ++i) {
        WorkerThread* worker = new WorkerThread(this);
        m_threads.append(worker);
    }
    
    LOG_DEBUG(QString("ThreadPool created with %1 threads").arg(count));
}

ThreadPool::~ThreadPool()
{
    stop();
    
    for (WorkerThread* thread : m_threads) {
        delete thread;
    }
    m_threads.clear();
}

void ThreadPool::start()
{
    if (m_running) return;
    
    m_running = true;
    
    for (WorkerThread* thread : m_threads) {
        thread->start();
    }
    
    LOG_DEBUG("ThreadPool started");
}

void ThreadPool::stop()
{
    if (!m_running) return;
    
    m_running = false;
    
    for (WorkerThread* thread : m_threads) {
        thread->stop();
    }
    
    LOG_DEBUG("ThreadPool stopped");
}

void ThreadPool::waitForDone()
{
    // 等待所有任务完成
    bool allDone = false;
    while (!allDone) {
        allDone = true;
        for (WorkerThread* thread : m_threads) {
            if (thread->isBusy()) {
                allDone = false;
                break;
            }
        }
        
        if (!allDone) {
            QThread::msleep(10);
        }
    }
    
    emit allTasksCompleted();
}

void ThreadPool::submit(std::function<void()> task)
{
    if (!m_running) {
        start();
    }
    
    WorkerThread* worker = selectWorker();
    worker->submitTask([this, task]() {
        task();
        emit taskCompleted();
    });
}

void ThreadPool::setThreadCount(int count)
{
    if (count == m_threads.size()) return;
    
    bool wasRunning = m_running;
    if (wasRunning) stop();
    
    // 删除现有线程
    for (WorkerThread* thread : m_threads) {
        delete thread;
    }
    m_threads.clear();
    
    // 创建新线程
    count = qMax(1, qMin(count, 32));
    for (int i = 0; i < count; ++i) {
        WorkerThread* worker = new WorkerThread(this);
        m_threads.append(worker);
    }
    
    if (wasRunning) start();
    
    LOG_DEBUG(QString("ThreadPool resized to %1 threads").arg(count));
}

int ThreadPool::pendingTaskCount() const
{
    QMutexLocker locker(&m_mutex);
    return m_taskQueue.size();
}

int ThreadPool::completedTaskCount() const
{
    int total = 0;
    for (WorkerThread* thread : m_threads) {
        total += thread->completedTaskCount();
    }
    return total;
}

WorkerThread* ThreadPool::selectWorker()
{
    // 简单的轮询调度
    QMutexLocker locker(&m_mutex);
    
    WorkerThread* worker = m_threads[m_nextWorker];
    m_nextWorker = (m_nextWorker + 1) % m_threads.size();
    
    return worker;
}

// ==================== Task ====================

Task::Task(QObject* parent)
    : QObject(parent)
    , m_state(State::Pending)
    , m_cancelled(false)
    , m_progress(0)
{
}

void Task::cancel()
{
    QMutexLocker locker(&m_mutex);
    m_cancelled = true;
    if (m_state == State::Pending) {
        m_state = State::Cancelled;
        emit cancelled();
    }
}

void Task::setProgress(int progress)
{
    QMutexLocker locker(&m_mutex);
    m_progress = qBound(0, progress, 100);
    emit progressChanged(m_progress);
}

void Task::setError(const QString& error)
{
    QMutexLocker locker(&m_mutex);
    m_errorMessage = error;
    m_state = State::Failed;
    emit failed(error);
}

void Task::setState(State state)
{
    QMutexLocker locker(&m_mutex);
    m_state = state;
    
    switch (state) {
        case State::Running:
            emit started();
            break;
        case State::Completed:
            emit finished();
            break;
        case State::Cancelled:
            emit cancelled();
            break;
        default:
            break;
    }
}

// ==================== LambdaTask ====================

LambdaTask::LambdaTask(std::function<void()> func, QObject* parent)
    : Task(parent)
    , m_func(func)
{
}

void LambdaTask::run()
{
    if (isCancelled()) return;
    
    setState(State::Running);
    
    try {
        m_func();
        if (!isCancelled()) {
            setState(State::Completed);
        }
    } catch (const std::exception& e) {
        setError(QString::fromStdString(e.what()));
    }
}

} // namespace yolo
