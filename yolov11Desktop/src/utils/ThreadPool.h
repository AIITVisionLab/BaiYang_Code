/**
 * @file ThreadPool.h
 * @brief 线程池实现（用于并行处理任务）
 */

#ifndef YOLO_THREADPOOL_H
#define YOLO_THREADPOOL_H

#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QQueue>
#include <QVector>
#include <functional>
#include <future>
#include <memory>

namespace yolo {

/**
 * @brief 工作线程类
 */
class WorkerThread : public QThread
{
    Q_OBJECT

public:
    explicit WorkerThread(QObject* parent = nullptr);
    ~WorkerThread();
    
    /**
     * @brief 提交任务
     */
    void submitTask(std::function<void()> task);
    
    /**
     * @brief 停止线程
     */
    void stop();
    
    /**
     * @brief 是否忙碌
     */
    bool isBusy() const { return m_busy; }
    
    /**
     * @brief 获取已完成任务数
     */
    int completedTaskCount() const { return m_completedTasks; }

protected:
    void run() override;

private:
    QQueue<std::function<void()>> m_taskQueue;
    QMutex m_mutex;
    QWaitCondition m_condition;
    bool m_running;
    bool m_busy;
    int m_completedTasks;
};

/**
 * @brief 线程池类
 *
 * 管理工作线程，支持：
 * - 任务队列
 * - 负载均衡
 * - 动态调整线程数
 */
class ThreadPool : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief 获取单例实例
     */
    static ThreadPool& instance();
    
    /**
     * @brief 构造函数
     * @param threadCount 线程数量，0表示自动检测
     */
    explicit ThreadPool(int threadCount = 0, QObject* parent = nullptr);
    
    /**
     * @brief 析构函数
     */
    ~ThreadPool();
    
    /**
     * @brief 启动线程池
     */
    void start();
    
    /**
     * @brief 停止线程池
     */
    void stop();
    
    /**
     * @brief 等待所有任务完成
     */
    void waitForDone();
    
    /**
     * @brief 提交任务
     */
    void submit(std::function<void()> task);
    
    /**
     * @brief 提交带返回值的任务
     */
    template<typename F, typename... Args>
    auto submitWithResult(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type>
    {
        using ReturnType = typename std::invoke_result<F, Args...>::type;
        
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<ReturnType> result = task->get_future();
        
        submit([task]() { (*task)(); });
        
        return result;
    }
    
    /**
     * @brief 并行执行任务
     */
    template<typename Iterator, typename Func>
    void parallelFor(Iterator begin, Iterator end, Func func)
    {
        std::vector<std::future<void>> futures;
        
        for (auto it = begin; it != end; ++it) {
            futures.push_back(submitWithResult([func, it]() { func(*it); }));
        }
        
        for (auto& future : futures) {
            future.wait();
        }
    }
    
    /**
     * @brief 获取线程数量
     */
    int threadCount() const { return m_threads.size(); }
    
    /**
     * @brief 设置线程数量
     */
    void setThreadCount(int count);
    
    /**
     * @brief 获取待处理任务数
     */
    int pendingTaskCount() const;
    
    /**
     * @brief 获取已完成任务数
     */
    int completedTaskCount() const;
    
    /**
     * @brief 是否正在运行
     */
    bool isRunning() const { return m_running; }

signals:
    /**
     * @brief 任务完成信号
     */
    void taskCompleted();
    
    /**
     * @brief 所有任务完成信号
     */
    void allTasksCompleted();

private:
    WorkerThread* selectWorker();
    
    QVector<WorkerThread*> m_threads;
    QQueue<std::function<void()>> m_taskQueue;
    mutable QMutex m_mutex;
    QWaitCondition m_condition;
    bool m_running;
    int m_nextWorker;
};

/**
 * @brief 简单任务类
 */
class Task : public QObject
{
    Q_OBJECT

public:
    enum class State {
        Pending,
        Running,
        Completed,
        Cancelled,
        Failed
    };
    
    explicit Task(QObject* parent = nullptr);
    virtual ~Task() = default;
    
    /**
     * @brief 执行任务（子类重写）
     */
    virtual void run() = 0;
    
    /**
     * @brief 获取任务状态
     */
    State state() const { return m_state; }
    
    /**
     * @brief 取消任务
     */
    void cancel();
    
    /**
     * @brief 是否已取消
     */
    bool isCancelled() const { return m_cancelled; }
    
    /**
     * @brief 获取进度 (0-100)
     */
    int progress() const { return m_progress; }
    
    /**
     * @brief 获取错误信息
     */
    QString errorMessage() const { return m_errorMessage; }

signals:
    void started();
    void finished();
    void cancelled();
    void failed(const QString& error);
    void progressChanged(int progress);

protected:
    void setProgress(int progress);
    void setError(const QString& error);
    void setState(State state);
    
    State m_state;
    bool m_cancelled;
    int m_progress;
    QString m_errorMessage;
    mutable QMutex m_mutex;
};

/**
 * @brief Lambda任务包装器
 */
class LambdaTask : public Task
{
    Q_OBJECT

public:
    explicit LambdaTask(std::function<void()> func, QObject* parent = nullptr);
    void run() override;

private:
    std::function<void()> m_func;
};

} // namespace yolo

#endif // YOLO_THREADPOOL_H
