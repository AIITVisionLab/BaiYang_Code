/**
 * @file Logger.h
 * @brief 日志管理系统（多级别、文件输出、控制台输出）
 */

#ifndef YOLO_LOGGER_H
#define YOLO_LOGGER_H

#include <QString>
#include <QFile>
#include <QTextStream>
#include <QMutex>
#include <QDateTime>
#include <memory>
#include <functional>

namespace yolo {

/**
 * @brief 日志级别枚举
 */
enum class LogLevel {
    Trace = 0,  ///< 最详细的跟踪信息
    Debug,      ///< 调试信息
    Info,       ///< 一般信息
    Warning,    ///< 警告信息
    Error,      ///< 错误信息
    Critical,   ///< 严重错误
    Off         ///< 关闭日志
};

/**
 * @brief 日志管理类（单例）
 *
 * 提供统一日志接口，支持：
 * - 多级别日志过滤
 * - 控制台彩色输出
 * - 文件日志轮转
 * - 自定义日志回调
 */
class Logger
{
public:
    /**
     * @brief 获取单例实例
     */
    static Logger& instance();
    
    /**
     * @brief 日志回调函数类型
     */
    using LogCallback = std::function<void(LogLevel, const QString&, const QString&)>;
    
    /**
     * @brief 设置日志级别
     */
    void setLevel(LogLevel level);
    
    /**
     * @brief 获取当前日志级别
     */
    LogLevel level() const { return m_level; }
    
    /**
     * @brief 启用控制台输出
     */
    void setConsoleEnabled(bool enabled);
    
    /**
     * @brief 是否启用了控制台输出
     */
    bool isConsoleEnabled() const { return m_consoleEnabled; }
    
    /**
     * @brief 启用文件日志
     * @param filePath 日志文件路径，空则使用默认路径
     */
    bool setFileEnabled(bool enabled, const QString& filePath = QString());
    
    /**
     * @brief 是否启用了文件日志
     */
    bool isFileEnabled() const { return m_fileEnabled; }
    
    /**
     * @brief 设置最大日志文件大小（字节）
     */
    void setMaxFileSize(qint64 size) { m_maxFileSize = size; }
    
    /**
     * @brief 设置最大日志文件数量
     */
    void setMaxFileCount(int count) { m_maxFileCount = count; }
    
    /**
     * @brief 设置自定义日志回调
     */
    void setCallback(LogCallback callback);
    
    /**
     * @brief 记录日志
     */
    void log(LogLevel level, const QString& message, const QString& file = QString(), 
             int line = -1, const QString& function = QString());
    
    /**
     * @brief 便捷日志方法
     */
    void trace(const QString& message, const QString& file = QString(), int line = -1);
    void debug(const QString& message, const QString& file = QString(), int line = -1);
    void info(const QString& message, const QString& file = QString(), int line = -1);
    void warning(const QString& message, const QString& file = QString(), int line = -1);
    void error(const QString& message, const QString& file = QString(), int line = -1);
    void critical(const QString& message, const QString& file = QString(), int line = -1);
    
    /**
     * @brief 获取日志级别名称
     */
    static QString levelName(LogLevel level);
    
    /**
     * @brief 从字符串解析日志级别
     */
    static LogLevel levelFromString(const QString& str);
    
    /**
     * @brief 刷新日志缓冲区
     */
    void flush();

private:
    Logger();
    ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    void writeToConsole(LogLevel level, const QString& formattedMessage);
    void writeToFile(const QString& formattedMessage);
    void rotateLogFile();
    QString formatMessage(LogLevel level, const QString& message, 
                          const QString& file, int line, const QString& function);
    QString colorize(LogLevel level, const QString& text);

private:
    LogLevel m_level;
    bool m_consoleEnabled;
    bool m_fileEnabled;
    QString m_filePath;
    std::unique_ptr<QFile> m_file;
    std::unique_ptr<QTextStream> m_stream;
    qint64 m_maxFileSize;
    int m_maxFileCount;
    LogCallback m_callback;
    mutable QMutex m_mutex;
};

} // namespace yolo

// 便捷宏定义
#define LOG_TRACE(msg)    yolo::Logger::instance().trace(msg, __FILE__, __LINE__)
#define LOG_DEBUG(msg)    yolo::Logger::instance().debug(msg, __FILE__, __LINE__)
#define LOG_INFO(msg)     yolo::Logger::instance().info(msg, __FILE__, __LINE__)
#define LOG_WARNING(msg)  yolo::Logger::instance().warning(msg, __FILE__, __LINE__)
#define LOG_ERROR(msg)    yolo::Logger::instance().error(msg, __FILE__, __LINE__)
#define LOG_CRITICAL(msg) yolo::Logger::instance().critical(msg, __FILE__, __LINE__)

#endif // YOLO_LOGGER_H
