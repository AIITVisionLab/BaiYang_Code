/**
 * @file Logger.cpp
 * @brief 日志管理系统实现
 */

#include "Logger.h"
#include <QCoreApplication>
#include <QStandardPaths>
#include <QDir>
#include <QFileInfo>
#include <iostream>

namespace yolo {

Logger& Logger::instance()
{
    static Logger instance;
    return instance;
}

Logger::Logger()
    : m_level(LogLevel::Info)
    , m_consoleEnabled(true)
    , m_fileEnabled(false)
    , m_maxFileSize(10 * 1024 * 1024)  // 10 MB
    , m_maxFileCount(5)
{
}

Logger::~Logger()
{
    flush();
    if (m_file && m_file->isOpen()) {
        m_file->close();
    }
}

void Logger::setLevel(LogLevel level)
{
    QMutexLocker locker(&m_mutex);
    m_level = level;
}

void Logger::setConsoleEnabled(bool enabled)
{
    QMutexLocker locker(&m_mutex);
    m_consoleEnabled = enabled;
}

bool Logger::setFileEnabled(bool enabled, const QString& filePath)
{
    QMutexLocker locker(&m_mutex);
    
    if (!enabled) {
        if (m_file && m_file->isOpen()) {
            m_stream.reset();
            m_file->close();
            m_file.reset();
        }
        m_fileEnabled = false;
        return true;
    }
    
    // 确定日志文件路径
    if (filePath.isEmpty()) {
        QString logDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
        QDir().mkpath(logDir);
        m_filePath = logDir + "/yolov11qt.log";
    } else {
        m_filePath = filePath;
        QDir().mkpath(QFileInfo(filePath).absolutePath());
    }
    
    // 打开日志文件
    m_file = std::make_unique<QFile>(m_filePath);
    if (!m_file->open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) {
        m_file.reset();
        return false;
    }
    
    m_stream = std::make_unique<QTextStream>(m_file.get());
    m_stream->setEncoding(QStringConverter::Utf8);
    m_fileEnabled = true;
    
    return true;
}

void Logger::setCallback(LogCallback callback)
{
    QMutexLocker locker(&m_mutex);
    m_callback = callback;
}

void Logger::log(LogLevel level, const QString& message, const QString& file, 
                 int line, const QString& function)
{
    if (level < m_level || level == LogLevel::Off) {
        return;
    }
    
    QMutexLocker locker(&m_mutex);
    
    QString formattedMessage = formatMessage(level, message, file, line, function);
    
    if (m_consoleEnabled) {
        writeToConsole(level, formattedMessage);
    }
    
    if (m_fileEnabled && m_stream) {
        writeToFile(formattedMessage);
    }
    
    if (m_callback) {
        m_callback(level, message, formattedMessage);
    }
}

void Logger::trace(const QString& message, const QString& file, int line)
{
    log(LogLevel::Trace, message, file, line);
}

void Logger::debug(const QString& message, const QString& file, int line)
{
    log(LogLevel::Debug, message, file, line);
}

void Logger::info(const QString& message, const QString& file, int line)
{
    log(LogLevel::Info, message, file, line);
}

void Logger::warning(const QString& message, const QString& file, int line)
{
    log(LogLevel::Warning, message, file, line);
}

void Logger::error(const QString& message, const QString& file, int line)
{
    log(LogLevel::Error, message, file, line);
}

void Logger::critical(const QString& message, const QString& file, int line)
{
    log(LogLevel::Critical, message, file, line);
}

QString Logger::levelName(LogLevel level)
{
    switch (level) {
        case LogLevel::Trace:    return "TRACE";
        case LogLevel::Debug:    return "DEBUG";
        case LogLevel::Info:     return "INFO";
        case LogLevel::Warning:  return "WARN";
        case LogLevel::Error:    return "ERROR";
        case LogLevel::Critical: return "CRIT";
        default:                 return "UNKNOWN";
    }
}

LogLevel Logger::levelFromString(const QString& str)
{
    QString lower = str.toLower();
    if (lower == "trace") return LogLevel::Trace;
    if (lower == "debug") return LogLevel::Debug;
    if (lower == "info")  return LogLevel::Info;
    if (lower == "warn" || lower == "warning") return LogLevel::Warning;
    if (lower == "error") return LogLevel::Error;
    if (lower == "crit" || lower == "critical") return LogLevel::Critical;
    if (lower == "off")   return LogLevel::Off;
    return LogLevel::Info;
}

void Logger::flush()
{
    QMutexLocker locker(&m_mutex);
    if (m_stream) {
        m_stream->flush();
    }
}

void Logger::writeToConsole(LogLevel level, const QString& formattedMessage)
{
    QString coloredMessage = colorize(level, formattedMessage);
    
    if (level >= LogLevel::Error) {
        std::cerr << coloredMessage.toStdString() << std::endl;
    } else {
        std::cout << coloredMessage.toStdString() << std::endl;
    }
}

void Logger::writeToFile(const QString& formattedMessage)
{
    if (!m_stream) return;
    
    *m_stream << formattedMessage << "\n";
    m_stream->flush();
    
    // 检查文件大小，需要时轮转
    if (m_file->size() > m_maxFileSize) {
        rotateLogFile();
    }
}

void Logger::rotateLogFile()
{
    m_stream.reset();
    m_file->close();
    
    // 删除最旧的日志
    QString oldestFile = QString("%1.%2").arg(m_filePath).arg(m_maxFileCount);
    QFile::remove(oldestFile);
    
    // 重命名现有日志文件
    for (int i = m_maxFileCount - 1; i >= 1; --i) {
        QString oldName = QString("%1.%2").arg(m_filePath).arg(i);
        QString newName = QString("%1.%2").arg(m_filePath).arg(i + 1);
        QFile::rename(oldName, newName);
    }
    
    // 重命名当前日志
    QFile::rename(m_filePath, m_filePath + ".1");
    
    // 创建新日志文件
    m_file = std::make_unique<QFile>(m_filePath);
    m_file->open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text);
    m_stream = std::make_unique<QTextStream>(m_file.get());
    m_stream->setEncoding(QStringConverter::Utf8);
}

QString Logger::formatMessage(LogLevel level, const QString& message, 
                              const QString& file, int line, const QString& function)
{
    QString timestamp = QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss.zzz");
    QString levelStr = levelName(level);
    
    QString result = QString("[%1] [%2]").arg(timestamp, levelStr);
    
    if (!file.isEmpty()) {
        QString fileName = QFileInfo(file).fileName();
        result += QString(" [%1:%2]").arg(fileName).arg(line);
    }
    
    if (!function.isEmpty()) {
        result += QString(" [%1]").arg(function);
    }
    
    result += " " + message;
    
    return result;
}

QString Logger::colorize(LogLevel level, const QString& text)
{
#ifdef Q_OS_WIN
    // Windows控制台可能不支持ANSI颜色
    return text;
#else
    QString colorCode;
    switch (level) {
        case LogLevel::Trace:    colorCode = "\033[90m"; break;  // 灰色
        case LogLevel::Debug:    colorCode = "\033[36m"; break;  // 青色
        case LogLevel::Info:     colorCode = "\033[32m"; break;  // 绿色
        case LogLevel::Warning:  colorCode = "\033[33m"; break;  // 黄色
        case LogLevel::Error:    colorCode = "\033[31m"; break;  // 红色
        case LogLevel::Critical: colorCode = "\033[35m"; break;  // 紫色
        default:                 colorCode = "\033[0m";  break;  // 默认
    }
    return colorCode + text + "\033[0m";
#endif
}

} // namespace yolo
