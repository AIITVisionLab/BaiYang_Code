/**
 * @file main.cpp
 * 应用入口
 */

#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QIcon>
#include <QDebug>

#include "gui/QmlBridge.h"
#include "gui/VideoFrameProvider.h"
#include "core/Detection.h"

int main(int argc, char *argv[])
{
    // 高 DPI 支持
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif

    QGuiApplication app(argc, argv);
    
    // 应用信息
    app.setApplicationName("YOLOv11Qt");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("YOLOv11Qt");
    app.setOrganizationDomain("yolov11qt.local");

    // 设置应用图标
    app.setWindowIcon(QIcon(":/resources/icons/app_icon.png"));

    // 注册元类型，方便跨线程/信号传递
    qRegisterMetaType<yolo::Detection>("Detection");
    qRegisterMetaType<yolo::DetectionResult>("DetectionResult");
    qRegisterMetaType<yolo::Frame>("Frame");
    qRegisterMetaType<QImage>("QImage");

    // 创建 QML 引擎
    QQmlApplicationEngine engine;

    // 创建视频帧提供器
    yolo::VideoFrameProvider* frameProvider = new yolo::VideoFrameProvider();
    engine.addImageProvider("video", frameProvider);

    // 创建帧更新发射器
    yolo::FrameEmitter* frameEmitter = new yolo::FrameEmitter(&app);

    // 创建 QML 桥接对象
    yolo::QmlBridge* bridge = new yolo::QmlBridge(&app);

    // 连接帧更新信号
    QObject::connect(bridge, &yolo::QmlBridge::frameWithDetectionsReady,
                     [frameProvider, frameEmitter](const QImage& frame) {
                         frameProvider->updateFrame(frame);
                         frameEmitter->emitFrameUpdate();
                     });

    // 注册到 QML 上下文
    QQmlContext* context = engine.rootContext();
    context->setContextProperty("backend", bridge);
    context->setContextProperty("frameEmitter", frameEmitter);

    // 注册 QML 类型
    qmlRegisterUncreatableType<yolo::QmlBridge>("YOLOv11", 1, 0, "Backend",
        "Backend is provided by the application");

    // 加载主 QML 文件
    const QUrl url(QStringLiteral("qrc:/YOLOv11App/qml/main.qml"));
    
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
                     &app, [url](QObject *obj, const QUrl &objUrl) {
                         if (!obj && url == objUrl)
                             QCoreApplication::exit(-1);
                     }, Qt::QueuedConnection);
    
    engine.load(url);

    qDebug() << "YOLOv11Qt started";
    qDebug() << "Available engines:" << bridge->availableEngines();

    return app.exec();
}
