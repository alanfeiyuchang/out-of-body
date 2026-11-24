import Foundation
import UIKit

/// 真机上看不到 Xcode 控制台，所以日志同时写到 Documents/log.txt（每次启动清空）。
/// 带 `-log YES` 启动时也打到标准输出，给 devicectl --console 看。
enum DebugLog {
    private static let queue = DispatchQueue(label: "outofbody.log")
    private static let toStdout = UserDefaults.standard.bool(forKey: "log")
    private static let start = CACurrentMediaTime()
    private static var handle: FileHandle?
    private static var mainBeat = CACurrentMediaTime()
    private static var watchdog: DispatchSourceTimer?
    private static var stalled = false

    static func write(_ text: String) {
        let line = String(format: "[%6.1f] %@\n", CACurrentMediaTime() - start, text)
        queue.async {
            if toStdout {
                print(line, terminator: "")
                fflush(stdout)
            }
            if let data = line.data(using: .utf8) { handle?.write(data) }
        }
    }

    /// 主线程上调一次：开日志文件，记前后台切换，起一个看门狗盯着主线程有没有卡住。
    static func install() {
        if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let url = dir.appendingPathComponent("log.txt")
            FileManager.default.createFile(atPath: url.path, contents: nil)
            queue.sync { handle = try? FileHandle(forWritingTo: url) }
        }
        let nc = NotificationCenter.default
        for (name, label) in [
            (UIApplication.didBecomeActiveNotification, "前台激活"),
            (UIApplication.willResignActiveNotification, "失去激活"),
            (UIApplication.didEnterBackgroundNotification, "进入后台"),
            (UIApplication.willEnterForegroundNotification, "回到前台"),
        ] {
            nc.addObserver(forName: name, object: nil, queue: .main) { _ in write("APP " + label) }
        }
        Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { _ in
            queue.async { mainBeat = CACurrentMediaTime() }
        }
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + 2, repeating: 1)
        t.setEventHandler {
            let gap = CACurrentMediaTime() - mainBeat
            if gap > 1.5, !stalled {
                stalled = true
                write(String(format: "MAIN 主线程 %.1f 秒没响应", gap))
            } else if gap < 0.6, stalled {
                stalled = false
                write("MAIN 主线程恢复")
            }
        }
        t.resume()
        watchdog = t
    }
}
