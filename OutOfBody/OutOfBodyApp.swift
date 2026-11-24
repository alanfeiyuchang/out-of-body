import SwiftUI

@main
struct OutOfBodyApp: App {
    init() {
        // `-probe YES`：把设备能力打到标准输出然后退出，给 devicectl --console 用。
        if UserDefaults.standard.bool(forKey: "probe") {
            print("PROBE-BEGIN")
            print(Capabilities.probe().json)
            print("PROBE-END")
            fflush(stdout)
            exit(0)
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
