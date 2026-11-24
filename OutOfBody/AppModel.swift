import Foundation
import Metal
import UIKit

/// 把数据来源、地图和渲染器连起来，顺便给界面攒一行状态文字。
final class AppModel: ObservableObject {
    @Published var hud = "启动中…"
    @Published var showLive = true

    let device: MTLDevice
    let map: SplatMap
    let source: FrameSource
    let predictor: ScenePredictor?
    let recorder = Recorder()
    let exporter = DatasetExporter()
    let neural: NeuralView
    /// 同一个服务上我们自己微调过的模型。
    let neuralTuned: NeuralView
    /// 手机上自己跑的小网络；包里没有模型文件时是 nil。
    let refiner: LocalRefiner?
    /// 最后那张图谁来画：0 = 自己渲染，1 = 服务器上的原版扩散模型重画整帧，
    /// 2 = 服务器上微调过的扩散模型（只画环境，人偶照常画在上面），3 = 手机上它的学生网络（同样的用法，不联网）。
    @Published var neuralMode = AppModel.startMode
    /// 一打开是哪一档。真机默认「服务器微调」；模拟器默认关（截图、回放、回归都是按「关」写的）。
    /// 启动参数能改：`-neural YES` / `-neuralFT YES|NO` / `-refine YES`。
    static let startMode: Int = {
        let d = UserDefaults.standard
        if d.bool(forKey: "neural") { return 1 }
        if d.bool(forKey: "refine") { return 3 }
        if d.object(forKey: "neuralFT") != nil { return d.bool(forKey: "neuralFT") ? 2 : 0 }
        #if targetEnvironment(simulator)
            return 0
        #else
            return 2
        #endif
    }()
    var neuralLabel: String { ["生成：关", "生成：服务器扩散", "生成：服务器微调", "生成：本机网络"][neuralMode] }
    @Published var recording = false
    /// 真机现在开合到什么程度：0 合上，1 完全展开；nil = 系统没给（不是折叠机，或者系统太旧），
    /// 那就按窗口大小猜。由界面上的铰链监听写，只在主线程读写。
    var hingeOpen: Float?
    /// 渲染器每秒写一次，只在主线程上读写。
    var fps = 0
    /// 人偶脚下的地面高度，渲染器每帧写。
    var floorY: Float?
    /// 渲染器建好之后把「渲一张全景图」的入口挂在这里，预测器用。
    var panorama: ((Vec3, @escaping (SplatMap.Panorama) -> Void) -> Bool)?

    private let lock = NSLock()
    private var pose: PoseSample?
    private var screenImage: (width: Int, height: Int, rgba: [UInt8])?
    private var pixelAngle: Float = 1.0 / 212
    private var timer: Timer?
    private var logTick = 0

    init() {
        DebugLog.install()
        device = MTLCreateSystemDefaultDevice()!
        // `-capacity 150000`：故意把地图容量设小，测试「快满了自动清理」。
        let cap = UserDefaults.standard.integer(forKey: "capacity")
        map = SplatMap(device: device, capacity: cap > 0 ? cap : 2_000_000)!
        predictor = ScenePredictor(device: device, map: map)
        neural = NeuralView(device: device)
        neuralTuned = NeuralView(device: device, tuned: true)
        predictor?.paused = AppModel.startMode != 0
        refiner = LocalRefiner(device: device)
        #if targetEnvironment(simulator)
            let synthetic = true
        #else
            let synthetic = UserDefaults.standard.bool(forKey: "synthetic")
        #endif
        let defaults = UserDefaults.standard
        MonoPipeline.mapSamples = { [map] c in map.alignmentSamples(for: c) }
        if defaults.bool(forKey: "replay") {
            let replay = ReplaySource()
            map.blocking = ReplaySource.fast
            let map = self.map
            let exporter = self.exporter
            let recorder = self.recorder
            replay.onFinished = { [weak replay] frames in
                // 边回放边录（给外部数据集补单目深度）的话，放完就收尾。
                if recorder.isRecording { recorder.stop() }
                if DatasetExporter.enabled, let replay {
                    let extra = DatasetExporter.thirdPersonViews(
                        poses: replay.capturePoses, time: Float(replay.lastCaptureTime),
                        count: DatasetExporter.thirdPersonViews)
                    let chase = DatasetExporter.chaseViews(
                        poses: replay.capturePoses, count: DatasetExporter.chaseViewCount)
                    exporter.enqueue(replay.heldViews + extra + chase)
                }
                let s = map.snapshot
                var line = "REPLAY 结束：\(frames) 帧（跳过 \(MonoPipeline.framesSkipped)），splat \(s.alive)"
                if defaults.bool(forKey: "evalSynthetic") {
                    let q = map.evaluate(
                        truth: SyntheticSource.distanceToTruth,
                        samples: SyntheticSource.truthSamples())
                    line += String(
                        format: "，画出来 %d，误差中位 %.1f cm，90 分位 %.1f cm，超过 8 cm 的占 %.1f%%，覆盖 %.1f%%",
                        q.drawn, q.median * 100, q.p90 * 100, q.far * 100, q.coverage * 100)
                    line += "｜地/中/顶 数量/中位/90分位：" + q.bands
                }
                DebugLog.write(line)
            }
            source = replay
        } else {
            source = synthetic ? SyntheticSource() : ARKitSource()
        }

        source.onPose = { [weak self] p in
            guard let self else { return }
            self.lock.lock()
            self.pose = p
            self.lock.unlock()
            self.recorder.add(pose: p)
        }
        source.onCapture = { [weak self] c in
            self?.recorder.add(c)
        }
        if defaults.bool(forKey: "record") { toggleRecording() }
        source.onDepth = { [weak self] f in
            guard let self else { return }
            self.lock.lock()
            self.screenImage = (f.width, f.height, f.color)
            self.pixelAngle = 1 / f.fx
            self.lock.unlock()
            self.map.submit(f)
        }
        source.onSparsePoints = { [weak self] pts in
            self?.map.submit(points: pts)
        }
        source.onThumbnail = { [weak self] w, h, rgba in
            guard let self else { return }
            self.lock.lock()
            self.screenImage = (w, h, rgba)
            self.lock.unlock()
        }
        source.start()
        UIApplication.shared.isIdleTimerDisabled = true

        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            guard let self else { return }
            let s = self.map.snapshot
            self.hud = String(
                format: "%@\nsplat %.0fk · 关键帧 %d · 累积 %.1f ms · %d fps\n%@",
                self.source.statusText, Double(s.alive) / 1000, s.keyframesUsed, s.integrateMs,
                self.fps, self.predictor?.status ?? "预测：不可用")
            if self.neuralMode == 1 { self.hud += "\n" + self.neural.status }
            if self.neuralMode == 2 { self.hud += "\n" + self.neuralTuned.status }
            if self.neuralMode == 3 { self.hud += "\n" + (self.refiner?.status ?? "本机网络：没有模型") }
            if self.recorder.isRecording {
                self.hud += "\n" + self.recorder.status
            } else if self.recording {
                self.recording = false  // 录满了自己停的
            }
            if self.logTick % 2 == 0, let p = self.currentPose() {
                self.predictor?.tick(userPosition: p.cameraToWorld.position, floorY: self.floorY)
            }
            // 每两秒把这行状态记进日志。
            self.logTick += 1
            if self.logTick % 4 == 0 {
                DebugLog.write("HUD " + self.hud.replacingOccurrences(of: "\n", with: " | "))
            }
        }
    }

    func toggleRecording() {
        if recorder.isRecording {
            recorder.stop()
            recording = false
        } else {
            recorder.start()
            recording = true
        }
    }

    /// 关 → 服务器扩散 → 服务器微调 → 本机网络（有模型才有这一档）→ 关。
    func toggleNeural() {
        neuralMode = (neuralMode + 1) % (refiner == nil ? 3 : 4)
        neural.enabled = neuralMode == 1
        neuralTuned.enabled = neuralMode == 2
        refiner?.enabled = neuralMode == 3
        // 画面交给网络出的时候，全景补全用不上（背景整张都是网络画的），停掉：
        // 服务器的显存也装不下它和扩散模型同时跑，硬跑会把那边的服务挤崩。
        predictor?.paused = neuralMode != 0
    }

    func currentPose() -> PoseSample? {
        lock.lock()
        defer { lock.unlock() }
        return pose
    }

    /// 取走最新的相机小图（给人偶手里那台手机的屏幕用），取过就清空。
    func takeScreenImage() -> (width: Int, height: Int, rgba: [UInt8])? {
        lock.lock()
        defer { lock.unlock() }
        let s = screenImage
        screenImage = nil
        return s
    }

    var livePixelAngle: Float {
        lock.lock()
        defer { lock.unlock() }
        return pixelAngle
    }
}
