import CoreML
import Foundation
import Metal
import QuartzCore
import simd

/// 手机上自己跑的「粗渲染 → 干净画面」小网络（tools/neural 里训练，CoreML）。
///
/// 它是 服务器上那个微调过的扩散模型的「学生」：训练时照着老师的输出学（tools/neural/train_student.py），
/// 所以用法也和老师那一档一样——交一张当前机位的地图渲染（颜色 + 有没有东西 + 逆距离，384×512），
/// 拿回来的图整张当背景，按画它时的机位跟着镜头重新投影，人偶画在上面。
/// 好处是不联网、出图快；它比老师小几百倍，细节会差一些。
/// 模型文件是 Models/Refiner.mlpackage（不进 git）；没有这个文件时这一档不出现。`-refine YES` 启动就开。
final class LocalRefiner {
    static let width = 384
    static let height = 512

    private let device: MTLDevice
    private let model: MLModel
    private let lock = NSLock()
    private let queue = DispatchQueue(label: "outofbody.refiner", qos: .userInitiated)
    private var busy = false
    private var _enabled = UserDefaults.standard.bool(forKey: "refine")
    private var _texture: MTLTexture?
    private var _viewProj = matrix_identity_float4x4
    private var _camera = matrix_identity_float4x4
    private var _status = "本机网络：等第一帧"
    private var averageMs: Double = 0

    init?(device: MTLDevice) {
        guard let url = Bundle.main.url(forResource: "Refiner", withExtension: "mlmodelc")
        else { return nil }
        let cfg = MLModelConfiguration()
        #if targetEnvironment(simulator)
            cfg.computeUnits = .cpuOnly
        #else
            cfg.computeUnits = .all
        #endif
        guard let m = try? MLModel(contentsOf: url, configuration: cfg) else { return nil }
        model = m
        self.device = device
    }

    var enabled: Bool {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _enabled
        }
        set {
            lock.lock()
            _enabled = newValue
            _texture = nil
            lock.unlock()
        }
    }

    var status: String {
        lock.lock()
        defer { lock.unlock() }
        return _status
    }

    var texture: MTLTexture? {
        lock.lock()
        defer { lock.unlock() }
        return _enabled ? _texture : nil
    }

    /// 最近一张图和画它时的机位。
    var frame: (texture: MTLTexture, viewProj: simd_float4x4, camera: simd_float4x4)? {
        lock.lock()
        defer { lock.unlock() }
        guard _enabled, let t = _texture else { return nil }
        return (t, _viewProj, _camera)
    }

    func wantsFrame() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard _enabled, !busy else { return false }
        busy = true
        return true
    }

    /// 渲染器把地图渲出来的颜色（RGBA，没东西的地方随便）和距离（0 = 没东西）交过来。
    func submit(rgba: [UInt8], dist: [Float], viewProj: simd_float4x4, camera: simd_float4x4) {
        queue.async {
            let t0 = CACurrentMediaTime()
            let tex = self.run(rgba: rgba, dist: dist)
            let ms = (CACurrentMediaTime() - t0) * 1000
            self.lock.lock()
            if let tex {
                self._texture = tex
                self._viewProj = viewProj
                self._camera = camera
            }
            self.averageMs = self.averageMs == 0 ? ms : self.averageMs * 0.8 + ms * 0.2
            self._status = tex == nil
                ? "本机网络：推理失败" : String(format: "本机网络：%.0f ms 一张", self.averageMs)
            self.busy = false
            self.lock.unlock()
        }
    }

    private var logged = false
    private func logOnce(_ text: String) {
        if !logged { DebugLog.write(text) }
        logged = true
    }

    private func run(rgba: [UInt8], dist: [Float]) -> MTLTexture? {
        let w = LocalRefiner.width, h = LocalRefiner.height
        let n = w * h
        guard rgba.count == n * 4, dist.count == n,
            let input = try? MLMultiArray(
                shape: [1, 5, NSNumber(value: h), NSNumber(value: w)], dataType: .float32)
        else { return nil }
        // 跟 tools/neural/data.py 的 load_pair 保持一致。
        let p = input.dataPointer.bindMemory(to: Float.self, capacity: n * 5)
        for i in 0..<n {
            let d = dist[i]
            let known: Float = d > 0 ? 1 : 0
            p[i] = Float(rgba[i * 4]) / 255 * known
            p[n + i] = Float(rgba[i * 4 + 1]) / 255 * known
            p[2 * n + i] = Float(rgba[i * 4 + 2]) / 255 * known
            p[3 * n + i] = known
            p[4 * n + i] = d > 0 ? 1 / max(d, 0.3) : 0
        }
        let out: MLMultiArray
        do {
            let provider = try MLDictionaryFeatureProvider(
                dictionary: ["x": MLFeatureValue(multiArray: input)])
            let result = try model.prediction(from: provider)
            guard let y = result.featureValue(for: "y")?.multiArrayValue,
                y.dataType == .float32 || y.dataType == .float16, y.count == n * 3
            else {
                let y = result.featureValue(for: "y")?.multiArrayValue
                logOnce("REFINER 输出对不上：\(String(describing: y?.shape)) 类型 \(y?.dataType.rawValue ?? -1)")
                return nil
            }
            out = y
        } catch {
            logOnce("REFINER 推理出错：\(error)")
            return nil
        }
        let s = out.strides.map { $0.intValue }  // [批, 通道, 行, 列]
        var bytes = [UInt8](repeating: 255, count: n * 4)
        // 半精度的模型输出也是半精度。
        let half = out.dataType == .float16
        let o32 = out.dataPointer.assumingMemoryBound(to: Float.self)
        let o16 = out.dataPointer.assumingMemoryBound(to: Float16.self)
        for y in 0..<h {
            for x in 0..<w {
                let base = y * s[2] + x * s[3]
                let i = (y * w + x) * 4
                for c in 0..<3 {
                    let k = base + c * s[1]
                    let v = half ? Float(o16[k]) : o32[k]
                    bytes[i + c] = UInt8(max(0, min(255, v * 255 + 0.5)))
                }
            }
        }
        let d = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
        d.usage = [.shaderRead]
        guard let tex = device.makeTexture(descriptor: d) else { return nil }
        tex.replace(
            region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: bytes,
            bytesPerRow: w * 4)
        return tex
    }
}
