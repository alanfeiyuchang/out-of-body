import Foundation
import Metal
import UIKit
import simd

/// 「生成式画面」：把渲染出来的一帧发给 服务器上的单步扩散模型重画一遍，拿回来铺满屏幕。
///
/// 这是照着 FSD v15 Reality View 的思路做的近似：最后那张图由网络来画，所以没有洞、没有颗粒；
/// 代价是细节是网络编的，而且要连着 服务器，画面比真实动作慢零点几秒（每秒只有几帧）。
/// 地图渲染出来的那一帧（带人偶）是网络的输入，网络只负责「照着它重画」。
///
/// `-neural YES` 启动就开；`-neuralStrength 499` 改强度（越高改得越多、编得越多）；
/// `-neuralKeep 0.3` 是把上一帧输出掺进这一帧输入的比例（默认 0，见下面的说明）。
/// 服务端见 tools/neural/refine_server.py，经 predict_server.py 的 /refine 转发。
final class NeuralView {
    private let device: MTLDevice
    private let session: URLSession
    private let hosts: [String]
    private let lock = NSLock()
    private var workingHost: String?
    /// 有几张图正在路上。微调那档允许两张同时在路上（`-genParallel n` 改）：
    /// 一张来回 150~200 ms，其中 服务器真正在算的只有 115 ms，其余是传输和手机这头的工夫；
    /// 等上一张回来再发下一张，服务器有三四成时间闲着。错开半个来回再发一张，出图快到 服务器的上限（每秒 8 张多）。
    /// 每张图在路上的时间会长一点，靠预判机位和「图立在世界里」兜着。
    private var inFlight = 0
    private var lastWanted: CFTimeInterval = 0
    private var nextSeq = 0
    private var shownSeq = -1
    private let parallel: Int = {
        let v = UserDefaults.standard.integer(forKey: "genParallel")
        return v > 0 ? v : 2
    }()
    private var _texture: MTLTexture?
    /// 现在这张图是从哪个机位画的（世界 → 裁剪空间），把它贴回场景时要用。
    private var _viewProj = matrix_identity_float4x4
    /// 画这张图时镜头在哪、朝哪（相机到世界）。镜头转开太多时这张图就不能再用了。
    private var _camera = matrix_identity_float4x4
    private var _status = ""
    private var _enabled = false
    private var needsReset = true
    private var lastDone = CACurrentMediaTime()
    private var rate: Double = 0
    private var avgGap: Double = 0
    private var _lead: Double = 0
    private var lastRetry: CFTimeInterval = 0
    private var encodeMs: Double = 0
    private var meshMs: Double = 0
    private var tripMs: Double = 0
    private var decodeMs: Double = 0
    private var lastTimingLog: CFTimeInterval = 0
    /// 微调那档：画好的图连同形状立在世界里，留几张（见 GenKeyframe）。
    private var keys: [GenKeyframe] = []
    /// 刚到的图。渲染器每帧取走：先画进全景贴图（见 GenShell），再用 insertKey 立成一层皮。
    struct Fresh {
        let texture: MTLTexture
        let dist: MTLTexture
        let mesh: MTLBuffer
        let viewProj: simd_float4x4
        let camera: simd_float4x4
    }
    private var fresh: [Fresh] = []
    /// 渲染器把一张图处理好（内容换成全景贴图里累积的那份）之后，交回来立成一层皮。
    /// viewer：屏幕上现在的镜头。图存满了的时候，扔掉离它最远的。
    func insertKey(
        texture: MTLTexture, mesh: MTLBuffer, camera: simd_float4x4, viewer: simd_float4x4,
        now: CFTimeInterval
    ) {
        lock.lock()
        defer { lock.unlock() }
        guard _enabled else { return }
        // 机位差不多的旧图让位（留一小会儿垫在下面）；太旧的、太多的删掉。
        let fwd = -camera.columns.2.xyz, pos = camera.columns.3.xyz
        for i in keys.indices where keys[i].replacedAt == nil {
            let k = keys[i].camera
            if simd_dot(-k.columns.2.xyz, fwd) > 0.966, simd_distance(k.columns.3.xyz, pos) < 0.6 {
                keys[i].replacedAt = now
            }
        }
        // 留得久、留得多：镜头转回来或者转得太快新图跟不上时，垫在下面的是这些旧图，而不是实时点云画面。
        keys.removeAll { ($0.replacedAt.map { now - $0 > 0.4 } ?? false) || now - $0.born > 90 }
        if keys.count >= 32 {
            // 满了就扔掉离屏幕上的镜头最远的那张（转开的角度 + 挪开的距离）。
            let vf = -viewer.columns.2.xyz, vp = viewer.columns.3.xyz
            var worst = 0
            var worstOff: Float = -1
            for i in keys.indices {
                let k = keys[i].camera
                let off = acos(min(1, max(-1, simd_dot(-k.columns.2.xyz, vf))))
                    + simd_distance(k.columns.3.xyz, vp)
                if off > worstOff { worst = i; worstOff = off }
            }
            keys.remove(at: worst)
        }
        keys.append(GenKeyframe(texture: texture, vertices: mesh, camera: camera, born: now, replacedAt: nil))
    }

    func takeFresh() -> [Fresh] {
        lock.lock()
        defer { lock.unlock() }
        let out = fresh
        fresh = []
        return _enabled ? out : []
    }
    var keyframes: [GenKeyframe] {
        lock.lock()
        defer { lock.unlock() }
        return _enabled ? keys : []
    }
    /// 取景要往前预判多少秒：手指拖着转视角时，按「这么久之后镜头会在哪」去画，拿回来才对得上。
    var lead: Float {
        lock.lock()
        defer { lock.unlock() }
        return Float(min(0.45, max(0.1, _lead == 0 ? 0.25 : _lead)))
    }
    private let queue = DispatchQueue(label: "outofbody.neural", qos: .userInitiated)

    let strength: Int = {
        let v = UserDefaults.standard.integer(forKey: "neuralStrength")
        return v > 0 ? v : 499
    }()
    let keep: Float = {
        let d = UserDefaults.standard
        // 默认不掺：试过 0.3，镜头一动上一帧的东西就留下一串残影。要掺得先按深度把上一帧挪到新机位。
        return d.object(forKey: "neuralKeep") != nil ? d.float(forKey: "neuralKeep") : 0
    }()
    /// 微调模型那一档交过去的渲染图多大。模型是在 384×512 上训的；默认放大 1.5 倍到 576×768，画面清楚不少。
    /// 服务端一张的时间（4090，2026-11-23 给编解码器提速之后，含 JPEG 和 HTTP）：576×768 约 116 ms（提速前 190），
    /// 480×640 约 98 ms，384×512 约 90 ms。分辨率降下去省不了多少，所以还是用大图。
    /// `-tunedScale 1.25`、`-tunedScale 1` 换小图。
    static let tunedSize: (width: Int, height: Int) = {
        let k = UserDefaults.standard.double(forKey: "tunedScale")
        let s = k > 0 ? k : 1.5
        return (Int(384 * s) / 8 * 8, Int(512 * s) / 8 * 8)
    }()
    /// 发过去的图长边多少像素。越大越清楚也越慢。
    static let longSide: Int = {
        let v = UserDefaults.standard.integer(forKey: "neuralSize")
        return v > 0 ? v : 640
    }()

    /// 用哪个模型：false = 原版（输入是带人偶的整帧画面），
    /// true = 我们自己微调过的（输入是只有扫到的表面、没扫到的地方中灰的渲染图，人偶之后再画上去）。
    let tuned: Bool
    private var label: String { tuned ? "微调" : "生成" }

    init(device: MTLDevice, tuned: Bool = false) {
        self.device = device
        self.tuned = tuned
        _enabled = AppModel.startMode == (tuned ? 2 : 1)
        _status = "\(tuned ? "微调" : "生成")：关"
        if let h = UserDefaults.standard.string(forKey: "predictHost"), !h.isEmpty {
            hosts = [h]
        } else {
            hosts = ScenePredictor.defaultHosts
        }
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 10
        cfg.waitsForConnectivity = false
        session = URLSession(configuration: cfg)
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
            keys = []
            fresh = []
            needsReset = true
            _status = newValue ? "\(label)：连接中…" : "\(label)：关"
            lock.unlock()
        }
    }

    var status: String {
        lock.lock()
        defer { lock.unlock() }
        return _status
    }

    /// 最近一张重画好的图；还没有就是 nil，渲染器照常显示自己画的。
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

    /// 渲染器每帧问一次：现在能不能接新的一帧。上一帧还在路上就不接。
    func wantsFrame() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard _enabled, inFlight < (tuned ? parallel : 1) else { return false }
        let now = CACurrentMediaTime()
        if inFlight > 0, now - lastWanted < max(0.05, tripMs / 2000) { return false }
        inFlight += 1
        lastWanted = now
        return true
    }

    private func finish(_ text: String? = nil) {
        lock.lock()
        inFlight = max(0, inFlight - 1)
        if let text, _enabled { _status = text }
        lock.unlock()
    }

    /// 渲染器把一帧交过来：原版模型给的是刚画完的整帧（BGRA），微调模型给的是只有表面的渲染图（RGBA）。
    /// dist：每个像素离镜头多远（只有微调那档给），用来把画好的图立成一张有形状的网。
    func submit(
        pixels: [UInt8], dist: [Float]? = nil, floorY: Float? = nil, width: Int, height: Int,
        viewProj: simd_float4x4, camera: simd_float4x4
    ) {
        lock.lock()
        let seq = nextSeq
        nextSeq += 1
        lock.unlock()
        queue.async {
            let t0 = CACurrentMediaTime()
            guard let jpeg = NeuralView.encode(pixels, width, height, rgba: self.tuned) else {
                self.finish()
                return
            }
            let t1 = CACurrentMediaTime()
            defer {
                self.lock.lock()
                self.encodeMs = self.encodeMs * 0.8 + (t1 - t0) * 200
                self.meshMs = self.meshMs * 0.8 + (CACurrentMediaTime() - t1) * 200
                self.lock.unlock()
            }
            let mesh = dist.flatMap {
                GenMesh.build(
                    dist: $0, width: width, height: height, viewProj: viewProj, camera: camera,
                    floorY: floorY, device: self.device)
            }
            var distTex: MTLTexture?
            if let dist, dist.count >= width * height {
                let td = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .r32Float, width: width, height: height, mipmapped: false)
                td.usage = [.shaderRead]
                distTex = self.device.makeTexture(descriptor: td)
                distTex?.replace(
                    region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: dist,
                    bytesPerRow: width * 4)
            }
            self.send(
                jpeg, mesh: mesh, dist: distTex, seq: seq, viewProj: viewProj, camera: camera, hostIndex: 0)
        }
    }

    private static func encode(_ pixels: [UInt8], _ w: Int, _ h: Int, rgba: Bool) -> Data? {
        var src = pixels
        let space = CGColorSpaceCreateDeviceRGB()
        let srcInfo = rgba
            ? CGImageAlphaInfo.noneSkipLast.rawValue
            : CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue
        // 微调模型的输入本来就是按它训练时的大小渲的，不再缩放。
        let s = rgba ? 1 : Float(longSide) / Float(max(w, h))
        let nw = max(8, Int(Float(w) * s) / 8 * 8), nh = max(8, Int(Float(h) * s) / 8 * 8)
        guard
            let sctx = CGContext(
                data: &src, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                space: space, bitmapInfo: srcInfo),
            let image = sctx.makeImage(),
            let dctx = CGContext(
                data: nil, width: nw, height: nh, bitsPerComponent: 8, bytesPerRow: nw * 4,
                space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }
        dctx.interpolationQuality = .medium
        dctx.draw(image, in: CGRect(x: 0, y: 0, width: nw, height: nh))
        guard let small = dctx.makeImage() else { return nil }
        return UIImage(cgImage: small).jpegData(compressionQuality: rgba ? 0.92 : 0.8)
    }

    private func send(
        _ jpeg: Data, mesh: MTLBuffer?, dist: MTLTexture?, seq: Int, viewProj: simd_float4x4,
        camera: simd_float4x4, hostIndex: Int
    ) {
        lock.lock()
        var order = hosts
        if let w = workingHost, let i = order.firstIndex(of: w) {
            order.remove(at: i)
            order.insert(w, at: 0)
        }
        let known = workingHost != nil
        let reset = needsReset
        lock.unlock()
        guard hostIndex < order.count else {
            DebugLog.write("NEURAL 所有地址都连不上")
            // 连不上就歇两秒再试，别一帧帧地空转。
            queue.asyncAfter(deadline: .now() + 2) { self.finish("\(self.label)：连不上服务") }
            return
        }
        let host = order[hostIndex]
        let base = "http://\(ScenePredictor.authority(host))"
        func post() {
            let query = tuned ? "m=ft" : "s=\(strength)&k=\(keep)" + (reset ? "&reset=1" : "")
            guard let url = URL(string: "\(base)/refine?\(query)") else { return finish() }
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.setValue("image/jpeg", forHTTPHeaderField: "Content-Type")
            let t0 = CACurrentMediaTime()
            session.uploadTask(with: req, from: jpeg) { data, response, error in
                let code = (response as? HTTPURLResponse)?.statusCode ?? 0
                let arrived = CACurrentMediaTime()
                guard error == nil, code == 200, let data, let tex = self.makeTexture(data) else {
                    let why = error?.localizedDescription
                        ?? "HTTP \(code) \(data.flatMap { String(data: $0.prefix(120), encoding: .utf8) } ?? "")"
                    DebugLog.write("NEURAL \(host) 失败：\(why)")
                    self.lock.lock()
                    self.workingHost = nil
                    self.lock.unlock()
                    self.queue.asyncAfter(deadline: .now() + 1) { self.finish("\(self.label)：服务出错") }
                    return
                }
                let now = CACurrentMediaTime()
                self.lock.lock()
                // 两张同时在路上时，后发的偶尔会先到：先发的那张回来就不用了。
                let stale = seq < self.shownSeq
                if !stale {
                    self.shownSeq = seq
                    self._texture = tex
                    self._viewProj = viewProj
                    self._camera = camera
                }
                if let dist, let mesh, !stale {
                    self.fresh.append(
                        Fresh(texture: tex, dist: dist, mesh: mesh, viewProj: viewProj, camera: camera))
                    if self.fresh.count > 4 { self.fresh.removeFirst() }
                }
                self.needsReset = false
                let gap = now - self.lastDone
                self.lastDone = now
                // 平均的是间隔，不是间隔的倒数：两张同时在路上时会有前后脚到的，按倒数平均会算出每秒三十张。
                self.avgGap = self.avgGap == 0 ? gap : self.avgGap * 0.85 + gap * 0.15
                self.rate = 1 / max(self.avgGap, 0.01)
                let fps = self.rate
                // 这张图从取景到显示完（下一张来之前）平均隔了多久：一张的来回时间，加半个出图间隔。
                let lag = (now - t0) + 0.5 / max(fps, 1)
                self._lead = self._lead == 0 ? lag : self._lead * 0.8 + lag * 0.2
                // 每 5 秒记一行各环节用时，真机日志里看时间花在哪。
                self.tripMs = self.tripMs * 0.8 + (arrived - t0) * 200
                self.decodeMs = self.decodeMs * 0.8 + (now - arrived) * 200
                if now - self.lastTimingLog > 5 {
                    self.lastTimingLog = now
                    DebugLog.write(String(
                        format: "NEURAL 用时 %@：压图 %.0f ms，建网 %.0f ms，来回 %.0f ms（发 %d KB 收 %d KB），解图 %.0f ms，每秒 %.1f 张，留着 %d 张",
                        host, self.encodeMs, self.meshMs, self.tripMs, jpeg.count / 1024, data.count / 1024,
                        self.decodeMs, fps, self.keys.count))
                }
                self.lock.unlock()
                self.finish(String(
                    format: "%@：%.0f ms 一张 · 每秒 %.1f 张", self.label, (now - t0) * 1000, fps))
            }.resume()
        }
        if known {
            retryPreferred(current: host)
            return post()
        }
        guard let health = URL(string: "\(base)/health") else { return finish() }
        var probe = URLRequest(url: health)
        probe.timeoutInterval = 3
        session.dataTask(with: probe) { _, response, error in
            guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                DebugLog.write("NEURAL \(host) 不通：\(error?.localizedDescription ?? "无响应")")
                self.send(
                    jpeg, mesh: mesh, dist: dist, seq: seq, viewProj: viewProj, camera: camera,
                    hostIndex: hostIndex + 1)
                return
            }
            self.lock.lock()
            self.workingHost = host
            self.lock.unlock()
            DebugLog.write("NEURAL 用 \(host)")
            post()
        }.resume()
    }

    /// 现在用的不是首选地址（直连 服务器）时，每 12 秒在旁边再试一次首选的，通了就换回去。
    /// 2026-11-23 真机日志：启动时直连 服务器两次都超时（原因没查清，服务器是无线网卡且开着省电），
    /// 之后整场都走 Mac 上的 ssh 转发，一张 203 ms；直连的时候只比服务端多 20 ms 左右。
    private func retryPreferred(current: String) {
        guard let first = hosts.first, first != current else { return }
        let now = CACurrentMediaTime()
        lock.lock()
        let due = now - lastRetry > 12
        if due { lastRetry = now }
        lock.unlock()
        guard due, let url = URL(string: "http://\(ScenePredictor.authority(first))/health") else { return }
        var probe = URLRequest(url: url)
        probe.timeoutInterval = 5
        let t0 = CACurrentMediaTime()
        session.dataTask(with: probe) { _, response, error in
            guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                DebugLog.write("NEURAL 再试 \(first)：还是不通（\(error?.localizedDescription ?? "无响应")）")
                return
            }
            self.lock.lock()
            self.workingHost = first
            self.lock.unlock()
            DebugLog.write(String(format: "NEURAL 换回 %@（应答 %.0f ms）", first, (CACurrentMediaTime() - t0) * 1000))
        }.resume()
    }

    private func makeTexture(_ jpeg: Data) -> MTLTexture? {
        guard let image = UIImage(data: jpeg)?.cgImage else { return nil }
        let w = image.width, h = image.height
        var rgba = [UInt8](repeating: 255, count: w * h * 4)
        guard
            let ctx = CGContext(
                data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        let d = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
        d.usage = [.shaderRead]
        guard let tex = device.makeTexture(descriptor: d) else { return nil }
        tex.replace(
            region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: rgba,
            bytesPerRow: w * 4)
        return tex
    }
}
