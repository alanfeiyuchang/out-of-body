import Foundation
import Metal
import UIKit
import simd

/// 生成式的「未扫区域预测」。
///
/// 每隔一阵把地图渲成一张 360° 全景图，连同「哪些像素还没扫到」的遮罩，发给局域网里一台有显卡的电脑。
/// 那边的图像模型只在缺的地方生成、已知部分原样保留，把补全的全景图传回来。
///
/// 回来的图在手机上再过一遍单目深度模型，对齐到已知部分的真实距离，
/// 于是缺的地方不只是一张贴在远处的画，而是一层有远近的「预测表面」（SplatMap.setPredicted），
/// 换个角度看有正确的遮挡和视差。真实数据一到，对应的预测表面就被删掉。
/// 深度模型看不到的方向（头顶、脚下）仍然用全景图当背景。
///
/// 服务端是 tools/predict_server.py，跑在 服务器上。连不上时什么都不做，app 照常用颜色外推的背景。
final class ScenePredictor {
    /// 依次尝试的服务地址，没写端口的用默认端口。`-predictHost 1.2.3.4` 可以覆盖。
    /// 地址不写在代码里：Config/Local.xcconfig 的 PREDICT_HOSTS（逗号隔开）经 Info.plist 带进来。
    /// 留空就是不连服务器，app 照常用自己渲染的画面。常见的填法：
    /// 1. 服务器的局域网地址：要它的防火墙放行 8765 才通。
    /// 2. 另一台电脑上的 SSH 中转（`ssh -N -L 0.0.0.0:8766:127.0.0.1:8765 <服务器>`）：防火墙没放行时的临时办法，
    ///    那台电脑睡眠或者换网络后要重开。
    /// 3. 服务器的 Tailscale 地址：要手机上开着 Tailscale。
    static let defaultHosts: [String] = {
        let raw = Bundle.main.object(forInfoDictionaryKey: "PredictHosts") as? String ?? ""
        return raw.split(whereSeparator: { $0 == "," || $0 == " " }).map(String.init)
    }()
    /// `-predictPort 8766` 可以覆盖（比如经过另一台电脑中转时）。
    static let port: Int = {
        let p = UserDefaults.standard.integer(forKey: "predictPort")
        return p > 0 ? p : 8765
    }()

    let texture: MTLTexture
    /// 房间的轮廓：从房间中间往每个水平方向看，墙有多远（128 个方向，一行贴图）。
    /// 没扫到的地方要有个东西接住视线才能上色。以前用一个球，但房间是方的：
    /// 球面上的「墙」比真墙远一截，换个角度看，猜出来的墙和扫到的墙就接不上。
    /// 按轮廓竖起来的墙和真墙在同一个位置，从哪看都接得上。
    let roomTexture: MTLTexture
    static let roomBins = 128
    private var roomProfile = [Float](repeating: 0, count: roomBins)  // 0 = 这个方向还不知道
    private var roomAnchor: Vec3?
    private var _room = SIMD4<Float>(0, 0, 0, 0)  // 中心 x、z，天花板高度，平均半径（0 = 还没有）
    private var lastPanoTime = 0.0
    /// 由 Renderer 接上：以某个位置为中心渲一张全景图。返回 false 表示上一张还没渲完。
    var panorama: ((Vec3, @escaping (SplatMap.Panorama) -> Void) -> Bool)?
    private let commandQueue: MTLCommandQueue?
    /// `-lift YES`：把补全的图再抬成有深度的预测表面（实验性，默认关）。
    /// 在合成房间里试下来，对齐数字没问题，但生成的画面本身透视不严格，
    /// 抬出来常是斜穿房间的大片，比一张平的背景更扎眼。等有真实房间的录像再调。
    private let liftEnabled = UserDefaults.standard.bool(forKey: "lift")
    private var _radius: Float = 4.5
    private let map: SplatMap
    private let hosts: [String]
    private let enabled = !UserDefaults.standard.bool(forKey: "noPredict")
    /// 暂停往服务端发补全请求。服务器的显存装不下两个模型同时跑：
    /// 「生成：服务器扩散」开着的时候，这边的全景补全先停。
    var paused = false
    private let session: URLSession

    private let lock = NSLock()
    private var busy = false
    private var _status = "预测：等数据"
    private var _center = Vec3(0, 0, 0)
    private var _valid = false
    private var _appliedAt = 0.0
    private var lastRequestTime = 0.0
    private var lastCoverage: Float = 0
    private var lastCenter: Vec3?
    private var workingHost: String?
    private var requestCount = 0
    private var _floorY: Float?

    init?(device: MTLDevice, map: SplatMap) {
        self.map = map
        // 带 mip：有预测表面的方向上，背景只取很糊的一层（见 apply）。
        let d = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 1536, height: 768, mipmapped: true)
        d.usage = [.shaderRead]
        d.storageMode = .shared
        let rd = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float, width: ScenePredictor.roomBins, height: 1, mipmapped: false)
        rd.usage = [.shaderRead]
        rd.storageMode = .shared
        guard let t = device.makeTexture(descriptor: d), let rt = device.makeTexture(descriptor: rd)
        else { return nil }
        texture = t
        roomTexture = rt
        commandQueue = device.makeCommandQueue()
        if let h = UserDefaults.standard.string(forKey: "predictHost"), !h.isEmpty {
            hosts = [h]
        } else {
            hosts = ScenePredictor.defaultHosts
        }
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 120
        cfg.timeoutIntervalForResource = 180
        cfg.waitsForConnectivity = false
        session = URLSession(configuration: cfg)
    }

    var status: String {
        lock.lock()
        defer { lock.unlock() }
        // 画面交给网络出的时候这边是停着的，别显示成「等数据」让人以为卡住了。
        if paused { return "预测：生成模式下不用" }
        return _status
    }

    /// 渲染器每帧取：预测图是以哪个位置为中心渲的、现在该用多大的强度（刚到时一秒内淡入），
    /// 以及背景那个球该多大（大致是房间的墙有多远）。
    var current: (center: Vec3, strength: Float, radius: Float) {
        lock.lock()
        defer { lock.unlock() }
        guard _valid else { return (_center, 0, _radius) }
        return (_center, Float(min(1, (CACurrentMediaTime() - _appliedAt) / 1.0)), _radius)
    }

    /// 渲染器每帧取：房间轮廓的中心（x、z）、天花板高度、平均半径（0 = 还没有轮廓，退回用球）。
    var room: SIMD4<Float> {
        lock.lock()
        defer { lock.unlock() }
        return _room
    }

    /// 从一张全景图（带每个像素的距离）里量出房间的轮廓。
    private func updateRoom(_ pano: SplatMap.Panorama, floorY: Float) {
        let n = ScenePredictor.roomBins
        let w = pano.width, h = pano.height

        // 天花板有多高：扫到的东西里最高的那一批（取 97 分位，留一点余量给飞点）。
        // 手机一般不会仰着拍，天花板本身多半没扫到，但墙扫到多高，天花板至少就有多高。
        var tops: [Float] = []
        for y in stride(from: 0, to: h / 2, by: 3) {
            let cosp = cos((Float(y) + 0.5) / Float(h) * .pi)
            for x in stride(from: 0, to: w, by: 3) where pano.distance[y * w + x] > 0.3 {
                tops.append(pano.center.y + cosp * pano.distance[y * w + x])
            }
        }
        tops.sort()
        var ceilY = floorY + 2.5
        if tops.count > 200 {
            ceilY = min(floorY + 4.5, max(floorY + 2.3, tops[tops.count * 97 / 100]))
        }

        // 每个方向上，离地半米到两米二之间的像素（墙那一段，不含地面和天花板）离中心的水平距离，
        // 取偏远的一档：近的是家具，远的才是墙。
        let lo = floorY + 0.5, hi = min(ceilY - 0.2, floorY + 2.2)
        var bucket = [[Float]](repeating: [], count: n)
        for y in stride(from: 0, to: h, by: 2) {
            let phi = (Float(y) + 0.5) / Float(h) * .pi
            let sinp = sin(phi), cosp = cos(phi)
            for x in stride(from: 0, to: w, by: 2) {
                let dist = pano.distance[y * w + x]
                if dist < 0.3 { continue }
                let height = pano.center.y + cosp * dist
                if height < lo || height > hi { continue }
                bucket[x * n / w].append(dist * sinp)
            }
        }

        // 轮廓以一个固定的点为中心，不跟着人走：人挪一步整圈墙就跟着晃的话很难看。
        // 人离它 2.5 米以上才换一个中心重新量。
        var anchor = roomAnchor ?? pano.center
        if simd_distance(Vec3(anchor.x, 0, anchor.z), Vec3(pano.center.x, 0, pano.center.z)) > 2.5 {
            anchor = pano.center
            roomProfile = [Float](repeating: 0, count: n)
        }
        roomAnchor = anchor
        var fresh = [Float](repeating: 0, count: n)
        for b in 0..<n where bucket[b].count >= 15 {
            bucket[b].sort()
            let r = bucket[b][bucket[b].count * 85 / 100]
            let th = ((Float(b) + 0.5) / Float(n) - 0.5) * 2 * .pi
            let px = pano.center.x + r * cos(th) - anchor.x
            let pz = pano.center.z + r * sin(th) - anchor.z
            let nb = min(n - 1, max(0, Int((atan2(pz, px) / (2 * .pi) + 0.5) * Float(n))))
            fresh[nb] = max(fresh[nb], (px * px + pz * pz).squareRoot())
        }
        for b in 0..<n where fresh[b] > 0 {
            roomProfile[b] = roomProfile[b] > 0 ? (roomProfile[b] + fresh[b]) * 0.5 : fresh[b]
        }

        // 不知道的方向从两边知道的方向之间连过去，再整圈抹平一点。
        let known = (0..<n).filter { roomProfile[$0] > 0 }
        guard known.count >= 6 else { return }
        var filled = roomProfile
        for (k, a) in known.enumerated() {
            let b = known[(k + 1) % known.count]
            let gap = (b - a + n) % n
            if gap <= 1 { continue }
            for j in 1..<gap {
                let t = Float(j) / Float(gap)
                filled[(a + j) % n] = roomProfile[a] * (1 - t) + roomProfile[b] * t
            }
        }
        for _ in 0..<2 {
            var out = filled
            for b in 0..<n {
                out[b] = (filled[(b + n - 1) % n] + 2 * filled[b] + filled[(b + 1) % n]) / 4
            }
            filled = out
        }
        // 往外推一点点：扫到的墙要盖在它前面，不能跟它抢。
        let half = filled.map { Float16(min(9, max(1, $0 * 1.04))) }
        half.withUnsafeBytes { raw in
            roomTexture.replace(
                region: MTLRegionMake2D(0, 0, n, 1), mipmapLevel: 0, withBytes: raw.baseAddress!,
                bytesPerRow: n * 2)
        }
        let average = filled.reduce(0, +) / Float(n)
        lock.lock()
        _room = SIMD4(anchor.x, anchor.z, ceilY, average)
        lock.unlock()
    }

    /// 已知像素里，水平方向附近那一圈的距离取偏远的一档（75 分位），当作「墙大概有多远」。
    private static func roomRadius(_ pano: SplatMap.Panorama) -> Float {
        var d: [Float] = []
        let w = pano.width, h = pano.height
        for y in stride(from: h / 3, to: h * 2 / 3, by: 4) {
            for x in stride(from: 0, to: w, by: 4) where pano.distance[y * w + x] > 0.3 {
                d.append(pano.distance[y * w + x])
            }
        }
        guard d.count > 50 else { return 4.5 }
        d.sort()
        return min(8, max(2.2, d[d.count * 3 / 4]))
    }

    private func setStatus(_ s: String) {
        lock.lock()
        _status = s
        lock.unlock()
    }

    /// 大约每秒调一次。每 3 秒渲一张全景图量房间的轮廓；要不要顺便发一次补全请求另算。
    /// `-noPredict YES` 只是不发请求，轮廓照量。
    func tick(userPosition: Vec3, floorY: Float?) {
        let now = CACurrentMediaTime()
        lock.lock()
        _floorY = floorY
        let skip = busy || now - lastPanoTime < 3
        if !skip {
            busy = true
            lastPanoTime = now
        }
        let maySend = enabled && !paused && now - lastRequestTime >= 6
        lock.unlock()
        if skip { return }

        let started = panorama?(userPosition) { pano in
            if pano.coverage > 0.03 {
                self.updateRoom(pano, floorY: floorY ?? pano.center.y - 1.3)
            }
            // 扫到的太少，模型没东西可参考；几乎扫全了，就没什么好补的。
            let moved = self.lastCenter.map { simd_distance($0, pano.center) } ?? .infinity
            let grew = pano.coverage - self.lastCoverage
            let worth =
                maySend && pano.coverage > 0.06 && pano.coverage < 0.97
                && (moved > 1.0 || grew > 0.04 || self.lastCenter == nil)
            guard worth, let png = ScenePredictor.encodePNG(pano) else {
                self.lock.lock()
                self.busy = false
                self.lock.unlock()
                return
            }
            self.lastRequestTime = CACurrentMediaTime()
            self.lastCoverage = pano.coverage
            self.lastCenter = pano.center
            self.requestCount += 1
            ScenePredictor.dump(png, name: "pano-in.png")
            self.setStatus(String(format: "预测：请求中（已扫 %.0f%%）", pano.coverage * 100))
            self.send(png, pano: pano, hostIndex: 0)
        }
        if started != true {
            lock.lock()
            busy = false
            lock.unlock()
        }
    }

    /// 先用一个 3 秒超时的小请求找出哪个地址是通的，再把图发过去。
    /// 直接发大请求的话，地址不通时要等很久才报错。
    private func send(_ png: Data, pano: SplatMap.Panorama, hostIndex: Int) {
        var order = hosts
        if let w = workingHost, let i = order.firstIndex(of: w) {
            order.remove(at: i)
            order.insert(w, at: 0)
        }
        guard hostIndex < order.count,
            let health = URL(string: "http://\(ScenePredictor.authority(order[hostIndex]))/health"),
            let url = URL(string: "http://\(ScenePredictor.authority(order[hostIndex]))/complete")
        else {
            setStatus("预测：连不上服务")
            DebugLog.write("PREDICT 所有地址都连不上")
            finish()
            return
        }
        let host = order[hostIndex]
        var probe = URLRequest(url: health)
        probe.timeoutInterval = 3
        session.dataTask(with: probe) { _, response, error in
            guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                DebugLog.write("PREDICT \(host) 不通：\(error?.localizedDescription ?? "无响应")")
                self.workingHost = nil
                self.send(png, pano: pano, hostIndex: hostIndex + 1)
                return
            }
            self.workingHost = host
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.setValue("image/png", forHTTPHeaderField: "Content-Type")
            req.timeoutInterval = 120
            let t0 = CACurrentMediaTime()
            self.session.uploadTask(with: req, from: png) { data, response, error in
                let code = (response as? HTTPURLResponse)?.statusCode ?? 0
                guard error == nil, code == 200, let data,
                    let image = UIImage(data: data)?.cgImage
                else {
                    let why = error?.localizedDescription
                        ?? "HTTP \(code) \(data.flatMap { String(data: $0.prefix(200), encoding: .utf8) } ?? "")"
                    DebugLog.write("PREDICT \(host) 补全失败：\(why)")
                    self.setStatus("预测：服务出错")
                    self.finish()
                    return
                }
                ScenePredictor.dump(data, name: "pano-out.png")
                let secs = CACurrentMediaTime() - t0
                let lifted = self.apply(image, pano: pano)
                self.setStatus(
                    lifted > 0
                        ? String(format: "预测：已更新（%.0f 秒，立体 %dk 点）", secs, lifted / 1000)
                        : String(format: "预测：已更新（用了 %.0f 秒）", secs))
                DebugLog.write(String(
                    format: "PREDICT %@ 成功，%.1f 秒，抬出 %d 个点，共 %.1f 秒", host, secs,
                    lifted, CACurrentMediaTime() - t0))
                self.finish()
            }.resume()
        }.resume()
    }

    static func authority(_ host: String) -> String {
        host.contains(":") ? host : "\(host):\(port)"
    }

    private func finish() {
        lock.lock()
        busy = false
        lastRequestTime = CACurrentMediaTime()
        lock.unlock()
    }

    /// 把模型补全的图放进贴图，并抬出预测表面。返回预测点的数量。
    ///
    /// 贴图的 alpha：0 = 原来就扫到了（不用预测）；约 0.5 = 缺、但有预测表面（背景只用很糊的一层，
    /// 免得从侧面看时同一件家具在表面上一份、远处背景上又一份）；1 = 缺、而且没有预测表面（背景用清楚的）。
    private func apply(_ image: CGImage, pano: SplatMap.Panorama) -> Int {
        let w = pano.width, h = pano.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        guard
            let ctx = CGContext(
                data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return 0 }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))

        let gw = w / 4, gh = h / 4
        var hasGeometry = [Bool](repeating: false, count: gw * gh)
        var points: [SplatMap.Splat] = []
        if liftEnabled {
            points = lift(bytes, pano: pano, coarseWidth: gw, coarseHeight: gh, marked: &hasGeometry)
        }
        map.setPredicted(points)
        if UserDefaults.standard.bool(forKey: "evalSynthetic"), !points.isEmpty {
            // 合成场景里知道真实表面在哪：给预测点打个分。
            let e = points.compactMap { SyntheticSource.distanceToTruth(Vec3($0.x, $0.y, $0.z)) }
                .sorted()
            if !e.isEmpty {
                DebugLog.write(String(
                    format: "LIFT 预测点 %d 个，到真实表面：中位 %.1f cm，90 分位 %.1f cm",
                    e.count, e[e.count / 2] * 100, e[e.count * 9 / 10] * 100))
            }
        }

        // 缺失遮罩糊几个像素，边界上预测和真实数据平滑过渡。
        var mask = [Float](repeating: 0, count: w * h)
        for i in 0..<w * h { mask[i] = pano.rgba[i * 4 + 3] == 0 ? 1 : 0 }
        for _ in 0..<3 {
            var out = mask
            for y in 0..<h {
                for x in 0..<w {
                    var s: Float = 0
                    for dy in -1...1 {
                        let yy = min(h - 1, max(0, y + dy))
                        for dx in -1...1 { s += mask[yy * w + (x + dx + w) % w] }
                    }
                    out[y * w + x] = s / 9
                }
            }
            mask = out
        }
        for y in 0..<h {
            for x in 0..<w {
                let i = y * w + x
                let level: Float = hasGeometry[(y / 4) * gw + x / 4] ? 0.5 : 1
                bytes[i * 4 + 3] = UInt8(max(0, min(255, mask[i] * level * 255)))
            }
        }

        bytes.withUnsafeBytes { raw in
            texture.replace(
                region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: raw.baseAddress!,
                bytesPerRow: w * 4)
        }
        if let cmd = commandQueue?.makeCommandBuffer(), let blit = cmd.makeBlitCommandEncoder() {
            blit.generateMipmaps(for: texture)
            blit.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        lock.lock()
        _center = pano.center
        _radius = ScenePredictor.roomRadius(pano)
        _valid = true
        _appliedAt = CACurrentMediaTime()
        lock.unlock()
        return points.count
    }

    /// 补全的全景图 → 预测表面。
    ///
    /// 朝四个水平方向各截一张普通视角的图（正着的，深度模型最拿手），跑单目深度，
    /// 再用已知像素的真实距离把它对齐成米。哪个方向的已知像素多就先做哪个；
    /// 做完的方向又给相邻方向（两张图有一条重叠带）当对照，这样全是预测的方向也能接上尺度。
    private func lift(
        _ rgba: [UInt8], pano: SplatMap.Panorama, coarseWidth gw: Int, coarseHeight gh: Int,
        marked: inout [Bool]
    ) -> [SplatMap.Splat] {
        guard let estimator = MonoDepthEstimator.shared else { return [] }
        let w = pano.width, h = pano.height
        let vw = 518, vh = 392
        let focal = Float(vw) / 2 / tan(50 * Float.pi / 180)  // 水平视场 100°
        let up = Vec3(0, 1, 0)

        struct View {
            var forward: Vec3
            var right: Vec3
            var done = false
        }
        var views = (0..<4).map { k -> View in
            let yaw = Float(k) * .pi / 2
            let f = Vec3(cos(yaw), 0, sin(yaw))
            return View(forward: f, right: simd_normalize(simd_cross(f, up)))
        }
        // 粗网格上记「这个方向已经抬出来的距离」，给后面的方向当对照。
        lock.lock()
        let floorY = _floorY
        lock.unlock()
        var liftedDistance = [Float](repeating: 0, count: gw * gh)
        var claimed = [UInt8](repeating: 0, count: gw * gh)
        var out: [SplatMap.Splat] = []

        func panoIndex(_ d: Vec3) -> (fine: Int, coarse: Int) {
            let p = SplatMap.panoPixel(d, width: w, height: h)
            let x = min(w - 1, max(0, Int(p.x))), y = min(h - 1, max(0, Int(p.y)))
            return (y * w + x, (y / 4) * gw + x / 4)
        }

        for _ in 0..<views.count {
            // 逆深度图是模型输出再 2x2 平均：259×196，焦距减半。
            let dw = vw / 2, dh = vh / 2
            let df = focal / 2
            func ray(_ v: View, _ px: Int, _ py: Int) -> Vec3 {
                v.forward + v.right * ((Float(px) + 0.5 - Float(dw) / 2) / df)
                    + up * (-(Float(py) + 0.5 - Float(dh) / 2) / df)
            }
            func samples(_ v: View) -> [DepthAlign.Sample] {
                var s: [DepthAlign.Sample] = []
                for py in stride(from: 2, to: dh - 2, by: 2) {
                    for px in stride(from: 2, to: dw - 2, by: 2) {
                        let d = ray(v, px, py)
                        let i = panoIndex(d)
                        let radial = pano.distance[i.fine] > 0
                            ? pano.distance[i.fine] : liftedDistance[i.coarse]
                        if radial < 0.3 { continue }
                        // 沿这张图光轴的深度 = 径向距离 × cos(夹角)，而方向的轴向分量恒为 1。
                        s.append(DepthAlign.Sample(u: px, v: py, z: radial / simd_length(d)))
                    }
                }
                if s.count > 900 {
                    let keep = s.count / 900 + 1
                    s = s.enumerated().filter { $0.offset % keep == 0 }.map { $0.element }
                }
                return s
            }

            var best = -1
            var bestSamples: [DepthAlign.Sample] = []
            for (k, v) in views.enumerated() where !v.done {
                let s = samples(v)
                if s.count > bestSamples.count {
                    best = k
                    bestSamples = s
                }
            }
            guard best >= 0, bestSamples.count >= 40 else { break }
            views[best].done = true
            let view = views[best]

            // 从全景图里截出这个方向的普通视角图。
            var image = [UInt8](repeating: 255, count: vw * vh * 4)
            for y in 0..<vh {
                for x in 0..<vw {
                    let d = view.forward + view.right * ((Float(x) + 0.5 - Float(vw) / 2) / focal)
                        + up * (-(Float(y) + 0.5 - Float(vh) / 2) / focal)
                    let j = panoIndex(d).fine * 4
                    let i = (y * vw + x) * 4
                    image[i] = rgba[j]
                    image[i + 1] = rgba[j + 1]
                    image[i + 2] = rgba[j + 2]
                }
            }
            guard let pb = MonoDepthEstimator.pixelBuffer(rgba: image, width: vw, height: vh),
                let result = estimator.run(pb), result.width == dw, result.height == dh
            else { continue }
            let fitted = DepthAlign.fit(disparity: result.values, width: dw, samples: bestSamples)
            DebugLog.write(String(
                format: "LIFT 方向 %d：对照点 %d，内点 %d，误差 %.1f%%", best, bestSamples.count,
                fitted?.inliers ?? 0, (fitted?.rms ?? 0) * 100))
            guard let fit = fitted, fit.inliers >= 30, fit.rms < 0.09 else { continue }
            var depth = DepthAlign.metricDepth(disparity: result.values, fit: fit, maxRange: 9)
            DepthAlign.localCorrection(
                &depth, width: dw, height: dh, disparity: result.values, fit: fit,
                samples: bestSamples)
            DepthAlign.removeEdges(&depth, width: dw, height: dh)

            let tag = UInt8(best + 1)
            for py in 0..<dh - 1 {
                for px in 0..<dw - 1 {
                    var z = depth[py * dw + px]
                    if z < 0.4 { continue }
                    // 几乎顺着视线躺着的表面（远处的地面、天花板）：一个像素对应很长一条，
                    // 抬出来换个角度看是一道道拉丝。不要，留给背景去画。
                    let zr = depth[py * dw + px + 1], zd = depth[(py + 1) * dw + px]
                    let lateral = z / df
                    if zr > 0, abs(zr - z) > 3 * lateral { continue }
                    if zd > 0, abs(zd - z) > 3 * lateral { continue }
                    let d = ray(view, px, py)
                    // 落在地面附近的点直接压到地面上，跟真实的地面齐平。
                    if let floorY, d.y < -0.05 {
                        let t = (floorY - pano.center.y) / d.y
                        if abs(t - z) < 0.12 + 0.08 * z { z = t }
                    }
                    let i = panoIndex(d)
                    // 只要原来缺的方向；两张图重叠的那一条，谁先做归谁。
                    if pano.rgba[i.fine * 4 + 3] != 0 { continue }
                    if claimed[i.coarse] != 0, claimed[i.coarse] != tag { continue }
                    claimed[i.coarse] = tag
                    marked[i.coarse] = true
                    liftedDistance[i.coarse] = z * simd_length(d)
                    let p = pano.center + d * z
                    let spacing = z / df
                    let cls: UInt32 = spacing < 0.025 ? 0 : (spacing < 0.04 ? 1 : (spacing < 0.06 ? 2 : 3))
                    let j = i.fine * 4
                    let color = UInt32(rgba[j]) | (UInt32(rgba[j + 1]) << 8)
                        | (UInt32(rgba[j + 2]) << 16) | (1 << 24)
                    out.append(SplatMap.Splat(
                        x: p.x, y: p.y, z: p.z, color: color, tex: (cls << 14) | SplatMap.noTex))
                }
            }
        }
        return out
    }

    /// 发给服务的图：上半张是全景图（缺的地方中灰），下半张是遮罩（白 = 缺）。
    private static func encodePNG(_ pano: SplatMap.Panorama) -> Data? {
        let w = pano.width, h = pano.height
        var bytes = [UInt8](repeating: 255, count: w * h * 2 * 4)
        for i in 0..<w * h {
            let missing = pano.rgba[i * 4 + 3] == 0
            bytes[i * 4] = pano.rgba[i * 4]
            bytes[i * 4 + 1] = pano.rgba[i * 4 + 1]
            bytes[i * 4 + 2] = pano.rgba[i * 4 + 2]
            let m: UInt8 = missing ? 255 : 0
            let j = (w * h + i) * 4
            bytes[j] = m
            bytes[j + 1] = m
            bytes[j + 2] = m
        }
        guard
            let ctx = CGContext(
                data: &bytes, width: w, height: h * 2, bitsPerComponent: 8,
                bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
            let image = ctx.makeImage()
        else { return nil }
        return UIImage(cgImage: image).pngData()
    }

    /// 调试：把最近一次发出去和收回来的图存到 Documents，方便拉回电脑看。
    private static func dump(_ data: Data, name: String) {
        guard let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        else { return }
        try? data.write(to: dir.appendingPathComponent(name))
    }
}
