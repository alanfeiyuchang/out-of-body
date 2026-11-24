import Foundation
import UIKit
import simd

/// 给「粗渲染 → 干净画面」的网络攒训练素材。
///
/// 回放一段录像时（`-dataset YES`）：
/// 1. 每隔几段扣下一段不喂给地图（`-holdEvery 4 -holdPhase 0 -holdBlock 8`），
///    回放完再从这些帧的机位把地图渲出来。地图没见过这些帧，渲出来的图带着真实的破洞和颗粒；
///    标准答案就是这一帧的相机照片。真机录像也能这么用。
/// 2. 合成房间的录像另外随机取一些第三人称机位（`-tpViews 40`），标准答案由合成场景直接算。
///
/// 每个机位在 Documents/dataset/ 下存三个文件：
///   <名字>_in.png  地图渲出来的颜色（没东西的地方是中灰）
///   <名字>_d.bin   每个像素到机位的距离，float32，0 = 没东西
///   <名字>_gt.png  标准答案
///   <名字>_up.txt  一个数字 k：把图逆时针转 k 个 90° 之后，世界的「上」就是图的上边
///                  （相机照片是按传感器方向存的，手机竖拿、横拿、平板各不一样，训练时要摆正）
final class DatasetExporter {
    struct View {
        var name: String
        var view: Mat4
        var proj: Mat4
        var width: Int
        var height: Int
        /// 标准答案（RGBA，width × height）。空的 = 没有标准答案（给蒸馏用：答案由老师模型来出）。
        var truth: [UInt8]
        /// 和 app 里跟拍镜头一样的裁剪：镜头和人之间的东西不画，镜头升高时掀掉屋顶。
        var nearClip: Float = 0.05
        var roofCut: Float = .greatestFiniteMagnitude
    }

    static let enabled = UserDefaults.standard.bool(forKey: "dataset")
    static let holdEvery = UserDefaults.standard.integer(forKey: "holdEvery")
    static let holdPhase = UserDefaults.standard.integer(forKey: "holdPhase")
    static let holdBlock = max(1, UserDefaults.standard.integer(forKey: "holdBlock"))
    static let thirdPersonViews = UserDefaults.standard.integer(forKey: "tpViews")
    /// `-chaseViews 60`：按 app 里跟拍镜头的算法随机取机位，只存粗渲染，不要标准答案。
    /// 真实录像也能用。蒸馏（让手机上的小网络学 服务器上那个大模型）时，输入要和 app 里实际看到的一样。
    static let chaseViewCount = UserDefaults.standard.integer(forKey: "chaseViews")
    static let tag = UserDefaults.standard.string(forKey: "datasetTag") ?? "d"
    /// 扣下来的帧渲多大（长边对着相机图的长边）。
    static let longSide = 512
    static let shortSide = 384

    /// 这一帧要不要扣下来。
    static func isHeld(frame: Int) -> Bool {
        guard enabled, holdEvery > 1 else { return false }
        return (frame / holdBlock) % holdEvery == holdPhase
    }

    static var directory: URL {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("dataset")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private let lock = NSLock()
    private var pending: [View] = []
    private var busy = false
    private var written = 0
    private var total = 0
    private let queue = DispatchQueue(label: "outofbody.dataset", qos: .utility)

    func enqueue(_ views: [View]) {
        lock.lock()
        pending += views
        total += views.count
        lock.unlock()
        if views.isEmpty { DebugLog.write("DATASET 完成：0 个机位") }
    }

    /// 渲染器空闲时来取下一个机位。上一个还没写完就先不给。
    func next() -> View? {
        lock.lock()
        defer { lock.unlock() }
        guard !busy, !pending.isEmpty else { return nil }
        busy = true
        return pending.removeFirst()
    }

    /// 渲染器把这个机位渲完读回来之后调。
    func finish(_ v: View, rgba: [UInt8], dist: [Float]) {
        queue.async {
            let dir = DatasetExporter.directory
            let base = dir.appendingPathComponent(v.name).path
            DatasetExporter.png(rgba, v.width, v.height)
                .map { try? $0.write(to: URL(fileURLWithPath: base + "_in.png")) }
            if !v.truth.isEmpty {
                DatasetExporter.png(v.truth, v.width, v.height)
                    .map { try? $0.write(to: URL(fileURLWithPath: base + "_gt.png")) }
            }
            dist.withUnsafeBytes { raw in
                try? Data(raw).write(to: URL(fileURLWithPath: base + "_d.bin"))
            }
            try? "\(DatasetExporter.upTurns(view: v.view))".write(
                toFile: base + "_up.txt", atomically: true, encoding: .utf8)
            self.lock.lock()
            self.busy = false
            self.written += 1
            let done = self.pending.isEmpty
            let n = self.written
            self.lock.unlock()
            if done { DebugLog.write("DATASET 完成：\(n) 个机位") }
        }
    }

    /// 世界的「上」在这张图里朝哪边，换算成「逆时针转几个 90° 能摆正」。
    static func upTurns(view: Mat4) -> Int {
        let up = view * SIMD4<Float>(0, 1, 0, 0)  // 相机坐标：x 朝图像右边，y 朝图像上边
        if abs(up.y) >= abs(up.x) { return up.y >= 0 ? 0 : 2 }
        // 上在图像右边：逆时针转一次；在左边：转三次（等于顺时针一次）。
        return up.x > 0 ? 1 : 3
    }

    private static func png(_ rgba: [UInt8], _ w: Int, _ h: Int) -> Data? {
        var bytes = rgba
        guard
            let ctx = CGContext(
                data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
            let image = ctx.makeImage()
        else { return nil }
        return UIImage(cgImage: image).pngData()
    }

    /// 把一张 RGBA 图缩到指定大小。
    static func resize(_ rgba: [UInt8], _ w: Int, _ h: Int, to nw: Int, _ nh: Int) -> [UInt8] {
        if w == nw, h == nh { return rgba }
        var src = rgba
        var out = [UInt8](repeating: 255, count: nw * nh * 4)
        let space = CGColorSpaceCreateDeviceRGB()
        let info = CGImageAlphaInfo.noneSkipLast.rawValue
        guard
            let sctx = CGContext(
                data: &src, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                space: space, bitmapInfo: info),
            let image = sctx.makeImage(),
            let dctx = CGContext(
                data: &out, width: nw, height: nh, bitsPerComponent: 8, bytesPerRow: nw * 4,
                space: space, bitmapInfo: info)
        else { return out }
        dctx.interpolationQuality = .high
        dctx.draw(image, in: CGRect(x: 0, y: 0, width: nw, height: nh))
        return out
    }

    /// 针孔相机的投影矩阵（主点可以不在正中），配合 `cameraToWorld.inverse` 当视图矩阵用，
    /// 渲出来的图和这台相机拍的照片逐像素对得上。fx…cy 是 width × height 这张图下的内参。
    static func projection(
        fx: Float, fy: Float, cx: Float, cy: Float, width: Int, height: Int, near: Float = 0.05,
        far: Float = 80
    ) -> Mat4 {
        let w = Float(width), h = Float(height)
        let zs = far / (near - far)
        return Mat4(columns: (
            SIMD4(2 * fx / w, 0, 0, 0), SIMD4(0, 2 * fy / h, 0, 0),
            SIMD4(1 - 2 * cx / w, 2 * cy / h - 1, zs, -1), SIMD4(0, 0, zs * near, 0)))
    }

    /// 扣下来的一帧 → 一个导出机位。
    static func heldView(_ c: MonoCapture, index: Int) -> View {
        let landscape = c.imageWidth >= c.imageHeight
        let w = landscape ? longSide : shortSide, h = landscape ? shortSide : longSide
        let sx = Float(w) / Float(c.width), sy = Float(h) / Float(c.height)
        return View(
            name: String(format: "%@_h%03d", tag, index),
            view: c.cameraToWorld.inverse,
            proj: projection(
                fx: c.fx * sx, fy: c.fy * sy, cx: c.cx * sx, cy: c.cy * sy, width: w, height: h),
            width: w, height: h,
            truth: resize(c.image, c.imageWidth, c.imageHeight, to: w, h))
    }

    /// app 里实际会出现的机位：人在录像里某一刻的位置，镜头在他后上方，角度和距离在默认值附近随机
    /// （三成的机位绕到任意方向，模拟用户用手指转着看）。裁剪规则和视角都照 Renderer 里的来。
    static func chaseViews(poses: [Mat4], count: Int) -> [View] {
        guard count > 0, !poses.isEmpty else { return [] }
        var state = tag.unicodeScalars.reduce(UInt32(84_696_351)) { ($0 ^ $1.value) &* 16_777_619 } | 1
        func rand() -> Float {
            state = state &* 1_664_525 &+ 1_013_904_223
            return Float(state >> 8) / Float(1 << 24)
        }
        let up = Vec3(0, 1, 0)
        let w = shortSide, h = longSide
        // 和 Renderer 里交给生成模型的那张图一样：视角比屏幕放大 12%，画面比例取手机屏幕的。
        let proj = Mat4.perspective(
            fovY: 2 * atan(tan(30 * Float.pi / 180) * 1.12), aspect: 0.70, near: 0.05, far: 80)
        var out: [View] = []
        for _ in 0..<count {
            let cam = poses[Int(rand() * Float(poses.count)) % poses.count]
            let heading = (-cam.zAxis).horizontal.normalized(or: Vec3(0, 0, -1))
            let head = cam.position - heading * 0.36 + up * 0.24
            let target = head - up * 0.5
            let yaw = rand() < 0.7 ? (rand() - 0.5) * 1.0 : (rand() - 0.5) * 2 * .pi
            let pitch = 0.15 + rand() * 0.75
            let dist = 1.8 + rand() * 3.2
            let b = -heading
            let c = cos(yaw), s = sin(yaw)
            let back = Vec3(b.x * c - b.z * s, 0, b.x * s + b.z * c)
            let eye = target + back * (cos(pitch) * dist) + up * (sin(pitch) * dist)
            let view = Mat4.lookAt(eye: eye, target: target - back * 1.1 + up * 0.15, up: up)
            out.append(View(
                name: String(format: "%@_c%03d", tag, out.count), view: view, proj: proj,
                width: w, height: h, truth: [],
                nearClip: max(0.05, simd_distance(eye, target) - 1.1),
                roofCut: eye.y > head.y + 1.0 ? head.y + 0.7 : .greatestFiniteMagnitude))
        }
        return out
    }

    /// 合成房间里随机取的第三人称机位：以录像里某一刻的手机位置为「人」，
    /// 按 app 里跟拍相机的算法退到后上方，角度和距离随机。标准答案由合成场景直接算。
    static func thirdPersonViews(poses: [Mat4], time: Float, count: Int) -> [View] {
        guard count > 0, !poses.isEmpty else { return [] }
        let scene = SyntheticScene.shared
        let boxes = scene.boxes(at: time)
        // 每个标签一组固定的随机数（String.hashValue 每次启动都不一样，不能用）。
        var state = tag.unicodeScalars.reduce(UInt32(2_166_136_261)) { ($0 ^ $1.value) &* 16_777_619 } | 1
        func rand() -> Float {
            state = state &* 1_664_525 &+ 1_013_904_223
            return Float(state >> 8) / Float(1 << 24)
        }
        let w = shortSide, h = longSide
        let fovY: Float = 60 * .pi / 180
        let focal = Float(h) / 2 / tan(fovY / 2)
        var out: [View] = []
        var tries = 0
        while out.count < count, tries < count * 30 {
            tries += 1
            let cam = poses[Int(rand() * Float(poses.count)) % poses.count]
            // 手机镜头朝 -z；人的朝向取它的水平分量。
            let heading = (-cam.zAxis).horizontal.normalized(or: Vec3(0, 0, -1))
            let target = cam.position - Vec3(0, 0.25, 0)
            let yaw = (rand() - 0.5) * 2 * .pi
            let pitch = 0.08 + rand() * 1.0
            let dist = 1.2 + rand() * 3.2
            let b = -heading
            let c = cos(yaw), s = sin(yaw)
            let back = Vec3(b.x * c - b.z * s, 0, b.x * s + b.z * c)
            var eye = target + back * (cos(pitch) * dist) + Vec3(0, sin(pitch) * dist, 0)
            // 标准答案是在房间里面算的，机位不能出墙。
            eye = simd_clamp(
                eye, scene.roomLo + Vec3(repeating: 0.55), scene.roomHi - Vec3(repeating: 0.55))
            // 贴着家具的机位（半米以内）不要：几厘米的误差在那里会变成整幅画面的差别。
            if scene.blocked(eye, boxes: boxes, margin: 0.5) { continue }
            let look = target - back * 1.1 + Vec3(0, 0.15, 0)
            if simd_distance(eye, look) < 0.5 { continue }
            let view = Mat4.lookAt(eye: eye, target: look, up: Vec3(0, 1, 0))
            let truth = scene.render(
                cam: view.inverse, boxes: boxes, width: w, height: h, focal: focal)
            out.append(View(
                name: String(format: "%@_t%03d", tag, out.count), view: view,
                proj: Mat4.perspective(
                    fovY: fovY, aspect: Float(w) / Float(h), near: 0.05, far: 80),
                width: w, height: h, truth: truth))
        }
        return out
    }
}
