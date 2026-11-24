import Foundation
import UIKit
import simd

/// 回放 Recorder 存下来的一段扫描，走跟真机完全一样的后处理和融合。
///
/// `-replay YES`：读 Documents/rec/capture.bin。
/// `-replayFast YES`：不按原来的时间等，能多快跑多快（配合 SplatMap.blocking），跑完打一行结果。
/// 真机上录的文件拷进模拟器的 Documents/rec/ 就能在电脑上反复重跑。
final class ReplaySource: FrameSource {
    var onPose: ((PoseSample) -> Void)?
    var onDepth: ((DepthFrame) -> Void)?
    var onSparsePoints: (([SplatMap.Splat]) -> Void)?
    var onThumbnail: ((Int, Int, [UInt8]) -> Void)?
    var onCapture: ((MonoCapture) -> Void)?
    /// 快放跑完时调一次。
    var onFinished: ((Int) -> Void)?
    private(set) var statusText = "回放"

    static let fast = UserDefaults.standard.bool(forKey: "replayFast")
    /// `-replayLimit 120`：只放前 N 帧深度。
    private let limit = UserDefaults.standard.integer(forKey: "replayLimit")
    private let queue = DispatchQueue(label: "outofbody.replay", qos: .userInitiated)
    private var stopped = false
    /// 攒训练素材时扣下来没喂给地图的帧（见 DatasetExporter），和所有帧的机位。
    private(set) var heldViews: [DatasetExporter.View] = []
    private(set) var capturePoses: [Mat4] = []
    private(set) var lastCaptureTime: Double = 0

    func start() {
        queue.async { self.run() }
    }

    func stop() {
        stopped = true
    }

    private struct Reader {
        let data: Data
        var at: Int

        mutating func u8() -> UInt8 {
            defer { at += 1 }
            return data[at]
        }

        mutating func u32() -> UInt32 {
            defer { at += 4 }
            return data.subdata(in: at..<at + 4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
        }

        mutating func f32() -> Float { Float(bitPattern: u32()) }

        mutating func f64() -> Double {
            defer { at += 8 }
            return Double(bitPattern: data.subdata(in: at..<at + 8).withUnsafeBytes {
                $0.loadUnaligned(as: UInt64.self)
            })
        }

        mutating func matrix() -> Mat4 {
            var c = [SIMD4<Float>]()
            for _ in 0..<4 { c.append(SIMD4(f32(), f32(), f32(), f32())) }
            return Mat4(columns: (c[0], c[1], c[2], c[3]))
        }
    }

    /// `-replayFile import.bin`：回放 Documents/rec/ 下的另一个文件。
    /// 配合 `-record YES` 可以把「原始录像」（类型 3，见下）补上单目深度后存成正常的录像：
    /// 读的和写的不能是同一个文件。
    private static var inputURL: URL {
        guard let name = UserDefaults.standard.string(forKey: "replayFile"), !name.isEmpty
        else { return Recorder.fileURL }
        return Recorder.fileURL.deletingLastPathComponent().appendingPathComponent(name)
    }
    /// `-bakeOnly YES`：只补单目深度交给录制，不建图（转换外部数据集时用，省时间）。
    private let bakeOnly = UserDefaults.standard.bool(forKey: "bakeOnly")
    /// `-floorBias 0.75`：把录像里的地面高度人为抬高，模拟「ARKit 把桌面认成了地面」，测人偶那边扛不扛得住。
    private let floorBias = UserDefaults.standard.float(forKey: "floorBias")

    private func run() {
        guard let data = try? Data(contentsOf: ReplaySource.inputURL, options: .alwaysMapped),
            data.count > 5
        else {
            statusText = "回放：没有录像"
            return
        }
        var r = Reader(data: data, at: 0)
        var frames = 0
        var firstTime: Double?
        let wallStart = CACurrentMediaTime()
        while !stopped, r.at + 5 <= data.count {
            let tag = r.u8()
            let length = Int(r.u32())
            let end = r.at + length
            if end > data.count { break }  // 录到一半断掉的最后一条
            let time = r.f64()
            if !ReplaySource.fast {
                let t0 = firstTime ?? time
                firstTime = t0
                let wait = (time - t0) - (CACurrentMediaTime() - wallStart)
                if wait > 0 { Thread.sleep(forTimeInterval: min(wait, 1)) }
            }
            if tag == 1 {
                let cam = r.matrix()
                let hasHead = r.u8() == 1
                let head = r.matrix()
                let hasFloor = r.u8() == 1
                let floor = r.f32()
                onPose?(PoseSample(
                    time: time, cameraToWorld: cam, headToWorld: hasHead ? head : nil,
                    floorY: hasFloor ? floor + floorBias : nil))
            } else if tag == 2 || tag == 3 {
                // 类型 3 是外部数据集转来的原始材料（tools/neural/prep_arkit.py）：有相机图、位姿、
                // 对照点和平面，还没有单目深度，在这里现算，之后和真机录的一样。
                let parsed = tag == 2
                    ? ReplaySource.capture(&r, time: time) : ReplaySource.rawCapture(&r, time: time)
                if let c = parsed {
                    if tag == 3 { onCapture?(c) }
                    if bakeOnly {
                        frames += 1
                        statusText = "转换 \(frames) 帧"
                        r.at = end
                        continue
                    }
                    frames += 1
                    capturePoses.append(c.cameraToWorld)
                    lastCaptureTime = time
                    if DatasetExporter.isHeld(frame: frames - 1) {
                        heldViews.append(DatasetExporter.heldView(c, index: frames - 1))
                    } else {
                        let result = MonoPipeline.process(c)
                        statusText = "回放 \(frames) 帧 · " + result.status
                        if let f = result.frame { onDepth?(f) }
                    }
                    if limit > 0, frames >= limit { break }
                }
            }
            r.at = end
        }
        statusText = "回放结束（\(frames) 帧）"
        onFinished?(frames)
    }

    private static func rawCapture(_ r: inout Reader, time: Double) -> MonoCapture? {
        let cam = r.matrix()
        let fx = r.f32(), fy = r.f32(), cx = r.f32(), cy = r.f32()
        var raw: [(Float, Float, Float)] = []
        for _ in 0..<Int(r.u32()) { raw.append((r.f32(), r.f32(), r.f32())) }
        var planes: [PlaneInfo] = []
        for _ in 0..<Int(r.u32()) {
            let c = Vec3(r.f32(), r.f32(), r.f32())
            let n = Vec3(r.f32(), r.f32(), r.f32())
            planes.append(PlaneInfo(center: c, normal: n, radius: r.f32()))
        }
        let iw = Int(r.u32()), ih = Int(r.u32())
        let jpegLength = Int(r.u32())
        let jpeg = r.data.subdata(in: r.at..<r.at + jpegLength)
        r.at += jpegLength
        guard let rgba = decode(jpeg, iw, ih),
            let pb = MonoDepthEstimator.pixelBuffer(rgba: rgba, width: iw, height: ih),
            let out = MonoDepthEstimator.shared?.run(pb)
        else { return nil }
        let w = out.width, h = out.height
        let sx = Float(w) / Float(iw), sy = Float(h) / Float(ih)
        let samples = raw.map { s in
            DepthAlign.Sample(
                u: min(w - 1, max(0, Int(s.0 * Float(w)))),
                v: min(h - 1, max(0, Int(s.1 * Float(h)))), z: s.2)
        }
        return MonoCapture(
            time: time, cameraToWorld: cam, width: w, height: h, disparity: out.values,
            fx: fx * sx, fy: fy * sy, cx: cx * sx, cy: cy * sy, samples: samples,
            imageWidth: iw, imageHeight: ih, image: rgba, planes: planes)
    }

    private static func decode(_ jpeg: Data, _ iw: Int, _ ih: Int) -> [UInt8]? {
        guard let image = UIImage(data: jpeg)?.cgImage else { return nil }
        var rgba = [UInt8](repeating: 255, count: iw * ih * 4)
        guard
            let ctx = CGContext(
                data: &rgba, width: iw, height: ih, bitsPerComponent: 8, bytesPerRow: iw * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: iw, height: ih))
        return rgba
    }

    private static func capture(_ r: inout Reader, time: Double) -> MonoCapture? {
        let cam = r.matrix()
        let fx = r.f32(), fy = r.f32(), cx = r.f32(), cy = r.f32()
        let w = Int(r.u32()), h = Int(r.u32())
        let half = r.data.subdata(in: r.at..<r.at + w * h * 2).withUnsafeBytes {
            Array($0.bindMemory(to: Float16.self))
        }
        r.at += w * h * 2
        let disparity = half.map { Float($0) }
        var samples: [DepthAlign.Sample] = []
        for _ in 0..<Int(r.u32()) {
            let u = Int(r.u32()), v = Int(r.u32())
            samples.append(DepthAlign.Sample(u: u, v: v, z: r.f32()))
        }
        var planes: [PlaneInfo] = []
        for _ in 0..<Int(r.u32()) {
            let c = Vec3(r.f32(), r.f32(), r.f32())
            let n = Vec3(r.f32(), r.f32(), r.f32())
            planes.append(PlaneInfo(center: c, normal: n, radius: r.f32()))
        }
        let iw = Int(r.u32()), ih = Int(r.u32())
        let jpegLength = Int(r.u32())
        let jpeg = r.data.subdata(in: r.at..<r.at + jpegLength)
        r.at += jpegLength
        guard let image = UIImage(data: jpeg)?.cgImage else { return nil }
        var rgba = [UInt8](repeating: 255, count: iw * ih * 4)
        guard
            let ctx = CGContext(
                data: &rgba, width: iw, height: ih, bitsPerComponent: 8, bytesPerRow: iw * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: iw, height: ih))
        return MonoCapture(
            time: time, cameraToWorld: cam, width: w, height: h, disparity: disparity,
            fx: fx, fy: fy, cx: cx, cy: cy, samples: samples,
            imageWidth: iw, imageHeight: ih, image: rgba, planes: planes)
    }
}
