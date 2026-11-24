import Foundation
import UIKit
import simd

/// 把一段扫描原样存下来：每一帧单目深度的原始材料（见 MonoCapture）和全部位姿。
///
/// 存的是算法的输入而不是结果，所以对齐、融合、去飞点怎么改，旧录像都能重跑。
/// 文件在 Documents/rec/capture.bin，回放见 ReplaySource。
///
/// 格式：一条条记录，每条 = 类型(1 字节) + 长度(4 字节) + 内容，全部小端。
///   类型 1 = 位姿，类型 2 = 一帧单目深度。
final class Recorder {
    static var fileURL: URL {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("rec")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("capture.bin")
    }
    /// 一分钟大约 100 MB，录太久既占地方也难拉回电脑。
    static let maxBytes = 400_000_000

    private let queue = DispatchQueue(label: "outofbody.recorder", qos: .utility)
    private let lock = NSLock()
    private var handle: FileHandle?
    private var pending = 0
    private var _frames = 0
    private var _bytes = 0
    private var _recording = false

    var isRecording: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _recording
    }

    var status: String {
        lock.lock()
        defer { lock.unlock() }
        return String(format: "录制中 %d 帧 · %.0f MB", _frames, Double(_bytes) / 1e6)
    }

    func start() {
        let url = Recorder.fileURL
        FileManager.default.createFile(atPath: url.path, contents: nil)
        guard let h = try? FileHandle(forWritingTo: url) else { return }
        lock.lock()
        handle = h
        _frames = 0
        _bytes = 0
        _recording = true
        lock.unlock()
        DebugLog.write("REC 开始")
    }

    func stop() {
        lock.lock()
        let was = _recording
        _recording = false
        lock.unlock()
        guard was else { return }
        queue.async {
            try? self.handle?.close()
            self.handle = nil
            DebugLog.write("REC 结束：\(self._frames) 帧，\(self._bytes / 1_000_000) MB")
        }
    }

    private func write(tag: UInt8, _ payload: Data) {
        var head = Data([tag])
        head.append(u32: UInt32(payload.count))
        handle?.write(head)
        handle?.write(payload)
        lock.lock()
        _bytes += payload.count + 5
        let full = _bytes > Recorder.maxBytes
        lock.unlock()
        if full { stop() }
    }

    func add(pose p: PoseSample) {
        guard isRecording else { return }
        queue.async {
            var d = Data()
            d.append(f64: p.time)
            d.append(matrix: p.cameraToWorld)
            d.append(p.headToWorld == nil ? 0 : 1)
            d.append(matrix: p.headToWorld ?? matrix_identity_float4x4)
            d.append(p.floorY == nil ? 0 : 1)
            d.append(f32: p.floorY ?? 0)
            self.write(tag: 1, d)
        }
    }

    func add(_ c: MonoCapture) {
        lock.lock()
        // 存得比来得慢的时候丢帧，不让内存里越积越多。
        let ok = _recording && pending < 3
        if ok { pending += 1 }
        lock.unlock()
        guard ok else { return }
        queue.async {
            defer {
                self.lock.lock()
                self.pending -= 1
                self.lock.unlock()
            }
            guard let jpeg = Recorder.jpeg(c.image, c.imageWidth, c.imageHeight) else { return }
            var d = Data()
            d.append(f64: c.time)
            d.append(matrix: c.cameraToWorld)
            for v in [c.fx, c.fy, c.cx, c.cy] { d.append(f32: v) }
            d.append(u32: UInt32(c.width))
            d.append(u32: UInt32(c.height))
            // 逆深度存成半精度，够用，体积减半。
            var half = c.disparity.map { Float16($0) }
            half.withUnsafeMutableBytes { d.append(contentsOf: $0) }
            d.append(u32: UInt32(c.samples.count))
            for s in c.samples {
                d.append(u32: UInt32(s.u))
                d.append(u32: UInt32(s.v))
                d.append(f32: s.z)
            }
            d.append(u32: UInt32(c.planes.count))
            for p in c.planes {
                for v in [p.center.x, p.center.y, p.center.z, p.normal.x, p.normal.y, p.normal.z,
                    p.radius]
                { d.append(f32: v) }
            }
            d.append(u32: UInt32(c.imageWidth))
            d.append(u32: UInt32(c.imageHeight))
            d.append(u32: UInt32(jpeg.count))
            d.append(jpeg)
            self.write(tag: 2, d)
            self.lock.lock()
            self._frames += 1
            self.lock.unlock()
        }
    }

    private static func jpeg(_ rgba: [UInt8], _ w: Int, _ h: Int) -> Data? {
        var bytes = rgba
        guard
            let ctx = CGContext(
                data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
            let image = ctx.makeImage()
        else { return nil }
        return UIImage(cgImage: image).jpegData(compressionQuality: 0.85)
    }
}

extension Data {
    mutating func append(u32 v: UInt32) {
        var x = v.littleEndian
        Swift.withUnsafeBytes(of: &x) { append(contentsOf: $0) }
    }

    mutating func append(f32 v: Float) { append(u32: v.bitPattern) }

    mutating func append(f64 v: Double) {
        var x = v.bitPattern.littleEndian
        Swift.withUnsafeBytes(of: &x) { append(contentsOf: $0) }
    }

    mutating func append(matrix m: simd_float4x4) {
        for c in [m.columns.0, m.columns.1, m.columns.2, m.columns.3] {
            for v in [c.x, c.y, c.z, c.w] { append(f32: v) }
        }
    }
}
