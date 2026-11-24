import Foundation
import simd

typealias Mat4 = simd_float4x4
typealias Vec3 = SIMD3<Float>

/// 每一帧都有的位姿。坐标约定跟 ARKit 一样：
/// 相机 +x 沿机身长边（竖拿时朝下），+y 沿短边，+z 从屏幕指向用户，镜头看 -z。
struct PoseSample {
    var time: Double
    var cameraToWorld: Mat4
    /// 前摄直接测到的头；测不到就是 nil，由 AvatarSolver 估。
    var headToWorld: Mat4?
    var floorY: Float?
}

/// ARKit 认出来的一块平面（地面、桌面、墙）。只记成一个圆盘：中心、法线、半径。
struct PlaneInfo {
    var center: Vec3
    var normal: Vec3
    var radius: Float
}

/// 一张深度图 + 同分辨率的颜色，大约 10 Hz。
struct DepthFrame {
    var width: Int
    var height: Int
    /// 沿相机光轴的距离（米），0 = 这个像素没有可信的深度。
    var depth: [Float]
    var color: [UInt8]  // RGBA
    var fx: Float
    var fy: Float
    var cx: Float
    var cy: Float
    var cameraToWorld: Mat4
    var maxRange: Float
    /// 雕刻时允许的深度误差 = carveBase + carvePerMeter × 距离。
    /// 默认值按 LiDAR 的精度定；单目深度误差大得多，要放宽。
    var carveBase: Float = 0.025
    var carvePerMeter: Float = 0.012
    /// 沿视线融合的容差 = fuseBase + fusePerMeter × 距离，两个都是 0 表示不做。
    /// 单目深度每帧的整体缩放都会飘几个百分点，同一面墙今天落在这层体素、明天落在隔壁层，
    /// 叠出好几层。打开后：新像素如果沿视线已经有一个差不多远的 splat，就只刷新那个 splat，不再新建。
    var fuseBase: Float = 0
    var fusePerMeter: Float = 0
    /// 同一帧的高清相机图（RGBA），视野和深度图完全一样，只是分辨率高。给贴图用，可以为空。
    var imageWidth = 0
    var imageHeight = 0
    var image: [UInt8] = []
    /// 这一刻已知的平面。深度落在平面附近的像素会被直接压到平面上（见 SplatMap.snapToPlanes）。
    var planes: [PlaneInfo] = []

    /// 单目深度的误差比 LiDAR 大一个量级，雕刻和融合的容差都要放宽。真机和合成测试共用这一组数。
    mutating func applyMonoTolerances() {
        let d = UserDefaults.standard
        func value(_ key: String, _ fallback: Float) -> Float {
            d.object(forKey: key) != nil ? d.float(forKey: key) : fallback
        }
        // 这几个数可以用同名的启动参数覆盖（比如 `-fusePerMeter 0.08`），回放调参用。
        // 默认值是在合成房间 + 真模型的录像上按「到真实表面的误差」调出来的（tools/replay.sh）。
        carveBase = value("carveBase", 0.05)
        carvePerMeter = value("carvePerMeter", 0.04)
        fuseBase = value("fuseBase", 0.03)
        fusePerMeter = value("fusePerMeter", 0.08)
    }
}

protocol FrameSource: AnyObject {
    var onPose: ((PoseSample) -> Void)? { get set }
    var onDepth: ((DepthFrame) -> Void)? { get set }
    /// 没有深度传感器时的兜底：稀疏的世界坐标点（x, y, z, rgb）。
    var onSparsePoints: (([SplatMap.Splat]) -> Void)? { get set }
    /// 没有深度图时单独给一张相机小图，贴到人偶手里的手机屏幕上。
    var onThumbnail: ((Int, Int, [UInt8]) -> Void)? { get set }
    /// 单目深度的原始材料（模型输出、对照点、相机图），给录制用。
    var onCapture: ((MonoCapture) -> Void)? { get set }
    var statusText: String { get }
    func start()
    func stop()
}

extension SIMD4 where Scalar == Float {
    var xyz: Vec3 { Vec3(x, y, z) }
}

extension SIMD3 where Scalar == Float {
    var horizontal: Vec3 { Vec3(x, 0, z) }

    func normalized(or fallback: Vec3) -> Vec3 {
        let l = simd_length(self)
        return l > 1e-5 ? self / l : fallback
    }
}

extension simd_float4x4 {
    var position: Vec3 { columns.3.xyz }
    var xAxis: Vec3 { columns.0.xyz }
    var yAxis: Vec3 { columns.1.xyz }
    var zAxis: Vec3 { columns.2.xyz }

    static func translation(_ t: Vec3) -> Mat4 {
        var m = matrix_identity_float4x4
        m.columns.3 = SIMD4(t.x, t.y, t.z, 1)
        return m
    }

    static func scale(_ s: Vec3) -> Mat4 {
        Mat4(diagonal: SIMD4(s.x, s.y, s.z, 1))
    }

    static func basis(x: Vec3, y: Vec3, z: Vec3, origin: Vec3) -> Mat4 {
        Mat4(columns: (
            SIMD4(x.x, x.y, x.z, 0), SIMD4(y.x, y.y, y.z, 0),
            SIMD4(z.x, z.y, z.z, 0), SIMD4(origin.x, origin.y, origin.z, 1)))
    }

    /// 右手系，看 -z，Metal 的 z ∈ [0, 1]。
    static func perspective(fovY: Float, aspect: Float, near: Float, far: Float) -> Mat4 {
        let ys = 1 / tan(fovY * 0.5)
        let xs = ys / aspect
        let zs = far / (near - far)
        return Mat4(columns: (
            SIMD4(xs, 0, 0, 0), SIMD4(0, ys, 0, 0),
            SIMD4(0, 0, zs, -1), SIMD4(0, 0, zs * near, 0)))
    }

    static func lookAt(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
        let z = simd_normalize(eye - target)
        let x = simd_normalize(simd_cross(up, z))
        let y = simd_cross(z, x)
        return Mat4(columns: (
            SIMD4(x.x, y.x, z.x, 0), SIMD4(x.y, y.y, z.y, 0),
            SIMD4(x.z, y.z, z.z, 0),
            SIMD4(-simd_dot(x, eye), -simd_dot(y, eye), -simd_dot(z, eye), 1)))
    }
}

func smoothstep(_ a: Float, _ b: Float, _ x: Float) -> Float {
    let t = min(max((x - a) / (b - a), 0), 1)
    return t * t * (3 - 2 * t)
}

/// 一阶低通的每帧系数：tau 秒后追上 63%。
func lowpass(_ dt: Float, _ tau: Float) -> Float {
    1 - exp(-dt / max(tau, 1e-4))
}
