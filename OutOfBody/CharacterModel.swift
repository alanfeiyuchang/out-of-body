import Foundation
import MetalKit
import simd

/// 现成的人物模型，带三段动作：待机循环、走路循环、坐下。
///
/// 不做骨骼蒙皮：tools/bake_animated.py 把每一帧的顶点位置和法线都烘焙好了，播放时在相邻两帧之间插值
/// （顶点着色器 character_vs）。帧的顺序：待机循环、走路循环（都首尾相接）、坐下（第 0 帧站着，最后一帧坐稳）。
/// 现在放的是 Sketchfab 上 Mat Rex 的「Tesla Optimus」（CC BY 4.0，见 Resources/CREDITS.txt）：
/// 原模型没有骨骼，骨骼是 服务器上 SkinTokens 那套流程自动绑的，动作是烘焙脚本按真人步态算的。
/// 换模型只要重新生成 character.bin / .png。`-capsule YES`：不用它，退回由手机位姿推出来的圆棒人偶。
final class CharacterModel {
    /// 所有帧的顶点连着放在一个缓冲里，每个顶点 24 字节（位置 + 法线）。
    let frames: MTLBuffer
    let uvBuffer: MTLBuffer
    let indexBuffer: MTLBuffer
    let vertexCount: Int
    let indexCount: Int
    /// 一共几帧。
    let frameCount: Int
    /// 三段动作各有几帧（手机模型只有开合一段，全算在 idleFrames 里）。
    let idleFrames: Int
    let walkFrames: Int
    let sitFrames: Int
    var walkStart: Int { idleFrames }
    var sitStart: Int { idleFrames + walkFrames }
    /// 模型自己的身高（模型单位）。画的时候按用户的身高缩放。
    let height: Float
    /// 走一个循环（左右各一步）前进多远、坐稳时胯比站着往后挪了多远（模型单位）。
    let stride: Float
    let sitBack: Float
    /// 每一帧手机画在模型的哪里（模型坐标：y 朝上，脚底 y = 0，脸朝 +z）。
    let hands: [Vec3]
    let texture: MTLTexture

    /// 第 n 帧的顶点在 frames 里从哪个字节开始。
    func offset(frame n: Int) -> Int { n * vertexCount * 24 }

    /// name = 资源文件名（不带扩展名）："character" 是人物；"phone" 是折叠手机，
    /// 它的帧不是动作而是开合：第 0 帧合上，最后一帧完全展开（tools/bake_phone.py）。
    init?(device: MTLDevice, name: String = "character") {
        if name == "character", UserDefaults.standard.bool(forKey: "capsule") { return nil }
        guard let url = Bundle.main.url(forResource: name, withExtension: "bin"),
            let data = try? Data(contentsOf: url, options: .alwaysMapped), data.count > 36,
            let imageURL = Bundle.main.url(forResource: name, withExtension: "png"),
            let tex = try? MTKTextureLoader(device: device).newTexture(
                URL: imageURL, options: [.SRGB: false, .generateMipmaps: name == "character"])
        else { return nil }
        func u32(_ at: Int) -> Int {
            Int(data.subdata(in: at..<at + 4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) })
        }
        func f32(_ at: Int) -> Float { Float(bitPattern: UInt32(u32(at))) }
        // 两种文件：TEC3 是带三段动作的人物（tools/bake_animated.py），TEC2 是只有一段的（折叠手机）。
        let vertices = u32(4), indices = u32(8)
        let idle: Int, walk: Int, sit: Int, uvStart: Int
        var handList: [Vec3] = []
        if data.prefix(4) == Data("TEC3".utf8) {
            idle = u32(12)
            walk = u32(16)
            sit = u32(20)
            height = f32(24)
            stride = f32(28)
            sitBack = f32(32)
            let count = idle + walk + sit
            guard data.count > 36 + count * 12 else { return nil }
            handList = (0..<count).map { Vec3(f32(36 + $0 * 12), f32(40 + $0 * 12), f32(44 + $0 * 12)) }
            uvStart = 36 + count * 12
        } else if data.prefix(4) == Data("TEC2".utf8) {
            idle = u32(12)
            walk = 0
            sit = 0
            height = f32(16)
            stride = 1
            sitBack = 0
            handList = Array(repeating: Vec3(f32(20), f32(24), f32(28)), count: idle)
            uvStart = 32
        } else {
            return nil
        }
        let frameCount = idle + walk + sit
        let uvBytes = vertices * 8, indexBytes = indices * 4, frameBytes = frameCount * vertices * 24
        let indexStart = uvStart + uvBytes, frameStart = indexStart + indexBytes
        guard frameCount >= 2, data.count >= frameStart + frameBytes else { return nil }
        func buffer(_ start: Int, _ length: Int) -> MTLBuffer? {
            data.subdata(in: start..<start + length).withUnsafeBytes {
                device.makeBuffer(bytes: $0.baseAddress!, length: length)
            }
        }
        guard let uv = buffer(uvStart, uvBytes), let ib = buffer(indexStart, indexBytes),
            let fb = buffer(frameStart, frameBytes)
        else { return nil }
        uvBuffer = uv
        indexBuffer = ib
        frames = fb
        vertexCount = vertices
        indexCount = indices
        self.frameCount = frameCount
        idleFrames = idle
        walkFrames = walk
        sitFrames = sit
        hands = handList
        texture = tex
    }
}
