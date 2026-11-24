import MetalKit
import simd

// 下面四个结构体的布局要跟 Shaders.metal 里同名的结构体一致。
struct SplatUniforms {
    var view: Mat4
    var proj: Mat4
    var phonePos: SIMD4<Float>
    var fog: SIMD4<Float>
    var params: SIMD4<Float>
    var params2: SIMD4<Float>
    var live: SIMD4<Float>
    var eye: SIMD4<Float>
}

struct ResolveUniforms {
    var invViewProj: Mat4
    var cameraPos: SIMD4<Float>
    var cameraFwd: SIMD4<Float>
    var bg: SIMD4<Float>
    var fog: SIMD4<Float>
    var params: SIMD4<Float>
    var misc: SIMD4<Float>
    var envCenter: SIMD4<Float>
    var genCenter: SIMD4<Float>
    var room: SIMD4<Float>
    var floorMap: SIMD4<Float>
    var extra: SIMD4<Float> = .zero
    var genViewProj: Mat4 = matrix_identity_float4x4
}

struct MeshUniforms {
    var viewProj: Mat4
    var lightDir: SIMD4<Float>
    var cameraPos: SIMD4<Float>
}

struct MeshInstance {
    var model: Mat4
    var color: SIMD4<Float>
}

struct GenMeshUniforms {
    var viewProj: Mat4
    /// x: 这张图现在用几成, y/z: 网格横纵几格。
    var params: SIMD4<Float>
    var screen: SIMD4<Float>
}

struct CharacterUniforms {
    var model: Mat4
    /// x 待机、y 走路、z 坐下：各自相邻两帧之间的位置。
    var blend: SIMD4<Float>
    /// x 走路的程度，y 坐下的程度。
    var weight: SIMD4<Float>
}

/// 人物这一帧播到哪：三段动作各取哪两帧、混多少。
struct CharacterFrames {
    var idle: (a: Int, b: Int, t: Float)
    var walk: (a: Int, b: Int, t: Float)
    var sit: (a: Int, b: Int, t: Float)
    var walkWeight: Float
    var sitWeight: Float
}

final class Renderer: NSObject, MTKViewDelegate {
    private let model: AppModel
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let meshes: Meshes
    private let splatDepthPipeline: MTLRenderPipelineState
    private let splatBlendPipeline: MTLRenderPipelineState
    private let resolvePipeline: MTLRenderPipelineState
    private let meshPipeline: MTLRenderPipelineState
    private let avatarPipeline: MTLRenderPipelineState
    private let panoPipeline: MTLRenderPipelineState
    private let blitPipeline: MTLRenderPipelineState
    private let characterPipeline: MTLRenderPipelineState
    private let genMeshPipeline: MTLRenderPipelineState
    private let genMeshIndices: MTLBuffer?
    /// `-genLive YES`：交给 服务器画的图带上「实时层」（当前这一帧刚拍到、还没并进地图的点）。默认不带：
    /// 训练素材里就没有它；它的深度是单帧的，窗户、屏幕这种地方会飘在半空，网络照着画出一团颜色。
    private let genLive = UserDefaults.standard.bool(forKey: "genLive")
    /// `-genBlend x`：新来的生成图盖住旧图几成（默认 0.6；1 = 整张替换）。
    private let genBlend: Float = UserDefaults.standard.object(forKey: "genBlend") == nil
        ? 0.6 : UserDefaults.standard.float(forKey: "genBlend")
    /// `-genSharpen x`：生成图锐化多少（默认 0.4，0.01 相当于关）。锐化会把每张图之间的细微差别也放大：
    /// 清晰那段回放里，1.2 时相邻帧平均差 1.01 → 1.46，0.6 时 0.87 → 1.06，所以只开一点。
    private let genSharpen: Float = UserDefaults.standard.object(forKey: "genSharpen") == nil
        ? 0.4 : UserDefaults.standard.float(forKey: "genSharpen")
    /// 镜头和人之间的东西从离人多远开始裁掉（`-nearMargin 米`）。原来是 1.1 米：人背后一米内的东西（比如背对着的
    /// 显示器）会留在画面里，而人偶是直接画在最上面的，看起来就像显示器穿过了人（2026-11-23 真机截图）。
    private let nearMargin: Float = UserDefaults.standard.object(forKey: "nearMargin") == nil
        ? 0.25 : UserDefaults.standard.float(forKey: "nearMargin")
    /// 全景壳（见 GenShell）。`-genShell NO` 关掉，退回每张图各自立一张皮的做法。
    private let genShell: GenShell?
    private let shellEnabled = UserDefaults.standard.object(forKey: "genShell") == nil
        || UserDefaults.standard.bool(forKey: "genShell")
    /// `-shellDraw YES`：直接把全景壳画出来（调试用，边缘全是毛刺）。
    private let shellDraw = UserDefaults.standard.bool(forKey: "shellDraw")
    /// `-genCull 1/2`：生成图那张网剔正面还是背面（调试用，默认剔背面）。
    private let genCull = UserDefaults.standard.integer(forKey: "genCull")
    /// 垫底图层边缘淡出的宽度（占图宽的比例）。`-genFeather 0.04` 回到以前的窄边做对比。
    private let genFeather: Float = UserDefaults.standard.object(forKey: "genFeather") != nil
        ? UserDefaults.standard.float(forKey: "genFeather") : 0.26
    /// 预先绕人画一圈垫底图（见发图那段）。`-genRing NO` 关掉做对比。
    private let ringEnabled = UserDefaults.standard.object(forKey: "genRing") == nil
        || UserDefaults.standard.bool(forKey: "genRing")
    private var ringTick = 0
    /// 一圈分几格：每格 24°，一张图左右各能看到 30° 上下，相邻两张有重叠。
    static let ringSlots = 15
    /// `-rawTint YES`：把实时点云画面染成洋红，量「生成的图没盖住的地方」占屏幕几成用。
    private let rawTint = UserDefaults.standard.bool(forKey: "rawTint")
    private var lastCamHeading = Vec3(0, 0, -1)
    private var headingRate: Float = 0
    /// `-walkTest YES`：让人物一直原地走路，截图检查走路动作用。
    private let walkTest = UserDefaults.standard.bool(forKey: "walkTest")
    /// `-yawSpin 弧度每秒`：镜头自己绕着人转，模拟用手拖动视角，截图检查「生成」画面跟不上时的样子。
    private let yawSpin = UserDefaults.standard.float(forKey: "yawSpin")
    /// `-yawSwing 弧度`：镜头左右来回摆（2.4 秒一个来回），模拟手指来回拖：速度一直在变，预判机位总会差一点，
    /// 是「生成图重新投影」最吃力的情况。
    private let yawSwing = UserDefaults.standard.float(forKey: "yawSwing")
    /// `-yawFlick 弧度`：镜头先不动 9 秒，之后每 4 秒朝同一个方向猛转这么多（0.4 秒转完），
    /// 模拟手指一甩：转过去的地方之前镜头没对着过，看那一下露出来的是什么。
    private let yawFlick = UserDefaults.standard.float(forKey: "yawFlick")
    /// `-genFreeze 秒`：启动这么久之后不再要新的生成图，镜头这时才开始摆。
    /// 屏幕上只剩下「旧图重新投影」，对不对一眼就能看出来（墙边、门框还竖不竖直）。
    private let genFreeze = UserDefaults.standard.double(forKey: "genFreeze")
    private let startedAt = CACurrentMediaTime()
    /// `-sitTest 0~1`：把人物固定在坐下动作的这个位置，截图检查用。不给就按实际情况。
    private let sitTest: Float = UserDefaults.standard.object(forKey: "sitTest") == nil
        ? -1 : UserDefaults.standard.float(forKey: "sitTest")
    /// 「生成式画面」开着时，自己画的那一帧先画到这里（不直接上屏），再决定屏幕上显示它还是重画后的图。
    private var rawTexture: MTLTexture?
    /// 要发出去重画的那一帧的拷贝：等回读的时候下一帧已经在往 rawTexture 上画了，不能直接读它。
    private var rawCopy: MTLTexture?
    private let depthWrite: MTLDepthStencilState
    private let depthTestOnly: MTLDepthStencilState
    private let depthOff: MTLDepthStencilState
    private let sampler: MTLSamplerState
    private let envSampler: MTLSamplerState
    /// 还没有任何关键帧时绑上去占位的 1×1 贴图。
    private let dummyFrames: MTLTexture
    private var accumTexture: MTLTexture?
    private var accumZTexture: MTLTexture?
    private var idTexture: MTLTexture?
    private var depthTexture: MTLTexture?
    /// `-noTex YES` 关掉投影贴图、`-noGuess YES` 关掉未扫区域预测，做前后对比用。
    private let texturingEnabled = !UserDefaults.standard.bool(forKey: "noTex")
    private let guessEnabled = !UserDefaults.standard.bool(forKey: "noGuess")
    /// `-noSurfel YES`：圆片不贴着表面，全部朝着镜头画（以前的画法），做前后对比用。
    private let surfelsEnabled = !UserDefaults.standard.bool(forKey: "noSurfel")
    /// `-showNormals YES`：把每个圆片的朝向当颜色画出来，检查法线用。
    private let showNormals = UserDefaults.standard.bool(forKey: "showNormals")
    /// `-noRoom YES`：没扫到的地方退回用一个球接住视线（以前的做法），做前后对比用。
    private let roomEnabled = !UserDefaults.standard.bool(forKey: "noRoom")
    /// `-noFill YES` 关掉点状表面补洞。
    private let fillEnabled = !UserDefaults.standard.bool(forKey: "noFill")
    private var screenTexture: MTLTexture?
    private let instanceBuffers: [MTLBuffer]
    private let maxInstances = 256
    private var frameIndex = 0

    private let solver = AvatarSolver()
    /// 现成的人物模型；包里没有或者 `-capsule YES` 时是 nil，用原来的圆棒人偶。
    private let character: CharacterModel?
    /// 手机的模型（带开合的 iPhone Duo）；没有的话画一个黑盒子。
    private let phoneModel: CharacterModel?

    // 跟拍相机：在人的后上方，朝向慢慢跟着人转；手指可以绕着看、捏合拉远近。
    static let defaultPitch: Float = 0.40
    static let defaultDistance: Float = 3.6
    /// 交给网络画的图比屏幕多画一圈（视角放大这么多倍）：镜头一动，原来在画面外的那一圈就要露出来。
    static let overscan: Float = 1.2
    private var orbitYaw: Float = 0
    private var lastOrbitYaw: Float = 0
    private var orbitRate: Float = 0
    private var fastDrag = false
    private var lastCameraNow: Mat4?
    private var camMotion: Float = 0
    private var trustSmooth: Float = 0
    private var orbitPitch = Renderer.defaultPitch
    private var distance = Renderer.defaultDistance
    private var camHeading = Vec3(0, 0, -1)
    private var camTarget: Vec3?
    private var lastTime = CACurrentMediaTime()
    private var fpsFrames = 0
    private var fpsStart = CACurrentMediaTime()
    /// `-shotEvery 3`：每 3 秒把渲染出来的一帧存到 Documents/shot-N.jpg（循环 6 张）。
    /// 真机上没法从电脑截屏，靠这个把画面拿回来看。
    private let shotInterval = Renderer.shotIntervalSetting
    static var shotIntervalSetting: Double {
        let v = UserDefaults.standard.double(forKey: "shotEvery")
        #if DEBUG && !targetEnvironment(simulator)
            return v > 0 ? v : 5
        #else
            return v
        #endif
    }
    private var lastShot = CACurrentMediaTime()
    private var shotIndex = 0

    static let background = Vec3(0.045, 0.055, 0.075)

    init?(model: AppModel) {
        self.model = model
        // 下面的局部函数不能在 super.init 之前碰 self，所以用局部变量。
        let device = model.device
        self.device = device
        guard let q = device.makeCommandQueue(), let lib = device.makeDefaultLibrary(),
            let meshes = Meshes(device: device)
        else { return nil }
        commandQueue = q
        self.meshes = meshes

        func makePipeline(
            _ vs: String, _ fs: String, _ format: MTLPixelFormat,
            _ configure: (MTLRenderPipelineDescriptor) -> Void = { _ in }
        ) -> MTLRenderPipelineState? {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = lib.makeFunction(name: vs)
            d.fragmentFunction = lib.makeFunction(name: fs)
            d.colorAttachments[0].pixelFormat = format
            d.depthAttachmentPixelFormat = .depth32Float
            configure(d)
            do {
                return try device.makeRenderPipelineState(descriptor: d)
            } catch {
                print("pipeline \(vs)/\(fs) 失败：\(error)")
                return nil
            }
        }

        guard
            let p1 = makePipeline("splat_vs", "splat_depth_fs", .rgba16Float, { d in
                d.colorAttachments[1].pixelFormat = .r32Float
                d.colorAttachments[2].pixelFormat = .r32Float
                for i in 0..<3 { d.colorAttachments[i].writeMask = [] }
            }),
            let p2 = makePipeline("splat_vs", "splat_blend_fs", .rgba16Float, { d in
                d.colorAttachments[1].pixelFormat = .r32Float
                d.colorAttachments[2].pixelFormat = .r32Float
                // 目标 0、1 相加；目标 2 取最大值。
                for i in 0..<3 {
                    let a = d.colorAttachments[i]!
                    a.isBlendingEnabled = true
                    a.sourceRGBBlendFactor = .one
                    a.destinationRGBBlendFactor = .one
                    a.sourceAlphaBlendFactor = .one
                    a.destinationAlphaBlendFactor = .one
                    if i == 2 {
                        a.rgbBlendOperation = .max
                        a.alphaBlendOperation = .max
                    }
                }
            }),
            let p3 = makePipeline("fullscreen_vs", "resolve_fs", .bgra8Unorm),
            let p4 = makePipeline("mesh_vs", "mesh_fs", .bgra8Unorm, { d in
                d.vertexDescriptor = Meshes.vertexDescriptor
                let a = d.colorAttachments[0]!
                a.isBlendingEnabled = true
                a.sourceRGBBlendFactor = .sourceAlpha
                a.destinationRGBBlendFactor = .oneMinusSourceAlpha
                a.sourceAlphaBlendFactor = .one
                a.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            }),
            let p5 = makePipeline("fullscreen_vs", "avatar_fs", .bgra8Unorm),
            let p6 = makePipeline("fullscreen_vs", "pano_fs", .rgba8Unorm, { d in
                d.colorAttachments[1].pixelFormat = .r32Float
            }),
            let p7 = makePipeline("fullscreen_vs", "blit_fs", .bgra8Unorm),
            let p8 = makePipeline("character_vs", "mesh_fs", .bgra8Unorm),
            let p9 = makePipeline("genmesh_vs", "genmesh_fs", .bgra8Unorm, { d in
                let a = d.colorAttachments[0]!
                a.isBlendingEnabled = true
                a.sourceRGBBlendFactor = .sourceAlpha
                a.destinationRGBBlendFactor = .oneMinusSourceAlpha
                a.sourceAlphaBlendFactor = .one
                a.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            })
        else { return nil }
        genMeshPipeline = p9
        genShell = GenShell(device: device, library: lib)
        genMeshIndices = GenMesh.indexBuffer(device)
        panoPipeline = p6
        blitPipeline = p7
        characterPipeline = p8
        splatDepthPipeline = p1
        splatBlendPipeline = p2
        resolvePipeline = p3
        meshPipeline = p4
        avatarPipeline = p5

        func makeDepth(_ compare: MTLCompareFunction, _ write: Bool) -> MTLDepthStencilState? {
            let d = MTLDepthStencilDescriptor()
            d.depthCompareFunction = compare
            d.isDepthWriteEnabled = write
            return device.makeDepthStencilState(descriptor: d)
        }
        let sd = MTLSamplerDescriptor()
        sd.minFilter = .linear
        sd.magFilter = .linear
        let esd = MTLSamplerDescriptor()
        esd.minFilter = .linear
        esd.magFilter = .linear
        esd.mipFilter = .linear
        esd.sAddressMode = .repeat
        let dfd = MTLTextureDescriptor()
        dfd.textureType = .type2DArray
        dfd.pixelFormat = .rgba8Unorm
        dfd.width = 1
        dfd.height = 1
        dfd.arrayLength = 1
        dfd.usage = [.shaderRead]
        guard let d1 = makeDepth(.less, true), let d2 = makeDepth(.lessEqual, false),
            let d3 = makeDepth(.always, false), let smp = device.makeSamplerState(descriptor: sd),
            let esmp = device.makeSamplerState(descriptor: esd),
            let dummy = device.makeTexture(descriptor: dfd)
        else { return nil }
        depthWrite = d1
        depthTestOnly = d2
        depthOff = d3
        sampler = smp
        envSampler = esmp
        dummyFrames = dummy

        var bufs: [MTLBuffer] = []
        for _ in 0..<3 {
            guard
                let b = device.makeBuffer(
                    length: maxInstances * MemoryLayout<MeshInstance>.stride,
                    options: .storageModeShared)
            else { return nil }
            bufs.append(b)
        }
        instanceBuffers = bufs
        character = CharacterModel(device: device)
        if let character {
            solver.strideFraction = character.stride / character.height
            solver.sitBackFraction = character.sitBack / character.height
        }
        phoneModel = CharacterModel(device: device, name: "phone")
        super.init()
        model.predictor?.panorama = { [weak self] center, done in
            self?.requestPanorama(center: center, completion: done) ?? false
        }

        // 调试用：`-yaw 1.2 -pitch 1.3 -dist 8` 固定一个机位，方便截图对比。
        let d = UserDefaults.standard
        if d.object(forKey: "yaw") != nil { orbitYaw = d.float(forKey: "yaw") }
        if d.object(forKey: "pitch") != nil { orbitPitch = d.float(forKey: "pitch") }
        if d.object(forKey: "dist") != nil { distance = d.float(forKey: "dist") }
    }

    // MARK: splat 遍

    /// 把地图（和实时层、预测层）画进一组累积目标。屏幕画面和全景图的每个面都走这里。
    private func encodeSplats(
        _ cmd: MTLCommandBuffer, accum: MTLTexture, accumZ: MTLTexture, id: MTLTexture,
        depth: MTLTexture, view: Mat4, proj: Mat4, nearClip: Float, phonePos: Vec3,
        snap: SplatMap.Snapshot, predicted: Bool, roofCut: Float = .greatestFiniteMagnitude,
        live: Bool = true
    ) {
        let texturing = texturingEnabled && snap.texture != nil
        let bg = Renderer.background
        let eyePos = view.inverse.position
        enum Layer { case map, live, predicted }
        func uniforms(push: Float, _ layer: Layer) -> SplatUniforms {
            SplatUniforms(
                view: view, proj: proj,
                phonePos: SIMD4(phonePos.x, phonePos.y, phonePos.z, 1),
                fog: SIMD4(bg.x, bg.y, bg.z, 7),
                params: SIMD4(
                    SplatMap.voxel * 0.95, nearClip, push,
                    layer == .live ? model.livePixelAngle * 1.15 : 0),
                params2: SIMD4(
                    layer == .map ? SplatMap.minHitsToDraw : (layer == .predicted ? 1 : 0), 16,
                    layer == .predicted ? 1 : 0, roofCut),
                live: SIMD4(
                    Float(snap.liveSlot), Float(snap.liveStamp), texturing ? 1 : 0,
                    snap.liveValid ? 1 : 0),
                eye: SIMD4(eyePos.x, eyePos.y, eyePos.z, showNormals ? 2 : (surfelsEnabled ? 1 : 0)))
        }

        let pass = MTLRenderPassDescriptor()
        for (i, tex) in [accum, accumZ, id].enumerated() {
            pass.colorAttachments[i].texture = tex
            pass.colorAttachments[i].loadAction = .clear
            pass.colorAttachments[i].clearColor = MTLClearColorMake(0, 0, 0, 0)
            pass.colorAttachments[i].storeAction = .store
        }
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        pass.depthAttachment.storeAction = .store
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.setCullMode(.none)
        let keyframeBytes = snap.keyframes.count * MemoryLayout<KeyframeGPU>.stride
        snap.keyframes.withUnsafeBytes { raw in
            enc.setVertexBytes(raw.baseAddress!, length: keyframeBytes, index: 2)
        }
        let showLive = live && model.showLive && snap.liveCount > 0
        let showPredicted = predicted && guessEnabled && snap.predictedCount > 0
        func draw(_ buffer: MTLBuffer, _ count: Int, _ layer: Layer, push: Float) {
            var u = uniforms(push: push, layer)
            enc.setVertexBuffer(buffer, offset: 0, index: 0)
            enc.setVertexBytes(&u, length: MemoryLayout<SplatUniforms>.stride, index: 1)
            enc.drawPrimitives(
                type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: count)
        }
        func drawAll(push: Float) {
            if snap.count > 0 { draw(model.map.buffer, snap.count, .map, push: push) }
            if showLive { draw(snap.live, snap.liveCount, .live, push: push) }
            if showPredicted { draw(snap.predicted, snap.predictedCount, .predicted, push: push) }
        }
        enc.setRenderPipelineState(splatDepthPipeline)
        enc.setDepthStencilState(depthWrite)
        drawAll(push: 2.5)
        enc.setRenderPipelineState(splatBlendPipeline)
        enc.setDepthStencilState(depthTestOnly)
        drawAll(push: 0)
        enc.endEncoding()
    }

    private func bindSurfaceInputs(
        _ enc: MTLRenderCommandEncoder, accum: MTLTexture, accumZ: MTLTexture, id: MTLTexture,
        snap: SplatMap.Snapshot, uniforms: inout ResolveUniforms
    ) {
        enc.setFragmentTexture(accum, index: 0)
        enc.setFragmentTexture(accumZ, index: 1)
        enc.setFragmentTexture(id, index: 2)
        enc.setFragmentTexture(snap.texture ?? dummyFrames, index: 3)
        enc.setFragmentTexture(model.map.envTexture, index: 4)
        enc.setFragmentTexture(model.predictor?.texture ?? model.map.envTexture, index: 5)
        enc.setFragmentTexture(snap.depthTexture ?? dummyFrames, index: 6)
        enc.setFragmentTexture(model.predictor?.roomTexture ?? model.map.envTexture, index: 7)
        enc.setFragmentTexture(model.map.floorTexture, index: 8)
        enc.setFragmentSamplerState(sampler, index: 0)
        enc.setFragmentSamplerState(envSampler, index: 1)
        enc.setFragmentBytes(&uniforms, length: MemoryLayout<ResolveUniforms>.stride, index: 0)
        let keyframeBytes = snap.keyframes.count * MemoryLayout<KeyframeGPU>.stride
        snap.keyframes.withUnsafeBytes { raw in
            enc.setFragmentBytes(raw.baseAddress!, length: keyframeBytes, index: 1)
        }
    }

    // MARK: 全景图

    /// 全景图分六个面渲，每帧只渲一个面，摊到六帧里，不让画面卡一下。
    private static let panoFaceSize = 640
    private static let panoWidth = 1536
    private static let panoHeight = 768
    private static let panoFaces: [(dir: Vec3, up: Vec3)] = [
        (Vec3(1, 0, 0), Vec3(0, 1, 0)), (Vec3(-1, 0, 0), Vec3(0, 1, 0)),
        (Vec3(0, 1, 0), Vec3(0, 0, 1)), (Vec3(0, -1, 0), Vec3(0, 0, -1)),
        (Vec3(0, 0, 1), Vec3(0, 1, 0)), (Vec3(0, 0, -1), Vec3(0, 1, 0)),
    ]

    private struct PanoJob {
        var center: Vec3
        var completion: (SplatMap.Panorama) -> Void
        var nextFace = 0
    }

    private let panoLock = NSLock()
    private var panoJob: PanoJob?
    private var panoAccum: [MTLTexture] = []  // 累积颜色、累积深度、槽位号、深度缓冲
    private var panoColor: [MTLTexture] = []
    private var panoDist: [MTLTexture] = []
    private let panoQueue = DispatchQueue(label: "outofbody.pano", qos: .utility)

    /// 以 center 为中心把地图渲成一张等距柱状全景图。用的是屏幕画面那套管线（投影贴图、补洞），
    /// 所以发给生成模型的图和用户看到的一样清楚。预测出来的表面不画进去：它们算「没扫到」。
    /// 结果在后台线程上回调。上一张还没渲完时再请求会被忽略（返回 false）。
    @discardableResult
    func requestPanorama(center: Vec3, completion: @escaping (SplatMap.Panorama) -> Void) -> Bool {
        panoLock.lock()
        defer { panoLock.unlock() }
        if panoJob != nil { return false }
        panoJob = PanoJob(center: center, completion: completion)
        return true
    }

    private func nextPanoFace() -> (job: PanoJob, face: Int)? {
        panoLock.lock()
        defer { panoLock.unlock() }
        guard var job = panoJob, job.nextFace < 6 else { return nil }
        let face = job.nextFace
        job.nextFace += 1
        panoJob = job
        return (job, face)
    }

    private static func panoViewProj(center: Vec3, face: Int) -> (view: Mat4, proj: Mat4) {
        let f = panoFaces[face]
        return (
            Mat4.lookAt(eye: center, target: center + f.dir, up: f.up),
            Mat4.perspective(fovY: .pi / 2, aspect: 1, near: 0.05, far: 80)
        )
    }

    private func ensurePanoTargets() -> Bool {
        if panoColor.count == 6 { return true }
        let s = Renderer.panoFaceSize
        func make(_ format: MTLPixelFormat, shared: Bool) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: s, height: s, mipmapped: false)
            d.usage = format == .depth32Float ? [.renderTarget] : [.renderTarget, .shaderRead]
            d.storageMode = shared ? .shared : .private
            return device.makeTexture(descriptor: d)
        }
        guard let a = make(.rgba16Float, shared: false), let z = make(.r32Float, shared: false),
            let i = make(.r32Float, shared: false), let dp = make(.depth32Float, shared: false)
        else { return false }
        var colors: [MTLTexture] = []
        var dists: [MTLTexture] = []
        for _ in 0..<6 {
            guard let c = make(.rgba8Unorm, shared: true), let d = make(.r32Float, shared: true)
            else { return false }
            colors.append(c)
            dists.append(d)
        }
        panoAccum = [a, z, i, dp]
        panoColor = colors
        panoDist = dists
        return true
    }

    private func renderPanoFace(
        _ cmd: MTLCommandBuffer, _ item: (job: PanoJob, face: Int), snap: SplatMap.Snapshot
    ) {
        guard ensurePanoTargets() else { return }
        let center = item.job.center
        let vp = Renderer.panoViewProj(center: center, face: item.face)
        // 离中心 30 cm 以内的不画：那是用户自己手边的东西，会把整个面糊住。
        encodeSplats(
            cmd, accum: panoAccum[0], accumZ: panoAccum[1], id: panoAccum[2], depth: panoAccum[3],
            view: vp.view, proj: vp.proj, nearClip: 0.3, phonePos: center, snap: snap,
            predicted: false)

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = panoColor[item.face]
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[1].texture = panoDist[item.face]
        pass.colorAttachments[1].loadAction = .dontCare
        pass.colorAttachments[1].storeAction = .store
        pass.depthAttachment.texture = panoAccum[3]
        pass.depthAttachment.loadAction = .load
        pass.depthAttachment.storeAction = .dontCare
        if let enc = cmd.makeRenderCommandEncoder(descriptor: pass) {
            let dir = Renderer.panoFaces[item.face].dir
            let bg = Renderer.background
            var ru = ResolveUniforms(
                invViewProj: (vp.proj * vp.view).inverse,
                cameraPos: SIMD4(center.x, center.y, center.z, 1),
                cameraFwd: SIMD4(dir.x, dir.y, dir.z, 0),
                bg: SIMD4(bg.x, bg.y, bg.z, 1), fog: SIMD4(bg.x, bg.y, bg.z, 7),
                params: SIMD4(
                    0, fillEnabled ? 1 : 0, 16, texturingEnabled && snap.texture != nil ? 1 : 0),
                misc: .zero, envCenter: .zero, genCenter: .zero, room: .zero, floorMap: .zero)
            enc.setRenderPipelineState(panoPipeline)
            enc.setDepthStencilState(depthOff)
            bindSurfaceInputs(
                enc, accum: panoAccum[0], accumZ: panoAccum[1], id: panoAccum[2], snap: snap,
                uniforms: &ru)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
        }

        guard item.face == 5 else { return }
        // 六个面都排进去了。等 GPU 画完，读回来拼成全景图。
        let colors = panoColor, dists = panoDist
        let job = item.job
        cmd.addCompletedHandler { [weak self] _ in
            guard let self else { return }
            self.panoQueue.async {
                let pano = Renderer.assemblePanorama(center: job.center, colors: colors, dists: dists)
                self.panoLock.lock()
                self.panoJob = nil
                self.panoLock.unlock()
                job.completion(pano)
            }
        }
    }

    private static func assemblePanorama(center: Vec3, colors: [MTLTexture], dists: [MTLTexture])
        -> SplatMap.Panorama
    {
        let s = panoFaceSize
        var faceColor = [[UInt8]](repeating: [UInt8](repeating: 0, count: s * s * 4), count: 6)
        var faceDist = [[Float]](repeating: [Float](repeating: 0, count: s * s), count: 6)
        let region = MTLRegionMake2D(0, 0, s, s)
        for f in 0..<6 {
            colors[f].getBytes(&faceColor[f], bytesPerRow: s * 4, from: region, mipmapLevel: 0)
            dists[f].getBytes(&faceDist[f], bytesPerRow: s * 4, from: region, mipmapLevel: 0)
        }
        let viewProjs = (0..<6).map { f -> Mat4 in
            let vp = panoViewProj(center: center, face: f)
            return vp.proj * vp.view
        }
        let w = panoWidth, h = panoHeight
        var rgba = [UInt8](repeating: 128, count: w * h * 4)
        var distance = [Float](repeating: 0, count: w * h)
        var covered = 0
        for y in 0..<h {
            for x in 0..<w {
                let d = SplatMap.panoDirection(
                    x: Float(x) + 0.5, y: Float(y) + 0.5, width: w, height: h)
                let ax = abs(d.x), ay = abs(d.y), az = abs(d.z)
                let face: Int
                if ax >= ay, ax >= az {
                    face = d.x > 0 ? 0 : 1
                } else if ay >= az {
                    face = d.y > 0 ? 2 : 3
                } else {
                    face = d.z > 0 ? 4 : 5
                }
                let p = center + d
                let clip = viewProjs[face] * SIMD4(p.x, p.y, p.z, 1)
                let fx = min(s - 1, max(0, Int((clip.x / clip.w * 0.5 + 0.5) * Float(s))))
                let fy = min(s - 1, max(0, Int((1 - (clip.y / clip.w * 0.5 + 0.5)) * Float(s))))
                let j = (fy * s + fx) * 4
                let i = y * w + x
                if faceColor[face][j + 3] > 127 {
                    rgba[i * 4] = faceColor[face][j]
                    rgba[i * 4 + 1] = faceColor[face][j + 1]
                    rgba[i * 4 + 2] = faceColor[face][j + 2]
                    rgba[i * 4 + 3] = 255
                    distance[i] = faceDist[face][fy * s + fx]
                    covered += 1
                } else {
                    rgba[i * 4 + 3] = 0
                }
            }
        }
        return SplatMap.Panorama(
            width: w, height: h, rgba: rgba, distance: distance, center: center,
            coverage: Float(covered) / Float(w * h))
    }

    // MARK: 导出训练素材

    private var exportTargets: [String: [MTLTexture]] = [:]

    /// 从指定机位把地图渲一张（只要扫到的表面，不要背景和人偶），读回颜色（RGBA）和每个像素到机位的距离。
    /// 攒训练素材和本机小网络的输入都走这里，所以两边看到的图是同一种。
    /// slot 区分不同用途各自的一套离屏贴图，免得互相覆盖。
    private func renderSurfaces(
        _ cmd: MTLCommandBuffer, slot: String, view: Mat4, proj: Mat4, width w: Int, height h: Int,
        nearClip: Float = 0.05, roofCut: Float = .greatestFiniteMagnitude, live: Bool,
        floorHint: (floorY: Float, center: Vec3)? = nil,
        snap: SplatMap.Snapshot, completion: @escaping ([UInt8], [Float]) -> Void
    ) {
        let key = "\(slot)\(w)x\(h)"
        if exportTargets[key] == nil {
            func make(_ format: MTLPixelFormat, shared: Bool) -> MTLTexture? {
                let d = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: format, width: w, height: h, mipmapped: false)
                d.usage = format == .depth32Float ? [.renderTarget] : [.renderTarget, .shaderRead]
                d.storageMode = shared ? .shared : .private
                return device.makeTexture(descriptor: d)
            }
            guard let a = make(.rgba16Float, shared: false), let z = make(.r32Float, shared: false),
                let i = make(.r32Float, shared: false), let dp = make(.depth32Float, shared: false),
                let c = make(.rgba8Unorm, shared: true), let ds = make(.r32Float, shared: true)
            else { return }
            exportTargets[key] = [a, z, i, dp, c, ds]
        }
        guard let t = exportTargets[key] else { return }
        let camera = view.inverse
        let eye = camera.position
        encodeSplats(
            cmd, accum: t[0], accumZ: t[1], id: t[2], depth: t[3], view: view, proj: proj,
            nearClip: nearClip, phonePos: eye, snap: snap, predicted: false, roofCut: roofCut,
            live: live)

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = t[4]
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[1].texture = t[5]
        pass.colorAttachments[1].loadAction = .dontCare
        pass.colorAttachments[1].storeAction = .store
        pass.depthAttachment.texture = t[3]
        pass.depthAttachment.loadAction = .load
        pass.depthAttachment.storeAction = .dontCare
        if let enc = cmd.makeRenderCommandEncoder(descriptor: pass) {
            let fwd = -camera.zAxis
            let bg = Renderer.background
            var ru = ResolveUniforms(
                invViewProj: (proj * view).inverse,
                cameraPos: SIMD4(eye.x, eye.y, eye.z, 1),
                cameraFwd: SIMD4(fwd.x, fwd.y, fwd.z, 0),
                bg: SIMD4(bg.x, bg.y, bg.z, 1), fog: SIMD4(bg.x, bg.y, bg.z, 7),
                params: SIMD4(
                    floorHint?.floorY ?? 0, fillEnabled ? 1 : 0, 16,
                    texturingEnabled && snap.texture != nil ? 1 : 0),
                misc: .zero,
                envCenter: floorHint.map { SIMD4($0.center.x, $0.center.y, $0.center.z, 0) } ?? .zero,
                genCenter: .zero, room: .zero,
                floorMap: floorHint == nil ? .zero : snap.floorInfo)
            enc.setRenderPipelineState(panoPipeline)
            enc.setDepthStencilState(depthOff)
            bindSurfaceInputs(enc, accum: t[0], accumZ: t[1], id: t[2], snap: snap, uniforms: &ru)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
        }
        let color = t[4], distTex = t[5]
        cmd.addCompletedHandler { _ in
            var rgba = [UInt8](repeating: 0, count: w * h * 4)
            var dist = [Float](repeating: 0, count: w * h)
            let region = MTLRegionMake2D(0, 0, w, h)
            color.getBytes(&rgba, bytesPerRow: w * 4, from: region, mipmapLevel: 0)
            distTex.getBytes(&dist, bytesPerRow: w * 4, from: region, mipmapLevel: 0)
            completion(rgba, dist)
        }
    }

    // MARK: 手势

    @objc func handlePan(_ g: UIPanGestureRecognizer) {
        let t = g.translation(in: g.view)
        g.setTranslation(.zero, in: g.view)
        orbitYaw += Float(t.x) * 0.008
        orbitPitch = min(1.35, max(0.05, orbitPitch + Float(t.y) * 0.005))
    }

    @objc func handlePinch(_ g: UIPinchGestureRecognizer) {
        distance = min(12, max(0.9, distance / Float(g.scale)))
        g.scale = 1
    }

    @objc func handleDoubleTap(_ g: UITapGestureRecognizer) {
        orbitYaw = 0
        orbitPitch = Renderer.defaultPitch
        distance = Renderer.defaultDistance
    }

    // MARK: 渲染

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    private func ensureTargets(_ w: Int, _ h: Int) {
        if let a = accumTexture, a.width == w, a.height == h { return }
        let ad = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: w, height: h, mipmapped: false)
        ad.usage = [.renderTarget, .shaderRead]
        ad.storageMode = .private
        accumTexture = device.makeTexture(descriptor: ad)
        let fd = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: w, height: h, mipmapped: false)
        fd.usage = [.renderTarget, .shaderRead]
        fd.storageMode = .private
        accumZTexture = device.makeTexture(descriptor: fd)
        idTexture = device.makeTexture(descriptor: fd)
        let dd = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: w, height: h, mipmapped: false)
        dd.usage = [.renderTarget]
        dd.storageMode = .private
        depthTexture = device.makeTexture(descriptor: dd)
        let rd = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
        rd.usage = [.renderTarget, .shaderRead]
        rd.storageMode = .private
        rawTexture = device.makeTexture(descriptor: rd)
        rd.storageMode = .shared
        rawCopy = device.makeTexture(descriptor: rd)
    }

    private func updateScreenTexture() {
        guard let img = model.takeScreenImage() else { return }
        if screenTexture == nil || screenTexture!.width != img.width
            || screenTexture!.height != img.height
        {
            let d = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba8Unorm, width: img.width, height: img.height, mipmapped: false)
            d.usage = [.shaderRead]
            screenTexture = device.makeTexture(descriptor: d)
        }
        img.rgba.withUnsafeBytes { raw in
            screenTexture?.replace(
                region: MTLRegionMake2D(0, 0, img.width, img.height), mipmapLevel: 0,
                withBytes: raw.baseAddress!, bytesPerRow: img.width * 4)
        }
    }

    /// 要跟 Shaders.metal 里的 AVATAR_LIMBS 一致。
    static let maxLimbs = 24

    /// 窗口有多宽（点）。系统不给铰链状态时靠它猜手机是合着还是展开：Duo 合上 466，展开 669。
    private var viewWidthPoints: CGFloat = 0

    func draw(in view: MTKView) {
        viewWidthPoints = view.bounds.width
        let size = view.drawableSize
        guard size.width > 1, size.height > 1 else { return }
        ensureTargets(Int(size.width), Int(size.height))
        guard let accum = accumTexture, let accumZ = accumZTexture, let idTex = idTexture,
            let depthTex = depthTexture,
            let drawable = view.currentDrawable, let cmd = commandQueue.makeCommandBuffer()
        else { return }

        let now = CACurrentMediaTime()
        let dt = Float(min(0.1, max(0.001, now - lastTime)))
        lastTime = now
        fpsFrames += 1
        if now - fpsStart >= 1 {
            model.fps = Int((Double(fpsFrames) / (now - fpsStart)).rounded())
            fpsFrames = 0
            fpsStart = now
        }

        let pose = model.currentPose()
        let avatar = pose.map { solver.update(pose: $0, dt: dt) }
        model.floorY = avatar?.floorY
        updateScreenTexture()

        // ---- 跟拍相机 ----
        let up = Vec3(0, 1, 0)
        // 用静态人物模型时，镜头也跟那尊「站着不动的像」，不跟着手的小动作晃。
        let still = character != nil
        let heading = (still ? avatar?.stillHeading : avatar?.heading) ?? Vec3(0, 0, -1)
        camHeading = (camHeading + (heading - camHeading) * lowpass(dt, 0.7)).normalized(or: heading)
        // 镜头跟着人转身的角速度（和手指拖的 orbitYaw 同一个转向约定），预判机位时两个加在一起。
        if dt > 1e-4 {
            let a = lastCamHeading, b = camHeading
            let turned = atan2(a.x * b.z - a.z * b.x, a.x * b.x + a.z * b.z)
            headingRate += (min(4, max(-4, turned / dt)) - headingRate) * lowpass(dt, 0.15)
        }
        lastCamHeading = camHeading
        let targetNow = avatar.map {
            still ? $0.stillCenter + up * ($0.standingHeight * (0.66 - 0.16 * $0.seatAmount)) : $0.chest
        } ?? Vec3(0, 1.1, 0)
        let target = camTarget.map { $0 + (targetNow - $0) * lowpass(dt, 0.12) } ?? targetNow
        camTarget = target
        if yawSpin != 0 { orbitYaw += yawSpin * dt }
        let frozen = genFreeze > 0 && now - startedAt > genFreeze
        if yawSwing != 0, genFreeze <= 0 || frozen {
            orbitYaw = yawSwing * sin(Float((now - startedAt).truncatingRemainder(dividingBy: 2.4)) / 2.4 * 2 * .pi)
        }
        if yawFlick != 0 {
            let t = Float(now - startedAt) - 9
            if t > 0 {
                let k = floor(t / 4)
                orbitYaw = yawFlick * (k + smoothstep(0, 0.4, t - k * 4))
            }
        }
        // 手指拖着转的角速度（弧度每秒）。给「生成」那档预判镜头用：见下面 leadView。
        if dt > 1e-4 {
            let rate = min(6, max(-6, (orbitYaw - lastOrbitYaw) / dt))
            orbitRate += (rate - orbitRate) * lowpass(dt, 0.08)
        }
        lastOrbitYaw = orbitYaw
        let distanceNow = distance, pitchNow = orbitPitch, headingNow = camHeading
        func chase(yaw: Float) -> (eye: Vec3, view: Mat4) {
            let b = -headingNow
            let c = cos(yaw), s = sin(yaw)
            let back = Vec3(b.x * c - b.z * s, 0, b.x * s + b.z * c)
            let eye = target + back * (cos(pitchNow) * distanceNow) + up * (sin(pitchNow) * distanceNow)
            // 往人前方看一点：人落在画面下半部，前面的路留得多。
            return (eye, Mat4.lookAt(eye: eye, target: target - back * 1.1 + up * 0.15, up: up))
        }
        let (eye, viewM) = chase(yaw: orbitYaw)
        let back = (eye - target).horizontal.normalized(or: -headingNow)
        let projM = Mat4.perspective(
            fovY: 60 * .pi / 180, aspect: Float(size.width / size.height), near: 0.05, far: 80)
        let viewProj = projM * viewM

        let snap = model.map.snapshot
        let texturing = texturingEnabled && snap.texture != nil
        let bg = Renderer.background
        let phonePos = pose?.cameraToWorld.position ?? Vec3(0, 0, 0)
        // 相机和人之间的东西不画，否则在小房间里相机一退到墙后就什么都看不见了。
        let nearClip = max(0.05, simd_distance(eye, target) - nearMargin)

        if let face = nextPanoFace() { renderPanoFace(cmd, face, snap: snap) }
        if let job = model.exporter.next() {
            let exporter = model.exporter
            renderSurfaces(
                cmd, slot: "dataset", view: job.view, proj: job.proj, width: job.width,
                height: job.height, nearClip: job.nearClip, roofCut: job.roofCut, live: false,
                snap: snap
            ) { rgba, dist in exporter.finish(job, rgba: rgba, dist: dist) }
        }

        // ---- 第一遍：splat 累积到浮点目标 ----
        // 相机升到人头顶上方往下看时，把高处的东西（天花板、门楣）掀掉，不然只能看到屋顶。
        let headY = avatar?.head.y ?? target.y + 0.5
        let roofCut: Float = eye.y > headY + 1.0 ? headY + 0.7 : .greatestFiniteMagnitude
        encodeSplats(
            cmd, accum: accum, accumZ: accumZ, id: idTex, depth: depthTex, view: viewM,
            proj: projM, nearClip: nearClip, phonePos: phonePos, snap: snap, predicted: true,
            roofCut: roofCut)

        // 「最后那张图让网络来画」的两档——服务器上微调过的扩散模型，或者手机上它的学生网络。
        // 做法一样：空闲时交一张只有表面的小图，拿回来的图连同当时的机位一起记着，
        // resolve 里把它当成立在人偶距离上的一幅画重新投影（见着色器里的说明）。
        // 比屏幕看到的范围多画一圈（视角放大 12%）：镜头一动，原来在画面外的那一圈就要露出来。
        let wide = Mat4.perspective(
            fovY: 2 * atan(tan(30 * Float.pi / 180) * Renderer.overscan),
            aspect: Float(size.width / size.height), near: 0.05, far: 80)
        let wideViewProj = wide * viewM
        let cameraNow = viewM.inverse
        let tuned = model.neuralTuned
        // 服务器那档从交图到拿回来要 0.15 秒左右，一张图还要顶到下一张回来。手指拖着转的时候，
        // 按现在的机位去画，画回来镜头早转走了。所以交的是「按现在的转速，过这么久之后镜头会在的位置」
        // 看到的画面：匀速拖的时候，一张图显示的那段时间里镜头正好从它的一侧转到另一侧。
        // 转得快的时候再做两件事，让转过去的那一侧少露空：
        //   - 画得更宽（多画的那一圈从 12% 加到 47%）。平时左右各只多 4°，一张图顶 0.1 秒，转快了前沿就没图可用；
        //   - 预判再往前多看一点。身后那一侧刚转过来，有旧图垫着，不怕。
        let spin = smoothstep(0.3, 1.5, abs(orbitRate + headingRate))
        let wideTuned = Mat4.perspective(
            fovY: 2 * atan(tan(30 * Float.pi / 180) * (Renderer.overscan + 0.35 * spin)),
            aspect: Float(size.width / size.height), near: 0.05, far: 80)
        let lead = chase(yaw: orbitYaw + (orbitRate + headingRate) * (tuned.lead + 0.06 * spin))
        // 预先画的那一圈用更宽的视角（多画 45%）：格子可以稀一点，少画几张；反正只是垫底。
        let ringWide = Mat4.perspective(
            fovY: 2 * atan(tan(30 * Float.pi / 180) * 1.45),
            aspect: Float(size.width / size.height), near: 0.05, far: 80)
        // `-noFloorHint YES` 关掉地面提示，做前后对比用。
        let floorHint: (floorY: Float, center: Vec3)? = UserDefaults.standard.bool(forKey: "noFloorHint")
            ? nil : avatar.map { ($0.floorY, still ? $0.stillRoot : $0.feetCenter) }
        if !frozen, tuned.wantsFrame() {
            let tw = NeuralView.tunedSize.width, th = NeuralView.tunedSize.height
            // 镜头不动的时候，每隔一张去画「绕人一圈」里还没画过（或者画了很久）的一个角度，存着不显示。
            // 手指一甩转过去，新图要 0.2 秒才到，这段时间垫在下面的就是这些预先画好的图，而不是点云。
            var shot = (view: lead.view, proj: wideTuned)
            // 「不动」放得宽一点（每秒转 9° 以内都算）：格子是钉在世界里的，人慢慢转身不影响。
            if ringEnabled, camMotion < 0.15, abs(orbitRate + headingRate) < 0.15 {
                ringTick += 1
                do {
                    let here = atan2(back.z, back.x)
                    let keys = tuned.keyframes
                    var best: (age: Double, view: Mat4)?
                    for j in 0..<Renderer.ringSlots {
                        var off = Float(j) * (2 * .pi / Float(Renderer.ringSlots)) - here
                        off -= (off / (2 * .pi)).rounded() * 2 * .pi
                        // 现在对着的这一格不用管，平时的图一直在画它。
                        if abs(off) < 0.7 * 2 * .pi / Float(Renderer.ringSlots) { continue }
                        let c = chase(yaw: orbitYaw + off)
                        let cam = c.view.inverse
                        var newest = -Double.greatestFiniteMagnitude
                        for k in keys where k.replacedAt == nil
                            && simd_dot(k.camera.zAxis, cam.zAxis) > 0.99
                            && simd_distance(k.camera.position, c.eye) < 0.2 * distanceNow {
                            newest = max(newest, k.born)
                        }
                        // 没画过的优先（离现在的角度近的先画）；都画过了就挑最旧的，超过 45 秒才重画。
                        let age = newest < 0 ? 1e6 - Double(abs(off)) : now - newest
                        if age > 45, age > (best?.age ?? 0) { best = (age, c.view) }
                    }
                    // 还有没画过的格子：三张里两张拿去画圈，尽快铺满（十来秒）；只是翻新旧图：四张里一张。
                    if let best, best.age > 1e5 ? ringTick % 3 != 0 : ringTick % 4 == 0 {
                        shot = (best.view, ringWide)
                    }
                }
            }
            let shotViewProj = shot.proj * shot.view, shotCamera = shot.view.inverse
            renderSurfaces(
                cmd, slot: "tuned", view: shot.view, proj: shot.proj, width: tw, height: th,
                nearClip: nearClip, roofCut: roofCut, live: genLive, floorHint: floorHint, snap: snap
            ) { rgba, dist in
                tuned.submit(
                    pixels: rgba, dist: dist, floorY: floorHint?.floorY, width: tw, height: th,
                    viewProj: shotViewProj, camera: shotCamera)
            }
        }
        // 全景壳：一帧量一个方向的形状；新量好的换上；刚到的生成图画进贴图。
        var shellReady = false
        if let shell = genShell, shellEnabled, tuned.enabled, let av = avatar {
            if let job = shell.nextFace(user: av.stillCenter, floorY: av.floorY, now: now) {
                renderSurfaces(
                    cmd, slot: "shell\(job.index)", view: job.view, proj: job.proj,
                    width: GenShell.faceSize, height: GenShell.faceSize, live: false, snap: snap
                ) { _, dist in shell.faceDone(job, dist: dist) }
            }
            shell.commit(cmd, now: now)
            for f in tuned.takeFresh() {
                shell.integrate(
                    cmd, gen: f.texture, dist: f.dist, viewProj: f.viewProj, camera: f.camera,
                    motion: smoothstep(0.08, 0.3, camMotion))
                let stable = shell.stabilized(
                    cmd, gen: f.texture, dist: f.dist, viewProj: f.viewProj, camera: f.camera)
                tuned.insertKey(texture: stable ?? f.texture, mesh: f.mesh, camera: f.camera, viewer: cameraNow, now: now)
            }
            shellReady = shellDraw && shell.ready
        } else {
            if !tuned.enabled || snap.count == 0 { genShell?.reset() }
            for f in tuned.takeFresh() {
                tuned.insertKey(texture: f.texture, mesh: f.mesh, camera: f.camera, viewer: cameraNow, now: now)
            }
        }
        let refiner = model.refiner
        if let refiner, refiner.wantsFrame() {
            renderSurfaces(
                cmd, slot: "refiner", view: viewM, proj: wide, width: LocalRefiner.width,
                height: LocalRefiner.height, nearClip: nearClip, roofCut: roofCut,
                live: true, floorHint: floorHint, snap: snap
            ) { rgba, dist in
                refiner.submit(rgba: rgba, dist: dist, viewProj: wideViewProj, camera: cameraNow)
            }
        }
        // 服务器那档：画好的图立在世界里，按现在的机位挑着用（见 GenKeyframe）。每张用几成：
        // 机位越近用得越多（转开 11° 以内全用，24° 以外不用；位置同理），刚来的 0.18 秒里淡入。
        // 范围收得紧是因为真机上每秒有八九张新图，离得远的旧图用不着，硬用就是重影。
        var layers: [(key: GenKeyframe, alpha: Float, sharpen: Float, feather: Float)] = []
        var under: [(key: GenKeyframe, alpha: Float, off: Float)] = []
        // 镜头自己动得多快（转动的弧度每秒 + 平移相对镜头到人距离的比例每秒）。
        if let last = lastCameraNow, dt > 1e-4 {
            let ang = acos(min(1, max(-1, simd_dot(last.zAxis, cameraNow.zAxis)))) / dt
            let lin = simd_distance(last.position, eye) / dt / max(simd_distance(eye, target), 1)
            camMotion += (min(ang + lin, 5) - camMotion) * lowpass(dt, 0.1)
        }
        lastCameraNow = cameraNow
        // 新图盖住旧图几成：镜头不动时只盖 genBlend 成，下面透出前几张，同一个地方连着几张取了平均
        // （网络每张图编出来的细节都不太一样，整张替换的话细节会一直「沸腾」）；
        // 镜头一动就整张盖：不同机位的几张图对不齐，叠在一起是重影。
        let blendNow = genBlend + (1 - genBlend) * smoothstep(0.08, 0.3, camMotion)
        for k in tuned.keyframes {
            let turn = acos(min(1, max(-1, simd_dot(-k.camera.zAxis, -cameraNow.zAxis))))
            let move = simd_distance(k.camera.position, eye) / max(simd_distance(eye, target), 1)
            // 淡入：镜头不动时 0.18 秒；镜头在动时缩到 0.07 秒，两张对不齐的图叠着的时间越短越好。
            let fresh = smoothstep(0, 0.18 - 0.11 * smoothstep(0.08, 0.3, camMotion), Float(now - k.born))
            let a = (1 - smoothstep(0.2, 0.42, turn)) * (1 - smoothstep(0.15, 0.4, move)) * fresh * blendNow
            if a > 0.01 { layers.append((k, a, genSharpen, genFeather * 0.45)) }
            // 垫底的：凡是还看得到一部分的旧图都算（转开 49° 以内整张用，63° 以外不用）。
            // 位置的门槛跟着放宽：镜头是绕着人转的，转 49° 机位本身就挪开了「到人的距离」的八成多。
            // 每张图是立在世界里的一层皮，从别的角度看，它拍到过的那部分位置还是对的。
            if fresh > 0.99 {
                let valid = (1 - smoothstep(0.85, 1.1, turn)) * (1 - smoothstep(0.85, 1.15, move))
                if valid > 0.01 { under.append((k, valid, turn + move)) }
            }
        }
        // 从远到近一张张不透明地盖上去：最近的在最上面，远的只在近的盖不到的地方（多半是屏幕边上）露出来。
        // 镜头转得快、新图还没到的时候，露出来的是这些旧图，而不是风格完全不同的实时点云画面。
        // 最多垫 6 张；只有最近的那张做锐化，其余的省掉。
        under.sort { $0.off > $1.off }
        let underDraw = under.suffix(6)
        // 边缘淡出的宽度：最近的那张（和上面正常显示的几张）窄一些，它平时盖满屏幕、边在屏幕外；
        // 其余垫底的宽得多，相邻两张之间是一大片渐变而不是一条缝。
        // 垫底的画两遍。第一遍窄边：能盖的地方都盖上，不让点云露出来，但图和图之间是一条条缝；
        // 第二遍宽边：每张图只有中间那一大块是实的，四周慢慢淡出，把第一遍的缝抹成渐变。
        // 只有一张图够得着的地方留着第一遍的样子。
        let softUnder = underDraw.enumerated().map { i, u in
            let nearest = i == underDraw.count - 1
            return (key: u.key, alpha: u.alpha, sharpen: nearest ? genSharpen : Float(0),
                    feather: nearest ? genFeather * 0.45 : genFeather)
        }
        layers.insert(contentsOf: softUnder, at: 0)
        if genFeather > 0.05 {
            layers.insert(contentsOf: underDraw.map {
                (key: $0.key, alpha: $0.alpha, sharpen: Float(0), feather: Float(0.04))
            }, at: 0)
        }
        let layerCover = layers.map(\.alpha).max() ?? 0
        // 手机上的小网络那档还是老办法：只有最新一张，当成平板重新投影。
        let generatedFrame = layers.isEmpty ? refiner?.frame : nil
        let generated = generatedFrame?.texture
        // 那张图是之前的机位画的。镜头只挪了一点时把它重新投影过来没问题；手指拖着转开一大截之后
        // 再硬投影，画面会被拉长变形。所以给它一个可信度，不可信时露出自己实时渲染的画面（转多快都跟得上）：
        //   - 现在的镜头和画图时的镜头差多少：转开 8° 以内全用，超过 22° 完全不用。镜头平移同理。
        //   - 手指拖得快（每秒 60° 以上）的时候干脆一直不用，慢到每秒 30° 以下才恢复。
        //     不这么做的话，快拖时每来一张新图就「生成的 / 实时的」来回跳一次（2026-11-23 截图里一秒跳四五次），
        //     比变形还难受。
        // 可信度本身再平滑一下（约 0.1 秒），切换是淡入淡出而不是硬跳。
        var trustNow: Float = 0
        if let g = generatedFrame {
            let turn = acos(min(1, max(-1, simd_dot(-g.camera.zAxis, -cameraNow.zAxis))))
            let move = simd_distance(g.camera.position, eye) / max(simd_distance(eye, target), 1)
            trustNow = (1 - smoothstep(0.14, 0.38, turn)) * (1 - smoothstep(0.3, 0.8, move))
        }
        if abs(orbitRate) > 1.05 { fastDrag = true } else if abs(orbitRate) < 0.5 { fastDrag = false }
        if fastDrag { trustNow = 0 }
        trustSmooth += (trustNow - trustSmooth) * lowpass(dt, 0.1)
        let generatedTrust = generated == nil ? 0 : trustSmooth

        // ---- 第二遍：resolve 到屏幕，再画人偶和手机 ----
        // 「生成式画面」开着：先画到离屏的 rawTexture，最后再把重画好的图（或原图）贴上屏幕。
        let neural = model.neural
        let neuralOn = neural.enabled && rawTexture != nil && rawCopy != nil
        let passB = MTLRenderPassDescriptor()
        passB.colorAttachments[0].texture = neuralOn ? rawTexture : drawable.texture
        passB.colorAttachments[0].loadAction = .dontCare
        passB.colorAttachments[0].storeAction = .store
        passB.depthAttachment.texture = depthTex
        // 背景换成微调模型画的整张图时，人偶不能再被自己那套点云挡：那些点画面上已经看不到了，
        // 挡出来是一身斑点。这时把深度清掉，人偶整个画在上面。
        passB.depthAttachment.loadAction = shellReady || max(generatedTrust, layerCover) > 0.5 ? .clear : .load
        passB.depthAttachment.clearDepth = 1
        passB.depthAttachment.storeAction = .dontCare
        if let enc = cmd.makeRenderCommandEncoder(descriptor: passB) {
            enc.setCullMode(.none)
            let fwd = simd_normalize(target - back * 1.1 + up * 0.15 - eye)
            let gen = model.predictor?.current ?? (Vec3(0, 0, 0), 0, 4.5)
            let center = avatar?.chest ?? phonePos
            var ru = ResolveUniforms(
                invViewProj: viewProj.inverse,
                cameraPos: SIMD4(eye.x, eye.y, eye.z, 1),
                cameraFwd: SIMD4(fwd.x, fwd.y, fwd.z, 0),
                bg: SIMD4(bg.x, bg.y, bg.z, 1),
                fog: SIMD4(bg.x, bg.y, bg.z, 7),
                params: SIMD4(avatar?.floorY ?? 0, fillEnabled ? 1 : 0, 16, texturing ? 1 : 0),
                // 相机拉得很远（比如俯视全图）时，那个「以用户为中心的球」会露馅，淡掉；地面照补。
                misc: SIMD4(0.16, gen.2, 1 - smoothstep(5, 8, simd_distance(eye, center)), roofCut),
                envCenter: SIMD4(
                    center.x, center.y, center.z, guessEnabled && snap.envValid ? 1 : 0),
                genCenter: SIMD4(gen.0.x, gen.0.y, gen.0.z, guessEnabled ? gen.1 : 0),
                room: roomEnabled ? (model.predictor?.room ?? .zero) : .zero,
                floorMap: snap.floorInfo,
                extra: SIMD4(
                    rawTint ? 1 : 0, generated == nil ? 0 : 1, max(0.5, simd_dot(target - eye, fwd)),
                    generatedTrust),
                genViewProj: generatedFrame?.viewProj ?? matrix_identity_float4x4)
            enc.setRenderPipelineState(resolvePipeline)
            enc.setDepthStencilState(depthOff)
            bindSurfaceInputs(enc, accum: accum, accumZ: accumZ, id: idTex, snap: snap, uniforms: &ru)
            // 网络画好的那张图（没有就随便绑一张占位）。
            enc.setFragmentTexture(generated ?? model.map.envTexture, index: 9)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

            // 先画各张图自己的那层皮垫底（全景壳上还没画到的地方露出来的是它），再画全景壳。
            if !layers.isEmpty, let indices = genMeshIndices {
                enc.setRenderPipelineState(genMeshPipeline)
                enc.setDepthStencilState(depthOff)
                enc.setFrontFacing(.counterClockwise)
                enc.setCullMode(genCull == 1 ? .front : (genCull == 2 ? .none : .back))
                enc.setFragmentSamplerState(sampler, index: 0)
                for layer in layers {
                    var gu = GenMeshUniforms(
                        viewProj: viewProj,
                        params: SIMD4(layer.alpha, Float(GenMesh.cols), Float(GenMesh.rows), layer.sharpen),
                        screen: SIMD4(Float(size.width), Float(size.height), max(0.01, layer.feather), 0))
                    enc.setVertexBuffer(layer.key.vertices, offset: 0, index: 0)
                    enc.setVertexBytes(&gu, length: MemoryLayout<GenMeshUniforms>.stride, index: 1)
                    enc.setFragmentBytes(&gu, length: MemoryLayout<GenMeshUniforms>.stride, index: 1)
                    enc.setFragmentTexture(layer.key.texture, index: 0)
                    enc.drawIndexedPrimitives(
                        type: .triangle, indexCount: GenMesh.indexCount, indexType: .uint16,
                        indexBuffer: indices, indexBufferOffset: 0)
                }
                enc.setCullMode(.none)
            }
            if shellReady, let shell = genShell {
                shell.draw(
                    enc, viewProj: viewProj, eye: eye, fwd: fwd, minDepth: nearClip, roofCut: roofCut,
                    depth: depthWrite)
                enc.setCullMode(.none)
            }

            if let avatar {
                drawAvatar(
                    enc, avatar, viewProj: viewProj, eye: eye,
                    envValid: guessEnabled && snap.envValid, size: size)
            }
            enc.endEncoding()
        }

        if neuralOn, let raw = rawTexture, let copy = rawCopy {
            if neural.wantsFrame() {
                if let blit = cmd.makeBlitCommandEncoder() {
                    blit.copy(from: raw, to: copy)
                    blit.endEncoding()
                }
                let w = copy.width, h = copy.height
                cmd.addCompletedHandler { _ in
                    var bytes = [UInt8](repeating: 0, count: w * h * 4)
                    copy.getBytes(
                        &bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h),
                        mipmapLevel: 0)
                    neural.submit(
                        pixels: bytes, width: w, height: h, viewProj: viewProj, camera: viewM.inverse)
                }
            }
            let passC = MTLRenderPassDescriptor()
            passC.colorAttachments[0].texture = drawable.texture
            passC.colorAttachments[0].loadAction = .dontCare
            passC.colorAttachments[0].storeAction = .store
            passC.depthAttachment.texture = depthTex
            passC.depthAttachment.loadAction = .dontCare
            passC.depthAttachment.storeAction = .dontCare
            if let enc = cmd.makeRenderCommandEncoder(descriptor: passC) {
                enc.setRenderPipelineState(blitPipeline)
                enc.setDepthStencilState(depthOff)
                enc.setFragmentTexture(neural.texture ?? raw, index: 0)
                enc.setFragmentSamplerState(sampler, index: 0)
                enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                enc.endEncoding()
            }
        }

        if shotInterval > 0, now - lastShot >= shotInterval, !drawable.texture.isFramebufferOnly {
            lastShot = now
            let index = shotIndex % 6
            shotIndex += 1
            cmd.addCompletedHandler { _ in
                Renderer.saveFrame(drawable.texture, name: "shot-\(index).jpg")
            }
        }

        cmd.present(drawable)
        cmd.commit()
    }

    private static func saveFrame(_ tex: MTLTexture, name: String) {
        let w = tex.width, h = tex.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        tex.getBytes(
            &bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        let info = CGBitmapInfo.byteOrder32Little.rawValue
            | CGImageAlphaInfo.noneSkipFirst.rawValue
        guard
            let ctx = CGContext(
                data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: info),
            let image = ctx.makeImage(),
            let data = UIImage(cgImage: image).jpegData(compressionQuality: 0.8),
            let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        else { return }
        try? data.write(to: dir.appendingPathComponent(name))
    }

    private func drawAvatar(
        _ enc: MTLRenderCommandEncoder, _ avatar: AvatarPose, viewProj: Mat4, eye: Vec3,
        envValid: Bool, size: CGSize
    ) {
        let light = simd_normalize(Vec3(0.35, 0.9, 0.25))

        if character == nil {
        // ---- 身体：距离场，一个全屏三角形，用剪裁框把像素限制在人偶附近 ----
        let top = avatar.head + Vec3(0, avatar.headRadius, 0)
        let center = (top + avatar.feetCenter) * 0.5
        var radius = simd_distance(top, avatar.feetCenter) * 0.5 + 0.12
        for l in avatar.limbs {
            radius = max(radius, simd_distance(l.a, center) + l.ra + 0.04)
            radius = max(radius, simd_distance(l.b, center) + l.rb + 0.04)
        }
        var u = [SIMD4<Float>]()
        u.reserveCapacity(15 + 2 * Renderer.maxLimbs)
        for m in [viewProj, viewProj.inverse] {
            u += [m.columns.0, m.columns.1, m.columns.2, m.columns.3]
        }
        let body = AvatarSolver.bodyColor, visor = AvatarSolver.visorColor
        let limbs = avatar.limbs.prefix(Renderer.maxLimbs)
        u += [
            SIMD4(eye.x, eye.y, eye.z, 1),
            SIMD4(light.x, light.y, light.z, envValid ? 1 : 0),
            SIMD4(center.x, center.y, center.z, radius),
            SIMD4(avatar.head.x, avatar.head.y, avatar.head.z, avatar.headRadius),
            SIMD4(avatar.faceDir.x, avatar.faceDir.y, avatar.faceDir.z, Float(limbs.count)),
            SIMD4(body.x, body.y, body.z, 1),
            SIMD4(visor.x, visor.y, visor.z, 1),
        ]
        var ends = [SIMD4<Float>](repeating: .zero, count: 2 * Renderer.maxLimbs)
        for (i, l) in limbs.enumerated() {
            ends[i] = SIMD4(l.a.x, l.a.y, l.a.z, l.ra)
            ends[Renderer.maxLimbs + i] = SIMD4(l.b.x, l.b.y, l.b.z, l.rb)
        }
        u += ends

        // 包围球投到屏幕上的方框。相机进到球里面时就整屏画。
        let w = Int(size.width), h = Int(size.height)
        var scissor = MTLScissorRect(x: 0, y: 0, width: w, height: h)
        let clip = viewProj * SIMD4(center.x, center.y, center.z, 1)
        if clip.w > radius * 1.2 {
            let dist = simd_distance(eye, center)
            let angular = radius / (dist * dist - radius * radius).squareRoot()
            // 竖直视场 60°：屏幕半高对应 tan(30°)。
            let ry = angular / tan(30 * Float.pi / 180) * Float(h) * 0.5 * 1.1
            let rx = ry
            let cxp = (clip.x / clip.w * 0.5 + 0.5) * Float(w)
            let cyp = (1 - (clip.y / clip.w * 0.5 + 0.5)) * Float(h)
            let x0 = max(0, Int(cxp - rx)), x1 = min(w, Int(cxp + rx) + 1)
            let y0 = max(0, Int(cyp - ry)), y1 = min(h, Int(cyp + ry) + 1)
            if x1 > x0, y1 > y0 {
                scissor = MTLScissorRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
            } else {
                scissor = MTLScissorRect(x: 0, y: 0, width: 1, height: 1)
            }
        }
        enc.setRenderPipelineState(avatarPipeline)
        enc.setDepthStencilState(depthWrite)
        enc.setScissorRect(scissor)
        u.withUnsafeBytes { raw in
            enc.setFragmentBytes(raw.baseAddress!, length: raw.count, index: 0)
        }
        enc.setFragmentTexture(model.map.envTexture, index: 0)
        enc.setFragmentSamplerState(envSampler, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.setScissorRect(MTLScissorRect(x: 0, y: 0, width: w, height: h))

        }

        // ---- 手机和地上的软阴影：网格 ----
        // 静态人物模型：整尊像按「站着不动」的位置和朝向摆，按用户身高缩放；手机放在它伸出来的那只手上。
        var phone = avatar.phone
        var feet = avatar.feetCenter
        var phoneScale: Float = 1
        // 手机模型摆成和真机一样的朝向：真机相对「人实际面朝的方向」怎么拿，模型相对那尊像就怎么拿。
        var phoneRotation = avatar.phone
        phoneRotation.columns.3 = SIMD4(0, 0, 0, 1)
        var figure: CharacterUniforms?
        var playing: CharacterFrames?
        if let character {
            let up = Vec3(0, 1, 0)
            let fwd = avatar.stillHeading
            // 模型是真人比例的，按用户身高放。
            let k = avatar.standingHeight / character.height
            let model = Mat4.basis(
                x: simd_cross(up, fwd) * k, y: up * k, z: fwd * k, origin: avatar.stillRoot)
            // 三段动作各播到哪。待机 4 秒一个循环；`-walkTest`、`-sitTest` 是截图检查动作用的。
            func loop(_ phase: Float, start: Int, count: Int) -> (a: Int, b: Int, t: Float) {
                let x = (phase - phase.rounded(.down)) * Float(count)
                let i = min(Int(x), count - 1)
                return (start + i, start + (i + 1) % count, x - Float(i))
            }
            let now = Float(CACurrentMediaTime().truncatingRemainder(dividingBy: 3600))
            let seat = sitTest >= 0 ? sitTest : avatar.seatAmount
            let sitAt = seat * Float(character.sitFrames - 1)
            let sitA = min(Int(sitAt), max(0, character.sitFrames - 2))
            let frames = CharacterFrames(
                idle: loop(now / 4, start: 0, count: character.idleFrames),
                walk: loop(walkTest ? now / 1.1 : avatar.walkPhase, start: character.walkStart, count: max(1, character.walkFrames)),
                sit: (character.sitStart + sitA, character.sitStart + min(sitA + 1, character.sitFrames - 1), sitAt - Float(sitA)),
                walkWeight: character.walkFrames > 0 ? (walkTest ? 1 : avatar.walkAmount) : 0,
                sitWeight: character.sitFrames > 1 ? smoothstep(0, 0.2, seat) : 0)
            playing = frames
            figure = CharacterUniforms(
                model: model, blend: SIMD4(frames.idle.t, frames.walk.t, frames.sit.t, 0),
                weight: SIMD4(frames.walkWeight, frames.sitWeight, 0, 0))
            // 手机跟着手：和顶点一样的混法。
            func at(_ f: (a: Int, b: Int, t: Float)) -> Vec3 {
                character.hands[f.a] + (character.hands[f.b] - character.hands[f.a]) * f.t
            }
            var palm = at(frames.idle)
            if frames.walkWeight > 0 { palm += (at(frames.walk) - palm) * frames.walkWeight }
            if frames.sitWeight > 0 { palm += (at(frames.sit) - palm) * frames.sitWeight }
            let hand = (model * SIMD4(palm.x, palm.y, palm.z, 1)).xyz
            // 竖着拿：机身长边朝下，屏幕朝着人。
            phone = Mat4.basis(
                x: -up, y: simd_cross(fwd, up), z: -fwd, origin: hand - fwd * 0.05 + up * 0.06)
            feet = (avatar.stillRoot + avatar.stillCenter) * 0.5
            phoneScale = 1.15
            // 像的朝向和人实际的朝向差一个绕竖直轴的角度，把真机的姿态也转过这个角度。
            let real = avatar.heading
            let delta = atan2(fwd.x, fwd.z) - atan2(real.x, real.z)
            let turn = Mat4.basis(
                x: Vec3(cos(delta), 0, -sin(delta)), y: up, z: Vec3(sin(delta), 0, cos(delta)),
                origin: Vec3(0, 0, 0))
            phoneRotation = turn * phoneRotation
            phoneRotation.columns.3 = SIMD4(hand.x, hand.y + 0.07, hand.z, 1)
        } else {
            phoneRotation.columns.3 = avatar.phone.columns.3
        }
        let phoneBox = MeshInstance(
            model: phone * Mat4.scale(Vec3(0.15, 0.075, 0.009) * phoneScale),
            color: SIMD4(0.07, 0.07, 0.09, 0))
        // 屏幕贴在朝用户的那一面（+z），显示后摄正在拍的画面。
        let phoneScreen = MeshInstance(
            model: phone * Mat4.translation(Vec3(0, 0, 0.0052 * phoneScale))
                * Mat4.scale(Vec3(0.142 * phoneScale, 0.068 * phoneScale, 1)),
            color: SIMD4(1, 1, 1, 1))
        let f = feet
        let shadow = MeshInstance(
            model: Mat4.basis(
                x: Vec3(0.95, 0, 0), y: Vec3(0, 0, -0.95), z: Vec3(0, 1, 0),
                origin: Vec3(f.x, f.y + 0.012, f.z)),
            color: SIMD4(0, 0, 0, 2))

        let all = [phoneBox, phoneScreen, shadow]
        frameIndex = (frameIndex + 1) % instanceBuffers.count
        let buffer = instanceBuffers[frameIndex]
        all.withUnsafeBytes { raw in
            buffer.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
        }

        var mu = MeshUniforms(
            viewProj: viewProj,
            lightDir: SIMD4(light, 0),
            cameraPos: SIMD4(eye.x, eye.y, eye.z, 1))
        enc.setRenderPipelineState(meshPipeline)
        enc.setDepthStencilState(depthWrite)
        enc.setVertexBuffer(meshes.vertexBuffer, offset: 0, index: 0)
        enc.setVertexBuffer(buffer, offset: 0, index: 1)
        enc.setVertexBytes(&mu, length: MemoryLayout<MeshUniforms>.stride, index: 2)
        enc.setFragmentBytes(&mu, length: MemoryLayout<MeshUniforms>.stride, index: 2)
        enc.setFragmentSamplerState(sampler, index: 0)
        if let tex = screenTexture { enc.setFragmentTexture(tex, index: 0) }

        func drawMesh(_ r: Meshes.Range, base: Int, count: Int) {
            guard count > 0 else { return }
            enc.drawIndexedPrimitives(
                type: .triangle, indexCount: r.indexCount, indexType: .uint16,
                indexBuffer: meshes.indexBuffer,
                indexBufferOffset: r.indexOffset * MemoryLayout<UInt16>.stride,
                instanceCount: count, baseVertex: 0, baseInstance: base)
        }
        if let phoneModel {
            // 开合程度：系统给了就用系统的；没给就按窗口宽度猜（普通手机会被当成合上的 Duo）。
            let open = model.hingeOpen ?? (viewWidthPoints > 560 ? 1 : 0)
            let position = min(max(open, 0), 1) * Float(phoneModel.frameCount - 1)
            let a = min(Int(position), phoneModel.frameCount - 2)
            // 模型的轴 → 相机的轴：模型的右是相机的 +y，模型的上（铰链方向）是相机的 -x，屏幕朝 +z。
            // 合上时只剩带摄像头的那一半，往回挪半个机身宽，让它还握在手心。
            let fit = Mat4.basis(
                x: Vec3(0, 1, 0), y: Vec3(-1, 0, 0), z: Vec3(0, 0, 1), origin: Vec3(0, 0, 0))
                * Mat4.translation(Vec3(-0.0415 * (1 - open), 0, 0))
            var pu = CharacterUniforms(
                model: phoneRotation * fit * Mat4.scale(Vec3(repeating: phoneScale)),
                blend: SIMD4(position - Float(a), 0, 0, 0), weight: SIMD4(0, 0, 0, 0))
            enc.setRenderPipelineState(characterPipeline)
            enc.setVertexBuffer(phoneModel.frames, offset: phoneModel.offset(frame: a), index: 0)
            enc.setVertexBuffer(phoneModel.frames, offset: phoneModel.offset(frame: a + 1), index: 7)
            for slot in [3, 4, 8, 9] {
                enc.setVertexBuffer(phoneModel.frames, offset: phoneModel.offset(frame: a), index: slot)
            }
            enc.setVertexBuffer(phoneModel.uvBuffer, offset: 0, index: 5)
            enc.setVertexBytes(&pu, length: MemoryLayout<CharacterUniforms>.stride, index: 6)
            enc.setFragmentTexture(screenTexture ?? phoneModel.texture, index: 0)
            enc.setFragmentTexture(phoneModel.texture, index: 1)
            enc.drawIndexedPrimitives(
                type: .triangle, indexCount: phoneModel.indexCount, indexType: .uint32,
                indexBuffer: phoneModel.indexBuffer, indexBufferOffset: 0)
            enc.setRenderPipelineState(meshPipeline)
            enc.setVertexBuffer(meshes.vertexBuffer, offset: 0, index: 0)
            if let tex = screenTexture { enc.setFragmentTexture(tex, index: 0) }
        }
        if let character, var cu = figure, let f = playing {
            enc.setRenderPipelineState(characterPipeline)
            for (slot, frame) in [(0, f.idle.a), (7, f.idle.b), (3, f.walk.a), (4, f.walk.b), (8, f.sit.a), (9, f.sit.b)] {
                enc.setVertexBuffer(
                    character.frames, offset: character.offset(frame: min(frame, character.frameCount - 1)),
                    index: slot)
            }
            enc.setVertexBuffer(character.uvBuffer, offset: 0, index: 5)
            enc.setVertexBytes(&cu, length: MemoryLayout<CharacterUniforms>.stride, index: 6)
            enc.setFragmentTexture(character.texture, index: 1)
            enc.drawIndexedPrimitives(
                type: .triangle, indexCount: character.indexCount, indexType: .uint32,
                indexBuffer: character.indexBuffer, indexBufferOffset: 0)
            enc.setRenderPipelineState(meshPipeline)
            enc.setVertexBuffer(meshes.vertexBuffer, offset: 0, index: 0)
        }
        if phoneModel == nil {
            drawMesh(meshes.cube, base: 0, count: 1)
            if screenTexture != nil { drawMesh(meshes.quad, base: 1, count: 1) }
        }
        let base = 2
        // 软阴影最后画：要测深度（被前面的家具挡住），但自己不写深度。
        enc.setDepthStencilState(depthTestOnly)
        drawMesh(meshes.quad, base: base, count: 1)
    }
}
