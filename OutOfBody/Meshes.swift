import Metal
import simd

/// 人偶和手机用的几个基本体，放在同一对顶点 / 索引缓冲里。
/// 顶点 = 位置(3) + 法线(3) + uv(2)，8 个 float。
struct Meshes {
    struct Range {
        var indexOffset: Int
        var indexCount: Int
    }

    let vertexBuffer: MTLBuffer
    let indexBuffer: MTLBuffer
    let sphere: Range
    let cylinder: Range  // 半径 1，y 从 0 到 1，没有盖
    let cube: Range  // 边长 1，中心在原点
    let quad: Range  // xy 平面，边长 1，法线 +z

    static let vertexDescriptor: MTLVertexDescriptor = {
        let d = MTLVertexDescriptor()
        d.attributes[0].format = .float3
        d.attributes[0].offset = 0
        d.attributes[1].format = .float3
        d.attributes[1].offset = 12
        d.attributes[2].format = .float2
        d.attributes[2].offset = 24
        d.layouts[0].stride = 32
        return d
    }()

    init?(device: MTLDevice) {
        var verts: [Float] = []
        var idx: [UInt16] = []
        func vertexCount() -> UInt16 { UInt16(verts.count / 8) }
        func add(_ p: Vec3, _ n: Vec3, _ u: Float, _ v: Float) {
            verts += [p.x, p.y, p.z, n.x, n.y, n.z, u, v]
        }

        // 球
        var start = idx.count
        var base = vertexCount()
        let rings = 12, segs = 20
        for r in 0...rings {
            let phi = Float(r) / Float(rings) * .pi
            for s in 0...segs {
                let th = Float(s) / Float(segs) * 2 * .pi
                let n = Vec3(sin(phi) * cos(th), cos(phi), sin(phi) * sin(th))
                add(n, n, Float(s) / Float(segs), Float(r) / Float(rings))
            }
        }
        for r in 0..<rings {
            for s in 0..<segs {
                let a = base + UInt16(r * (segs + 1) + s)
                let b = a + UInt16(segs + 1)
                idx += [a, b, a + 1, a + 1, b, b + 1]
            }
        }
        sphere = Range(indexOffset: start, indexCount: idx.count - start)

        // 圆柱
        start = idx.count
        base = vertexCount()
        for s in 0...segs {
            let th = Float(s) / Float(segs) * 2 * .pi
            let n = Vec3(cos(th), 0, sin(th))
            add(Vec3(n.x, 0, n.z), n, Float(s) / Float(segs), 0)
            add(Vec3(n.x, 1, n.z), n, Float(s) / Float(segs), 1)
        }
        for s in 0..<segs {
            let a = base + UInt16(s * 2)
            idx += [a, a + 1, a + 2, a + 2, a + 1, a + 3]
        }
        cylinder = Range(indexOffset: start, indexCount: idx.count - start)

        // 立方体
        start = idx.count
        let faces: [(Vec3, Vec3, Vec3)] = [
            (Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)),
            (Vec3(-1, 0, 0), Vec3(0, 0, 1), Vec3(0, 1, 0)),
            (Vec3(0, 1, 0), Vec3(0, 0, 1), Vec3(1, 0, 0)),
            (Vec3(0, -1, 0), Vec3(1, 0, 0), Vec3(0, 0, 1)),
            (Vec3(0, 0, 1), Vec3(1, 0, 0), Vec3(0, 1, 0)),
            (Vec3(0, 0, -1), Vec3(0, 1, 0), Vec3(1, 0, 0)),
        ]
        for (n, t, b) in faces {
            base = vertexCount()
            add((n - t - b) * 0.5, n, 0, 0)
            add((n + t - b) * 0.5, n, 1, 0)
            add((n + t + b) * 0.5, n, 1, 1)
            add((n - t + b) * 0.5, n, 0, 1)
            idx += [base, base + 1, base + 2, base, base + 2, base + 3]
        }
        cube = Range(indexOffset: start, indexCount: idx.count - start)

        // 方片：u 沿 +x，v 沿 -y（跟图像一样 v 朝下）。
        start = idx.count
        base = vertexCount()
        let n = Vec3(0, 0, 1)
        add(Vec3(-0.5, -0.5, 0), n, 0, 1)
        add(Vec3(0.5, -0.5, 0), n, 1, 1)
        add(Vec3(0.5, 0.5, 0), n, 1, 0)
        add(Vec3(-0.5, 0.5, 0), n, 0, 0)
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
        quad = Range(indexOffset: start, indexCount: idx.count - start)

        guard
            let vb = device.makeBuffer(
                bytes: verts, length: verts.count * MemoryLayout<Float>.stride),
            let ib = device.makeBuffer(
                bytes: idx, length: idx.count * MemoryLayout<UInt16>.stride)
        else { return nil }
        vertexBuffer = vb
        indexBuffer = ib
    }
}
