import Foundation
import simd

/// 人偶身上的一段：两头半径可以不一样的圆棒。渲染时所有圆棒在距离场里平滑地并在一起。
struct Limb {
    var a: Vec3
    var ra: Float
    var b: Vec3
    var rb: Float
}

struct AvatarPose {
    var limbs: [Limb] = []
    var head = Vec3(0, 1.6, 0)
    var headRadius: Float = 0.1
    var faceDir = Vec3(0, 0, -1)
    /// 手机盒子的位姿（原点在机身中心，轴跟相机一致）。
    var phone: Mat4 = matrix_identity_float4x4
    var chest = Vec3(0, 1.2, 0)
    var heading = Vec3(0, 0, -1)
    var floorY: Float = 0
    var feetCenter = Vec3(0, 0, 0)
    /// 「站着不动的那尊像」的位置（脚底）和朝向：用户小幅度晃动、转头时它不跟着动，
    /// 走开半步以上或者转过一个明显的角度才挪过去。静态人物模型和跟拍镜头用这两个。
    var stillRoot = Vec3(0, 0, 0)
    var stillHeading = Vec3(0, 0, -1)
    var standingHeight: Float = 1.6
    /// 那尊像在走还是站着（0~1），和走路循环放到哪了（0~1，一圈是左右各迈一步）。
    var walkAmount: Float = 0
    var walkPhase: Float = 0
    /// 那尊像坐下了多少（0 站着，1 坐稳）。
    var seatAmount: Float = 0
    /// 那尊像身体中心在地上的投影。站着时就是 stillRoot；坐下时脚留在 stillRoot，胯往后坐，中心跟着往后。
    var stillCenter = Vec3(0, 0, 0)
}

/// 从「手机在哪」推出「人是什么姿势」。
///
/// 真实数据只有两样：手机的 6DoF 位姿（= 手在哪），以及前摄测到的头（没有就估）。
/// 躯干是按头的位置摆的，手臂和腿是 IK，迈步是按移动速度播的循环。
/// 原理跟 VR 头显用头和手推全身一样，腿是推出来的，不是测出来的。
final class AvatarSolver {
    static let bodyColor = Vec3(0.90, 0.91, 0.93)
    static let visorColor = Vec3(0.16, 0.20, 0.28)

    private var heading = Vec3(0, 0, -1)
    private var head: Vec3?
    private var lastHead: Vec3?
    private var floorY: Float?
    private var velocity = Vec3(0, 0, 0)
    private var gaitPhase: Float = 0
    private var standingHeight: Float = 1.6
    /// 按「头离地至少 1.15 米」推出来的地面高度上限里，见过的最低的那个。
    private var headFloor: Float?
    private var stillRoot: Vec3?
    private var stillHeading: Vec3?
    private var stillMoving = false
    private var stillTurning = false
    private var stillSpeed: Float = 0
    private var stillPhase: Float = 0
    private var turnRate: Float = 0
    private var sitting = false
    private var seatTimer: Float = 0
    private var seatAmount: Float = 0
    private var logTimer: Float = 0
    private var sinceStart: Float = 0
    private var seatAnchor: Vec3?
    /// 人物模型的两个尺寸，都是身高的几分之几：走一个循环前进多远，坐稳时胯往后挪多远。
    var strideFraction: Float = 0.46
    var sitBackFraction: Float = 0.2

    func update(pose: PoseSample, dt: Float) -> AvatarPose {
        let up = Vec3(0, 1, 0)
        let cam = pose.cameraToWorld
        // ARKit 的原点在镜头上，镜头在机身顶端；往机身中心挪一点。
        let phoneCenter = cam.position + cam.xAxis * 0.055
        let camForward = -cam.zAxis

        // ---- 朝向和头 ----
        var measuredHead: Vec3?
        var targetHeading = heading
        if let h = pose.headToWorld {
            measuredHead = h.position
            targetHeading = (phoneCenter - h.position).horizontal.normalized(or: heading)
        } else {
            let hf = camForward.horizontal
            if simd_length(hf) > 0.35 {
                targetHeading = simd_normalize(hf)
            } else {
                // 镜头朝正下或正上：用机身顶端的指向代替。
                let top = (camForward.y < 0 ? -cam.xAxis : cam.xAxis).horizontal
                targetHeading = top.normalized(or: heading)
            }
        }
        heading = (heading + (targetHeading - heading) * lowpass(dt, 0.25)).normalized(or: targetHeading)

        let headTarget = measuredHead ?? (phoneCenter - heading * 0.36 + up * 0.24)
        let headNow: Vec3
        if let prev = head {
            headNow = prev + (headTarget - prev) * lowpass(dt, measuredHead == nil ? 0.12 : 0.04)
        } else {
            headNow = headTarget
        }
        head = headNow

        // ---- 地面和身高 ----
        // ARKit 给的「地面」是它认出来的最低的那块水平面。坐在桌前、镜头没照到地板的时候，
        // 最低的平面是桌面：2026-11-23 的真机截图里人偶就蹲在了桌面那么高的地方，身材还被缩小了。
        // 所以再加一条常识：头离地至少 1.15 米（坐着的成年人）。两个里取低的那个。
        // 这个上限记最低值，慢慢才往回升：坐下过一次，之后站起来走也不会又把桌面当地面。
        let limit = headNow.y - 1.15
        if let h = headFloor {
            headFloor = limit < h ? limit : h + (limit - h) * lowpass(dt, 60)
        } else {
            headFloor = limit
        }
        let target = min(pose.floorY ?? .greatestFiniteMagnitude, headFloor ?? limit)
        floorY = floorY.map { $0 + (target - $0) * lowpass(dt, 0.5) } ?? target
        let floor = floorY ?? 0
        // 见过的最高头部高度当作站立身高，蹲下、坐下时腿会自己弯。不让它低于 1.5 米：
        // 一直坐着的话不能把人偶缩成一个站着的小矮人。
        standingHeight = min(2.1, max(1.5, max(standingHeight * 0.9999, headNow.y - floor)))
        let s = min(1.3, max(0.7, standingHeight / 1.6))

        // ---- 速度和步态 ----
        if let last = lastHead, dt > 1e-4 {
            let v = (headNow - last).horizontal / dt
            velocity += (v - velocity) * lowpass(dt, 0.25)
        }
        lastHead = headNow
        let speed = simd_length(velocity)
        let walk = smoothstep(0.12, 0.5, speed)
        gaitPhase += speed * dt / (1.4 * s) * 2 * .pi
        if gaitPhase > 2 * .pi { gaitPhase -= 2 * .pi }
        let moveDir = velocity.normalized(or: heading)

        let right = simd_normalize(simd_cross(heading, up))
        let bob = walk * 0.018 * s * cos(2 * gaitPhase)

        // ---- 骨架 ----
        let neck = headNow - up * (0.15 * s) - heading * (0.02 * s)
        var pelvis = neck - up * (0.50 * s) - heading * (0.03 * s) + up * bob
        pelvis.y = max(pelvis.y, floor + 0.28 * s)
        let shoulderL = neck - right * (0.19 * s) - up * (0.04 * s)
        let shoulderR = neck + right * (0.19 * s) - up * (0.04 * s)
        let hipL = pelvis - right * (0.095 * s)
        let hipR = pelvis + right * (0.095 * s)

        // 两只手握在手机更「横」的那一对边上。
        var side = cam.xAxis
        var half: Float = 0.075
        if abs(cam.yAxis.y) < abs(cam.xAxis.y) {
            side = cam.yAxis
            half = 0.0375
        }
        if simd_dot(side, right) < 0 { side = -side }
        let gripR = phoneCenter + side * (half + 0.02) - up * 0.015
        let gripL = phoneCenter - side * (half + 0.02) - up * 0.015
        // 手机的位置是测出来的，肩膀是估的。估偏了手就够不到，手机会孤零零飘在手外面，
        // 所以够不到时把手臂拉长（最多三成）去够它：手臂长一点没人看得出，手机脱手很扎眼。
        func arm(_ shoulder: Vec3, _ grip: Vec3, pole: Vec3) -> (mid: Vec3, end: Vec3) {
            let full = 0.58 * s
            let stretch = min(1.3, max(1, simd_distance(shoulder, grip) / (full * 0.98)))
            return Self.ik(shoulder, grip, 0.30 * s * stretch, 0.28 * s * stretch, pole: pole)
        }
        let armL = arm(shoulderL, gripL, pole: -up * 0.8 - right * 0.6 - heading * 0.2)
        let armR = arm(shoulderR, gripR, pole: -up * 0.8 + right * 0.6 - heading * 0.2)

        // 脚：站着时在胯正下方；走起来时沿移动方向前后摆，往前摆的那只抬起来。
        // 胯离地不到站着时的七成（坐着或蹲着）：脚往前伸，大腿放平，看起来是坐着而不是蹲成一团。
        let seated = (1 - smoothstep(0.55, 0.75, (pelvis.y - floor) / (0.93 * s))) * (1 - walk)
        func foot(_ hip: Vec3, _ phase: Float) -> Vec3 {
            var p = Vec3(hip.x, floor + 0.07 * s, hip.z) + heading * ((0.02 + 0.36 * seated) * s)
            p += moveDir * (sin(phase) * 0.27 * s * walk)
            p.y += max(0, cos(phase)) * 0.09 * s * walk
            return p
        }
        let ankleL = foot(hipL, gaitPhase)
        let ankleR = foot(hipR, gaitPhase + .pi)
        let legL = Self.ik(hipL, ankleL, 0.46 * s, 0.45 * s, pole: heading - right * 0.15)
        let legR = Self.ik(hipR, ankleR, 0.46 * s, 0.45 * s, pole: heading + right * 0.15)

        // ---- 站着不动的那尊像 ----
        // 坐着还是站着。可靠的依据只有一个：前摄测到的头离地多高（低于 1.28 米坐，高于 1.40 米站）。
        // 但真机上前摄每隔十来秒才认到一下脸，其余时间头是按手机位置估的，那其实只知道手机举多高：
        //   坐着：估的头高 1.28~1.49，把手机举高看四周时到过 1.58（测到的真实头高只有 1.15）；
        //   站着走：1.77~1.85，放下手机去点按钮时掉到 1.40。
        // 两边重叠，所以估的头高不再用来来回改主意（2026-11-23 22:33 真机：人一直坐着，手机一举高机器人就站起来了）：
        //   - 刚启动的两秒里还没有别的依据，先按估的猜一个（低于 1.39 算坐着）；
        //   - 之后只有很确定才改：估的低于 1.20 算坐下，高于 1.75 算站起来；
        //   - 坐着的人离开坐下的位置 0.8 米以上，算站起来走了（坐在转椅上转圈，估的头大致不动）。
        let headHeight = headNow.y - floor
        logTimer += dt
        if logTimer > 2 {
            logTimer = 0
            DebugLog.write(String(
                format: "AVATAR 头高 %.2f（%@） 身高 %.2f 坐 %.1f 速度 %.2f", headHeight,
                measuredHead == nil ? "按手机估的" : "测的", standingHeight, seatAmount, speed))
        }
        let measured = measuredHead != nil
        sinceStart += dt
        var seatWant: Bool?
        if measured {
            if headHeight < 1.28 { seatWant = true } else if headHeight > 1.40 { seatWant = false }
        } else if sinceStart < 2 {
            seatWant = headHeight < 1.39
        } else if headHeight < 1.20 {
            seatWant = true
        } else if headHeight > 1.75 {
            seatWant = false
        }
        var walkedAway = false
        if sitting, let anchor = seatAnchor, simd_length((headNow - anchor).horizontal) > 0.8 {
            seatWant = false
            walkedAway = true
        }
        if let w = seatWant, w != sitting, w == false || speed < 0.3 {
            seatTimer += dt
            if seatTimer > (measured ? 0.5 : 1.0) || stillRoot == nil || walkedAway {
                sitting = w
                seatTimer = 0
                seatAnchor = w ? headNow : nil
            }
        } else {
            seatTimer = 0
        }
        if stillRoot == nil { seatAmount = sitting ? 1 : 0 }  // 一打开就坐着：不用演一遍坐下
        seatAmount = min(1, max(0, seatAmount + (sitting ? dt : -dt) / 0.8))
        let ease = seatAmount * seatAmount * (3 - 2 * seatAmount)
        let back = sitBackFraction * standingHeight * ease

        // 朝向偏了 30 度以上才开始转，转到 5 度以内停。绕身体中心转，不是绕脚。
        var facing = stillHeading ?? heading
        let oldFacing = facing
        let turn = acos(min(1, max(-1, simd_dot(facing, heading))))
        if turn > 30 * .pi / 180 { stillTurning = true }
        if stillTurning {
            facing = (facing + (heading - facing) * lowpass(dt, 0.5)).normalized(or: heading)
            if turn < 5 * .pi / 180 { stillTurning = false }
        }
        stillHeading = facing
        if dt > 1e-4 {
            let w = acos(min(1, max(-1, simd_dot(facing, oldFacing)))) / dt
            turnRate += (min(w, 4) - turnRate) * lowpass(dt, 0.15)
        }

        // 人的位置按头算，像的位置按身体中心算（脚在 root，坐下时中心在脚后面 back 那么远）。
        // 中心离人 30 厘米以上才开始跟，跟到 6 厘米以内停，中间的小动作一律不理。
        let want = Vec3(headNow.x, floor, headNow.z)
        var center = stillRoot.map { $0 - oldFacing * back } ?? want
        let gap = simd_length((want - center).horizontal)
        if gap > 0.30 { stillMoving = true }
        if gap > 1.5 { center = want }  // 差太远（刚启动、跟踪跳了）直接过去
        if stillMoving {
            center += (want - center).horizontal * lowpass(dt, 0.45)
            if gap < 0.06 { stillMoving = false }
        }
        var root = center + facing * back
        root.y = floor
        center.y = floor
        // 像挪动的时候播走路：迈步的快慢跟着它实际挪动的速度，脚才不会在地上打滑。
        if let last = stillRoot, dt > 1e-4 {
            let v = simd_length((root - last).horizontal) / dt
            stillSpeed += (min(v, 3) - stillSpeed) * lowpass(dt, 0.15)
        }
        // 原地转身也要倒脚：转四分之一圈大约迈一步。
        let stride = max(0.3, strideFraction * standingHeight)
        stillPhase = (stillPhase + stillSpeed * dt / stride + turnRate * dt * 0.32)
            .truncatingRemainder(dividingBy: 1)
        stillRoot = root

        var out = AvatarPose()
        out.stillRoot = root
        out.stillCenter = center
        out.stillHeading = facing
        out.standingHeight = standingHeight
        out.seatAmount = seatAmount
        out.walkAmount = max(smoothstep(0.04, 0.22, stillSpeed), 0.55 * smoothstep(0.2, 0.7, turnRate))
            * (1 - ease)
        out.walkPhase = stillPhase
        let faceDir = pose.headToWorld?.zAxis ?? (phoneCenter - headNow).normalized(or: heading)
        func hand(_ wrist: Vec3, _ elbow: Vec3) -> Limb {
            let dir = (wrist - elbow).normalized(or: heading)
            return Limb(a: wrist, ra: 0.040 * s, b: wrist + dir * (0.055 * s), rb: 0.034 * s)
        }
        func foot(_ ankle: Vec3) -> Limb {
            Limb(
                a: ankle - up * (0.015 * s) - heading * (0.02 * s), ra: 0.045 * s,
                b: ankle + heading * (0.15 * s) - up * (0.035 * s), rb: 0.036 * s)
        }
        out.limbs = [
            // 躯干：中间一根脊柱，两侧各一根，并起来是一个前后扁、左右宽的身子。
            Limb(a: neck - up * (0.08 * s), ra: 0.105 * s, b: pelvis + up * (0.05 * s), rb: 0.095 * s),
            Limb(
                a: neck - right * (0.085 * s) - up * (0.11 * s), ra: 0.100 * s,
                b: pelvis - right * (0.07 * s) + up * (0.04 * s), rb: 0.088 * s),
            Limb(
                a: neck + right * (0.085 * s) - up * (0.11 * s), ra: 0.100 * s,
                b: pelvis + right * (0.07 * s) + up * (0.04 * s), rb: 0.088 * s),
            Limb(a: shoulderL, ra: 0.060 * s, b: shoulderR, rb: 0.060 * s),
            Limb(a: hipL, ra: 0.092 * s, b: hipR, rb: 0.092 * s),
            Limb(a: neck - up * (0.02 * s), ra: 0.048 * s, b: headNow - up * (0.07 * s), rb: 0.044 * s),
            Limb(a: headNow - up * (0.018 * s), ra: 0.098 * s, b: headNow + up * (0.028 * s), rb: 0.096 * s),
            Limb(a: shoulderL, ra: 0.054 * s, b: armL.mid, rb: 0.043 * s),
            Limb(a: armL.mid, ra: 0.042 * s, b: armL.end, rb: 0.033 * s),
            hand(armL.end, armL.mid),
            Limb(a: shoulderR, ra: 0.054 * s, b: armR.mid, rb: 0.043 * s),
            Limb(a: armR.mid, ra: 0.042 * s, b: armR.end, rb: 0.033 * s),
            hand(armR.end, armR.mid),
            Limb(a: hipL, ra: 0.082 * s, b: legL.mid, rb: 0.058 * s),
            Limb(a: legL.mid, ra: 0.056 * s, b: legL.end, rb: 0.040 * s),
            foot(legL.end),
            Limb(a: hipR, ra: 0.082 * s, b: legR.mid, rb: 0.058 * s),
            Limb(a: legR.mid, ra: 0.056 * s, b: legR.end, rb: 0.040 * s),
            foot(legR.end),
        ]
        out.head = headNow + up * (0.005 * s)
        out.headRadius = 0.098 * s
        out.faceDir = faceDir
        out.phone = cam * Mat4.translation(Vec3(0.055, 0, 0))
        out.chest = (neck + pelvis) * 0.5
        out.heading = heading
        out.floorY = floor
        out.feetCenter = Vec3(pelvis.x, floor, pelvis.z)
        return out
    }

    /// 两段骨头的 IK：给定根、目标和两段长度，返回中间关节和实际够到的末端。
    static func ik(_ root: Vec3, _ target: Vec3, _ l1: Float, _ l2: Float, pole: Vec3)
        -> (mid: Vec3, end: Vec3)
    {
        let d = target - root
        let len = simd_length(d)
        let dir = len > 1e-5 ? d / len : Vec3(0, -1, 0)
        let reach = min(max(len, abs(l1 - l2) + 0.01), (l1 + l2) * 0.999)
        let end = root + dir * reach
        let x = (l1 * l1 - l2 * l2 + reach * reach) / (2 * reach)
        let hgt = max(0, l1 * l1 - x * x).squareRoot()
        let perp = (pole - dir * simd_dot(pole, dir)).normalized(or: Vec3(0, 0, 1))
        return (root + dir * x + perp * hgt, end)
    }
}
