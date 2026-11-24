import ARKit
import AVFoundation

/// 这台设备能给我们什么：决定深度从哪来、头的位置能不能直接测。
struct Capabilities: Codable {
    var model = ""
    var osVersion = ""
    var worldTracking = false
    var sceneDepth = false
    var smoothedSceneDepth = false
    var meshReconstruction = false
    var userFaceTracking = false
    var faceTracking = false
    var faceTrackingWithWorld = false
    var personSegmentationWithDepth = false
    var bodyTracking = false
    var lidarCamera = false
    var trueDepthCamera = false
    var cameras: [String] = []
    var videoFormats: [String] = []

    static func probe() -> Capabilities {
        var c = Capabilities()
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var machine = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.machine", &machine, &size, nil, 0)
        c.model = String(cString: machine)
        c.osVersion = UIDevice.current.systemVersion

        c.worldTracking = ARWorldTrackingConfiguration.isSupported
        c.sceneDepth = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
        c.smoothedSceneDepth = ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth)
        c.meshReconstruction = ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh)
        c.userFaceTracking = ARWorldTrackingConfiguration.supportsUserFaceTracking
        c.faceTracking = ARFaceTrackingConfiguration.isSupported
        c.faceTrackingWithWorld = ARFaceTrackingConfiguration.supportsWorldTracking
        c.personSegmentationWithDepth =
            ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentationWithDepth)
        c.bodyTracking = ARBodyTrackingConfiguration.isSupported

        let types: [AVCaptureDevice.DeviceType] = [
            .builtInWideAngleCamera, .builtInUltraWideCamera, .builtInTelephotoCamera,
            .builtInDualCamera, .builtInDualWideCamera, .builtInTripleCamera,
            .builtInTrueDepthCamera, .builtInLiDARDepthCamera,
        ]
        let session = AVCaptureDevice.DiscoverySession(
            deviceTypes: types, mediaType: .video, position: .unspecified)
        for d in session.devices {
            let pos = d.position == .back ? "back" : d.position == .front ? "front" : "?"
            c.cameras.append("\(pos) \(d.deviceType.rawValue)")
            if d.deviceType == .builtInLiDARDepthCamera { c.lidarCamera = true }
            if d.deviceType == .builtInTrueDepthCamera { c.trueDepthCamera = true }
        }
        for f in ARWorldTrackingConfiguration.supportedVideoFormats {
            let r = f.imageResolution
            c.videoFormats.append(
                "\(Int(r.width))x\(Int(r.height))@\(f.framesPerSecond) \(f.captureDeviceType.rawValue)")
        }
        return c
    }

    var json: String {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        return (try? String(data: enc.encode(self), encoding: .utf8)) ?? "{}"
    }
}
