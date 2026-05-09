// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "AppleSpeechBenchmark",
    platforms: [
        .macOS(.v14)
    ],
    targets: [
        .executableTarget(
            name: "AppleSpeechBenchmark",
            path: "Sources/AppleSpeechBenchmark"
        )
    ]
)
