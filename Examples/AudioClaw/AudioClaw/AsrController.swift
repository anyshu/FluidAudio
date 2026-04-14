@preconcurrency import AVFoundation
import FluidAudio
import Foundation

struct AudioInputDevice: Identifiable, Equatable {
    let id: String
    let name: String
}

enum VadDebugEvent: Sendable {
    case speechStart(probability: Float)
    case speechEnd(probability: Float)
    case forcedSplit(probability: Float, seconds: Double)
}

actor CapturedAudioStore {
    private(set) var sampleCount: Int = 0
    private var samples: [Float] = []

    func reset() {
        samples.removeAll(keepingCapacity: true)
        sampleCount = 0
    }

    func append(_ newSamples: [Float]) {
        guard !newSamples.isEmpty else { return }
        samples.append(contentsOf: newSamples)
        sampleCount = samples.count
    }

    func snapshot() -> [Float] {
        samples
    }

    func writeDebugWav() throws -> URL {
        guard !samples.isEmpty else {
            throw AudioClawError.noDebugAudioAvailable
        }

        let fileName = "fluidaudio-debug-\(Self.timestamp()).wav"
        let downloadsURL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Downloads", isDirectory: true)
        let outputURL = downloadsURL.appendingPathComponent(fileName)
        let wavData = try AudioWAV.data(from: samples, sampleRate: 16_000)
        try wavData.write(to: outputURL)
        return outputURL
    }

    private static func timestamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter.string(from: Date())
    }
}

private struct VadGateResult {
    let finalizedUtterances: [[Float]]
    let isSpeechActive: Bool
    let probability: Float
    let events: [VadDebugEvent]
}

actor VadSpeechGate {
    private let vadManager: VadManager
    private let config: VadSegmentationConfig
    private let maxSpeechSamples: Int
    private let carryoverSamples: Int
    private var streamState = VadStreamState.initial()
    private var pendingSamples: [Float] = []
    private var activeSpeechSamples: [Float] = []

    init(vadManager: VadManager, config: VadSegmentationConfig) {
        self.vadManager = vadManager
        self.config = config
        self.maxSpeechSamples = Int(config.maxSpeechDuration * Double(VadManager.sampleRate))
        self.carryoverSamples = Int(config.speechPadding * Double(VadManager.sampleRate))
    }

    fileprivate func append(samples: [Float]) async throws -> VadGateResult {
        pendingSamples.append(contentsOf: samples)

        var finalized: [[Float]] = []
        var probability: Float = 0
        var events: [VadDebugEvent] = []

        while pendingSamples.count >= VadManager.chunkSize {
            let chunk = Array(pendingSamples.prefix(VadManager.chunkSize))
            pendingSamples.removeFirst(VadManager.chunkSize)

            let result = try await vadManager.processStreamingChunk(
                chunk,
                state: streamState,
                config: config,
                returnSeconds: false,
                timeResolution: 1
            )
            streamState = result.state
            probability = result.probability

            if streamState.triggered || result.event?.kind == .speechStart {
                activeSpeechSamples.append(contentsOf: chunk)
            }

            if result.event?.kind == .speechStart {
                events.append(.speechStart(probability: result.probability))
            }

            if result.event?.kind == .speechEnd, !activeSpeechSamples.isEmpty {
                events.append(.speechEnd(probability: result.probability))
                finalized.append(activeSpeechSamples)
                activeSpeechSamples.removeAll(keepingCapacity: true)
                continue
            }

            if streamState.triggered, activeSpeechSamples.count >= maxSpeechSamples, !activeSpeechSamples.isEmpty {
                let forcedSeconds = Double(activeSpeechSamples.count) / Double(VadManager.sampleRate)
                events.append(.forcedSplit(probability: result.probability, seconds: forcedSeconds))
                finalized.append(activeSpeechSamples)
                if carryoverSamples > 0 {
                    activeSpeechSamples = Array(activeSpeechSamples.suffix(carryoverSamples))
                } else {
                    activeSpeechSamples.removeAll(keepingCapacity: true)
                }
            }
        }

        return VadGateResult(
            finalizedUtterances: finalized,
            isSpeechActive: streamState.triggered,
            probability: probability,
            events: events
        )
    }

    func finish() -> [Float]? {
        if streamState.triggered {
            activeSpeechSamples.append(contentsOf: pendingSamples)
        }
        pendingSamples.removeAll(keepingCapacity: true)
        streamState = VadStreamState.initial()

        guard !activeSpeechSamples.isEmpty else { return nil }
        let result = activeSpeechSamples
        activeSpeechSamples.removeAll(keepingCapacity: true)
        return result
    }
}

actor FixedIntervalBuffer {
    private let chunkSamples: Int
    private var samples: [Float] = []

    init(chunkDuration: Double, sampleRate: Int = 16_000) {
        self.chunkSamples = Int(chunkDuration * Double(sampleRate))
    }

    func append(_ newSamples: [Float]) -> [[Float]] {
        samples.append(contentsOf: newSamples)
        var chunks: [[Float]] = []
        while samples.count >= chunkSamples {
            chunks.append(Array(samples.prefix(chunkSamples)))
            samples.removeFirst(chunkSamples)
        }
        return chunks
    }

    func finish() -> [Float]? {
        guard !samples.isEmpty else { return nil }
        let result = samples
        samples.removeAll(keepingCapacity: true)
        return result
    }
}

@MainActor
final class AsrController {
    private let audioEngine = AVAudioEngine()
    private let audioConverter = AudioConverter()
    private let capturedAudioStore = CapturedAudioStore()
    private let audioProcessingQueue = DispatchQueue(label: "AudioClaw.AudioProcessing", qos: .utility)

    private var asrManager = AsrManager()
    private var ctcZhCnManager: CtcZhCnManager?
    private var vadManager: VadManager?
    private var cachedModels: AsrModels?
    private var activeModelVersion: AsrModelVersion?
    private var updatesContinuation: AsyncStream<SlidingWindowTranscriptionUpdate>.Continuation?
    private var gate: VadSpeechGate?
    private var fixedBuffer: FixedIntervalBuffer?
    private var transcriptionTask: Task<Void, Never>?
    private var fileTranscriptionTask: Task<Void, Never>?
    private var pendingUtterance: [Float]?
    private var isTranscribingUtterance = false
    private var committedTranscriptSegments: [String] = []

    func availableInputDevices() -> [AudioInputDevice] {
        AVCaptureDevice.DiscoverySession(
            deviceTypes: [.microphone, .external, .continuityCamera],
            mediaType: .audio,
            position: .unspecified
        ).devices
            .map { AudioInputDevice(id: $0.uniqueID, name: $0.localizedName) }
            .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    func defaultInputDevice() -> AudioInputDevice? {
        guard let device = AVCaptureDevice.default(for: .audio) else { return nil }
        return AudioInputDevice(id: device.uniqueID, name: device.localizedName)
    }

    private func ensureModelsLoaded(
        modelVersion: AsrModelVersion,
        needsVad: Bool,
        progressHandler: @escaping DownloadUtils.ProgressHandler
    ) async throws {
        if let currentVersion = activeModelVersion, currentVersion != modelVersion {
            cachedModels = nil
            activeModelVersion = nil
            ctcZhCnManager = nil
        }

        if modelVersion == .ctcZhCn {
            if ctcZhCnManager == nil {
                ctcZhCnManager = try await CtcZhCnManager.load(progressHandler: progressHandler)
            }
            activeModelVersion = modelVersion
        } else {
            let models: AsrModels
            if let cachedModels {
                models = cachedModels
            } else {
                models = try await AsrModels.downloadAndLoad(version: modelVersion, progressHandler: progressHandler)
                self.cachedModels = models
                activeModelVersion = modelVersion
            }
            asrManager = AsrManager()
            try await asrManager.loadModels(models)
        }

        if needsVad, vadManager == nil {
            vadManager = try await VadManager(config: VadConfig(defaultThreshold: 0.4))
        }
    }

    func start(
        modelVersion: AsrModelVersion,
        selectedDeviceID _: String?,
        vadEnabled: Bool,
        progressHandler: @escaping DownloadUtils.ProgressHandler,
        onAudioLevel: @escaping @Sendable (Float) -> Void,
        onVadStateChanged: @escaping @Sendable (Bool, Float) -> Void,
        onVadEvent: @escaping @Sendable (VadDebugEvent) -> Void
    ) async throws -> AsyncStream<SlidingWindowTranscriptionUpdate> {
        try await ensureModelsLoaded(
            modelVersion: modelVersion,
            needsVad: vadEnabled,
            progressHandler: progressHandler
        )

        if vadEnabled {
            guard let vadManager else {
                throw VadError.notInitialized
            }
            gate = VadSpeechGate(
                vadManager: vadManager,
                config: VadSegmentationConfig(
                    minSpeechDuration: 0.12,
                    minSilenceDuration: 0.15,
                    maxSpeechDuration: 20.0,
                    speechPadding: 0.1
                )
            )
            fixedBuffer = nil
        } else {
            gate = nil
            fixedBuffer = FixedIntervalBuffer(chunkDuration: 15.0)
        }

        await capturedAudioStore.reset()
        committedTranscriptSegments.removeAll(keepingCapacity: true)
        transcriptionTask?.cancel()
        transcriptionTask = nil
        pendingUtterance = nil
        isTranscribingUtterance = false
        updatesContinuation?.finish()

        let stream = AsyncStream<SlidingWindowTranscriptionUpdate> { continuation in
            self.updatesContinuation = continuation
        }

        try startAudioEngine(
            onAudioLevel: onAudioLevel,
            onVadStateChanged: onVadStateChanged,
            onVadEvent: onVadEvent
        )
        return stream
    }

    func stop() async throws -> String {
        stopAudioEngine()

        if let trailingUtterance = await gate?.finish() {
            try await transcribeUtteranceAndPublish(trailingUtterance)
        }
        if let trailingChunk = await fixedBuffer?.finish() {
            try await transcribeUtteranceAndPublish(trailingChunk)
        }
        gate = nil
        fixedBuffer = nil

        await transcriptionTask?.value
        transcriptionTask = nil
        pendingUtterance = nil
        isTranscribingUtterance = false

        updatesContinuation?.finish()
        updatesContinuation = nil
        return committedTranscriptSegments.joined(separator: " ")
    }

    func saveDebugWav() async throws -> URL {
        try await capturedAudioStore.writeDebugWav()
    }

    func transcribeFile(
        url: URL,
        modelVersion: AsrModelVersion,
        vadEnabled: Bool,
        modelProgressHandler: @escaping DownloadUtils.ProgressHandler,
        onProgress: @escaping @Sendable (Double) -> Void
    ) async throws -> AsyncStream<SlidingWindowTranscriptionUpdate> {
        try await ensureModelsLoaded(
            modelVersion: modelVersion,
            needsVad: vadEnabled,
            progressHandler: modelProgressHandler
        )

        let resolvedVadManager: VadManager?
        if vadEnabled {
            guard let vadManager else { throw VadError.notInitialized }
            resolvedVadManager = vadManager
        } else {
            resolvedVadManager = nil
        }

        let accessing = url.startAccessingSecurityScopedResource()
        let converter = audioConverter
        let samples: [Float]
        do {
            samples = try await Task.detached(priority: .userInitiated) {
                defer { if accessing { url.stopAccessingSecurityScopedResource() } }
                return try converter.resampleAudioFile(url)
            }.value
        } catch {
            if accessing { url.stopAccessingSecurityScopedResource() }
            throw error
        }

        let fileGate: VadSpeechGate? = resolvedVadManager.map { vm in
            VadSpeechGate(
                vadManager: vm,
                config: VadSegmentationConfig(
                    minSpeechDuration: 0.12,
                    minSilenceDuration: 0.15,
                    maxSpeechDuration: 20.0,
                    speechPadding: 0.1
                )
            )
        }

        committedTranscriptSegments.removeAll(keepingCapacity: true)
        updatesContinuation?.finish()
        let stream = AsyncStream<SlidingWindowTranscriptionUpdate> { continuation in
            self.updatesContinuation = continuation
        }

        let capturedModelVersion = activeModelVersion
        let capturedAsrManager = self.asrManager
        let capturedCtcZhCnManager = self.ctcZhCnManager

        fileTranscriptionTask?.cancel()
        fileTranscriptionTask = Task { [weak self] in
            guard let self else { return }

            let totalSamples = samples.count
            var utterances: [[Float]] = []

            if let fileGate {
                let processingChunkSize = 8_192
                var processedCount = 0

                // VAD phase: report 0% – 50%
                while processedCount < totalSamples, !Task.isCancelled {
                    let end = min(processedCount + processingChunkSize, totalSamples)
                    let chunk = Array(samples[processedCount..<end])
                    processedCount = end

                    if let result = try? await fileGate.append(samples: chunk) {
                        utterances.append(contentsOf: result.finalizedUtterances)
                    }
                    onProgress(Double(processedCount) / Double(totalSamples) * 0.5)
                }

                if !Task.isCancelled, let trailing = await fileGate.finish() {
                    utterances.append(trailing)
                }
            } else {
                // No VAD: split into fixed 15s chunks
                let chunkSamples = 15 * 16_000
                var idx = 0
                while idx < totalSamples, !Task.isCancelled {
                    let end = min(idx + chunkSamples, totalSamples)
                    utterances.append(Array(samples[idx..<end]))
                    idx = end
                }
                onProgress(0.5)
            }

            // Transcription phase: report 50% – 100%
            let totalUtterances = utterances.count
            for (i, utterance) in utterances.enumerated() {
                guard !Task.isCancelled else { break }

                if let text = try? await Self.transcribeUtterance(
                    samples: utterance,
                    modelVersion: capturedModelVersion,
                    asrManager: capturedAsrManager,
                    ctcZhCnManager: capturedCtcZhCnManager
                ) {
                    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty {
                        await MainActor.run {
                            self.committedTranscriptSegments.append(trimmed)
                            self.updatesContinuation?.yield(
                                SlidingWindowTranscriptionUpdate(
                                    text: trimmed,
                                    isConfirmed: true,
                                    confidence: 1.0,
                                    timestamp: Date()
                                )
                            )
                        }
                    }
                }
                onProgress(0.5 + Double(i + 1) / Double(max(1, totalUtterances)) * 0.5)
            }

            await MainActor.run {
                self.updatesContinuation?.finish()
                self.updatesContinuation = nil
            }
        }

        return stream
    }

    func cancelFileTranscription() {
        fileTranscriptionTask?.cancel()
        fileTranscriptionTask = nil
        updatesContinuation?.finish()
        updatesContinuation = nil
    }

    func currentInputDeviceName() -> String {
        defaultInputDevice()?.name ?? "System Default Input"
    }

    private func startAudioEngine(
        onAudioLevel: @escaping @Sendable (Float) -> Void,
        onVadStateChanged: @escaping @Sendable (Bool, Float) -> Void,
        onVadEvent: @escaping @Sendable (VadDebugEvent) -> Void
    ) throws {
        stopAudioEngine()

        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        let converter = audioConverter
        let capturedAudioStore = self.capturedAudioStore
        let audioProcessingQueue = self.audioProcessingQueue
        let gate = self.gate
        let fixedBuffer = self.fixedBuffer

        inputNode.installTap(onBus: 0, bufferSize: 4_096, format: inputFormat) { buffer, _ in
            guard let copiedBuffer = buffer.deepCopy() else { return }

            let level = copiedBuffer.normalizedLevel
            onAudioLevel(level)

            audioProcessingQueue.async { [weak self] in
                guard self != nil else { return }
                guard let resampled = try? converter.resampleBuffer(copiedBuffer) else { return }

                Task {
                    await capturedAudioStore.append(resampled)

                    if let gate {
                        guard let gateResult = try? await gate.append(samples: resampled) else { return }
                        onVadStateChanged(gateResult.isSpeechActive, gateResult.probability)
                        for event in gateResult.events {
                            onVadEvent(event)
                        }

                        for utterance in gateResult.finalizedUtterances {
                            guard let self else { return }
                            try? await self.transcribeUtteranceAndPublish(utterance)
                        }
                    } else if let fixedBuffer {
                        let chunks = await fixedBuffer.append(resampled)
                        for chunk in chunks {
                            guard let self else { return }
                            try? await self.transcribeUtteranceAndPublish(chunk)
                        }
                    }
                }
            }
        }

        audioEngine.prepare()
        try audioEngine.start()
    }

    private func stopAudioEngine() {
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        audioEngine.reset()
    }

    private func transcribeUtteranceAndPublish(_ samples: [Float]) async throws {
        guard samples.count >= 4_000 else { return }
        if isTranscribingUtterance {
            // Keep only the latest fallback/VAD segment so ASR backlog cannot starve audio capture.
            pendingUtterance = samples
            return
        }

        isTranscribingUtterance = true
        pendingUtterance = nil
        scheduleTranscriptionLoop(initialSamples: samples)
    }

    private func scheduleTranscriptionLoop(initialSamples: [Float]) {
        let modelVersion = activeModelVersion
        let asrManager = self.asrManager
        let ctcZhCnManager = self.ctcZhCnManager
        let continuation = updatesContinuation

        transcriptionTask = Task<Void, Never> { [weak self] in
            guard let self else { return }

            var currentSamples: [Float]? = initialSamples

            while !Task.isCancelled, let samples = currentSamples {
                if
                    let text = try? await Self.transcribeUtterance(
                        samples: samples,
                        modelVersion: modelVersion,
                        asrManager: asrManager,
                        ctcZhCnManager: ctcZhCnManager
                    )
                {
                    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty {
                        await MainActor.run {
                            self.committedTranscriptSegments.append(trimmed)
                            continuation?.yield(
                                SlidingWindowTranscriptionUpdate(
                                    text: trimmed,
                                    isConfirmed: true,
                                    confidence: 1.0,
                                    timestamp: Date()
                                )
                            )
                        }
                    }
                }

                currentSamples = await MainActor.run {
                    let next = self.pendingUtterance
                    self.pendingUtterance = nil
                    if next == nil {
                        self.isTranscribingUtterance = false
                    }
                    return next
                }
            }
        }
    }

    private static func transcribeUtterance(
        samples: [Float],
        modelVersion: AsrModelVersion?,
        asrManager: AsrManager,
        ctcZhCnManager: CtcZhCnManager?
    ) async throws -> String {
        if modelVersion == .ctcZhCn {
            guard let ctcZhCnManager else { return "" }
            return try await ctcZhCnManager.transcribe(audio: samples)
        }

        var decoderState = TdtDecoderState.make(decoderLayers: await asrManager.decoderLayerCount)
        let result = try await asrManager.transcribe(samples, decoderState: &decoderState)
        return result.text
    }
}

private extension AVAudioPCMBuffer {
    var normalizedLevel: Float {
        let frameCount = Int(frameLength)
        guard frameCount > 0 else { return 0 }

        let channelCount = Int(format.channelCount)
        guard channelCount > 0 else { return 0 }

        switch format.commonFormat {
        case .pcmFormatFloat32:
            guard let channels = floatChannelData else { return 0 }
            var sum: Float = 0
            for channel in 0..<channelCount {
                let samples = UnsafeBufferPointer(start: channels[channel], count: frameCount)
                for sample in samples {
                    sum += sample * sample
                }
            }
            let mean = sum / Float(frameCount * channelCount)
            return min(max(sqrt(mean) * 3.0, 0), 1)
        case .pcmFormatInt16:
            guard let channels = int16ChannelData else { return 0 }
            var sum: Float = 0
            for channel in 0..<channelCount {
                let samples = UnsafeBufferPointer(start: channels[channel], count: frameCount)
                for sample in samples {
                    let normalized = Float(sample) / Float(Int16.max)
                    sum += normalized * normalized
                }
            }
            let mean = sum / Float(frameCount * channelCount)
            return min(max(sqrt(mean) * 3.0, 0), 1)
        case .pcmFormatInt32:
            guard let channels = int32ChannelData else { return 0 }
            var sum: Float = 0
            for channel in 0..<channelCount {
                let samples = UnsafeBufferPointer(start: channels[channel], count: frameCount)
                for sample in samples {
                    let normalized = Float(sample) / Float(Int32.max)
                    sum += normalized * normalized
                }
            }
            let mean = sum / Float(frameCount * channelCount)
            return min(max(sqrt(mean) * 3.0, 0), 1)
        default:
            return 0
        }
    }

    func deepCopy() -> AVAudioPCMBuffer? {
        guard let copy = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCapacity) else {
            return nil
        }

        copy.frameLength = frameLength

        let channelCount = Int(format.channelCount)
        let frameCount = Int(frameLength)

        switch format.commonFormat {
        case .pcmFormatFloat32:
            guard let source = floatChannelData, let destination = copy.floatChannelData else {
                return nil
            }
            for channel in 0..<channelCount {
                memcpy(destination[channel], source[channel], frameCount * MemoryLayout<Float>.size)
            }
        case .pcmFormatInt16:
            guard let source = int16ChannelData, let destination = copy.int16ChannelData else {
                return nil
            }
            for channel in 0..<channelCount {
                memcpy(destination[channel], source[channel], frameCount * MemoryLayout<Int16>.size)
            }
        case .pcmFormatInt32:
            guard let source = int32ChannelData, let destination = copy.int32ChannelData else {
                return nil
            }
            for channel in 0..<channelCount {
                memcpy(destination[channel], source[channel], frameCount * MemoryLayout<Int32>.size)
            }
        default:
            return nil
        }

        return copy
    }
}
