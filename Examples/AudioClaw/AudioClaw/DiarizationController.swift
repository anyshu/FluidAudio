import FluidAudio
import Foundation

// MARK: - Engine selection

enum DiarizationEngineType: String, CaseIterable, Identifiable {
    case sortformer = "Sortformer"
    case lseend = "LS-EEND"
    case offline = "Offline VBx"

    var id: String { rawValue }

    var isStreamingCapable: Bool { self != .offline }

    var detail: String {
        switch self {
        case .sortformer: return "Max 4 speakers • ~1s latency"
        case .lseend: return "Up to 10 speakers • ~100ms latency"
        case .offline: return "Unlimited speakers • batch only"
        }
    }
}

// MARK: - Output types

struct DiarizedTranscriptSegment: Identifiable, Sendable {
    let id = UUID()
    let speakerIndex: Int
    let text: String
    let startTime: Float
    let endTime: Float

    var speakerLabel: String { "Speaker \(speakerIndex + 1)" }
}

struct SpeakerTimeRange: Sendable {
    let speakerIndex: Int
    let startTime: Float
    let endTime: Float
}

enum DiarizationError: LocalizedError {
    case notInitialized
    case offlineEngineNotSupportedForStreaming

    var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "Diarization engine is not initialized."
        case .offlineEngineNotSupportedForStreaming:
            return "Offline VBx is batch-only. Use Sortformer or LS-EEND for live recording."
        }
    }
}

// MARK: - Controller

actor DiarizationController {
    private enum StreamingEngine {
        case sortformer(SortformerDiarizer)
        case lseend(LSEENDDiarizer)
    }

    // Multi-engine caches: once an engine is loaded we keep it around so switching
    // back doesn't trigger another ANE compile. Sortformer's heavy MLModel object
    // is cached separately so threshold changes only rebuild the lightweight wrapper.
    private var sortformerModelsCache: [String: SortformerModels] = [:]
    private var sortformerDiarizer: SortformerDiarizer?
    private var sortformerActiveKey: String?

    private var lseendCache: [LSEENDVariant: LSEENDDiarizer] = [:]
    private var lseendActiveVariant: LSEENDVariant?

    private var offlineManager: OfflineDiarizerManager?

    private(set) var engineType: DiarizationEngineType = .sortformer
    private(set) var lseendVariant: LSEENDVariant = .dihard3
    private(set) var isInitialized: Bool = false
    private var cumulativeSamples: Int = 0

    /// Initialize (or activate) the requested engine. Reuses cached engines when possible.
    /// First call for each engine triggers download + ANE compile; subsequent activations are instant.
    func initialize(
        engine: DiarizationEngineType,
        lseendVariant: LSEENDVariant = .dihard3,
        sortformerConfig: SortformerConfig = .fastV2_1,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws {
        switch engine {
        case .sortformer:
            let variantKey = ModelNames.Sortformer.bundle(for: sortformerConfig) ?? "default"
            // Load + compile the MLModel only once per variant.
            let models: SortformerModels
            if let cached = sortformerModelsCache[variantKey] {
                models = cached
            } else {
                models = try await SortformerModels.loadFromHuggingFace(
                    config: sortformerConfig,
                    progressHandler: progressHandler
                )
                sortformerModelsCache[variantKey] = models
            }
            // Rebuild the lightweight diarizer wrapper if config changed (e.g. threshold slider).
            let diarizerKey = "\(variantKey):\(sortformerConfig.predScoreThreshold)"
            if sortformerActiveKey != diarizerKey {
                let diarizer = SortformerDiarizer(config: sortformerConfig)
                diarizer.initialize(models: models)
                sortformerDiarizer = diarizer
                sortformerActiveKey = diarizerKey
            } else {
                // Same config — just reset state for a fresh session.
                sortformerDiarizer?.reset()
            }

        case .lseend:
            if lseendCache[lseendVariant] == nil {
                let diarizer = LSEENDDiarizer()
                let descriptor = try await LSEENDModelDescriptor.loadFromHuggingFace(
                    variant: lseendVariant,
                    progressHandler: progressHandler
                )
                try diarizer.initialize(descriptor: descriptor)
                lseendCache[lseendVariant] = diarizer
            } else {
                lseendCache[lseendVariant]?.reset()
            }
            lseendActiveVariant = lseendVariant

        case .offline:
            if offlineManager == nil {
                let manager = OfflineDiarizerManager()
                try await manager.prepareModels()
                offlineManager = manager
            }
        }

        self.engineType = engine
        self.lseendVariant = lseendVariant
        cumulativeSamples = 0
        isInitialized = true
    }

    /// Resolves the currently active streaming engine from the per-engine caches.
    private var activeStreamingEngine: StreamingEngine? {
        switch engineType {
        case .sortformer:
            return sortformerDiarizer.map { .sortformer($0) }
        case .lseend:
            guard let variant = lseendActiveVariant, let d = lseendCache[variant] else { return nil }
            return .lseend(d)
        case .offline:
            return nil
        }
    }

    /// Whether the controller is ready to handle streaming (microphone) input with the current engine.
    var canHandleStreaming: Bool {
        isInitialized && engineType.isStreamingCapable
    }

    /// Feed 16 kHz mono samples to the streaming diarizer. Errors are swallowed (non-fatal for ASR).
    func feedAudio(_ samples: [Float]) {
        guard let engine = activeStreamingEngine else { return }
        do {
            switch engine {
            case .sortformer(let d):
                try d.addAudio(samples, sourceSampleRate: nil)
                _ = try d.process()
            case .lseend(let d):
                try d.addAudio(samples, sourceSampleRate: nil)
                _ = try d.process()
            }
            cumulativeSamples += samples.count
        } catch {
            // Diarization is best-effort; never break ASR
        }
    }

    /// Dominant speaker index in the given time window (seconds since recording start).
    func dominantSpeakerIndex(from startTime: Float, to endTime: Float) -> Int {
        let timeline: DiarizerTimeline
        switch activeStreamingEngine {
        case .sortformer(let d): timeline = d.timeline
        case .lseend(let d): timeline = d.timeline
        case nil: return 0
        }

        var speakerDuration: [Int: Float] = [:]
        for (index, speaker) in timeline.speakers {
            for seg in speaker.finalizedSegments {
                let lo = max(seg.startTime, startTime)
                let hi = min(seg.endTime, endTime)
                if hi > lo {
                    speakerDuration[index, default: 0] += hi - lo
                }
            }
        }
        return speakerDuration.max(by: { $0.value < $1.value })?.key ?? 0
    }

    var currentTimeSeconds: Float { Float(cumulativeSamples) / 16_000.0 }

    /// Process a complete file and return all speaker time ranges.
    /// Works for any engine (streaming or offline).
    func processFileToSegments(_ url: URL) async throws -> [SpeakerTimeRange] {
        switch engineType {
        case .sortformer, .lseend:
            let timeline = try await processFileWithStreamingEngine(url)
            var ranges: [SpeakerTimeRange] = []
            for (index, speaker) in timeline.speakers {
                for seg in speaker.finalizedSegments {
                    ranges.append(
                        SpeakerTimeRange(
                            speakerIndex: index,
                            startTime: seg.startTime,
                            endTime: seg.endTime
                        )
                    )
                }
            }
            return ranges.sorted { $0.startTime < $1.startTime }

        case .offline:
            guard let manager = offlineManager else {
                throw DiarizationError.notInitialized
            }
            let result = try await manager.process(url)
            // Map string speakerId to a stable integer index
            let uniqueIds = Set(result.segments.map(\.speakerId)).sorted()
            let idToIndex = Dictionary(uniqueKeysWithValues: uniqueIds.enumerated().map { ($1, $0) })
            return result.segments.map { seg in
                SpeakerTimeRange(
                    speakerIndex: idToIndex[seg.speakerId] ?? 0,
                    startTime: seg.startTimeSeconds,
                    endTime: seg.endTimeSeconds
                )
            }
        }
    }

    private func processFileWithStreamingEngine(_ url: URL) async throws -> DiarizerTimeline {
        switch activeStreamingEngine {
        case .sortformer(let d):
            return try await Task.detached(priority: .userInitiated) { [d] in
                try d.processComplete(
                    audioFileURL: url,
                    keepingEnrolledSpeakers: nil,
                    finalizeOnCompletion: true,
                    progressCallback: nil
                )
            }.value
        case .lseend(let d):
            return try await Task.detached(priority: .userInitiated) { [d] in
                try d.processComplete(
                    audioFileURL: url,
                    keepingEnrolledSpeakers: nil,
                    finalizeOnCompletion: true,
                    progressCallback: nil
                )
            }.value
        case nil:
            throw DiarizationError.notInitialized
        }
    }

    /// Reset only the currently active engine's session state. Cached engines stay loaded.
    func reset() {
        switch activeStreamingEngine {
        case .sortformer(let d): d.reset()
        case .lseend(let d): d.reset()
        case nil: break
        }
        cumulativeSamples = 0
    }

    /// Tear down all cached engines. Call only when the app is shutting down.
    func cleanup() {
        for d in lseendCache.values { d.cleanup() }
        lseendCache.removeAll()
        lseendActiveVariant = nil

        sortformerDiarizer?.cleanup()
        sortformerDiarizer = nil
        sortformerActiveKey = nil
        sortformerModelsCache.removeAll()

        offlineManager = nil
        isInitialized = false
        cumulativeSamples = 0
    }
}

// MARK: - Helpers

enum SpeakerLookup {
    /// Find the speaker with the most total overlap in the given range.
    static func dominantSpeaker(
        from segments: [SpeakerTimeRange],
        in range: ClosedRange<Float>
    ) -> Int {
        var duration: [Int: Float] = [:]
        for seg in segments {
            let lo = max(seg.startTime, range.lowerBound)
            let hi = min(seg.endTime, range.upperBound)
            if hi > lo {
                duration[seg.speakerIndex, default: 0] += hi - lo
            }
        }
        return duration.max(by: { $0.value < $1.value })?.key ?? 0
    }
}
