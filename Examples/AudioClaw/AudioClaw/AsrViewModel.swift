import AVFoundation
import FluidAudio
import Foundation
import SwiftUI

@MainActor
final class AsrViewModel: ObservableObject {
    @Published private(set) var availableInputDevices: [AudioInputDevice] = []
    @Published var selectedInputDeviceID: String = "" {
        didSet {
            updateDisplayedInputDeviceName()
        }
    }
    @Published var selectedModel: DemoModel = .english
    @Published var vadEnabled: Bool = true
    @Published var diarizationEnabled: Bool = false
    @Published var selectedDiarizationEngine: DiarizationEngineType = .sortformer
    @Published var selectedLSEENDVariant: LSEENDVariant = .dihard3
    @Published var sortformerPredScoreThreshold: Double = 0.18 {
        didSet {
            // Force re-init on next run so the new threshold takes effect.
            loadedDiarizationKey = nil
        }
    }
    @Published private(set) var diarizedSegments: [DiarizedTranscriptSegment] = []
    @Published var pendingFileURL: URL? = nil
    @Published private(set) var fullTranscript: String = ""
    @Published private(set) var liveTranscript: String = ""
    @Published private(set) var statusText: String = "Idle"
    @Published private(set) var downloadProgress: Double?
    @Published private(set) var isRecording: Bool = false
    @Published private(set) var isPreparing: Bool = false
    @Published private(set) var sessionDuration: TimeInterval = 0
    @Published private(set) var inputDeviceName: String = "System Default Input"
    @Published private(set) var inputLevel: Float = 0
    @Published private(set) var vadProbability: Float = 0
    @Published private(set) var isVadSpeechActive: Bool = false
    @Published private(set) var vadSpeechStartCount: Int = 0
    @Published private(set) var vadSpeechEndCount: Int = 0
    @Published private(set) var vadForcedSplitCount: Int = 0
    @Published private(set) var lastVadEventText: String = "No VAD events yet"
    @Published private(set) var lastPartialText: String = ""
    @Published private(set) var lastConfidence: Float = 0
    @Published private(set) var recentUpdates: [RecentTranscriptUpdate] = []
    @Published private(set) var lastSavedDebugWavPath: String = ""
    @Published var isShowingError: Bool = false
    @Published var errorMessage: String = ""
    @Published var isShowingFilePicker: Bool = false
    @Published private(set) var isTranscribingFile: Bool = false
    @Published private(set) var fileTranscriptionProgress: Double?

    var isBusy: Bool { isPreparing || isTranscribingFile }

    var primaryButtonTitle: String {
        isRecording ? "Stop Recording" : "Start Recording"
    }

    var sessionSummary: String {
        let seconds = Int(sessionDuration.rounded())
        return isRecording ? "REC  \(seconds)s" : "Ready"
    }

    var displayedTranscript: String {
        fullTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var selectedModelShortTitle: String {
        selectedModel.shortTitle
    }

    private let controller = AsrController()
    private let diarizationController = DiarizationController()
    private var updatesTask: Task<Void, Never>?
    private var fileUpdatesTask: Task<Void, Never>?
    private var timerTask: Task<Void, Never>?
    private var levelDecayTask: Task<Void, Never>?
    private var sessionStartedAt: Date?
    private var loadedDiarizationKey: String?

    func prepareIfNeeded() async {
        let devices = controller.availableInputDevices()
        let defaultDeviceID = controller.defaultInputDevice()?.id ?? devices.first?.id ?? ""
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            availableInputDevices = devices
            selectedInputDeviceID = defaultDeviceID
            statusText = "Idle"
            updateDisplayedInputDeviceName()
        }
    }

    func clearTranscript() {
        fullTranscript = ""
        liveTranscript = ""
        lastPartialText = ""
        lastConfidence = 0
        recentUpdates = []
        diarizedSegments = []
        lastSavedDebugWavPath = ""
        vadSpeechStartCount = 0
        vadSpeechEndCount = 0
        vadForcedSplitCount = 0
        lastVadEventText = isRecording ? "Waiting for VAD events" : "No VAD events yet"
        statusText = isRecording ? "Listening..." : "Idle"
    }

    /// Ensure the diarization engine is initialized for the requested configuration.
    /// Reloads if engine or LS-EEND variant changed since last load.
    private func ensureDiarizationLoaded(forStreaming: Bool) async throws {
        let engine = selectedDiarizationEngine
        if forStreaming, !engine.isStreamingCapable {
            throw DiarizationError.offlineEngineNotSupportedForStreaming
        }
        let thresholdKey = String(format: "%.3f", sortformerPredScoreThreshold)
        let key = "\(engine.rawValue):\(selectedLSEENDVariant.rawValue):\(thresholdKey)"
        if loadedDiarizationKey == key, await diarizationController.isInitialized {
            return
        }
        statusText = "Loading \(engine.rawValue) diarizer..."
        try await diarizationController.initialize(
            engine: engine,
            lseendVariant: selectedLSEENDVariant,
            sortformerConfig: SortformerConfig(
                modelVariant: .balancedV2,
                chunkLen: 6,
                chunkLeftContext: 1,
                chunkRightContext: 7,
                fifoLen: 188,
                spkcacheLen: 188,
                spkcacheUpdatePeriod: 144,
                predScoreThreshold: Float(sortformerPredScoreThreshold)
            ),
            progressHandler: { [weak self] progress in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.downloadProgress = progress.fractionCompleted
                    self.statusText = Self.makeStatusText(for: progress)
                }
            }
        )
        loadedDiarizationKey = key
    }

    func toggleRecording() async {
        if isRecording {
            await stopRecording()
            return
        }

        await startRecording()
    }

    private func startRecording() async {
        guard !isPreparing else { return }

        do {
            isPreparing = true
            statusText = "Requesting microphone access..."
            downloadProgress = nil
            try await requestMicrophonePermission()

            statusText = "Loading \(selectedModel.title)..."

            if diarizationEnabled {
                try await ensureDiarizationLoaded(forStreaming: true)
                controller.diarizationController = diarizationController
            } else {
                controller.diarizationController = nil
            }

            var onDiarizedSegment: (@Sendable (DiarizedTranscriptSegment) -> Void)? = nil
            if diarizationEnabled {
                onDiarizedSegment = { [weak self] segment in
                    Task { @MainActor in
                        self?.appendDiarizedSegment(segment)
                    }
                }
            }

            let updates = try await controller.start(
                modelVersion: selectedModel.asrModelVersion,
                selectedDeviceID: selectedInputDeviceID.isEmpty ? nil : selectedInputDeviceID,
                vadEnabled: vadEnabled,
                progressHandler: { [weak self] progress in
                    Task { @MainActor [weak self] in
                        guard let self else { return }
                        downloadProgress = progress.fractionCompleted
                        statusText = Self.makeStatusText(for: progress)
                    }
                },
                onAudioLevel: { [weak self] level in
                    Task { @MainActor [weak self] in
                        self?.inputLevel = level
                    }
                },
                onVadStateChanged: { [weak self] isSpeechActive, probability in
                    Task { @MainActor [weak self] in
                        self?.isVadSpeechActive = isSpeechActive
                        self?.vadProbability = probability
                    }
                },
                onVadEvent: { [weak self] event in
                    Task { @MainActor [weak self] in
                        self?.handleVadEvent(event)
                    }
                },
                onDiarizedSegment: onDiarizedSegment
            )

            updatesTask?.cancel()
            updatesTask = Task { [weak self] in
                guard let self else { return }
                for await update in updates {
                    await self.consume(update: update)
                }
            }

            downloadProgress = nil
            updateDisplayedInputDeviceName()
            statusText = "Listening on \(inputDeviceName)... first text usually appears in 2-3s"
            isRecording = true
            isPreparing = false
            lastPartialText = ""
            lastConfidence = 0
            recentUpdates = []
            lastSavedDebugWavPath = ""
            sessionStartedAt = Date()
            isVadSpeechActive = false
            vadProbability = 0
            vadSpeechStartCount = 0
            vadSpeechEndCount = 0
            vadForcedSplitCount = 0
            lastVadEventText = "Waiting for VAD events"
            startSessionClock()
            startLevelDecay()
        } catch {
            present(error: error)
            isPreparing = false
            isRecording = false
            sessionStartedAt = nil
            stopSessionClock()
            stopLevelDecay()
        }
    }

    private func stopRecording() async {
        guard isRecording || isPreparing else { return }

        isPreparing = true
        statusText = "Finalizing transcript..."

        do {
            let finalTranscript = try await controller.stop()
            let trimmedFinal = finalTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmedFinal.isEmpty {
                fullTranscript = trimmedFinal
            }
            liveTranscript = ""
            lastPartialText = ""
            statusText = "Finished"
        } catch {
            present(error: error)
        }

        updatesTask?.cancel()
        updatesTask = nil
        isRecording = false
        isPreparing = false
        stopSessionClock()
        stopLevelDecay()
        inputLevel = 0
        isVadSpeechActive = false
        vadProbability = 0
        vadForcedSplitCount = 0
    }

    func saveDebugWav() async {
        do {
            let url = try await controller.saveDebugWav()
            lastSavedDebugWavPath = url.path
            statusText = "Saved debug WAV to Downloads"
        } catch {
            present(error: error)
        }
    }

    /// Stash a picked file URL without starting transcription. The user must press "Run".
    func selectFile(url: URL) {
        pendingFileURL = url
        statusText = "Selected \(url.lastPathComponent) — press Run to transcribe"
    }

    func clearPendingFile() {
        pendingFileURL = nil
        if !isRecording, !isTranscribingFile {
            statusText = "Idle"
        }
    }

    /// Run transcription on the currently selected pending file.
    func runPendingFile() async {
        guard let url = pendingFileURL else { return }
        await transcribeFile(url: url)
    }

    func transcribeFile(url: URL) async {
        guard !isBusy, !isRecording else { return }

        isTranscribingFile = true
        fileTranscriptionProgress = 0
        downloadProgress = nil
        fullTranscript = ""
        liveTranscript = ""
        recentUpdates = []
        diarizedSegments = []
        statusText = "Loading \(selectedModel.title)..."

        do {
            if diarizationEnabled {
                try await ensureDiarizationLoaded(forStreaming: false)
                controller.diarizationController = diarizationController
            } else {
                controller.diarizationController = nil
            }

            var onDiarizedSegment: (@Sendable (DiarizedTranscriptSegment) -> Void)? = nil
            if diarizationEnabled {
                onDiarizedSegment = { [weak self] segment in
                    Task { @MainActor in
                        self?.appendDiarizedSegment(segment)
                    }
                }
            }

            // VAD is intentionally forced off for file transcription: sliding-window ASR
            // already segments long audio, and a VAD pre-pass would double the latency
            // for no quality gain.
            let updates = try await controller.transcribeFile(
                url: url,
                modelVersion: selectedModel.asrModelVersion,
                vadEnabled: false,
                modelProgressHandler: { [weak self] progress in
                    Task { @MainActor [weak self] in
                        guard let self else { return }
                        downloadProgress = progress.fractionCompleted
                        statusText = Self.makeStatusText(for: progress)
                    }
                },
                onProgress: { [weak self] progress in
                    Task { @MainActor [weak self] in
                        guard let self else { return }
                        fileTranscriptionProgress = progress
                        let pct = Int(progress * 100)
                        statusText =
                            pct < 50
                            ? "Analyzing audio... \(pct * 2)%" : "Transcribing... \(Int((progress - 0.5) * 200))%"
                    }
                },
                onDiarizedSegment: onDiarizedSegment
            )

            downloadProgress = nil
            statusText = "Transcribing \(url.lastPathComponent)..."

            fileUpdatesTask?.cancel()
            fileUpdatesTask = Task { [weak self] in
                guard let self else { return }
                for await update in updates {
                    await self.consume(update: update)
                }
                await MainActor.run {
                    self.isTranscribingFile = false
                    self.fileTranscriptionProgress = nil
                    self.statusText = self.fullTranscript.isEmpty ? "No speech detected" : "Done"
                }
            }
            await fileUpdatesTask?.value
        } catch {
            present(error: error)
            isTranscribingFile = false
            fileTranscriptionProgress = nil
            downloadProgress = nil
        }
    }

    func cancelFileTranscription() {
        controller.cancelFileTranscription()
        fileUpdatesTask?.cancel()
        fileUpdatesTask = nil
        isTranscribingFile = false
        fileTranscriptionProgress = nil
        statusText = "Cancelled"
    }

    private func consume(update: SlidingWindowTranscriptionUpdate) async {
        lastConfidence = update.confidence

        let trimmedText = update.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if update.isConfirmed {
            if !trimmedText.isEmpty {
                lastPartialText = trimmedText
                fullTranscript = mergeTranscript(fullTranscript, with: trimmedText)
                appendRecentUpdate(text: trimmedText, confidence: update.confidence, confirmed: true)
            }
            liveTranscript = ""
            return
        }

        if !trimmedText.isEmpty {
            lastPartialText = trimmedText
            liveTranscript = trimmedText
            appendRecentUpdate(text: trimmedText, confidence: update.confidence, confirmed: false)
        }
    }

    private func mergeTranscript(_ existing: String, with newText: String) -> String {
        guard !newText.isEmpty else { return existing }
        guard !existing.isEmpty else { return newText }

        if existing.hasSuffix(newText) {
            return existing
        }

        let needsSpace =
            !existing.hasSuffix(" ")
            && !newText.hasPrefix(",")
            && !newText.hasPrefix(".")
            && !newText.hasPrefix("!")
            && !newText.hasPrefix("?")

        return needsSpace ? existing + " " + newText : existing + newText
    }

    private func appendRecentUpdate(text: String, confidence: Float, confirmed: Bool) {
        let clippedText: String
        if text.count > 120 {
            clippedText = String(text.prefix(120)) + "..."
        } else {
            clippedText = text
        }
        let entry = RecentTranscriptUpdate(
            timestamp: Date(),
            phase: confirmed ? .confirmed : .volatile,
            confidence: confidence,
            text: clippedText
        )
        recentUpdates.append(entry)
        if recentUpdates.count > 8 {
            recentUpdates = Array(recentUpdates.suffix(8))
        }
    }

    private func handleVadEvent(_ event: VadDebugEvent) {
        switch event {
        case .speechStart(let probability):
            vadSpeechStartCount += 1
            lastVadEventText = "speechStart @ \(String(format: "%.2f", probability))"
            appendSystemRecentUpdate(text: "VAD speechStart", confidence: probability)
        case .speechEnd(let probability):
            vadSpeechEndCount += 1
            lastVadEventText = "speechEnd @ \(String(format: "%.2f", probability))"
            appendSystemRecentUpdate(text: "VAD speechEnd", confidence: probability)
        case .forcedSplit(let probability, let seconds):
            vadForcedSplitCount += 1
            lastVadEventText =
                "forcedSplit \(String(format: "%.1f", seconds))s @ \(String(format: "%.2f", probability))"
            appendSystemRecentUpdate(
                text: "Fallback split after \(String(format: "%.1f", seconds))s",
                confidence: probability,
                phase: .fallback
            )
        }
    }

    private func appendDiarizedSegment(_ segment: DiarizedTranscriptSegment) {
        diarizedSegments.append(segment)
    }

    var uniqueSpeakerCount: Int {
        Set(diarizedSegments.map(\.speakerIndex)).count
    }

    private func appendSystemRecentUpdate(
        text: String,
        confidence: Float,
        phase: RecentTranscriptUpdate.Phase = .vadEvent
    ) {
        let entry = RecentTranscriptUpdate(
            timestamp: Date(),
            phase: phase,
            confidence: confidence,
            text: text
        )
        recentUpdates.append(entry)
        if recentUpdates.count > 8 {
            recentUpdates = Array(recentUpdates.suffix(8))
        }
    }

    private func requestMicrophonePermission() async throws {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return
        case .notDetermined:
            let granted = await withCheckedContinuation { continuation in
                AVCaptureDevice.requestAccess(for: .audio) { granted in
                    continuation.resume(returning: granted)
                }
            }
            guard granted else {
                throw AudioClawError.microphonePermissionDenied
            }
        case .denied, .restricted:
            throw AudioClawError.microphonePermissionDenied
        @unknown default:
            throw AudioClawError.microphonePermissionDenied
        }
    }

    private func present(error: Error) {
        errorMessage = error.localizedDescription
        isShowingError = true
        statusText = "Error"
    }

    private func updateDisplayedInputDeviceName() {
        if let selected = availableInputDevices.first(where: { $0.id == selectedInputDeviceID }) {
            inputDeviceName = selected.name
            return
        }
        inputDeviceName = controller.currentInputDeviceName()
    }

    private func startSessionClock() {
        stopSessionClock()
        timerTask = Task { [weak self] in
            guard let self else { return }
            while !Task.isCancelled {
                if let sessionStartedAt {
                    sessionDuration = Date().timeIntervalSince(sessionStartedAt)
                }
                try? await Task.sleep(for: .seconds(1))
            }
        }
    }

    private func stopSessionClock() {
        timerTask?.cancel()
        timerTask = nil
        sessionStartedAt = nil
    }

    private func startLevelDecay() {
        stopLevelDecay()
        levelDecayTask = Task { [weak self] in
            guard let self else { return }
            while !Task.isCancelled {
                inputLevel *= 0.82
                try? await Task.sleep(for: .milliseconds(80))
            }
        }
    }

    private func stopLevelDecay() {
        levelDecayTask?.cancel()
        levelDecayTask = nil
    }

    private static func makeStatusText(for progress: DownloadUtils.DownloadProgress) -> String {
        switch progress.phase {
        case .listing:
            return "Checking model files..."
        case .downloading(let completedFiles, let totalFiles):
            return "Downloading models (\(completedFiles)/\(totalFiles))..."
        case .compiling(let modelName):
            return "Compiling \(modelName)..."
        }
    }
}

struct RecentTranscriptUpdate: Identifiable, Equatable {
    enum Phase: String {
        case confirmed
        case volatile
        case vadEvent
        case fallback

        var label: String {
            switch self {
            case .confirmed:
                return "Confirmed"
            case .volatile:
                return "Volatile"
            case .vadEvent:
                return "VAD"
            case .fallback:
                return "Fallback"
            }
        }
    }

    let id = UUID()
    let timestamp: Date
    let phase: Phase
    let confidence: Float
    let text: String

    var timestampText: String {
        Self.timestampFormatter.string(from: timestamp)
    }

    private static let timestampFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter
    }()
}

enum DemoModel: String, CaseIterable, Identifiable {
    case english
    case multilingual
    case chinese

    var id: String { rawValue }

    var title: String {
        switch self {
        case .english:
            return "English v2"
        case .multilingual:
            return "Multilingual v3"
        case .chinese:
            return "Parakeet CTC Chinese"
        }
    }

    var shortTitle: String {
        switch self {
        case .english:
            return "English"
        case .multilingual:
            return "Multi"
        case .chinese:
            return "Chinese"
        }
    }

    var asrModelVersion: AsrModelVersion {
        switch self {
        case .english:
            return .v2
        case .multilingual:
            return .v3
        case .chinese:
            return .ctcZhCn
        }
    }
}

enum AudioClawError: LocalizedError {
    case microphonePermissionDenied
    case noAudioInputDevice
    case cannotAddAudioInput(String)
    case cannotAddAudioOutput
    case noDebugAudioAvailable

    var errorDescription: String? {
        switch self {
        case .microphonePermissionDenied:
            return
                "Microphone permission is required. Enable it for AudioClaw in System Settings > Privacy & Security > Microphone."
        case .noAudioInputDevice:
            return "No audio input device is available."
        case .cannotAddAudioInput(let deviceName):
            return "Unable to capture from the selected device: \(deviceName)."
        case .cannotAddAudioOutput:
            return "Unable to create the audio capture output."
        case .noDebugAudioAvailable:
            return "No debug audio has been captured yet. Record a short sample first."
        }
    }
}
