import FluidAudio
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var viewModel = AsrViewModel()

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            progressBars

            HStack(spacing: 0) {
                transcriptPanel
                Divider()
                sidebar
                    .frame(width: 280)
            }
        }
        .frame(minWidth: 880, minHeight: 560)
        .background(Color(nsColor: .windowBackgroundColor))
        .task {
            await viewModel.prepareIfNeeded()
        }
        .alert("ASR Error", isPresented: $viewModel.isShowingError) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(viewModel.errorMessage)
        }
        .fileImporter(
            isPresented: $viewModel.isShowingFilePicker,
            allowedContentTypes: [.audio],
            allowsMultipleSelection: false
        ) { result in
            guard case .success(let urls) = result, let url = urls.first else { return }
            viewModel.selectFile(url: url)
        }
    }

    // MARK: - Toolbar

    private var toolbar: some View {
        VStack(spacing: 8) {
            HStack(spacing: 12) {
                Picker("Model", selection: $viewModel.selectedModel) {
                    ForEach(DemoModel.allCases) { model in
                        Text(model.shortTitle).tag(model)
                    }
                }
                .pickerStyle(.segmented)
                .fixedSize()
                .disabled(viewModel.isBusy)

                Picker("Input", selection: $viewModel.selectedInputDeviceID) {
                    ForEach(viewModel.availableInputDevices) { device in
                        Text(device.name).tag(device.id)
                    }
                }
                .frame(maxWidth: 220)
                .disabled(viewModel.isBusy || viewModel.isRecording)

                Toggle("VAD", isOn: $viewModel.vadEnabled)
                    .toggleStyle(.switch)
                    .controlSize(.small)
                    .disabled(
                        viewModel.isBusy
                            || viewModel.isRecording
                            || viewModel.pendingFileURL != nil
                    )
                    .help(
                        viewModel.pendingFileURL != nil
                            ? "VAD is unnecessary in file mode — sliding-window ASR already handles segmentation."
                            : "Voice Activity Detection (live recording only)."
                    )

                Toggle("Diarize", isOn: $viewModel.diarizationEnabled)
                    .toggleStyle(.switch)
                    .controlSize(.small)
                    .disabled(viewModel.isBusy || viewModel.isRecording)

                Spacer()

                statusBadge(
                    title: viewModel.statusText,
                    tint: viewModel.isRecording ? .red : (viewModel.isTranscribingFile ? .purple : .secondary)
                )

                if viewModel.isRecording {
                    statusBadge(title: viewModel.sessionSummary, tint: .red)
                }
            }

            HStack(spacing: 8) {
                Button {
                    Task { await viewModel.toggleRecording() }
                } label: {
                    Label(
                        viewModel.primaryButtonTitle,
                        systemImage: viewModel.isRecording ? "stop.fill" : "mic.fill"
                    )
                }
                .buttonStyle(.borderedProminent)
                .tint(viewModel.isRecording ? .red : .accentColor)
                .disabled(viewModel.isBusy)
                .controlSize(.regular)

                Button {
                    viewModel.isShowingFilePicker = true
                } label: {
                    Label("Choose File...", systemImage: "doc.fill")
                }
                .buttonStyle(.bordered)
                .disabled(viewModel.isBusy || viewModel.isRecording)
                .controlSize(.regular)

                Button {
                    Task { await viewModel.runPendingFile() }
                } label: {
                    Label("Run", systemImage: "play.fill")
                }
                .buttonStyle(.borderedProminent)
                .tint(.purple)
                .disabled(
                    viewModel.isBusy
                        || viewModel.isRecording
                        || viewModel.pendingFileURL == nil
                )
                .controlSize(.regular)

                Button("Clear") {
                    viewModel.clearTranscript()
                }
                .buttonStyle(.bordered)
                .disabled(viewModel.isBusy || viewModel.fullTranscript.isEmpty)
                .controlSize(.regular)

                Spacer()

                Button {
                    Task { await viewModel.saveDebugWav() }
                } label: {
                    Label("Save WAV", systemImage: "square.and.arrow.down")
                }
                .buttonStyle(.bordered)
                .disabled(viewModel.isBusy)
                .controlSize(.small)
            }

            if let url = viewModel.pendingFileURL {
                pendingFileRow(url: url)
            }

            if viewModel.diarizationEnabled {
                diarizationOptionsRow
                if viewModel.selectedDiarizationEngine == .sortformer {
                    sortformerThresholdRow
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    private func pendingFileRow(url: URL) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "doc.fill")
                .font(.caption)
                .foregroundStyle(.purple)
            Text(url.lastPathComponent)
                .font(.caption.weight(.medium))
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            Button {
                viewModel.clearPendingFile()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.tertiary)
            }
            .buttonStyle(.borderless)
            .disabled(viewModel.isBusy)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(Color.purple.opacity(0.06), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
    }

    private var sortformerThresholdRow: some View {
        HStack(spacing: 10) {
            Image(systemName: "slider.horizontal.3")
                .font(.caption)
                .foregroundStyle(.purple)
            Text("Speaker change sensitivity")
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
            Slider(
                value: $viewModel.sortformerPredScoreThreshold,
                in: 0.10...0.40,
                step: 0.01
            )
            .frame(maxWidth: 220)
            .disabled(viewModel.isBusy || viewModel.isRecording)
            Text(String(format: "%.2f", viewModel.sortformerPredScoreThreshold))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 36, alignment: .trailing)
            Text("(lower = more sensitive)")
                .font(.caption2)
                .foregroundStyle(.tertiary)
            Spacer()
        }
        .padding(.vertical, 4)
    }

    private var diarizationOptionsRow: some View {
        HStack(spacing: 10) {
            Image(systemName: "person.2.wave.2.fill")
                .font(.caption)
                .foregroundStyle(.purple)
            Text("Engine")
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)

            Picker("Engine", selection: $viewModel.selectedDiarizationEngine) {
                ForEach(DiarizationEngineType.allCases) { engine in
                    Text(engine.rawValue).tag(engine)
                }
            }
            .pickerStyle(.segmented)
            .fixedSize()
            .disabled(viewModel.isBusy || viewModel.isRecording)

            if viewModel.selectedDiarizationEngine == .lseend {
                Text("Variant")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                Picker("Variant", selection: $viewModel.selectedLSEENDVariant) {
                    ForEach(LSEENDVariant.allCases) { variant in
                        Text(variant.rawValue).tag(variant)
                    }
                }
                .frame(maxWidth: 160)
                .disabled(viewModel.isBusy || viewModel.isRecording)
            }

            Text(viewModel.selectedDiarizationEngine.detail)
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .lineLimit(1)

            Spacer()
        }
        .padding(.vertical, 4)
    }

    // MARK: - Progress Bars

    @ViewBuilder
    private var progressBars: some View {
        if let progress = viewModel.downloadProgress {
            HStack(spacing: 8) {
                Text("Model")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                ProgressView(value: progress)
                Text("\(Int(progress * 100))%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(width: 32, alignment: .trailing)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 6)
            .background(Color(nsColor: .controlBackgroundColor))
            Divider()
        }

        if let progress = viewModel.fileTranscriptionProgress {
            HStack(spacing: 8) {
                Text("File")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                ProgressView(value: progress)
                    .tint(.purple)
                Text("\(Int(progress * 100))%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(width: 32, alignment: .trailing)
                Button("Cancel") {
                    viewModel.cancelFileTranscription()
                }
                .buttonStyle(.borderless)
                .font(.caption.weight(.medium))
                .foregroundStyle(.red)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 6)
            .background(Color.purple.opacity(0.04))
            Divider()
        }
    }

    // MARK: - Transcript Panel

    private var transcriptPanel: some View {
        VStack(alignment: .leading, spacing: 0) {
            if !viewModel.liveTranscript.isEmpty {
                HStack(spacing: 6) {
                    Circle()
                        .fill(.orange)
                        .frame(width: 6, height: 6)
                    Text(viewModel.liveTranscript)
                        .font(.callout)
                        .italic()
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.orange.opacity(0.05))
                Divider()
            }

            if viewModel.displayedTranscript.isEmpty {
                VStack(spacing: 10) {
                    Image(systemName: "waveform")
                        .font(.system(size: 36, weight: .thin))
                        .foregroundStyle(.quaternary)
                    Text("Record from microphone or transcribe a local audio file.")
                        .font(.callout)
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if viewModel.diarizationEnabled, !viewModel.diarizedSegments.isEmpty {
                diarizedTranscriptView
            } else {
                TranscriptTextView(text: viewModel.displayedTranscript)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var diarizedTranscriptView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                ForEach(viewModel.diarizedSegments) { segment in
                    HStack(alignment: .top, spacing: 10) {
                        Circle()
                            .fill(Self.speakerColor(for: segment.speakerIndex))
                            .frame(width: 10, height: 10)
                            .padding(.top, 6)
                        VStack(alignment: .leading, spacing: 2) {
                            HStack(spacing: 6) {
                                Text(segment.speakerLabel)
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(Self.speakerColor(for: segment.speakerIndex))
                                Text(String(format: "%.1fs – %.1fs", segment.startTime, segment.endTime))
                                    .font(.caption2.monospacedDigit())
                                    .foregroundStyle(.tertiary)
                            }
                            Text(segment.text)
                                .font(.callout)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        Self.speakerColor(for: segment.speakerIndex).opacity(0.06),
                        in: RoundedRectangle(cornerRadius: 8, style: .continuous)
                    )
                }
            }
            .padding(16)
        }
    }

    private static let speakerPalette: [Color] = [
        .blue, .orange, .green, .pink, .purple, .teal, .yellow, .red, .indigo, .mint,
    ]

    static func speakerColor(for index: Int) -> Color {
        speakerPalette[index % speakerPalette.count]
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                sidebarSection("Live Monitor", icon: "mic.fill") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text(viewModel.inputDeviceName)
                                .font(.caption)
                                .lineLimit(1)
                            Spacer()
                            Text(viewModel.isRecording ? "Active" : "Standby")
                                .font(.caption2.weight(.medium))
                                .foregroundStyle(viewModel.isRecording ? Color.green : Color.secondary)
                        }

                        GeometryReader { proxy in
                            ZStack(alignment: .leading) {
                                Capsule().fill(Color.primary.opacity(0.06))
                                Capsule()
                                    .fill(
                                        LinearGradient(
                                            colors: [.green, .orange, .red],
                                            startPoint: .leading,
                                            endPoint: .trailing
                                        )
                                    )
                                    .frame(
                                        width: max(proxy.size.width * CGFloat(viewModel.inputLevel), 3)
                                    )
                            }
                        }
                        .frame(height: 6)

                        HStack {
                            Circle()
                                .fill(viewModel.isVadSpeechActive ? Color.red : Color.secondary.opacity(0.2))
                                .frame(width: 7, height: 7)
                            Text(viewModel.isVadSpeechActive ? "Speech" : "Silence")
                                .font(.caption.weight(.medium))
                            Spacer()
                            Text(String(format: "P %.2f", viewModel.vadProbability))
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.tertiary)
                        }

                        HStack(spacing: 12) {
                            vadStat("Start", viewModel.vadSpeechStartCount)
                            vadStat("End", viewModel.vadSpeechEndCount)
                            vadStat("Split", viewModel.vadForcedSplitCount)
                            Spacer()
                        }

                        if !viewModel.lastPartialText.isEmpty {
                            VStack(alignment: .leading, spacing: 2) {
                                HStack {
                                    Text("Partial")
                                        .font(.caption2.weight(.medium))
                                        .foregroundStyle(.tertiary)
                                    Spacer()
                                    Text(String(format: "%.2f", viewModel.lastConfidence))
                                        .font(.caption2.monospacedDigit())
                                        .foregroundStyle(.tertiary)
                                }
                                Text(viewModel.lastPartialText)
                                    .font(.caption)
                                    .lineLimit(3)
                            }
                        }
                    }
                }

                if viewModel.diarizationEnabled {
                    Divider().padding(.horizontal, 12)
                    sidebarSection("Speakers", icon: "person.2.wave.2.fill") {
                        if viewModel.diarizedSegments.isEmpty {
                            Text("Engine: \(viewModel.selectedDiarizationEngine.rawValue)")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                            Text("Speaker labels appear after the first utterance.")
                                .font(.caption2)
                                .foregroundStyle(.quaternary)
                        } else {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("\(viewModel.uniqueSpeakerCount) speaker(s) detected")
                                    .font(.caption.weight(.medium))
                                    .foregroundStyle(.secondary)
                                ForEach(speakerStats(), id: \.index) { stat in
                                    HStack(spacing: 6) {
                                        Circle()
                                            .fill(Self.speakerColor(for: stat.index))
                                            .frame(width: 8, height: 8)
                                        Text("Speaker \(stat.index + 1)")
                                            .font(.caption)
                                        Spacer()
                                        Text("\(stat.count) seg")
                                            .font(.caption2.monospacedDigit())
                                            .foregroundStyle(.tertiary)
                                    }
                                }
                            }
                        }
                    }
                }

                Divider().padding(.horizontal, 12)

                sidebarSection("Recent Updates", icon: "list.bullet") {
                    if viewModel.recentUpdates.isEmpty {
                        Text("Transcript chunks appear here.")
                            .font(.caption)
                            .foregroundStyle(.quaternary)
                    } else {
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(viewModel.recentUpdates) { entry in
                                VStack(alignment: .leading, spacing: 2) {
                                    HStack(spacing: 4) {
                                        Text(entry.timestampText)
                                            .font(.caption2.monospacedDigit())
                                            .foregroundStyle(.quaternary)
                                        Text(entry.phase.label)
                                            .font(.caption2.weight(.semibold))
                                            .foregroundStyle(phaseColor(entry.phase))
                                        Spacer()
                                        Text(String(format: "%.2f", entry.confidence))
                                            .font(.caption2.monospacedDigit())
                                            .foregroundStyle(.quaternary)
                                    }
                                    Text(entry.text)
                                        .font(.caption)
                                        .lineLimit(2)
                                }
                                .padding(6)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(
                                    Color.primary.opacity(0.03),
                                    in: RoundedRectangle(cornerRadius: 5, style: .continuous)
                                )
                            }
                        }
                    }
                }

                if !viewModel.lastSavedDebugWavPath.isEmpty {
                    Divider().padding(.horizontal, 12)

                    sidebarSection("Debug WAV", icon: "waveform.circle") {
                        Text(viewModel.lastSavedDebugWavPath)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .lineLimit(3)
                    }
                }
            }
        }
        .background(Color(nsColor: .controlBackgroundColor))
    }

    // MARK: - Sidebar Helpers

    private func sidebarSection<Content: View>(
        _ title: String,
        icon: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: icon)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            content()
        }
        .padding(12)
    }

    private func vadStat(_ label: String, _ value: Int) -> some View {
        HStack(spacing: 3) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.quaternary)
            Text("\(value)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
    }

    private func statusBadge(title: String, tint: Color) -> some View {
        Text(title)
            .font(.caption.weight(.semibold))
            .foregroundStyle(tint)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(tint.opacity(0.1), in: Capsule())
    }

    private struct SpeakerStat {
        let index: Int
        let count: Int
    }

    private func speakerStats() -> [SpeakerStat] {
        var counts: [Int: Int] = [:]
        for seg in viewModel.diarizedSegments {
            counts[seg.speakerIndex, default: 0] += 1
        }
        return
            counts
            .sorted { $0.key < $1.key }
            .map { SpeakerStat(index: $0.key, count: $0.value) }
    }

    private func phaseColor(_ phase: RecentTranscriptUpdate.Phase) -> Color {
        switch phase {
        case .confirmed: return .green
        case .volatile: return .orange
        case .vadEvent: return .blue
        case .fallback: return .pink
        }
    }
}
