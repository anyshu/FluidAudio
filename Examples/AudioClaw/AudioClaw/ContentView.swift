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
            Task {
                await viewModel.transcribeFile(url: url)
            }
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
                    Label("Transcribe File...", systemImage: "doc.fill")
                }
                .buttonStyle(.bordered)
                .disabled(viewModel.isBusy || viewModel.isRecording)
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
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
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
            } else {
                TranscriptTextView(text: viewModel.displayedTranscript)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
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

    private func phaseColor(_ phase: RecentTranscriptUpdate.Phase) -> Color {
        switch phase {
        case .confirmed: return .green
        case .volatile: return .orange
        case .vadEvent: return .blue
        case .fallback: return .pink
        }
    }
}
