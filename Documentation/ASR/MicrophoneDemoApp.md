# AudioClaw Demo App

This repo includes a macOS SwiftUI demo app that captures microphone audio
with `AVAudioEngine`, uses FluidAudio VAD to gate utterances, and then
transcribes each finalized speech segment locally.

- Project: `Examples/AudioClaw/AudioClaw.xcodeproj`
- Platform: macOS 14+
- Models: Parakeet English v2, multilingual v3, or Parakeet CTC Chinese

## What the demo shows

- Requesting microphone permission
- Downloading and caching ASR models on first launch
- Capturing live microphone buffers with `AVAudioEngine`
- Running streaming VAD on 16 kHz mono chunks
- Transcribing finalized utterances and appending them into the transcript area
- Saving captured audio to a debug WAV file for inspection

## Running the demo

1. Open `Examples/AudioClaw/AudioClaw.xcodeproj` in Xcode.
2. Select the `AudioClaw` scheme.
3. Run on macOS.
4. Grant microphone permission when prompted.
5. Pick a model and click `Start Recording`.

The app downloads the selected model bundle the first time you use it, then
reuses the cached copy for later runs.

## Implementation notes

- `AsrController.swift`
  - Owns `AVAudioEngine`
  - Installs the microphone tap
  - Runs `VadManager.processStreamingChunk(...)` in the same chunk shape used by
    the VAD tests
  - Uses a more aggressive live VAD profile tuned closer to the Electron app:
    `threshold=0.4`, `minSpeechDuration=0.12`, `minSilenceDuration=0.15`,
    `maxSpeechDuration=20.0`, `speechPadding=0.1`
  - Queues finalized utterances for ASR so long sessions do not keep
    retranscribing the full recording
- `AsrViewModel.swift`
  - Handles permission flow
  - Tracks download / compile progress
  - Publishes appended transcript text and raw incremental updates for SwiftUI
- `ContentView.swift`
  - Keeps the UI intentionally small so the capture and transcription flow is
    easy to follow and reuse in your own app

## Adapting it to your app

- Replace the demo's transcript view with your own editor or subtitle UI.
- Keep the deep-copy step for tap buffers before handing them to async work.
- If you need more or fewer ASR updates during continuous speech, tune the
  demo's `VadConfig.defaultThreshold` and `VadSegmentationConfig`.
- If you only need English, prefer model `v2` for better recall.
