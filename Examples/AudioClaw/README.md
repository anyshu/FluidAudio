# AudioClaw

A macOS SwiftUI app that transcribes audio locally with FluidAudio.
Supports both real-time microphone capture and local audio file transcription.

## Open the demo

1. Open `Examples/AudioClaw/AudioClaw.xcodeproj` in Xcode.
2. Select the `AudioClaw` scheme.
3. Run the app on macOS 14 or later.
4. Grant microphone permission when prompted.

The first launch downloads the selected ASR model bundle from Hugging Face.
Subsequent runs reuse the cached models.
