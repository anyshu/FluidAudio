# Benchmark Output Convention

All benchmark commands write a results JSON to `benchmark_results/` at the repo root by default. Pass `--output <path>` to override or `--output none` to skip the JSON.

```
benchmark_results/
├── parakeet_v3_test_clean.json        # asr-benchmark --subset test-clean
├── parakeet_v3_test_other.json        # asr-benchmark --subset test-other
├── parakeet_zh_cn_int8.json           # ctc-zh-cn-benchmark
├── parakeet_ja_jsut.json              # ja-benchmark --dataset jsut
├── sensevoice_all.json                # SenseVoiceBenchmark/benchmark.py
├── qwen3_all.json                     # Qwen3RemoteBenchmark/benchmark.py
└── apple_all.json                     # AppleSpeechBenchmark
```

The directory is `.gitignore`'d.

---

# Parakeet TDT-CTC-110M Benchmark Results

## LibriSpeech test-clean (Full Dataset)

| Metric | Value |
|--------|-------|
| Files processed | 2,620 |
| **Average WER** | **3.01%** |
| **Median WER** | **0.0%** |
| Average CER | 1.09% |
| Audio duration | 19,452.5s (~5.4 hours) |
| Processing time | 201.5s (~3.4 minutes) |
| **Overall RTFx** | **96.5x** |
| **Median RTFx** | **86.4x** |

## Configuration

- Model: Parakeet TDT-CTC-110M (CoreML)
- Architecture: Hybrid TDT-CTC with fused preprocessor+encoder
- Platform: Apple Silicon (M2)
- Date: March 26, 2026

## Key Features

- **96.5x real-time factor** - 1 hour of audio transcribes in 37 seconds
- **3.01% WER** - Competitive accuracy on LibriSpeech test-clean
- **0% median WER** - Most files transcribed perfectly
- **iOS compatible** - Runs on iPhone with full CoreML optimization
- **Stateless processing** - No encoder state carryover needed

## Running the Benchmark

```bash
# Build release
swift build -c release

# Run full benchmark (auto-downloads dataset and models)
.build/release/fluidaudiocli asr-benchmark --subset test-clean --model-version tdt-ctc-110m

# Run with limited files
.build/release/fluidaudiocli asr-benchmark --subset test-clean --model-version tdt-ctc-110m --max-files 100

# Process single file
.build/release/fluidaudiocli asr-benchmark --single-file 1089-134686-0000 --model-version tdt-ctc-110m
```

## Notes

- TDT (Token-and-Duration Transducer) decoder with CTC-constrained beam search
- Fused preprocessor+encoder reduces model load time and memory usage
- Models available at: [FluidInference/parakeet-tdt-ctc-110m-coreml](https://huggingface.co/FluidInference/parakeet-tdt-ctc-110m-coreml)
- iOS test app validates on-device performance with LibriSpeech ground truth

---

# Nemotron Speech Streaming 0.6B Benchmark Results

## LibriSpeech test-clean (Full Dataset)

| Metric | Value |
|--------|-------|
| Files processed | 2,620 |
| Total words | 53,120 |
| Total errors | 1,334 |
| **WER** | **2.51%** |
| Audio duration | 19,452.5s (~5.4 hours) |
| Processing time | 3,393.7s (~56.6 minutes) |
| **RTFx** | **5.7x** |
| Peak memory | 1.452 GB |

## Configuration

- Model: Nemotron Speech Streaming 0.6B (CoreML)
- Encoder variant: int8
- Platform: Apple Silicon (M4 Pro)
- Date: January 15, 2026

## Running the Benchmark

```bash
# Build release
swift build -c release

# Run full benchmark (auto-downloads dataset and models)
.build/release/fluidaudiocli nemotron-benchmark --subset test-clean

# Run with limited files
.build/release/fluidaudiocli nemotron-benchmark --subset test-clean --max-files 100

# Use float32 encoder variant
.build/release/fluidaudiocli nemotron-benchmark --encoder float32 --max-files 50
```

## Notes

- True streaming with 1.12s audio chunks and encoder state carryover
- RNNT greedy decoding with proper decoder LSTM state management
- Models available at: [alexwengg/nemotron-speech-streaming-en-0.6b-coreml](https://huggingface.co/alexwengg/nemotron-speech-streaming-en-0.6b-coreml)

---

# Parakeet TDT 0.6B v3 Benchmark Results

## LibriSpeech test-clean (2,620 files, 19,452.5 s audio)

| Encoder           | On-disk | Avg WER | Avg CER | Overall RTFx | Peak RAM |
|-------------------|--------:|--------:|--------:|-------------:|---------:|
| 8-bit palettized  | 425 MB  | 2.64%   | 1.03%   | 47.1×        | 153 MB   |
| int4 linear/ch    | 285 MB  | 3.76%   | 1.59%   | 43.1×        | 139 MB   |

Apple M2, `.cpuAndNeuralEngine`. Decoder/joint/preprocessor fp16 in both. Models: [FluidInference/parakeet-tdt-0.6b-v3-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml).

---

# Parakeet CTC zh-CN

## Configuration

- Model: Parakeet CTC 0.6B zh-CN (CoreML)
- Architecture: CTC-only, 600M parameters, INT8 quantized encoder (0.55GB) or FP32 (1.1GB)
- Dataset: THCHS-30 (Tsinghua Chinese speech dataset)
- Metric: CER (Character Error Rate)
- Models available at: [FluidInference/parakeet-ctc-0.6b-zh-cn-coreml](https://huggingface.co/FluidInference/parakeet-ctc-0.6b-zh-cn-coreml)

## Running the Benchmark

```bash
swift build -c release

# Full dataset, INT8 encoder (default)
.build/release/fluidaudiocli ctc-zh-cn-benchmark --auto-download

# FP32 encoder
.build/release/fluidaudiocli ctc-zh-cn-benchmark --auto-download --fp32

# Limited files
.build/release/fluidaudiocli ctc-zh-cn-benchmark --auto-download --samples 100

# Save results to JSON
.build/release/fluidaudiocli ctc-zh-cn-benchmark --auto-download --output results.json
```

## Notes

- `--auto-download` requires `huggingface_hub` (`pip install huggingface_hub`)
- Newer versions of `huggingface_hub` install as `hf`, not `huggingface-cli`. The CLI handles this automatically.
- Dataset auto-downloads to `~/Library/Application Support/FluidAudio/Datasets/THCHS-30/`
- Text normalization: Arabic digits converted to Chinese characters (0→零, 1→一…), spaces removed, non-CJK stripped

---

# Parakeet TDT ja (Japanese)

## Configuration

- Model: Parakeet TDT 0.6B ja (CoreML)
- Architecture: Hybrid CTC encoder + TDT decoder/joint v2, INT8 quantized
- Datasets: JSUT-basic5000 or Common Voice Japanese (test split)
- Metric: CER (Character Error Rate)
- Models available at: [FluidInference/parakeet-0.6b-ja-coreml](https://huggingface.co/FluidInference/parakeet-0.6b-ja-coreml)
- The same repo also contains a CTC-only Japanese model (`CtcDecoder.mlmodelc`)

## Running the Benchmark

```bash
swift build -c release

# JSUT-basic5000 (default)
.build/release/fluidaudiocli ja-benchmark --auto-download

# Common Voice Japanese test set
.build/release/fluidaudiocli ja-benchmark --dataset cv-test --auto-download

# Limited files
.build/release/fluidaudiocli ja-benchmark --samples 100 --auto-download

# Save results to JSON
.build/release/fluidaudiocli ja-benchmark --auto-download --output results.json
```

## Notes

- Text normalization: kanji numbers converted to Arabic digits (二十一→21) with compound numbers handled before simple tens to avoid partial matches
- Both `jsut` and `cv-test` datasets auto-download via `huggingface_hub`

---

# SenseVoice Small (Python ONNX)

## Configuration

- Model: SenseVoice Small (ONNX, via `funasr_onnx`)
- Datasets: THCHS-30 (CER), JSUT-basic5000 (CER), LibriSpeech test-clean (WER), LibriSpeech test-other (WER)
- Script: `Examples/SenseVoiceBenchmark/benchmark.py`
- Reuses datasets already downloaded by the Swift CLI benchmarks

## Setup

```bash
pip install funasr_onnx soundfile numpy onnx
```

## Running the Benchmark

```bash
cd Examples/SenseVoiceBenchmark

# All datasets
python3 benchmark.py

# Single dataset
python3 benchmark.py --dataset thchs30
python3 benchmark.py --dataset librispeech
python3 benchmark.py --dataset librispeech-other
python3 benchmark.py --dataset jsut

# Limited files (quick validation)
python3 benchmark.py --max-files 50

# FP32 model
python3 benchmark.py --quantize

# Save results to JSON
python3 benchmark.py --output results.json
```

## Notes

- First run exports the PyTorch model to ONNX and downloads from ModelScope (~download once)
- The exported ONNX model has a type mismatch bug in `Less` operator nodes (tensor(float) vs tensor(int64)) that causes `onnxruntime >= 1.17` to reject it. The script automatically patches the ONNX file on first run and writes a `.patched` sentinel to avoid re-patching.
- Datasets are read from `~/Library/Application Support/FluidAudio/Datasets/` — the same paths used by the Swift CLI. Run the Swift CLI benchmarks first to download datasets, or use `--auto-download` on each Swift benchmark command.
- Text normalization is kept identical to the Swift benchmarks for fair cross-model comparison:
  - Chinese: Arabic digits → Chinese characters, non-CJK stripped
  - Japanese: kanji numbers → Arabic digits (same ordered replacement as Swift)
  - English: full HuggingFace ASR leaderboard normalization (British→American via `english.json`, abbreviations, contractions, number words → digits)

---

# Apple SFSpeechRecognizer (Swift)

## Configuration

- Model: Apple `SFSpeechRecognizer` (on-device by default)
- Datasets: THCHS-30 (CER, zh-CN), JSUT-basic5000 (CER, ja-JP), LibriSpeech test-clean / test-other (WER, en-US)
- Location: `Examples/AppleSpeechBenchmark/`
- Reuses the datasets already downloaded by the Swift CLI benchmarks (`~/Library/Application Support/FluidAudio/Datasets/`)

## Running the Benchmark

```bash
cd Examples/AppleSpeechBenchmark
swift build -c release

# All datasets (THCHS-30, LibriSpeech test-clean/test-other, JSUT)
.build/release/AppleSpeechBenchmark

# Single dataset
.build/release/AppleSpeechBenchmark --dataset thchs30
.build/release/AppleSpeechBenchmark --dataset librispeech
.build/release/AppleSpeechBenchmark --dataset librispeech-other
.build/release/AppleSpeechBenchmark --dataset jsut

# Limit files per dataset
.build/release/AppleSpeechBenchmark --dataset librispeech --max-files 100

# Allow Apple's server-side recognition as a fallback (default is on-device only)
.build/release/AppleSpeechBenchmark --allow-server

# Save results to JSON
.build/release/AppleSpeechBenchmark --output apple_speech_results.json
```

## Notes

- **Authorization**: First run triggers a Speech Recognition permission prompt. If it is denied or not shown (CLI has no Info.plist), grant access in `System Settings > Privacy & Security > Speech Recognition`. The benchmark exits immediately if authorization is not granted.
- **Dictation must be enabled**: Even with authorization granted, macOS returns `kLSRErrorDomain 201 "Siri and Dictation are disabled"` unless Dictation is turned on in `System Settings > Keyboard > Dictation`. Turn it on, let it download the offline language pack(s) you need (English (US), Chinese (Simplified), Japanese), then re-run.
- **On-device language packs**: on-device mode (default) requires the locale's offline pack to be downloaded via the Dictation settings above. Without it, the benchmark will skip that dataset. Use `--allow-server` to fall back to network recognition (subject to Apple's rate limits and 1-minute per-file limit — not suitable for full-dataset runs).
- **Main run loop**: the executable uses `CFRunLoopRun()` rather than blocking the main thread with a semaphore, because `SFSpeechRecognizer` delivers its recognitionTask callbacks via the main run loop. Blocking the main thread causes every recognition to hang forever.

---

# Qwen3-ASR-1.7B (remote, OpenAI-compatible)

## Configuration

- Model: Qwen3-ASR-1.7B served behind an OpenAI-compatible `/v1/chat/completions` endpoint
- Default endpoint: `https://next-api.fazhiplus.com/v1/chat/completions`
- Datasets: THCHS-30 (CER), JSUT-basic5000 (CER), LibriSpeech test-clean (WER), LibriSpeech test-other (WER)
- Location: `Examples/Qwen3RemoteBenchmark/`
- Audio is sent inline as `data:audio/wav;base64,...` after downmixing to mono and resampling to 16 kHz, so flac/wav/mp3 inputs all work without a separate upload step.
- Reuses the dataset loaders, normalizers, and metric calculations from `Examples/SenseVoiceBenchmark/benchmark.py` so the CER/WER numbers are directly comparable to SenseVoice and the Swift CLI benchmarks.

## Setup

```bash
pip install -r Examples/Qwen3RemoteBenchmark/requirements.txt
export QWEN3_API_KEY=sk-...   # or pass via --api-key
```

## Running the Benchmark

```bash
cd Examples/Qwen3RemoteBenchmark

# All datasets, all files (large — many thousands of API calls)
python benchmark.py

# Single dataset
python benchmark.py --dataset thchs30
python benchmark.py --dataset librispeech
python benchmark.py --dataset librispeech-other
python benchmark.py --dataset jsut

# Quick sample
python benchmark.py --dataset librispeech --max-files 100

# Tune concurrency (default 4). Server may rate-limit at higher values.
python benchmark.py --dataset librispeech --max-files 200 --concurrency 8

# Custom endpoint / model
python benchmark.py --api-url https://your.endpoint/v1/chat/completions --model Qwen3-ASR-1.7B

# Save aggregate JSON
python benchmark.py --output qwen3_remote.json
```

## Notes

- **Response format**: the model returns `language English<asr_text>...transcript...` (sometimes with a closing `</asr_text>`). The script strips both before scoring.
- **Audio encoding**: each request inlines the WAV as a base64 data URL. Typical 10-30s LibriSpeech utterances produce 0.4-1.5 MB request bodies. The endpoint accepts these without extra config.
- **Concurrency & rate limits**: `--concurrency` controls the thread pool size. The script retries on 429/5xx with exponential backoff (up to 4 retries, 1s → 30s). If you see persistent 429s, lower concurrency.
- **Cost**: every file is a billable chat-completions call. Token usage from the `usage` field in each response is summed and printed at the end.
- **Text normalization** is identical to SenseVoiceBenchmark (and the Swift CLI benchmarks) for fair cross-model comparison:
  - Chinese: Arabic digits → Chinese characters, non-CJK stripped
  - Japanese: kanji numbers → Arabic digits (compound tens before simple tens)
  - English: full HuggingFace ASR leaderboard normalization (British→American, abbreviations, contractions, number words → digits)
- **Dependencies**: pulls dataset loaders and normalizers from `../SenseVoiceBenchmark/benchmark.py` via `sys.path`. Keep both directories side-by-side.
- **Datasets root**: defaults to `~/Library/Application Support/FluidAudio/Datasets/` — run the Swift CLI benchmarks first (e.g. `fluidaudiocli ctc-zh-cn-benchmark --auto-download`, `fluidaudiocli ja-benchmark --auto-download`, `fluidaudiocli asr-benchmark --subset test-clean`) to populate it. Override with `--datasets-root`.
- **Text normalization** mirrors the other benchmarks for cross-model comparability:
  - Chinese: Arabic digits → Chinese characters, non-CJK stripped
  - Japanese: kanji numbers → Arabic digits (compound tens handled before simple tens), keep hiragana/katakana/CJK/digits
  - English: basic Unicode-aware normalization (lowercase, NFKD, strip punctuation/symbols, collapse whitespace). This is the "basic" HF normalizer — if you need strict Open ASR Leaderboard WER, post-process hypotheses with the full `TextNormalizer` in `Sources/FluidAudioCLI/Utils/TextNormalizer.swift`.
- **Metrics reported**: Aggregate/Average/Median WER or CER, total audio duration, total processing time, Overall RTFx, Median RTFx.
- **Punctuation**: `request.addsPunctuation = false` on macOS 13+ to avoid biasing WER with recognizer-inserted punctuation.

---

# VibeVoice-ASR-7B (remote, OpenAI-compatible)

## Configuration

- Model: `VibeVoice-ASR-7B` served behind an OpenAI-compatible `/v1/chat/completions` endpoint (non-streaming)
- Default endpoint: `https://next-api.fazhiplus.com/v1/chat/completions`
- Output schema: VibeVoice produces structured diarization-style JSON, e.g.
  `[{"Start time": "0:00", "End time": "0:05", "Speaker ID": "S1", "Content": "..."}, ...]`. The benchmark concatenates every `Content` field in order to form a single transcript before scoring.
- Datasets: THCHS-30 (CER), JSUT-basic5000 (CER), LibriSpeech test-clean (WER), LibriSpeech test-other (WER)
- Location: `Examples/VibeVoiceBenchmark/`
- Reuses the dataset loaders, normalizers, and metric calculations from `Examples/SenseVoiceBenchmark/benchmark.py` so the numbers are directly comparable to SenseVoice / Qwen3 / Parakeet / Apple.

## Setup

```bash
pip install -r Examples/VibeVoiceBenchmark/requirements.txt
export VIBEVOICE_API_KEY=sk-...   # or pass via --api-key
```

## Running the Benchmark

```bash
cd Examples/VibeVoiceBenchmark

# All datasets, full size (slow — streaming JSON output for each file)
python benchmark.py

# Single dataset
python benchmark.py --dataset thchs30
python benchmark.py --dataset librispeech
python benchmark.py --dataset librispeech-other
python benchmark.py --dataset jsut

# Quick sample
python benchmark.py --dataset librispeech --max-files 100

# Lower concurrency if the endpoint rate-limits (default: 2)
python benchmark.py --dataset librispeech --max-files 200 --concurrency 1

# Override prompts (the {duration} placeholder is substituted with the audio length)
python benchmark.py --user-prompt-template "Transcribe this {duration:.1f}s audio." \
                    --system-prompt "You are an ASR engine. Output JSON with key 'Content' only."

# Save aggregate JSON (default goes to <repo>/benchmark_results/vibevoice_<dataset>.json)
python benchmark.py --output benchmark_results/vibevoice_all.json
```

## Notes

- **Streaming**: requests use `stream=True` and consume Server-Sent Events line by line. Each SSE chunk's `delta.content` is concatenated, then parsed once after `[DONE]`.
- **Robust JSON extraction**: handles three formats out of the wild:
  1. Clean JSON array — parsed directly
  2. JSON wrapped in ` ```json ... ``` ` markdown fences — fences are stripped first
  3. Plain text with `Content: ...` lines — regex fallback as a last resort
- **Concurrency default is 2**, lower than the Qwen3 benchmark, because streaming connections held open for a long audio file are heavier on the endpoint.
- **Timeout default is 600 s** per request — VibeVoice can take a while for long-form audio.
- **Cost**: every file is a billable streaming chat-completion call, with `max_tokens=32768` allocated. If your provider charges by max-tokens reservation rather than emitted tokens, this could be expensive.
- **Apples-to-apples scoring**: VibeVoice does diarization + transcription, but for ASR comparison we score only the concatenated transcript. Diarization quality is **not** evaluated here.
- **Text normalization** is identical to the other benchmarks (HF leaderboard English; digit-mapping Chinese; kanji-to-digit Japanese).
