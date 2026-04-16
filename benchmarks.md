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
