#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SenseVoice Small ONNX Benchmark
Tests on THCHS-30 (Chinese, CER) and LibriSpeech test-clean (English, WER).

Install dependencies:
    pip install funasr_onnx soundfile numpy

Usage:
    python benchmark.py                          # both datasets, all files
    python benchmark.py --dataset thchs30        # Chinese only
    python benchmark.py --dataset librispeech    # English only
    python benchmark.py --max-files 100          # limit files
    python benchmark.py --language zh            # force language
"""

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path

import soundfile as sf
import numpy as np

# ---------------------------------------------------------------------------
# Dataset paths (same as FluidAudio Swift CLI)
# ---------------------------------------------------------------------------
FLUIDAL_DATASETS = Path.home() / "Library/Application Support/FluidAudio/Datasets"
THCHS30_DIR = FLUIDAL_DATASETS / "THCHS-30"
LIBRISPEECH_DIR = FLUIDAL_DATASETS / "LibriSpeech"
JSUT_DIR = FLUIDAL_DATASETS / "JSUT-basic5000"


# ---------------------------------------------------------------------------
# Edit distance helpers
# ---------------------------------------------------------------------------

def levenshtein(a, b):
    """Token-level Levenshtein distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def wer(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens:
        return 0.0
    return levenshtein(ref_tokens, hyp_tokens) / len(ref_tokens)


def cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if not ref_chars:
        return 0.0
    return levenshtein(ref_chars, hyp_chars) / len(ref_chars)


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def _load_british_to_american() -> dict[str, str]:
    json_path = Path(__file__).parent / "english.json"
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    return {}

_BRITISH_TO_AMERICAN = _load_british_to_american()

_ADDITIONAL_DIACRITICS = {
    "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE",
    "ß": "ss", "ẞ": "SS", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "th", "ł": "l", "Ł": "L",
}

_ABBREVIATIONS = {
    "mr": "mister", "mrs": "missus", "ms": "miss", "dr": "doctor",
    "prof": "professor", "st": "saint", "jr": "junior", "sr": "senior",
    "esq": "esquire", "capt": "captain", "gov": "governor", "ald": "alderman",
    "gen": "general", "sen": "senator", "rep": "representative",
    "pres": "president", "rev": "reverend", "hon": "honorable",
    "asst": "assistant", "assoc": "associate", "lt": "lieutenant",
    "col": "colonel", "vs": "versus", "inc": "incorporated",
    "ltd": "limited", "co": "company", "am": "a m", "pm": "p m",
    "ad": "ad", "bc": "bc",
}

_CONTRACTIONS = {
    "can't": "can not", "won't": "will not", "ain't": "aint",
    "let's": "let us", "n't": " not", "'re": " are", "'ve": " have",
    "'ll": " will", "'d": " would", "'m": " am", "'t": " not", "'s": " is",
    "y'all": "you all", "wanna": "want to", "gonna": "going to",
    "gotta": "got to", "i'ma": "i am going to", "imma": "i am going to",
    "woulda": "would have", "coulda": "could have", "shoulda": "should have",
    "ma'am": "madam", "'d been": " had been", "'s been": " has been",
    "'d gone": " had gone", "'s gone": " has gone", "'d done": " had done",
    "'s got": " has got", "it's": "it is", "that's": "that is",
    "there's": "there is", "here's": "here is", "what's": "what is",
    "where's": "where is", "who's": "who is", "how's": "how is",
    "i'm": "i am", "you're": "you are", "we're": "we are",
    "they're": "they are", "you've": "you have", "we've": "we have",
    "they've": "they have", "i've": "i have", "you'll": "you will",
    "we'll": "we will", "they'll": "they will", "i'll": "i will",
    "you'd": "you would", "we'd": "we would", "they'd": "they would",
    "i'd": "i would", "she's": "she is", "he's": "he is",
    "she'll": "she will", "he'll": "he will", "she'd": "she would",
    "he'd": "he would",
}

_NUMBER_WORDS: dict[str, int] = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_MULTIPLIERS: dict[str, int] = {
    "hundred": 100, "thousand": 1000, "million": 1_000_000,
    "billion": 1_000_000_000,
}
_NUMBER_WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "billion": "1000000000",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
    "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
    "ninth": "9th", "tenth": "10th",
}


def _is_number_word(word: str) -> bool:
    return word in _NUMBER_WORDS or word in _MULTIPLIERS


def _parse_number_sequence(words: list[str]) -> str:
    results: list[str] = []
    current_sum = 0
    last_scale = 0

    for word in words:
        val = _NUMBER_WORDS.get(word) or _MULTIPLIERS.get(word) or 0
        is_multiplier = word in _MULTIPLIERS

        if is_multiplier:
            if current_sum == 0:
                current_sum = 1
            current_sum *= val
            last_scale = val
        else:
            if current_sum == 0:
                current_sum = val
                last_scale = 1
            else:
                can_merge = (last_scale >= 100 and val < last_scale) or \
                            (last_scale == 1 and (current_sum % 100 >= 20) and
                             (current_sum % 10 == 0) and val < 10)
                if can_merge:
                    current_sum += val
                    last_scale = 1
                else:
                    results.append(str(current_sum))
                    current_sum = val
                    last_scale = 1

    if current_sum > 0:
        results.append(str(current_sum))
    return " ".join(results)


def _convert_numbers(text: str) -> str:
    words = text.split()
    result: list[str] = []
    current_number_words: list[str] = []

    for word in words:
        if _is_number_word(word):
            current_number_words.append(word)
        else:
            if current_number_words:
                result.append(_parse_number_sequence(current_number_words))
                current_number_words = []
            result.append(word)

    if current_number_words:
        result.append(_parse_number_sequence(current_number_words))

    return " ".join(result)


def normalize_english(text: str) -> str:
    """Port of Swift TextNormalizer.normalize() — HuggingFace ASR leaderboard standard."""
    text = text.lower()

    # British → American spelling
    for british, american in _BRITISH_TO_AMERICAN.items():
        text = re.sub(r"\b" + re.escape(british) + r"\b", american, text)

    # Abbreviations
    for abbrev, expansion in _ABBREVIATIONS.items():
        text = re.sub(r"\b" + re.escape(abbrev) + r"\b", expansion, text)

    # Remove bracketed / parenthetical content
    text = re.sub(r"[<\[].*?[>\]]", "", text)
    text = re.sub(r"\([^)]+?\)", "", text)

    # Filler words
    text = re.sub(r"\b(hmm|mm|mhm|mmm|uh|um)\b", "", text)

    # Stuttering patterns
    text = re.sub(r"\b[a-z]+-\s*", "", text)

    # Apostrophe spacing
    text = text.replace(" '", "'")

    # "and a half" → "point five"
    text = text.replace(" and a half", " point five")

    # Letter/digit boundaries
    text = re.sub(r"([a-z])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([a-z])", r"\1 \2", text)

    # Remove spaces before ordinal suffixes
    text = re.sub(r"([0-9])\s+(st|nd|rd|th|s)\b", r"\1\2", text)

    # Diacritics
    for char, replacement in _ADDITIONAL_DIACRITICS.items():
        text = text.replace(char, replacement)

    # Currency / symbols
    text = text.replace("$", " dollar ")
    text = text.replace("&", " and ")
    text = text.replace("%", " percent ")
    text = text.replace("€", " euro ")
    text = text.replace("£", " pound ")
    text = text.replace("¥", " yen ")

    # Remove punctuation (keep word chars, spaces, apostrophes)
    text = re.sub(r"[^\w\s']", " ", text)

    # Contractions
    for contraction, expansion in _CONTRACTIONS.items():
        text = text.replace(contraction, expansion)

    # Number word → digit conversion
    text = _convert_numbers(text)
    for word, digit in _NUMBER_WORD_TO_DIGIT.items():
        text = re.sub(r"\b" + re.escape(word) + r"\b", digit, text)

    # Remove commas between digits
    text = re.sub(r"(\d),(\d)", r"\1\2", text)

    # Remove periods not followed by numbers
    text = re.sub(r"\.([^0-9]|$)", r" \1", text)

    # "a d" → "ad"
    text = text.replace("a d", "ad")

    # Final punctuation cleanup
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


_ARABIC_TO_CHINESE = {
    "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
    "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
}

def normalize_chinese(text: str) -> str:
    # Convert Arabic digits to Chinese characters (matches Swift CtcZhCnBenchmark)
    for digit, chinese in _ARABIC_TO_CHINESE.items():
        text = text.replace(digit, chinese)
    # Remove spaces, keep only CJK characters
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbf]", "", text)
    return text


# Kanji digit normalization matching Swift JapaneseAsrBenchmark behavior.
# Compound tens (21-99) must be replaced BEFORE simple tens (20, 30, …)
# to avoid partial matches (e.g. 二十一 must not become 2十1).
_KANJI_DIGIT_TABLE: list[tuple[str, str]] = []

def _build_kanji_table() -> list[tuple[str, str]]:
    units = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
    tens  = ["二十", "三十", "四十", "五十", "六十", "七十", "八十", "九十"]
    rows: list[tuple[str, str]] = []
    # 21-99: compound tens first
    for ti, ten in enumerate(tens, start=2):
        for ui, unit in enumerate(units, start=1):
            rows.append((ten + unit, str(ti * 10 + ui)))
    # 11-19
    for ui, unit in enumerate(units, start=1):
        rows.append(("十" + unit, str(10 + ui)))
    # Simple tens 20,30,...,90
    for ti, ten in enumerate(tens, start=2):
        rows.append((ten, str(ti * 10)))
    # 10
    rows.append(("十", "10"))
    # Single digits (must come after compound replacements)
    for ui, unit in enumerate(units, start=1):
        rows.append((unit, str(ui)))
    rows.append(("〇", "0"))
    rows.append(("零", "0"))
    # Full-width digits
    for i, fw in enumerate("０１２３４５６７８９"):
        rows.append((fw, str(i)))
    return rows

_KANJI_DIGIT_TABLE = _build_kanji_table()


def normalize_japanese(text: str) -> str:
    # Convert kanji/full-width numbers to ASCII digits (same as Swift benchmark)
    for kanji, digit in _KANJI_DIGIT_TABLE:
        text = text.replace(kanji, digit)
    # Remove spaces
    text = re.sub(r"\s+", "", text)
    # Keep kana, CJK, ASCII digits, half-width katakana
    text = re.sub(
        r"[^\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3400-\u4dbf\uff66-\uff9f0-9]",
        "",
        text,
    )
    return text


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_thchs30(max_files=None):
    """Load THCHS-30 samples from metadata.jsonl."""
    metadata_path = THCHS30_DIR / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"THCHS-30 not found at {THCHS30_DIR}\n"
            "Run: .build/release/fluidaudiocli ctc-zh-cn-benchmark --auto-download"
        )

    samples = []
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            audio_path = THCHS30_DIR / entry["file_name"]
            if audio_path.exists():
                samples.append({"audio": str(audio_path), "text": entry["text"]})
            if max_files and len(samples) >= max_files:
                break

    return samples


def load_jsut(max_files=None):
    """Load JSUT-basic5000 samples from metadata.jsonl."""
    metadata_path = JSUT_DIR / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"JSUT-basic5000 not found at {JSUT_DIR}\n"
            "Run: .build/release/fluidaudiocli ja-benchmark --auto-download"
        )

    samples = []
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            audio_path = JSUT_DIR / "audio" / entry["file_name"]
            if audio_path.exists():
                samples.append({"audio": str(audio_path), "text": entry["text"]})
            if max_files and len(samples) >= max_files:
                break

    return samples


def load_librispeech(max_files=None, subset="test-clean"):
    """Load LibriSpeech samples from .trans.txt files."""
    subset_dir = LIBRISPEECH_DIR / subset
    if not subset_dir.exists():
        raise FileNotFoundError(
            f"LibriSpeech {subset} not found at {subset_dir}\n"
            f"Run: .build/release/fluidaudiocli asr-benchmark --subset {subset}"
        )

    samples = []
    for trans_file in sorted(subset_dir.rglob("*.trans.txt")):
        chapter_dir = trans_file.parent
        with open(trans_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                file_id, *words = line.split()
                transcript = " ".join(words)
                audio_path = chapter_dir / f"{file_id}.flac"
                if audio_path.exists():
                    samples.append({"audio": str(audio_path), "text": transcript})
                if max_files and len(samples) >= max_files:
                    return samples

    return samples


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def get_audio_duration(audio_path: str) -> float:
    info = sf.info(audio_path)
    return info.frames / info.samplerate


def run_benchmark(model, samples, metric: str, language: str, batch_size: int):
    """
    metric: 'wer' or 'cer'
    Returns list of result dicts.
    """
    from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess

    results = []
    total = len(samples)

    for i, sample in enumerate(samples):
        audio_path = sample["audio"]
        reference = sample["text"]

        duration = get_audio_duration(audio_path)

        t0 = time.perf_counter()
        raw = model([audio_path], language=language, use_itn=True)
        elapsed = time.perf_counter() - t0

        hypothesis = rich_transcription_postprocess(raw[0]) if raw else ""

        if metric == "wer":
            ref_norm = normalize_english(reference)
            hyp_norm = normalize_english(hypothesis)
            score = wer(ref_norm, hyp_norm)
        elif metric == "cer_ja":
            ref_norm = normalize_japanese(reference)
            hyp_norm = normalize_japanese(hypothesis)
            score = cer(ref_norm, hyp_norm)
        else:
            ref_norm = normalize_chinese(reference)
            hyp_norm = normalize_chinese(hypothesis)
            score = cer(ref_norm, hyp_norm)

        rtfx = duration / elapsed if elapsed > 0 else 0.0

        results.append({
            "audio": audio_path,
            "reference": reference,
            "hypothesis": hypothesis,
            "score": score,
            "duration_sec": duration,
            "latency_sec": elapsed,
            "rtfx": rtfx,
        })

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total} files...")

    return results


def print_results(results, dataset_name: str, metric: str):
    scores = [r["score"] for r in results]
    rtfxs = [r["rtfx"] for r in results]
    total_audio = sum(r["duration_sec"] for r in results)
    total_proc = sum(r["latency_sec"] for r in results)
    overall_rtfx = total_audio / total_proc if total_proc > 0 else 0.0

    metric_label = {"wer": "WER", "cer": "CER", "cer_ja": "CER"}.get(metric, "CER")
    mean_score = statistics.mean(scores) * 100
    median_score = statistics.median(scores) * 100
    median_rtfx = statistics.median(rtfxs)

    print()
    print("=== Benchmark Results ===")
    print(f"Dataset: {dataset_name}")
    print(f"Model: SenseVoice Small")
    print(f"Files processed: {len(results)}")
    print()
    print(f"Average {metric_label}: {mean_score:.1f}%")
    print(f"Median {metric_label}: {median_score:.1f}%")
    print(f"Median RTFx: {median_rtfx:.1f}x")
    print(
        f"Overall RTFx: {overall_rtfx:.1f}x "
        f"({total_audio:.1f}s / {total_proc:.1f}s)"
    )

    # Score distribution
    below5 = sum(1 for s in scores if s < 0.05)
    below10 = sum(1 for s in scores if s < 0.10)
    below20 = sum(1 for s in scores if s < 0.20)
    n = len(scores)
    print()
    print(f"{metric_label} Distribution:")
    print(f"  <5%:  {below5} files ({below5/n*100:.1f}%)")
    print(f"  <10%: {below10} files ({below10/n*100:.1f}%)")
    print(f"  <20%: {below20} files ({below20/n*100:.1f}%)")


# ---------------------------------------------------------------------------
# ONNX model patch
# ---------------------------------------------------------------------------

def _patch_sensevoice_onnx(model_dir: str):
    """
    Fix type mismatch in the exported SenseVoice ONNX model.

    funasr_onnx exports a model where some comparison operators (Less, Greater,
    etc.) receive inputs of different types (tensor(float) vs tensor(int64)).
    onnxruntime >= 1.17 rejects this. We fix it by inserting Cast nodes so
    both inputs share the same element type.
    """
    import onnx
    from onnx import helper, shape_inference

    # Resolve the actual .onnx path (funasr_onnx caches under modelscope)
    cache_root = Path.home() / ".cache/modelscope/hub/models"
    if (cache_root / model_dir).exists():
        onnx_path = cache_root / model_dir / "model.onnx"
    else:
        onnx_path = Path(model_dir) / "model.onnx"

    if not onnx_path.exists():
        return  # Not exported yet; funasr_onnx will export on first load

    sentinel = onnx_path.with_suffix(".onnx.patched")
    if sentinel.exists():
        return  # Already patched

    print(f"Patching ONNX model at {onnx_path} ...")
    model = onnx.load(str(onnx_path))

    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    graph = model.graph

    # Build name → elem_type map from all typed value_infos + initializers
    type_map: dict[str, int] = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        tt = vi.type
        if tt.HasField("tensor_type"):
            type_map[vi.name] = tt.tensor_type.elem_type
    for init in graph.initializer:
        type_map[init.name] = init.data_type

    CMP_OPS = {"Less", "Greater", "LessOrEqual", "GreaterOrEqual", "Equal"}
    new_nodes = []
    cast_idx = 0

    for node in graph.node:
        if node.op_type in CMP_OPS and len(node.input) >= 2:
            t0 = type_map.get(node.input[0])
            t1 = type_map.get(node.input[1])
            if t0 is not None and t1 is not None and t0 != t1:
                cast_name = f"_type_fix_cast_{cast_idx}"
                cast_idx += 1
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[node.input[1]],
                    outputs=[cast_name],
                    to=t0,
                )
                new_nodes.append(cast_node)
                node.input[1] = cast_name

        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)

    onnx.save(model, str(onnx_path))
    sentinel.touch()
    print("ONNX model patched.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SenseVoice Small ONNX Benchmark")
    parser.add_argument(
        "--dataset",
        choices=["thchs30", "librispeech", "librispeech-other", "jsut", "all"],
        default="all",
        help="Dataset to benchmark (default: all)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files per dataset (default: all)",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="Language hint passed to SenseVoice (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size (default: 1)",
    )
    parser.add_argument(
        "--model-dir",
        default="iic/SenseVoiceSmall",
        help="Model directory or ModelScope ID (default: iic/SenseVoiceSmall)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use quantized model",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    # Load model (with ONNX type-mismatch fix applied first)
    from funasr_onnx import SenseVoiceSmall
    print(f"Loading SenseVoice Small from: {args.model_dir}")
    _patch_sensevoice_onnx(args.model_dir)
    model = SenseVoiceSmall(args.model_dir, batch_size=args.batch_size, quantize=args.quantize)
    print("Model loaded.\n")

    all_results = {}

    # THCHS-30 (Chinese CER)
    if args.dataset in ("thchs30", "all"):
        print("=== THCHS-30 (Chinese) ===")
        print("Loading dataset...")
        samples = load_thchs30(max_files=args.max_files)
        print(f"Loaded {len(samples)} files.")
        lang = args.language if args.language != "auto" else "zh"
        results = run_benchmark(model, samples, metric="cer", language=lang, batch_size=args.batch_size)
        print_results(results, "THCHS-30", "cer")
        all_results["thchs30"] = results

    # JSUT-basic5000 (Japanese CER)
    if args.dataset in ("jsut", "all"):
        print("\n=== JSUT-basic5000 (Japanese) ===")
        print("Loading dataset...")
        samples = load_jsut(max_files=args.max_files)
        print(f"Loaded {len(samples)} files.")
        lang = args.language if args.language != "auto" else "ja"
        results = run_benchmark(model, samples, metric="cer_ja", language=lang, batch_size=args.batch_size)
        print_results(results, "JSUT-basic5000", "cer_ja")
        all_results["jsut"] = results

    # LibriSpeech test-clean (English WER)
    if args.dataset in ("librispeech", "all"):
        print("\n=== LibriSpeech test-clean (English) ===")
        print("Loading dataset...")
        samples = load_librispeech(max_files=args.max_files, subset="test-clean")
        print(f"Loaded {len(samples)} files.")
        lang = args.language if args.language != "auto" else "en"
        results = run_benchmark(model, samples, metric="wer", language=lang, batch_size=args.batch_size)
        print_results(results, "LibriSpeech test-clean", "wer")
        all_results["librispeech_clean"] = results

    # LibriSpeech test-other (English WER)
    if args.dataset in ("librispeech-other", "all"):
        print("\n=== LibriSpeech test-other (English) ===")
        print("Loading dataset...")
        samples = load_librispeech(max_files=args.max_files, subset="test-other")
        print(f"Loaded {len(samples)} files.")
        lang = args.language if args.language != "auto" else "en"
        results = run_benchmark(model, samples, metric="wer", language=lang, batch_size=args.batch_size)
        print_results(results, "LibriSpeech test-other", "wer")
        all_results["librispeech_other"] = results

    # Save JSON output
    if args.output:
        output = {}
        for key, results in all_results.items():
            scores = [r["score"] for r in results]
            rtfxs = [r["rtfx"] for r in results]
            total_audio = sum(r["duration_sec"] for r in results)
            total_proc = sum(r["latency_sec"] for r in results)
            output[key] = {
                "files_processed": len(results),
                "average_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "median_rtfx": statistics.median(rtfxs),
                "overall_rtfx": total_audio / total_proc if total_proc > 0 else 0,
                "total_audio_sec": total_audio,
                "total_processing_sec": total_proc,
                "results": results,
            }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
