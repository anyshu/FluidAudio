#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-ASR-1.7B Remote Benchmark (OpenAI-compatible chat completions endpoint).

Tests on the same datasets and uses the same text normalization as
Examples/SenseVoiceBenchmark/benchmark.py so the WER/CER numbers are directly
comparable.

Install dependencies:
    pip install -r requirements.txt

Usage:
    # All datasets (THCHS-30, LibriSpeech test-clean/test-other, JSUT)
    python benchmark.py

    # Single dataset, limited files
    python benchmark.py --dataset librispeech --max-files 100
    python benchmark.py --dataset thchs30 --max-files 200
    python benchmark.py --dataset jsut --max-files 200
    python benchmark.py --dataset librispeech-other --max-files 100

    # Save aggregate JSON
    python benchmark.py --output qwen3_remote_results.json

    # Tune concurrency (default 4). Server-side rate limits apply.
    python benchmark.py --dataset librispeech --max-files 100 --concurrency 8

API key resolution order (first non-empty wins):
    --api-key
    $QWEN3_API_KEY
    $OPENAI_API_KEY
"""

import argparse
import base64
import io
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests
import soundfile as sf

# Reuse dataset loaders, normalizers, and metrics from the SenseVoice benchmark
# so both benchmarks measure the exact same way (apples-to-apples WER/CER).
SHARED_DIR = Path(__file__).resolve().parent.parent / "SenseVoiceBenchmark"
sys.path.insert(0, str(SHARED_DIR))
try:
    from benchmark import (  # type: ignore
        cer,
        get_audio_duration,
        load_jsut,
        load_librispeech,
        load_thchs30,
        normalize_chinese,
        normalize_english,
        normalize_japanese,
        wer,
    )
except ImportError as e:
    print(
        "ERROR: cannot import shared helpers from "
        f"{SHARED_DIR}/benchmark.py: {e}\n"
        "Make sure the SenseVoiceBenchmark example exists alongside this one."
    )
    sys.exit(2)


DEFAULT_API_URL = "https://next-api.fazhiplus.com/v1/chat/completions"
DEFAULT_MODEL = "Qwen3-ASR-1.7B"

# Default output: <repo_root>/benchmark_results/<file>. Resolved from this
# script's location (Examples/Qwen3RemoteBenchmark/benchmark.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark_results"


def default_output_path(dataset: str) -> Path:
    name = "qwen3_all.json" if dataset == "all" else f"qwen3_{dataset.replace('-', '_')}.json"
    return DEFAULT_OUTPUT_DIR / name


# ---------------------------------------------------------------------------
# Response post-processing
# ---------------------------------------------------------------------------

# Qwen3-ASR returns:
#   "language English<asr_text>...transcript..."
# or sometimes
#   "language Chinese<asr_text>...transcript...</asr_text>"
# Strip the language tag prefix and any closing tag.
_ASR_TAG_OPEN = re.compile(r"^\s*language\s+\S+\s*<asr_text>", re.IGNORECASE)
_ASR_TAG_CLOSE = re.compile(r"</asr_text>\s*$", re.IGNORECASE)


def strip_qwen3_asr_tags(text: str) -> str:
    text = _ASR_TAG_OPEN.sub("", text)
    text = _ASR_TAG_CLOSE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Audio → base64 data URL
# ---------------------------------------------------------------------------

def audio_to_data_url(audio_path: str, target_sr: int = 16000) -> tuple[str, float]:
    """
    Decode any soundfile-supported audio (wav/flac/mp3 with libsndfile),
    downmix to mono, resample to 16 kHz, encode as 16-bit PCM WAV in memory,
    return ("data:audio/wav;base64,...", duration_seconds).
    """
    data, sr = sf.read(audio_path, dtype="float32", always_2d=False)

    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != target_sr:
        # Linear resample — good enough for ASR. SciPy is optional; do it manually.
        ratio = target_sr / sr
        new_len = int(round(len(data) * ratio))
        x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        data = np.interp(x_new, x_old, data).astype(np.float32)
        sr = target_sr

    duration = len(data) / sr

    pcm16 = np.clip(data * 32768.0, -32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    sf.write(buf, pcm16, sr, subtype="PCM_16", format="WAV")
    raw = buf.getvalue()

    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:audio/wav;base64,{b64}", duration


# ---------------------------------------------------------------------------
# Remote ASR call
# ---------------------------------------------------------------------------

class TransientError(Exception):
    pass


def call_qwen3(
    audio_data_url: str,
    api_url: str,
    api_key: str,
    model: str,
    timeout: float,
    user_prompt: str | None = None,
    system_prompt: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    POST to the OpenAI-compatible /chat/completions endpoint.

    Returns (transcript, raw_response_dict). Raises TransientError on 429/5xx
    so the caller can retry.
    """
    content: list[dict[str, Any]] = [
        {"type": "audio_url", "audio_url": {"url": audio_data_url}},
    ]
    if user_prompt:
        content.append({"type": "text", "text": user_prompt})

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    body = {
        "model": model,
        "messages": messages,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(api_url, headers=headers, json=body, timeout=timeout)

    if resp.status_code in (429, 500, 502, 503, 504):
        raise TransientError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

    payload = resp.json()
    try:
        message = payload["choices"][0]["message"]
        # Qwen3-ASR returns plain text in `content`. Some providers wrap it as a
        # list of content blocks — handle both.
        c = message.get("content", "")
        if isinstance(c, list):
            transcript = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in c
            )
        else:
            transcript = str(c)
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected response shape: {payload}") from e

    return strip_qwen3_asr_tags(transcript), payload


def call_with_retries(
    audio_data_url: str,
    api_url: str,
    api_key: str,
    model: str,
    timeout: float,
    user_prompt: str | None = None,
    system_prompt: str | None = None,
    max_retries: int = 4,
) -> tuple[str, dict[str, Any]]:
    delay = 1.0
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return call_qwen3(
                audio_data_url, api_url, api_key, model, timeout,
                user_prompt=user_prompt, system_prompt=system_prompt,
            )
        except TransientError as e:
            last_err = e
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError(f"Exceeded retries: {last_err}")


# ---------------------------------------------------------------------------
# Per-file work
# ---------------------------------------------------------------------------

def process_one(
    sample: dict,
    api_url: str,
    api_key: str,
    model: str,
    timeout: float,
    metric: str,
    user_prompt: str | None,
    system_prompt: str | None,
) -> dict | None:
    audio = sample["audio"]
    reference = sample["text"]

    try:
        data_url, duration = audio_to_data_url(audio)
    except Exception as e:
        return {"audio": audio, "error": f"decode: {e}"}

    t0 = time.perf_counter()
    try:
        transcript, raw = call_with_retries(
            data_url, api_url, api_key, model, timeout,
            user_prompt=user_prompt, system_prompt=system_prompt,
        )
    except Exception as e:
        return {"audio": audio, "error": f"api: {e}", "duration_sec": duration}
    latency = time.perf_counter() - t0

    if metric == "wer":
        ref_n = normalize_english(reference)
        hyp_n = normalize_english(transcript)
        score = wer(ref_n, hyp_n)
    elif metric == "cer":
        ref_n = normalize_chinese(reference)
        hyp_n = normalize_chinese(transcript)
        score = cer(ref_n, hyp_n)
    elif metric == "cer_ja":
        ref_n = normalize_japanese(reference)
        hyp_n = normalize_japanese(transcript)
        score = cer(ref_n, hyp_n)
    else:
        raise ValueError(f"unknown metric {metric}")

    rtfx = duration / latency if latency > 0 else 0.0

    usage = (raw or {}).get("usage", {}) if isinstance(raw, dict) else {}
    return {
        "audio": audio,
        "reference_norm": ref_n,
        "hypothesis_raw": transcript,
        "hypothesis_norm": hyp_n,
        "score": score,
        "duration_sec": duration,
        "latency_sec": latency,
        "rtfx": rtfx,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_dataset(
    samples: list[dict],
    metric: str,
    api_url: str,
    api_key: str,
    model: str,
    timeout: float,
    concurrency: int,
    user_prompt: str | None,
    system_prompt: str | None,
) -> tuple[list[dict], list[dict]]:
    results: list[dict] = []
    failed: list[dict] = []

    total = len(samples)
    print(f"Submitting {total} files (concurrency={concurrency})...")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                process_one, s, api_url, api_key, model, timeout, metric,
                user_prompt, system_prompt,
            ): s
            for s in samples
        }
        done = 0
        running_scores: list[float] = []
        running_rtfs: list[float] = []
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            if not res or "error" in (res or {}):
                failed.append(res or {})
            else:
                results.append(res)
                running_scores.append(res["score"])
                running_rtfs.append(res["rtfx"])

            if done % 10 == 0 or done == total or done <= 3:
                if running_scores:
                    avg = statistics.mean(running_scores) * 100
                    avg_rtfx = statistics.mean(running_rtfs)
                    label = "WER" if metric == "wer" else "CER"
                    print(
                        f"  [{done}/{total}] avg {label}={avg:.2f}% "
                        f"avg rtfx={avg_rtfx:.2f}x failed={len(failed)}"
                    )
                else:
                    print(f"  [{done}/{total}] no successful results yet, failed={len(failed)}")

    return results, failed


def print_results(
    results: list[dict],
    failed: list[dict],
    dataset_name: str,
    metric: str,
    model_label: str = "Qwen3 (remote)",
):
    """Match the SenseVoice benchmark output format exactly."""
    metric_label = {"wer": "WER", "cer": "CER", "cer_ja": "CER"}[metric]

    if not results:
        print()
        print("=== Benchmark Results ===")
        print(f"Dataset: {dataset_name}")
        print(f"Model: {model_label}")
        print(f"Files processed: 0 (failed: {len(failed)})")
        if failed:
            print()
            print("First failures:")
            for f in failed[:5]:
                print(f"  {f.get('audio', '?')}: {f.get('error', '?')}")
        return

    scores = [r["score"] for r in results]
    rtfxs = [r["rtfx"] for r in results]
    total_audio = sum(r["duration_sec"] for r in results)
    total_proc = sum(r["latency_sec"] for r in results)
    overall_rtfx = total_audio / total_proc if total_proc > 0 else 0.0

    mean_score = statistics.mean(scores) * 100
    median_score = statistics.median(scores) * 100
    median_rtfx = statistics.median(rtfxs)

    print()
    print("=== Benchmark Results ===")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_label}")
    print(f"Files processed: {len(results)}")
    print()
    print(f"Average {metric_label}: {mean_score:.1f}%")
    print(f"Median {metric_label}: {median_score:.1f}%")
    print(f"Median RTFx: {median_rtfx:.1f}x")
    print(f"Overall RTFx: {overall_rtfx:.1f}x ({total_audio:.1f}s / {total_proc:.1f}s)")

    n = len(scores)
    below5 = sum(1 for s in scores if s < 0.05)
    below10 = sum(1 for s in scores if s < 0.10)
    below20 = sum(1 for s in scores if s < 0.20)
    print()
    print(f"{metric_label} Distribution:")
    print(f"  <5%:  {below5} files ({below5 / n * 100:.1f}%)")
    print(f"  <10%: {below10} files ({below10 / n * 100:.1f}%)")
    print(f"  <20%: {below20} files ({below20 / n * 100:.1f}%)")

    # Token cost summary if the provider returned usage info on at least one call.
    used_tokens = [r.get("total_tokens") for r in results if r.get("total_tokens")]
    if used_tokens:
        print()
        print(f"Total tokens reported (sum across {len(used_tokens)} successful responses): "
              f"{sum(used_tokens)}")

    if failed:
        print()
        print(f"{len(failed)} failure(s). First 3:")
        for f in failed[:3]:
            print(f"  {f.get('audio', '?')}: {f.get('error', '?')}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dataset",
        choices=["thchs30", "librispeech", "librispeech-other", "jsut", "all"],
        default="all",
    )
    p.add_argument("--max-files", type=int, default=None, help="Limit per dataset (default: all)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout (seconds)")
    p.add_argument("--api-url", default=os.environ.get("QWEN3_API_URL", DEFAULT_API_URL))
    p.add_argument("--model", default=os.environ.get("QWEN3_MODEL", DEFAULT_MODEL))
    p.add_argument(
        "--api-key",
        default=os.environ.get("QWEN3_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        help="API bearer token (or set $QWEN3_API_KEY)",
    )
    p.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt. For general multimodal models (e.g. Qwen3-Omni) defaults to an "
             "ASR-only instruction; for Qwen3-ASR-* models defaults to none. Pass empty string to disable.",
    )
    p.add_argument(
        "--user-prompt",
        default=None,
        help="User text prompt sent alongside the audio. Default depends on model.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Write aggregate JSON here (default: <repo>/benchmark_results/qwen3_<dataset>.json; pass 'none' to skip)",
    )
    args = p.parse_args()

    # Auto-detect: general multimodal models like Qwen3-Omni need to be told to act as ASR.
    is_omni_like = (
        "omni" in args.model.lower()
        or "vl" in args.model.lower()
        or "audio" in args.model.lower() and "asr" not in args.model.lower()
    )
    default_system = (
        "You are an automatic speech recognition system. Output only the verbatim "
        "transcription of the input audio in the original spoken language. Do not add "
        "commentary, translation, summary, or any explanatory text. Just the transcript."
        if is_omni_like else ""
    )
    default_user = "Transcribe this audio verbatim." if is_omni_like else ""

    if args.system_prompt is None:
        args.system_prompt = default_system
    if args.user_prompt is None:
        args.user_prompt = default_user
    args.system_prompt = args.system_prompt or None  # empty -> None
    args.user_prompt = args.user_prompt or None

    # Use a model-specific output filename so two runs (e.g. Qwen3-ASR vs Qwen3-Omni)
    # don't clobber each other.
    if args.output is None:
        model_tag = re.sub(r"[^a-zA-Z0-9]+", "_", args.model).strip("_").lower() or "qwen3"
        ds_tag = "all" if args.dataset == "all" else args.dataset.replace("-", "_")
        args.output = str(DEFAULT_OUTPUT_DIR / f"{model_tag}_{ds_tag}.json")
    elif args.output.lower() in ("none", "no", "off", ""):
        args.output = None

    if not args.api_key:
        print("ERROR: no API key. Pass --api-key or set $QWEN3_API_KEY.", file=sys.stderr)
        sys.exit(2)

    print(f"Endpoint: {args.api_url}")
    print(f"Model:    {args.model}")
    print(f"Concurrency: {args.concurrency}")
    if args.system_prompt:
        print(f"System prompt: {args.system_prompt[:80]}{'...' if len(args.system_prompt) > 80 else ''}")
    if args.user_prompt:
        print(f"User prompt:   {args.user_prompt[:80]}")
    if args.output:
        print(f"Output:   {args.output}")
    print()

    datasets = (
        ["thchs30", "librispeech", "librispeech-other", "jsut"]
        if args.dataset == "all"
        else [args.dataset]
    )

    aggregate: list[dict] = []
    for ds in datasets:
        if ds == "thchs30":
            display, metric, loader = "THCHS-30", "cer", lambda: load_thchs30(args.max_files)
        elif ds == "jsut":
            display, metric, loader = (
                "JSUT-basic5000", "cer_ja", lambda: load_jsut(args.max_files),
            )
        elif ds == "librispeech":
            display, metric, loader = (
                "LibriSpeech test-clean", "wer",
                lambda: load_librispeech(args.max_files, subset="test-clean"),
            )
        elif ds == "librispeech-other":
            display, metric, loader = (
                "LibriSpeech test-other", "wer",
                lambda: load_librispeech(args.max_files, subset="test-other"),
            )
        else:
            print(f"Unknown dataset: {ds}")
            continue

        print(f"=== {display} ===")
        try:
            samples = loader()
        except FileNotFoundError as e:
            print(f"Skipping {display}: {e}")
            print()
            continue
        print(f"Loaded {len(samples)} files.")

        results, failed = run_dataset(
            samples, metric, args.api_url, args.api_key, args.model,
            args.timeout, args.concurrency,
            user_prompt=args.user_prompt, system_prompt=args.system_prompt,
        )
        print_results(results, failed, display, metric, model_label=f"{args.model} (remote)")
        print()

        aggregate.append({
            "dataset": display,
            "metric": {"wer": "WER", "cer": "CER", "cer_ja": "CER"}[metric],
            "files_processed": len(results),
            "failed": len(failed),
            "average_score_pct": (
                statistics.mean(r["score"] for r in results) * 100 if results else None
            ),
            "median_score_pct": (
                statistics.median(r["score"] for r in results) * 100 if results else None
            ),
            "overall_rtfx": (
                sum(r["duration_sec"] for r in results)
                / sum(r["latency_sec"] for r in results)
                if results and sum(r["latency_sec"] for r in results) > 0
                else None
            ),
            "median_rtfx": (
                statistics.median(r["rtfx"] for r in results) if results else None
            ),
            "audio_duration_s": sum(r["duration_sec"] for r in results) if results else 0,
            "processing_time_s": sum(r["latency_sec"] for r in results) if results else 0,
        })

    if args.output:
        out = {
            "endpoint": args.api_url,
            "model": args.model,
            "datasets": aggregate,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON: {out_path}")


if __name__ == "__main__":
    main()
