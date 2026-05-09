#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VibeVoice-ASR-7B Remote Benchmark (OpenAI-compatible chat completions endpoint).

The model produces structured diarization-style JSON output:
    [{"Start time": "...", "End time": "...", "Speaker ID": "...", "Content": "..."}, ...]

For ASR benchmarking we concatenate all `Content` fields in order and score that
against the dataset reference using the same WER/CER pipeline as the SenseVoice
and Qwen3 benchmarks (so the numbers stay directly comparable).

Install:
    pip install -r requirements.txt

Usage:
    # All datasets (THCHS-30, LibriSpeech test-clean/test-other, JSUT)
    python benchmark.py

    # One dataset, limited
    python benchmark.py --dataset librispeech --max-files 50

    # Tune concurrency
    python benchmark.py --dataset librispeech --max-files 200 --concurrency 4

API key resolution: --api-key > $VIBEVOICE_API_KEY > $QWEN3_API_KEY > $OPENAI_API_KEY
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

import numpy as np
import requests
import soundfile as sf

# Reuse dataset loaders, normalizers, metrics from the SenseVoice benchmark.
SHARED_DIR = Path(__file__).resolve().parent.parent / "SenseVoiceBenchmark"
sys.path.insert(0, str(SHARED_DIR))
try:
    from benchmark import (  # type: ignore
        cer,
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
        f"ERROR: cannot import shared helpers from {SHARED_DIR}/benchmark.py: {e}\n"
        "Keep VibeVoiceBenchmark next to SenseVoiceBenchmark."
    )
    sys.exit(2)


DEFAULT_API_URL = "https://next-api.fazhiplus.com/v1/chat/completions"
DEFAULT_MODEL = "VibeVoice-ASR-7B"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that transcribes audio input into text "
    "output in JSON format."
)
# Duration-aware user prompt template; {duration} is substituted at call time.
DEFAULT_USER_PROMPT_TEMPLATE = (
    "This is a {duration:.2f} seconds audio, please transcribe it with "
    "these keys: Start time, End time, Speaker ID, Content"
)

# Default output: <repo_root>/benchmark_results/<file>.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark_results"


def default_output_path(dataset: str) -> Path:
    name = "vibevoice_all.json" if dataset == "all" else f"vibevoice_{dataset.replace('-', '_')}.json"
    return DEFAULT_OUTPUT_DIR / name


# ---------------------------------------------------------------------------
# Audio → base64 wav data URL (16 kHz mono PCM_16)
# ---------------------------------------------------------------------------

def audio_to_data_url(audio_path: str, target_sr: int = 16000) -> tuple[str, float]:
    data, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
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
# Streaming SSE call
# ---------------------------------------------------------------------------

class TransientError(Exception):
    pass


def call_vibevoice(
    audio_data_url: str,
    duration: float,
    api_url: str,
    api_key: str,
    model: str,
    timeout: float,
    system_prompt: str,
    user_prompt_template: str,
    max_tokens: int = 32768,
    temperature: float = 0.0,
) -> tuple[str, dict]:
    """
    POST a non-streaming chat completion. Returns (assistant_content, full_response_dict).
    Raises TransientError on 429/5xx so the caller can retry.
    """
    user_prompt = user_prompt_template.format(duration=duration)

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "audio_url", "audio_url": {"url": audio_data_url}},
                {"type": "text", "text": user_prompt},
            ]},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "top_p": 1.0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(api_url, headers=headers, json=body, timeout=timeout)
    if resp.status_code in (429, 500, 502, 503, 504):
        body_text = resp.text[:200] if resp.text else ""
        raise TransientError(f"HTTP {resp.status_code}: {body_text}")
    if not resp.ok:
        body_text = resp.text[:500] if resp.text else ""
        raise RuntimeError(f"HTTP {resp.status_code}: {body_text}")

    payload = resp.json()  # `requests` decodes JSON bodies as UTF-8 by default — no mojibake.
    try:
        message = payload["choices"][0]["message"]
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )
        return str(content), payload
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected response shape: {payload}") from e


def call_with_retries(
    audio_data_url: str,
    duration: float,
    api_url: str,
    api_key: str,
    model: str,
    timeout: float,
    system_prompt: str,
    user_prompt_template: str,
    max_tokens: int,
    temperature: float,
    max_retries: int = 4,
) -> tuple[str, dict]:
    delay = 1.0
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return call_vibevoice(
                audio_data_url, duration, api_url, api_key, model, timeout,
                system_prompt, user_prompt_template, max_tokens, temperature,
            )
        except TransientError as e:
            last_err = e
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError(f"Exceeded retries: {last_err}")


# ---------------------------------------------------------------------------
# Response parsing — extract a clean transcript from the JSON
# ---------------------------------------------------------------------------

def _maybe_unmojibake(text: str) -> str:
    """
    Detect and repair UTF-8-as-Latin-1 mojibake. If `text` looks like it was
    decoded via latin-1 but actually contains UTF-8 byte sequences (very common
    on misconfigured SSE endpoints), round-trip it back to a real Unicode string.
    No-op for clean text.
    """
    if not text:
        return text
    # The hallmark of this mojibake is a high density of typical Latin-1
    # mid-byte renderings of UTF-8: "Ã", "Â", and especially "ä¸" / "ä¹" / "æ".
    suspicious_chars = ("Ã", "Â", "ä¸", "ä¹", "æ", "è", "é", "ï¼", "ã€", "ãƒ")
    if not any(ch in text for ch in suspicious_chars):
        return text
    try:
        repaired = text.encode("latin-1", errors="strict").decode("utf-8", errors="strict")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    # Only accept the repair if it didn't lose characters and actually produced
    # a recognizable script.
    return repaired


# Strip markdown ```json fences if present.
_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")
# Try to find a JSON array inside arbitrary text.
_ARRAY_RE = re.compile(r"\[[\s\S]*?\]")
# Plain "Content: ..." line fallback.
_CONTENT_LINE_RE = re.compile(
    r'(?im)^[\s\-\*]*"?Content"?\s*[:：]\s*"?(.+?)"?\s*[,}]?\s*$'
)


def extract_transcript(raw: str) -> str:
    """
    VibeVoice returns a JSON list like:
        [{"Start time": "0:00", "End time": "0:05", "Speaker ID": "S1", "Content": "..."}, ...]
    Sometimes wrapped in ```json fences, sometimes with leading/trailing prose.
    Concatenate every "Content" value in order, separated by single spaces.
    Falls back gracefully if the model didn't follow the schema.
    """
    if not raw:
        return ""

    # Repair UTF-8-as-Latin-1 mojibake before parsing (cheap no-op when clean).
    raw = _maybe_unmojibake(raw)

    # Strip markdown fences first (they confuse json.loads).
    text = _FENCE_RE.sub("", raw.strip())

    # 1. Direct JSON parse.
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # 2. Pull the first JSON array out of the text.
        m = _ARRAY_RE.search(text)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                parsed = None

    if isinstance(parsed, list):
        contents: list[str] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            v = item.get("Content") or item.get("content") or item.get("text") or ""
            if not isinstance(v, str):
                continue
            v = v.strip()
            if not v:
                continue
            # VibeVoice loves to fill remaining max_tokens with "[Silence]" or
            # "[silence]" / "[noise]" placeholders — drop them.
            if re.fullmatch(r"\[\s*(silence|noise|music|inaudible)\s*\]", v, re.IGNORECASE):
                continue
            # Drop consecutive duplicates (the model loops once it has nothing
            # left to transcribe, repeating the last real segment verbatim).
            if contents and contents[-1] == v:
                continue
            contents.append(v)
        if contents:
            return " ".join(contents).strip()

    if isinstance(parsed, dict):
        v = parsed.get("Content") or parsed.get("content") or parsed.get("text")
        if isinstance(v, str):
            return v.strip()

    # 3. Last-resort: scrape "Content: ..." lines.
    matches = _CONTENT_LINE_RE.findall(text)
    if matches:
        return " ".join(m.strip().strip('"').strip(",") for m in matches).strip()

    # 4. Plain text.
    return text.strip()


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
    system_prompt: str,
    user_prompt_template: str,
    max_tokens: int,
    temperature: float,
) -> dict | None:
    audio = sample["audio"]
    reference = sample["text"]

    try:
        data_url, duration = audio_to_data_url(audio)
    except Exception as e:
        return {"audio": audio, "error": f"decode: {e}"}

    t0 = time.perf_counter()
    try:
        raw, _ = call_with_retries(
            data_url, duration, api_url, api_key, model, timeout,
            system_prompt, user_prompt_template, max_tokens, temperature,
        )
    except Exception as e:
        return {"audio": audio, "error": f"api: {e}", "duration_sec": duration}
    latency = time.perf_counter() - t0

    transcript = extract_transcript(raw)

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

    return {
        "audio": audio,
        "raw_response_preview": raw[:300],
        "hypothesis_extracted": transcript[:500],
        "reference_norm": ref_n,
        "hypothesis_norm": hyp_n,
        "score": score,
        "duration_sec": duration,
        "latency_sec": latency,
        "rtfx": rtfx,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_dataset(samples, metric, *, api_url, api_key, model, timeout, concurrency,
                system_prompt, user_prompt_template, max_tokens, temperature):
    results: list[dict] = []
    failed: list[dict] = []
    total = len(samples)
    print(f"Submitting {total} files (concurrency={concurrency})...")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                process_one, s, api_url, api_key, model, timeout, metric,
                system_prompt, user_prompt_template, max_tokens, temperature,
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
                if len(failed) <= 5:
                    print(f"  [FAIL {len(failed)}] {res.get('audio', '?')}: {res.get('error', '?')}")
            else:
                results.append(res)
                running_scores.append(res["score"])
                running_rtfs.append(res["rtfx"])

            if done % 10 == 0 or done == total or done <= 3:
                if running_scores:
                    label = "WER" if metric == "wer" else "CER"
                    avg = statistics.mean(running_scores) * 100
                    avg_rtfx = statistics.mean(running_rtfs)
                    print(
                        f"  [{done}/{total}] avg {label}={avg:.2f}% "
                        f"avg rtfx={avg_rtfx:.2f}x failed={len(failed)}"
                    )
                else:
                    print(f"  [{done}/{total}] no successful results yet, failed={len(failed)}")
    return results, failed


def print_results(results, failed, dataset_name, metric, model_label):
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
    p.add_argument("--concurrency", type=int, default=4,
                   help="Parallel requests (default: 4)")
    p.add_argument("--timeout", type=float, default=180.0,
                   help="Per-request timeout in seconds (default: 180)")
    p.add_argument("--api-url", default=os.environ.get("VIBEVOICE_API_URL", DEFAULT_API_URL))
    p.add_argument("--model", default=os.environ.get("VIBEVOICE_MODEL", DEFAULT_MODEL))
    p.add_argument(
        "--api-key",
        default=(os.environ.get("VIBEVOICE_API_KEY")
                 or os.environ.get("QWEN3_API_KEY")
                 or os.environ.get("OPENAI_API_KEY")),
        help="API bearer token (or set $VIBEVOICE_API_KEY)",
    )
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--user-prompt-template", default=DEFAULT_USER_PROMPT_TEMPLATE,
                   help="Use {duration} as placeholder for the audio duration in seconds")
    p.add_argument(
        "--output",
        default=None,
        help="Write JSON here (default: <repo>/benchmark_results/vibevoice_<dataset>.json; pass 'none' to skip)",
    )
    args = p.parse_args()

    if args.output is None:
        args.output = str(default_output_path(args.dataset))
    elif args.output.lower() in ("none", "no", "off", ""):
        args.output = None

    if not args.api_key:
        print("ERROR: no API key. Pass --api-key or set $VIBEVOICE_API_KEY.", file=sys.stderr)
        sys.exit(2)

    print(f"Endpoint: {args.api_url}")
    print(f"Model:    {args.model}")
    print(f"Concurrency: {args.concurrency}, timeout: {args.timeout}s")
    if args.output:
        print(f"Output:   {args.output}")
    print()

    datasets = (
        ["thchs30", "librispeech", "librispeech-other", "jsut"]
        if args.dataset == "all" else [args.dataset]
    )

    aggregate: list[dict] = []
    for ds in datasets:
        if ds == "thchs30":
            display, metric, loader = "THCHS-30", "cer", lambda: load_thchs30(args.max_files)
        elif ds == "jsut":
            display, metric, loader = "JSUT-basic5000", "cer_ja", lambda: load_jsut(args.max_files)
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
            samples, metric,
            api_url=args.api_url, api_key=args.api_key, model=args.model,
            timeout=args.timeout, concurrency=args.concurrency,
            system_prompt=args.system_prompt,
            user_prompt_template=args.user_prompt_template,
            max_tokens=args.max_tokens, temperature=args.temperature,
        )
        model_label = f"{args.model} (remote)"
        print_results(results, failed, display, metric, model_label)
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
                if results and sum(r["latency_sec"] for r in results) > 0 else None
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
