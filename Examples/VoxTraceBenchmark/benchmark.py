#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VoxTrace remote ASR + diarization benchmark.

The API call shape follows ~/working/temp/voxtrace.py: an OpenAI-compatible
/chat/completions request with a base64 WAV audio_url payload. The default model
returns diarization-style JSON segments with Start time, End time, Speaker ID,
and Content fields.

Install:
    pip install -r requirements.txt

Usage:
    # ASR benchmark on THCHS-30, LibriSpeech test-clean/test-other, and JSUT
    python benchmark.py --task asr

    # Raw diarization benchmark on local AMI SDM files + RTTM references
    python benchmark.py --task diarization --dataset ami --diarization-api-url http://127.0.0.1:28211/v1/audio/diarization --max-files 2

    # Final diarized_json benchmark through the chat API
    python benchmark.py --task diarization --dataset ami --diarization-source chat --mode full --single-file ES2004a

API key resolution: --api-key > $VOXTRACE_API_KEY > $OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import base64
import io
import itertools
import json
import os
import re
import threading
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
import soundfile as sf

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
        "Keep VoxTraceBenchmark next to SenseVoiceBenchmark."
    )
    sys.exit(2)


DEFAULT_API_URL = "https://next-api.fazhipro.com/v1/chat/completions"
DEFAULT_MODEL = "arcships-asr-diarize"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that transcribes audio input into text output in JSON format."
DEFAULT_USER_PROMPT_TEMPLATE = ""
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark_results"
LOCAL_AUDIO_SERVER: tuple[Any, str] | None = None
LOCAL_AUDIO_SERVER_LOCK = threading.Lock()
AMI_TEST_SET = [
    "EN2002a", "EN2002b", "EN2002c", "EN2002d",
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
    "IS1009a", "IS1009b", "IS1009c", "IS1009d",
    "TS3003a", "TS3003b", "TS3003c", "TS3003d",
]


class TransientError(Exception):
    pass


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: str
    text: str = ""


def default_output_path(task: str, dataset: str, mode: str | None, diarization_source: str = "raw") -> Path:
    dataset_name = dataset.replace("-", "_")
    if task == "asr":
        return DEFAULT_OUTPUT_DIR / f"voxtrace_asr_{dataset_name}.json"
    if diarization_source == "chat":
        return DEFAULT_OUTPUT_DIR / f"voxtrace_{task}_{mode}_{dataset_name}.json"
    return DEFAULT_OUTPUT_DIR / f"voxtrace_{task}_raw_{dataset_name}.json"


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
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{b64}", duration


def call_voxtrace(
    audio_data_url: str,
    api_url: str,
    api_key: str,
    model: str,
    timeout: float,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": audio_data_url}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "top_p": 1.0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(api_url, headers=headers, json=body, timeout=timeout)
    if resp.status_code in (429, 500, 502, 503, 504):
        raise TransientError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

    payload = resp.json()
    try:
        content = payload["choices"][0]["message"].get("content", "")
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected response shape: {payload}") from e
    if isinstance(content, list):
        content = "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
    return str(content), payload


def call_with_retries(*args: Any, max_retries: int = 4, **kwargs: Any) -> tuple[str, dict[str, Any]]:
    delay = 1.0
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return call_voxtrace(*args, **kwargs)
        except TransientError as e:
            last_err = e
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError(f"Exceeded retries: {last_err}")


def call_raw_diarization(
    audio_url: str,
    api_url: str,
    api_key: str | None,
    model: str,
    timeout: float,
    speaker_options: dict[str, int] | None = None,
) -> list[Segment]:
    if api_url.rstrip("/").endswith("/audio/diarization"):
        body: dict[str, Any] = {"audio_url": audio_url}
        if speaker_options:
            body["speaker_options"] = speaker_options
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = requests.post(api_url, headers=headers, json=body, timeout=timeout)
    else:
        data: dict[str, str] = {"model": model}
        data["response_format"] = "diarized_json"
        if speaker_options:
            if speaker_options.get("min_speakers") is not None:
                data["min_speakers"] = str(speaker_options["min_speakers"])
            if speaker_options.get("max_speakers") is not None:
                data["max_speakers"] = str(speaker_options["max_speakers"])
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        with open(audio_url, "rb") as audio_file:
            files = {"file": (Path(audio_url).name, audio_file)}
            resp = requests.post(api_url, headers=headers, data=data, files=files, timeout=timeout)
    if resp.status_code in (429, 500, 502, 503, 504):
        raise TransientError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    payload = resp.json()
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    turns = payload.get("speaker_turns") or metadata.get("speaker_turns") or payload.get("segments")
    if not isinstance(turns, list):
        raise RuntimeError(f"Unexpected diarization response shape: {payload}")
    segments: list[Segment] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        start = parse_time(turn.get("start"))
        end = parse_time(turn.get("end"))
        speaker = turn.get("speaker") or turn.get("Speaker ID") or turn.get("speaker_id")
        if start is None or end is None or end <= start or speaker is None:
            continue
        segments.append(Segment(start=start, end=end, speaker=str(speaker)))
    return sorted(segments, key=lambda s: (s.start, s.end))


def call_raw_diarization_with_retries(*args: Any, max_retries: int = 4, **kwargs: Any) -> list[Segment]:
    delay = 1.0
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return call_raw_diarization(*args, **kwargs)
        except TransientError as e:
            last_err = e
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError(f"Exceeded retries: {last_err}")


def raw_diarization_audio_input(audio_path: Path, data_url: str, options: argparse.Namespace) -> str:
    if not options.diarization_api_url.rstrip("/").endswith("/audio/diarization"):
        return str(audio_path)
    if options.audio_base_url:
        return f"{options.audio_base_url.rstrip('/')}/{audio_path.name}"
    return data_url


def local_audio_url(audio_path: Path) -> str:
    global LOCAL_AUDIO_SERVER
    with LOCAL_AUDIO_SERVER_LOCK:
        if LOCAL_AUDIO_SERVER is None:
            from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
            from functools import partial

            directory = str(audio_path.parent)
            handler = partial(SimpleHTTPRequestHandler, directory=directory)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_port}"
            LOCAL_AUDIO_SERVER = (server, base_url)
        _, base_url = LOCAL_AUDIO_SERVER
    return f"{base_url}/{audio_path.name}"


_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")
_ARRAY_RE = re.compile(r"\[[\s\S]*?\]")
_CONTENT_LINE_RE = re.compile(r'(?im)^[\s\-\*]*"?Content"?\s*[:：]\s*"?(.+?)"?\s*[,}]?\s*$')


def parse_time(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip().lower().replace("seconds", "").replace("second", "").replace("sec", "").strip()
    if not text:
        return None
    if ":" not in text:
        try:
            return float(text)
        except ValueError:
            return None
    parts = text.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    return None


def _json_candidates(raw: str) -> list[Any]:
    text = _FENCE_RE.sub("", raw.strip())
    candidates = [text]
    match = _ARRAY_RE.search(text)
    if match:
        candidates.append(match.group(0))
    parsed: list[Any] = []
    for candidate in candidates:
        try:
            parsed.append(json.loads(candidate))
        except json.JSONDecodeError:
            continue
    return parsed


def extract_segments(raw: str) -> list[Segment]:
    for parsed in _json_candidates(raw):
        items = parsed if isinstance(parsed, list) else [parsed]
        segments: list[Segment] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            start = parse_time(item.get("Start time") or item.get("start") or item.get("start_time"))
            end = parse_time(item.get("End time") or item.get("end") or item.get("end_time"))
            speaker = item.get("Speaker ID") or item.get("speaker") or item.get("speaker_id") or item.get("spk")
            text = item.get("Content") or item.get("content") or item.get("text") or ""
            if start is None or end is None or end <= start:
                continue
            segments.append(Segment(start=start, end=end, speaker=str(speaker or "S0"), text=str(text).strip()))
        if segments:
            return sorted(segments, key=lambda s: (s.start, s.end))
    return []


def extract_transcript(raw: str) -> str:
    segments = extract_segments(raw)
    if segments:
        parts: list[str] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            if re.fullmatch(r"\[\s*(silence|noise|music|inaudible)\s*\]", text, re.IGNORECASE):
                continue
            if parts and parts[-1] == text:
                continue
            parts.append(text)
        return " ".join(parts).strip()

    text = _FENCE_RE.sub("", raw.strip())
    matches = _CONTENT_LINE_RE.findall(text)
    if matches:
        return " ".join(m.strip().strip('"').strip(",") for m in matches).strip()
    return text


def process_asr_sample(sample: dict[str, str], options: argparse.Namespace, metric: str) -> dict[str, Any]:
    audio = sample["audio"]
    reference = sample["text"]
    try:
        data_url, duration = audio_to_data_url(audio)
    except Exception as e:
        return {"audio": audio, "error": f"decode: {e}"}

    user_prompt = options.user_prompt_template.format(duration=duration)
    t0 = time.perf_counter()
    try:
        raw, _ = call_with_retries(
            data_url,
            options.api_url,
            options.api_key,
            options.model,
            options.timeout,
            options.system_prompt,
            user_prompt,
            options.max_tokens,
            options.temperature,
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

    return {
        "audio": audio,
        "raw_response_preview": raw[:300],
        "hypothesis_extracted": transcript[:500],
        "reference_norm": ref_n,
        "hypothesis_norm": hyp_n,
        "score": score,
        "duration_sec": duration,
        "latency_sec": latency,
        "rtfx": duration / latency if latency > 0 else 0.0,
    }


def load_rttm(path: Path) -> list[Segment]:
    segments: list[Segment] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            segments.append(Segment(start=start, end=start + duration, speaker=parts[7]))
    return segments


def active_speakers(segments: list[Segment], t: float) -> set[str]:
    return {s.speaker for s in segments if s.start <= t < s.end}


def in_collar(reference: list[Segment], t: float, collar: float) -> bool:
    if collar <= 0:
        return False
    return any(abs(t - s.start) < collar or abs(t - s.end) < collar for s in reference)


def best_speaker_mapping(reference: list[Segment], hypothesis: list[Segment], frame_times: list[float]) -> dict[str, str]:
    ref_speakers = sorted({s.speaker for s in reference})
    hyp_speakers = sorted({s.speaker for s in hypothesis})
    if not ref_speakers or not hyp_speakers:
        return {}

    overlap = {h: {r: 0 for r in ref_speakers} for h in hyp_speakers}
    for t in frame_times:
        ref_active = active_speakers(reference, t)
        hyp_active = active_speakers(hypothesis, t)
        for h in hyp_active:
            for r in ref_active:
                if h in overlap and r in overlap[h]:
                    overlap[h][r] += 1

    if len(hyp_speakers) <= 8 and len(ref_speakers) <= 8:
        best_score = -1
        best_map: dict[str, str] = {}
        assignment_size = min(len(hyp_speakers), len(ref_speakers))
        for selected_hyps in itertools.combinations(hyp_speakers, assignment_size):
            for refs in itertools.permutations(ref_speakers, assignment_size):
                mapping = dict(zip(selected_hyps, refs))
                score = sum(overlap[h][r] for h, r in mapping.items())
                if score > best_score:
                    best_score = score
                    best_map = mapping
        return best_map

    mapping: dict[str, str] = {}
    used_refs: set[str] = set()
    pairs = sorted(
        ((overlap[h][r], h, r) for h in hyp_speakers for r in ref_speakers),
        reverse=True,
    )
    for _, h, r in pairs:
        if h in mapping or r in used_refs:
            continue
        mapping[h] = r
        used_refs.add(r)
    return mapping


def compute_der(
    reference: list[Segment],
    hypothesis: list[Segment],
    *,
    duration: float,
    collar: float,
    ignore_overlap: bool,
    frame_step: float,
) -> dict[str, Any]:
    frame_times: list[float] = []
    frame_count = int(duration / frame_step) + 1
    for i in range(frame_count):
        t = i * frame_step + frame_step / 2
        if t > duration or in_collar(reference, t, collar):
            continue
        ref_active = active_speakers(reference, t)
        if ignore_overlap and len(ref_active) > 1:
            continue
        frame_times.append(t)

    mapping = best_speaker_mapping(reference, hypothesis, frame_times)
    miss = 0
    false_alarm = 0
    speaker_error = 0
    reference_speech = 0

    for t in frame_times:
        ref_active = active_speakers(reference, t)
        hyp_active = active_speakers(hypothesis, t)
        n_ref = len(ref_active)
        n_hyp = len(hyp_active)
        reference_speech += n_ref
        correct = sum(1 for h in hyp_active if mapping.get(h) in ref_active)
        miss += max(n_ref - n_hyp, 0)
        false_alarm += max(n_hyp - n_ref, 0)
        speaker_error += max(min(n_ref, n_hyp) - correct, 0)

    if reference_speech == 0:
        return {"der": None, "miss_rate": None, "false_alarm_rate": None, "speaker_error_rate": None, "scored_frames": 0}

    der = (miss + false_alarm + speaker_error) / reference_speech
    return {
        "der": der,
        "miss_rate": miss / reference_speech,
        "false_alarm_rate": false_alarm / reference_speech,
        "speaker_error_rate": speaker_error / reference_speech,
        "scored_frames": len(frame_times),
        "mapping": mapping,
    }


def diarization_paths(dataset: str, meeting: str, datasets_root: Path) -> tuple[Path, Path]:
    if dataset == "ami":
        return (
            datasets_root / "ami_official" / "sdm" / f"{meeting}.Mix-Headset.wav",
            datasets_root / "ami_official" / "rttm" / f"{meeting}.rttm",
        )
    if dataset == "voxconverse":
        return (
            datasets_root / "voxconverse" / "voxconverse_test_wav" / f"{meeting}.wav",
            datasets_root / "voxconverse" / "rttm_repo" / "test" / f"{meeting}.rttm",
        )
    if dataset == "callhome":
        return (datasets_root / "callhome_eng" / f"{meeting}.wav", datasets_root / "callhome_eng" / "rttm" / f"{meeting}.rttm")
    raise ValueError(f"unknown diarization dataset {dataset}")


def list_diarization_meetings(dataset: str, datasets_root: Path, max_files: int | None, single_file: str | None) -> list[str]:
    if single_file:
        return [single_file]
    if dataset == "ami":
        meetings = [m for m in AMI_TEST_SET if all(p.exists() for p in diarization_paths(dataset, m, datasets_root))]
    elif dataset == "voxconverse":
        wav_dir = datasets_root / "voxconverse" / "voxconverse_test_wav"
        meetings = sorted(p.stem for p in wav_dir.glob("*.wav") if diarization_paths(dataset, p.stem, datasets_root)[1].exists())
    elif dataset == "callhome":
        wav_dir = datasets_root / "callhome_eng"
        meetings = sorted(p.stem for p in wav_dir.glob("*.wav") if diarization_paths(dataset, p.stem, datasets_root)[1].exists())
    else:
        raise ValueError(f"unknown diarization dataset {dataset}")
    return meetings[:max_files] if max_files else meetings


def process_diarization_meeting(meeting: str, options: argparse.Namespace) -> dict[str, Any]:
    audio_path, rttm_path = diarization_paths(options.dataset, meeting, Path(options.datasets_root).expanduser())
    if not audio_path.exists():
        return {"meeting": meeting, "error": f"audio not found: {audio_path}"}
    if not rttm_path.exists():
        return {"meeting": meeting, "error": f"rttm not found: {rttm_path}"}

    try:
        data_url, duration = audio_to_data_url(str(audio_path))
        reference = load_rttm(rttm_path)
    except Exception as e:
        return {"meeting": meeting, "error": f"load: {e}"}

    user_prompt = options.user_prompt_template.format(duration=duration)
    t0 = time.perf_counter()
    try:
        if options.diarization_source == "raw":
            hypothesis = call_raw_diarization_with_retries(
                raw_diarization_audio_input(audio_path, data_url, options),
                options.diarization_api_url,
                options.api_key,
                options.model,
                options.timeout,
                speaker_options=options.speaker_options,
            )
            raw_preview = ""
        else:
            raw, _ = call_with_retries(
                data_url,
                options.api_url,
                options.api_key,
                options.model,
                options.timeout,
                options.system_prompt,
                user_prompt,
                options.max_tokens,
                options.temperature,
            )
            hypothesis = extract_segments(raw)
            raw_preview = raw[:300]
    except Exception as e:
        return {"meeting": meeting, "error": f"api: {e}", "duration_sec": duration}
    latency = time.perf_counter() - t0
    metrics = compute_der(
        reference,
        hypothesis,
        duration=duration,
        collar=options.collar,
        ignore_overlap=options.ignore_overlap,
        frame_step=options.frame_step,
    )
    if metrics["der"] is None:
        return {"meeting": meeting, "error": "no scored reference frames", "duration_sec": duration}

    return {
        "meeting": meeting,
        "audio": str(audio_path),
        "rttm": str(rttm_path),
        "raw_response_preview": raw_preview,
        "source": options.diarization_source,
        "der": metrics["der"],
        "miss_rate": metrics["miss_rate"],
        "false_alarm_rate": metrics["false_alarm_rate"],
        "speaker_error_rate": metrics["speaker_error_rate"],
        "duration_sec": duration,
        "latency_sec": latency,
        "rtfx": duration / latency if latency > 0 else 0.0,
        "detected_speakers": len({s.speaker for s in hypothesis}),
        "reference_speakers": len({s.speaker for s in reference}),
        "hypothesis_segments": len(hypothesis),
        "reference_segments": len(reference),
        "scored_frames": metrics["scored_frames"],
    }


def run_asr_dataset(name: str, options: argparse.Namespace) -> dict[str, Any] | None:
    if name == "thchs30":
        display, metric, samples = "THCHS-30", "cer", load_thchs30(options.max_files)
    elif name == "jsut":
        display, metric, samples = "JSUT-basic5000", "cer_ja", load_jsut(options.max_files)
    elif name == "librispeech":
        display, metric, samples = "LibriSpeech test-clean", "wer", load_librispeech(options.max_files, subset="test-clean")
    elif name == "librispeech-other":
        display, metric, samples = "LibriSpeech test-other", "wer", load_librispeech(options.max_files, subset="test-other")
    else:
        raise ValueError(f"unknown ASR dataset {name}")

    print(f"=== ASR: {display} ===")
    results, failed = run_parallel(samples, lambda s: process_asr_sample(s, options, metric), options.concurrency)
    print_asr_results(results, failed, display, metric)
    return summarize_asr(display, metric, results, failed)


def run_diarization_dataset(options: argparse.Namespace) -> dict[str, Any]:
    meetings = list_diarization_meetings(options.dataset, Path(options.datasets_root).expanduser(), options.max_files, options.single_file)
    print(f"=== Diarization: {options.dataset} ({len(meetings)} file(s)) ===")
    results, failed = run_parallel(meetings, lambda m: process_diarization_meeting(m, options), options.concurrency)
    print_diarization_results(results, failed, options.dataset)
    return summarize_diarization(options.dataset, results, failed)


def run_parallel(items: list[Any], worker: Any, concurrency: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    total = len(items)
    print(f"Submitting {total} files (concurrency={concurrency})...")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(worker, item): item for item in items}
        for done, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            if "error" in res:
                failed.append(res)
                if len(failed) <= 5:
                    print(f"  [FAIL {len(failed)}] {res.get('audio') or res.get('meeting', '?')}: {res['error']}")
            else:
                results.append(res)
            print(f"  [{done}/{total}] ok={len(results)} failed={len(failed)}")
    return results, failed


def print_asr_results(results: list[dict[str, Any]], failed: list[dict[str, Any]], display: str, metric: str) -> None:
    label = "WER" if metric == "wer" else "CER"
    if not results:
        print(f"No successful ASR results for {display}; failed={len(failed)}")
        return
    scores = [r["score"] for r in results]
    rtfxs = [r["rtfx"] for r in results]
    total_audio = sum(r["duration_sec"] for r in results)
    total_latency = sum(r["latency_sec"] for r in results)
    print(f"Average {label}: {statistics.mean(scores) * 100:.1f}%")
    print(f"Median {label}:  {statistics.median(scores) * 100:.1f}%")
    print(f"Median RTFx: {statistics.median(rtfxs):.1f}x")
    print(f"Overall RTFx: {total_audio / total_latency if total_latency > 0 else 0.0:.1f}x")
    if failed:
        print(f"Failed: {len(failed)}")
    print()


def print_diarization_results(results: list[dict[str, Any]], failed: list[dict[str, Any]], dataset: str) -> None:
    if not results:
        print(f"No successful diarization results for {dataset}; failed={len(failed)}")
        return
    ders = [r["der"] for r in results]
    rtfxs = [r["rtfx"] for r in results]
    print(f"Average DER: {statistics.mean(ders) * 100:.1f}%")
    print(f"Median DER:  {statistics.median(ders) * 100:.1f}%")
    print(f"Median RTFx: {statistics.median(rtfxs):.1f}x")
    print("Meeting        DER %    Miss %     FA %     SE %   Speakers     RTFx")
    for r in sorted(results, key=lambda item: item["der"]):
        print(
            f"{r['meeting']:<12} {r['der'] * 100:6.1f} {r['miss_rate'] * 100:9.1f} "
            f"{r['false_alarm_rate'] * 100:8.1f} {r['speaker_error_rate'] * 100:8.1f} "
            f"{r['detected_speakers']:>3}/{r['reference_speakers']:<3} {r['rtfx']:8.1f}x"
        )
    if failed:
        print(f"Failed: {len(failed)}")
    print()


def summarize_asr(display: str, metric: str, results: list[dict[str, Any]], failed: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task": "asr",
        "dataset": display,
        "metric": "WER" if metric == "wer" else "CER",
        "files_processed": len(results),
        "failed": len(failed),
        "average_score_pct": statistics.mean(r["score"] for r in results) * 100 if results else None,
        "median_score_pct": statistics.median(r["score"] for r in results) * 100 if results else None,
        "overall_rtfx": (
            sum(r["duration_sec"] for r in results) / sum(r["latency_sec"] for r in results)
            if results and sum(r["latency_sec"] for r in results) > 0 else None
        ),
        "items": results,
        "failed_items": failed,
    }


def summarize_diarization(dataset: str, results: list[dict[str, Any]], failed: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task": "diarization",
        "dataset": dataset,
        "metric": "DER",
        "files_processed": len(results),
        "failed": len(failed),
        "average_der_pct": statistics.mean(r["der"] for r in results) * 100 if results else None,
        "median_der_pct": statistics.median(r["der"] for r in results) * 100 if results else None,
        "overall_rtfx": (
            sum(r["duration_sec"] for r in results) / sum(r["latency_sec"] for r in results)
            if results and sum(r["latency_sec"] for r in results) > 0 else None
        ),
        "items": results,
        "failed_items": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["asr", "diarization"], default="asr")
    parser.add_argument(
        "--mode",
        choices=["full", "chunk_global"],
        help="Required for --task diarization --diarization-source chat. Ignored for ASR/raw diarization.",
    )
    parser.add_argument(
        "--diarization-source",
        choices=["raw", "chat"],
        default="raw",
        help="raw uses /v1/audio/diarization speaker_turns; chat scores final diarized_json from /v1/chat/completions.",
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help="ASR: thchs30/librispeech/librispeech-other/jsut/all. Diarization: ami/voxconverse/callhome.",
    )
    parser.add_argument("--single-file", help="Diarization meeting id, e.g. ES2004a")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=12000.0)
    parser.add_argument("--api-url", default=os.environ.get("VOXTRACE_API_URL", DEFAULT_API_URL))
    parser.add_argument(
        "--diarization-api-url",
        default=os.environ.get("VOXTRACE_DIARIZATION_API_URL", "http://127.0.0.1:28211/v1/audio/diarization"),
        help="/v1/audio/diarization JSON audio_url endpoint, or /v1/audio/transcriptions multipart fallback.",
    )
    parser.add_argument(
        "--audio-base-url",
        default=os.environ.get("VOXTRACE_AUDIO_BASE_URL"),
        help="Base URL that exposes the local dataset audio files by basename for /v1/audio/diarization.",
    )
    parser.add_argument("--model", default=os.environ.get("VOXTRACE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.environ.get("VOXTRACE_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-prompt-template", default=DEFAULT_USER_PROMPT_TEMPLATE)
    parser.add_argument("--datasets-root", default=str(Path.home() / "FluidAudioDatasets"))
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--ignore-overlap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-step", type=float, default=0.01)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--output", default=None, help="Write JSON here (default: benchmark_results/voxtrace_*.json; 'none' disables)")
    args = parser.parse_args()

    if args.task == "asr" or (args.task == "diarization" and args.diarization_source == "chat"):
        if not args.api_key:
            print("ERROR: no API key. Pass --api-key or set $VOXTRACE_API_KEY.", file=sys.stderr)
            sys.exit(2)

    if args.min_speakers is not None and args.max_speakers is not None and args.min_speakers > args.max_speakers:
        parser.error("--min-speakers must be <= --max-speakers")
    args.speaker_options = None
    if args.min_speakers is not None or args.max_speakers is not None:
        args.speaker_options = {
            key: value
            for key, value in (("min_speakers", args.min_speakers), ("max_speakers", args.max_speakers))
            if value is not None
        }

    if args.task == "diarization":
        if args.diarization_source == "chat" and not args.mode:
            parser.error("--mode is required when --task diarization")
        if args.dataset == "all":
            args.dataset = "ami"

    if args.output is None:
        args.output = str(default_output_path(args.task, args.dataset, args.mode, args.diarization_source))
    elif args.output.lower() in ("none", "no", "off", ""):
        args.output = None

    print(f"Endpoint: {args.api_url}")
    print(f"Model:    {args.model}")
    print(f"Task:     {args.task}")
    if args.task == "diarization":
        print(f"Source:   {args.diarization_source}")
        print(f"Diar API: {args.diarization_api_url}")
        if args.diarization_source == "chat":
            print(f"Mode:     {args.mode}")
    print(f"Output:   {args.output or 'none'}")
    print()

    aggregate: list[dict[str, Any]] = []
    if args.task == "asr":
        datasets = ["thchs30", "librispeech", "librispeech-other", "jsut"] if args.dataset == "all" else [args.dataset]
        for dataset in datasets:
            try:
                summary = run_asr_dataset(dataset, args)
            except FileNotFoundError as e:
                print(f"Skipping {dataset}: {e}\n")
                continue
            if summary:
                aggregate.append(summary)
    else:
        aggregate.append(run_diarization_dataset(args))

    if args.output:
        out = {
            "endpoint": args.api_url,
            "model": args.model,
            "task": args.task,
            "datasets": aggregate,
        }
        if args.task == "diarization":
            out["source"] = args.diarization_source
            if args.diarization_source == "chat":
                out["mode"] = args.mode
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON: {out_path}")


if __name__ == "__main__":
    main()
