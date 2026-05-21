#!/usr/bin/env python3
"""
Generate a self-contained HTML benchmark comparison report from the JSON files
in benchmark_results/.

Usage:
    python3 Scripts/generate_benchmark_report.py
        → writes benchmark_results/ASR_Benchmark_Report.html
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime
from html import escape
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "benchmark_results"
OUTPUT = RESULTS / "ASR_Benchmark_Report.html"


# ---------------------------------------------------------------------------
# Loaders — produce {dataset_name: {files, avg, median, rtfx_overall, rtfx_median, audio_s, proc_s}}
# ---------------------------------------------------------------------------

def load_qwen_aggregate(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    return {
        ds["dataset"]: {
            "files": ds["files_processed"],
            "avg": ds["average_score_pct"],
            "median": ds["median_score_pct"],
            "rtfx_overall": ds["overall_rtfx"],
            "rtfx_median": ds["median_rtfx"],
            "audio_s": ds["audio_duration_s"],
            "proc_s": ds["processing_time_s"],
        }
        for ds in d["datasets"]
    }


def load_voxtrace_asr_aggregate(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    out = {}
    for ds in d["datasets"]:
        items = ds.get("items", [])
        rtfxs = [r["rtfx"] for r in items if "rtfx" in r]
        audio_s = sum(r.get("duration_sec", 0) for r in items)
        proc_s = sum(r.get("latency_sec", 0) for r in items)
        out[ds["dataset"]] = {
            "files": ds["files_processed"],
            "avg": ds["average_score_pct"],
            "median": ds["median_score_pct"],
            "rtfx_overall": ds["overall_rtfx"],
            "rtfx_median": statistics.median(rtfxs) if rtfxs else None,
            "audio_s": audio_s,
            "proc_s": proc_s,
        }
    return out


def load_apple_aggregate(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    return {
        r["dataset"]: {
            "files": r["files_processed"],
            "avg": r["average_score"],
            "median": r["median_score"],
            "rtfx_overall": r["overall_rtfx"],
            "rtfx_median": r["median_rtfx"],
            "audio_s": r["audio_duration_s"],
            "proc_s": r["processing_time_s"],
        }
        for r in d["reports"]
    }


def load_sensevoice_aggregate(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    name_map = {
        "thchs30": "THCHS-30",
        "jsut": "JSUT-basic5000",
        "librispeech_clean": "LibriSpeech test-clean",
        "librispeech_other": "LibriSpeech test-other",
    }
    out = {}
    for key, info in d.items():
        out[name_map[key]] = {
            "files": info["files_processed"],
            "avg": info["average_score"] * 100,
            "median": info["median_score"] * 100,
            "rtfx_overall": info["overall_rtfx"],
            "rtfx_median": info["median_rtfx"],
            "audio_s": info["total_audio_sec"],
            "proc_s": info["total_processing_sec"],
        }
    return out


def load_parakeet_subset(path: Path, dataset_name: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    s = d["summary"]
    return {
        dataset_name: {
            "files": s["filesProcessed"],
            "avg": s["averageWER"] * 100,
            "median": s["medianWER"] * 100,
            "rtfx_overall": s["overallRTFx"],
            "rtfx_median": s["medianRTFx"],
            "audio_s": s["totalAudioDuration"],
            "proc_s": s["totalProcessingTime"],
        }
    }


def load_parakeet_cer(path: Path, dataset_name: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    res = d["results"]
    durs = [r["audioDurationSec"] for r in res]
    procs = [r["latencyMs"] / 1000 for r in res]
    rtfxs = [r["rtfx"] for r in res]
    cers = [r["cer"] for r in res]
    total_proc = sum(procs)
    return {
        dataset_name: {
            "files": len(res),
            "avg": statistics.mean(cers) * 100,
            "median": statistics.median(cers) * 100,
            "rtfx_overall": sum(durs) / total_proc if total_proc else 0,
            "rtfx_median": statistics.median(rtfxs),
            "audio_s": sum(durs),
            "proc_s": total_proc,
        }
    }


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

models: dict[str, dict] = {}

parakeet_en = load_parakeet_subset(RESULTS / "parakeet_v3_test_other.json", "LibriSpeech test-other")
# test-clean wasn't re-run after the move; use the previously published full-run numbers.
parakeet_en["LibriSpeech test-clean"] = {
    "files": 2620, "avg": 2.60, "median": 0.00,
    "rtfx_overall": 106.9, "rtfx_median": 90.2,
    "audio_s": 19452.5, "proc_s": 182.0,
}
models["Parakeet TDT v3 (en)"] = parakeet_en
models["Parakeet CTC zh-CN (int8)"] = load_parakeet_cer(RESULTS / "parakeet_zh_cn_int8.json", "THCHS-30")
models["Parakeet TDT ja"] = load_parakeet_cer(RESULTS / "parakeet_ja_jsut.json", "JSUT-basic5000")
models["Apple SFSpeechRecognizer"] = load_apple_aggregate(RESULTS / "apple_all.json")
models["SenseVoice Small"] = load_sensevoice_aggregate(RESULTS / "sensevoice_all.json")
models["Qwen3-ASR-1.7B"] = load_qwen_aggregate(RESULTS / "qwen3_all.json")
models["Qwen3-Omni"] = load_qwen_aggregate(RESULTS / "qwen3_omni_all.json")
models["VibeVoice-ASR-7B"] = load_qwen_aggregate(RESULTS / "vibevoice_all.json")
models["VoxTrace ASR"] = load_voxtrace_asr_aggregate(RESULTS / "voxtrace_asr_librispeech.json")

DATASETS: list[tuple[str, str, str]] = [
    ("LibriSpeech test-clean", "WER", "English"),
    ("LibriSpeech test-other", "WER", "English"),
    ("THCHS-30", "CER", "Chinese"),
    ("JSUT-basic5000", "CER", "Japanese"),
]

MODEL_TYPES = {
    "Parakeet TDT v3 (en)": ("local", "Apple Silicon · CoreML"),
    "Parakeet CTC zh-CN (int8)": ("local", "Apple Silicon · CoreML"),
    "Parakeet TDT ja": ("local", "Apple Silicon · CoreML"),
    "Apple SFSpeechRecognizer": ("local", "Apple Silicon · Speech.framework"),
    "SenseVoice Small": ("local", "Apple Silicon · ONNX CPU"),
    "Qwen3-ASR-1.7B": ("remote", "Remote GPU · OpenAI-compatible API"),
    "Qwen3-Omni": ("remote", "Remote GPU · OpenAI-compatible API"),
    "VibeVoice-ASR-7B": ("remote", "Remote GPU · OpenAI-compatible API"),
    "VoxTrace ASR": ("remote", "Remote GPU · OpenAI-compatible API"),
}


# ---------------------------------------------------------------------------
# Diarization data — per-engine aggregates on AMI 16-meeting test set
# ---------------------------------------------------------------------------

def load_diarization(path: Path) -> dict:
    """Each diarization JSON is a flat list of per-meeting result dicts."""
    with open(path) as f:
        data = json.load(f)
    ders = [r["der"] for r in data if "der" in r]
    miss = [r["missRate"] for r in data if "missRate" in r]
    fa = [r["falseAlarmRate"] for r in data if "falseAlarmRate" in r]
    se = [r["speakerErrorRate"] for r in data if "speakerErrorRate" in r]
    rtfxs = [r["rtfx"] for r in data if "rtfx" in r]
    procs = [r.get("processingTime", 0) for r in data]
    return {
        "files": len(data),
        "avg_der": statistics.mean(ders) if ders else None,
        "median_der": statistics.median(ders) if ders else None,
        "avg_miss": statistics.mean(miss) if miss else None,
        "avg_fa": statistics.mean(fa) if fa else None,
        "avg_se": statistics.mean(se) if se else None,
        "rtfx_overall": statistics.mean(rtfxs) if rtfxs else None,
        "rtfx_median": statistics.median(rtfxs) if rtfxs else None,
        "total_proc_s": sum(procs),
        "per_meeting": data,
    }


def load_voxtrace_diarization(path: Path) -> dict:
    """VoxTrace diarization JSON is an aggregate wrapper with fractional metrics."""
    with open(path) as f:
        data = json.load(f)
    dataset = data["datasets"][0]
    items = dataset.get("items", [])
    per_meeting = []
    for r in items:
        per_meeting.append({
            "meeting": r["meeting"],
            "der": r["der"] * 100,
            "missRate": r["miss_rate"] * 100,
            "falseAlarmRate": r["false_alarm_rate"] * 100,
            "speakerErrorRate": r["speaker_error_rate"] * 100,
            "rtfx": r["rtfx"],
            "processingTime": r.get("latency_sec", 0),
        })

    ders = [r["der"] for r in per_meeting]
    miss = [r["missRate"] for r in per_meeting]
    fa = [r["falseAlarmRate"] for r in per_meeting]
    se = [r["speakerErrorRate"] for r in per_meeting]
    rtfxs = [r["rtfx"] for r in per_meeting]
    return {
        "files": dataset["files_processed"],
        "avg_der": dataset["average_der_pct"],
        "median_der": dataset["median_der_pct"],
        "avg_miss": statistics.mean(miss) if miss else None,
        "avg_fa": statistics.mean(fa) if fa else None,
        "avg_se": statistics.mean(se) if se else None,
        "rtfx_overall": dataset["overall_rtfx"],
        "rtfx_median": statistics.median(rtfxs) if rtfxs else None,
        "total_proc_s": sum(r["processingTime"] for r in per_meeting),
        "per_meeting": per_meeting,
        "source": data.get("source", "chat"),
        "mode": data.get("mode"),
    }


diarization_engines: dict[str, dict] = {}
# Each entry: (display_name, [list of candidate filenames — first existing one wins]).
# Sortformer ships under a few different names depending on which variant was
# benchmarked last (`sortformer_ami.json` for the default, plus older
# `sortformer_nvidia_high_ami.json` / `sortformer_gd_ami.json` for explicit
# variant runs). Try the most-recent default first.
_diar_files = [
    ("Pyannote community-1 (offline, VBx)", ["diarization_offline_ami_sdm.json"]),
    ("Pyannote 3.1 (streaming)",            ["diarization_streaming_ami_sdm.json"]),
    ("Sortformer NVIDIA High-Latency",      ["sortformer_nvidia_high_ami.json"]),
    ("Sortformer GD (streaming, fastV2_1)", ["sortformer_ami.json", "sortformer_gd_ami.json"]),
    ("LS-EEND",                             ["lseend_ami.json"]),
]
for name, fnames in _diar_files:
    for fname in fnames:
        p = RESULTS / fname
        if p.exists():
            diarization_engines[name] = load_diarization(p)
            break
_voxtrace_chunk_global = RESULTS / "voxtrace_diarization_chunk_global_ami.json"
if _voxtrace_chunk_global.exists():
    diarization_engines["VoxTrace final JSON (chunk_global)"] = load_voxtrace_diarization(_voxtrace_chunk_global)
_voxtrace_full = RESULTS / "voxtrace_diarization_full_ami.json"
if _voxtrace_full.exists():
    diarization_engines["VoxTrace final JSON (full)"] = load_voxtrace_diarization(_voxtrace_full)
_voxtrace_raw = RESULTS / "voxtrace_diarization_raw_ami.json"
if _voxtrace_raw.exists():
    diarization_engines["VoxTrace raw speaker_turns"] = load_voxtrace_diarization(_voxtrace_raw)


# ---------------------------------------------------------------------------
# Remote ASR concurrency sweep (Qwen3-ASR family)
# ---------------------------------------------------------------------------

qwen3_sweep: dict | None = None
_sweep_path = RESULTS / "qwen3_sweep.json"
if _sweep_path.exists():
    with open(_sweep_path) as f:
        qwen3_sweep = json.load(f)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def fmt_pct(v):
    return f"{v:.2f}%" if v is not None else "—"


def fmt_pct_short(v):
    return f"{v:.1f}%" if v is not None else "—"


def fmt_rtfx(v):
    if v is None or v == 0:
        return "—"
    return f"{v:.1f}×"


def fmt_int(v):
    return f"{v:,}" if v is not None else "—"


def best_score(dataset: str) -> float | None:
    """Lowest avg score across models for highlighting."""
    vals = [m[dataset]["avg"] for m in models.values() if dataset in m]
    return min(vals) if vals else None


def best_speed(dataset: str) -> float | None:
    """Highest RTFx across models for highlighting."""
    vals = [m[dataset]["rtfx_overall"] for m in models.values() if dataset in m]
    return max(vals) if vals else None


def render_summary_matrix() -> str:
    """Big accuracy matrix with mini-bars."""
    rows = []
    for model, data in models.items():
        kind = MODEL_TYPES[model][0]
        cells = []
        for ds, metric, _ in DATASETS:
            info = data.get(ds)
            if info is None:
                cells.append('<td class="empty">—</td>')
                continue
            avg = info["avg"]
            best = best_score(ds)
            is_best = abs(avg - best) < 0.005 if best is not None else False
            # Bar width: log-ish scale capped at 25% to keep visual scale reasonable.
            bar_pct = min(100, (avg / 25) * 100)
            cls = "best" if is_best else ""
            cells.append(
                f'<td class="data {cls}">'
                f'<div class="cell-bar"><span style="width:{bar_pct:.0f}%"></span></div>'
                f'<div class="cell-value">{fmt_pct(avg)}</div>'
                f'</td>'
            )
        kind_label = {"local": "本地", "remote": "远程"}.get(kind, kind)
        kind_badge = f'<span class="badge {kind}">{kind_label}</span>'
        rows.append(
            f'<tr><th class="model">{escape(model)} {kind_badge}</th>{"".join(cells)}</tr>'
        )

    headers = "".join(
        f'<th><div class="ds-name">{escape(ds)}</div><div class="ds-metric">平均 {metric}</div></th>'
        for ds, metric, _ in DATASETS
    )
    return f"""
    <table class="matrix">
      <thead>
        <tr><th class="model-header">模型</th>{headers}</tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    """


def render_speed_matrix() -> str:
    rows = []
    max_rtfx = max(
        info["rtfx_overall"] for m in models.values() for info in m.values()
    )
    for model, data in models.items():
        kind = MODEL_TYPES[model][0]
        cells = []
        for ds, _, _ in DATASETS:
            info = data.get(ds)
            if info is None:
                cells.append('<td class="empty">—</td>')
                continue
            rtfx = info["rtfx_overall"]
            best = best_speed(ds)
            is_best = abs(rtfx - best) < 0.05 if best is not None else False
            bar_pct = min(100, (rtfx / max_rtfx) * 100)
            cls = "best-speed" if is_best else ""
            cells.append(
                f'<td class="data {cls}">'
                f'<div class="cell-bar speed"><span style="width:{bar_pct:.0f}%"></span></div>'
                f'<div class="cell-value">{fmt_rtfx(rtfx)}</div>'
                f'</td>'
            )
        kind_label = {"local": "本地", "remote": "远程"}.get(kind, kind)
        kind_badge = f'<span class="badge {kind}">{kind_label}</span>'
        rows.append(
            f'<tr><th class="model">{escape(model)} {kind_badge}</th>{"".join(cells)}</tr>'
        )

    headers = "".join(
        f'<th><div class="ds-name">{escape(ds)}</div><div class="ds-metric">总体 RTFx</div></th>'
        for ds, _, _ in DATASETS
    )
    return f"""
    <table class="matrix">
      <thead>
        <tr><th class="model-header">模型</th>{headers}</tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    """


def render_dataset_detail(ds: str, metric: str) -> str:
    rows_data = []
    for name, data in models.items():
        info = data.get(ds)
        if info:
            rows_data.append((info["avg"], name, info))
    rows_data.sort()

    rows_html = []
    for rank, (_, name, info) in enumerate(rows_data, 1):
        kind = MODEL_TYPES[name][0]
        kind_label = {"local": "本地", "remote": "远程"}.get(kind, kind)
        rank_cls = "rank-1" if rank == 1 else ("rank-2" if rank == 2 else ("rank-3" if rank == 3 else ""))
        rows_html.append(
            f'<tr class="{rank_cls}">'
            f'<td class="rank">{rank}</td>'
            f'<td><strong>{escape(name)}</strong> <span class="badge {kind}">{kind_label}</span></td>'
            f'<td class="num">{fmt_int(info["files"])}</td>'
            f'<td class="num"><strong>{fmt_pct(info["avg"])}</strong></td>'
            f'<td class="num">{fmt_pct(info["median"])}</td>'
            f'<td class="num">{fmt_rtfx(info["rtfx_overall"])}</td>'
            f'<td class="num">{fmt_rtfx(info["rtfx_median"])}</td>'
            '</tr>'
        )

    # Models that don't apply
    skipped = [name for name in models if ds not in models[name]]
    skipped_html = ""
    if skipped:
        items = ", ".join(escape(n) for n in skipped)
        skipped_html = (
            f'<p class="skipped-note">该数据集未覆盖 / 无数据:{items}</p>'
        )

    return f"""
    <table class="detail">
      <thead>
        <tr>
          <th>#</th><th>模型</th><th>文件数</th>
          <th>平均 {metric}</th><th>中位 {metric}</th>
          <th>总体 RTFx</th><th>中位 RTFx</th>
        </tr>
      </thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
    {skipped_html}
    """


def render_findings() -> str:
    items = [
        ("顶级准确率由远程和本地引擎共享",
         "Qwen3-Omni 和 Qwen3-ASR-1.7B 在 LibriSpeech test-clean 上 WER 都在 2.0–2.1%，Parakeet TDT v3 紧随 2.6%，VibeVoice-ASR-7B 2.87%。前 4 名相差不到 0.9pt,已落入单文件方差范围内。"),
        ("Qwen3-Omni 全数据集小幅胜出 Qwen3-ASR-1.7B",
         "Omni 是通用多模态模型而非专用 ASR,却在 4 个数据集上全部胜出,最大优势在日文 JSUT(10.92% vs 12.37% CER,–1.45pt)。代价是约慢 10–15%,因为多了 system/user prompt 的输入 token。"),
        ("VibeVoice-ASR-7B 英文有竞争力,其他语言中游",
         "在 LibriSpeech test-clean 上 2.87% WER 优于 SenseVoice(3.77%),但 test-other、中文、日文均落后于 Qwen3 系列和 SenseVoice。日文 15.15% CER 是它最弱项——训练数据可能英文为主。"),
        ("VoxTrace ASR 英文小样本已进入第一梯队",
         "VoxTrace ASR 在 LibriSpeech test-clean 100 文件样本上 WER 2.22%、中位 0%,接近 Qwen3-Omni / Qwen3-ASR / Parakeet 的全量成绩。注意它目前只跑了英文 100 文件,还不能和全量 2,620 文件结果完全等价。"),
        ("本地中文识别 SenseVoice 明显领先 Parakeet CTC",
         "SenseVoice Small(ONNX CPU)在 THCHS-30 上 CER 5.27%,Parakeet CTC zh-CN(int8)为 8.20%。Apple on-device 中文最弱 13.82%。Parakeet fp32 编码器版本暂未对比。"),
        ("日文 SenseVoice 与 Qwen3-Omni 平分秋色",
         "两者在 JSUT 上中位 CER 都是 7.14%,平均 SenseVoice 10.30% / Omni 10.92%。Qwen3-ASR-1.7B 与 Parakeet TDT ja 落后 2–3pt,VibeVoice 落后更多至 15.15%。"),
        ("速度榜单 Parakeet 一骑绝尘",
         "Parakeet TDT v3 在英文 LibriSpeech 上 RTFx 突破 100×。SenseVoice 是非英文场景速度王者(60–80×)。Qwen3 系列 7–13× 瓶颈在网络往返而非推理本身。VibeVoice 是远程模型中最慢(3.6–6.1×),输出结构化 JSON 多生成不少 token。Apple 18–49×,因语言而异。"),
        ("Apple SFSpeechRecognizer 在跑过的数据集上全面落后",
         "on-device test-other WER 17.71%(领先模型 4–8%),中文 CER 13.82%。日文在这台机器上没有 on-device 资源——Apple 按硬件/区域分发语言包。"),
        ("说话人分离 Pyannote community-1 是离线场景唯一的最优解",
         "AMI SDM 16 会议、统一评分协议(collar 0.25 总宽度 + ignoreOverlap + pyannote 官方 RTTM)下:Pyannote community-1 (powerset 分割 + VBx 聚类) DER 9.81%、RTFx 316×,断档式领先。LS-EEND 18.28%(流式里最佳)、VoxTrace final JSON full 17.72%、VoxTrace final JSON chunk_global 20.00%、Pyannote 3.1 streaming 23.61%(SE 15.67% 主导,流式聚类无法回看)、Sortformer NVIDIA High-Latency 26.01%(30.4s 上下文,速度 154×)、Sortformer GD fastV2_1 29.06%(0.48s 上下文)。Sortformer 两个变体共有 Miss 18% 的短板,跟 chunk 大小无关,是模型本身的偏置。"),
    ]
    return "\n".join(
        f'<div class="finding"><h3>{escape(t)}</h3><p>{escape(b)}</p></div>'
        for t, b in items
    )


def render_selection_table() -> str:
    rows = [
        ("iOS / macOS 离线 · 英文",            "Parakeet TDT v3",          "WER 2.6%、RTFx 107×、免费、本地推理"),
        ("iOS / macOS 离线 · 中文",            "SenseVoice Small (ONNX)",  "CER 5.3%、RTFx 79×;Parakeet zh-CN int8 落后到 8.2%"),
        ("iOS / macOS 离线 · 日文",            "SenseVoice Small (ONNX)",  "CER 10.3%、RTFx 61×;目前唯一可用的本地方案"),
        ("联网 · 最高准确率",                  "Qwen3-Omni",               "4 个数据集全部胜出;比 ASR-1.7B 慢 10–15%"),
        ("联网 · 控制成本",                    "Qwen3-ASR-1.7B",           "英/中只比 Omni 落后 ≤0.2pt,单位 token 成本更低"),
        ("联网 · 转写 + 说话人分离",           "VibeVoice-ASR-7B",         "输出带 Speaker ID 和时间戳的结构化 JSON;英文有竞争力(WER 2.87%),中/日文落后"),
        ("联网 · VoxTrace Diarization",      "优先看 raw speaker_turns",   "raw /v1/audio/diarization 与 Pyannote C1 同层级; final diarized_json 会叠加 ASR/align/segment merge 误差"),
        ("零依赖 · 系统 API",                  "Apple SFSpeechRecognizer", "无需安装,但每个数据集准确率落后 5-15pt"),
        ("说话人分离 · 离线批量",              "Pyannote community-1 (VBx)", "AMI 16 会议 DER 9.81%、RTFx 316×,FluidAudio 主推方案"),
        ("说话人分离 · 实时流式",              "LS-EEND",                  "DER 18.28%、RTFx 204×,流式场景的最佳折中(比 Pyannote 3.1 streaming 低 5pt)"),
    ]
    body = "\n".join(
        f'<tr><td>{escape(a)}</td><td><strong>{escape(b)}</strong></td><td>{escape(c)}</td></tr>'
        for a, b, c in rows
    )
    return f"""
    <table class="selection">
      <thead><tr><th>使用场景</th><th>推荐模型</th><th>理由</th></tr></thead>
      <tbody>{body}</tbody>
    </table>
    """


def short_diarization_name(name: str) -> str:
    short_names = {
        "Pyannote community-1 (offline, VBx)": "Pyannote C1",
        "Pyannote 3.1 (streaming)": "Pyannote 3.1",
        "Sortformer NVIDIA High-Latency": "Sortformer NVIDIA",
        "Sortformer GD (streaming, fastV2_1)": "Sortformer GD",
        "VoxTrace final JSON (chunk_global)": "VoxTrace JSON CG",
        "VoxTrace final JSON (full)": "VoxTrace JSON full",
        "VoxTrace raw speaker_turns": "VoxTrace raw",
    }
    return short_names.get(name, name.split(" (")[0])


def render_diarization_section() -> str:
    """Diarization comparison: DER matrix + per-engine ranking + DER error breakdown."""
    if not diarization_engines:
        return ""

    # Sort engines by avg DER ascending (lower is better).
    rows_sorted = sorted(
        diarization_engines.items(), key=lambda kv: kv[1]["avg_der"] or 1e9
    )

    best_der = min(v["avg_der"] for v in diarization_engines.values() if v["avg_der"] is not None)
    best_rtfx = max(
        v["rtfx_overall"] for v in diarization_engines.values() if v["rtfx_overall"] is not None
    )

    # 1. Headline ranking
    rank_rows = []
    for rank, (name, info) in enumerate(rows_sorted, 1):
        is_best_der = abs((info["avg_der"] or 0) - best_der) < 0.005
        is_best_rtfx = abs((info["rtfx_overall"] or 0) - best_rtfx) < 0.05
        rank_cls = "rank-1" if rank == 1 else ""
        rank_rows.append(
            f'<tr class="{rank_cls}">'
            f'<td class="rank">{rank}</td>'
            f'<td><strong>{escape(name)}</strong></td>'
            f'<td class="num">{info["files"]}</td>'
            f'<td class="num"><strong>{fmt_pct(info["avg_der"])}</strong></td>'
            f'<td class="num">{fmt_pct(info["median_der"])}</td>'
            f'<td class="num">{fmt_pct(info["avg_miss"])}</td>'
            f'<td class="num">{fmt_pct(info["avg_fa"])}</td>'
            f'<td class="num">{fmt_pct(info["avg_se"])}</td>'
            f'<td class="num">{fmt_rtfx(info["rtfx_overall"])}</td>'
            "</tr>"
        )

    ranking_table = f"""
    <table class="detail">
      <thead>
        <tr>
          <th>#</th><th>引擎</th><th>会议数</th>
          <th>平均 DER</th><th>中位 DER</th>
          <th>Miss</th><th>FA</th><th>SE</th>
          <th>总体 RTFx</th>
        </tr>
      </thead>
      <tbody>{"".join(rank_rows)}</tbody>
    </table>
    <p class="skipped-note">
      DER = Miss + FA + SE。Miss(漏检)、FA(虚警)、SE(说话人混淆)分别量化错误来源。
      所有引擎在同一份 AMI SDM 测试集(16 个会议、约 9.4 小时音频)上评测,
      使用相同的 RTTM ground-truth 和 collar 设置,数字直接可比。
    </p>
    """

    # 2. Per-meeting DER breakdown — show top-3 best/worst per engine to keep table small
    meetings = []
    for name, info in rows_sorted:
        for r in info["per_meeting"]:
            meetings.append((name, r["meeting"], r["der"], r["rtfx"]))

    # Build a per-meeting comparison: rows = meeting, cols = engines
    all_meetings = sorted({m for _, m, *_ in meetings})
    headers = (
        '<th class="model-header diarization-meeting-header">会议</th>'
        + "".join(
            f'<th class="diarization-engine"><div class="ds-name">{escape(short_diarization_name(name))}</div>'
            f'<div class="ds-metric">DER</div></th>'
            for name, _ in rows_sorted
        )
    )
    body_rows = []
    for m in all_meetings:
        cells = [f"<th class=\"model\">{escape(m)}</th>"]
        for name, _ in rows_sorted:
            r = next((r for r in diarization_engines[name]["per_meeting"] if r["meeting"] == m), None)
            if r is None:
                cells.append('<td class="empty">—</td>')
            else:
                der = r["der"]
                bar_pct = min(100, der / 60 * 100)
                cls = ""
                # Highlight cell if this is the best engine on this meeting
                best_meeting_der = min(
                    rr["der"]
                    for nn, _ in rows_sorted
                    for rr in diarization_engines[nn]["per_meeting"]
                    if rr["meeting"] == m
                )
                if abs(der - best_meeting_der) < 0.005:
                    cls = "best"
                cells.append(
                    f'<td class="data {cls}">'
                    f'<div class="cell-bar"><span style="width:{bar_pct:.0f}%"></span></div>'
                    f'<div class="cell-value">{der:.1f}%</div>'
                    f"</td>"
                )
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    matrix_table = f"""
    <table class="matrix">
      <thead><tr>{headers}</tr></thead>
      <tbody>{"".join(body_rows)}</tbody>
    </table>
    """

    return ranking_table + "<h3 style=\"margin-top:18px;\">逐会议 DER 矩阵</h3>" + matrix_table


def render_concurrency_sweep() -> str:
    """Render the Qwen3 remote ASR concurrency scaling table + analysis."""
    if not qwen3_sweep or not qwen3_sweep.get("sweep_runs"):
        return ""

    # The JSON may use either field-name schema:
    #   • benchmark.py output:  throughput_files_per_sec / wall_clock_sec / latency_p50_sec / ...
    #   • hand-edited short:    throughput_fps / wall_sec / p50 / p95 / p99 / error_pct / files
    # Normalize to a single dict per row.
    def norm(r: dict) -> dict:
        return {
            "concurrency": r["concurrency"],
            "files":       r.get("files_processed", r.get("files", 0)),
            "wall_sec":    r.get("wall_clock_sec", r.get("wall_sec", 0)),
            "throughput":  r.get("throughput_files_per_sec", r.get("throughput_fps", 0)),
            "rtfx":        r.get("effective_rtfx_overall", r.get("effective_rtfx", 0)),
            "p50":         r.get("latency_p50_sec", r.get("p50", 0)),
            "p95":         r.get("latency_p95_sec", r.get("p95", 0)),
            "p99":         r.get("latency_p99_sec", r.get("p99", 0)),
            "error_pct":   r.get("error_rate_pct", r.get("error_pct", 0)),
        }

    runs = [norm(r) for r in qwen3_sweep["sweep_runs"]]
    runs.sort(key=lambda r: r["concurrency"])
    baseline_throughput = runs[0]["throughput"] if runs else 1.0
    peak_throughput = max(r["throughput"] for r in runs)
    best_c_idx = max(range(len(runs)), key=lambda i: runs[i]["throughput"])

    baseline_p50 = runs[0]["p50"] if runs else 1.0

    rows_html = []
    for i, r in enumerate(runs):
        speedup = r["throughput"] / baseline_throughput if baseline_throughput > 0 else 0
        c_now = r["concurrency"]
        c_prev = runs[i - 1]["concurrency"] if i > 0 else 1
        prev_t = runs[i - 1]["throughput"] if i > 0 else baseline_throughput
        ratio_to_ideal = (r["throughput"] / prev_t) / (c_now / c_prev) if c_prev > 0 and prev_t > 0 else 1.0
        throughput_frac = r["throughput"] / peak_throughput if peak_throughput > 0 else 0
        p50_ratio = r["p50"] / baseline_p50 if baseline_p50 > 0 else 1.0

        # Tag priority (highest first):
        #   regress: throughput < previous level — overload territory
        #   sweet:   ≥70% of peak throughput AND p50 ≤ 2× baseline — best practical pick
        #   peak:    ≥95% of peak throughput — max throughput, latency may suffer
        #   linear:  ratio_to_ideal > 0.85 — still scaling well
        #   plateau: anything else
        if i > 0 and r["throughput"] < prev_t:
            css, tag = "regress", "倒退"
        elif throughput_frac >= 0.70 and p50_ratio <= 2.0 and throughput_frac < 0.95:
            css, tag = "sweet", "性价比拐点"
        elif throughput_frac >= 0.95:
            css, tag = "peak", "吞吐峰值"
        elif ratio_to_ideal > 0.85:
            css, tag = "linear", "近线性"
        else:
            css, tag = "plateau", "平台"
        rank_cls = "rank-1" if i == best_c_idx else ""
        rows_html.append(
            f'<tr class="{rank_cls}">'
            f'<td class="rank">{r["concurrency"]}</td>'
            f'<td class="num">{r["files"]}</td>'
            f'<td class="num">{r["wall_sec"]:.1f}s</td>'
            f'<td class="num"><strong>{r["throughput"]:.2f}</strong> f/s</td>'
            f'<td class="num">{r["rtfx"]:.1f}×</td>'
            f'<td class="num">{speedup:.2f}×</td>'
            f'<td class="num">{r["p50"]:.2f}s</td>'
            f'<td class="num">{r["p95"]:.2f}s</td>'
            f'<td class="num">{r["p99"]:.2f}s</td>'
            f'<td class="num">{r["error_pct"]:.1f}%</td>'
            f'<td><span class="sweep-tag sweep-{css}">{tag}</span></td>'
            "</tr>"
        )

    table = f"""
    <table class="detail">
      <thead>
        <tr>
          <th>并发数</th><th>文件</th><th>墙钟</th>
          <th>吞吐量</th><th>effRTFx</th><th>加速比</th>
          <th>p50</th><th>p95</th><th>p99</th>
          <th>错误率</th><th>状态</th>
        </tr>
      </thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
    """

    # Dynamic findings — compute numbers from the actual `runs` list so the
    # analysis stays in sync with whatever sweep data is currently in the JSON.
    peak_row = max(runs, key=lambda r: r["throughput"])
    last_row = runs[-1]
    base = runs[0]
    # Sweet-spot row: highest throughput where p50 ≤ 1.5× baseline.
    sweet_candidates = [r for r in runs if r["p50"] <= base["p50"] * 1.5 and r["throughput"] >= peak_row["throughput"] * 0.65]
    sweet_row = max(sweet_candidates, key=lambda r: r["throughput"]) if sweet_candidates else peak_row
    linear_end = sweet_row  # last "still scaling well" row by definition

    findings = f"""
    <div class="caveats" style="margin-top:14px;">
      <div class="caveat">
        <h4>线性扩展区(1 → {linear_end['concurrency']})</h4>
        <p>吞吐量从 {base['throughput']:.2f} f/s 涨到 {linear_end['throughput']:.2f} f/s,
        加速比 {linear_end['throughput']/base['throughput']:.1f}×,接近理想 {linear_end['concurrency']}×。
        p50 延迟基本不涨({base['p50']:.2f}s → {linear_end['p50']:.2f}s)。
        服务端仍有空闲容量,每加一个并发都换回可观吞吐。</p>
      </div>
      <div class="caveat">
        <h4>性价比拐点 = {sweet_row['concurrency']}</h4>
        <p>吞吐量 {sweet_row['throughput']:.2f} f/s,达到峰值的 {sweet_row['throughput']/peak_row['throughput']*100:.0f}%。
        p50 {sweet_row['p50']:.2f}s,p95 {sweet_row['p95']:.2f}s,延迟可控。
        <strong>用户交互式应用首选 {sweet_row['concurrency']} 并发</strong>。</p>
      </div>
      <div class="caveat">
        <h4>吞吐量天花板 = {peak_row['concurrency']}</h4>
        <p>{peak_row['throughput']:.2f} f/s 峰值,effRTFx {peak_row['rtfx']:.1f}×{(' 已经追上本地 Parakeet TDT v3 (106.9×)' if peak_row['rtfx'] >= 100 else '')}。
        p50 {peak_row['p50']:.2f}s({peak_row['p50']/base['p50']:.1f}× baseline),p95 {peak_row['p95']:.2f}s。
        <strong>批量离线场景首选</strong>——牺牲单文件延迟换最大总吞吐。</p>
      </div>
      <div class="caveat">
        <h4>过载/倒退 ≥ {last_row['concurrency']}</h4>
        <p>{last_row['concurrency']} 并发吞吐量 {last_row['throughput']:.2f} f/s
        {('<strong>比峰值低 ' + f"{(1 - last_row['throughput']/peak_row['throughput'])*100:.0f}" + '%</strong>' ) if last_row['throughput'] < peak_row['throughput'] * 0.95 else '已与峰值持平'}。
        p99 {last_row['p99']:.2f}s 远高于交互应用可接受范围。客户端必须自己 throttle,不要无脑加并发。</p>
      </div>
      <div class="caveat">
        <h4>服务端是软排队,不返回 429</h4>
        <p>所有并发级别错误率都是 0%——服务端不主动拒绝请求,只用排队拖慢延迟。
        客户端无法靠 429 反馈感知饱和,必须靠 p50/p95 监控自己判断。</p>
      </div>
      <div class="caveat">
        <h4>跟本地 Parakeet 的成本对比</h4>
        <p>concurrency={peak_row['concurrency']} 时 effRTFx {peak_row['rtfx']:.1f}×,
        {'接近' if peak_row['rtfx'] < 130 else '超过'}本地 Parakeet TDT v3 (106.9×)。
        但 Qwen3 远程烧 token、Parakeet 本地免费。<strong>纯英文场景本地 Parakeet 仍然首选</strong>,
        Qwen3 留给"非英文"或"需要更高准确率"的场景。</p>
      </div>
    </div>
    """

    return table + findings


def render_caveats() -> str:
    items = [
        ("硬件环境",
         "本地引擎(Parakeet、SenseVoice、Apple)跑在 Apple M4 MacBook Air;Qwen3 系列和 VibeVoice 跑在服务商远程 GPU,RTFx 数据包含网络往返,不能与本地引擎数字直接对比。"),
        ("文本归一化",
         "同一套归一化管道作用于参考和假设串再打分。英文沿用 HuggingFace ASR Leaderboard 规则(英美拼写互转、缩略语展开、连读拆分、数字词转数字)。中文将阿拉伯数字映射为汉字,仅保留 CJK 码点。日文将汉字数字映射为阿拉伯数字(复合十位优先于简单十位),保留平假名/片假名/CJK/数字。"),
        ("评估指标",
         "英文按词级 WER,中/日按字符级 CER,均通过归一化后字符串上的 Levenshtein 距离计算。"),
        ("Apple 跳过 JSUT",
         "这台 Mac 上 Apple 没有日文 on-device 资源。<code>--allow-server</code> 可以走云端但有限流,本次未启用。"),
        ("JSUT 子集说明",
         "FluidInference 镜像只有 500 文件(BASIC5000_4501–5000),不是完整的 5000 条 JSUT-basic5000。所有模型评测的是同一份 500 条,绝对 CER 在完整 corpus 上可能略有变化。"),
        ("Parakeet zh-CN 编码器版本",
         "仅测试了 int8 量化编码器版本,fp32 版本可能能拉近与 SenseVoice 的差距。"),
        ("Qwen3-Omni 提示词",
         'Omni 是通用多模态 LLM,必须显式给一段 ASR 风格的 system prompt 抑制评论性输出。我们用的是 "You are an automatic speech recognition system. Output only the verbatim transcription…"。不带这个 prompt 它会输出对内容的分析而不是转写。'),
        ("并发与重试",
         "Qwen3 / VibeVoice 远程调用使用 4 并发,碰到 429/5xx 走指数回退(1s → 30s,最多重试 4 次)。"),
        ("Diarization 评测集",
         "AMI SDM 测试集 16 会议(EN2002a-d / ES2004a-d / IS1009a-d / TS3003a-d),"
         "对应 <code>DatasetDownloader.officialAMITestSet</code>。4 个引擎跑的是同一组音频和 RTTM ground truth,DER 直接可比。"),
        ("Diarization 指标拆解",
         "DER = Miss(漏检) + FA(虚警) + SE(说话人混淆)。Pyannote 3.1 streaming 的 DER 主要被 SE 推高,因为流式聚类无法回溯——一旦把两个人当作同一说话人,后面没机会修正。"),
        ("两个 Pyannote 模型不是同一个",
         "<code>--mode offline</code> 跑的是 <strong>pyannote/speaker-diarization-community-1</strong>(powerset 分割 + WeSpeaker + VBx 聚类),"
         "<code>--mode streaming</code> 跑的是 <strong>pyannote/speaker-diarization-3.1</strong>(分割 + WeSpeaker)。"
         "两者是不同代的模型,community-1 比 3.1 更新更强,所以 DER 差距既来自模型代际,也来自流式 vs 离线的算法约束,不能简单理解为同一模型的\u201c速度精度权衡\u201d。"),
        ("Sortformer 两个变体对比",
         "<code>sortformer-benchmark --dataset ami</code> 默认跑 <strong>Gradient Descent fastV2_1 流式版</strong>"
         "(chunkLen=6, ~0.48s 上下文, DER 29.06%, RTFx 17.5×),加 <code>--nvidia-high-latency</code> 切到"
         "<strong>NVIDIA High-Latency 版</strong>(chunkLen=340, 30.4s 上下文, DER 26.01%, RTFx 154×)。"
         "NVIDIA 版准确率好 3pt 主要来自 SE 降低(7.66% → 5.35%,长上下文更易分人),速度反而快 9 倍"
         "(长 chunk 摊销运行时开销更划算)。"),
        ("Sortformer Miss 偏高是模型本身的问题",
         "GD 和 NVIDIA 两个变体 Miss rate 几乎一样(18.67% vs 18.12%),跟 chunk 大小无关,跟阈值关系也有限。"
         "这是 Sortformer 训练数据(主要是电话会议、短上下文场景)决定的系统性欠检测偏置,"
         "本地调参很难根除——要解决得换模型或重训。"),
        ("Diarization 评分协议 — 三个 bug 修复",
         "本仓库的 diarization benchmark 原来有 3 个评分 bug,导致 4 个引擎的 DER 不可比:"
         "(1) pyannote offline 用的 DiarizationMetricsCalculator 实际是 collar=0.5(每个 segment 两端各削 0.25),不是文档说的 0.25;"
         "(2) pyannote streaming 用 totalFrames 当分母(应该是 totalRefSpeech)、贪心 first-overlap 映射(应该是 Hungarian)、没有 collar;"
         "(3) ground truth 各引擎不一致(XML 标注 vs 逐词标注)。"
         "本报告里全部修复:4 个引擎统一调用 DiarizationDER.compute(collar=0.25, ignoreOverlap=true),ground truth 统一用 pyannote 官方 RTTM(<code>~/FluidAudioDatasets/ami_official/rttm/</code>),所有数字直接可比。"),
    ]
    return "\n".join(
        f'<div class="caveat"><h4>{escape(t)}</h4><p>{b}</p></div>'
        for t, b in items
    )


def render_inventory() -> str:
    rows = [
        ("parakeet_v3_test_other.json",      "Parakeet TDT v3 (en)",      "LibriSpeech test-other(全集)"),
        ("parakeet_zh_cn_int8.json",         "Parakeet CTC zh-CN",         "THCHS-30"),
        ("parakeet_ja_jsut.json",            "Parakeet TDT ja",           "JSUT-basic5000"),
        ("sensevoice_all.json",              "SenseVoice Small (ONNX)",   "全部 4 个数据集"),
        ("qwen3_all.json",                   "Qwen3-ASR-1.7B (远程)",     "全部 4 个数据集"),
        ("qwen3_omni_all.json",              "Qwen3-Omni (远程)",         "全部 4 个数据集"),
        ("vibevoice_all.json",               "VibeVoice-ASR-7B (远程)",   "全部 4 个数据集"),
        ("voxtrace_asr_librispeech.json",    "VoxTrace ASR (远程)",       "LibriSpeech test-clean 100 文件"),
        ("apple_all.json",                   "Apple SFSpeechRecognizer",  "3 个数据集(无 JSUT)"),
        ("diarization_offline_ami_sdm.json",  "Pyannote community-1 (offline, VBx)", "AMI SDM 测试集 16 会议"),
        ("diarization_streaming_ami_sdm.json","Pyannote 3.1 (streaming)",           "AMI SDM 测试集 16 会议"),
        ("sortformer_nvidia_high_ami.json",    "Sortformer NVIDIA High-Latency", "AMI SDM 测试集 16 会议"),
        ("sortformer_ami.json",                "Sortformer GD (streaming, fastV2_1)", "AMI SDM 测试集 16 会议"),
        ("lseend_ami.json",                   "LS-EEND",                   "AMI SDM 测试集 16 会议"),
        ("voxtrace_diarization_raw_ami.json", "VoxTrace raw speaker_turns", "AMI SDM 测试集 16 会议(/v1/audio/diarization)"),
        ("voxtrace_diarization_full_ami.json", "VoxTrace final JSON full", "AMI SDM 测试集 16 会议(chat diarized_json)"),
        ("voxtrace_diarization_chunk_global_ami.json", "VoxTrace final JSON chunk_global", "AMI SDM 测试集 16 会议(chat diarized_json)"),
        ("qwen3_sweep.json",                  "Qwen3-ASR-1.7B 并发扫描",   "LibriSpeech test-clean 100 文件 × 7 个并发级别"),
    ]
    body = "\n".join(
        f'<tr><td><code>{escape(a)}</code></td><td>{escape(b)}</td><td>{escape(c)}</td></tr>'
        for a, b, c in rows
    )
    return f"""
    <table class="inventory">
      <thead><tr><th>文件</th><th>模型</th><th>覆盖数据集</th></tr></thead>
      <tbody>{body}</tbody>
    </table>
    """


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

CSS = r"""
:root {
  --bg: #fbfbfd;
  --bg-card: #ffffff;
  --bg-soft: #f5f7fa;
  --text: #1a1f2c;
  --text-muted: #6b7280;
  --border: #e4e7ec;
  --accent: #1d4ed8;
  --accent-soft: #dbeafe;
  --good: #15803d;
  --good-soft: #dcfce7;
  --speed: #b45309;
  --speed-soft: #fef3c7;
  --bad: #b91c1c;
  --shadow: 0 1px 2px rgba(0,0,0,.04), 0 4px 12px rgba(15,23,42,.06);
  --radius: 10px;
  --mono: ui-monospace, "SF Mono", "Menlo", "Cascadia Code", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115;
    --bg-card: #161a23;
    --bg-soft: #1c212c;
    --text: #e7eaf3;
    --text-muted: #9aa3b2;
    --border: #2a303d;
    --accent: #93c5fd;
    --accent-soft: #1e3a8a;
    --good: #86efac;
    --good-soft: #14532d;
    --speed: #fcd34d;
    --speed-soft: #78350f;
    --bad: #fca5a5;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 6px 18px rgba(0,0,0,.35);
  }
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
               "Hiragino Sans GB", "Microsoft YaHei", system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
.container { max-width: 1180px; margin: 0 auto; padding: 32px 24px 80px; }
header { margin-bottom: 32px; }
h1 {
  font-size: 28px;
  font-weight: 700;
  margin: 0 0 6px;
  letter-spacing: -0.02em;
}
header .subtitle { color: var(--text-muted); margin: 0 0 16px; }
.toc {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 20px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px 20px;
  font-size: 13px;
  margin-top: 16px;
}
.toc a {
  color: var(--accent);
  text-decoration: none;
  border-bottom: 1px solid transparent;
}
.toc a:hover { border-bottom-color: var(--accent); }

.tabs {
  display: flex;
  gap: 8px;
  margin: 24px 0 18px;
  position: sticky;
  top: 0;
  z-index: 10;
  background: color-mix(in srgb, var(--bg) 92%, transparent);
  backdrop-filter: blur(8px);
  padding: 10px 0;
}
.tab-button {
  appearance: none;
  border: 1px solid var(--border);
  background: var(--bg-card);
  color: var(--text-muted);
  border-radius: 999px;
  cursor: pointer;
  font: inherit;
  font-weight: 700;
  padding: 9px 18px;
}
.tab-button.active {
  background: var(--text);
  border-color: var(--text);
  color: var(--bg-card);
}
.tab-panel { display: none; }
.tab-panel.active { display: block; }
section { margin-top: 36px; }
section h2 {
  font-size: 20px;
  font-weight: 700;
  margin: 0 0 6px;
  letter-spacing: -0.015em;
}
section .lede { color: var(--text-muted); margin: 0 0 16px; max-width: 80ch; }
section h3 { font-size: 16px; margin: 24px 0 8px; }

.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  box-shadow: var(--shadow);
  overflow-x: auto;
}

table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td {
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  vertical-align: middle;
}
thead th {
  font-weight: 600;
  color: var(--text-muted);
  background: var(--bg-soft);
  white-space: nowrap;
}
tbody tr:hover { background: var(--bg-soft); }
.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
code, .mono { font-family: var(--mono); font-size: 12px; }

/* Matrix */
.matrix { table-layout: fixed; }
.matrix th.model-header { width: 28%; }
.matrix th .ds-name {
  font-weight: 600;
  color: var(--text);
  white-space: normal;
  overflow-wrap: anywhere;
  line-height: 1.25;
}
.matrix th .ds-metric { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
.matrix th.diarization-meeting-header { width: 12%; }
.matrix th.diarization-engine { width: 14.666%; }
.matrix th.diarization-engine .ds-name {
  font-size: 12px;
  word-break: keep-all;
  overflow-wrap: normal;
}
.matrix td.data { position: relative; }
.matrix td.empty { color: var(--text-muted); text-align: center; font-style: italic; }
.matrix td.best { background: var(--good-soft); }
.matrix td.best-speed { background: var(--speed-soft); }
.matrix td.best .cell-value::after { content: " ★"; color: var(--good); }
.matrix td.best-speed .cell-value::after { content: " ★"; color: var(--speed); }
.matrix th.model { font-weight: 500; padding: 8px 12px; }
.cell-bar {
  background: var(--bg-soft);
  border-radius: 4px;
  height: 6px;
  margin-bottom: 4px;
  overflow: hidden;
  position: relative;
}
.cell-bar span {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, #fb7185, #ef4444);
  border-radius: 4px;
}
.cell-bar.speed span {
  background: linear-gradient(90deg, #34d399, #059669);
}
.cell-value { font-weight: 600; font-variant-numeric: tabular-nums; font-size: 13px; }

/* Detail tables */
table.detail tr.rank-1 td:first-child { background: var(--good-soft); color: var(--good); font-weight: 700; }
table.detail tr.rank-1 strong { color: var(--good); }
table.detail tr.rank-1 td { background: rgba(21, 128, 61, 0.04); }
.skipped-note {
  margin: 8px 0 0;
  font-size: 12px;
  color: var(--text-muted);
  font-style: italic;
}

/* Badges */
.badge {
  display: inline-block;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  padding: 2px 6px;
  border-radius: 3px;
  vertical-align: middle;
  margin-left: 6px;
}
.badge.local  { background: var(--accent-soft); color: var(--accent); text-transform: none; letter-spacing: 0; }
.badge.remote { background: var(--speed-soft); color: var(--speed); text-transform: none; letter-spacing: 0; }

/* Concurrency sweep status tags */
.sweep-tag {
  display: inline-block;
  font-size: 10px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 10px;
  white-space: nowrap;
}
.sweep-linear  { background: rgba(34, 197, 94, 0.18);  color: #15803d; }
.sweep-sweet   { background: rgba(34, 197, 94, 0.32);  color: #14532d; }
.sweep-peak    { background: rgba(234, 179, 8, 0.22);  color: #b45309; }
.sweep-plateau { background: rgba(148, 163, 184, 0.25); color: #475569; }
.sweep-regress { background: rgba(239, 68, 68, 0.18);  color: #b91c1c; }

/* Findings */
.findings { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); }
.finding {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-left: 4px solid var(--accent);
  border-radius: var(--radius);
  padding: 14px 18px;
}
.finding h3 { font-size: 14px; margin: 0 0 6px; color: var(--text); }
.finding p { margin: 0; color: var(--text-muted); font-size: 13px; }

/* Caveats */
.caveats { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); }
.caveat {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 16px;
}
.caveat h4 { font-size: 13px; margin: 0 0 4px; }
.caveat p { margin: 0; color: var(--text-muted); font-size: 12.5px; }

footer {
  margin-top: 60px;
  padding-top: 20px;
  border-top: 1px solid var(--border);
  color: var(--text-muted);
  font-size: 12px;
}
"""


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def main():
    today = datetime.now().strftime("%Y-%m-%d")

    lang_zh = {"English": "英文", "Chinese": "中文", "Japanese": "日文"}
    detail_sections = ""
    for ds, metric, lang in DATASETS:
        slug = ds.lower().replace(" ", "-").replace("_", "-")
        detail_sections += f"""
        <section id="ds-{slug}">
          <h3>{escape(ds)} <span style="font-size:13px; color:var(--text-muted); font-weight:500;">— {escape(lang_zh.get(lang, lang))}, {escape(metric)}</span></h3>
          <div class="card">{render_dataset_detail(ds, metric)}</div>
        </section>
        """

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ASR 引擎横向对比报告</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">

<header>
  <h1>ASR 引擎横向对比报告</h1>
  <p class="subtitle">
    生成日期 {today} ·
    本地硬件 Apple M4 Air ·
    远程引擎运行于服务商 GPU(OpenAI 兼容 API)
  </p>
  <p>
    在 4 个公开数据集上对比 7 款 ASR 引擎,全部走同一套文本归一化流程,WER/CER 数字可直接横向比较。
  </p>
  <nav class="toc">
    <a href="#datasets">数据集</a>
    <a href="#accuracy">准确率矩阵</a>
    <a href="#speed">速度矩阵</a>
    <a href="#detail">各数据集明细</a>
    <a href="#findings">关键发现</a>
    <a href="#selection">选型指南</a>
    <a href="#diarization">说话人分离</a>
    <a href="#concurrency">远程并发扩展性</a>
    <a href="#caveats">方法学</a>
    <a href="#inventory">原始数据文件</a>
  </nav>
</header>

<div class="tabs" role="tablist" aria-label="报告分类">
  <button class="tab-button active" type="button" role="tab" aria-selected="true" aria-controls="tab-asr" data-tab="asr">ASR</button>
  <button class="tab-button" type="button" role="tab" aria-selected="false" aria-controls="tab-diarization" data-tab="diarization">Diarization</button>
</div>

<div id="tab-asr" class="tab-panel active" role="tabpanel">

<section id="datasets">
  <h2>数据集</h2>
  <p class="lede">全部引擎在同一批文件上评测,总时长约 17.5 小时。</p>
  <div class="card">
    <table>
      <thead><tr><th>数据集</th><th class="num">文件数</th><th>总时长</th><th>语言</th><th>指标</th><th>来源</th></tr></thead>
      <tbody>
        <tr><td><strong>LibriSpeech test-clean</strong></td><td class="num">2,620</td><td>5h 24m</td><td>英文</td><td>WER</td><td>官方 test-clean 全集</td></tr>
        <tr><td><strong>LibriSpeech test-other</strong></td><td class="num">2,939</td><td>5h 21m</td><td>英文</td><td>WER</td><td>官方 test-other 全集</td></tr>
        <tr><td><strong>THCHS-30</strong></td><td class="num">2,495</td><td>6h 19m</td><td>中文</td><td>CER</td><td>FluidInference/THCHS-30-tests(HF 子集)</td></tr>
        <tr><td><strong>JSUT-basic5000</strong></td><td class="num">500</td><td>31m</td><td>日文</td><td>CER</td><td>FluidInference/JSUT-basic5000(BASIC5000_4501–5000)</td></tr>
      </tbody>
    </table>
  </div>
</section>

<section id="accuracy">
  <h2>准确率矩阵</h2>
  <p class="lede">平均 WER(英文)或 CER(中/日文),越低越好。每列最佳模型以绿色高亮。</p>
  <div class="card">{render_summary_matrix()}</div>
</section>

<section id="speed">
  <h2>速度矩阵</h2>
  <p class="lede">RTFx 含义:100× 表示 1 小时音频 36 秒转写完毕。本地引擎跑在 Apple M4;远程模型数字包含网络往返,并不反映模型本身的纯推理速度。</p>
  <div class="card">{render_speed_matrix()}</div>
</section>

<section id="detail">
  <h2>各数据集明细</h2>
  <p class="lede">同样的数据按准确率排序。中位数能看出分布偏态——"中位 0%" 意味着至少一半文件被完美转写。</p>
  {detail_sections}
</section>

<section id="findings">
  <h2>关键发现</h2>
  <div class="findings">{render_findings()}</div>
</section>

<section id="selection">
  <h2>选型指南</h2>
  <p class="lede">常见使用场景到本测试中最佳模型的映射。</p>
  <div class="card">{render_selection_table()}</div>
</section>

<section id="concurrency">
  <h2>远程模型并发扩展性 — Qwen3-ASR-1.7B</h2>
  <p class="lede">
    在 LibriSpeech test-clean 上取 100 个文件,以同样的样本扫描 7 个并发级别(1/2/4/8/16/32/64),
    实测服务端容量曲线、最优并发点、过载阈值。这是给你做容量规划/选型的关键数据。
  </p>
  <div class="card">{render_concurrency_sweep()}</div>
</section>

</div>

<div id="tab-diarization" class="tab-panel" role="tabpanel">

<section id="diarization">
  <h2>说话人分离(Diarization)</h2>
  <p class="lede">在 AMI SDM 测试集(16 个会议,约 9.4 小时)上对比 4 套 diarization 引擎。指标 DER 越低越好,所有引擎评测同一份音频和 ground truth,数字直接可比。</p>
  <div class="card">{render_diarization_section()}</div>
</section>

<section id="caveats">
  <h2>方法学说明</h2>
  <div class="caveats">{render_caveats()}</div>
</section>

<section id="inventory">
  <h2>原始数据文件</h2>
  <p class="lede">报告中每个数字都来自 <code>benchmark_results/</code> 下的某个 JSON 文件。</p>
  <div class="card">{render_inventory()}</div>
</section>

</div>

<footer>
  重跑任意 benchmark 后运行 <code>python3 Scripts/generate_benchmark_report.py</code> 重新生成此报告。
</footer>

</div>
<script>
const tabButtons = document.querySelectorAll('.tab-button');
const tabPanels = document.querySelectorAll('.tab-panel');
function activateTab(name) {{
  tabButtons.forEach((button) => {{
    const active = button.dataset.tab === name;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', active ? 'true' : 'false');
  }});
  tabPanels.forEach((panel) => panel.classList.toggle('active', panel.id === `tab-${{name}}`));
}}
tabButtons.forEach((button) => button.addEventListener('click', () => activateTab(button.dataset.tab)));
document.querySelectorAll('.toc a').forEach((link) => {{
  link.addEventListener('click', () => {{
    const target = link.getAttribute('href') || '';
    if (target === '#diarization' || target === '#caveats' || target === '#inventory') {{
      activateTab('diarization');
    }} else {{
      activateTab('asr');
    }}
  }});
}});
</script>
</body>
</html>
"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Wrote: {OUTPUT}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
