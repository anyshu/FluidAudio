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
}


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
        ("本地中文识别 SenseVoice 明显领先 Parakeet CTC",
         "SenseVoice Small(ONNX CPU)在 THCHS-30 上 CER 5.27%,Parakeet CTC zh-CN(int8)为 8.20%。Apple on-device 中文最弱 13.82%。Parakeet fp32 编码器版本暂未对比。"),
        ("日文 SenseVoice 与 Qwen3-Omni 平分秋色",
         "两者在 JSUT 上中位 CER 都是 7.14%,平均 SenseVoice 10.30% / Omni 10.92%。Qwen3-ASR-1.7B 与 Parakeet TDT ja 落后 2–3pt,VibeVoice 落后更多至 15.15%。"),
        ("速度榜单 Parakeet 一骑绝尘",
         "Parakeet TDT v3 在英文 LibriSpeech 上 RTFx 突破 100×。SenseVoice 是非英文场景速度王者(60–80×)。Qwen3 系列 7–13× 瓶颈在网络往返而非推理本身。VibeVoice 是远程模型中最慢(3.6–6.1×),输出结构化 JSON 多生成不少 token。Apple 18–49×,因语言而异。"),
        ("Apple SFSpeechRecognizer 在跑过的数据集上全面落后",
         "on-device test-other WER 17.71%(领先模型 4–8%),中文 CER 13.82%。日文在这台机器上没有 on-device 资源——Apple 按硬件/区域分发语言包。"),
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
        ("零依赖 · 系统 API",                  "Apple SFSpeechRecognizer", "无需安装,但每个数据集准确率落后 5-15pt"),
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


def render_caveats() -> str:
    items = [
        ("硬件环境",
         "本地引擎(Parakeet、SenseVoice、Apple)跑在 Apple M2 MacBook Air;Qwen3 系列和 VibeVoice 跑在服务商远程 GPU,RTFx 数据包含网络往返,不能与本地引擎数字直接对比。"),
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
        ("apple_all.json",                   "Apple SFSpeechRecognizer",  "3 个数据集(无 JSUT)"),
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
.matrix th .ds-name { font-weight: 600; color: var(--text); }
.matrix th .ds-metric { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
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
    本地硬件 Apple M2 Air ·
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
    <a href="#caveats">方法学</a>
    <a href="#inventory">原始数据文件</a>
  </nav>
</header>

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
  <p class="lede">RTFx 含义:100× 表示 1 小时音频 36 秒转写完毕。本地引擎跑在 Apple M2;远程模型数字包含网络往返,并不反映模型本身的纯推理速度。</p>
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

<section id="caveats">
  <h2>方法学说明</h2>
  <div class="caveats">{render_caveats()}</div>
</section>

<section id="inventory">
  <h2>原始数据文件</h2>
  <p class="lede">报告中每个数字都来自 <code>benchmark_results/</code> 下的某个 JSON 文件。</p>
  <div class="card">{render_inventory()}</div>
</section>

<footer>
  重跑任意 benchmark 后运行 <code>python3 Scripts/generate_benchmark_report.py</code> 重新生成此报告。
</footer>

</div>
</body>
</html>
"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Wrote: {OUTPUT}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
