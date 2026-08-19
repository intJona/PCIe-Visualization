#!/usr/bin/env python3
"""
dashboard.py - Streamlit + Plotly viewer for bench_run.py's ./logs/ output.

Scans ./logs/<topic>/ for bench-<N>p_<I>i_<O>o_run<R>.json (+ paired raw-metrics/engine-metrics
CSVs and .meta.json sidecar). Each distinct (topic, prompts, input_len, output_len) combination
is its own selectable "configuration" -- prompts/input/output are part of the filename and the
configuration label, not a separate cascading filter -- and each configuration gets its own
independent run-number picker. Renders a 3-row shared-x-axis figure: PCIe bandwidth,
hardware/engine utilization, cumulative tokens generated.

Visual encoding (strict):
  Configuration -> base color family (hue), e.g. config 0 = Blues, config 1 = Oranges, ...
  Run index     -> lightness within that family (first run = darkest)
  Metric type   -> line dash style, fixed per row regardless of GPU/run (row 1: Total PCIe/Tx
                    = solid, Rx = dashed; row 2: SM = solid, Mem = dashed, KV = dotted) -- this
                    is the thing that actually needs to stay distinguishable when more than one
                    metric is shown on the same row at once (e.g. "Tx + Rx (both)").
  GPU index     -> marker symbol (circle/square/diamond/...), plotted sparsely on every line

Color already carries configuration+run identity, so GPU used to double up onto dash style too
-- leaving metric type (Tx vs Rx, SM vs Mem vs KV) distinguishable only by a subtle line-width
difference, which was unreadable once more than one metric was shown for the same run. GPU now
has its own channel (marker shape) instead, freeing dash style to mean "which metric" alone.

The legend is synthesized into two compact groups (real data traces are showlegend=False, so
the legend never grows with the number of GPUs/metrics): one swatch per configuration+run
(color/shade key) and one per GPU index (marker-symbol key). Prefill/decode phase shading is only
drawn when exactly one run is displayed, to avoid overlapping shaded regions from stacking.

Sidebar order: GPU selection comes first (before Configurations) so it's the first thing you
see, then Configurations, then a per-configuration run picker, then display options grouped
under the plot row each one affects (Row 1 PCIe mode, Row 2 utilization toggles) so it's always
clear what's currently being shown.

Usage:
  streamlit run dashboard.py
"""

import json
import re
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

LOGS_ROOT = Path("./logs")
FILENAME_RE = re.compile(
    r"^bench-(?P<prompts>\d+)p_(?P<input>\d+)i_(?P<output>\d+)o_run(?P<run>\d+)\.json$"
)
COLOR_FAMILIES = ["Blues", "Oranges", "Greens", "Reds", "Purples", "Greys"]
# GPU identity is conveyed by marker shape (color/dash are already spoken for by run and metric).
GPU_MARKERS = ["circle", "square", "diamond", "triangle-up", "triangle-down", "star", "cross", "x"]
MARKER_EVERY = 20


def discover_runs():
    """Scan LOGS_ROOT for every topic subfolder's bench-<N>p_<I>i_<O>o_run<R>.json files, and
    group them into "configurations" -- one per distinct (topic, prompts, input_len, output_len)
    combination, each holding its own list of runs. That full parameter set (not just the topic
    folder) is what actually distinguishes one displayable trace-group from another.
    Returns {config_key: {"topic", "prompts", "input", "output", "label", "runs": [...]}}."""
    configs = {}
    if not LOGS_ROOT.exists():
        return configs
    for topic_dir in sorted(p for p in LOGS_ROOT.iterdir() if p.is_dir()):
        for jpath in sorted(topic_dir.glob("bench-*.json")):
            m = FILENAME_RE.match(jpath.name)
            if not m:
                continue
            prompts = int(m.group("prompts"))
            inp, out, run = int(m.group("input")), int(m.group("output")), int(m.group("run"))
            run_id = f"{prompts}p_{inp}i_{out}o_run{run}"
            csv_path = topic_dir / f"raw-metrics-{run_id}.csv"
            config_key = f"{topic_dir.name}__{prompts}p_{inp}i_{out}o"
            cfg = configs.setdefault(config_key, {
                "topic": topic_dir.name, "prompts": prompts, "input": inp, "output": out,
                "label": f"{topic_dir.name} | {prompts}p, {inp}i/{out}o", "runs": [],
            })
            meta_path = csv_path.with_suffix(".meta.json")
            num_gpus = 0
            if meta_path.exists():
                try:
                    num_gpus = json.loads(meta_path.read_text()).get("num_gpus", 0)
                except (json.JSONDecodeError, OSError):
                    num_gpus = 0
            cfg["runs"].append({
                "run": run, "json": jpath, "csv": csv_path,
                "engine": topic_dir / f"engine-metrics-{run_id}.csv",
                "meta": meta_path, "num_gpus": num_gpus,
            })
    for cfg in configs.values():
        cfg["runs"].sort(key=lambda e: e["run"])
    return configs


def read_csv(path):
    """Plain CSV -> {col: [float, ...]}. None if the file is missing/empty."""
    if not path.exists():
        return None
    with open(path) as f:
        lines = [ln.strip().split(",") for ln in f if ln.strip()]
    if len(lines) < 2:
        return None
    headers, rows = lines[0], [[float(x) for x in r] for r in lines[1:]]
    return {headers[i]: [row[i] for row in rows] for i in range(len(headers))}


def load_run(entry):
    """Load + time-align one run: shifts CSV/engine timestamps and reconstructs the full
    token-generation timeline so the first vLLM request lands at x=0 (per meta.json's
    t0_monotonic, the CLOCK_MONOTONIC reading at the CSV's own t=0 -- the same clock vLLM's own
    'start_times' use). TTFT is the time-to-first-token of whichever request started first.
    Returns None if the run's JSON lacks usable request data."""
    with open(entry["json"]) as f:
        jdata = json.load(f)
    start_times, ttfts, itls = jdata.get("start_times", []), jdata.get("ttfts", []), jdata.get("itls", [])
    if not start_times or not ttfts or not itls:
        return None

    meta = json.loads(entry["meta"].read_text()) if entry["meta"].exists() else {}
    t0_monotonic = meta.get("t0_monotonic")
    first_idx = min(range(len(start_times)), key=lambda i: start_times[i])
    first_request_abs = start_times[first_idx]
    origin_shift = (first_request_abs - t0_monotonic) if t0_monotonic is not None else 0.0
    ttft = ttfts[first_idx]

    events = []
    for i in range(len(itls)):
        t = (start_times[i] - first_request_abs) + ttfts[i]
        events.append(t)
        for itl in itls[i]:
            t += itl
            events.append(t)
    events.sort()

    csv_cols = read_csv(entry["csv"])
    if csv_cols is not None:
        csv_cols["timestamp"] = [t - origin_shift for t in csv_cols["timestamp"]]
        csv_cols = clip_negative_time(csv_cols)

    engine_cols = read_csv(entry["engine"])
    preempt_events = []
    if engine_cols is not None:
        engine_cols["timestamp"] = [t - origin_shift for t in engine_cols["timestamp"]]
        engine_cols = clip_negative_time(engine_cols)
        ts, pt = engine_cols["timestamp"], engine_cols["num_preemptions_total"]
        preempt_events = [ts[i] for i in range(1, len(pt)) if pt[i] > pt[i - 1]]

    return {
        "events": events, "token_counts": list(range(1, len(events) + 1)), "ttft": ttft,
        "csv_cols": csv_cols, "engine_cols": engine_cols, "preempt_events": preempt_events,
        "n_gpus": meta.get("num_gpus", 0),
    }


def clip_negative_time(cols):
    """Drop every sample recorded before this run's t=0 -- telemetry polling starts slightly
    before the first vLLM request is actually sent, so a few leading rows are otherwise
    negative once shifted to align with the first request. Keeps every column in sync."""
    ts = cols["timestamp"]
    start = next((i for i, t in enumerate(ts) if t >= 0), len(ts))
    return {k: v[start:] for k, v in cols.items()}


def sparse_marker_sizes(n, every=MARKER_EVERY, size=6):
    return [size if i % every == 0 else 0 for i in range(n)]


def main():
    st.set_page_config(layout="wide", page_title="vLLM PCIe Benchmark Dashboard")
    st.title("vLLM PCIe / GPU / Token Benchmark Dashboard")

    configs = discover_runs()
    if not configs:
        st.warning(f"No runs found under {LOGS_ROOT}/. Run bench_run.py first.")
        return

    # GPU selection comes first, before Configurations -- picking e.g. just GPU 1 is otherwise
    # an easy way to end up with an empty-looking PCIe/Utilization row without noticing why.
    max_gpus_available = max((r["num_gpus"] for cfg in configs.values() for r in cfg["runs"]), default=0)
    st.sidebar.header("GPUs")
    selected_gpus = set(st.sidebar.multiselect(
        "Which GPU(s) to show in the PCIe/Utilization rows", list(range(max_gpus_available)),
        default=list(range(max_gpus_available)), format_func=lambda i: f"GPU {i}",
    ))
    if max_gpus_available and not selected_gpus:
        st.sidebar.caption("No GPUs selected -- PCIe/Utilization rows will show only "
                            "KV cache/tokens (nothing per-GPU) until you pick one.")

    st.sidebar.header("Configurations")
    all_keys = sorted(configs, key=lambda k: configs[k]["label"])
    selected_keys = st.sidebar.multiselect(
        "Topic | prompts, input/output tokens", all_keys, default=all_keys[:1],
        format_func=lambda k: configs[k]["label"],
    )
    if not selected_keys:
        st.info("Select at least one configuration.")
        return

    # Each configuration gets its own independent run-number picker -- prompts/input/output no
    # longer need a separate cascading filter since they're already part of the configuration.
    st.sidebar.header("Runs (per configuration)")
    selected_runs = {}
    for key in selected_keys:
        cfg = configs[key]
        run_nums = [r["run"] for r in cfg["runs"]]
        chosen = st.sidebar.multiselect(cfg["label"], run_nums, default=run_nums, key=f"runs_{key}")
        if chosen:
            selected_runs[key] = set(chosen)
    if not selected_runs:
        st.info("Select at least one run for at least one configuration.")
        return

    # Display options are grouped under the plot row they affect (matching each row's own
    # title) instead of one flat "Display" list, so it's unambiguous what a given toggle
    # controls and what's currently shown at a glance.
    st.sidebar.header("Row 1: PCIe Data Transfer")
    pcie_mode = st.sidebar.radio(
        "Show", ["Total PCIe", "Tx only (D2H)", "Rx only (H2D)", "Tx + Rx (both)"], index=0,
    )
    st.sidebar.caption("Line style: solid = Total/Tx, dashed = Rx. Marker shape = GPU.")

    st.sidebar.header("Row 2: Hardware / Engine Utilization")
    show_sm = st.sidebar.checkbox("SM Compute %", value=True)
    show_mem = st.sidebar.checkbox("Memory Controller %", value=False)
    show_kv = st.sidebar.checkbox("KV Cache %", value=True)
    st.sidebar.caption("Line style: solid = SM, dashed = Mem, dotted = KV. Marker shape = GPU.")

    st.sidebar.header("Row 3: Cumulative Tokens Generated")
    st.sidebar.caption("Always shown (token timeline + first-token marker).")

    plot_entries = [
        (key, r) for key in selected_keys if key in selected_runs
        for r in configs[key]["runs"] if r["run"] in selected_runs[key]
    ]
    if not plot_entries:
        st.info("No runs match that selection.")
        return

    # Run index -> lightness rank, computed per-configuration across only the selected runs.
    runs_by_config = {}
    for key, r in plot_entries:
        runs_by_config.setdefault(key, set()).add(r["run"])
    run_rank = {(k, r): rank for k, runs in runs_by_config.items() for rank, r in enumerate(sorted(runs))}
    config_family = {k: COLOR_FAMILIES[i % len(COLOR_FAMILIES)] for i, k in enumerate(selected_keys)}

    loaded = []
    with st.spinner("Loading runs..."):
        for key, entry in plot_entries:
            data = load_run(entry)
            if data is None:
                continue
            n = len(runs_by_config[key])
            rank = run_rank[(key, entry["run"])]
            # Widened lightness span (0.80 -> 0.10) for stronger run-to-run contrast within a
            # configuration than the previous 0.85 -> 0.30 range.
            shade = 0.72 if n <= 1 else 0.80 - rank * (0.70 / (n - 1))
            color = sample_colorscale(config_family[key], [shade])[0]
            label = f"{configs[key]['label']} run{entry['run']}"
            loaded.append((entry, data, color, label))

    if not loaded:
        st.warning("None of the selected runs have usable data (missing start_times/ttfts/itls).")
        return
    single_run = len(loaded) == 1

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.4, 0.3, 0.3],
        subplot_titles=("PCIe Data Transfer (MiB/s)", "Hardware / Engine Utilization (%)",
                         "Cumulative Tokens Generated"),
    )

    for entry, data, color, label in loaded:
        csv_cols, n_gpus = data["csv_cols"], data["n_gpus"]
        if csv_cols is not None and n_gpus:
            ts = csv_cols["timestamp"]
            for gpu_i in range(n_gpus):
                if gpu_i not in selected_gpus:
                    continue
                marker = GPU_MARKERS[gpu_i % len(GPU_MARKERS)]
                gpu_label = f"{label} GPU{gpu_i}"

                if pcie_mode == "Total PCIe":
                    total = csv_cols[f"total_pcie{gpu_i}"]
                    fig.add_trace(go.Scatter(
                        x=ts, y=total, mode="lines+markers",
                        line=dict(color=color, width=2, dash="solid"),
                        marker=dict(symbol=marker, size=sparse_marker_sizes(len(total)), color=color),
                        name=f"{gpu_label} Total PCIe", legendgroup=gpu_label, showlegend=False,
                    ), row=1, col=1)

                if pcie_mode in ("Tx only (D2H)", "Tx + Rx (both)"):
                    d2h = csv_cols[f"d2h_mb{gpu_i}"]
                    fig.add_trace(go.Scatter(  # Tx (D2H): solid dash
                        x=ts, y=d2h, mode="lines+markers", line=dict(color=color, width=2.2, dash="solid"),
                        marker=dict(symbol=marker, size=sparse_marker_sizes(len(d2h)), color=color),
                        name=f"{gpu_label} Tx (D2H)", legendgroup=gpu_label, showlegend=False,
                    ), row=1, col=1)
                if pcie_mode in ("Rx only (H2D)", "Tx + Rx (both)"):
                    h2d = csv_cols[f"h2d_mb{gpu_i}"]
                    fig.add_trace(go.Scatter(  # Rx (H2D): dashed, so it stays distinct from Tx even in the same run color
                        x=ts, y=h2d, mode="lines+markers", line=dict(color=color, width=1.6, dash="dash"),
                        marker=dict(symbol=marker, size=sparse_marker_sizes(len(h2d)), color=color),
                        name=f"{gpu_label} Rx (H2D)", legendgroup=gpu_label, showlegend=False,
                    ), row=1, col=1)

                if show_sm:
                    sm = csv_cols[f"sm_util{gpu_i}"]
                    fig.add_trace(go.Scatter(
                        x=ts, y=sm, mode="lines+markers", line=dict(color=color, width=2.2, dash="solid"),
                        marker=dict(symbol=marker, size=sparse_marker_sizes(len(sm)), color=color),
                        name=f"{gpu_label} SM Compute %", legendgroup=gpu_label, showlegend=False,
                    ), row=2, col=1)
                if show_mem:
                    mem = csv_cols[f"mem_util{gpu_i}"]
                    fig.add_trace(go.Scatter(  # Mem: dashed, so it stays distinct from SM even in the same run color
                        x=ts, y=mem, mode="lines+markers", line=dict(color=color, width=1.6, dash="dash"),
                        marker=dict(symbol=marker, size=sparse_marker_sizes(len(mem)), color=color),
                        name=f"{gpu_label} Mem Controller %", legendgroup=gpu_label, showlegend=False,
                    ), row=2, col=1)

        if data["engine_cols"] is not None:
            ec = data["engine_cols"]
            if show_kv:
                fig.add_trace(go.Scatter(  # KV cache is per-run (not per-GPU): dotted, distinct from SM/Mem
                    x=ec["timestamp"], y=[v * 100 for v in ec["kv_cache_usage_perc"]], mode="lines",
                    line=dict(color=color, width=2, dash="dot"),
                    name=f"{label} KV Cache %", legendgroup=label, showlegend=False,
                ), row=2, col=1)
            for pt in data["preempt_events"]:
                fig.add_vline(x=pt, line_dash="dash", line_color=color, opacity=0.6, row=2, col=1)

        fig.add_trace(go.Scatter(
            x=data["events"], y=data["token_counts"], mode="lines",
            line=dict(color=color, width=2), name=f"{label} Tokens", legendgroup=label, showlegend=False,
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=[data["ttft"]], y=[1], mode="markers",
            marker=dict(symbol="x", size=12, color=color, line=dict(width=2)),
            name=f"{label} First Token", legendgroup=label, showlegend=False,
        ), row=3, col=1)

    # Prefill/decode phase shading is only meaningful for a single displayed run -- with more
    # than one run active, overlapping vrects from different runs would stack into a visually
    # confusing muddle rather than showing either run's phases clearly.
    if single_run:
        entry, data, color, label = loaded[0]
        run_end = data["events"][-1] if data["events"] else data["ttft"]
        # Annotation text only on row 1 -- row="all" would otherwise repeat the same label on
        # every subplot row, which is redundant clutter since the x-axis is shared.
        fig.add_vrect(x0=0, x1=data["ttft"], fillcolor="#FFEB3B", opacity=0.28, line_width=0,
                      row=1, col=1, annotation_text="Prefill (TTFT)", annotation_position="top left")
        fig.add_vrect(x0=data["ttft"], x1=run_end, fillcolor="#66BB6A", opacity=0.22, line_width=0,
                      row=1, col=1, annotation_text="Decode", annotation_position="top left")
        fig.add_vrect(x0=0, x1=data["ttft"], fillcolor="#FFEB3B", opacity=0.28, line_width=0, row=2, col=1)
        fig.add_vrect(x0=data["ttft"], x1=run_end, fillcolor="#66BB6A", opacity=0.22, line_width=0, row=2, col=1)
        fig.add_vrect(x0=0, x1=data["ttft"], fillcolor="#FFEB3B", opacity=0.28, line_width=0, row=3, col=1)
        fig.add_vrect(x0=data["ttft"], x1=run_end, fillcolor="#66BB6A", opacity=0.22, line_width=0, row=3, col=1)

    # Synthesized legend: real data traces above are all showlegend=False (hover still shows
    # their individual name), so the legend itself never grows with GPU count/metric count.
    # Two groups instead: one swatch per configuration+run (color/shade key) and one per GPU
    # index actually in use (dash-style key).
    for entry, data, color, label in loaded:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", line=dict(color=color, width=6), name=label,
            legendgroup="colorkey", legendgrouptitle_text="Configuration / run (color)",
        ), row=1, col=1)
    max_gpus = max((data["n_gpus"] for _, data, _, _ in loaded), default=0)
    for gpu_i in range(max_gpus):
        if gpu_i not in selected_gpus:
            continue
        marker = GPU_MARKERS[gpu_i % len(GPU_MARKERS)]
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", marker=dict(symbol=marker, size=11, color="black"),
            name=f"GPU {gpu_i}", legendgroup="markerkey", legendgrouptitle_text="GPU (marker)",
        ), row=1, col=1)

    fig.update_yaxes(title_text="MiB/s", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative tokens", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    # Safety net on top of clip_negative_time()'s data trimming: autorange padding can still
    # nudge the displayed range slightly below 0 even when every sample is >= 0.
    fig.update_xaxes(rangemode="nonnegative")
    fig.update_layout(height=1000, legend=dict(font=dict(size=11), groupclick="togglegroup", tracegroupgap=18))

    st.plotly_chart(fig, use_container_width=True)

    try:
        png_bytes = fig.to_image(format="png", width=1920, height=1080, engine="kaleido")
        st.download_button("Download High-Res PNG (1920x1080)", data=png_bytes,
                            file_name="pcie_benchmark_dashboard.png", mime="image/png")
    except Exception as e:
        st.caption(f"PNG export unavailable ({e}). Install the 'kaleido' package to enable it.")


if __name__ == "__main__":
    main()
