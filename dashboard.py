#!/usr/bin/env python3
"""
dashboard.py - Streamlit + Plotly viewer for bench_run.py's ./logs/ output.

Scans ./logs/<topic>/ for bench-<N>p_<I>i_<O>o_run<R>.json (+ paired raw-metrics/engine-metrics
CSVs and .meta.json sidecar). Each distinct (topic, prompts, input_len, output_len) combination
is its own selectable "configuration" -- prompts/input/output are part of the filename and the
configuration label, not a separate cascading filter -- and each configuration gets its own
independent run-number picker. Renders a 3-row shared-x-axis figure: PCIe bandwidth,
hardware/engine utilization, request lifecycle timeline (cumulative tokens + per-request TTFT),
plus a run-level summary table for server-measured prefill/queue time (see below for why that's
a table and not a fourth thing plotted on the timeline).

Row placement follows the signal's nature, not just its data source: rows 1-2 hold continuously
sampled instrument gauges (NVML PCIe/util, /metrics KV cache %) where every poll is meaningful,
plotted as connected lines. Row 3 holds discrete per-request lifecycle events -- token-generation
timestamps and per-request TTFT -- plotted as sparse markers rather than forcing a continuous-line
treatment onto data that isn't continuous. TTFT is drawn ON the cumulative-token line itself (y =
token count reached at that instant, via bisect against the token timeline) rather than on a
separate value axis, so each marker visibly sits where the curve actually was at that moment;
hover shows the request's own TTFT duration.

Visual encoding follows a single-mode/multi-mode STATE MACHINE, not a per-dimension fixed
scheme: single_mode is "exactly one run selected in total, across every selected configuration",
multi_mode is "more than one". This replaces two earlier approaches -- a dynamic, cardinality-
based channel reassignment (kept changing which channel a dimension rendered through as the
selection changed, making sidebar swatches unreliable) and a fully-fixed GPU=hue/Config=shade/
Run=marker scheme (fine with one run selected, but gave GPU sole ownership of color even when
comparing many topics/runs, which is rarely what you want to compare at that point). There is
no marker-shape channel at all anymore (rows 1-2 render as plain lines) -- color and shade alone
carry the encoding, with shade_color()/shade_factor() spreading each configuration's own runs
across a fixed darkest-lightest range (same recipe as compare_topics.py) so contrast stays
maximal regardless of how many runs happen to be selected.
  Single_mode (rows 1-2, per-GPU traces): GPU index -> hue. Configuration -> lightness within
    that hue (Run can't vary here -- single_mode means exactly one run is selected, period).
    Prefill/decode phase shading is drawn (only one run on screen, so no overlap).
  Multi_mode (rows 1-2): Configuration -> hue. Run -> lightness within that hue (same color a
    trace would get in row 3). GPU has no channel of its own here -- the GPU dimension control
    (see below) restricts multi_mode to one GPU or an aggregate sum by default, and even with
    "All GPUs" active, per-GPU lines are only distinguishable via hover, not color. No phase
    shading (would overlap across runs).
  Row 3 (per-run only: tokens, TTFT -- no GPU dimension, same in both modes) -- Configuration ->
    hue, Run -> lightness within it. TTFT keeps its own "x" marker (see below), unrelated to run
    identity -- it's the one deliberate marker left in the whole figure.
Metric type itself (Total PCIe/Tx vs Rx, SM vs Mem vs KV) always maps to line dash style,
independent of mode, since it identifies *what signal* a trace is, not *which instance*.

The GPU dimension has its own segmented control -- "GPU N" / "All GPUs" / "Total (Sum)" -- above
the PCIe/Utilization rows. "Total (Sum)" collapses every selected GPU into one line per run
(PCIe bytes summed, utilization % averaged) using that run's Configuration/Run color, since
there's no per-GPU distinction left to encode. "GPU N" reveals a secondary picker: a multiselect
in single_mode (pick several specific GPUs at once) but only a single-select dropdown in
multi_mode (color there is already spoken for by Configuration/Run, so a second freely-varying
per-GPU channel has nowhere to go without colliding). The PCIe metric picker mirrors this: "Total
/ TX / RX / TX+RX" in single_mode, but "TX+RX" (both at once) is removed in multi_mode for the
same reason -- Tx and Rx would render in the identical Configuration/Run color and become
visually indistinguishable from each other.

Server-measured prefill/queue time (vLLM's own request_prefill_time_seconds/request_queue_time_
seconds histograms) are deliberately NOT plotted on the timeline. They were originally shown as
sparse hexagon/pentagon markers, but vLLM only records a request's stats into these histograms
once the request *finishes* entirely -- so the value only becomes visible at completion, and a
marker's x-position was therefore always "when it was recorded" (completion time), not when
prefill/queue actually happened. For a single long-decode request that meant the prefill-time
marker landed near the very end of its run, nowhere near its actual start -- confusing on both
axes and not fixable by better labeling, since there genuinely is no single correct instant on
this timeline for a value that's only knowable in hindsight. They're reported instead as a
run-level mean (weighted by sample count, so it's the true mean over the whole run, not a mean-
of-per-window-means) in a plain summary table below the chart, where no x-position is implied.
Per-request TTFT (the x marker) doesn't have this problem -- it comes straight from vLLM bench's
own client-side start_times/ttfts, with no server polling involved, so both its value and its
timing are exact; it just can't split out how much of that time was queueing vs. prefill.

There is no separate Plotly legend (every trace is showlegend=False, and the figure legend is
disabled outright) -- the sidebar doubles as the legend, kept intentionally compact: a segmented
control ("Total|TX|RX|TX+RX" etc.) or pills widget stands in for what used to be a radio button
or checkbox group plus a paragraph of caption text explaining it, and only Configuration's own
multiselect keeps a per-item swatch (hue_for() on that configuration's global rank) since it's
the one dimension whose color role stays the same in every mode.

Sidebar order: Configurations, then a per-configuration run picker (this pair also feeds the
single_mode/multi_mode state machine), then the GPU dimension control, then Row 1 (PCIe) and
Row 2 (utilization) pickers, so each widget appears in the same top-to-bottom order it affects
the plot.

Usage:
  streamlit run dashboard.py
"""

import bisect
import json
import math
import re
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

LOGS_ROOT = Path("./logs")
FILENAME_RE = re.compile(
    r"^bench-(?P<prompts>\d+)p_(?P<input>\d+)i_(?P<output>\d+)o_run(?P<run>\d+)\.json$"
)
COLOR_FAMILIES = ["Blues", "Oranges", "Greens", "Reds", "Purples", "Greys"]
# Same base palette compare_topics.py uses (matplotlib's "tab10" order) -- one vivid, fully
# saturated hex per family, shaded per-run by shade_color() below rather than sampled from a
# sequential colorscale (whose usable range compresses well before either end, which was why
# runs looked too similar to tell apart).
BASE_HEX = {"Blues": "#1f77b4", "Oranges": "#ff7f0e", "Greens": "#2ca02c",
            "Reds": "#d62728", "Purples": "#9467bd", "Greys": "#7f7f7f"}


def hue_for(rank):
    """Generic rank -> color-family mapping. Used for GPU (rows 1-2) and Configuration (row 3),
    the two dimensions that own the hue channel in the fixed scheme."""
    return COLOR_FAMILIES[rank % len(COLOR_FAMILIES)]


def shade_color(base_hex, factor):
    """Darken (factor<1) or lighten (factor>1) base_hex -- same recipe as compare_topics.py's
    shade_color(), reimplemented without a matplotlib dependency. factor=1.0 returns base_hex
    unchanged."""
    r, g, b = (int(base_hex[i:i + 2], 16) / 255 for i in (1, 3, 5))
    if factor <= 1.0:
        r, g, b = r * factor, g * factor, b * factor
    else:
        extra = factor - 1.0
        r, g, b = r + (1 - r) * extra, g + (1 - g) * extra, b + (1 - b) * extra
    r, g, b = (min(max(c, 0.0), 1.0) for c in (r, g, b))
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def shade_factor(i, n):
    """0-based rank i out of n items -> shade_color() factor, spread across the SAME fixed
    0.55 (darkest)-1.15 (lightest) range compare_topics.py uses regardless of n -- normalizing
    by (n - 1) rather than a fixed step table means two runs always land on the two endpoints
    of that range (maximum contrast) instead of two adjacent slots of some larger fixed cycle."""
    return 1.0 if n <= 1 else 0.55 + (1.15 - 0.55) * (i / (n - 1))


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
    'start_times' use). "ttft" is the time of the first token emitted anywhere in the run --
    i.e. min(events), NOT necessarily the TTFT of whichever request happened to start first.
    With concurrent requests (num_prompts > 1) those two can differ: a request that started
    slightly later can still finish prefill first, so anchoring to the earliest-starting
    request's own TTFT could place the "first token" marker after tokens the cumulative count
    line already shows, which is inconsistent by construction.
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

    events = []
    ttft_points = []
    for i in range(len(itls)):
        t = (start_times[i] - first_request_abs) + ttfts[i]
        events.append(t)
        # Exact per-request (time, TTFT) pair -- both values come straight from vLLM bench's own
        # client-side measurement, with no server polling/histogram involved, so unlike the
        # engine-metrics prefill/queue markers below there's no ambiguity about *when* this
        # happened: it's the instant this specific request's first token arrived.
        ttft_points.append((t, ttfts[i]))
        for itl in itls[i]:
            t += itl
            events.append(t)
    events.sort()
    ttft_points.sort()
    # Marker position ON the token curve itself (y = cumulative token count at that instant),
    # rather than a separate value axis -- bisect_right finds where this exact instant falls in
    # the sorted events list, which is the token count reached at that point. The TTFT duration
    # itself is kept alongside only for hover text, not for where the marker sits.
    ttft_markers = [(t, bisect.bisect_right(events, t), ttft_val) for t, ttft_val in ttft_points]
    # The first entry appended per request (before its itl loop) is that request's own first
    # token; the earliest one of those across every request is the true first token of the run.
    ttft = events[0]

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
        "ttft_markers": ttft_markers,
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


def add_pcie_util_traces(fig, ts, total, d2h, h2d, sm, mem, color, label,
                          pcie_mode, show_sm, show_mem):
    """Add one GPU-or-aggregate line's worth of Row 1 (PCIe)/Row 2 (utilization) traces.
    Shared by the per-GPU loop and the "Total (Sum)" aggregate path so both stay in sync.
    Plain lines, no markers -- color/shade/dash alone carry the encoding (see shade_color())."""
    if pcie_mode == "Total":
        fig.add_trace(go.Scatter(
            x=ts, y=total, mode="lines", line=dict(color=color, width=3, dash="solid"),
            name=f"{label} Total PCIe", showlegend=False,
        ), row=1, col=1)
    if pcie_mode in ("TX", "TX+RX"):
        fig.add_trace(go.Scatter(  # Tx (D2H): solid, so Total-only vs Tx-shown stays visually distinct
            x=ts, y=d2h, mode="lines", line=dict(color=color, width=3.2, dash="solid"),
            name=f"{label} Tx (D2H)", showlegend=False,
        ), row=1, col=1)
    if pcie_mode in ("RX", "TX+RX"):
        fig.add_trace(go.Scatter(  # Rx (H2D): dashed, so it stays distinct from Tx even in the same color
            x=ts, y=h2d, mode="lines", line=dict(color=color, width=2.4, dash="dash"),
            name=f"{label} Rx (H2D)", showlegend=False,
        ), row=1, col=1)
    if show_sm:
        fig.add_trace(go.Scatter(
            x=ts, y=sm, mode="lines", line=dict(color=color, width=3.2, dash="solid"),
            name=f"{label} SM Compute %", showlegend=False,
        ), row=2, col=1)
    if show_mem:
        fig.add_trace(go.Scatter(  # Mem: dashed, so it stays distinct from SM even in the same color
            x=ts, y=mem, mode="lines", line=dict(color=color, width=2.4, dash="dash"),
            name=f"{label} Mem Controller %", showlegend=False,
        ), row=2, col=1)


def main():
    st.set_page_config(layout="wide", page_title="vLLM PCIe Benchmark Dashboard")
    st.title("vLLM PCIe / GPU / Token Benchmark Dashboard")

    configs = discover_runs()
    if not configs:
        st.warning(f"No runs found under {LOGS_ROOT}/. Run bench_run.py first.")
        return

    st.sidebar.header("Configurations")
    all_keys = sorted(configs, key=lambda k: configs[k]["label"])
    selected_keys = st.sidebar.multiselect(
        "Topic | prompts, input/output tokens", all_keys, default=all_keys[:1],
    )
    if not selected_keys:
        st.info("Select at least one configuration.")
        return

    # Dynamic selection-based rank: first selected config -> color 0, second -> color 1, etc.
    # This ensures maximum contrast (always using the full palette's distinct hues) regardless
    # of which configs are selected or their global alphabetical order.
    selection_rank = {k: i for i, k in enumerate(selected_keys)}

    # Each configuration gets its own independent run-number picker -- prompts/input/output no
    # longer need a separate cascading filter since they're already part of the configuration.
    st.sidebar.header("Runs (per configuration)")
    selected_runs = {}
    for key in selected_keys:
        cfg = configs[key]
        run_nums = [r["run"] for r in cfg["runs"]]
        chosen = st.sidebar.multiselect(
            cfg["label"], run_nums, default=run_nums, key=f"runs_{key}",
        )
        if chosen:
            selected_runs[key] = set(chosen)
            # Streamlit's multiselect can't recolor individual pills, so mirror the plot's exact
            # per-run shade_color() as a small legend right under the picker instead -- recomputed
            # every rerun so it always reflects the CURRENT selection (shade depends on n_runs).
            run_ranks = sorted(chosen)
            base_hex = BASE_HEX[hue_for(selection_rank[key])]
            swatches = "&nbsp;&nbsp;".join(
                f'<span style="color:{shade_color(base_hex, shade_factor(run_ranks.index(r), len(run_ranks)))}">\u25cf</span> run{r}'
                for r in run_ranks
            )
            st.sidebar.markdown(
                f"<div style='margin-top:-10px;font-size:0.85em;'>{swatches}</div>",
                unsafe_allow_html=True,
            )
    if not selected_runs:
        st.info("Select at least one run for at least one configuration.")
        return

    # State machine: single_mode vs multi_mode, decided purely from how many runs are selected
    # (not how many end up loading successfully) -- this is what governs which GPU/PCIe controls
    # are available below and how color/shade/marker are assigned on the plot.
    total_selected_runs = sum(len(v) for v in selected_runs.values())
    single_mode = total_selected_runs == 1

    max_gpus_available = max((r["num_gpus"] for cfg in configs.values() for r in cfg["runs"]), default=0)
    st.sidebar.header("GPUs")
    gpu_view = st.sidebar.segmented_control(
        "View", ["GPU N", "All GPUs", "Total (Sum)"], default="All GPUs", required=True,
        help="GPU N picks specific GPU(s). Total (Sum) sums PCIe / averages utilization across "
             "every GPU into a single line.",
    )
    if gpu_view == "GPU N" and max_gpus_available:
        if single_mode:
            gpu_list = st.sidebar.multiselect(
                "Which GPU(s)", list(range(max_gpus_available)), default=[0],
                format_func=lambda i: f"GPU {i}",
            )
        else:
            # Multi-run view only allows one GPU at a time here -- color is already spoken for
            # by Configuration/Run below, so a second per-GPU channel would have nowhere to go.
            gpu_list = [st.sidebar.selectbox(
                "Which GPU", list(range(max_gpus_available)), format_func=lambda i: f"GPU {i}",
            )]
    else:
        gpu_list = list(range(max_gpus_available))
    gpu_aggregate = gpu_view == "Total (Sum)"
    selected_gpus = set(gpu_list)
    if max_gpus_available and not selected_gpus:
        st.sidebar.caption("No GPU selected -- PCIe/Utilization rows will show only KV cache/tokens.")

    st.sidebar.header("Row 1: PCIe Data Transfer")
    # TX+RX (both at once) is only offered in single-run view -- in multi-run view Run already
    # owns the shade channel, so a same-color Tx/Rx pair would be indistinguishable from a
    # same-color Total/Tx pair; restricting to one metric at a time avoids that ambiguity.
    pcie_options = ["Total", "TX", "RX", "TX+RX"] if single_mode else ["Total", "TX", "RX"]
    pcie_mode = st.sidebar.segmented_control(
        "Show", pcie_options, default="Total", required=True,
        key=f"pcie_mode_{'single' if single_mode else 'multi'}",
        help="TX = device-to-host, RX = host-to-device.",
    )

    st.sidebar.header("Row 2: Hardware / Engine Utilization")
    util_selected = st.sidebar.pills(
        "Show", ["SM", "Mem", "KV"], selection_mode="multi", default=["SM", "KV"],
        help="SM/Mem are per-GPU; KV cache is per-run.",
    )
    show_sm, show_mem, show_kv = "SM" in util_selected, "Mem" in util_selected, "KV" in util_selected

    st.sidebar.header("Row 3: Request Lifecycle")
    st.sidebar.caption("Always on: token timeline + per-request TTFT (from vLLM bench's own "
                        "client data). Server-measured prefill/queue time is in the summary "
                        "table below the chart instead -- vLLM only records it at request "
                        "completion, so it has no honest instant on this timeline.")

    if single_mode:
        st.sidebar.caption("Single-run view -- color = GPU, shade = Configuration. "
                            "Prefill/decode shading shown.")
    else:
        st.sidebar.caption("Multi-run view -- color = Configuration, shade = Run. "
                            "No phase shading (avoids overlap).")

    plot_entries = [
        (key, r) for key in selected_keys if key in selected_runs
        for r in configs[key]["runs"] if r["run"] in selected_runs[key]
    ]
    if not plot_entries:
        st.info("No runs match that selection.")
        return

    # Configuration owns hue, Run owns shade within it -- matches the Row 3 caption above
    # exactly. Shade is relative to THIS configuration's own selected run count (shade_factor),
    # not a fixed global table, so e.g. two selected runs always land on the darkest/lightest
    # endpoints regardless of how many runs exist overall -- same approach compare_topics.py
    # uses for its per-topic run shading.
    loaded = []
    with st.spinner("Loading runs..."):
        for key, entry in plot_entries:
            data = load_run(entry)
            if data is None:
                continue
            run_ranks = sorted(selected_runs[key])
            factor = shade_factor(run_ranks.index(entry["run"]), len(run_ranks))
            color = shade_color(BASE_HEX[hue_for(selection_rank[key])], factor)
            label = f"{configs[key]['label']} run{entry['run']}"
            loaded.append((key, entry, data, color, label))

    if not loaded:
        st.warning("None of the selected runs have usable data (missing start_times/ttfts/itls).")
        return
    single_run = len(loaded) == 1

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.4, 0.3, 0.3],
        subplot_titles=("PCIe Data Transfer (MiB/s)", "Hardware / Engine Utilization (%)",
                         "Request Lifecycle Timeline"),
    )
    summary_rows = []

    for key, entry, data, color, label in loaded:
        csv_cols, n_gpus = data["csv_cols"], data["n_gpus"]
        if csv_cols is not None and n_gpus:
            ts = csv_cols["timestamp"]
            gpus_here = [g for g in range(n_gpus) if g in selected_gpus]
            if gpu_aggregate and gpus_here:
                # No GPU dimension left once summed -- reuse the same Configuration=hue/Run=
                # shade color Row 3 uses below, since there's nothing else left to encode.
                n = len(gpus_here)
                total = [sum(csv_cols[f"total_pcie{g}"][i] for g in gpus_here) for i in range(len(ts))]
                d2h = [sum(csv_cols[f"d2h_mb{g}"][i] for g in gpus_here) for i in range(len(ts))]
                h2d = [sum(csv_cols[f"h2d_mb{g}"][i] for g in gpus_here) for i in range(len(ts))]
                sm = [sum(csv_cols[f"sm_util{g}"][i] for g in gpus_here) / n for i in range(len(ts))]
                mem = [sum(csv_cols[f"mem_util{g}"][i] for g in gpus_here) / n for i in range(len(ts))]
                add_pcie_util_traces(fig, ts, total, d2h, h2d, sm, mem, color,
                                      f"{label} \u03a3GPUs", pcie_mode, show_sm, show_mem)
            else:
                for gpu_i in gpus_here:
                    # Single-run view: GPU owns hue outright (Configuration/Run can't vary --
                    # single_mode means exactly one is selected, period -- so no shading needed).
                    # Multi-run view: GPU has no channel of its own (Configuration=hue, Run=
                    # shade already claim both; distinguishable via hover if "All GPUs" is on).
                    gpu_color = BASE_HEX[hue_for(gpu_i)] if single_mode else color
                    gpu_label = f"{label} GPU{gpu_i}"
                    add_pcie_util_traces(
                        fig, ts, csv_cols[f"total_pcie{gpu_i}"], csv_cols[f"d2h_mb{gpu_i}"],
                        csv_cols[f"h2d_mb{gpu_i}"], csv_cols[f"sm_util{gpu_i}"],
                        csv_cols[f"mem_util{gpu_i}"], gpu_color, gpu_label,
                        pcie_mode, show_sm, show_mem,
                    )

        if data["engine_cols"] is not None:
            ec = data["engine_cols"]
            if show_kv:
                fig.add_trace(go.Scatter(  # KV cache is per-run (not per-GPU): dotted, distinct from SM/Mem
                    x=ec["timestamp"], y=[v * 100 for v in ec["kv_cache_usage_perc"]], mode="lines",
                    line=dict(color=color, width=3, dash="dot"),
                    name=f"{label} KV Cache %", showlegend=False,
                ), row=2, col=1)
            for pt in data["preempt_events"]:
                fig.add_vline(x=pt, line_dash="dash", line_color=color, opacity=0.6, row=2, col=1)

            # Prefill/queue time used to be plotted as sparse markers on this timeline, but their
            # x-position is always "when vLLM recorded the sample" (request completion), not when
            # prefill/queue actually happened -- there's no honest instant on this timeline for
            # them, so they're reported as a run-level mean in the summary table below instead.
            # Weighted by prefill_samples_count so this is the true mean over every completed
            # request in the run, not a mean-of-per-window-means.
            counts = ec.get("prefill_samples_count")
            if counts and "avg_prefill_time_seconds" in ec and "avg_queue_time_seconds" in ec:
                total_n = sum(n for n in counts if not math.isnan(n))
                if total_n > 0:
                    prefill_mean = sum(v * n for v, n in zip(ec["avg_prefill_time_seconds"], counts)
                                        if not math.isnan(v) and not math.isnan(n)) / total_n
                    queue_mean = sum(v * n for v, n in zip(ec["avg_queue_time_seconds"], counts)
                                      if not math.isnan(v) and not math.isnan(n)) / total_n
                    summary_rows.append({
                        "Run": label, "Requests (server)": int(total_n),
                        "Mean prefill (s)": round(prefill_mean, 4),
                        "Mean queue (s)": round(queue_mean, 4),
                    })

        fig.add_trace(go.Scatter(
            x=data["events"], y=data["token_counts"], mode="lines",
            line=dict(color=color, width=3), name=f"{label} Tokens", showlegend=False,
        ), row=3, col=1)
        if data["ttft_markers"]:
            xs_ttft, ys_ttft, vals_ttft = zip(*data["ttft_markers"])
            fig.add_trace(go.Scatter(
                x=list(xs_ttft), y=list(ys_ttft), mode="markers",
                marker=dict(symbol="x", size=10, color=color, line=dict(width=1.5)),
                text=[f"{v:.3f}s" for v in vals_ttft], hovertemplate="TTFT %{text}<extra></extra>",
                name=f"{label} TTFT (per-request)", showlegend=False,
            ), row=3, col=1)

    # Prefill/decode phase shading is only meaningful for a single displayed run -- with more
    # than one run active, overlapping vrects from different runs would stack into a visually
    # confusing muddle rather than showing either run's phases clearly.
    if single_run:
        _, entry, data, color, label = loaded[0]
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

    fig.update_yaxes(title_text="MiB/s", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative tokens", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    # Safety net on top of clip_negative_time()'s data trimming: autorange padding can still
    # nudge the displayed range slightly below 0 even when every sample is >= 0.
    fig.update_xaxes(rangemode="nonnegative")
    # No Plotly legend -- the sidebar is the legend now (see module docstring), and every trace
    # above is already showlegend=False.
    fig.update_layout(height=1400, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    if summary_rows:
        st.caption("Server-measured prefill/queue time -- mean over every request completed "
                    "this run (vLLM's own histograms, weighted by sample count; not tied to "
                    "any single instant on the timeline above, see the Row 3 caption).")
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)

    try:
        png_bytes = fig.to_image(format="png", width=1920, height=1080, engine="kaleido")
        st.download_button("Download High-Res PNG (1920x1080)", data=png_bytes,
                            file_name="pcie_benchmark_dashboard.png", mime="image/png")
    except Exception as e:
        st.caption(f"PNG export unavailable ({e}). Install the 'kaleido' package to enable it.")


if __name__ == "__main__":
    main()
