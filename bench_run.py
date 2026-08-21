#!/usr/bin/env python3
"""
bench_run.py - Condensed vLLM benchmark runner with PCIe/GPU/engine telemetry.

Runs `vllm bench serve` while two background threads poll hardware (NVML: PCIe H2D/D2H
throughput, SM/memory-controller utilization) and software (vLLM /metrics) telemetry.
Scales to any number of GPUs (pynvml.nvmlDeviceGetCount()) -- no hardcoded GPU count.

Per run, writes into ./logs/<detail>/:
  bench-<N>p_<I>i_<O>o_run<R>.json          vLLM's own --save-result output
  raw-metrics-<N>p_<I>i_<O>o_run<R>.csv     timestamp, and per GPU: total_pcie/h2d_mb/d2h_mb/sm_util/mem_util
  engine-metrics-<N>p_<I>i_<O>o_run<R>.csv  timestamp, num_requests_running/waiting, kv_cache_usage_perc,
    num_preemptions_total, num_requests_completed (cumulative since this run started),
    avg_prefill_time_seconds, avg_queue_time_seconds, prefill_samples_count (number of requests
    whose prefill completed since the PREVIOUS sample -- a per-interval count, NOT the same as
    num_requests_completed; avg_* is EXACT for that one request when this is 1, else the mean
    over however many completed in that interval)
  raw-metrics-<N>p_<I>i_<O>o_run<R>.meta.json  t0_monotonic, input_len, output_len, num_gpus, etc.

Usage:
  python bench_run.py --detail "tp_x16" --prompts 1 20 50 --runs 2
  python bench_run.py --detail "tp_x4" --prompts 20 --input-len 4000 --output-len 512
"""

import argparse
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from threading import Event, Thread

import pynvml

ENGINE_METRIC_PATTERNS = {
    "num_requests_running": re.compile(r"^vllm:num_requests_running\{[^}]*\}\s+([0-9.eE+-]+)", re.MULTILINE),
    "num_requests_waiting": re.compile(r"^vllm:num_requests_waiting\{[^}]*\}\s+([0-9.eE+-]+)", re.MULTILINE),
    "kv_cache_usage_perc": re.compile(r"^vllm:kv_cache_usage_perc\{[^}]*\}\s+([0-9.eE+-]+)", re.MULTILINE),
    "num_preemptions_total": re.compile(r"^vllm:num_preemptions_total\{[^}]*\}\s+([0-9.eE+-]+)", re.MULTILINE),
    # Lifetime-cumulative counter shared by the whole server process (never resets per run) --
    # baselined to each run's first sample in poll_engine_metrics() so the CSV shows "completed
    # since this run started", not a huge near-flat count from every prior run against this server.
    "request_success_total": re.compile(r"^vllm:request_success_total\{[^}]*\}\s+([0-9.eE+-]+)", re.MULTILINE),
    # request_prefill_time_seconds / request_queue_time_seconds are Prometheus Histograms, not
    # gauges -- /metrics never exposes individual observations, only cumulative _sum/_count (plus
    # _bucket lines, which we don't need). Also lifetime-cumulative, so diffed against the
    # PREVIOUS sample (not the run-start baseline) in poll_engine_metrics() -- diffing against
    # the run-start baseline instead would make avg_prefill_time/avg_queue_time a slowly
    # converging running average over the whole run rather than a per-sample-window snapshot.
    # prefill_time = first_token_ts - scheduled_ts (excludes queue wait, unlike TTFT) -- the
    # accurate "how long did loading+processing the input prompt take" metric.
    "prefill_time_sum": re.compile(r"^vllm:request_prefill_time_seconds_sum\{[^}]*\}\s+([0-9.eE+-]+)", re.MULTILINE),
    "prefill_time_count": re.compile(r"^vllm:request_prefill_time_seconds_count\{[^}]*\}\s+([0-9.eE+-]+)", re.MULTILINE),
    "queue_time_sum": re.compile(r"^vllm:request_queue_time_seconds_sum\{[^}]*\}\s+([0-9.eE+-]+)", re.MULTILINE),
    "queue_time_count": re.compile(r"^vllm:request_queue_time_seconds_count\{[^}]*\}\s+([0-9.eE+-]+)", re.MULTILINE),
}

# Lifetime-cumulative metrics that must be baselined to each run's first sample (see comments above).
CUMULATIVE_METRICS = (
    "request_success_total", "prefill_time_sum", "prefill_time_count", "queue_time_sum", "queue_time_count",
)


class GpuProbe:
    """Per-GPU sampler for PCIe throughput + utilization.

    sm_util prefers NVML's GPM API (NVML_GPM_METRIC_SM_UTIL: true % of SMs busy, sampled as a
    delta between two GPM snapshots) when the device supports it (Hopper+/some Ampere) --
    nvmlDeviceGetUtilizationRates().gpu is a much coarser "any kernel ran this sample" bit that
    reports pinned-100% under vLLM's continuous batching. Falls back to it when GPM isn't
    available, so this always returns a value either way.
    """

    def __init__(self, handle):
        self.h = handle
        self.gpm_ok = False
        try:
            if pynvml.nvmlGpmQueryDeviceSupport(handle).isSupportedDevice:
                self.s_prev = pynvml.nvmlGpmSampleAlloc()
                self.s_cur = pynvml.nvmlGpmSampleAlloc()
                pynvml.nvmlGpmSampleGet(handle, self.s_prev)
                self.gpm_ok = True
        except pynvml.NVMLError:
            pass

    def read(self):
        """Return (h2d_mb, d2h_mb, sm_util_pct, mem_util_pct) for one sample."""
        h2d = pynvml.nvmlDeviceGetPcieThroughput(self.h, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024
        d2h = pynvml.nvmlDeviceGetPcieThroughput(self.h, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024
        util = pynvml.nvmlDeviceGetUtilizationRates(self.h)
        sm_util = util.gpu
        if self.gpm_ok:
            try:
                pynvml.nvmlGpmSampleGet(self.h, self.s_cur)
                m = pynvml.c_nvmlGpmMetricsGet_t()
                m.version, m.numMetrics = pynvml.NVML_GPM_METRICS_GET_VERSION, 1
                m.metrics[0].metricId = pynvml.NVML_GPM_METRIC_SM_UTIL
                m.sample1, m.sample2 = self.s_prev, self.s_cur
                pynvml.nvmlGpmMetricsGet(m)
                if m.metrics[0].nvmlReturn == pynvml.NVML_SUCCESS:
                    sm_util = m.metrics[0].value
                self.s_prev, self.s_cur = self.s_cur, self.s_prev
            except pynvml.NVMLError:
                pass
        return h2d, d2h, sm_util, util.memory


def csv_header(n_gpus):
    cols = ["timestamp"]
    for i in range(n_gpus):
        cols += [f"total_pcie{i}", f"h2d_mb{i}", f"d2h_mb{i}", f"sm_util{i}", f"mem_util{i}"]
    return ",".join(cols)


def gpu_monitor(rows, stop_event, start_time, anchor_holder, sample_interval=0.02):
    """Poll every GPU on a fixed interval until stop_event is set. anchor_holder already holds
    t0_monotonic (set by the caller at the exact same instant as start_time, before this thread
    even started -- NVML init / GPM probe setup below can take a non-trivial amount of time, so
    capturing t0_monotonic here instead would skew it away from the CSV's true t=0 by however
    long that setup took). This function only fills in num_gpus.
    sample_interval default of 0.02s (20ms) matches NVML's own internal averaging window for
    nvmlDeviceGetPcieThroughput -- polling faster just re-reads the same stale value."""
    try:
        pynvml.nvmlInit()
        n_gpus = pynvml.nvmlDeviceGetCount()
        probes = [GpuProbe(pynvml.nvmlDeviceGetHandleByIndex(i)) for i in range(n_gpus)]
    except pynvml.NVMLError as e:
        print(f"Error: NVML init failed ({e}); no GPU telemetry will be captured.")
        anchor_holder["num_gpus"] = 0
        return

    anchor_holder["num_gpus"] = n_gpus
    error_count, last_error = 0, None
    while not stop_event.is_set():
        loop_start = time.monotonic()
        try:
            row = [time.time() - start_time]
            for p in probes:
                h2d, d2h, sm, mem = p.read()
                row += [h2d + d2h, h2d, d2h, sm, mem]
            rows.append(row)
        except pynvml.NVMLError as e:
            error_count += 1
            last_error = e
            if error_count == 1:
                print(f"Warning: NVML sample failed ({e}); dropping affected samples.")
        # Subtract however long the reads themselves took -- without this, a slow NVML call
        # (e.g. nvmlDeviceGetPcieThroughput can take ~20ms/call with persistence mode off,
        # observed via direct benchmarking) silently stacks on TOP of the full sleep instead of
        # being absorbed by it, so the achieved interval drifts well past sample_interval
        # (measured ~105ms vs the intended 20ms on this workstation) instead of just being
        # floored by whatever the reads cost.
        elapsed = time.monotonic() - loop_start
        stop_event.wait(max(0.0, sample_interval - elapsed))
    pynvml.nvmlShutdown()
    if error_count:
        print(f"Warning: {error_count}/{error_count + len(rows)} GPU telemetry samples were "
              f"dropped this run due to NVML errors (last error: {last_error}).")


def poll_engine_metrics(rows, stop_event, base_url, start_time, sample_interval=0.5):
    """Poll vLLM's own /metrics endpoint for engine-side state. Disables itself (returns) the
    first time the endpoint is unreachable -- best-effort enrichment, never fatal."""
    url = base_url.rstrip("/") + "/metrics"
    baseline = None
    prev = None
    while not stop_event.is_set():
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"Warning: engine metrics endpoint unreachable ({url}: {e}); skipping engine metrics.")
            return

        values = {}
        for name, pattern in ENGINE_METRIC_PATTERNS.items():
            matches = [float(m) for m in pattern.findall(text)]
            if not matches:
                values[name] = float("nan")
            elif name == "kv_cache_usage_perc":
                values[name] = max(matches)
            else:
                values[name] = sum(matches)

        if baseline is None:
            baseline = {k: values[k] for k in CUMULATIVE_METRICS}
            prev = dict(baseline)
        completed_this_run = values["request_success_total"] - baseline["request_success_total"]
        # Windowed deltas (since the PREVIOUS sample, not run start) -- see the CUMULATIVE_METRICS
        # comment above for why run-start baselining would be wrong here.
        prefill_sum_delta = values["prefill_time_sum"] - prev["prefill_time_sum"]
        prefill_count_delta = values["prefill_time_count"] - prev["prefill_time_count"]
        queue_sum_delta = values["queue_time_sum"] - prev["queue_time_sum"]
        queue_count_delta = values["queue_time_count"] - prev["queue_time_count"]
        # avg_*_time is exact for a single request when its matching count delta is 1 (e.g. the
        # 1-prompt warm-up run, or the only request to complete between two samples); otherwise
        # it's the mean over however many requests completed since the previous sample.
        avg_prefill_time = prefill_sum_delta / prefill_count_delta if prefill_count_delta > 0 else float("nan")
        avg_queue_time = queue_sum_delta / queue_count_delta if queue_count_delta > 0 else float("nan")
        prev = {k: values[k] for k in CUMULATIVE_METRICS}

        rows.append([
            time.time() - start_time,
            values["num_requests_running"], values["num_requests_waiting"],
            values["kv_cache_usage_perc"], values["num_preemptions_total"], completed_this_run,
            avg_prefill_time, avg_queue_time, prefill_count_delta,
        ])
        stop_event.wait(sample_interval)


def topic_dirname(detail):
    return detail.strip().lower().replace(" ", "_")


def run_warmup():
    """Fire a single throwaway request (1 prompt, 1 input/output token) before any logged runs
    begin, to absorb one-time cold-start costs (CUDA/cuBLAS kernel selection, NCCL channel
    setup) that would otherwise inflate the TTFT of whatever the first LOGGED run happens to
    be. No --save-result, so nothing from this is written to disk."""
    cmd = [
        "vllm", "bench", "serve", "--input-len", "1", "--output-len", "1",
        "--num-prompts", "1", "--request-rate", "inf", "--trust-remote-code", "--ignore-eos",
    ]
    print("Warming up server (1 throwaway request, output discarded)...")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, KeyboardInterrupt) as e:
        print(f"Warning: warm-up request failed/interrupted ({e}); continuing anyway.")


def run_bench(detail, num_prompts, output_len, input_len, run_idx, engine_metrics_url):
    log_dir = Path("./logs") / topic_dirname(detail)
    log_dir.mkdir(parents=True, exist_ok=True)
    # Encode the full parameter set (not just prompts/run) into every filename -- otherwise
    # sweeping the same --detail/--prompts with a different --input-len/--output-len would
    # silently overwrite a previous run's files.
    run_id = f"{num_prompts}p_{input_len}i_{output_len}o_run{run_idx}"

    gpu_rows, engine_rows, stop_event = [], [], Event()
    # Captured together, back-to-back, in the main thread -- BEFORE any thread/NVML/subprocess
    # work starts -- so t0_monotonic is guaranteed to correspond to the exact same real-world
    # instant as start_time (the CSV's own t=0). Letting gpu_monitor() capture t0_monotonic
    # itself (the old approach) meant it was measured after NVML init + per-GPU GPM probe setup
    # had already run in that thread, silently skewing every aligned timestamp by however long
    # that setup took.
    start_time = time.time()
    anchor = {"t0_monotonic": time.monotonic()}
    gpu_thread = Thread(target=gpu_monitor, args=(gpu_rows, stop_event, start_time, anchor))
    gpu_thread.start()

    engine_thread = None
    if engine_metrics_url:
        engine_thread = Thread(target=poll_engine_metrics,
                                args=(engine_rows, stop_event, engine_metrics_url, start_time))
        engine_thread.start()

    json_name = f"bench-{run_id}.json"
    cmd = [
        "vllm", "bench", "serve", "--input-len", str(input_len), "--output-len", str(output_len),
        "--num-prompts", str(num_prompts), "--request-rate", "inf", "--trust-remote-code",
        "--ignore-eos", "--save-result", "--save-detailed", "--result-dir", str(log_dir),
        "--result-filename", json_name,
    ]
    print(f"Running vLLM bench serve: {num_prompts} prompts, run {run_idx} (detail={detail})")
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, KeyboardInterrupt) as e:
        print(f"Warning: benchmark failed/interrupted ({e})")
    finally:
        stop_event.set()
        gpu_thread.join()
        if engine_thread is not None:
            engine_thread.join()

    n_gpus = anchor.get("num_gpus", 0)
    csv_path = log_dir / f"raw-metrics-{run_id}.csv"
    with open(csv_path, "w") as f:
        f.write(csv_header(n_gpus) + "\n")
        f.write("\n".join(",".join(map(str, row)) for row in gpu_rows))
    print(f"Telemetry saved: {csv_path}")

    engine_captured = bool(engine_rows)
    if engine_captured:
        engine_path = log_dir / f"engine-metrics-{run_id}.csv"
        with open(engine_path, "w") as f:
            f.write("timestamp,num_requests_running,num_requests_waiting,kv_cache_usage_perc,"
                    "num_preemptions_total,num_requests_completed,avg_prefill_time_seconds,"
                    "avg_queue_time_seconds,prefill_samples_count\n")
            f.write("\n".join(",".join(map(str, row)) for row in engine_rows))
        print(f"Engine metrics saved: {engine_path}")

    meta = {
        "detail": detail, "num_prompts": num_prompts, "output_len": output_len, "input_len": input_len,
        "run_idx": run_idx, "num_gpus": n_gpus, "engine_metrics_captured": engine_captured,
        "t0_monotonic": anchor.get("t0_monotonic"),
    }
    meta_path = csv_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"Metadata saved: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Condensed vLLM PCIe/GPU/engine telemetry benchmark runner.")
    parser.add_argument("--detail", required=True, help="Topic label; runs land in logs/<detail>/.")
    parser.add_argument("--prompts", type=int, nargs="+", default=[1, 20], help="Prompt counts to benchmark.")
    parser.add_argument("--input-len", type=int, default=8000, help="Input (prompt) tokens per request.")
    parser.add_argument("--output-len", type=int, default=1000, help="Output tokens per request.")
    parser.add_argument("--runs", type=int, default=1, help="Repeat runs per prompt count.")
    parser.add_argument("--engine-metrics-url", type=str, default="http://localhost:8000",
                        help="Base URL of the running `vllm serve` server. Pass '' to disable.")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip the throwaway warm-up request normally run once before the first "
                             "logged run, to absorb one-time CUDA/NCCL cold-start costs.")
    args = parser.parse_args()

    if not args.no_warmup:
        run_warmup()

    for num_prompts in args.prompts:
        for run_idx in range(1, args.runs + 1):
            print(f"=== {args.detail}: {num_prompts} prompts, run {run_idx}/{args.runs} ===")
            run_bench(args.detail, num_prompts, args.output_len, args.input_len, run_idx,
                      args.engine_metrics_url or None)
            if run_idx < args.runs:
                print("Cooling down for 5 seconds before the next run...")
                time.sleep(5)


if __name__ == "__main__":
    main()
