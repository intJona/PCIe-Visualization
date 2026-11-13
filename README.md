# PCIe Visualization

Runs a vLLM serving benchmark on NVIDIA GPUs while recording PCIe, GPU and engine telemetry,
then shows the results in an interactive dashboard. Use it to see how a PCIe configuration
(topology, lane width, ...) affects tensor-parallel inference.

- [`bench_run.py`](bench_run.py) — runs `vllm bench serve` and records telemetry to `./logs/`
- [`dashboard.py`](dashboard.py) — Streamlit + Plotly viewer for `./logs/`

## Why this exists

`vllm bench serve` reports request-level latency and throughput, but nothing about what the
PCIe link was doing at the time. `bench_run.py` wraps it so that, in one command:

- **PCIe traffic** (H2D / D2H per GPU, via NVML) is sampled in the background during the run
- **Engine state** (KV cache usage, queue, preemptions, prefill time) is sampled from vLLM's
  `/metrics` endpoint
- **Everything shares one clock** and lands in a fixed file layout, so the dashboard can line
  up PCIe traffic with the request timeline
- **Runs are repeatable and labelled** (`--detail`, `--runs`), so different PCIe configurations
  can be compared side by side

## Requirements

- Linux, 1+ NVIDIA GPUs with a working vLLM install (tensor parallel needs 2+)
- Python environment with the packages in `requirements.txt`

```bash
conda create -n vllm python=3.13
conda activate vllm
pip install -r requirements.txt
```

`vllm`/`torch` are sensitive to your CUDA version. If the install fails on them, install vLLM
first following the [official instructions](https://docs.vllm.ai/en/latest/getting_started/installation/index.html),
then re-run `pip install -r requirements.txt`.

Install only what you need:

- `requirements-bench.txt` — `bench_run.py` (vLLM + NVML), needs a GPU
- `requirements-dashboard.txt` — `dashboard.py` only (Streamlit + Plotly); useful for viewing
  `logs/` from a machine that does not run benchmarks

## 1. Start the vLLM server

In its own terminal:

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/model \
    --tensor-parallel-size 2 \
    --max-model-len auto \
    --disable-custom-all-reduce \
    --enforce-eager --no-enable-prefix-caching
```

Adjust the model, GPU list and tensor-parallel size to your setup. These flags are the ones used
for testing. `--no-enable-prefix-caching` matters: it stops vLLM reusing KV cache between
prompts, so every request is a cold pass.

## 2. Run a benchmark

```bash
python bench_run.py --detail "my_test" --prompts 20 --input-len 200 --output-len 100 --runs 2
```

Results are written to `logs/my_test/`.

| Flag | Default | Description |
|---|---|---|
| `--detail` | *required* | Topic label; results go to `logs/<detail>/` |
| `--prompts` | `1 20` | One or more prompt counts (space-separated); one run set per value |
| `--input-len` | `8000` | Input tokens per request |
| `--output-len` | `1000` | Output tokens per request |
| `--runs` | `1` | Repetitions per prompt count |
| `--engine-metrics-url` | `http://localhost:8000` | Base URL of the vLLM server for `/metrics`; `""` disables |
| `--no-warmup` | off | Skip the warm-up request normally sent before the first logged run |

## 3. View the dashboard

```bash
streamlit run dashboard.py
```

Pick a configuration and its runs in the sidebar.

| Control | Effect |
|---|---|
| Configurations | Which (topic, prompts, input/output tokens) combinations to plot |
| Runs | Which repeated runs to plot |
| GPU view | `GPU N`, `All GPUs` (one line each) or `Total (Sum)` |
| Rows to Display | Show or hide rows 1–3 |
| Row 1: PCIe Data Transfer | `Total`, `TX`, `RX`, or `TX+RX` (single-run view only) |
| Row 2: KV cache | KV cache % and preemption events |
| Row 3: Request Lifecycle | Cumulative token count over time with a per-request TTFT marker |

## Storage

Logs location, highest priority first:

1. `PCIE_VIZ_LOGS_DIR` environment variable
2. `.storage-config` (gitignored; copy from `.storage-config.example`)
3. `./logs`

## Glossary

- **PCIe H2D / D2H** — bytes per second moving into / out of the GPU over PCIe. `Total` is both.
- **SM / memory utilization** — recorded in `raw-metrics-*.csv` (`sm_util`, `mem_util`) but not
  plotted. Without NVML GPM support on the GPU, `sm_util` is only a coarse "any kernel running"
  bit that sits near 100% under continuous batching, so it is easy to misread.
- **KV cache usage** — share of vLLM's GPU KV cache in use. Near 100% causes preemptions
  (requests paused/evicted to free space).
- **TTFT** — time from request submission to its first output token, measured client-side by
  `vllm bench serve`. Includes queueing and prefill.
- **Prefill** — processing of the input prompt before the first output token; compute-heavy,
  roughly proportional to input length.
- **Decode** — token-by-token generation after prefill; typically memory-bandwidth-bound.
- **Queue time** — time a request waits before the engine starts on it. Shown only as a run-level
  average, since vLLM records it once a request completes.
