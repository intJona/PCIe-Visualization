# PCIe Visualization

Benchmark vLLM, capture PCIe/GPU/engine telemetry alongside it, and view the results in an
interactive dashboard.

- [`bench_run.py`](bench_run.py) — runs `vllm bench serve` while polling GPU (NVML) and engine
  (`/metrics`) telemetry in the background, and writes everything to `./logs/`.
- [`dashboard.py`](dashboard.py) — a Streamlit + Plotly viewer for that `./logs/` output.

## Setup

```bash
conda create -n vllm python=3.13
conda activate vllm
pip install -r requirements.txt
```

`vllm`/`torch` are hardware- and CUDA-version-sensitive. If `pip install -r requirements.txt`
fails on them, install vLLM first per the [official instructions](https://docs.vllm.ai/en/latest/getting_started/installation/index.html)
for your CUDA version, then re-run `pip install -r requirements.txt` to pick up the rest.

## Serving the model

Start the vLLM server first, in its own terminal — the commands below are what we've been using
to test with (Qwen3-14B-FP8 across 2 GPUs), flags and all:

```bash
conda activate vllm

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export VLLM_USE_FLASHINFER_SAMPLER=0

CUDA_VISIBLE_DEVICES=0,1 vllm serve /home/intel/.cache/huggingface/hub/models--Qwen--Qwen3-14B-FP8/snapshots/9a283b4a5efbc09ce247e0ae5b02b744739e525a/ \
    --tensor-parallel-size 2 \
    --max-model-len auto \
    --disable-custom-all-reduce \
    --enforce-eager --no-enable-prefix-caching
```

`--no-enable-prefix-caching` matters for benchmarking: it stops vLLM from reusing KV cache
between prompts, so every request is measured as a genuine cold pass instead of a cache hit.

## Running a benchmark

Once the server is up, run a benchmark in a single command — pick a topic label, how many
prompts to send, how many tokens per prompt/response, and how many times to repeat it:

```bash
python bench_run.py --detail "my_test" --prompts 20 --input-len 200 --output-len 100 --runs 2
```

Results land in `logs/my_test/`.

## Viewing the dashboard

```bash
streamlit run dashboard.py
```

Opens in your browser. Pick a configuration and its runs in the sidebar to plot PCIe transfer,
GPU/engine utilization, and the request lifecycle timeline.

## Flags and terminology

### `bench_run.py`

| Flag | Default | Description |
|---|---|---|
| `--detail` | *required* | Topic label. Runs are written to `logs/<detail>/`. |
| `--prompts` | `1 20` | One or more prompt counts to benchmark (space-separated), one full run per value. |
| `--input-len` | `8000` | Input (prompt) tokens per request. |
| `--output-len` | `1000` | Output tokens generated per request. |
| `--runs` | `1` | How many times to repeat each prompt count (`bench-..._run1.json`, `_run2.json`, ...). |
| `--engine-metrics-url` | `http://localhost:8000` | Base URL of the running `vllm serve` server, for polling `/metrics`. Pass `""` to disable. |
| `--no-warmup` | off | Skip the throwaway warm-up request normally run once before the first logged run, to absorb one-time CUDA/NCCL cold-start cost. |

### `dashboard.py` sidebar

| Control | Effect |
|---|---|
| Configurations | Selects which (topic, prompts, input/output tokens) combinations to plot. |
| Runs (per configuration) | Selects which repeated runs of each configuration to plot; a colored dot legend under each picker matches each run's plotted color/shade. |
| GPU view | `GPU N` (pick specific GPUs), `All GPUs` (one line per GPU), or `Total (Sum)` (GPUs summed/averaged into one line). |
| Rows to Display | Show/hide Row 1, 2, and 3 independently to reclaim vertical space. |
| Row 1: PCIe Data Transfer | `Total`, `TX`, `RX`, or (single-run view only) `TX+RX` together. |
| Row 2: Engine Utilization | Always on when shown: KV cache % and preemption events. |
| Row 3: Request Lifecycle | Always on when shown — cumulative token count over time plus a per-request TTFT marker. |

### Glossary

- **PCIe H2D / D2H** — Host-to-Device / Device-to-Host: bytes per second moving *into* vs *out
  of* the GPU over PCIe. `Total` is H2D + D2H combined.
- **SM / memory controller utilization** — still recorded in `raw-metrics-*.csv` (`sm_util`,
  `mem_util`) but not plotted: this workstation's GPUs don't support NVML's GPM API, so these
  fall back to a coarse, easily-misread utilization estimate rather than true SM occupancy.
- **KV cache usage** — percent of vLLM's GPU KV cache (the storage holding every request's past
  attention keys/values, so it doesn't have to be recomputed each token) currently occupied.
  Running near 100% causes preemptions (requests paused/evicted to free space).
- **TTFT (Time To First Token)** — wall-clock time from when a request was submitted to when its
  first output token came back. Measured client-side by `vllm bench serve`, so it's exact but
  doesn't split out queueing vs. prefill.
- **Prefill** — the pass over a request's *input* prompt (computing its initial KV cache) before
  any output tokens can be generated. Compute-heavy and roughly proportional to input length.
- **Decode** — the token-by-token generation phase after prefill, one output token per step.
  Memory-bandwidth-bound rather than compute-bound (see SM vs. Mem above).
- **Queue time** — time a request spent waiting before the engine started processing it at all,
  separate from prefill itself. Reported only as a run-level average in the summary table (vLLM
  only records it once a request completes, so it has no honest single instant to plot on a
  timeline).
