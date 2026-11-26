# Methods in python

Python pipelines for the scRNA pipelines benchmark, one omnibenchmark (api 0.8)
entrypoint per pipeline:

| entrypoint | script | environment(s) |
|---|---|---|
| `scanpy` | `pipelines/run_scanpy.py` | `envs/scanpy` |
| `rapids` | `pipelines/run_rapids.py` | `envs/rsc-0.13`, `envs/rsc-0.14` (GPU, CUDA 12) |
| `brisc` | `pipelines/run_brisc.py` | `envs/brisc` (no Louvain: that column is NA) |

Shared code lives in `src/`: argument parsing (`cli.py`), inputs (`data.py`),
the resolution search (`search.py`), output writers (`writers.py`) and phase
timings (`timing.py`), which are recorded as [obkit](https://github.com/omnibenchmark/obkit)
events in `obkit-events.jsonl`.

## Environments

Each `envs/<env>/` is its own [pixi](https://pixi.sh) project (linux-64).
CI (`.github/workflows/containers.yml`) builds each into
`ghcr.io/scrna-bench/pipelines-py:<env>-<sha>`, which the benchmark plan
references. After editing a `pixi.toml`, run `pixi lock` in its folder.
