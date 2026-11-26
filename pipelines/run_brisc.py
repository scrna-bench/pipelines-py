#!/usr/bin/env python3
# Original author: Ilaria Billato (https://github.com/billila)
# Source: billila/pca_scwf_paper, wfsc/sc_mix/scanpy_sc_mix.py @ commit 99bdef3
# Modified by: Tom Kitak (Omnibenchmark compatibility)
# Modified by: Mark Robinson (brisc pipeline, omni-training/pipelines-py)
# License: MIT (see repository LICENSE)
"""brisc pipeline. brisc has no Louvain: the louvain column is left empty (NA)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import polars as pl  # noqa: E402
from brisc import SingleCell  # noqa: E402

import cli  # noqa: E402
from data import n_cluster_target, qc_thresholds  # noqa: E402
from timing import init, phase  # noqa: E402
from writers import clustering_info, write_outputs  # noqa: E402


def leiden_to_target(sc, n_target, max_runs=50, max_resolution=100.0):
    """Leiden resolution giving n_target clusters: double the resolution until
    there are enough clusters, then bisect. Returns (dataset with
    obs['leiden'], resolution, number of runs) for the closest result."""
    runs = 0
    best = None  # (abs diff, dataset, resolution)

    def run(res):
        nonlocal runs, best
        runs += 1
        out = sc.cluster(resolution=res, cluster_column="leiden", overwrite=True)
        k = out.obs["leiden"].drop_nulls().n_unique()
        diff = abs(k - n_target)
        if best is None or diff < best[0]:
            best = (diff, out, res)
        return k

    lo, hi = 0.0, 1.0
    k = run(hi)
    while k < n_target and hi < max_resolution and runs < max_runs:
        lo, hi = hi, hi * 2
        k = run(hi)
    while k != n_target and runs < max_runs:
        mid = (lo + hi) / 2
        k = run(mid)
        if k < n_target:
            lo = mid
        elif k > n_target:
            hi = mid
    return best[1], best[2], runs


args = cli.parse(cli.parser("brisc pipeline"))
init(args.outputs["methods_events"])

with phase("load"):
    sc = SingleCell(args.data_h5ad)
    # brisc refuses to dedupe when a name already contains the separator, and
    # gene names like "MT-CO1" contain "-", so only dedupe when needed
    if sc.var_names.is_duplicated().any():
        sc = sc.make_var_names_unique(separator="__")
n_cluster = n_cluster_target(args)

with phase("find_mit_gene"):
    # allow_float: many h5ad files store raw counts as float32
    sc = sc.qc_metrics(allow_float=True)

with phase("filter"):
    if args.filter == "manual":
        qc = qc_thresholds(sc.uns["qc_thresholds"])
        sc = sc.qc(
            subset=True,
            allow_float=True,
            max_mito_fraction=qc["max_mt_pct"] / 100,
            min_genes=int(qc["min_genes"]),
            custom_filter=(pl.col("num_counts") <= qc["max_counts"])
            & (pl.col("num_genes") <= qc["max_genes"]),
            nonzero_MALAT1=False,
        )

# brisc selects HVGs on raw counts, so before normalisation
with phase("hvg"):
    sc = sc.hvg(num_genes=args.n_hvg)

with phase("normalization"):
    sc = sc.normalize()

with phase("pca"):
    sc = sc.pca(num_PCs=args.n_comp)

with phase("neighbors"):
    sc = sc.neighbors(num_neighbors=args.n_neig).shared_neighbors()

with phase("umap"):
    sc = sc.umap()

with phase("leiden") as attrs:
    sc, leiden_res, leiden_runs = leiden_to_target(sc, n_cluster)
    attrs.update(resolution=leiden_res, num_runs=leiden_runs)

with phase("write"):
    hvgs = sc.var.filter(pl.col("highly_variable")).sort("highly_variable_rank")
    write_outputs(
        args.outputs,
        cell_ids=sc.obs_names.to_list(),
        pca=sc.obsm["pca"],
        hvgs=hvgs["_index"].to_list(),
        louvain=None,
        leiden=sc.obs["leiden"].cast(pl.String).to_list(),
        clustering_info=clustering_info(leiden=(leiden_res, leiden_runs)),
    )
