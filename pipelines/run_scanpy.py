#!/usr/bin/env python3
# Original author: Ilaria Billato (https://github.com/billila)
# Source: billila/pca_scwf_paper, wfsc/sc_mix/scanpy_sc_mix.py @ commit 99bdef3
# Modified by: Tom Kitak (Omnibenchmark compatibility)
# License: MIT (see repository LICENSE)
"""scanpy pipeline: QC, normalisation, HVGs, PCA, t-SNE, UMAP, Louvain/Leiden."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import scanpy as sc  # noqa: E402

import cli  # noqa: E402
from data import n_cluster_target, qc_thresholds  # noqa: E402
from search import binary_search  # noqa: E402
from timing import init, phase  # noqa: E402
from writers import clustering_info, write_outputs  # noqa: E402

args = cli.parse(cli.parser("scanpy pipeline"))
init(args.outputs["methods_events"])
sc.settings.verbosity = 3

with phase("load"):
    adata = sc.read_h5ad(args.data_h5ad)
    adata.var_names_make_unique()
n_cluster = n_cluster_target(args)

with phase("find_mit_gene"):
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
print("cells after loading:", adata.shape, file=sys.stderr)

with phase("filter"):
    if args.filter == "manual":
        qc = qc_thresholds(adata.uns["qc_thresholds"])
        sc.pp.filter_cells(adata, min_genes=qc["min_genes"])
        sc.pp.filter_cells(adata, max_genes=qc["max_genes"])
        sc.pp.filter_cells(adata, max_counts=qc["max_counts"])
        sc.pp.filter_genes(adata, min_cells=3)
        adata = adata[adata.obs.pct_counts_mt < qc["max_mt_pct"], :]
print("cells after filtering:", adata.shape, file=sys.stderr)

with phase("normalization"):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

with phase("hvg"):
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=args.n_hvg)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]

with phase("scaling"):
    sc.pp.scale(adata, max_value=10)

with phase("pca"):
    sc.tl.pca(adata, svd_solver="arpack", n_comps=args.n_comp)

with phase("t_sne"):
    sc.tl.tsne(adata)

with phase("umap"):
    sc.pp.neighbors(adata, n_neighbors=args.n_neig, n_pcs=args.n_comp)
    sc.tl.umap(adata)

with phase("louvain") as attrs:
    _, louvain_res, louvain_runs = binary_search(adata, n_cluster, sc.tl.louvain)
    attrs.update(resolution=louvain_res, num_runs=louvain_runs)

with phase("leiden") as attrs:
    _, leiden_res, leiden_runs = binary_search(adata, n_cluster, sc.tl.leiden)
    attrs.update(resolution=leiden_res, num_runs=leiden_runs)

with phase("write"):
    write_outputs(
        args.outputs,
        cell_ids=adata.obs_names,
        pca=adata.obsm["X_pca"],
        hvgs=adata.var_names,
        louvain=adata.obs["louvain"].values,
        leiden=adata.obs["leiden"].values,
        clustering_info=clustering_info(
            louvain=(louvain_res, louvain_runs), leiden=(leiden_res, leiden_runs)
        ),
    )
