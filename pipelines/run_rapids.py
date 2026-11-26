#!/usr/bin/env python3
# Original author: Ilaria Billato (https://github.com/billila)
# Source: billila/pca_scwf_paper, wfsc/sc_mix/scanpy_sc_mix.py @ commit 99bdef3
# Modified by: Tom Kitak (Omnibenchmark compatibility)
# License: MIT (see repository LICENSE)
"""rapids-singlecell (GPU) pipeline. The rapids-singlecell version comes from
the environment (envs/rsc-0.13, envs/rsc-0.14), not from this script."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cupy as cp  # noqa: E402
import rapids_singlecell as rsc  # noqa: E402
import rmm  # noqa: E402
import scanpy as sc  # noqa: E402
from rmm.allocators.cupy import rmm_cupy_allocator  # noqa: E402

import cli  # noqa: E402
from data import n_cluster_target, qc_thresholds  # noqa: E402
from search import binary_search  # noqa: E402
from timing import init, phase  # noqa: E402
from writers import clustering_info, write_outputs  # noqa: E402

args = cli.parse(cli.parser("rapids-singlecell pipeline"))
init(args.outputs["methods_events"])
sc.settings.verbosity = 3

with phase("load", rapids_singlecell=rsc.__version__):
    adata = sc.read_h5ad(args.data_h5ad)
    adata.var_names_make_unique()
n_cluster = n_cluster_target(args)

with phase("gpu_load"):
    rmm.reinitialize(managed_memory=False, pool_allocator=False)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    rsc.get.anndata_to_GPU(adata)

with phase("find_mit_gene"):
    rsc.pp.flag_gene_family(adata, gene_family_name="mt", gene_family_prefix="MT-")
    rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt"])
print("cells after loading:", adata.shape, file=sys.stderr)

with phase("filter"):
    if args.filter == "manual":
        qc = qc_thresholds(adata.uns["qc_thresholds"])
        rsc.pp.filter_cells(adata, min_genes=qc["min_genes"])
        rsc.pp.filter_cells(adata, max_genes=qc["max_genes"])
        rsc.pp.filter_cells(adata, max_counts=qc["max_counts"])
        rsc.pp.filter_genes(adata, min_cells=3)
        adata = adata[adata.obs.pct_counts_mt < qc["max_mt_pct"], :]
print("cells after filtering:", adata.shape, file=sys.stderr)

with phase("normalization"):
    adata.layers["counts"] = adata.X.copy()
    rsc.pp.normalize_total(adata, target_sum=1e4)
    rsc.pp.log1p(adata)

with phase("hvg"):
    rsc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg, flavor="seurat_v3", layer="counts")
    adata.raw = adata
    adata = adata[:, adata.var["highly_variable"]]
    rsc.pp.regress_out(adata, keys=["total_counts", "pct_counts_mt"])

with phase("scaling"):
    rsc.pp.scale(adata, max_value=10)

with phase("pca"):
    rsc.pp.pca(adata, n_comps=args.n_comp)

with phase("t_sne"):
    rsc.tl.tsne(adata, n_pcs=args.n_comp)

with phase("umap"):
    rsc.pp.neighbors(adata, n_neighbors=args.n_neig, n_pcs=args.n_comp)
    rsc.tl.umap(adata)

with phase("louvain") as attrs:
    _, louvain_res, louvain_runs = binary_search(adata, n_cluster, rsc.tl.louvain)
    attrs.update(resolution=louvain_res, num_runs=louvain_runs)

with phase("leiden") as attrs:
    _, leiden_res, leiden_runs = binary_search(adata, n_cluster, rsc.tl.leiden)
    attrs.update(resolution=leiden_res, num_runs=leiden_runs)

with phase("write"):
    rsc.get.anndata_to_CPU(adata, convert_all=True)
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
