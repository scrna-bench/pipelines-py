from time import time

import scanpy as sc
from search_res import binary_search
import sys


def run_scanpy(
    adata: sc.AnnData,
    n_cluster: int,
    n_comp: int,
    n_neig: int,
    n_hvg: int,
    filter: str,
    timings: dict[str, None | float],
    starts_ends: dict[str, None | float],
    clustering_info: dict[str, dict[str, None | float | int]],
) -> sc.AnnData:
    # find mitocondrial genes ####
    starts_ends["find_mit_gene"] = start_time = time()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    end_time = time()
    starts_ends["find_mit_gene"].append(end_time)
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["find_mit_gene"] = time_elapsed
    

    print("after loading:", adata.shape)
    sys.stderr.write("cells after loading: " + str(adata.shape) + "\n")

    # filter data ####
    starts_ends["filter"] = start_time = time()
    if filter == "manual":
        sys.stderr.write(
            "mt pcts [0:3]: " + str(adata.obs.pct_counts_mt.values[0:3]) + "\n"
        )
        sys.stderr.write(
            "cells detected [0:3]: "
            + str(adata.obs.n_genes_by_counts.values[0:3])
            + "\n"
        )

        qc = adata.uns["qc_thresholds"]
        min_genes = qc[qc["metric"] == "nFeature"]["min"].values[0]
        max_genes = qc[qc["metric"] == "nFeature"]["max"].values[0]
        max_counts = qc[qc["metric"] == "nCount"]["max"].values[0]
        max_mt = qc[qc["metric"] == "percent.mt"]["max"].values[0]

        sc.pp.filter_cells(adata, min_genes=min_genes)
        sys.stderr.write("cells after filtering1: " + str(adata.shape) + "\n")
        sc.pp.filter_cells(adata, max_genes=max_genes)
        sys.stderr.write("cells after filtering2: " + str(adata.shape) + "\n")
        sc.pp.filter_cells(adata, max_counts=max_counts)
        sys.stderr.write("cells after filtering3: " + str(adata.shape) + "\n")
        sc.pp.filter_genes(adata, min_cells=3)

        adata = adata[adata.obs.pct_counts_mt < max_mt, :]

    end_time = time()
    starts_ends["filter"].append(end_time)
    print("after filtering4:", adata.shape)
    sys.stderr.write("cells after filtering4: " + str(adata.shape) + "\n")
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["filter"] = time_elapsed

    # normalization ####
    starts_ends["normalization"] = start_time = time()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    end_time = time()
    starts_ends["normalization"].append(end_time)
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["normalization"] = time_elapsed

    # Identification of highly variable features (feature selection) ####
    starts_ends["hvg"] = start_time = time()
    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=n_hvg
    )
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    end_time = time()
    starts_ends["hvg"].append(end_time)
    print("'adata.raw' after HVGs:", adata.raw.shape)
    print("'adata' after HVGs:", adata.shape)
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["hvg"] = time_elapsed

    # Scaling the data ####
    starts_ends["scaling"] = start_time = time()
    sc.pp.scale(adata, max_value=10)
    end_time = time()
    starts_ends["scaling"].append(end_time)
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["scaling"] = time_elapsed

    # PCA ####
    starts_ends["pca"] = start_time = time()
    sc.tl.pca(adata, svd_solver="arpack", n_comps=n_comp)
    end_time = time()
    starts_ends["pca"].append(end_time)
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["pca"] = time_elapsed

    # t-sne ####
    starts_ends["t_sne"] = start_time = time()
    sc.tl.tsne(adata)
    end_time = time()
    starts_ends["t_sne"].append(end_time)
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["t_sne"] = time_elapsed

    # UMAP ####
    starts_ends["umap"] = start_time = time()
    sc.pp.neighbors(adata, n_neighbors=n_neig, n_pcs=n_comp)
    sc.tl.umap(adata)
    end_time = time()
    starts_ends["umap"].append(end_time)
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["umap"] = time_elapsed

    # louvain ####
    starts_ends["louvain"] = start_time = time()
    _, res, num_runs = binary_search(adata, n_cluster, sc.tl.louvain)
    end_time = time()
    starts_ends["louvain"].append(end_time)
    time_elapsed = end_time - start_time
    avg_time_elapsed = time_elapsed / num_runs
    print(f"Louvain resolution: {res}")
    print(f"Louvain runs: {num_runs}")
    print("Total Search Time Elapsed:", time_elapsed)
    print("Average Time per Clustering Run:", avg_time_elapsed)
    timings["louvain"] = avg_time_elapsed
    clustering_info["resolutions"]["louvain"] = res
    clustering_info["num_runs"]["louvain"] = num_runs

    # leiden ####
    starts_ends["leiden"] = start_time = time()
    _, res, num_runs = binary_search(adata, n_cluster, sc.tl.leiden)
    end_time = time()
    time_elapsed = end_time - start_time
    avg_time_elapsed = time_elapsed / num_runs
    print(f"Leiden resolution: {res}")
    print(f"Leiden runs: {num_runs}")
    print("Total Search Time Elapsed:", time_elapsed)
    print("Average Time per Clustering Run:", avg_time_elapsed)
    timings["leiden"] = avg_time_elapsed
    clustering_info["resolutions"]["leiden"] = res
    clustering_info["num_runs"]["leiden"] = num_runs

    return adata
