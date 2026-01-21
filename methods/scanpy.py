from time import time

import numpy as np
import scanpy as sc
from sklearn.metrics import silhouette_score


def run_scanpy(adata, resolution: float, timings: dict["str", None | float]):
    # find mitocondrial genes ####
    start_time = time()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["find_mit_gene"] = time_elapsed

    print("after loading:", adata.shape)

    # filter data ####
    start_time = time()
    qc = adata.uns["qc_thresholds"]
    min_genes = qc[qc["metric"] == "nFeature"]["min"].values[0]
    max_genes = qc[qc["metric"] == "nFeature"]["max"].values[0]
    max_mt = qc[qc["metric"] == "percent.mt"]["max"].values[0]

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=3)
    print("after filtering1:", adata.shape)

    adata = adata[adata.obs.n_genes_by_counts < max_genes, :]
    adata = adata[adata.obs.pct_counts_mt < max_mt, :]

    end_time = time()
    print("after filtering2:", adata.shape)
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["filter"] = time_elapsed

    # normalization ####
    start_time = time()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["normalization"] = time_elapsed

    # Identification of highly variable features (feature selection) ####
    start_time = time()
    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=1000
    )
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    end_time = time()
    print("'adata.raw' after HVGs:", adata.raw.shape)
    print("'adata' after HVGs:", adata.shape)
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["hvg"] = time_elapsed

    # Scaling the data ####
    start_time = time()
    sc.pp.scale(adata, max_value=10)
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["scaling"] = time_elapsed

    # PCA ####
    start_time = time()
    sc.tl.pca(adata, svd_solver="arpack")
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["pca"] = time_elapsed

    # t-sne ####
    start_time = time()
    sc.tl.tsne(adata)
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["t_sne"] = time_elapsed

    # UMAP ####
    start_time = time()
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
    sc.tl.umap(adata)
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["umap"] = time_elapsed

    # louvain ####
    start_time = time()
    sc.tl.louvain(adata, resolution=resolution)
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["louvain"] = time_elapsed

    # Calculate silhouette scores
    silhouette_avg = silhouette_score(adata.X, adata.obs["louvain"])

    # Print the average silhouette score
    print("Average Silhouette Score:", silhouette_avg)

    # leiden ####
    start_time = time()
    sc.tl.leiden(adata, resolution=resolution)
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["leiden"] = time_elapsed

    # Calculate silhouette scores
    silhouette_avg = silhouette_score(adata.X, adata.obs["leiden"])

    # Print the average silhouette score
    print("Average Silhouette Score:", silhouette_avg)

    return adata
