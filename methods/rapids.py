from time import time

import scanpy as sc
import cupy as cp
import rmm
import rapids_singlecell as rsc
from rmm.allocators.cupy import rmm_cupy_allocator
from sklearn.metrics import silhouette_score


def run_rapids(
    adata: sc.AnnData,
    resolution: float,
    filter: str,
    timings: dict[str, None | float],
) -> sc.AnnData:
    rmm.reinitialize(
        managed_memory=False,
        pool_allocator=False,
    )
    cp.cuda.set_allocator(rmm_cupy_allocator)

    rsc.get.anndata_to_GPU(adata)

    # find mitocondrial genes ####
    start_time = time()
    rsc.pp.flag_gene_family(adata, gene_family_name="mt", gene_family_prefix="MT-")
    rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt"])
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["find_mit_gene"] = time_elapsed

    print("after loading:", adata.shape)

    # filter data ####
    start_time = time()
    if filter == "manual":
        qc = adata.uns["qc_thresholds"]
        min_genes = qc[qc["metric"] == "nFeature"]["min"].values[0]
        max_genes = qc[qc["metric"] == "nFeature"]["max"].values[0]
        max_mt = qc[qc["metric"] == "percent.mt"]["max"].values[0]

        rsc.pp.filter_cells(adata, min_genes=min_genes)
        rsc.pp.filter_genes(adata, min_cells=3)
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
    adata.layers["counts"] = adata.X.copy()
    rsc.pp.normalize_total(adata, target_sum=1e4)
    rsc.pp.log1p(adata)
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["normalization"] = time_elapsed

    # Identification of highly variable features (feature selection) ####
    start_time = time()
    rsc.pp.highly_variable_genes(
        adata,
        n_top_genes=1000,
        flavor="seurat_v3",
        layer="counts",
    )
    adata.raw = adata
    adata = adata[:, adata.var["highly_variable"]]
    rsc.pp.regress_out(adata, keys=["total_counts", "pct_counts_mt"])
    end_time = time()
    print("'adata.raw' after HVGs:", adata.raw.shape)
    print("'adata' after HVGs:", adata.shape)
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["hvg"] = time_elapsed

    # Scaling the data ####
    start_time = time()
    rsc.pp.scale(adata, max_value=10)
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["scaling"] = time_elapsed

    # PCA ####
    start_time = time()
    rsc.pp.pca(adata, n_comps=50)
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["pca"] = time_elapsed

    # t-sne ####
    start_time = time()
    rsc.tl.tsne(adata, n_pcs=50)
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["t_sne"] = time_elapsed

    # UMAP ####
    start_time = time()
    rsc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
    rsc.tl.umap(adata)
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["umap"] = time_elapsed

    # louvain ####
    start_time = time()
    rsc.tl.louvain(adata, resolution=resolution)
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["louvain"] = time_elapsed

    # leiden ####
    start_time = time()
    rsc.tl.leiden(adata, resolution=resolution)
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["leiden"] = time_elapsed

    rsc.get.anndata_to_CPU(adata, convert_all=True)

    # Calculate silhouette scores
    silhouette_avg = silhouette_score(adata.X, adata.obs["louvain"])

    # Print the average silhouette score
    print("Average Silhouette Score:", silhouette_avg)

    # Calculate silhouette scores
    silhouette_avg = silhouette_score(adata.X, adata.obs["leiden"])

    # Print the average silhouette score
    print("Average Silhouette Score:", silhouette_avg)

    return adata
