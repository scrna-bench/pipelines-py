"""Writers for the PIPELINE stage outputs, identical across pipelines."""

import json

import numpy as np
import polars as pl


def write_outputs(outputs, cell_ids, pca, hvgs, louvain, leiden, clustering_info):
    """cell_ids: per-cell ids; pca: cells x PCs array; hvgs: gene names;
    louvain/leiden: per-cell labels. louvain is None for pipelines without
    Louvain, and the column is then left empty."""
    cell_ids = pl.Series("cell_id", np.asarray(cell_ids).astype(str))
    n = len(cell_ids)

    def labels(name, values):
        if values is None:
            return pl.Series(name, [None] * n, dtype=pl.String)
        return pl.Series(name, [None if v is None else str(v) for v in values], dtype=pl.String)

    pl.DataFrame([cell_ids, labels("louvain", louvain), labels("leiden", leiden)]).write_csv(
        outputs["methods_clusters"], separator="\t"
    )

    pca = np.asarray(pca)  # keep float32 as float32: shorter, exact output
    pl.DataFrame(pca, schema=[f"PC{i + 1}" for i in range(pca.shape[1])], orient="row").insert_column(
        0, cell_ids
    ).write_csv(outputs["methods_pca"], separator="\t")

    with open(outputs["methods_hvgs"], "w") as f:
        f.writelines(f"{g}\n" for g in hvgs)

    with open(outputs["methods_clustering_info"], "w") as f:
        json.dump(clustering_info, f, indent=2)


def clustering_info(louvain=(None, None), leiden=(None, None)):
    """{'resolutions': {...}, 'num_runs': {...}} from (resolution, num_runs) pairs."""
    return {
        "resolutions": {"louvain": louvain[0], "leiden": leiden[0]},
        "num_runs": {"louvain": louvain[1], "leiden": leiden[1]},
    }
