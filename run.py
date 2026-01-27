# Original author: Ilaria Billato (https://github.com/billila)
# Source: billila/pca_scwf_paper, wfsc/sc_mix/scanpy_sc_mix.py @ commit 99bdef3
# Modified by: Tom Kitak (Omnibenchmark compatibility)
# License: MIT (see repository LICENSE)

import os
import sys
import argparse
import json
from pathlib import Path

import pandas as pd
import scanpy as sc

repo_dir = Path(__file__).parent
sys.path.insert(0, str(repo_dir))

from methods import run_scanpy, run_rapids


# Parse command line arguments
parser = argparse.ArgumentParser(description="Scanpy single-cell analysis pipeline")
parser.add_argument(
    "--output_dir", "-o", type=str, required=True, help="Output directory"
)
parser.add_argument("--name", "-n", type=str, required=True, help="Dataset name")
parser.add_argument(
    "--data.h5ad", dest="data_h5ad", type=str, required=True, help="Input h5ad file"
)
parser.add_argument(
    "--method_name", type=str, choices=["scanpy", "rapids"], help="Method to run"
)
parser.add_argument("--resolution", type=float, help="clustering resolution")
parser.add_argument(
    "--n_comp",
    type=int,
    default=50,
    help="number of PCA components to use for KNN graph construction",
)
parser.add_argument(
    "--n_neig",
    type=int,
    default=15,
    help="number of neighbors to use for KNN graph construction",
)
parser.add_argument(
    "--n_hvg",
    type=int,
    default=1000,
    help="number of highly variable genes to use",
)
# only have manual filtering
parser.add_argument(
    "--filter",
    type=str,
    choices=["manual"],
    default="manual",
    help="filtering strategy (manual uses suggested cutoffs; auto uses package QC)",
)
args, _ = parser.parse_known_args()

# time object to store time involved (in seconds) in each step
timings: dict[str, float | None] = {
    "find_mit_gene": None,
    "filter": None,
    "normalization": None,
    "hvg": None,
    "scaling": None,
    "pca": None,
    "t_sne": None,
    "umap": None,
    "louvain": None,
    "leiden": None,
}

sc.settings.verbosity = 3

# data ####
adata = sc.read_h5ad(args.data_h5ad)
adata.var_names_make_unique()

if args.method_name == "scanpy":
    adata = run_scanpy(
        adata,
        args.resolution,
        args.n_comp,
        args.n_neig,
        args.n_hvg,
        args.filter,
        timings,
    )
elif args.method_name == "rapids":
    adata = run_rapids(
        adata,
        args.resolution,
        args.n_comp,
        args.n_neig,
        args.n_hvg,
        args.filter,
        timings,
    )


# Save timings as JSON
with open(os.path.join(args.output_dir, f"{args.name}.timings.json"), "w") as f:
    json.dump(timings, f, indent=2)

# Save cluster assignments as TSV
adata.obs[["louvain", "leiden"]].reset_index(names="cell_id").to_csv(
    os.path.join(args.output_dir, f"{args.name}.clusters.tsv"),
    sep="\t",
    index=False,
)

hvg_list = adata.var.highly_variable.index.to_series()
hvg_list.to_csv(
    os.path.join(args.output_dir, f"{args.name}.hvgs.tsv"),
    sep="\t",
    index=False,
    header=False,
)

# Save PCA coordinates as TSV
pca_df = pd.DataFrame(
    adata.obsm["X_pca"],
    index=adata.obs_names,
    columns=[f"PC{i+1}" for i in range(adata.obsm["X_pca"].shape[1])],
)
pca_df.to_csv(
    os.path.join(args.output_dir, f"{args.name}.pca.tsv"),
    sep="\t",
    index=True,
    index_label="cell_id",
)
