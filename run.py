# Original author: Ilaria Billato (https://github.com/billila)
# Source: billila/pca_scwf_paper, wfsc/sc_mix/scanpy_sc_mix.py @ commit 99bdef3
# Modified by: Tom Kitak (Omnibenchmark compatibility)
# License: MIT (see repository LICENSE)

import argparse
import json
import os

import numpy as np
import pandas as pd
import scanpy as sc
import time
import anndata
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# Parse command line arguments
parser = argparse.ArgumentParser(description="Scanpy single-cell analysis pipeline")
parser.add_argument("--output_dir", "-o", type=str, required=True, help="Output directory")
parser.add_argument("--name", "-n", type=str, required=True, help="Dataset name")
parser.add_argument("--data.h5ad", dest="data_h5ad", type=str, required=True, help="Input h5ad file")
parser.add_argument("--methods", type=str, default="scanpy", help="Method to run")
args, _ = parser.parse_known_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

sc.settings.verbosity = 3

# save time usage ####
time_sc = pd.DataFrame(index=["find_mit_gene", "filter", "normalization", "hvg",
                           "scaling", "PCA", "t-sne", "umap", "louvain", "leiden"],
                    columns=["time_sec"])

# data ####
adata = sc.read_h5ad(args.data_h5ad)
adata.var_names_make_unique()
adata

eprint("after loading: ", adata.shape)



#sc.pl.highest_expr_genes(adata, n_top=20, )

# find mitocondrial genes ####
start_time = time.time()

adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

#sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
#             jitter=0.4, multi_panel=True)

end_time = time.time()
time_elapsed = end_time - start_time
print("Time Elapsed:", time_elapsed)
time_sc.iloc[0, 0] = time_elapsed

# filter data ####
start_time = time.time()

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
eprint("after filtering1: ", adata.shape)

#sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
#sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

adata = adata[adata.obs.n_genes_by_counts < 5000, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]

end_time = time.time()
eprint("after filtering2: ", adata.shape)
time_elapsed = end_time - start_time
print("Time Elapsed:", time_elapsed)
time_sc.iloc[1, 0] = time_elapsed

# normalization ####
start_time = time.time()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

end_time = time.time()
time_elapsed = end_time - start_time
print("Time Elapsed:", time_elapsed)
time_sc.iloc[2, 0] = time_elapsed

# Identification of highly variable features (feature selection) ####
start_time = time.time()

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = 1000)
#sc.pl.highly_variable_genes(adata)

adata.raw = adata
adata = adata[:, adata.var.highly_variable]

end_time = time.time()
eprint("'adata.raw' after HVGs: ", adata.raw.shape)
eprint("'adata' after HVGs: ", adata.shape)

time_elapsed = end_time - start_time
print("Time Elapsed:", time_elapsed)
time_sc.iloc[3, 0] = time_elapsed


#df = pd.DataFrame(adata.raw.var.highly_variable, columns=['hvg'])
#df.to_csv(os.path.join(args.output_dir, f'{args.name}.hvgs.tsv'), sep='\t', index=False)
#df.to_excel(os.path.join(args.output_dir, f'{args.name}_hvg.xlsx'), index=False)

hvg_list = list(adata.var.highly_variable.index)
fn = os.path.join(args.output_dir, f'{args.name}.hvgs.tsv')

with open(fn, "w") as outfile:
    outfile.write("\n".join(hvg_list))

# Scaling the data ####
start_time = time.time()

sc.pp.scale(adata, max_value=10)

end_time = time.time()
time_elapsed = end_time - start_time
print("Time Elapsed:", time_elapsed)
time_sc.iloc[4, 0] = time_elapsed




# PCA ####
start_time = time.time()

sc.tl.pca(adata, svd_solver='arpack')
#sc.pl.pca(adata, color='CST3')
#sc.pl.pca_variance_ratio(adata, log=True)

adata

end_time = time.time()
time_elapsed = end_time - start_time
print("Time Elapsed:", time_elapsed)
time_sc.iloc[5, 0] = time_elapsed

# t-sne ####
start_time = time.time()

sc.tl.tsne(adata)

end_time = time.time()
time_elapsed = end_time - start_time
print("Time Elapsed:", time_elapsed)
time_sc.iloc[6, 0] = time_elapsed

# UMAP ####
start_time = time.time()
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
sc.tl.umap(adata)

end_time = time.time()
time_elapsed = end_time - start_time
print("Time Elapsed:", time_elapsed)
time_sc.iloc[7, 0] = time_elapsed

# louvain ####
start_time = time.time()

sc.tl.louvain(adata, resolution = 0.13)

end_time = time.time()
time_elapsed = end_time - start_time
print("Time Elapsed:", time_elapsed)
time_sc.iloc[8, 0] = time_elapsed

#true_labels = adata.obs['cell_line'].astype(str)
#predicted_labels = adata.obs['louvain'].astype(str)



# Compute the ARI
#ari_score = adjusted_rand_score(true_labels, predicted_labels)

# Print the ARI score
#print("Adjusted Rand Index (ARI):", ari_score)

cluster_labels = adata.obs['louvain']

# Calculate silhouette scores
silhouette_avg = silhouette_score(adata.X, cluster_labels)

# Print the average silhouette score
print("Average Silhouette Score:", silhouette_avg)


# leiden ####
start_time = time.time()

sc.tl.leiden(adata, resolution = 0.13)

end_time = time.time()
time_elapsed = end_time - start_time
print("Time Elapsed:", time_elapsed)
time_sc.iloc[9, 0] = time_elapsed

#true_labels = adata.obs['cell_line'].astype(str)
#predicted_labels = adata.obs['leiden'].astype(str)

# Compute the ARI
#ari_score = adjusted_rand_score(true_labels, predicted_labels)

# Print the ARI score
#print("Adjusted Rand Index (ARI):", ari_score)

cluster_labels = adata.obs['leiden']

# Calculate silhouette scores
silhouette_avg = silhouette_score(adata.X, cluster_labels)

# Print the average silhouette score
print("Average Silhouette Score:", silhouette_avg)

time_sc

# Save outputs for omnibenchmark ####

# Save timings as JSON
timings_dict = time_sc['time_sec'].to_dict()
with open(os.path.join(args.output_dir, f'{args.name}.timings.json'), 'w') as f:
    json.dump(timings_dict, f, indent=2)

# Save cluster assignments as TSV
cluster_df = pd.DataFrame({
    'cell_id': adata.obs_names,
    'louvain': adata.obs['louvain'].values,
    'leiden': adata.obs['leiden'].values
})
#if 'cell_line' in adata.obs.columns:
#    cluster_df['cell_line'] = adata.obs['cell_line'].values
cluster_df.to_csv(os.path.join(args.output_dir, f'{args.name}.clusters.tsv'), sep='\t', index=False)

# Save PCA coordinates as TSV
pca_df = pd.DataFrame(
    adata.obsm['X_pca'],
    index=adata.obs_names,
    columns=[f'PC{i+1}' for i in range(adata.obsm['X_pca'].shape[1])]
)
pca_df.to_csv(os.path.join(args.output_dir, f'{args.name}.pca.tsv'), sep='\t', index=True)

# Also save the full h5ad file
#adata.write(os.path.join(args.output_dir, f'{args.name}.h5ad'))
