"""Command-line parsing for the pipeline entrypoints (omnibenchmark api 0.8).

ob calls an entrypoint as
    <script> --output_dir D --name N --output <id>=<path> ... --<input> <path> --<param> <value>
"""

import argparse
import os

OUTPUTS = {
    # output id -> pre-0.8 suffix (<output_dir>/<name><suffix>)
    "methods_clusters": ".clusters.tsv",
    "methods_pca": ".pca.tsv",
    "methods_hvgs": ".hvgs.tsv",
    "methods_clustering_info": ".clustering_info.json",
}


def _key_path(s):
    key, sep, path = s.partition("=")
    if not sep or not key or not path:
        raise argparse.ArgumentTypeError(f"--output wants ID=PATH, got {s!r}")
    return key, path


def parser(description, filters=("manual",)):
    """Parser with the arguments every pipeline takes."""
    p = argparse.ArgumentParser(description=description, allow_abbrev=False)
    p.add_argument("--output_dir", help="node output directory")
    p.add_argument("--name", help="module id")
    p.add_argument("--output", type=_key_path, action="append", default=[],
                   help="repeatable: output id=path")
    p.add_argument("--data_h5ad", required=True, help="input h5ad")
    p.add_argument("--data_clusters_truth_num", required=True,
                   help="file holding the true number of clusters")
    p.add_argument("--d_cluster", type=int, required=True,
                   help="target clusters = true clusters + d_cluster")
    p.add_argument("--n_comp", type=int, default=50, help="number of PCs")
    p.add_argument("--n_neig", type=int, default=15, help="number of neighbours (kNN)")
    p.add_argument("--n_hvg", type=int, default=1000, help="number of highly variable genes")
    p.add_argument("--filter", choices=filters, default=filters[0],
                   help="manual: the dataset's suggested QC thresholds")
    return p


def parse(p):
    """Parse and attach `args.outputs`: output id -> path, plus the events path."""
    args = p.parse_args()
    given = dict(args.output)
    outputs = {}
    for oid, suffix in OUTPUTS.items():
        if oid in given:
            outputs[oid] = given[oid]
        elif args.output_dir and args.name:
            outputs[oid] = os.path.join(args.output_dir, args.name + suffix)
        else:
            p.error(f"no --output {oid}=PATH and no --output_dir/--name to fall back on")
    outputs["methods_events"] = given.get(
        "methods_events", os.path.join(args.output_dir or ".", "obkit-events.jsonl"))
    args.outputs = outputs
    return args
