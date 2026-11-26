"""Inputs shared by the pipelines: target cluster count and QC thresholds."""


def n_cluster_target(args):
    with open(args.data_clusters_truth_num) as f:
        return int(f.read().strip()) + args.d_cluster


def _columns(table):
    """A metric/min/max table as a dict of lists, whichever form the reader
    returned it in: pandas or polars DataFrame, or a dict of columns (brisc)."""
    if hasattr(table, "to_pandas"):  # polars
        return table.to_dict(as_series=False)
    if hasattr(table, "to_dict"):  # pandas
        return table.to_dict("list")
    return {k: list(v) for k, v in table.items()}


def qc_thresholds(table):
    """The dataset's suggested QC thresholds (uns['qc_thresholds'], written by
    the datasets module as a metric/min/max table) as plain numbers."""
    cols = _columns(table)
    rows = {m: i for i, m in enumerate(cols["metric"])}

    def get(metric, bound):
        return float(cols[bound][rows[metric]])

    return {
        "min_genes": get("nFeature", "min"),
        "max_genes": get("nFeature", "max"),
        "max_counts": get("nCount", "max"),
        "max_mt_pct": get("percent.mt", "max"),
    }
