# Derived from search_res.py
# Source: https://github.com/SpatialHackathon/SACCELERATOR
# Original code licensed under MIT-0


def binary_search(
    adata,
    n_clust_target,
    cluster_fn,
    resolution_boundaries=None,
    resolution_init=1,
    resolution_update=2,
    num_rs=1e2,
    tolerance=1e-3,
    **kwargs,
):
    """
    Uses binary search to find the resolution parameter that results in the target number of clusters.

    Parameters
    ------------
    adata (Anndata)
        AnnData object for clustering. Should have neighbor graphs already.
    target_n_clusters (int)
        The desired number of clusters.
    cluster_fn (callable)
        The clustering function to call, e.g. sc.tl.louvain or rsc.tl.louvain.
        The function name must match the adata.obs column it writes to ("louvain" or "leiden").
    resolution_boundary (list)
        a list defining the boundary for resolution search:
            - If `None`, the function will find a rough boundary that contains the target resolution using `resolution_init` and `resolution_update`;
            - If defined, follows [`left_boundary`, `right_boundary`];
    resolution_init (float)
        initial resolution to start the search with.
    resolution_update(float)
        a positive number for the scale of coarse boundary definition. Only used when `resolution_boundary = None`.
    num_rs (int):
        Highest number of iteration for search.
    tolerance (float):
        Smallest gap between search boundaries. search will stop if the interval is smaller than tolerance.

    Returns
    ------------
    y:
        A dataframe with the clustering results from the selected resolution.
    final_res:
        The selected clustering resolution.
    num_runs:
        The number of times the clustering function was invoked during search.
    """
    import warnings

    method = cluster_fn.__name__
    y = None
    res = resolution_init
    mid = None
    final_res = None
    num_rs = max(1, int(num_rs))
    num_runs = 0

    def do_clustering(res):
        nonlocal num_runs
        num_runs += 1
        cluster_fn(adata, resolution=res)
        y = adata.obs[[method]].astype(int)
        n_clust = y[method].nunique()
        return y, n_clust

    lb = rb = None
    n_clust = -1
    if resolution_boundaries is not None:
        lb, rb = resolution_boundaries
        if lb > rb:
            lb, rb = rb, lb
    else:
        y, n_clust = do_clustering(res)
        final_res = res
        # coarse search for the boundary containing n_clust_target
        if n_clust > n_clust_target:
            coarse_i = 0
            while n_clust > n_clust_target and res > 1e-4 and coarse_i < num_rs:
                rb = res
                res /= resolution_update
                y, n_clust = do_clustering(res)
                final_res = res
                coarse_i += 1
            lb = res
        elif n_clust < n_clust_target:
            coarse_i = 0
            while n_clust < n_clust_target and coarse_i < num_rs:
                lb = res
                res *= resolution_update
                y, n_clust = do_clustering(res)
                final_res = res
                coarse_i += 1
            rb = res
        if n_clust == n_clust_target:
            lb = rb = res

    if lb is None or rb is None:
        lb = rb = res

    i = 0
    while (rb - lb > tolerance or lb == rb) and i < num_rs:
        mid = (lb * rb) ** 0.5
        y, n_clust = do_clustering(mid)
        final_res = mid
        if n_clust == n_clust_target or lb == rb:
            break
        if n_clust > n_clust_target:
            rb = mid
        else:
            lb = mid
        i += 1

    # Check if the situation is met
    if n_clust != n_clust_target:
        warnings.warn(
            f"Warning: n_clust = {n_clust_target} not found in binary search, \
        return best proximation with res = {final_res} and \
        n_clust = {n_clust}. (rb = {rb}, lb = {lb}, i = {i})"
        )

    return y, final_res, num_runs
