import logging

import pandas as pd

logger = logging.getLogger("nxbench")


def load_data(file_path="results/results.csv") -> pd.DataFrame:
    """Load the raw CSV data into a Pandas DataFrame."""
    df = pd.read_csv(file_path, dtype=str)
    # hack to handle the deprecated "iteritems" rename in recent Pandas versions
    pd.DataFrame.iteritems = pd.DataFrame.items
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform the raw DataFrame"""
    essential_columns = ["algorithm", "execution_time", "memory_used"]
    if "execution_time_with_preloading" not in df.columns:
        df["execution_time_with_preloading"] = df["execution_time"]
    df = df.dropna(subset=essential_columns)

    df["execution_time"] = pd.to_numeric(df["execution_time"], errors="coerce")
    df["execution_time_with_preloading"] = pd.to_numeric(
        df["execution_time_with_preloading"], errors="coerce"
    )
    df["memory_used"] = pd.to_numeric(df["memory_used"], errors="coerce")
    df["num_nodes"] = pd.to_numeric(df["num_nodes"], errors="coerce")
    df["num_edges"] = pd.to_numeric(df["num_edges"], errors="coerce")
    df["num_thread"] = pd.to_numeric(df["num_thread"], errors="coerce")

    df["execution_time_with_preloading"] = df["execution_time_with_preloading"].fillna(
        df["execution_time"]
    )

    df = df.dropna(
        subset=[
            "algorithm",
            "execution_time",
            "execution_time_with_preloading",
            "memory_used",
        ]
    )

    string_columns = [
        "algorithm",
        "dataset",
        "backend",
        "is_directed",
        "is_weighted",
        "python_version",
        "backend_version",
        "cpu",
        "os",
    ]
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    unique_n_nodes = df["num_nodes"].nunique(dropna=True)
    if unique_n_nodes > 1:
        num_nodes_binned = pd.cut(df["num_nodes"], bins=min(unique_n_nodes, 4))

        node_labels = []
        for interval in num_nodes_binned.cat.categories:
            lower = int(interval.left) if pd.notnull(interval.left) else float("-inf")
            upper = int(interval.right) if pd.notnull(interval.right) else float("inf")
            node_labels.append(f"{lower} <= x < {upper}")

        node_label_map = dict(zip(num_nodes_binned.cat.categories, node_labels))
        df["num_nodes_bin"] = num_nodes_binned.replace(node_label_map)
    else:
        df["num_nodes_bin"] = df["num_nodes"]

    df["num_nodes_bin"] = (
        df["num_nodes_bin"].astype("category").cat.remove_unused_categories()
    )

    unique_n_edges = df["num_edges"].nunique(dropna=True)
    if unique_n_edges > 1:
        num_edges_binned = pd.cut(df["num_edges"], bins=min(unique_n_edges, 4))

        edge_labels = []
        for interval in num_edges_binned.cat.categories:
            lower = int(interval.left) if pd.notnull(interval.left) else float("-inf")
            upper = int(interval.right) if pd.notnull(interval.right) else float("inf")
            edge_labels.append(f"{lower} <= x < {upper}")

        edge_label_map = dict(zip(num_edges_binned.cat.categories, edge_labels))
        df["num_edges_bin"] = num_edges_binned.replace(edge_label_map)
    else:
        df["num_edges_bin"] = df["num_edges"]

    df["num_edges_bin"] = (
        df["num_edges_bin"].astype("category").cat.remove_unused_categories()
    )
    return df


def aggregate_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    """Aggregate the preprocessed DataFrame by computing the mean execution time,
    mean memory usage, etc., grouping by relevant columns.
    """
    group_columns = [
        "algorithm",
        "dataset",
        "backend",
        "num_nodes_bin",
        "num_edges_bin",
        "is_directed",
        "is_weighted",
        "python_version",
        "cpu",
        "os",
        "num_thread",
    ]

    if "backend_version" in df.columns:
        df["backend_version"] = df["backend_version"].apply(
            lambda x: (
                x.split("==")[1] if isinstance(x, str) and "==" in x else "unknown"
            )
        )
        df["backend_full"] = df.apply(
            lambda row: (
                f"{row['backend']} ({row['backend_version']})"
                if row["backend_version"] != "unknown"
                else row["backend"]
            ),
            axis=1,
        )
        group_columns = [
            c if c != "backend" else "backend_full"
            for c in group_columns
            if c != "backend_version"
        ]
    else:
        logger.warning("No 'backend_version' column found in the dataframe.")
        group_columns = [c for c in group_columns if c not in ("backend_version",)]

    df_agg = df.groupby(group_columns, as_index=False, observed=True).agg(
        mean_execution_time=("execution_time", "mean"),
        mean_memory_used=("memory_used", "mean"),
        sample_count=("execution_time", "size"),
        mean_preload_execution_time=("execution_time_with_preloading", "mean"),
    )

    df_agg.set_index(group_columns, inplace=True)

    df_index = df_agg.index.to_frame()
    unique_counts = df_index.nunique()
    available_parcats_columns = [
        col for col in group_columns if col != "algorithm" and unique_counts[col] > 1
    ]

    df_agg.reset_index(inplace=True)
    # remove unused categories
    for col in ["num_nodes_bin", "num_edges_bin"]:
        if col in df_agg.columns and pd.api.types.is_categorical_dtype(df_agg[col]):
            df_agg[col] = df_agg[col].cat.remove_unused_categories()

    df_agg.set_index(group_columns, inplace=True)

    return df_agg, group_columns, available_parcats_columns


def load_and_prepare_data(file_path: str, logger: logging.Logger):
    """Orchestrate data loading, preprocessing, and aggregation. Returns the final
    aggregated data, along with any metadata needed for downstream visualization.
    """
    raw_df = load_data(file_path)
    cleaned_df = preprocess_data(raw_df)
    df_agg, group_columns, available_parcats_columns = aggregate_data(cleaned_df)
    return cleaned_df, df_agg, group_columns, available_parcats_columns
