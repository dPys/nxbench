import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from nxbench.benchmarking.config import BenchmarkResult
from nxbench.data.db import BenchmarkDB
from nxbench.data.loader import BenchmarkDataManager

logger = logging.getLogger("nxbench")


class ResultsExporter:
    """Handle loading, processing, and exporting of benchmark results."""

    def __init__(self, results_file: Path, config: dict | None = None):
        """
        Parameters
        ----------
        results_file : Path
            Path to the benchmark results file (JSON or CSV).
        config : dict, optional
            Configuration dictionary. For SQL export, this should contain the key
            "db_conn_str" (e.g., "dbname=prefect_db user=prefect_user password=pass
            host=localhost").
            If not provided, a default connection string is used.
        """
        self.results_file = results_file
        self.config = config or {}
        self.data_manager = BenchmarkDataManager()
        self._cached_results: list[BenchmarkResult] | None = None

    def _safely_parse_entry(self, entry: dict[str, Any]) -> BenchmarkResult | None:
        try:
            return self._create_benchmark_result_from_entry(entry)
        except Exception:
            logger.exception("Skipping one entry due to error")
            return None

    def load_results(self) -> list[BenchmarkResult]:
        if self._cached_results is not None:
            return self._cached_results

        results = []
        try:
            suffix = self.results_file.suffix.lower()
            if suffix == ".json":
                with self.results_file.open("r") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    logger.error(
                        f"Expected a list of results in JSON file, got {type(data)}"
                    )
                    return []
                for entry in data:
                    result = self._safely_parse_entry(entry)
                    if result:
                        results.append(result)
            elif suffix == ".csv":
                df = pd.read_csv(self.results_file)
                for _, row in df.iterrows():
                    entry = row.to_dict()
                    result = self._safely_parse_entry(entry)
                    if result:
                        results.append(result)
        except Exception:
            logger.exception(f"Failed to load results from: {self.results_file}")
            return []

        self._cached_results = results
        logger.info(f"Loaded {len(results)} benchmark results from {self.results_file}")
        return results

    def _create_benchmark_result_from_entry(
        self, entry: dict[str, Any]
    ) -> BenchmarkResult:
        known_fields = {
            "algorithm",
            "dataset",
            "execution_time",
            "execution_time_with_preloading",
            "memory_used",
            "num_nodes",
            "num_edges",
            "is_directed",
            "is_weighted",
            "backend",
            "num_thread",
            "date",
            "validation",
            "validation_message",
            "error",
        }

        def as_float(value, default=float("nan")):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def as_int(value, default=0):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        algorithm = entry.get("algorithm", "unknown")
        dataset = entry.get("dataset", "unknown")
        backend = entry.get("backend", "unknown")
        execution_time = as_float(entry.get("execution_time"))
        execution_time_with_preloading = as_float(
            entry.get("execution_time_with_preloading")
        )
        memory_used = as_float(entry.get("memory_used"))
        num_nodes = as_int(entry.get("num_nodes"))
        num_edges = as_int(entry.get("num_edges"))
        is_directed = bool(entry.get("is_directed", False))
        is_weighted = bool(entry.get("is_weighted", False))
        num_thread = as_int(entry.get("num_thread"), default=1)
        date = as_int(entry.get("date"), default=0)
        validation = entry.get("validation", "unknown")
        validation_message = entry.get("validation_message", "")
        error = entry.get("error")
        metadata = {k: v for k, v in entry.items() if k not in known_fields}

        return BenchmarkResult(
            algorithm=algorithm,
            dataset=dataset,
            execution_time=execution_time,
            execution_time_with_preloading=execution_time_with_preloading,
            memory_used=memory_used,
            num_nodes=num_nodes,
            num_edges=num_edges,
            is_directed=is_directed,
            is_weighted=is_weighted,
            backend=backend,
            num_thread=num_thread,
            date=date,
            metadata=metadata,
            validation=validation,
            validation_message=validation_message,
            error=error,
        )

    def to_dataframe(self) -> pd.DataFrame:
        results = self.load_results()
        if not results:
            raise ValueError("No benchmark results found")

        records = []
        all_metadata_keys = set()
        for result in results:
            record = {
                "algorithm": result.algorithm,
                "dataset": result.dataset,
                "backend": result.backend,
                "execution_time": result.execution_time,
                "execution_time_with_preloading": result.execution_time_with_preloading,
                "memory_used": result.memory_used,
                "num_nodes": result.num_nodes,
                "num_edges": result.num_edges,
                "is_directed": result.is_directed,
                "is_weighted": result.is_weighted,
                "num_thread": result.num_thread,
                "date": result.date,
                "validation": result.validation,
                "validation_message": result.validation_message,
                "error": result.error,
            }
            for k, v in result.metadata.items():
                record[k] = v
                all_metadata_keys.add(k)
            records.append(record)
        df = pd.DataFrame(records)
        for mk in all_metadata_keys:
            if mk not in df.columns:
                df[mk] = pd.NA
        return df

    def export_results(
        self,
        output_path: Path | None = None,
        form: str = "csv",
        if_exists: str = "replace",
    ) -> None:
        df = self.to_dataframe()
        if form == "csv":
            if output_path is None:
                raise ValueError("An output file must be specified for CSV export")
            df.to_csv(output_path, index=False)
            logger.info(f"Exported results to CSV: {output_path}")
        elif form == "json":
            if output_path is None:
                raise ValueError("An output file must be specified for JSON export")
            df.to_json(output_path, orient="records", indent=2)
            logger.info(f"Exported results to JSON: {output_path}")
        elif form == "sql":
            conn_str = self.config.get(
                "db_conn_str",
                "dbname=prefect_db user=prefect_user password=pass host=localhost",
            )
            db = BenchmarkDB(conn_str=conn_str)
            if if_exists == "replace":
                db.delete_results()
            results = self.load_results()
            db.save_results(results=results, machine_info={}, package_versions={})
            logger.info(
                f"Exported results to PostgreSQL database using connection string: "
                f"{conn_str}"
            )
        else:
            raise ValueError(f"Unsupported export format: {form}")

    def query_results(
        self,
        algorithm: str | None = None,
        backend: str | None = None,
        dataset: str | None = None,
        date_range: tuple[str, str] | None = None,
    ) -> pd.DataFrame:
        df = self.to_dataframe()
        if algorithm:
            df = df[df["algorithm"] == algorithm]
        if backend:
            df = df[df["backend"] == backend]
        if dataset:
            df = df[df["dataset"] == dataset]
        if date_range:
            start_date, end_date = date_range
            df = df[
                (df["date"] >= pd.to_datetime(start_date))
                & (df["date"] <= pd.to_datetime(end_date))
            ]
        return df.sort_values(["algorithm", "dataset", "backend"])
