import sqlite3
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nxbench.benchmarking.config import BenchmarkResult

warnings.filterwarnings("ignore")

SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    backend TEXT NOT NULL,
    dataset TEXT NOT NULL,
    timing REAL NOT NULL,
    num_nodes INTEGER NOT NULL,
    num_edges INTEGER NOT NULL,
    directed INTEGER NOT NULL,
    weighted INTEGER NOT NULL,
    parameters TEXT,
    error TEXT,
    memory_usage REAL,
    git_commit TEXT,
    machine_info TEXT,
    python_version TEXT,
    package_versions TEXT
);

CREATE INDEX IF NOT EXISTS idx_algorithm ON benchmarks(algorithm);
CREATE INDEX IF NOT EXISTS idx_backend ON benchmarks(backend);
CREATE INDEX IF NOT EXISTS idx_dataset ON benchmarks(dataset);
CREATE INDEX IF NOT EXISTS idx_timestamp ON benchmarks(timestamp);
"""


class BenchmarkDB:
    """Database interface for storing and querying benchmark results."""

    def __init__(self, db_path: str | Path | None = None):
        """Initialize the database connection.

        Parameters
        ----------
        db_path : str or Path, optional
            Path to SQLite database file. If None, uses default location
        """
        if db_path is None:
            db_path = Path.home() / ".nxbench" / "benchmarks.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._connection() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def save_results(
        self,
        results: BenchmarkResult | list[BenchmarkResult],
        git_commit: str | None = None,
        machine_info: dict | None = None,
        python_version: str | None = None,
        package_versions: dict | None = None,
    ) -> None:
        """Save benchmark results to database.

        Parameters
        ----------
        results : BenchmarkResult or list of BenchmarkResult
            Results to save
        git_commit : str, optional
            Git commit hash for version tracking
        machine_info : dict, optional
            System information
        python_version : str, optional
            Python version used
        package_versions : dict, optional
            Versions of key packages
        """
        valid_columns = {
            "id",
            "timestamp",
            "algorithm",
            "backend",
            "dataset",
            "timing",
            "num_nodes",
            "num_edges",
            "directed",
            "weighted",
            "parameters",
            "error",
            "memory_usage",
            "git_commit",
            "machine_info",
            "python_version",
            "package_versions",
        }

        if isinstance(results, BenchmarkResult):
            results = [results]

        with self._connection() as conn:
            for result in results:
                result_dict = asdict(result)

                result_dict["timing"] = result_dict.pop("execution_time")
                result_dict["memory_usage"] = result_dict.pop("memory_used")

                result_dict["directed"] = int(result_dict.pop("is_directed"))
                result_dict["weighted"] = int(result_dict.pop("is_weighted"))

                result_dict.update(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "git_commit": git_commit,
                        "machine_info": str(machine_info) if machine_info else None,
                        "python_version": python_version,
                        "package_versions": (
                            str(package_versions) if package_versions else None
                        ),
                    }
                )

                filtered_dict = {
                    k: v for k, v in result_dict.items() if k in valid_columns
                }

                if not filtered_dict:
                    continue

                columns = list(filtered_dict.keys())

                query = "INSERT INTO benchmarks ("
                query += ",".join(f'"{col}"' for col in columns)
                query += ") VALUES ("
                query += ",".join("?" for _ in columns)
                query += ")"

                conn.execute(query, list(filtered_dict.values()))
            conn.commit()

    def get_results(
        self,
        algorithm: str | None = None,
        backend: str | None = None,
        dataset: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        as_pandas: bool = True,
    ) -> pd.DataFrame | list[dict]:
        """Query benchmark results with optional filters.

        Parameters
        ----------
        algorithm : str, optional
            Filter by algorithm name
        backend : str, optional
            Filter by backend
        dataset : str, optional
            Filter by dataset
        start_date : str, optional
            Filter results after this date (ISO format)
        end_date : str, optional
            Filter results before this date (ISO format)
        as_pandas : bool, default=True
            Return results as pandas DataFrame

        Returns
        -------
        DataFrame or list of dict
            Filtered benchmark results
        """
        query = "SELECT * FROM benchmarks WHERE 1=1"
        params = []

        if algorithm:
            query += " AND algorithm = ?"
            params.append(algorithm)
        if backend:
            query += " AND backend = ?"
            params.append(backend)
        if dataset:
            query += " AND dataset = ?"
            params.append(dataset)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        with self._connection() as conn:
            if as_pandas:
                return pd.read_sql_query(query, conn, params=params)

            cursor = conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_unique_values(self, column: str) -> list[str]:
        """Get unique values for a given column."""
        valid_columns = {
            "id",
            "timestamp",
            "algorithm",
            "backend",
            "dataset",
            "timing",
            "num_nodes",
            "num_edges",
            "directed",
            "weighted",
            "parameters",
            "error",
            "memory_usage",
            "git_commit",
            "machine_info",
            "python_version",
            "package_versions",
        }

        if column not in valid_columns:
            raise ValueError(f"Invalid column name: {column}")

        query = f"SELECT DISTINCT {column} FROM benchmarks"  # noqa: S608
        with self._connection() as conn:
            cursor = conn.execute(query)
            return [row[0] for row in cursor.fetchall()]

    def delete_results(
        self,
        algorithm: str | None = None,
        backend: str | None = None,
        dataset: str | None = None,
        before_date: str | None = None,
    ) -> int:
        """Delete benchmark results matching criteria.

        Parameters
        ----------
        algorithm : str, optional
            Delete results for this algorithm
        backend : str, optional
            Delete results for this backend
        dataset : str, optional
            Delete results for this dataset
        before_date : str, optional
            Delete results before this date

        Returns
        -------
        int
            Number of records deleted
        """
        query = "DELETE FROM benchmarks WHERE 1=1"
        params = []

        if algorithm:
            query += " AND algorithm = ?"
            params.append(algorithm)
        if backend:
            query += " AND backend = ?"
            params.append(backend)
        if dataset:
            query += " AND dataset = ?"
            params.append(dataset)
        if before_date:
            query += " AND timestamp < ?"
            params.append(before_date)

        with self._connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount
