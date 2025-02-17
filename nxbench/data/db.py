import warnings
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd
from psycopg2 import connect, sql

from nxbench.benchmarking.config import BenchmarkResult

warnings.filterwarnings("ignore")

SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmarks (
    id SERIAL PRIMARY KEY,
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
    package_versions TEXT
);

CREATE INDEX IF NOT EXISTS idx_algorithm ON benchmarks(algorithm);
CREATE INDEX IF NOT EXISTS idx_backend ON benchmarks(backend);
CREATE INDEX IF NOT EXISTS idx_dataset ON benchmarks(dataset);
CREATE INDEX IF NOT EXISTS idx_timestamp ON benchmarks(timestamp);
"""


class BenchmarkDB:
    """Database interface for storing and querying benchmark results in PostgreSQL."""

    def __init__(self, conn_str: str | None = None):
        """
        Initialize the database connection.

        Parameters
        ----------
        conn_str : str, optional
            PostgreSQL connection string. If None, a default connection string is used.
        """
        if conn_str is None:
            conn_str = (
                "dbname=prefect_db user=prefect_user password=pass host=localhost"
            )
        self.conn_str = conn_str
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA)
            conn.commit()

    def truncate(self) -> None:
        """Completely clear the benchmarks table and reset the serial counter."""
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE benchmarks RESTART IDENTITY CASCADE")
            conn.commit()

    @contextmanager
    def _connection(self):
        """Context manager for PostgreSQL database connections."""
        conn = connect(self.conn_str)
        try:
            yield conn
        finally:
            conn.close()

    def save_results(
        self,
        results: BenchmarkResult | list[BenchmarkResult],
        git_commit: str | None = None,
        machine_info: dict | None = None,
        package_versions: dict | None = None,
    ) -> None:
        """Save benchmark results to the database.

        Parameters
        ----------
        results : BenchmarkResult or list of BenchmarkResult
            Results to save.
        git_commit : str, optional
            Git commit hash for version tracking.
        machine_info : dict, optional
            System information.
        package_versions : dict, optional
            Versions of key packages.
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
            "package_versions",
        }

        if isinstance(results, BenchmarkResult):
            results = [results]

        with self._connection() as conn:
            with conn.cursor() as cur:
                for result in results:
                    result_dict = asdict(result)
                    # Rename fields
                    result_dict["timing"] = result_dict.pop("execution_time")
                    result_dict["memory_usage"] = result_dict.pop("memory_used")
                    # Convert booleans to integers
                    result_dict["directed"] = int(result_dict.pop("is_directed"))
                    result_dict["weighted"] = int(result_dict.pop("is_weighted"))
                    result_dict.update(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "git_commit": git_commit,
                            "machine_info": str(machine_info) if machine_info else None,
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
                    query = sql.SQL(
                        "INSERT INTO benchmarks ({fields}) VALUES ({values})"
                    ).format(
                        fields=sql.SQL(",").join(map(sql.Identifier, columns)),
                        values=sql.SQL(",").join(sql.Placeholder() for _ in columns),
                    )
                    cur.execute(query, list(filtered_dict.values()))
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
            Filter by algorithm name.
        backend : str, optional
            Filter by backend.
        dataset : str, optional
            Filter by dataset.
        start_date : str, optional
            Filter results after this date (ISO format).
        end_date : str, optional
            Filter results before this date (ISO format).
        as_pandas : bool, default=True
            Return results as a pandas DataFrame.

        Returns
        -------
        DataFrame or list of dict
            Filtered benchmark results.
        """
        query = "SELECT * FROM benchmarks WHERE TRUE"
        params = []
        if algorithm:
            query += " AND algorithm = %s"
            params.append(algorithm)
        if backend:
            query += " AND backend = %s"
            params.append(backend)
        if dataset:
            query += " AND dataset = %s"
            params.append(dataset)
        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)
        with self._connection() as conn:
            if as_pandas:
                return pd.read_sql_query(query, conn, params=params)
            with conn.cursor() as cur:
                cur.execute(query, params)
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
            return [dict(zip(columns, row)) for row in rows]

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
            "package_versions",
        }
        if column not in valid_columns:
            raise ValueError(f"Invalid column name: {column}")

        query = f'SELECT DISTINCT "{column}" FROM benchmarks'  # noqa: S608
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                return [row[0] for row in cur.fetchall()]

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
            Delete results for this algorithm.
        backend : str, optional
            Delete results for this backend.
        dataset : str, optional
            Delete results for this dataset.
        before_date : str, optional
            Delete results before this date.

        Returns
        -------
        int
            Number of records deleted.
        """
        query = "DELETE FROM benchmarks WHERE TRUE"
        params = []
        if algorithm:
            query += " AND algorithm = %s"
            params.append(algorithm)
        if backend:
            query += " AND backend = %s"
            params.append(backend)
        if dataset:
            query += " AND dataset = %s"
            params.append(dataset)
        if before_date:
            query += " AND timestamp < %s"
            params.append(before_date)

        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                affected = cur.rowcount
            conn.commit()
        return affected
