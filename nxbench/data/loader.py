import importlib.resources as importlib_resources
import logging
import os
import warnings
import zipfile
from pathlib import Path
from typing import Any, ClassVar

import aiofiles
import aiohttp
import networkx as nx
import pandas as pd
from scipy.io import mmread

from nxbench.benchmarking.config import DatasetConfig
from nxbench.data.synthesize import generate_graph
from nxbench.data.utils import detect_delimiter, fix_matrix_market_file

warnings.filterwarnings("ignore")

logger = logging.getLogger("nxbench")


class BenchmarkDataManager:
    """Manages loading and caching of networks for benchmarking."""

    SUPPORTED_FORMATS: ClassVar[list[str]] = [".edgelist", ".mtx", ".graphml", ".edges"]

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = (
            Path(data_dir) if data_dir else Path.home() / ".nxbench" / "data"
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._network_cache: dict[str, tuple[nx.Graph | nx.DiGraph, dict[str, Any]]] = (
            {}
        )
        self._metadata_df = self._load_metadata()

    def _normalize_name(self, name: str) -> str:
        return name.lower().strip().replace("-", "_")

    def _load_metadata(self) -> pd.DataFrame:
        try:
            with importlib_resources.open_text(
                "nxbench.data", "network_directory.csv"
            ) as f:
                df = pd.read_csv(f)
                df["name"] = df["name"].apply(self._normalize_name)
                # logger.debug(f"Loaded metadata names: {df['name'].tolist()}")
                return df
        except Exception:
            logger.exception("Failed to load network metadata from package data")
            raise RuntimeError("Failed to load network metadata from package data")

    def get_metadata(self, name: str) -> dict[str, Any]:
        normalized_name = self._normalize_name(name)
        network = self._metadata_df[self._metadata_df["name"] == normalized_name]
        if len(network) == 0:
            raise ValueError(f"Network {name} not found in metadata cache")
        return network.iloc[0].to_dict()

    async def load_network(
        self, config: DatasetConfig, session: aiohttp.ClientSession | None = None
    ) -> tuple[nx.Graph | nx.DiGraph, dict[str, Any]]:
        """Load or generate a network based on config."""
        source_lower = config.source.lower()

        if source_lower == "generator":
            return self._generate_graph(config)

        metadata = self.get_metadata(config.name)

        if config.name in self._network_cache:
            logger.debug(f"Loading network '{config.name}' from cache")
            return self._network_cache[config.name]

        graph_file = None

        for ext in self.SUPPORTED_FORMATS:
            potential_file = self.data_dir / f"{config.name}{ext}"
            if potential_file.exists():
                graph_file = potential_file
                logger.debug(f"Found existing graph file: {graph_file}")
                break

        if graph_file:
            graph = self._load_graph_file(graph_file, metadata)
            self._network_cache[config.name] = (graph, metadata)
            logger.debug(f"Loaded network '{config.name}' from existing file.")
            return graph, metadata

        source_lower = config.source.lower()
        if source_lower == "networkrepository":
            graph, metadata = await self._load_nr_graph(config.name, metadata, session)
        elif source_lower == "local":
            graph, metadata = self._load_local_graph(config)
        elif source_lower == "generator":
            graph, metadata = self._generate_graph(config)
        else:
            raise ValueError(f"Invalid network source: {config.source}")

        self._network_cache[config.name] = (graph, metadata)
        logger.debug(f"Loaded network '{config.name}' successfully")
        return graph, metadata

    def _load_graph_file(
        self, graph_file: Path, metadata: dict[str, Any]
    ) -> nx.Graph | nx.DiGraph:
        def handle_unsupported_format(suffix):
            """Handle unsupported file formats."""
            raise ValueError(f"Unsupported file format: {suffix}")

        def check_graph_validity(graph, file_path):
            """Check if the graph is valid and contains edges."""
            if graph.number_of_edges() == 0:
                raise ValueError(f"Graph file {file_path} contains no valid edges.")

        try:
            suffix = graph_file.suffix.lower()
            if suffix == ".mtx":
                logger.info(f"Loading Matrix Market file from {graph_file}")
                graph_path = Path(graph_file)
                corrected_file = graph_path.with_name(
                    f"{graph_path.stem}_corrected{graph_path.suffix}"
                )

                try:
                    # check if the corrected file already exists
                    if corrected_file.exists():
                        logger.info(
                            f"Using existing corrected Matrix Market file: "
                            f"{corrected_file}"
                        )
                        sparse_matrix = mmread(corrected_file)
                    else:
                        try:
                            # attempt to read the original file
                            sparse_matrix = mmread(graph_file)
                        except Exception:
                            logger.info(f"Fixing Matrix Market file: {graph_file}")
                            # fix the file and load the corrected version
                            corrected_file = fix_matrix_market_file(graph_path)
                            sparse_matrix = mmread(corrected_file)
                except Exception:
                    logger.exception(f"Failed to load Matrix Market file {graph_file}")
                    raise ValueError("Matrix Market file not in expected format")
                else:
                    graph = nx.from_scipy_sparse_array(
                        sparse_matrix,
                        create_using=(
                            nx.DiGraph()
                            if metadata.get("directed", False)
                            else nx.Graph()
                        ),
                    )
                    graph.graph.update(metadata)
                    return graph
            elif suffix in [".edgelist", ".edges"]:
                try:
                    delimiter = detect_delimiter(graph_file)
                    logger.debug(f"Detected delimiter: '{delimiter}'")
                except Exception:
                    logger.debug(
                        "No valid delimiter found, falling back to whitespace split"
                    )
                    delimiter = " "

                create_using = (
                    nx.DiGraph() if metadata.get("directed", False) else nx.Graph()
                )
                weighted = metadata.get("weighted", False)
                logger.info(f"Loading edgelist from {graph_file}")

                has_weights = False
                with graph_file.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith(("#", "%")):
                            continue
                        parts = line.split()
                        if len(parts) >= 3:
                            has_weights = True
                        break

                if has_weights and weighted:
                    logger.debug("Detected weights in the edge list.")
                    try:
                        with graph_file.open("r") as f:

                            def edge_parser():
                                for line in f:
                                    line = line.strip()
                                    if not line or line.startswith(("#", "%")):
                                        continue
                                    parts = line.split()
                                    u, v, w = parts[:3]
                                    yield u, v, float(w)

                            graph = create_using
                            graph.add_weighted_edges_from(edge_parser())
                    except ValueError as ve:
                        logger.warning(
                            f"ValueError while parsing weights from {graph_file}: "
                            f"{ve}. Resuming without weights..."
                        )
                        with graph_file.open("r") as f:
                            edge_iter = (
                                line for line in f if not line.startswith(("#", "%"))
                            )
                            graph = nx.read_edgelist(
                                edge_iter,
                                delimiter=delimiter,
                                nodetype=str,
                                create_using=create_using,
                                data=False,
                            )
                    except Exception:
                        logger.warning(
                            f"Unexpected error while parsing weights from "
                            f"{graph_file}. Resuming without weights..."
                        )
                        with graph_file.open("r") as f:
                            edge_iter = (
                                line for line in f if not line.startswith(("#", "%"))
                            )
                            graph = nx.read_edgelist(
                                edge_iter,
                                delimiter=delimiter,
                                nodetype=str,
                                create_using=create_using,
                                data=False,
                            )
                    check_graph_validity(graph, graph_file)
                else:
                    logger.debug(
                        "No weights detected or weights not required. Reading as "
                        "unweighted."
                    )
                    try:
                        with graph_file.open("r") as f:
                            edge_iter = (
                                line for line in f if not line.startswith(("#", "%"))
                            )
                            graph = nx.read_edgelist(
                                edge_iter,
                                delimiter=delimiter,
                                nodetype=str,
                                create_using=create_using,
                                data=False,
                            )
                    except Exception:
                        logger.exception(
                            f"Failed to read unweighted edgelist from {graph_file}"
                        )
                        raise
                    check_graph_validity(graph, graph_file)
                initial_num_edges = graph.number_of_edges()
                graph.remove_edges_from(nx.selfloop_edges(graph))
                final_num_edges = graph.number_of_edges()
                if initial_num_edges != final_num_edges:
                    logger.debug(
                        f"Removed {initial_num_edges - final_num_edges} self-loop(s) "
                        f"from {graph_file}"
                    )

                if not all(isinstance(node, str) for node in graph.nodes()):
                    logger.debug("Converting node IDs to strings.")
                    mapping = {node: str(node) for node in graph.nodes()}
                    graph = nx.relabel_nodes(graph, mapping)

            elif suffix == ".graphml":
                graph = nx.read_graphml(graph_file)
                check_graph_validity(graph, graph_file)
            else:
                return handle_unsupported_format(suffix)
        except Exception:
            logger.exception(f"Failed to load graph file {graph_file}")
            raise
        else:
            graph.graph.update(metadata)
            if graph.number_of_edges() == 0:
                raise ValueError(f"Graph file {graph_file} contains no valid edges.")
            logger.info(f"Loaded network from '{graph_file}' successfully.")
            return graph

    async def _load_nr_graph(
        self,
        name: str,
        metadata: dict[str, Any],
        session: aiohttp.ClientSession | None = None,
    ) -> nx.Graph | nx.DiGraph:
        for ext in self.SUPPORTED_FORMATS:
            graph_file = self.data_dir / f"{name}{ext}"
            if graph_file.exists():
                return self._load_graph_file(graph_file, metadata)

        url = metadata.get("download_url")
        if not url:
            raise ValueError(f"No download URL found for network {name}")

        logger.info(
            f"Network '{name}' not found in local cache. Attempting to download from "
            f"repository."
        )
        await self._download_and_extract_network(name, url, session)

        for ext in self.SUPPORTED_FORMATS:
            graph_file = self.data_dir / f"{name}{ext}"
            if graph_file.exists():
                return self._load_graph_file(graph_file, metadata)

        logger.error(f"No suitable graph file found after downloading '{name}'")
        raise FileNotFoundError(
            f"No suitable graph file found after downloading '{name}'. Ensure the "
            f"download was successful and the graph file exists."
        )

    async def _download_and_extract_network(
        self, name: str, url: str, session: aiohttp.ClientSession | None = None
    ):
        zip_path = self.data_dir / f"{name}.zip"
        extracted_folder = self.data_dir / f"{name}_extracted"

        if not zip_path.exists():
            logger.info(f"Downloading network '{name}' from {url}")
            await self._download_file(url, zip_path, session)
            logger.info(f"Downloaded network '{name}' to {zip_path}")

        if not extracted_folder.exists():
            logger.info(f"Extracting network '{name}'")
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extracted_folder)
                logger.info(f"Extracted network '{name}' to {extracted_folder}")
            except zipfile.BadZipFile:
                logger.exception(f"Failed to extract zip file {zip_path}")
                raise

        graph_file = self._find_graph_file(extracted_folder)
        if not graph_file:
            logger.error(f"No suitable graph file found after extracting '{name}'")
            logger.error(f"Contents of '{extracted_folder}':")
            for item in extracted_folder.iterdir():
                logger.error(f"- {item.name}")
            raise FileNotFoundError(
                f"No suitable graph file found after extracting '{name}'"
            )

        target_graph_file = self.data_dir / graph_file.name
        if not target_graph_file.exists():
            try:
                graph_file.rename(target_graph_file)
                logger.info(f"Moved graph file to {target_graph_file}")
            except Exception:
                logger.exception(
                    f"Failed to move graph file {graph_file} to {target_graph_file}"
                )
                raise

    async def _download_file(
        self, url: str, dest: Path, session: aiohttp.ClientSession | None = None
    ):
        if session is None:
            session = aiohttp.ClientSession()
        async with session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(
                        f"Failed to download file from {url}. Status code: "
                        f"{response.status}"
                    )
                    raise ConnectionError(
                        f"Failed to download file from {url}. Status code: "
                        f"{response.status}"
                    )
                async with aiofiles.open(dest, "wb") as f:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        await f.write(chunk)
        logger.info(f"Downloaded file from {url} to {dest}")

    def _find_graph_file(self, extracted_folder: Path) -> Path | None:
        """Search for supported graph files within the extracted folder and its
        immediate files.
        """
        for file in extracted_folder.glob("*"):
            if file.suffix in self.SUPPORTED_FORMATS:
                logger.debug(f"Found graph file: {file}")
                return file

        for root, _, files in os.walk(extracted_folder):
            for file in files:
                if file.endswith(tuple(self.SUPPORTED_FORMATS)):
                    graph_file = Path(root) / file
                    logger.debug(f"Found graph file: {graph_file}")
                    return graph_file

        logger.error(f"No suitable graph file found. Contents of {extracted_folder}:")
        for item in extracted_folder.rglob("*"):
            logger.error(f"- {item.relative_to(extracted_folder)}")
        return None

    def _load_local_graph(
        self, config: DatasetConfig
    ) -> tuple[nx.Graph | nx.DiGraph, dict[str, Any]]:
        paths_to_try = [
            Path(config.params["path"]),
            self.data_dir / config.params["path"],
            self.data_dir / Path(config.params["path"]).name,
        ]

        path = None
        for p in paths_to_try:
            if p.exists():
                path = p
                break

        if path is None:
            raise FileNotFoundError(
                f"Network file not found in any location: "
                f"{[str(p) for p in paths_to_try]}"
            )

        graph = self._load_graph_file(path, config.metadata)
        return graph, config.metadata

    def _generate_graph(
        self, config: DatasetConfig
    ) -> tuple[nx.Graph | nx.DiGraph, dict[str, Any]]:
        """Generate a synthetic network using a generator function."""
        generator_name = config.params.get("generator")
        if not generator_name:
            raise ValueError("Generator name must be specified in params.")

        gen_params = config.params.copy()
        gen_params.pop("generator", None)

        directed = config.metadata.get("directed", False)

        try:
            graph = generate_graph(generator_name, gen_params, directed)
        except Exception:
            logger.exception(
                f"Failed to generate graph with generator '{generator_name}'"
            )
            raise

        graph.graph.update(config.metadata)
        return graph, config.metadata

    def load_network_sync(
        self, config: DatasetConfig
    ) -> tuple[nx.Graph | nx.DiGraph, dict[str, Any]]:
        import asyncio

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.load_network(config))
