import logging
import zipfile
import aiohttp
import os
import warnings
import importlib.util
import importlib.resources as importlib_resources
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union

from scipy.io import mmread

from nxbench.config import DatasetConfig

warnings.filterwarnings("ignore")

logger = logging.getLogger("nxbench")


class BenchmarkDataManager:
    """Manages loading and caching of networks for benchmarking."""

    SUPPORTED_FORMATS = [
        ".edgelist",
        ".mtx",
        ".graphml",
    ]

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        self.data_dir = (
            Path(data_dir) if data_dir else Path.home() / ".nxbench" / "data"
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._network_cache: Dict[
            str, Tuple[Union[nx.Graph, nx.DiGraph], Dict[str, Any]]
        ] = {}
        self._metadata_df = self._load_metadata()

    def _normalize_name(self, name: str) -> str:
        return name.lower().replace("-", "_")

    def _load_metadata(self) -> pd.DataFrame:
        try:
            with importlib_resources.open_text(
                "nxbench.data", "network_directory.csv"
            ) as f:
                df = pd.read_csv(f)
                df["name"] = df["name"].apply(self._normalize_name)
                logger.debug(f"Loaded metadata names: {df['name'].tolist()}")
                return df
        except Exception as e:
            logger.error("Failed to load network metadata from package data")
            raise RuntimeError(
                "Failed to load network metadata from package data"
            ) from e

    def get_metadata(self, name: str) -> Dict[str, Any]:
        normalized_name = self._normalize_name(name)
        network = self._metadata_df[self._metadata_df["name"] == normalized_name]
        if len(network) == 0:
            raise ValueError(f"Network {name} not found in metadata cache")
        return network.iloc[0].to_dict()

    async def load_network(
        self, config: DatasetConfig
    ) -> Tuple[Union[nx.Graph, nx.DiGraph], Dict[str, Any]]:
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
            graph, metadata = await self._load_networkrepository_graph(
                config.name, metadata
            )
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
        self, graph_file: Path, metadata: Dict[str, Any]
    ) -> Union[nx.Graph, nx.DiGraph]:
        try:
            if graph_file.suffix == ".mtx":
                logger.info(f"Loading Matrix Market file from {graph_file}")
                sparse_matrix = mmread(graph_file)
                graph = nx.from_scipy_sparse_array(
                    sparse_matrix,
                    create_using=(
                        nx.DiGraph() if metadata.get("directed", False) else nx.Graph()
                    ),
                )
            elif graph_file.suffix == ".edgelist":
                create_using = (
                    nx.DiGraph() if metadata.get("directed", False) else nx.Graph()
                )
                weighted = metadata.get("weighted", False)
                logger.info(f"Loading edgelist from {graph_file}")
                if weighted:
                    graph = nx.read_edgelist(
                        graph_file,
                        nodetype=int,
                        create_using=create_using,
                        data=["weight"],
                    )
                else:
                    graph = nx.read_edgelist(
                        graph_file, nodetype=int, create_using=create_using, data=False
                    )
            elif graph_file.suffix == ".graphml":
                logger.info(f"Loading GraphML from {graph_file}")
                graph = nx.read_graphml(graph_file)
            else:
                raise ValueError(f"Unsupported file format: {graph_file.suffix}")
        except Exception as e:
            logger.error(f"Failed to load graph file {graph_file}: {e}")
            raise e

        graph.graph.update(metadata)
        logger.info(f"Loaded network from '{graph_file}' successfully.")
        return graph

    async def _load_networkrepository_graph(
        self, name: str, metadata: Dict[str, Any]
    ) -> Union[nx.Graph, nx.DiGraph]:
        for ext in self.SUPPORTED_FORMATS:
            graph_file = self.data_dir / f"{name}{ext}"
            if graph_file.exists():
                return self._load_graph_file(graph_file, metadata)

        url = metadata.get("download_url")
        if not url:
            raise ValueError(f"No download URL found for network {name}")

        logger.info(
            f"Network '{name}' not found in local cache. Attempting to download from repository."
        )
        await self._download_and_extract_network(name, url)

        for ext in self.SUPPORTED_FORMATS:
            graph_file = self.data_dir / f"{name}{ext}"
            if graph_file.exists():
                return self._load_graph_file(graph_file, metadata)

        logger.error(f"No suitable graph file found after downloading '{name}'")
        raise FileNotFoundError(
            f"No suitable graph file found after downloading '{name}'. Ensure the download was successful and the graph file exists."
        )

    async def _download_and_extract_network(self, name: str, url: str):
        zip_path = self.data_dir / f"{name}.zip"
        extracted_folder = self.data_dir / f"{name}_extracted"

        if not zip_path.exists():
            logger.info(f"Downloading network '{name}' from {url}")
            await self._download_file(url, zip_path)
            logger.info(f"Downloaded network '{name}' to {zip_path}")

        if not extracted_folder.exists():
            logger.info(f"Extracting network '{name}'")
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extracted_folder)
                logger.info(f"Extracted network '{name}' to {extracted_folder}")
            except zipfile.BadZipFile as e:
                logger.error(f"Failed to extract zip file {zip_path}: {e}")
                raise e

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
            except Exception as e:
                logger.error(
                    f"Failed to move graph file {graph_file} to {target_graph_file}: {e}"
                )
                raise e

    async def _download_file(self, url: str, dest: Path):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(
                        f"Failed to download file from {url}. Status code: {response.status}"
                    )
                    raise ConnectionError(
                        f"Failed to download file from {url}. Status code: {response.status}"
                    )
                with open(dest, "wb") as f:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        f.write(chunk)
        logger.info(f"Downloaded file from {url} to {dest}")

    def _find_graph_file(self, extracted_folder: Path) -> Optional[Path]:
        """Search for supported graph files within the extracted folder and its immediate files."""
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
    ) -> Tuple[Union[nx.Graph, nx.DiGraph], Dict[str, Any]]:
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
                f"Network file not found in any location: {[str(p) for p in paths_to_try]}"
            )

        _format = config.params.get("format", path.suffix[1:])
        weighted = config.metadata.get("weighted", False)
        directed = config.metadata.get("directed", False)

        graph_file = path
        return self._load_graph_file(graph_file, config.metadata)

    def _generate_graph(
        self, config: DatasetConfig
    ) -> Tuple[Union[nx.Graph, nx.DiGraph], Dict[str, Any]]:
        """Generate a synthetic network using networkx generator functions."""
        generator_name = config.params.get("generator")
        if not generator_name:
            raise ValueError("Generator name must be specified in params.")

        try:
            module_path, func_name = generator_name.rsplit(".", 1)
            module = importlib.import_module(module_path)
            generator = getattr(module, func_name)
        except Exception as e:
            raise ValueError(f"Invalid generator {generator_name}: {e}")

        gen_params = config.params.copy()
        gen_params.pop("generator", None)

        try:
            graph = generator(**gen_params)
        except Exception as e:
            raise ValueError(
                f"Failed to generate graph with {generator_name} and params {gen_params}: {e}"
            )

        directed = config.metadata.get("directed", False)
        if directed and not graph.is_directed():
            graph = graph.to_directed()
        elif not directed and graph.is_directed():
            graph = graph.to_undirected()

        graph.graph.update(config.metadata)

        return graph, config.metadata

    def load_network_sync(
        self, config: DatasetConfig
    ) -> Tuple[Union[nx.Graph, nx.DiGraph], Dict[str, Any]]:
        import asyncio

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.load_network(config))
