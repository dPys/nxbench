import asyncio
import hashlib
import json
import logging
import os
import random
import tarfile
import traceback
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import aiofiles
import aiofiles.os
import chardet
from aiohttp import ClientResponse, ClientSession, ClientTimeout, TCPConnector
from aiohttp.client_exceptions import ClientError, ClientResponseError
from aiohttp.client_reqrep import RequestInfo
from bs4 import BeautifulSoup
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from nxbench.data.constants import BASE_URL, COLLECTIONS, HEADERS
from nxbench.data.utils import normalize_name, safe_extract

warnings.filterwarnings("ignore")

logger = logging.getLogger("nxbench")


@dataclass
class NetworkStats:
    """Network statistics that are consistently reported."""

    n_nodes: int
    n_edges: int
    density: float
    max_degree: int
    min_degree: int
    avg_degree: float
    assortativity: float
    n_triangles: int
    avg_triangles: float
    max_triangles: int
    avg_clustering: float
    transitivity: float
    max_kcore: int
    max_clique_lb: int


@dataclass
class NetworkMetadata:
    """Flexible metadata container for network datasets."""

    name: str
    category: str = "Unknown"
    description: str | None = None
    source: str = "Unknown"
    directed: bool = False
    weighted: bool = False

    vertex_type: str | None = "Unknown"
    edge_type: str | None = "Unknown"
    collection: str | None = "Unknown"
    tags: list[str] | None = field(default_factory=list)

    citations: list[str] = field(default_factory=list)

    network_statistics: NetworkStats | None = None
    download_url: str | None = None


class NetworkRepository:
    """Asynchronous interface for downloading and working with networks from the
    networkrepository
    """

    def __init__(
        self,
        data_home: str | Path | None = None,
        scrape_delay: float = 1.0,
        timeout: int = 30,
        max_connections: int = 10,
        max_keepalive_connections: int = 5,
        keepalive_timeout: int = 30,
    ):
        """Initialize dataset loader with optional custom data directory.

        Parameters
        ----------
        data_home : str or Path, optional
            Directory for storing downloaded datasets. If None, defaults to
            ~/nxdata
        scrape_delay : float, default=1.0
            Delay between scraping requests to avoid overloading the server.
        timeout : int, default=10
            Timeout for HTTP requests in seconds.
        max_connections : int, default=1
            Maximum number of concurrent HTTP connections.
        """
        self.data_home = self._get_data_home(data_home)
        self.cache_file = self.data_home / "metadata_cache.json"
        self.metadata_cache = {}
        self.scrape_delay = scrape_delay
        self.timeout = timeout
        self.connector = TCPConnector(
            limit=max_connections,
            limit_per_host=max_connections,
            enable_cleanup_closed=True,
            force_close=False,
            keepalive_timeout=keepalive_timeout,
        )
        self.pool_settings = {
            "max_connections": max_connections,
            "max_keepalive_connections": max_keepalive_connections,
            "keepalive_timeout": keepalive_timeout,
        }
        self.session: ClientSession | None = None
        self.networks_by_category: dict[str, list[str]] = {}

    def _get_data_home(self, data_home: str | Path | None = None) -> Path:
        """Return the path of the dataset directory."""
        if data_home is None:
            data_home = Path(os.environ.get("NXBENCH_HOME", "~/nxbench"))
        else:
            data_home = Path(data_home)

        data_home = data_home.expanduser().resolve()
        data_home.mkdir(parents=True, exist_ok=True)
        return data_home

    async def _load_metadata_cache(self) -> dict[str, NetworkMetadata]:
        if self.cache_file.exists():
            async with aiofiles.open(self.cache_file) as f:
                data = await f.read()
            data = json.loads(data)
            return {k: NetworkMetadata(**v) for k, v in data.items()}
        return {}

    async def _save_metadata_cache(self):
        async with aiofiles.open(self.cache_file, "w") as f:
            serialized = {
                k: self._serialize_metadata(v) for k, v in self.metadata_cache.items()
            }
            await f.write(json.dumps(serialized, indent=4))

    def _serialize_metadata(self, metadata: NetworkMetadata) -> dict[str, Any]:
        """Serialize NetworkMetadata to a dictionary, handling nested dataclasses."""
        data = metadata.__dict__.copy()
        if isinstance(data.get("network_statistics"), NetworkStats):
            data["network_statistics"] = data["network_statistics"].__dict__
        return data

    async def __aenter__(self):
        self.session = ClientSession(
            connector=self.connector,
            timeout=ClientTimeout(total=self.timeout),
            headers=HEADERS,
            trust_env=True,
            raise_for_status=True,
            connector_owner=False,
        )
        self.metadata_cache = await self._load_metadata_cache()
        self.networks_by_category = await self.discover_networks_by_category()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session and not self.session.closed:
            await self.session.close()
        if self.connector and not self.connector.closed:
            await self.connector.close()
        await self._save_metadata_cache()

    async def _fetch_text(
        self, url: str, method: str = "GET", retries: int = 3, **kwargs
    ) -> str | None:
        """Fetch the text content of a URL using aiohttp with retries and robust
        encoding handling.
        """
        if not self.session:
            raise RuntimeError("HTTP session is not initialized.")
        attempt = 0
        while attempt < retries:
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    logger.debug(
                        f"Fetching URL: {url} | Method: {method} | "
                        f"Status: {response.status}"
                    )
                    response.raise_for_status()
                    raw_bytes = await response.read()

                    detected = chardet.detect(raw_bytes)
                    encoding = detected["encoding"]
                    confidence = detected["confidence"]

                    if encoding and confidence > 0.5:
                        text = raw_bytes.decode(encoding, errors="replace")
                        logger.debug(
                            f"Decoded content from {url} using detected encoding: "
                            f"{encoding} (Confidence: {confidence})"
                        )
                    else:
                        text = raw_bytes.decode("utf-8", errors="replace")
                        logger.debug(
                            f"Decoded content from {url} using fallback encoding: "
                            f"utf-8 with replacement"
                        )
                    return text
            except UnicodeDecodeError:
                logger.warning(
                    f"Unicode decode error for {url}. Attempt {attempt + 1}/{retries}"
                )
                if attempt >= retries - 1:
                    logger.exception(
                        f"Failed to decode {url} after {retries} attempts."
                    )
                    logger.exception(traceback.format_exc())
                    raise
            except ClientResponseError:
                logger.warning(
                    f"HTTP response error for {url}. Attempt "
                    f"{attempt + 1}/{retries}"
                )
                if attempt >= retries - 1:
                    logger.exception(f"Failed to fetch {url} after {retries} attempts.")
                    logger.exception(traceback.format_exc())
                    raise
            except ClientError:
                logger.warning(
                    f"HTTP client error for {url}. Attempt {attempt + 1}/{retries}"
                )
                if attempt >= retries - 1:
                    logger.exception(f"Failed to fetch {url} after {retries} attempts.")
                    logger.exception(traceback.format_exc())
                    raise
            except Exception:
                logger.warning(
                    f"Unexpected error for {url}. Attempt {attempt + 1}/{retries}"
                )
                if attempt >= retries - 1:
                    logger.exception(f"Failed to fetch {url} after {retries} attempts.")
                    logger.exception(traceback.format_exc())
                    raise
            attempt += 1
            backoff = min(2**attempt + random.uniform(0, 1), 60)
            logger.debug(f"Retrying after {backoff:.2f} seconds...")
            await asyncio.sleep(backoff)

    async def _fetch_response(
        self, url: str, method: str = "GET", retries: int = 3, **kwargs
    ) -> ClientResponse | None:
        """Fetch the response object of a URL using aiohttp with retries."""
        if not self.session:
            raise RuntimeError("HTTP session is not initialized.")
        attempt = 0
        while attempt < retries:
            try:
                response = await self.session.request(method, url, **kwargs)
                logger.debug(
                    f"Fetching URL: {url} | Method: {method} | "
                    f"Status: {response.status}"
                )
                response.raise_for_status()
            except ClientResponseError:
                logger.warning(
                    f"HTTP response error for {url}. Attempt {attempt + 1}/{retries}"
                )
                if attempt >= retries - 1:
                    logger.exception(f"Failed to fetch {url} after {retries} attempts.")
                    logger.exception(traceback.format_exc())
                    raise
            except ClientError:
                logger.warning(
                    f"HTTP client error for {url}. Attempt {attempt + 1}/{retries}"
                )
                if attempt >= retries - 1:
                    logger.exception(f"Failed to fetch {url} after {retries} attempts.")
                    logger.exception(traceback.format_exc())
                    raise
            except Exception:
                logger.warning(
                    f"Unexpected error for {url}. Attempt {attempt + 1}/{retries}"
                )
                if attempt >= retries - 1:
                    logger.exception(f"Failed to fetch {url} after {retries} attempts.")
                    logger.exception(traceback.format_exc())
                    raise
            else:
                return response
            attempt += 1
            backoff = 2**attempt + random.uniform(0, 1)
            logger.debug(f"Retrying after {backoff:.2f} seconds...")
            await asyncio.sleep(backoff)

    async def _download_file(
        self, url: str, dest: Path, sha256: str | None = None
    ) -> Path:
        """Asynchronously download a file from a URL to a destination path."""
        logger.info(f"Starting download from {url} to {dest}")
        try:
            response = await self._fetch_response(url, method="GET")
            async with aiofiles.open(dest, "wb") as f:
                async for chunk in response.content.iter_chunked(1024 * 64):
                    await f.write(chunk)
            logger.debug(f"Download completed for {dest}")
        except Exception:
            logger.exception(f"Failed to download '{url}'")
            if dest.exists():
                await aiofiles.os.remove(dest)
                logger.debug(f"Removed incomplete download '{dest}'")
            raise

        if sha256:
            logger.info(f"Verifying checksum for {dest}")
            try:
                async with aiofiles.open(dest, "rb") as f:
                    data = await f.read()

                def verify_checksum(
                    data_hash: str, expected: str, filename: str
                ) -> None:
                    """Verify file checksum matches expected value."""
                    if data_hash != expected:
                        logger.error(
                            f"Checksum mismatch for '{filename}'. "
                            f"Expected: {expected}, got: {data_hash}"
                        )

                file_hash = hashlib.sha256(data).hexdigest()
                verify_checksum(file_hash, sha256, dest.name)
                logger.debug(f"Checksum verification passed for {dest}")
            except Exception:
                logger.exception(f"Checksum verification failed for '{dest.name}'")
                raise

        return dest

    async def _extract_file(self, filepath: Path) -> Path:
        """Safely extract compressed files."""
        extracted_path = filepath.with_suffix("")
        try:
            if zipfile.is_zipfile(filepath):
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, safe_extract, filepath, extracted_path)
                logger.info(f"Extracted ZIP file to '{extracted_path}'")
            elif tarfile.is_tarfile(filepath):
                loop = asyncio.get_event_loop()

                def safe_tar_extract():
                    """Safely extract tar file with path checking."""

                    def validate_member(member):
                        """Validate tar member paths."""
                        if member.name.startswith("/") or ".." in member.name:
                            return None
                        return member

                    with tarfile.open(filepath) as tar:
                        valid_members = []
                        invalid_members = []

                        for member in tar.getmembers():
                            if validate_member(member) is not None:
                                valid_members.append(member)
                            else:
                                invalid_members.append(member.name)

                        if invalid_members:
                            logger.warning(
                                f"Skipping unsafe paths in tar: "
                                f"{', '.join(invalid_members)}"
                            )

                        if not valid_members:
                            logger.warning("No valid members found in tar file")
                            return

                        extracted_path.mkdir(parents=True, exist_ok=True)

                        tar.extractall(
                            path=extracted_path,
                            members=valid_members,
                            filter="data",
                        )
                        logger.debug(
                            f"Successfully extracted {len(valid_members)} members"
                        )

                await loop.run_in_executor(None, safe_tar_extract)
                logger.info(f"Extracted TAR file to '{extracted_path}'")
            else:
                logger.warning(f"Unsupported archive format for '{filepath}'")
        except Exception:
            logger.exception(f"Failed to extract '{filepath}'")
            raise
        return extracted_path

    async def _fetch_remote(
        self,
        name: str,
        url: str,
        sha256: str | None = None,
        download_if_missing: bool = True,
    ) -> Path:
        """Download dataset if not present, verify checksum, and handle compressed
        files asynchronously.

        Parameters
        ----------
        name : str
            Name of the dataset.
        url : str
            URL to download the dataset from.
        sha256 : str, optional
            Expected SHA256 checksum of the downloaded file.
        download_if_missing : bool, default=True
            If True, download the dataset if not already present.

        Returns
        -------
        Path
            Path to the downloaded or extracted dataset file.
        """
        filepath = self.data_home / name
        extracted_path = filepath.with_suffix("")

        if not filepath.exists():
            if not download_if_missing:
                raise FileNotFoundError(
                    f"Dataset {name} not found and download_if_missing=False"
                )

            try:
                await self._download_file(url, filepath, sha256)
            except Exception:
                logger.exception(f"Failed to download '{name}' from '{url}'")
                raise

            if zipfile.is_zipfile(filepath) or tarfile.is_tarfile(filepath):
                try:
                    extracted_path = await self._extract_file(filepath)
                except Exception:
                    logger.exception(f"Failed to extract '{name}'")
                    raise
                else:
                    return extracted_path

        return filepath

    async def list_networks(
        self,
        category: str | None = None,
        collection: str | None = None,
        min_nodes: int | None = None,
        max_nodes: int | None = None,
        directed: bool | None = None,
        weighted: bool | None = None,
        limit: int | None = None,
    ) -> list[NetworkMetadata]:
        """List available networks matching specified criteria asynchronously."""
        matching_networks = []
        categories = [category] if category else list(self.networks_by_category.keys())

        for cat in categories:
            networks = self.networks_by_category.get(cat, [])
            logger.debug(f"Listing networks in category '{cat}'")
            for name in networks:
                try:
                    metadata = await self.get_network_metadata(name, cat)
                except Exception:
                    logger.exception(f"Error fetching metadata for network '{name}'")
                    continue

                if directed is not None and metadata.directed != directed:
                    continue
                if weighted is not None and metadata.weighted != weighted:
                    continue
                if collection is not None and metadata.collection != collection:
                    continue
                if metadata.network_statistics is None:
                    continue
                if min_nodes is not None:
                    n_nodes = (
                        metadata.network_statistics.n_nodes
                        if metadata.network_statistics
                        else None
                    )
                    if n_nodes is None or n_nodes < min_nodes:
                        continue
                if max_nodes is not None:
                    n_nodes = (
                        metadata.network_statistics.n_nodes
                        if metadata.network_statistics
                        else None
                    )
                    if n_nodes is None or n_nodes > max_nodes:
                        continue

                matching_networks.append(metadata)
                logger.debug(f"Added network '{name}' to matching list")

                if limit and len(matching_networks) >= limit:
                    logger.info(f"Limit of {limit} networks reached")
                    return matching_networks

        return matching_networks

    async def verify_url(self, url: str) -> bool:
        """Check if the URL is valid and reachable."""
        async with self.session.head(url) as response:
            try:
                is_valid_url = response.status == 200
            except Exception:
                logger.exception(f"Error verifying URL '{url}'")
                return False
            else:
                return is_valid_url

    async def discover_networks_by_category(self) -> dict[str, list[str]]:
        """Asynchronously scrape network names from networkrepository.com for each
        category.
        """
        networks_by_category = {}

        for category in COLLECTIONS:
            php_page = f"{category}.php"
            logger.debug(f"Processing category '{category}' with page '{php_page}'")

            networks_by_category[category] = []

            url = urljoin(BASE_URL, php_page)
            try:
                text = await self._fetch_text(url, method="GET")
                soup = BeautifulSoup(text, "lxml")

                networks = set()
                for a_tag in soup.select("table a[href*='.php']"):
                    href = a_tag.get("href")
                    if not href or not isinstance(href, str):
                        continue
                    href = href.strip()
                    if href.startswith("/"):
                        href = href[1:]
                    if href.endswith(".php"):
                        network_name = href[:-4]  # e.g. "network1.php" => "network1"
                        if network_name not in COLLECTIONS:
                            networks.add(network_name)

                # store sorted network names in the dictionary
                networks_by_category[category] = sorted(networks)

                logger.info(
                    f"Discovered {len(networks)} networks in category '{category}'"
                )
                logger.debug(f"Networks found: {networks}")

                await asyncio.sleep(self.scrape_delay)

            except ClientError:
                logger.exception(f"Error fetching networks for category '{category}'")
            except Exception:
                logger.exception(f"Unexpected error for category '{category}'")

        return networks_by_category

    def get_category_for_network(self, network_name: str) -> str:
        """Get the category for a given network name.

        Parameters
        ----------
        network_name : str
            Name of the network

        Returns
        -------
        str or None
            The category name if found, else None
        """
        for category, networks in self.networks_by_category.items():
            if network_name in networks:
                return category
        return "Unknown"

    async def fetch_with_retry(self, name: str) -> str | None:
        """Attempt to fetch the metadata URL using alternative naming patterns."""
        alternative_patterns = [
            f"{name}.php",
            f"{name.replace('-', '_')}.php",
            f"{name.lower()}.php",
        ]

        for pattern in alternative_patterns:
            url = urljoin(BASE_URL, pattern)
            if await self.verify_url(url):
                logger.debug(f"Alternative pattern matched for '{name}': {url}")
                return url
            logger.debug(f"Alternative pattern did not match for '{name}': {url}")

        logger.error(f"No valid alternative URLs found for network '{name}'.")
        return None

    async def extract_download_url(
        self, soup: BeautifulSoup, name: str, base_url: str = BASE_URL
    ) -> str | None:
        archive_extensions = [
            ".zip",
            ".7z",
            ".tar.gz",
            ".tar.bz2",
            ".mtx",
            ".mtx.gz",
            ".edges",
            ".edges.gz",
            ".txt",
            ".txt.gz",
            ".graph",
            ".graph.gz",
        ]

        all_links = soup.find_all("a", href=True)
        for a_tag in all_links:
            href = a_tag["href"]
            if any(href.lower().endswith(ext) for ext in archive_extensions):
                download_url = urljoin(base_url, href)
                logger.debug(f"Extracted download URL for '{name}': {download_url}")
                return download_url

        logger.warning(f"No download URL found for network '{name}'.")
        return None

    async def get_network_metadata(self, name: str, category: str) -> NetworkMetadata:
        """Asynchronously fetch and parse the metadata for a specific network.

        Parameters
        ----------
        name : str
            Name of the network
        category : str
            Category of the network

        Returns
        -------
        NetworkMetadata
            The metadata object populated with information from the network's page
        """
        if name in self.metadata_cache:
            logger.debug(f"Metadata for '{name}' retrieved from cache")
            return self.metadata_cache[name]

        normalized_name = normalize_name(name)
        url = urljoin(BASE_URL, f"{normalized_name}.php")
        logger.info(f"Fetching metadata from URL: {url}")

        if not await self.verify_url(url):
            logger.warning(
                f"URL '{url}' is not valid. Trying alternative patterns for '{name}'."
            )
            alternative_url = await self.fetch_with_retry(name)
            if alternative_url:
                url = alternative_url
                logger.info(f"Using alternative URL '{url}' for '{name}'.")
            else:
                logger.error(f"URL '{url}' is not valid. Skipping network '{name}'.")
                request_info = RequestInfo(
                    url=URL(url),
                    method="GET",
                    headers=CIMultiDictProxy(CIMultiDict()),
                    real_url=URL(url),
                )
                raise ClientResponseError(
                    request_info=request_info,
                    history=(),
                    status=404,
                    message=f"URL '{url}' is not valid. Skipping network '{name}'.",
                    headers=None,
                )

        try:
            html_content = await self._fetch_text(url, method="GET")
            logger.debug(f"Fetched metadata content for '{name}'")
        except ClientError:
            logger.exception(f"Failed to fetch metadata for '{name}' from '{url}'")
            raise

        soup = BeautifulSoup(html_content, "lxml")

        download_url = await self.extract_download_url(soup, name)
        if not download_url:
            logger.warning(f"No download URL found for network '{name}'.")
            return None

        stats_table = soup.find(
            "table", summary="Network data statistics", id="sortTableExample"
        )
        scraped_net_stats = {}

        if stats_table:
            for row in stats_table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True).replace(":", "")
                    value = cells[1].get_text(strip=True)
                    scraped_net_stats[key] = value
            logger.debug(
                f"Extracted network statistics for '{name}': {scraped_net_stats}"
            )
        else:
            logger.warning(
                f"No network data statistics table found for network '{name}'"
            )

        network_statistics_data = self._parse_network_stats(scraped_net_stats)

        metadata_table = soup.find("table", summary="Dataset metadata")

        if metadata_table is None:
            logger.warning(
                f"No metadata table found for network '{name}' at URL '{url}'. Using "
                f"default metadata."
            )
            metadata = NetworkMetadata(
                name=name,
                category=category,
                description=None,
                directed=False,
                weighted=False,
                vertex_type="Unknown",
                edge_type="Unknown",
                collection=category,
                tags=[],
                source="Unknown",
                citations=[],
                network_statistics=network_statistics_data,
                download_url=download_url,
            )
            self.metadata_cache[name] = metadata
            logger.debug(f"Metadata for '{name}' added to cache with default values")
            return metadata

        data = {}
        fields = [
            "Category",
            "Collection",
            "Tags",
            "Source",
            "Short",
            "Edge type",
            "Edge weights",
            "Description",
            "Name",
            "Vertex type",
            "Format",
        ]

        try:
            for row in metadata_table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue  # skip rows without a label and value pair

                label = cells[0].get_text(strip=True)
                value_cell = cells[1]

                if label in fields:
                    if label == "Tags":
                        data["Tags"] = [
                            tag.get_text(strip=True)
                            for tag in value_cell.find_all("a", class_="tag")
                        ]
                    else:
                        data[label] = value_cell.get_text(strip=True)
            logger.debug(f"Extracted metadata fields for '{name}': {data}")
        except Exception:
            logger.exception(f"Error parsing metadata table for '{name}'")

        citations = []
        acknowledgements_section = soup.find("div", id="collapse_ack")

        if acknowledgements_section:
            for blockquote in acknowledgements_section.find_all("blockquote"):
                citation_text = blockquote.get_text(separator="\n", strip=True)
                citations.append(citation_text)
            logger.debug(f"Extracted citations for '{name}': {citations}")
        else:
            logger.info(f"No acknowledgements section found for network '{name}'.")

        directed = False
        if "Format" in data:
            format_lower = data["Format"].lower()
            if "directed" in format_lower:
                directed = True

        weighted = False
        if "Edge weights" in data:
            edge_weights_lower = data["Edge weights"].lower()
            if "weighted" in edge_weights_lower:
                weighted = True

        metadata = NetworkMetadata(
            name=name,
            category=data.get("Category", category),
            description=data.get("Description"),
            directed=directed,
            weighted=weighted,
            vertex_type=data.get("Vertex type", "Unknown"),
            edge_type=data.get("Edge type", "Unknown"),
            collection=data.get("Collection", category),
            tags=data.get("Tags", []),
            source=data.get("Source", "Unknown"),
            citations=citations,
            network_statistics=network_statistics_data,
            download_url=download_url,
        )

        logger.info(f"Fetched metadata for network '{name}'")

        self.metadata_cache[name] = metadata
        await self._save_metadata_cache()
        logger.debug(f"Metadata for '{name}' saved to cache")

        return metadata

    def _parse_numeric_value(self, value: str) -> int | float | None:
        """Parse a numeric value that may include suffixes like K, M, etc.

        Returns None if the value cannot be parsed into a number.
        """
        if isinstance(value, (int, float)):
            return value
        value = str(value).upper().replace(",", "").strip()

        if value in {"NAN", "INF", "-INF"}:
            return None

        multiplier = 1
        if value.endswith("K"):
            multiplier = 1_000
            value = value[:-1]
        elif value.endswith("M"):
            multiplier = 1_000_000
            value = value[:-1]
        elif value.endswith("B"):
            multiplier = 1_000_000_000
            value = value[:-1]

        try:
            if "." in value:
                return float(value) * multiplier
            return int(value) * multiplier
        except ValueError:
            return None

    def _parse_network_stats(self, stats_dict: dict[str, Any]) -> NetworkStats:
        """Convert a dictionary of stats into a NetworkStats dataclass."""

        def get_numeric(key: str, default: int | float) -> int | float:
            """Parse numeric values with defaults."""
            value = self._parse_numeric_value(stats_dict.get(key, default))
            if value is None:
                return default
            return value

        stats = NetworkStats(
            n_nodes=int(get_numeric("Nodes", 0)),
            n_edges=int(get_numeric("Edges", 0)),
            density=float(get_numeric("Density", 0.0)),
            max_degree=int(get_numeric("Maximum degree", 0)),
            min_degree=int(get_numeric("Minimum degree", 0)),
            avg_degree=float(get_numeric("Average degree", 0.0)),
            assortativity=float(get_numeric("Assortativity", 0.0)),
            n_triangles=int(get_numeric("Number of triangles", 0)),
            avg_triangles=float(get_numeric("Average number of triangles", 0.0)),
            max_triangles=int(get_numeric("Maximum number of triangles", 0)),
            avg_clustering=float(get_numeric("Average clustering coefficient", 0.0)),
            transitivity=float(get_numeric("Fraction of closed triangles", 0.0)),
            max_kcore=int(get_numeric("Maximum k-core", 0)),
            max_clique_lb=int(get_numeric("Lower bound of Maximum Clique", 0)),
        )
        logger.debug(f"Parsed network statistics: {stats}")
        return stats
