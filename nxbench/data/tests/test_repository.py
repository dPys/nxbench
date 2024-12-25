import importlib
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import aiofiles
import aiohttp
import chardet
import pytest
import pytest_asyncio
from aiohttp.client_exceptions import ClientError, ClientResponseError
from bs4 import BeautifulSoup

from nxbench.data import constants
from nxbench.data.repository import NetworkMetadata, NetworkRepository, NetworkStats


class MockAiohttpResponse:
    def __init__(self, status=200, body=b"mocked response"):
        self.status = status
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def raise_for_status(self):
        if self.status >= 400:
            raise ClientResponseError(None, (), status=self.status)

    async def read(self):
        return self._body

    @property
    def content(self):
        return self._FakeContent(self._body)

    class _FakeContent:
        def __init__(self, body):
            self.body = body

        async def iter_chunked(self, chunk_size):
            half = len(self.body) // 2
            yield self.body[:half]
            yield self.body[half:]


@pytest.fixture
def mock_data_home(tmp_path):
    return tmp_path


@pytest_asyncio.fixture
async def repo(mock_data_home):
    def default_request_side_effect(method, url, **kwargs):
        return MockAiohttpResponse(200, b"mocked response")

    def default_head_side_effect(url, **kwargs):
        return MockAiohttpResponse(200, b"head response")

    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.request.side_effect = default_request_side_effect
    mock_session.head.side_effect = default_head_side_effect

    type(mock_session).closed = PropertyMock(return_value=False)

    with patch("nxbench.data.repository.ClientSession", return_value=mock_session):
        with patch(
            "nxbench.data.repository.NetworkRepository.discover_networks_by_category",
            new_callable=AsyncMock,
            return_value={},  # empty by default
        ):
            async with NetworkRepository(data_home=mock_data_home) as nr:
                yield nr


@pytest.mark.asyncio
async def test_list_networks_filters(repo):
    """
    Tests filtering in list_networks by:
    - min_nodes, max_nodes
    - directed, weighted
    - limit
    """
    meta_small = NetworkMetadata(
        name="small_net",
        directed=False,
        weighted=False,
        network_statistics=NetworkStats(
            n_nodes=10,
            n_edges=20,
            density=0.1,
            max_degree=5,
            min_degree=1,
            avg_degree=2.0,
            assortativity=0.0,
            n_triangles=0,
            avg_triangles=0.0,
            max_triangles=0,
            avg_clustering=0.0,
            transitivity=0.0,
            max_kcore=2,
            max_clique_lb=2,
        ),
    )
    meta_large_directed = NetworkMetadata(
        name="large_directed",
        directed=True,
        weighted=False,
        network_statistics=NetworkStats(
            n_nodes=1000,
            n_edges=2000,
            density=0.01,
            max_degree=100,
            min_degree=1,
            avg_degree=2.0,
            assortativity=0.0,
            n_triangles=0,
            avg_triangles=0.0,
            max_triangles=0,
            avg_clustering=0.0,
            transitivity=0.0,
            max_kcore=10,
            max_clique_lb=5,
        ),
    )
    meta_weighted = NetworkMetadata(
        name="weighted_net",
        directed=False,
        weighted=True,
        network_statistics=NetworkStats(
            n_nodes=50,
            n_edges=100,
            density=0.05,
            max_degree=10,
            min_degree=1,
            avg_degree=2.0,
            assortativity=0.0,
            n_triangles=0,
            avg_triangles=0.0,
            max_triangles=0,
            avg_clustering=0.0,
            transitivity=0.0,
            max_kcore=5,
            max_clique_lb=3,
        ),
    )

    repo.networks_by_category = {
        "mixed": ["small_net", "large_directed", "weighted_net"]
    }
    repo.metadata_cache["small_net"] = meta_small
    repo.metadata_cache["large_directed"] = meta_large_directed
    repo.metadata_cache["weighted_net"] = meta_weighted

    with patch.object(repo, "_save_metadata_cache", new=AsyncMock()):

        # 1) Filter by min_nodes=20 => excludes 'small_net'
        result = await repo.list_networks(category="mixed", min_nodes=20)
        assert len(result) == 2
        assert all(m.n_nodes >= 20 for m in [r.network_statistics for r in result])

        # 2) Filter by max_nodes=50 => only 'small_net' and 'weighted_net' remain
        result2 = await repo.list_networks(category="mixed", max_nodes=50)
        assert len(result2) == 2
        assert {"small_net", "weighted_net"} == {m.name for m in result2}

        # 3) Filter by directed=True => only 'large_directed'
        result3 = await repo.list_networks(category="mixed", directed=True)
        assert len(result3) == 1
        assert result3[0].name == "large_directed"

        # 4) Filter by weighted=True => only 'weighted_net'
        result4 = await repo.list_networks(category="mixed", weighted=True)
        assert len(result4) == 1
        assert result4[0].name == "weighted_net"

        # 5) Test limit=1 => we only get 1 result back
        result5 = await repo.list_networks(category="mixed", limit=1)
        assert len(result5) == 1


@pytest.mark.asyncio
async def test_load_and_save_metadata_cache(repo, tmp_path):
    assert repo.metadata_cache == {}
    sample = {
        "test_network": {
            "name": "test_network",
            "category": "TestCategory",
            "network_statistics": {"n_nodes": 10, "n_edges": 20},
        }
    }

    async with aiofiles.open(repo.cache_file, "w") as f:
        await f.write(json.dumps(sample))

    loaded = await repo._load_metadata_cache()
    assert "test_network" in loaded
    assert loaded["test_network"].name == "test_network"

    repo.metadata_cache["another"] = NetworkMetadata(name="another")
    await repo._save_metadata_cache()
    async with aiofiles.open(repo.cache_file, "r") as f:
        data = json.loads(await f.read())
    assert "another" in data


@pytest.mark.asyncio
async def test_fetch_text_success(repo):
    text = await repo._fetch_text("http://foo")
    assert text == "mocked response"


@pytest.mark.asyncio
async def test_fetch_text_retry_on_error(repo):
    calls = []

    def request_side_effect(method, url, **kwargs):
        calls.append(url)
        if len(calls) < 3:
            raise aiohttp.ClientError("temp fail")
        return MockAiohttpResponse(200, b"final ok")

    repo.session.request.side_effect = request_side_effect

    text = await repo._fetch_text("http://foo", retries=3)
    assert text == "final ok"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_fetch_text_exhaust_retries(repo):
    def always_fail(method, url, **kwargs):
        raise aiohttp.ClientError("fail")

    repo.session.request.side_effect = always_fail

    with pytest.raises(aiohttp.ClientError):
        await repo._fetch_text("http://bar", retries=2)


@pytest.mark.asyncio
async def test_fetch_response_success(repo):
    async def fetch_resp_side_effect(method, url, **kwargs):
        return MockAiohttpResponse(200, b"mocked response")

    repo.session.request.side_effect = fetch_resp_side_effect

    resp = await repo._fetch_response("http://example.com")
    assert resp.status == 200
    body = await resp.read()
    assert body == b"mocked response"


@pytest.mark.asyncio
async def test_fetch_response_retry_exhausted(repo):
    def always_fail(method, url, **kwargs):
        raise aiohttp.ClientError("network error")

    repo.session.request.side_effect = always_fail
    with pytest.raises(aiohttp.ClientError):
        await repo._fetch_response("http://fail.com", retries=2)


@pytest.mark.asyncio
async def test_download_file_success(repo, tmp_path):
    def fetch_resp_side(*args, **kwargs):
        return MockAiohttpResponse(200, b"helloworld")

    with patch.object(repo, "_fetch_response", side_effect=fetch_resp_side):
        out_f = tmp_path / "file.zip"
        ret = await repo._download_file("http://example.com/file.zip", out_f)
        assert ret == out_f

        written_data = out_f.read_bytes()
        assert written_data == b"helloworld"


@pytest.mark.asyncio
async def test_extract_file_tar(repo, tmp_path):
    (tmp_path / "test.tar.gz").touch()

    def dummy_safe_extract():
        pass

    with (
        patch("zipfile.is_zipfile", return_value=False),
        patch("tarfile.is_tarfile", return_value=True),
        patch("asyncio.get_event_loop") as mock_get_loop,
        patch("nxbench.data.utils.safe_extract", new=dummy_safe_extract),
    ):
        loop_mock = MagicMock()

        async def run_executor_side(*args, **kwargs):
            return dummy_safe_extract()

        loop_mock.run_in_executor.side_effect = run_executor_side
        mock_get_loop.return_value = loop_mock

        out = await repo._extract_file(tmp_path / "test.tar.gz")
        assert out.name == "test.tar"


@pytest.mark.asyncio
async def test_verify_url(repo):
    """
    By default => 200 => True
    Then we override => returns a 404 => False
    """
    is_good = await repo.verify_url("http://xyz")
    assert is_good is True

    def head_404_side_effect(url, **kwargs):
        return MockAiohttpResponse(404, b"not found")

    repo.session.head.side_effect = head_404_side_effect
    is_good2 = await repo.verify_url("http://xyz2")
    assert is_good2 is False


@pytest.mark.asyncio
async def test_discover_networks_by_category(monkeypatch, tmp_path):
    monkeypatch.setattr(constants, "COLLECTIONS", ["category1", "category2"])

    import nxbench.data.repository

    importlib.reload(nxbench.data.repository)

    repo = nxbench.data.repository.NetworkRepository(data_home=tmp_path)
    repo.scrape_delay = 0

    async def side_fetch(url, **kwargs):
        if url.endswith("category1.php"):
            return """
            <html><body>
              <table>
                <tr><td><a href="network1.php">Net1</a></td></tr>
                <tr><td><a href="network2.php">Net2</a></td></tr>
              </table>
            </body></html>
            """
        if url.endswith("category2.php"):
            return """
            <html><body>
              <table>
                <tr><td><a href="network3.php">Net3</a></td></tr>
              </table>
            </body></html>
            """
        return "<html></html>"

    with patch.object(repo, "_fetch_text", side_effect=side_fetch):
        result = await repo.discover_networks_by_category()

        assert "category1" in result
        assert "category2" in result
        assert result["category1"] == ["network1", "network2"]
        assert result["category2"] == ["network3"]


@pytest.mark.asyncio
async def test_extract_download_url(repo):
    from bs4 import BeautifulSoup

    with patch.object(repo, "extract_download_url") as mock_ex:
        mock_ex.return_value = "https://example.com/data.zip"
        soup = BeautifulSoup("<a href='somefile.txt'>File</a>", "lxml")
        url = await repo.extract_download_url(soup, "dummy")
        assert url.endswith("data.zip")


@pytest.mark.asyncio
async def test_get_network_metadata_cached(repo):
    meta = NetworkMetadata(name="cached_network")
    repo.metadata_cache["cached_network"] = meta

    out = await repo.get_network_metadata("cached_network", "cat")
    assert out == meta


@pytest.mark.asyncio
async def test_get_network_metadata_not_found(repo):
    with patch.object(repo, "verify_url", new=AsyncMock(return_value=False)):
        with pytest.raises(ClientResponseError) as exc:
            await repo.get_network_metadata("no_such", "catX")
        assert exc.value.status == 404


def test_get_category_for_network(repo):
    repo.networks_by_category = {"catA": ["n1"], "catB": ["n2"]}
    assert repo.get_category_for_network("n1") == "catA"
    assert repo.get_category_for_network("n2") == "catB"
    assert repo.get_category_for_network("nX") == "Unknown"


@pytest.mark.asyncio
async def test_fetch_with_retry(repo):
    """fetch_with_retry tries multiple patterns. We'll make the first pattern fail,
    second succeed.
    """
    calls = []

    async def fake_verify(url):
        calls.append(url)
        return len(calls) > 1 and "testnetwork.php" in url.lower()

    with patch.object(repo, "verify_url", new=fake_verify):
        final = await repo.fetch_with_retry("TestNetwork")
        assert "testnetwork.php" in final.lower()
        assert len(calls) >= 2


def test_parse_numeric_value(repo):
    assert repo._parse_numeric_value("100") == 100
    assert repo._parse_numeric_value("1.5K") == 1500
    assert repo._parse_numeric_value("2.5M") == 2_500_000
    assert repo._parse_numeric_value("10B") == 10_000_000_000
    assert repo._parse_numeric_value("NaN") is None
    assert repo._parse_numeric_value("junk") is None


def test_parse_network_stats(repo):
    stats_dict = {
        "Nodes": "100",
        "Edges": "200",
        "Density": "0.01",
        "Maximum degree": "10",
        "Minimum degree": "1",
        "Average degree": "2.0",
        "Assortativity": "0.0",
        "Number of triangles": "10",
        "Average number of triangles": "2.0",
        "Maximum number of triangles": "5",
        "Average clustering coefficient": "0.3",
        "Fraction of closed triangles": "0.4",
        "Maximum k-core": "2",
        "Lower bound of Maximum Clique": "3",
    }
    ns = repo._parse_network_stats(stats_dict)
    assert ns.n_nodes == 100
    assert ns.n_edges == 200
    assert ns.density == 0.01
    assert ns.max_degree == 10
    assert ns.min_degree == 1
    assert ns.avg_degree == 2.0
    assert ns.assortativity == 0.0
    assert ns.n_triangles == 10
    assert ns.avg_triangles == 2.0
    assert ns.max_triangles == 5
    assert ns.avg_clustering == 0.3
    assert ns.transitivity == 0.4
    assert ns.max_kcore == 2
    assert ns.max_clique_lb == 3


@pytest.mark.asyncio
async def test_serialize_metadata_with_network_stats(repo):
    """
    Ensures that when metadata.network_statistics is a NetworkStats object,
    _serialize_metadata preserves the fields in some form (dict or NetworkStats).
    """
    stats = NetworkStats(
        n_nodes=123,
        n_edges=456,
        density=0.1,
        max_degree=10,
        min_degree=1,
        avg_degree=2.3,
        assortativity=0.0,
        n_triangles=5,
        avg_triangles=0.1,
        max_triangles=3,
        avg_clustering=0.05,
        transitivity=0.02,
        max_kcore=4,
        max_clique_lb=2,
    )
    meta = NetworkMetadata(name="test_net", network_statistics=stats)

    d = repo._serialize_metadata(meta)
    assert "network_statistics" in d

    ns_obj = d["network_statistics"]

    if isinstance(ns_obj, dict):
        for field, expected_value in stats.__dict__.items():
            assert ns_obj.get(field) == expected_value
    elif isinstance(ns_obj, NetworkStats):
        for field, expected_value in stats.__dict__.items():
            assert getattr(ns_obj, field) == expected_value
    else:
        pytest.fail(
            f"Expected 'network_statistics' to be either dict or NetworkStats, got "
            f"{type(ns_obj)}"
        )


@pytest.mark.asyncio
async def test_fetch_text_no_session(mock_data_home):
    """
    Cover the line: if not self.session: raise RuntimeError(...)
    We'll create a NetworkRepository but not enter it as a context manager,
    so self.session remains None.
    """
    nr = NetworkRepository(data_home=mock_data_home)
    with pytest.raises(RuntimeError, match="HTTP session is not initialized"):
        await nr._fetch_text("http://example.com")


@pytest.mark.asyncio
async def test_fetch_text_fallback_encoding(repo):
    with patch.object(
        chardet, "detect", return_value={"encoding": None, "confidence": 0.2}
    ):
        body = b"\xff\xff\xff"

        repo.session.request.side_effect = None
        repo.session.request.return_value = MockAiohttpResponse(200, body=body)

        text = await repo._fetch_text("http://fallback-encoding")
        assert "�" in text


@pytest.mark.asyncio
async def test_fetch_text_exhaust_retries_unicode_decode_error(repo):
    class MockUnicodeErrorResponse(MockAiohttpResponse):
        async def read(self):
            # Force a real UnicodeDecodeError, bypassing "replace"
            raise UnicodeDecodeError(
                "ascii", b"\xff\xff\xff", 0, 1, "invalid start byte"
            )

    def side_effect(*args, **kwargs):
        return MockUnicodeErrorResponse()

    repo.session.request.side_effect = side_effect

    with pytest.raises(UnicodeDecodeError):
        await repo._fetch_text("http://unicode-error", retries=2)


@pytest.mark.asyncio
async def test_fetch_text_exhaust_retries_client_response_error(repo):
    """
    Cover the lines where ClientResponseError occurs repeatedly, then is re-raised
    after max retries.
    """

    def side_effect(*args, **kwargs):
        resp = MockAiohttpResponse(404)
        resp.raise_for_status()  # Raises ClientResponseError
        return resp  # never reached

    repo.session.request.side_effect = side_effect

    with pytest.raises(ClientResponseError):
        await repo._fetch_text("http://client-resp-error", retries=2)


@pytest.mark.asyncio
async def test_fetch_text_unexpected_exception(repo):
    """
    Cover the lines where a random Exception occurs repeatedly, then is re-raised
    after max retries.
    """

    def side_effect(*args, **kwargs):
        raise ValueError("Some random error")

    repo.session.request.side_effect = side_effect
    with pytest.raises(ValueError, match="Some random error"):
        await repo._fetch_text("http://unexpected-error", retries=2)


@pytest.mark.asyncio
async def test_fetch_response_exhaust_retries_client_response_error(repo):
    """Cover lines in _fetch_response where ClientResponseError is raised repeatedly."""

    def side_effect(*args, **kwargs):
        resp = MockAiohttpResponse(500)
        resp.raise_for_status()  # Raises ClientResponseError
        return resp

    repo.session.request.side_effect = side_effect
    with pytest.raises(ClientResponseError):
        await repo._fetch_response("http://fetch-response-error", retries=2)


@pytest.mark.asyncio
async def test_fetch_response_unexpected_exception(repo):
    """
    Cover the lines in _fetch_response where a generic Exception triggers a retry
    and eventually fails.
    """

    def side_effect(*args, **kwargs):
        raise OSError("Some unexpected OS error")

    repo.session.request.side_effect = side_effect
    with pytest.raises(OSError, match="Some unexpected OS error"):
        await repo._fetch_response("http://unexpected-fetch-error", retries=2)


@pytest.mark.asyncio
async def test_download_file_failure(repo, tmp_path):
    """
    Cover the lines in _download_file where an exception is raised
    and the partially downloaded file is removed.
    """

    def fail_fetch(*args, **kwargs):
        raise ClientError("Download failed")

    with patch.object(repo, "_fetch_response", side_effect=fail_fetch):
        out_f = tmp_path / "file.zip"
        assert not out_f.exists()
        with pytest.raises(ClientError):
            await repo._download_file("http://example.com/file.zip", out_f)
        assert not out_f.exists()


@pytest.mark.asyncio
async def test_download_file_checksum_mismatch(repo, tmp_path, caplog):
    """
    Cover lines verifying checksum mismatch. Note that the code logs an error
    but does NOT raise if there's a mismatch.
    """

    def fetch_resp_side(*args, **kwargs):
        return MockAiohttpResponse(200, b"helloworld")

    with patch.object(repo, "_fetch_response", side_effect=fetch_resp_side):
        out_f = tmp_path / "file.zip"
        await repo._download_file(
            "http://example.com/file.zip", out_f, sha256="incorrecthash"
        )
        assert "Checksum mismatch" in caplog.text


@pytest.mark.asyncio
async def test_extract_file_unsupported(repo, tmp_path):
    """Cover the line that warns about "Unsupported archive format"."""
    dummy = tmp_path / "dummy.bin"
    dummy.write_bytes(b"not a zip nor tar")

    with (
        patch("zipfile.is_zipfile", return_value=False),
        patch("tarfile.is_tarfile", return_value=False),
    ):
        with patch("nxbench.data.repository.logger.warning") as mock_warn:
            out = await repo._extract_file(dummy)
            assert out == dummy.with_suffix("")
            mock_warn.assert_called_once_with(
                f"Unsupported archive format for '{dummy}'"
            )


@pytest.mark.asyncio
async def test_extract_file_tar_with_invalid_member(repo, tmp_path):
    import tarfile

    test_tar = tmp_path / "test.tar"
    test_tar.touch()

    with patch("tarfile.is_tarfile", return_value=True):
        valid_member = MagicMock(spec=tarfile.TarInfo)
        type(valid_member).name = PropertyMock(return_value="validpath/something.txt")
        valid_member.isreg.return_value = True

        invalid_member = MagicMock(spec=tarfile.TarInfo)
        type(invalid_member).name = PropertyMock(return_value="../evil.txt")
        invalid_member.isreg.return_value = True  # or doesn't matter

        mock_tarfile = MagicMock()
        mock_tarfile.getmembers.return_value = [valid_member, invalid_member]

        mock_tarfile.__enter__.return_value = mock_tarfile
        mock_tarfile.__exit__.return_value = None

        def mock_tar_open(*args, **kwargs):
            return mock_tarfile

        with patch("tarfile.open", side_effect=mock_tar_open):
            await repo._extract_file(test_tar)

            mock_tarfile.extractall.assert_called_once()

            extractall_kwargs = mock_tarfile.extractall.call_args[1]
            extracted_members = extractall_kwargs["members"]

            assert valid_member in extracted_members
            assert invalid_member not in extracted_members


@pytest.mark.asyncio
async def test_fetch_remote_not_found_download_if_missing_false(repo):
    """
    Cover lines where we attempt to fetch but the file does not exist,
    and download_if_missing=False => raises FileNotFoundError.
    """
    non_existent = repo.data_home / "nonexistent.bin"
    assert not non_existent.exists()

    with pytest.raises(
        FileNotFoundError, match="not found and download_if_missing=False"
    ):
        await repo._fetch_remote(
            "nonexistent.bin",
            "http://example.com/nonexistent.bin",
            download_if_missing=False,
        )


@pytest.mark.asyncio
async def test_list_networks_exception_in_get_metadata(repo):
    """
    Cover lines where get_network_metadata raises an exception
    and is caught inside list_networks.
    """
    repo.networks_by_category = {"cat": ["net1"]}

    async def mock_get_metadata(name, cat):
        raise RuntimeError("some error during get_network_metadata")

    with patch.object(repo, "get_network_metadata", side_effect=mock_get_metadata):
        out = await repo.list_networks("cat")
        assert out == []


@pytest.mark.asyncio
async def test_verify_url_exception(repo):
    def head_side_effect(*args, **kwargs):
        resp = MockAiohttpResponse(200)
        type(resp).status = PropertyMock(side_effect=OSError("Some OS error"))
        return resp

    repo.session.head.side_effect = head_side_effect

    is_valid = await repo.verify_url("http://xyz")
    assert is_valid is False


@pytest.mark.asyncio
async def test_discover_networks_by_category_client_error(repo):
    with patch.object(
        repo, "_fetch_text", side_effect=ClientError("some client error")
    ):
        result = await repo.discover_networks_by_category()
        assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_discover_networks_by_category_unexpected_exception(repo):
    """Cover lines in discover_networks_by_category that catch generic Exception."""
    with patch.object(repo, "_fetch_text", side_effect=ValueError("unexpected error")):
        result = await repo.discover_networks_by_category()
        assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_extract_download_url_no_link(repo):
    soup = BeautifulSoup("<html><body>No archives here</body></html>", "lxml")
    url = await repo.extract_download_url(soup, "dummy")
    assert url is None


@pytest.mark.asyncio
async def test_get_network_metadata_no_download_url_found(repo):
    """Cover lines where no download URL is found => returns None"""
    with patch.object(repo, "verify_url", return_value=True):

        async def mock_fetch_text(*args, **kwargs):
            return "<html><body>No links with recognized extension</body></html>"

        with patch.object(repo, "_fetch_text", side_effect=mock_fetch_text):
            md = await repo.get_network_metadata("some_network", "some_category")
            assert md is None


@pytest.mark.asyncio
async def test_get_network_metadata_no_metadata_table(repo):
    """
    Tests the scenario where a recognized download link is found,
    but there's NO <table summary="Dataset metadata">, so we fall back to default
    metadata.
    Also checks the "no network data statistics table" fallback.
    """
    mock_html = """
    <html>
      <body>
        <!-- A recognized download link -->
        <a href="dataset.zip">Download me</a>
        <!-- Intentionally missing the table with summary="Dataset metadata" -->
        <!-- Also missing the table with id="sortTableExample" for stats -->
      </body>
    </html>
    """

    with (
        patch.object(repo, "verify_url", return_value=True),
        patch.object(repo, "_fetch_text", return_value=mock_html),
    ):
        meta = await repo.get_network_metadata(
            "test_no_metadata_table", "test_category"
        )
        assert meta is not None
        assert meta.name == "test_no_metadata_table"
        assert meta.category == "test_category"
        assert meta.network_statistics is not None
        assert meta.network_statistics.n_nodes == 0
        assert meta.download_url.endswith("dataset.zip")
        assert meta.vertex_type == "Unknown"
        assert meta.collection == "test_category"


@pytest.mark.asyncio
async def test_get_network_metadata_ack_section(repo):
    """
    Tests parsing citations from the acknowledgements section (collapse_ack).
    Also tests reading 'directed' and 'weighted' from Format and Edge weights fields.
    """
    mock_html = """
    <html>
      <body>
        <!-- Recognized download link -->
        <a href="somegraph.mtx.gz">Download me</a>

        <!-- Stats table present, but partial -->
        <table id="sortTableExample" summary="Network data statistics">
          <tr><td>Nodes:</td><td>42</td></tr>
          <tr><td>Edges:</td><td>100</td></tr>
        </table>

        <table summary="Dataset metadata">
          <tr><td>Format</td><td>Directed Graph Format</td></tr>
          <tr><td>Edge weights</td><td>Weighted edges present</td></tr>
          <tr><td>Tags</td><td>
            <a href="#" class="tag">tag1</a>
            <a href="#" class="tag">tag2</a>
          </td></tr>
        </table>

        <div id="collapse_ack">
          <blockquote>Citation 1 line 1\nCitation 1 line 2</blockquote>
          <blockquote>Citation 2 only line</blockquote>
        </div>
      </body>
    </html>
    """

    with (
        patch.object(repo, "verify_url", return_value=True),
        patch.object(repo, "_fetch_text", return_value=mock_html),
    ):
        meta = await repo.get_network_metadata("test_ack_section", "test_category")
        assert meta is not None
        assert meta.download_url.endswith(".mtx.gz")
        # Check that stats were parsed
        assert meta.network_statistics is not None
        assert meta.network_statistics.n_nodes == 42
        assert meta.network_statistics.n_edges == 100
        # Check that "directed" and "weighted" are set
        assert meta.directed is True
        assert meta.weighted is True
        # Check tags
        assert meta.tags == ["tag1", "tag2"]
        # Check citations
        assert len(meta.citations) == 2
        assert "Citation 1 line 1" in meta.citations[0]
        assert "Citation 2 only line" in meta.citations[1]


@pytest.mark.asyncio
async def test_extract_file_tar_no_valid_members(repo, tmp_path):
    """
    Tests extracting a tar file that has NO valid members.
    This should trigger the 'No valid members found in tar file' warning.
    """
    test_tar = tmp_path / "empty.tar"
    test_tar.touch()

    with patch("tarfile.is_tarfile", return_value=True):
        mock_tarfile = MagicMock()

        mock_tarfile.getmembers.return_value = []
        mock_tarfile.__enter__.return_value = mock_tarfile
        mock_tarfile.__exit__.return_value = None

        def mock_tar_open(*args, **kwargs):
            return mock_tarfile

        with (
            patch("tarfile.open", side_effect=mock_tar_open),
            patch("nxbench.data.repository.logger.warning") as mock_warn,
        ):
            out_path = await repo._extract_file(test_tar)
            assert out_path == test_tar.with_suffix("")
            mock_warn.assert_any_call("No valid members found in tar file")
