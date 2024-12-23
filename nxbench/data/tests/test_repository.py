import importlib
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import aiofiles
import aiohttp
import pytest
import pytest_asyncio
from aiohttp.client_exceptions import ClientResponseError

from nxbench.data import constants
from nxbench.data.repository import NetworkMetadata, NetworkRepository


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
            # Split the body in two chunks
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
async def test_get_data_home_env(monkeypatch, tmp_path):
    monkeypatch.setenv("NXBENCH_HOME", str(tmp_path))
    nr = NetworkRepository()
    assert nr.data_home == tmp_path.resolve()


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
