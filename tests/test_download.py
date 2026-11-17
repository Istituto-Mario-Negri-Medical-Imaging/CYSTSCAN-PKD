"""Model weights are checked against the md5 that Zenodo lists for each file."""
from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from cystscan.segmentation import download


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("nnUNet_results/placeholder.txt", "weights")
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self.headers = {"content-length": str(len(payload))}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self._payload), chunk_size):
            yield self._payload[i:i + chunk_size]


@pytest.fixture
def fake_zenodo(monkeypatch):
    payload = _zip_bytes()
    url = "https://example.invalid/nnUNet_n5_weights.zip"

    def use(md5: str):
        monkeypatch.setitem(download.ZENODO_FILES, "n5", (url, md5))
        monkeypatch.setattr(download, "_http_get", lambda u: _FakeResponse(payload))

    use.payload = payload
    return use


def test_every_model_has_url_and_md5():
    assert set(download.ZENODO_FILES) == {"n5", "n10", "n15", "n20", "n5_1C"}
    for url, md5 in download.ZENODO_FILES.values():
        assert url.startswith("https://zenodo.org/records/19097241/files/")
        assert len(md5) == 32 and int(md5, 16) >= 0


def test_checksum_ok(tmp_path, fake_zenodo):
    fake_zenodo(hashlib.md5(fake_zenodo.payload).hexdigest())
    results = download.download_model("n5", tmp_path)
    assert results == tmp_path / "nnUNet_results"
    assert (results / "placeholder.txt").read_text() == "weights"


def test_checksum_mismatch_deletes_and_raises(tmp_path, fake_zenodo):
    fake_zenodo("0" * 32)
    with pytest.raises(RuntimeError, match="checksum mismatch for n5"):
        download.download_model("n5", tmp_path)
    assert not any(tmp_path.glob("*.zip*"))
    assert not (tmp_path / "nnUNet_results").exists()


def test_existing_archive_is_verified(tmp_path, fake_zenodo):
    fake_zenodo(hashlib.md5(fake_zenodo.payload).hexdigest())
    (tmp_path / "nnUNet_n5_weights.zip").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum mismatch for n5"):
        download.download_model("n5", tmp_path)
    assert not (tmp_path / "nnUNet_n5_weights.zip").exists()
