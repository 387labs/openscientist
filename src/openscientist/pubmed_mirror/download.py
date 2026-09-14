"""Download and md5-verify the NCBI MEDLINE annual baseline."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import requests
from tqdm import tqdm  # type: ignore[import-untyped]

BASELINE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
_FILE_RE = re.compile(r"pubmed\d+n\d+\.xml\.gz")
_MD5_RE = re.compile(r"MD5\([^)]*\)\s*=\s*([0-9a-fA-F]{32})")
_CHUNK = 1 << 20


def list_baseline_files(base_url: str = BASELINE_URL, *, timeout: int = 60) -> list[str]:
    """Return the sorted ``.xml.gz`` file names listed in the baseline index."""
    response = requests.get(base_url, timeout=timeout)
    response.raise_for_status()
    names = sorted(set(_FILE_RE.findall(response.text)))
    if not names:
        raise RuntimeError(f"No baseline files found at {base_url}")
    return names


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - NCBI publishes md5, not our choice
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_md5(url: str, *, timeout: int) -> str | None:
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        return None
    match = _MD5_RE.search(response.text)
    return match.group(1).lower() if match else None


def _stream_to_file(url: str, dest: Path, *, timeout: int) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0)) or None
        tmp = dest.with_suffix(dest.suffix + ".part")
        with (
            tmp.open("wb") as handle,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, leave=False) as bar,
        ):
            for chunk in response.iter_content(chunk_size=_CHUNK):
                handle.write(chunk)
                bar.update(len(chunk))
        tmp.rename(dest)


def download_baseline(
    dest_dir: Path,
    *,
    limit: int | None = None,
    base_url: str = BASELINE_URL,
    timeout: int = 600,
) -> list[Path]:
    """Download baseline files into ``dest_dir``; already-verified files resume."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    names = list_baseline_files(base_url, timeout=timeout)
    if limit is not None:
        names = names[:limit]

    verified: list[Path] = []
    for name in tqdm(names, unit="file", desc="baseline"):
        dest = dest_dir / name
        expected = _expected_md5(f"{base_url}{name}.md5", timeout=timeout)
        if dest.exists() and expected is not None and _md5(dest) == expected:
            verified.append(dest)
            continue

        _stream_to_file(f"{base_url}{name}", dest, timeout=timeout)
        if expected is not None and _md5(dest) != expected:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"md5 mismatch for {name}")
        verified.append(dest)

    return verified
