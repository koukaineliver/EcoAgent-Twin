"""Acquire the pinned public Southern California OpenStreetMap extract."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import file_digest
import json
from pathlib import Path
from urllib.request import Request, urlopen


SOURCE = "https://download.geofabrik.de/north-america/us/california/socal/socal-190101.osm.pbf"
SIZE = 423_963_662
ETAG = '"19452c0e-57e6d561f9a40"'
CHUNK_BYTES = 16 * 1024 * 1024


def _range(start: int) -> tuple[int, bytes]:
    end = min(SIZE, start + CHUNK_BYTES) - 1
    request = Request(SOURCE, headers={
        "Range": f"bytes={start}-{end}", "If-Match": ETAG, "User-Agent": "EcoAgentTwin research rebuild",
    })
    with urlopen(request, timeout=90) as stream:
        if (
            stream.status != 206 or stream.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}"
            or stream.headers.get("ETag") != ETAG
        ):
            raise ValueError("the archive did not return the requested range from the pinned snapshot")
        contents = stream.read(CHUNK_BYTES + 1)
    if len(contents) != end - start + 1:
        raise ValueError("the archive returned an incomplete range")
    return start, contents


def acquire(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    name = "socal-190101.osm.pbf"
    incomplete = output / f"{name}.part"
    final = output / name
    with incomplete.open("xb") as destination, ThreadPoolExecutor(max_workers=4) as pool:
        for start, contents in pool.map(_range, range(0, SIZE, CHUNK_BYTES)):
            destination.seek(start)
            destination.write(contents)
            print(f"Acquired {(start + len(contents)) / 1_000_000:.1f} of {SIZE / 1_000_000:.1f} MB", flush=True)
    if incomplete.stat().st_size != SIZE:
        raise ValueError("incomplete pinned road extract")
    incomplete.rename(final)
    with final.open("rb") as source:
        fingerprint = file_digest(source, "sha256").hexdigest()
    receipt = {
        "url": SOURCE, "filename": name, "bytes": SIZE, "etag": ETAG, "sha256": fingerprint,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": "OpenStreetMap contributors, ODbL, https://www.openstreetmap.org/copyright",
    }
    with (output / "acquisition.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(acquire(parser.parse_args().output), indent=2))


if __name__ == "__main__":
    main()
