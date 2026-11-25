"""Acquire an attributed, bounded subset of the official Amazon challenge data."""

from __future__ import annotations

import argparse
import codecs
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


REGISTRY = "https://registry.opendata.aws/amazon-last-mile-challenges/"
LICENSE = "https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt"
BASE = "https://amazon-last-mile-challenges.s3.us-west-2.amazonaws.com/almrrc2021/"
EVALUATION = BASE + "almrrc2021-data-evaluation/"
URLS = {
    "route": EVALUATION + "model_apply_inputs/eval_route_data.json",
    "packages": EVALUATION + "model_apply_inputs/eval_package_data.json",
    "travel_seconds": EVALUATION + "model_apply_inputs/eval_travel_times.json",
    "actual_sequence": EVALUATION + "model_score_inputs/eval_actual_sequences.json",
}
CHUNK = 256 * 1024
MAX_PREFIX_BYTES = 128 * 1024 * 1024


class _ObjectPrefix:
    """Read complete route-keyed JSON values without loading the full S3 object."""

    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()
        self.bytes = 0
        self.utf8 = codecs.getincrementaldecoder("utf-8")()
        self.parser = json.JSONDecoder(parse_constant=lambda value: None)
        self.buffer = ""
        self.ended = False

    def _more(self) -> bool:
        if self.ended:
            return False
        data = self.stream.read(CHUNK)
        self.ended = not data
        self.hash.update(data)
        self.bytes += len(data)
        self.buffer += self.utf8.decode(data, final=self.ended)
        return bool(data)

    def _leading(self) -> str:
        self.buffer = self.buffer.lstrip()
        while not self.buffer:
            if not self._more():
                raise ValueError("source object or requested prefix ended mid-object")
            self.buffer = self.buffer.lstrip()
        return self.buffer[0]

    def _take(self, symbol: str) -> None:
        if self._leading() != symbol:
            raise ValueError(f"expected {symbol!r} in source object")
        self.buffer = self.buffer[1:]

    def _value(self):
        self._leading()
        while True:
            try:
                value, end = self.parser.raw_decode(self.buffer)
            except json.JSONDecodeError as exc:
                if not self._more():
                    raise ValueError("source record is incomplete") from exc
            else:
                self.buffer = self.buffer[end:]
                return value

    def records(self):
        self._take("{")
        if self._leading() != "}":
            while True:
                key = self._value()
                self._take(":")
                value = self._value()
                if not isinstance(key, str) or not isinstance(value, dict):
                    raise ValueError("expected a route identifier and object-valued record")
                yield key, value
                if self._leading() == "}":
                    break
                self._take(",")
        self._take("}")


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def acquire(output: Path, *, count: int = 40, station_prefix: str = "DLA") -> dict:
    """Select the first matching route IDs from ordered, joined official objects."""
    if type(count) is not int or count <= 0 or not station_prefix or not station_prefix.isascii():
        raise ValueError("select a positive route count and a nonempty ASCII station prefix")
    attributes = {}
    output.mkdir(parents=True, exist_ok=False)
    archive_path = output / "records.jsonl"
    selected = []
    examined = 0
    with ExitStack() as stack:
        sources = {}
        for field, url in URLS.items():
            request = Request(url, headers={"Range": f"bytes=0-{MAX_PREFIX_BYTES - 1}"})
            response = stack.enter_context(urlopen(request, timeout=60))
            content_range = response.headers.get("Content-Range", "")
            if response.status != 206 or not content_range.startswith("bytes 0-"):
                raise ValueError(f"source did not honor the bounded prefix: {field}")
            sources[field] = _ObjectPrefix(response)
            attributes[field] = {
                "url": url, "etag": response.headers.get("ETag"), "content_range": content_range,
                "last_modified": response.headers.get("Last-Modified"),
            }
        with archive_path.open("x", encoding="utf-8", newline="\n") as archive:
            for records in zip(*(reader.records() for reader in sources.values()), strict=True):
                examined += 1
                route_id = records[0][0]
                if any(key != route_id for key, _ in records):
                    raise ValueError(f"official objects are out of route-key order near {route_id}")
                row = dict(zip(URLS, (value for _, value in records), strict=True))
                if not row["route"]["station_code"].startswith(station_prefix):
                    continue
                archive.write(json.dumps({"route_id": route_id, **row}, allow_nan=False) + "\n")
                selected.append(route_id)
                if len(selected) == count:
                    break
        for field, reader in sources.items():
            attributes[field].update({
                "prefix_bytes_read": reader.bytes, "prefix_sha256": reader.hash.hexdigest(),
            })
    if len(selected) != count:
        raise ValueError(f"bounded source prefix yielded {len(selected)} requested routes")
    license_path = output / "LICENSE-AMAZON.txt"
    with urlopen(BASE + "License.txt", timeout=30) as response, license_path.open("xb") as stream:
        stream.write(response.read())
    receipt = {
        "schema": 1,
        "accessed_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "2021 Amazon Last Mile Routing Research Challenge Dataset",
        "registry": REGISTRY,
        "attribution": "Copyright Amazon.com, Inc. or its affiliates. Merchán et al., 2022, DOI 10.1287/trsc.2022.1173.",
        "license": "CC BY-NC 4.0",
        "license_url": LICENSE,
        "license_file_sha256": file_sha256(license_path),
        "sample": {
            "release_split": "official evaluation",
            "station_prefix": station_prefix,
            "route_ids": selected,
            "records_examined_in_official_file_order": examined,
            "selection": "first matching records in joined official order without filtering on outcomes",
            "normalization": "nonstandard JSON NaN literals -> null",
        },
        "objects": attributes,
        "records_sha256": file_sha256(archive_path),
    }
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def read_records(directory: Path) -> list[dict]:
    receipt = json.loads((directory / "manifest.json").read_bytes())
    source = directory / "records.jsonl"
    if file_sha256(source) != receipt["records_sha256"]:
        raise ValueError("official route subset checksum mismatch")
    with source.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if [row["route_id"] for row in rows] != receipt["sample"]["route_ids"]:
        raise ValueError("official route subset differs from its selection manifest")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--station-prefix", default="DLA")
    args = parser.parse_args()
    try:
        receipt = acquire(args.output, count=args.count, station_prefix=args.station_prefix.upper())
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    print(args.output.resolve(), len(receipt["sample"]["route_ids"]), receipt["records_sha256"])


if __name__ == "__main__":
    main()
