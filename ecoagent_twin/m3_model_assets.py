"""Fetch the pinned public model files for the offline Month 3 supervisor."""

from __future__ import annotations

import argparse
from hashlib import file_digest
import json
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

from .m3_local_model import MODEL_REPO, MODEL_REVISION


SHA256 = {
    "config.json": "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a",
    "generation_config.json": "87b916edaaab66b3899b9d0dd0752727dff6666686da0504d89ae0a6e055a013",
    "model.safetensors": "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c",
    "tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    "tokenizer_config.json": "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df",
    "special_tokens_map.json": "2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52",
    "README.md": "4f97533ad95b1b2fea15fbc075c01b94578ebdd7c8138888fa43fa3abd530dc4",
}


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def fetch(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, expected in SHA256.items():
        path = directory / name
        if not path.exists():
            address = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}?download=true"
            with NamedTemporaryFile(prefix=f".{name}.", suffix=".part", dir=directory, delete=False) as partial:
                temporary = Path(partial.name)
                try:
                    with urlopen(address, timeout=60) as response:
                        copyfileobj(response, partial)
                    partial.flush()
                except OSError:
                    partial.close()
                    temporary.unlink(missing_ok=True)
                    raise
            if _digest(temporary) != expected:
                temporary.unlink()
                raise ValueError(f"download failed fingerprint check: {name}")
            if path.exists():
                temporary.unlink()
            else:
                temporary.rename(path)
        if _digest(path) != expected:
            raise ValueError(f"existing file differs from the pinned revision: {path}")
        files[name] = {"bytes": path.stat().st_size, "sha256": expected}
    manifest = {"repo": MODEL_REPO, "revision": MODEL_REVISION, "files": files}
    receipt = directory / "source.json"
    if receipt.exists():
        if json.loads(receipt.read_bytes()) != manifest:
            raise ValueError("existing local model receipt differs from verified files")
    else:
        with receipt.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("artifacts/models/smollm2-135m-instruct"))
    args = parser.parse_args()
    try:
        result = fetch(args.directory)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(f"Verified {result['repo']} at {result['revision']} in {args.directory.resolve()}")


if __name__ == "__main__":
    main()
