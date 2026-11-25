"""Score approved supervisor menu entries with an offline language model."""

from __future__ import annotations

from hashlib import file_digest
import json
import math
from pathlib import Path
from string import ascii_uppercase
from time import perf_counter


MODEL_REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"


class LocalMenuModel:
    def __init__(self, directory: Path, *, threads: int = 2) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        directory = directory.resolve(strict=True)
        provenance = json.loads((directory / "source.json").read_bytes())
        if provenance["repo"] != MODEL_REPO or provenance["revision"] != MODEL_REVISION:
            raise ValueError("unexpected local model repository or revision")
        for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
            entry = provenance["files"][name]
            path = directory / name
            with path.open("rb") as stream:
                digest = file_digest(stream, "sha256").hexdigest()
            if digest != entry["sha256"] or path.stat().st_size != entry["bytes"]:
                raise ValueError(f"model artifact fingerprint mismatch: {name}")
        torch.set_num_threads(threads)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        self._model = AutoModelForCausalLM.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False, use_safetensors=True, dtype=torch.float32,
        ).eval()
        self.provenance = provenance
        self.last_scores: dict[str, float] = {}

    def __call__(self, view, *, timeout_s: float) -> dict:
        start = perf_counter()
        self.last_scores = {}
        if timeout_s <= 0:
            raise TimeoutError("decision budget expired")
        visible = view.to_dict()
        entries = visible["candidates"]
        if not 1 <= len(entries) <= len(ascii_uppercase):
            raise ValueError("offline model expects 1 to 26 certified candidates")
        labels = ascii_uppercase[:len(entries)]
        candidates = [f"{letter}: {json.dumps(entry, ensure_ascii=True, sort_keys=True)}"
                      for letter, entry in zip(labels, entries, strict=True)]
        prompt = "\n".join([
            "Choose one certified action after a parcel-locker outage.",
            "First maximize parcels served by deadline. If tied, minimize travel Wh, then recovery minutes.",
            f"Alert: {json.dumps(visible['alert'], ensure_ascii=True, sort_keys=True)}",
            *candidates,
            "Answer with the single letter of the best action.",
        ])
        input_ids = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        )
        token_ids = [self._tokenizer.encode(letter, add_special_tokens=False) for letter in labels]
        if not all(len(ids) == 1 for ids in token_ids) or len({ids[0] for ids in token_ids}) != len(entries):
            raise ValueError("model cannot score the specified menu labels as unique single tokens")
        with self._torch.inference_mode():
            logits = self._model(input_ids, use_cache=False).logits[0, -1]
        scores = [float(logits[ids[0]]) for ids in token_ids]
        if not all(math.isfinite(value) for value in scores):
            raise ValueError("model returned a nonfinite menu score")
        self.last_scores = {entry["id"]: score for entry, score in zip(entries, scores, strict=True)}
        if perf_counter() - start > timeout_s:
            raise TimeoutError("offline menu scoring exceeded the decision budget")
        winner = max(range(len(entries)), key=scores.__getitem__)
        return {"candidate_id": entries[winner]["id"],
                "reason": "Highest model next-token score among the certified menu labels."}
