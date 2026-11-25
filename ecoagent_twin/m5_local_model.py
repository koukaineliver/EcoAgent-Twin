"""Score a two-option menu with a compact prompt in both label orders."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from time import perf_counter

from .m3_local_model import LocalMenuModel
from .m3_supervisor import Candidate, DecisionView


ALERT_VALUES = {
    "kind": (
        "locker_outage", "locker_saturation", "charger_outage",
        "road_blockage", "demand_surge", "compound",
    ),
    "severity": ("moderate", "severe"),
    "traffic": ("light_traffic", "heavy_traffic"),
}


def _compact_prompt(alert, first: Candidate, second: Candidate) -> str:
    context = ", ".join(
        alert[field].replace("_", " ")
        for field, allowed in ALERT_VALUES.items()
        if alert.get(field) in allowed
    )
    entries = ["Choose the better plan. Maximize parcels served, then minimize Wh, then recovery minutes."]
    if context:
        entries.append(f"Alert: {context}.")
    for letter, candidate in (("A", first), ("B", second)):
        total = candidate.served + candidate.unserved
        entries.append(
            f"{letter}: {candidate.served}/{total} served, "
            f"{candidate.travel_wh} Wh, {candidate.restricted_recovery_min:g} min."
        )
    entries.append("Answer A or B.")
    return "\n".join(entries)


class OrderInvariantMenuModel(LocalMenuModel):
    """Average each approved candidate's logits when it occupies A and B."""

    def __init__(self, directory: Path, *, threads: int = 2) -> None:
        super().__init__(directory, threads=threads)
        self.last_diagnostics: dict = {}

    def __call__(self, view: DecisionView, *, timeout_s: float) -> dict:
        start = perf_counter()
        self.last_scores = {}
        self.last_diagnostics = {}
        if type(timeout_s) not in (int, float) or not isfinite(timeout_s) or timeout_s <= 0:
            raise TimeoutError("Decision budget expired or is invalid")
        if len(view.candidates) != 2 or len({candidate.id for candidate in view.candidates}) != 2:
            raise ValueError("Order-invariant local scoring requires two distinct approved candidates")
        first, second = sorted(view.candidates, key=lambda candidate: candidate.id)
        prompts = [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": _compact_prompt(view.alert, left, right)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for left, right in ((first, second), (second, first))
        ]
        tokenized = self._tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False,
        )
        token_ids = [
            self._tokenizer.encode(letter, add_special_tokens=False) for letter in ("A", "B")
        ]
        if any(len(ids) != 1 for ids in token_ids) or token_ids[0] == token_ids[1]:
            raise ValueError("Model cannot score A and B as distinct single tokens")
        mask = tokenized["attention_mask"].bool()
        positions = self._torch.arange(mask.size(-1)).expand_as(mask)
        last_positions = positions.masked_fill(~mask, -1).amax(dim=-1)
        with self._torch.inference_mode():
            logits = self._model(**tokenized, use_cache=False).logits
        a_id, b_id = (ids[0] for ids in token_ids)
        ab_a, ab_b = (float(logits[0, last_positions[0], index]) for index in (a_id, b_id))
        ba_a, ba_b = (float(logits[1, last_positions[1], index]) for index in (a_id, b_id))
        if not all(isfinite(score) for score in (ab_a, ab_b, ba_a, ba_b)):
            raise ValueError("Model returned a nonfinite menu score")
        self.last_scores = {
            first.id: (ab_a + ba_b) / 2,
            second.id: (ab_b + ba_a) / 2,
        }
        elapsed = perf_counter() - start
        self.last_diagnostics = {
            "canonical_ids": [first.id, second.id],
            "prompt_tokens": mask.sum(dim=-1).tolist(),
            "canonical_order_scores": {first.id: ab_a, second.id: ab_b},
            "reversed_order_scores": {first.id: ba_b, second.id: ba_a},
            "elapsed_ms": elapsed * 1_000,
        }
        if elapsed >= timeout_s:
            raise TimeoutError("Both-order menu scoring exceeded the decision budget")
        winner = min(self.last_scores, key=lambda candidate_id: (-self.last_scores[candidate_id], candidate_id))
        return {
            "candidate_id": winner,
            "reason": "Highest mean next-token score across both candidate positions.",
        }
