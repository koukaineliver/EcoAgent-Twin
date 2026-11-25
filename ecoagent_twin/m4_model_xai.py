"""Explain one fixed candidate pair's local language-model logit margin."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
import json
from math import factorial, fsum, isclose, isfinite
from pathlib import Path
from time import perf_counter
from typing import Protocol

from .m3_local_model import LocalMenuModel
from .m3_supervisor import Candidate, DecisionView, _freeze
from .m4_xai import _load_case


class _MenuScorer(Protocol):
    last_scores: Mapping[str, float]

    def __call__(self, view: DecisionView, *, timeout_s: float) -> object: ...


def _scores(
    model: _MenuScorer, view: DecisionView, menu_ids: set[str], deadline: float
) -> dict[str, float]:
    remaining = deadline - perf_counter()
    if remaining <= 0:
        raise TimeoutError("The model attribution budget expired before all coalitions were scored")
    model(view, timeout_s=remaining)
    if perf_counter() >= deadline:
        raise TimeoutError("The model attribution exceeded its total timeout")
    scores = getattr(model, "last_scores", None)
    if not isinstance(scores, Mapping) or set(scores) != menu_ids:
        raise ValueError("The model must return one next-token score for every menu ID")
    if any(type(value) not in (int, float) or not isfinite(value) for value in scores.values()):
        raise ValueError("Model next-token scores must be finite numbers")
    return {candidate_id: float(score) for candidate_id, score in scores.items()}


def attribute(
    alert: dict,
    menu: list[dict],
    selected_id: str,
    alternative_id: str,
    model: _MenuScorer,
    *,
    timeout_s: float = 4.0,
) -> dict:
    """Compute exact Shapley values for three paired prompt metric groups.

    The deadline applies to the eight model calls in total. The synchronous
    scorer receives the remaining time and must interrupt its own work to
    return promptly. This function rejects a score returned after the deadline.
    """
    started = perf_counter()
    if type(timeout_s) not in (int, float) or not isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Attribution timeout must be finite and positive")
    if not isinstance(alert, dict) or not isinstance(menu, list) or not all(
        isinstance(entry, dict) for entry in menu
    ):
        raise ValueError("Expected an alert dictionary and a list of candidate dictionaries")
    if not callable(model):
        raise ValueError("The model must be callable")
    if not isinstance(selected_id, str) or not isinstance(alternative_id, str):
        raise ValueError("Both comparison candidate IDs must be strings")
    if selected_id == alternative_id:
        raise ValueError("The selected and alternative candidate IDs must be distinct")

    candidates = tuple(Candidate(**entry) for entry in menu)
    ids = {candidate.id for candidate in candidates}
    compared = {selected_id, alternative_id}
    if len(ids) != len(candidates) or not compared <= ids:
        raise ValueError("Menu IDs must be unique and include both comparison candidates")
    cohorts = {candidate.served + candidate.unserved for candidate in candidates}
    if len(cohorts) != 1:
        raise ValueError("The menu must describe one fixed parcel cohort")
    cohort = next(iter(cohorts))

    frozen_alert = _freeze(alert)
    neutral = {
        "service": {"served": 0, "unserved": cohort},
        "energy": {"travel_wh": 0},
        "recovery": {"restricted_recovery_min": 0},
    }
    groups = tuple(neutral)
    size = len(groups)
    full_coalition = (1 << size) - 1
    deadline = started + timeout_s
    margins = {}
    coalitions = []
    for mask in range(full_coalition + 1):
        retained = [group for bit, group in enumerate(groups) if mask & (1 << bit)]
        masked_fields = {
            field: value
            for group in groups
            if group not in retained
            for field, value in neutral[group].items()
        }
        visible = tuple(
            replace(candidate, **masked_fields) if candidate.id in compared else candidate
            for candidate in candidates
        )
        scores = _scores(model, DecisionView(frozen_alert, visible), ids, deadline)
        margin = scores[selected_id] - scores[alternative_id]
        if not isfinite(margin):
            raise ValueError("The next-token score margin must be finite")
        margins[mask] = margin
        coalitions.append({
            "retained_groups": retained,
            "selected_score": scores[selected_id],
            "alternative_score": scores[alternative_id],
            "margin": margin,
        })

    values = {}
    for bit, group in enumerate(groups):
        singleton = 1 << bit
        values[group] = fsum(
            factorial(mask.bit_count()) * factorial(size - mask.bit_count() - 1)
            / factorial(size) * (margins[mask | singleton] - margins[mask])
            for mask in margins
            if not mask & singleton
        )

    difference = margins[full_coalition] - margins[0]
    attribution_sum = fsum(values.values())
    if not isfinite(difference) or not isclose(attribution_sum, difference, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Shapley efficiency check failed to reconstruct the observed score change")
    return {
        "method": "exact Shapley attribution over three paired prompt metric groups",
        "score_definition": "selected candidate letter's next-token logit minus alternative's",
        "selected_id": selected_id,
        "alternative_id": alternative_id,
        "menu_order": [candidate.id for candidate in candidates],
        "cohort_size": cohort,
        "neutral_values_for_compared_pair": neutral,
        "baseline_margin": margins[0],
        "full_margin": margins[full_coalition],
        "full_minus_baseline": difference,
        "shapley_values": values,
        "shapley_sum": attribution_sum,
        "efficiency_error": attribution_sum - difference,
        "efficiency_verified": True,
        "coalitions": coalitions,
        "elapsed_ms": (perf_counter() - started) * 1_000,
        "interpretation": (
            "Attributions are specific to the menu order, alert, scoring model, and declared "
            "neutral values. They describe the model's prompt scores only. The neutral inputs "
            "are reference values for attribution, and no routing or offer is applied."
        ),
    }


def audit_case(run_dir: Path, case_id: str, model_dir: Path, *, timeout_s: float = 30.0) -> dict:
    """Replay the saved scores, explain their margin, and reverse the two entries."""
    run_dir = run_dir.resolve(strict=True)
    case, source = _load_case(run_dir, case_id)
    decision = case["selections"]["agentic_supervisor"]
    if decision["status"] != "selected":
        raise ValueError("this case used a fallback; use m4_xai to explain the executed decision")
    menu = case["menu"]
    selected = decision["decision_id"]
    if len(menu) != 2 or sum(entry["id"] == selected for entry in menu) != 1:
        raise ValueError("expected a two-candidate menu containing the recorded decision")
    alternative = next(entry["id"] for entry in menu if entry["id"] != selected)
    model = LocalMenuModel(model_dir)
    manifest = json.loads((run_dir / "manifest.json").read_bytes())
    if model.provenance != manifest["model"]:
        raise ValueError("the attribution model differs from the model recorded in the run")

    deadline = perf_counter() + timeout_s
    attribution = attribute(case["alert"], menu, selected, alternative, model, timeout_s=timeout_s)
    original = model.last_scores.copy()
    saved = case["model_scores"]
    if original.keys() != saved.keys() or any(
        not isclose(score, saved[key], rel_tol=0.0, abs_tol=1e-4) for key, score in original.items()
    ) or max(original, key=original.__getitem__) != selected:
        raise ValueError("recomputed full-menu scores differ from the saved decision")

    reversed_candidates = tuple(Candidate(**entry) for entry in reversed(menu))
    reversed_order = [entry.id for entry in reversed_candidates]
    reversed_view = DecisionView(_freeze(case["alert"]), reversed_candidates)
    reversed_scores = _scores(model, reversed_view, set(original), deadline)
    return {
        "case_id": case_id,
        "source": source,
        "model": {key: model.provenance[key] for key in ("repo", "revision")},
        "attribution": attribution,
        "reversed_menu": {
            "original": {"order": [entry["id"] for entry in menu], "decision_id": selected, "scores": original},
            "reversed": {"order": reversed_order,
                         "decision_id": max(reversed_order, key=reversed_scores.__getitem__),
                         "scores": reversed_scores},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/smollm2-135m-instruct"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        receipt = audit_case(args.run, args.case, args.model, timeout_s=args.timeout)
        payload = json.dumps(receipt, ensure_ascii=False, indent=2)
        if args.output is None:
            print(payload)
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload + "\n")
    except (ImportError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
