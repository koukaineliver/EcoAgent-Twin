"""Replay archived Month 4 menus through the order-invariant local scorer."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from itertools import zip_longest
import json
from math import ceil, isfinite
from pathlib import Path
from statistics import fmean
from time import perf_counter

from .m3_supervisor import Candidate, DecisionView, _freeze
from .m5_local_model import OrderInvariantMenuModel


DEFAULT_MODEL = Path("artifacts/models/smollm2-135m-instruct")


def select_cases(cases: list[dict], limit: int | None) -> list[dict]:
    """Round-robin across chaos and traffic groups for a bounded smoke run."""
    if limit is None or limit >= len(cases):
        return cases
    if limit < 1:
        raise ValueError("The case limit must be positive")
    groups = defaultdict(list)
    for index, row in enumerate(cases):
        groups[row["chaos_kind"], row["traffic"]].append(index)
    indexes = []
    for tier in zip_longest(*groups.values()):
        indexes.extend(index for index in tier if index is not None)
        if len(indexes) >= limit:
            break
    return [cases[index] for index in sorted(indexes[:limit])]


def _score(view: DecisionView, model, timeout_s: float) -> dict:
    started = perf_counter()
    result = model(view, timeout_s=timeout_s)
    elapsed_ms = 1_000 * (perf_counter() - started)
    choice = result["candidate_id"]
    order = [candidate.id for candidate in view.candidates]
    scores = dict(model.last_scores)
    if choice not in order or set(scores) != set(order) or not all(isfinite(score) for score in scores.values()):
        raise ValueError("The backend did not return finite scores and an approved choice")
    return {
        "submitted_order": order,
        "candidate_id": choice,
        "scores": scores,
        "elapsed_ms": elapsed_ms,
        "prompt_tokens": getattr(model, "last_diagnostics", {}).get("prompt_tokens"),
    }


def _rank(candidate: Candidate) -> tuple:
    return -candidate.served, candidate.travel_wh, candidate.restricted_recovery_min


def _latency(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "mean": fmean(ordered),
        "p95": ordered[ceil(0.95 * len(ordered)) - 1],
        "max": ordered[-1],
    }


def evaluate(cases: list[dict], model, *, timeout_s: float = 2.0) -> dict:
    """Return exact order checks and outcome comparisons on the archived menus."""
    if not cases or not isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Provide at least one archived case and a finite positive timeout")
    rows = []
    counts: Counter = Counter()
    status: Counter = Counter()
    totals = Counter()
    for trace in cases:
        case_id = trace["case_id"]
        menu = tuple(Candidate(**row) for row in trace["menu"])
        if len(menu) != 2 or len({candidate.id for candidate in menu}) != 2:
            raise ValueError(f"Expected two distinct candidates for case {case_id}")
        by_id = {candidate.id: candidate for candidate in menu}
        total = menu[0].served + menu[0].unserved
        if total <= 0 or menu[1].served + menu[1].unserved != total:
            raise ValueError(f"Invalid shared demand for case {case_id}")

        alert = _freeze(trace["alert"])
        original = _score(DecisionView(alert, menu), model, timeout_s)
        reversed_order = _score(DecisionView(alert, menu[::-1]), model, timeout_s)
        if any(original[key] != reversed_order[key] for key in ("candidate_id", "scores")):
            raise ValueError(f"Model scores or choice changed under submitted order reversal: {case_id}")

        archived = trace["selections"]["agentic_supervisor"]
        previous = by_id[archived["decision_id"]]
        current = by_id[original["candidate_id"]]
        best = min(menu, key=lambda candidate: (*_rank(candidate), candidate.id))
        status[archived["status"]] += 1
        counts["m5_matches_m4_choice"] += current.id == previous.id
        counts["m5_first_in_archived_order"] += current.id == menu[0].id
        counts["m4_accepted_first_in_archived_order"] += (
            archived["status"] == "selected" and previous.id == menu[0].id
        )
        for stage, candidate in (("m4", previous), ("m5", current)):
            counts[f"{stage}_maximum_service"] += candidate.served == best.served
            counts[f"{stage}_matches_deterministic_id"] += candidate.id == best.id
            counts[f"{stage}_matches_deterministic_metrics"] += _rank(candidate) == _rank(best)
            totals[f"{stage}_served"] += candidate.served
        totals["deterministic_served"] += best.served
        totals["demand"] += total
        rows.append({
            "case_id": case_id,
            "chaos_kind": trace["chaos_kind"],
            "traffic": trace["traffic"],
            "demand": total,
            "archived_m4": {
                "candidate_id": previous.id,
                "status": archived["status"],
                "selector_elapsed_ms": archived["elapsed_ms"],
                "served": previous.served,
            },
            "deterministic": {"candidate_id": best.id, "maximum_served": best.served},
            "m5_served": current.served,
            "m5_in_archived_order": original,
            "m5_in_reversed_order": reversed_order,
        })
    return {
        "case_count": len(rows),
        "exact_order_invariance_cases": len(rows),
        "model_calls": 2 * len(rows),
        "m4_archived_statuses": dict(status),
        "matching_choices": dict(counts),
        "served_counts": dict(totals),
        "m5_model_call_ms": _latency([
            row[key]["elapsed_ms"]
            for row in rows
            for key in ("m5_in_archived_order", "m5_in_reversed_order")
        ]),
        "m4_archived_selector_ms": _latency([row["archived_m4"]["selector_elapsed_ms"] for row in rows]),
        "cases": rows,
    }


def _archived(directory: Path) -> tuple[list[dict], dict]:
    manifest = (directory / "manifest.json").read_bytes()
    document = json.loads(manifest)
    traces = (directory / "traces.jsonl").read_bytes()
    trace_digest = sha256(traces).hexdigest()
    if document["kind"] != "month_4_chaos_experiment" or document["sha256"]["traces.jsonl"] != trace_digest:
        raise ValueError("The archived trace does not match its Month 4 experiment manifest")
    cases = [json.loads(line) for line in traces.splitlines()]
    if not cases or len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("Expected a nonempty archive with unique matched case IDs")
    return cases, {
        "archive": str(directory),
        "manifest_sha256": sha256(manifest).hexdigest(),
        "traces_sha256": trace_digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Saved Month 4 evaluation directory")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Pinned local model directory")
    parser.add_argument("--output", type=Path, required=True, help="New JSON receipt path")
    parser.add_argument("--limit", type=int, help="Run a smaller sample across traffic and chaos groups")
    parser.add_argument("--timeout-s", type=float, default=2.0, help="Timeout for each submitted order")
    args = parser.parse_args()
    try:
        archive = args.archive.resolve(strict=True)
        output = args.output.resolve()
        if output.exists() or output.is_relative_to(archive):
            raise ValueError("The receipt must be a new file outside the archived evaluation directory")
        if not output.parent.is_dir():
            raise ValueError(f"Create the receipt parent directory first: {output.parent}")
        cases, provenance = _archived(archive)
        selected = select_cases(cases, args.limit)
        model = OrderInvariantMenuModel(args.model)
        report = evaluate(selected, model, timeout_s=args.timeout_s)
        report.update({
            "schema": 1,
            "scope": "Retrospective score and choice audit on saved Month 4 menus",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "archived_case_count": len(cases),
            "subset_selection": "round-robin by chaos kind and traffic" if len(selected) < len(cases) else "all",
            "timeout_seconds_per_m5_call": args.timeout_s,
            "provenance": provenance,
            "source_sha256": {
                name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("m5_model_audit.py", "m5_local_model.py")
            },
            "model": model.provenance,
            "latency_boundary": (
                "M5 wall time per scorer call includes both label permutations. M4 archived selector "
                "time also includes LangGraph overhead. Route planning is outside both measurements."
            ),
        })
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Verified {report['exact_order_invariance_cases']} reversed-menu cases. Receipt {output}")
    except (ImportError, OSError, RuntimeError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
