"""Explain a crossed sample of actual LA supervisor decisions from frozen traces."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from .la_solver_independence import ROADS, RUN, _source, _state_for
from .m3_cases import digest, read_json, write_json
from .m3_experiment import DEFAULT_MODEL
from .m3_local_model import LocalMenuModel
from .m5_local_model import OrderInvariantMenuModel
from .m6_decision_xai import _CachedModel, _case_receipt


OUTPUT = Path("runs/la_osmnx_decision_xai_20261125.json")
METHODS = ("M4", "M5")


def _inputs(run: Path, roads: Path) -> tuple[dict, dict, list]:
    provenance, config, offers, traces = _source(run, roads)
    first = config["traffic_seeds"][0]
    traces = [trace for trace in traces if trace["traffic_seed"] == first and trace["severity"] == "moderate"]
    combinations = {(profile, kind) for profile in ("light_traffic", "heavy_traffic") for kind in config["chaos_kinds"]}
    if {(trace["traffic"], trace["chaos_kind"]) for trace in traces} != combinations or len(traces) != len(combinations):
        raise ValueError("Archived LA trace did not supply the exact registered XAI sample")
    source = run / "traces.jsonl"
    payload = source.read_bytes()
    trace_digest = sha256(payload).hexdigest()
    fingerprints = {json.loads(line)["case_id"]: (number, sha256(line).hexdigest())
                    for number, line in enumerate(payload.splitlines(), 1)}
    for trace in traces:
        _state_for(trace, offers)
    cases = []
    for trace in traces:
        case_id = trace["case_id"]
        number, fingerprint = fingerprints[case_id]
        binding = {"trace_file": source.name, "trace_line": number, "trace_sha256": trace_digest,
                   "row_sha256": fingerprint, "manifest_sha256": provenance["run_manifest_sha256"]}
        cases.append((case_id, binding, trace))
    provenance.update({"disruption_severity": "moderate", "traffic_seed": first,
                       "explanation_code_sha256": digest(Path(__file__)),
                       "shapley_code_sha256": digest(Path(__file__).with_name("m6_decision_xai.py"))})
    return provenance, read_json(run / "manifest.json")["model"], cases


def _trace_evidence(trace: dict) -> dict:
    receipt = trace["repair_receipt"]
    offers = trace["offers"]["metrics"]
    route_rows = {}
    for candidate in trace["menu"]:
        name = candidate["id"]
        route_rows[name] = {"on_time_parcels": candidate["served"], "traction_wh": candidate["travel_wh"],
                            "restricted_recovery_min": candidate["restricted_recovery_min"],
                            "network_distance_m": trace["route_distances"][name]["network_distance_m"]}
    return {
        "road_evidence": {"tool": "OSMnx", "profile": trace["traffic"], "route_candidates": route_rows},
        "customer_evidence": {"tool": "Mesa", "issued": offers["issued"], "committed": offers["reassignments"],
                              "paid_cents": offers["spent_cents"], "dispatch_min": trace["dispatch"]["dispatch_after_min"]},
        "repair_evidence": {"tool": "OR-Tools CP-SAT", "status": receipt["solver_status"],
                            "strategy": receipt["strategy"], "fallback_reason": receipt["fallback_reason"]},
        "alert": trace["alert"],
        "narrative": (
            f"The shared {trace['traffic'].replace('_', ' ')} OSMnx profile supplied the route geometry and modeled costs. "
            f"Mesa committed {offers['reassignments']} alternate locker slots for {offers['spent_cents']} cents before routing. "
            f"Both supervisors received the same two certified routes after the {trace['chaos_kind'].replace('_', ' ')} alert. "
            f"The repair solver reported {receipt['solver_status']}. The route and offer evidence describes this simulated case."
        ),
    }


def _explain(case: tuple, models: dict | None, saved: dict | None = None) -> dict:
    case_id, binding, trace = case
    original_order = {**trace, "selections": {**trace["selections"], "agentic_supervisor": trace["m4_agentic"]},
                      "model_scores": trace["m4_model_scores"]}
    branches = {"M4": original_order, "M5": trace}
    decisions = {method: _case_receipt(method, (case_id, binding, branches[method]),
                                       None if models is None else models[method],
                                       None if saved is None else saved["scorers"][method]) for method in METHODS}
    if decisions["M4"]["decisions"]["deterministic_supervisor"] != decisions["M5"]["decisions"]["deterministic_supervisor"]:
        raise ValueError("The two scorer explanations differ on the common rule")
    return {"case_id": case_id, "source": binding, "evidence": _trace_evidence(trace), "scorers": decisions}


def _summary(rows: list[dict]) -> dict:
    counts = {}
    for method in METHODS:
        decisions = [row["scorers"][method]["decisions"]["agentic_supervisor"] for row in rows]
        counts[method] = dict(Counter(decision["attribution"]["unit"] if decision["attribution"] is not None
                                    else "planning_deadline" for decision in decisions))
    return {"cases": len(rows), "shared_deterministic_explanations": len(rows), "model_or_fallback_explanations": counts}


def run(source: Path, roads: Path, output: Path, model_path: Path) -> dict:
    provenance, pin, cases = _inputs(source, roads)
    models = {"M4": LocalMenuModel(model_path), "M5": OrderInvariantMenuModel(model_path)}
    if any(model.provenance != pin for model in models.values()):
        raise ValueError("Local scorer model differs from the archived LA model provenance")
    scorers = {method: _CachedModel(model, method) for method, model in models.items()}
    rows = [_explain(case, scorers) for case in cases]
    summary = _summary(rows)
    write_json(output, {"schema": 1, "summary": summary, "provenance": provenance, "model": pin, "rows": rows,
                        "scope": "First archived workload seed, all six moderate disruptions and both modeled traffic profiles. Illustrative route-choice explanations with exact three-group Shapley arithmetic for the declared prompt or rule score."})
    return summary


def verify(source: Path, roads: Path, output: Path) -> dict:
    provenance, pin, cases = _inputs(source, roads)
    stored = read_json(output)
    if stored["schema"] != 1 or stored["provenance"] != provenance or stored["model"] != pin:
        raise ValueError("LA XAI source or scorer fingerprints differ")
    regenerated = [_explain(case, None, row) for case, row in zip(cases, stored["rows"], strict=True)]
    if regenerated != stored["rows"] or _summary(regenerated) != stored["summary"]:
        raise ValueError("LA XAI conclusions differ from saved traces or the coalition arithmetic")
    return stored["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--source", type=Path, default=RUN)
    parser.add_argument("--roads", type=Path, default=ROADS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    result = (run(args.source, args.roads, args.output, args.model) if args.command == "run"
              else verify(args.source, args.roads, args.output))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
