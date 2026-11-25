"""Compare two repair backends on a matched, frozen Month 5 workload."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from platform import python_version
from statistics import fmean, median

from .m3_cases import digest, read_json, write_json
from .m3_experiment import _certify_record, _restricted
from .m5_experiment import _state
from .m6_solver import DEFAULT_TRANSITION_LIMIT, certified_repair, cp_sat_backend, simulator_backend


METHODS = ("cp_sat", "simulator_search")


def _fingerprint(path: Path, expected: str) -> None:
    if digest(path) != expected:
        raise ValueError(f"fingerprint mismatch for {path}")


def _inputs(m5_run: Path, corpus: Path) -> tuple[dict, dict, dict, list[dict]]:
    manifest = read_json(m5_run / "manifest.json")
    if manifest["kind"] != "month_5_matched_ablation":
        raise ValueError("expected a frozen Month 5 evaluation")
    corpus_manifest_path = corpus / "manifest.json"
    _fingerprint(corpus_manifest_path, manifest["corpus_manifest_sha256"])
    corpus_manifest = read_json(corpus_manifest_path)
    protocol_path = corpus / "protocol.json"
    _fingerprint(protocol_path, corpus_manifest["sha256"]["protocol.json"])
    config = read_json(protocol_path)
    offer_path, traces_path = m5_run / "offer_protocol.json", m5_run / "traces.jsonl"
    for path in (offer_path, traces_path):
        _fingerprint(path, manifest["sha256"][path.name])
    for source, expected in manifest["source_sha256"].items():
        _fingerprint(Path(__file__).with_name(source), expected)
    selected_seeds = config["scale_traffic_seeds"][:2]
    cases = []
    with traces_path.open(encoding="utf-8") as handle:
        for line in handle:
            trace = json.loads(line)
            world = trace["world"]
            if world["fleet_size"] == 2 and world["traffic_seed"] in selected_seeds:
                cases.append(trace)
    observed = Counter((t["world"]["traffic_seed"], t["world"]["traffic"], t["world"]["chaos_kind"],
                        t["world"]["severity"]) for t in cases)
    expected = Counter((seed, traffic, chaos, severity) for seed in selected_seeds
                       for traffic in ("light_traffic", "heavy_traffic") for chaos in config["chaos_kinds"]
                       for severity in config["case_seeds"])
    if len(selected_seeds) != 2 or observed != expected or len({t["case_id"] for t in cases}) != len(cases):
        raise ValueError("archived panel does not contain the complete crossed design")
    provenance = {
        "m5_run": m5_run.name, "corpus": corpus.name, "m5_manifest_sha256": digest(m5_run / "manifest.json"),
        "corpus_manifest_sha256": digest(corpus_manifest_path), "selected_traffic_seeds": selected_seeds,
        "archived_source_files_confirmed": len(manifest["source_sha256"]),
        "new_source_sha256": {name: digest(Path(__file__).with_name(name))
                              for name in ("m6_solver.py", "m6_solver_experiment.py")},
    }
    return provenance, config, read_json(offer_path), cases


def _shared_state(trace: dict, offer_protocol: dict):
    scenario, dispatch, offers, fallbacks, alert = _state(trace["world"], offer_protocol)
    if (offers, fallbacks, alert) != (trace["offers"], trace["fallbacks"], trace["alert"]):
        raise ValueError("shared dispatch state differs from archived offer commitments")
    _certify_record(scenario, dispatch, trace["candidates"]["route_repair"])
    return scenario, dispatch, offers


def _summary(rows: list[dict]) -> dict:
    by_method = {}
    for method in METHODS:
        arms = [row["backends"][method] for row in rows]
        metrics = [arm["record"]["result"]["metrics"] for arm in arms]
        latency = sorted(arm["planning_ms"] for arm in arms)
        by_method[method] = {
            "cases": len(arms), "mean_service_rate": fmean(item["service_rate"] for item in metrics),
            "parcels_served": sum(item["served_parcels"] for item in metrics),
            "mean_travel_min": fmean(item["travel_min"] for item in metrics),
            "mean_travel_wh": fmean(item["travel_wh"] for item in metrics),
            "mean_restricted_recovery_min": fmean(arm["restricted_recovery_min"] for arm in arms),
            "planning_ms": {"mean": fmean(latency), "median": median(latency),
                            "p95_nearest_rank": latency[(95 * len(latency) + 99) // 100 - 1], "max": latency[-1]},
            "mean_shared_replay_ms": fmean(arm["shared_replay_ms"] for arm in arms),
            "status_counts": dict(Counter(arm["planner"]["solver_status"] for arm in arms)),
            "strategy_counts": dict(Counter(arm["planner"]["strategy"] for arm in arms)),
        }
    differences = []
    equal_objectives = 0
    for row in rows:
        cp, search = (row["backends"][method]["record"]["result"]["metrics"] for method in METHODS)
        differences.append(search["served_parcels"] - cp["served_parcels"])
        equal_objectives += all(cp[k] == search[k] for k in ("served_parcels", "travel_min", "travel_wh"))
    return {
        "matched_cases": len(rows), "by_backend": by_method,
        "shared_offer_credit_cents": sum(row["offer_metrics"]["spent_cents"] for row in rows),
        "search_service_win_tie_loss": [sum(n > 0 for n in differences), sum(n == 0 for n in differences),
                                        sum(n < 0 for n in differences)],
        "equal_service_time_energy_objective": equal_objectives,
        "cp_sat_record_matches_archived": sum(row["cp_sat_record_matches_archived"] for row in rows),
        "mean_search_minus_cp_sat_service_pp": 100 * fmean(
            diff / row["backends"]["cp_sat"]["record"]["result"]["metrics"]["total_parcels"]
            for diff, row in zip(differences, rows, strict=True)
        ),
    }


def run(m5_run: Path, corpus: Path, output: Path, limit: int) -> dict:
    from ortools import __version__ as ortools_version

    provenance, config, protocol, traces = _inputs(m5_run, corpus)
    planners = {"cp_sat": cp_sat_backend(**config["repair"]), "simulator_search": simulator_backend(limit)}
    rows = []
    for i, trace in enumerate(traces):
        scenario, dispatch, offers = _shared_state(trace, protocol)
        order = METHODS if i % 2 == 0 else METHODS[::-1]
        arms = {}
        for method in order:
            arm = certified_repair(scenario, dispatch, planners[method])
            arm["restricted_recovery_min"] = _restricted(scenario, arm["record"]["result"]["metrics"])
            arms[method] = arm
        rows.append({
            **{k: trace[k] for k in ("case_id", "traffic", "traffic_seed", "chaos_kind", "severity")},
            "execution_order": list(order), "dispatch": dispatch, "offer_metrics": offers["metrics"],
            "backends": arms, "cp_sat_record_matches_archived": (
                arms["cp_sat"]["record"] == trace["candidates"]["route_repair"]),
        })
    report = {
        "schema": "m6_solver_verification_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "basis": "two-EV synthetic SUMO-conditioned M5 cases with matched simulated offer commitments",
        "objective": ["maximize on-time deposits", "minimize travel minutes", "minimize traction watt-hours"],
        "cp_sat_scope": "existing restricted route columns, up to one charge per vehicle and shared CP-SAT wall-time budget",
        "search_scope": "simulator-masked dispatch action tree under fixed offers/custody; exact only if completed with admissible pruning, otherwise a feasible incumbent",
        "latency_scope": "single local run per backend per case; planner initialization, optimization, and its internal certification included; common replay separate; offers excluded; backend order alternated",
        "runtime": {"python": python_version(), "ortools": ortools_version},
        "provenance": provenance, "repair_limits": config["repair"], "search_transition_limit": limit,
        "summary": _summary(rows), "rows": rows,
    }
    write_json(output, report)
    return report["summary"]


def verify(m5_run: Path, corpus: Path, output: Path) -> dict:
    saved = read_json(output)
    provenance, config, protocol, traces = _inputs(m5_run, corpus)
    if (saved["schema"] != "m6_solver_verification_v1" or saved["provenance"] != provenance
            or saved["repair_limits"] != config["repair"]):
        raise ValueError("receipt input or code provenance differs")
    search = simulator_backend(saved["search_transition_limit"])
    for i, (trace, row) in enumerate(zip(traces, saved["rows"], strict=True)):
        scenario, dispatch, offers = _shared_state(trace, protocol)
        if ((row["dispatch"], row["offer_metrics"]) != (dispatch, offers["metrics"])
                or any(row[k] != trace[k] for k in ("case_id", "traffic", "traffic_seed", "chaos_kind", "severity"))
                or set(row["backends"]) != set(METHODS)
                or row["execution_order"] != list(METHODS if i % 2 == 0 else METHODS[::-1])):
            raise ValueError("receipt case, execution order, or shared offers differ")
        for arm in row["backends"].values():
            record = arm["record"]
            _certify_record(scenario, dispatch, record)
            if (arm["planner"]["certified_metrics"] != record["result"]["metrics"]
                    or arm["planner"]["certified_unserved_parcel_ids"] != record["result"]["unserved_parcel_ids"]
                    or arm["restricted_recovery_min"] != _restricted(scenario, record["result"]["metrics"])):
                raise ValueError("receipt metrics differ from the common replay")
        cp, alt = (row["backends"][method] for method in METHODS)
        rebuilt = certified_repair(scenario, dispatch, search)
        if (row["cp_sat_record_matches_archived"] != (cp["record"] == trace["candidates"]["route_repair"])
                or any(alt[k] != rebuilt[k] for k in ("record", "planner"))):
            raise ValueError("archival CP-SAT comparison or deterministic search differs")
    if _summary(saved["rows"]) != saved["summary"]:
        raise ValueError("saved aggregate differs from certified records")
    return saved["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "verify"):
        command = commands.add_parser(name)
        command.add_argument("m5_run", type=Path)
        command.add_argument("corpus", type=Path)
        command.add_argument("receipt", type=Path)
        if name == "run":
            command.add_argument("--transition-limit", type=int, default=DEFAULT_TRANSITION_LIMIT)
    args = parser.parse_args()
    if args.command == "run":
        result = run(args.m5_run, args.corpus, args.receipt, args.transition_limit)
    else:
        result = verify(args.m5_run, args.corpus, args.receipt)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
