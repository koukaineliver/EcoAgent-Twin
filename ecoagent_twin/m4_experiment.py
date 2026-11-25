"""Compare frozen MAPPO and matched supervisors under injected disruptions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from statistics import fmean
from time import perf_counter
import json

from .m1_benchmark import _perturb
from .m2_offer_experiment import DEFAULT_PROTOCOL as OFFER_PROTOCOL
from .m2_sumo_network import CONDITIONS
from .m3_cases import digest, read_json, write_json
from .m3_experiment import _certify_record, _core, _credit_commitments, _environment_settings, _menu, _repair
from .m3_local_model import LocalMenuModel
from .m3_supervisor import supervise
from .m4_chaos import inject
from .m4_corpus import verify as verify_corpus
from .model import Scenario


ARMS = ("mappo_alone", "mappo_agentic_supervisor", "mappo_deterministic_supervisor")
SEVERITIES = ("moderate", "severe")
DEFAULT_MODEL = Path("artifacts/models/smollm2-135m-instruct")
FIELDS = (
    "case_id", "traffic", "traffic_seed", "chaos_kind", "severity", "arm", "demand", "served", "service_rate",
    "recovery_min", "restricted_recovery_min", "travel_wh", "charged_wh", "credit_spent_cents", "budget_cents",
    "selected_intervention", "status", "end_to_end_decision_ms", "first_menu_option",
)
ARTIFACTS = ("policy.pt", "offer_protocol.json", "traces.jsonl", "comparison.csv", "summary.json")


def _worlds(config: dict, corpus: Path):
    kinds = config["chaos_kinds"]
    if not kinds or len(set(kinds)) != len(kinds):
        raise ValueError("chaos kinds must be distinct")
    for site_index, traffic_seed in enumerate(config["traffic_seeds"]):
        for traffic_index, traffic in enumerate(CONDITIONS):
            base = read_json(corpus / "sumo" / str(traffic_seed) / traffic / "scenario.json")
            for severity_index, severity in enumerate(SEVERITIES):
                case_seed = config["case_seeds"][severity]
                common = _perturb(base, config, traffic_seed, case_seed)
                for kind_index, kind in enumerate(kinds):
                    identifier = f"{traffic_seed}-{traffic}-{kind}-{severity}"
                    raw, charger, alerts = inject(common, kind, severity, traffic_seed * 1000 + case_seed,
                                                   **_environment_settings(config))
                    raw["id"] = identifier
                    backwards = (site_index + traffic_index + severity_index + kind_index) % 2
                    order = ["route_repair", "continue_mappo"] if backwards else ["continue_mappo", "route_repair"]
                    key = {"day": traffic_seed, "seed": case_seed, "scenario": f"{kind}:{severity}"}
                    yield {"case_id": identifier, "traffic": traffic, "traffic_seed": traffic_seed,
                           "chaos_kind": kind, "severity": severity, "case": key, "snapshot": raw,
                           "charger": charger, "alerts": alerts, "menu_order": order}


def _take_decision(alert: dict, menu: tuple, budget_s: float, planning_ms: float, *, model=None) -> dict:
    remaining = budget_s - planning_ms / 1000
    if remaining <= 0:
        return {"status": "planning_timeout", "decision_id": "continue_mappo", "fallback_used": True,
                "reason": "Planning used the deadline. Use the certified MAPPO continuation.",
                "elapsed_ms": 0.0, "end_to_end_decision_ms": planning_ms}
    receipt = supervise(alert, menu, model=model, budget_s=remaining)
    return {**receipt, "end_to_end_decision_ms": planning_ms + receipt["elapsed_ms"]}


def _metadata(world: dict) -> dict:
    return {name: world[name] for name in ("case_id", "traffic", "traffic_seed", "chaos_kind", "severity")}


def _row(world: dict, scenario: Scenario, record: dict, arm: str, choice: dict, offers: dict) -> dict:
    metric = record["result"]["metrics"]
    recovery = metric["recovery_95_min"]
    return {
        **_metadata(world), "arm": arm, "demand": metric["total_parcels"], "served": metric["served_parcels"],
        "service_rate": metric["service_rate"], "recovery_min": recovery,
        "restricted_recovery_min": scenario.horizon_min - scenario.outage_min if recovery is None else recovery,
        "travel_wh": metric["travel_wh"], "charged_wh": metric["charged_wh"],
        "credit_spent_cents": offers["metrics"]["spent_cents"], "budget_cents": offers["metrics"]["budget_cents"],
        "selected_intervention": choice["decision_id"], "status": choice["status"],
        "end_to_end_decision_ms": choice["end_to_end_decision_ms"], "first_menu_option": world["menu_order"][0],
    }


def _offer_state(world: dict, offer_protocol: dict):
    initial = Scenario.from_dict(world["snapshot"])
    routed, expiry, offers, fallback = _credit_commitments(initial, world["case"], {**offer_protocol, **world["charger"]})
    dispatch = {**world["charger"], "preassigned_slots": offers["assignments"], "dispatch_after_min": expiry}
    alert = {"kind": world["chaos_kind"], "severity": world["severity"], "traffic": world["traffic"],
             "alert_min": initial.outage_min, "horizon_min": initial.horizon_min, "dispatch_min": expiry,
             "revealed_disruptions": world["alerts"], "onboard_parcels": len(initial.parcels),
             "accepted_reassignments": offers["metrics"]["reassignments"],
             "offer_budget_cents": offers["metrics"]["budget_cents"], "credit_spent_cents": offers["metrics"]["spent_cents"]}
    return routed, dispatch, offers, fallback, alert


def _evaluate_case(world: dict, config: dict, offers_protocol: dict, policy, model) -> tuple[dict, list[dict]]:
    scenario, dispatch, offers, fallback, alert = _offer_state(world, offers_protocol)
    clock = perf_counter()
    baseline = _core(scenario, dispatch, policy.choose)
    baseline_ms = 1000 * (perf_counter() - clock)
    repaired, receipt = _repair(scenario, dispatch, config["repair"])
    plans = {"continue_mappo": baseline, "route_repair": repaired}
    candidates = {name: plans[name] for name in world["menu_order"]}
    menu = _menu(scenario, candidates)
    planning_ms = 1000 * (perf_counter() - clock)
    model.last_scores = {}
    selections = {
        "deterministic_supervisor": _take_decision(alert, menu, config["supervisor_budget_s"], planning_ms),
        "agentic_supervisor": _take_decision(alert, menu, config["supervisor_budget_s"], planning_ms, model=model),
    }
    solo = {"decision_id": "continue_mappo", "status": "selected", "end_to_end_decision_ms": baseline_ms}
    choices = (solo, selections["agentic_supervisor"], selections["deterministic_supervisor"])
    rows = [_row(world, scenario, candidates[choice["decision_id"]], arm, choice, offers)
            for arm, choice in zip(ARMS, choices, strict=True)]
    trace = {**_metadata(world), "world": world, "alerts": world["alerts"], "alert": alert,
             "offers": offers, "fallbacks": fallback, "candidates": candidates, "repair_receipt": receipt,
             "menu": [asdict(candidate) for candidate in menu], "model_scores": model.last_scores.copy(),
             "selections": selections, "mappo_alone": solo, "preselection_planning_ms": planning_ms}
    return trace, rows


def _summary(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    by_case = defaultdict(dict)
    for row in rows:
        groups[row["arm"], "all"].append(row)
        groups[row["arm"], row["chaos_kind"]].append(row)
        if row["arm"] in by_case[row["case_id"]]:
            raise ValueError("duplicate arm in matched experiment")
        by_case[row["case_id"]][row["arm"]] = row
    report = {}
    for (arm, kind), group in sorted(groups.items()):
        ordered = sorted(row["end_to_end_decision_ms"] for row in group)
        report.setdefault(kind, {})[arm] = {
            "episodes": len(group), "mean_service_rate": fmean(row["service_rate"] for row in group),
            "reached_95_percent_service": sum(row["recovery_min"] is not None for row in group),
            "mean_restricted_recovery_min": fmean(row["restricted_recovery_min"] for row in group),
            "mean_travel_wh": fmean(row["travel_wh"] for row in group),
            "credit_spent_cents": sum(row["credit_spent_cents"] for row in group),
            "mean_decision_ms": fmean(ordered), "p95_decision_ms": ordered[(95 * len(ordered) + 99) // 100 - 1],
            "statuses": dict(Counter(row["status"] for row in group)),
            "interventions": dict(Counter(row["selected_intervention"] for row in group)),
        }
    paired = {}
    for arm in ARMS[1:]:
        results = Counter()
        differences = []
        for cohort in by_case.values():
            solo, other = cohort[ARMS[0]], cohort[arm]
            if any(solo[key] != other[key] for key in ("demand", "budget_cents", "credit_spent_cents")):
                raise ValueError("matched tiers changed the cohort or the common offer stage")
            delta = other["served"] - solo["served"]
            differences.append(delta / solo["demand"])
            results["better" if delta > 0 else "worse" if delta < 0 else "tie"] += 1
        paired[arm] = {"service_change_percentage_points": 100 * fmean(differences),
                       "pairs": len(differences), **{label: results[label] for label in ("better", "tie", "worse")}}
    agent_rows = groups["mappo_agentic_supervisor", "all"]
    return {"data_basis": "fresh synthetic SUMO traffic with seeded chaos overlays",
            "matched_cases": len(by_case), "episode_rows": len(rows), "by_chaos_kind": report,
            "paired_against_mappo_alone": paired,
            "agentic_selected_first_option": sum(row["selected_intervention"] == row["first_menu_option"] for row in agent_rows),
            "agentic_first_option_exposure": dict(Counter(row["first_menu_option"] for row in agent_rows)),
            "latency_boundary": "wall time of post-offer MAPPO rollout, optional route repair and selector; common offer time excluded"}


def run(corpus: Path, training_run: Path, model_path: Path, output: Path) -> dict:
    from .m3_rllib import RLlibMAPPO

    config = verify_corpus(corpus)
    checkpoint = training_run / "policy.pt"
    trained = read_json(training_run / "training.json")
    if digest(checkpoint) != trained["checkpoint_sha256"]:
        raise ValueError("frozen MAPPO checkpoint fingerprint differs from training")
    used_seeds = {*trained["training_traffic_seeds"], *trained["validation_traffic_seeds"]}
    if used_seeds.intersection(config["traffic_seeds"]):
        raise ValueError("Month 4 traffic overlaps the frozen training or validation split")
    policy = RLlibMAPPO.load(checkpoint)
    model = LocalMenuModel(model_path)
    offer_protocol = read_json(OFFER_PROTOCOL)
    output.mkdir(parents=True, exist_ok=False)
    (output / "policy.pt").write_bytes(checkpoint.read_bytes())
    write_json(output / "offer_protocol.json", offer_protocol)
    rows = []
    with (output / "traces.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for world in _worlds(config, corpus):
            trace, cohort = _evaluate_case(world, config, offer_protocol, policy, model)
            stream.write(json.dumps(trace, allow_nan=False) + "\n")
            rows.extend(cohort)
    with (output / "comparison.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary(rows)
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", {
        "schema": 1, "kind": "month_4_chaos_experiment", "corpus_manifest_sha256": digest(corpus / "manifest.json"),
        "training_receipt_sha256": digest(training_run / "training.json"), "model": model.provenance,
        "supervisor_budget_s": config["supervisor_budget_s"],
        "sha256": {name: digest(output / name) for name in ARTIFACTS},
        "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in (
            "m4_experiment.py", "m4_corpus.py", "m4_chaos.py", "m3_route_repair.py", "m3_supervisor.py",
            "m3_local_model.py", "m3_rllib.py", "m2_offer_experiment.py", "m2_simulator.py",
        )},
    })
    return summary


def _verify_choice(selection: dict, menu: list[dict], planning_ms: float, budget_ms: float,
                   scores: dict | None) -> None:
    deterministic = min(menu, key=lambda item: (-item["served"], item["travel_wh"], item["restricted_recovery_min"], item["id"]))
    elapsed = selection["end_to_end_decision_ms"]
    if not isfinite(elapsed) or elapsed < planning_ms:
        raise ValueError("invalid recorded supervisor time")
    status = selection["status"]
    if status == "planning_timeout":
        expected = "continue_mappo"
    elif selection["fallback_used"] or scores is None:
        expected = deterministic["id"]
    else:
        if set(scores) != {candidate["id"] for candidate in menu} or not all(isfinite(score) for score in scores.values()):
            raise ValueError("saved model scores differ from the intervention menu")
        expected = max(menu, key=lambda candidate: scores[candidate["id"]])["id"]
    if (selection["decision_id"] != expected or (status == "selected" and elapsed > budget_ms)
            or (status == "planning_timeout" and planning_ms < budget_ms)):
        raise ValueError("saved supervisor decision violates the recorded menu or deadline")


def verify(corpus: Path, output: Path) -> dict:
    from .m3_rllib import RLlibMAPPO

    config = verify_corpus(corpus)
    manifest = read_json(output / "manifest.json")
    if (manifest["schema"] != 1 or manifest["kind"] != "month_4_chaos_experiment"
            or manifest["corpus_manifest_sha256"] != digest(corpus / "manifest.json")):
        raise ValueError("experiment is bound to a different corpus or schema")
    for name in ARTIFACTS:
        if digest(output / name) != manifest["sha256"][name]:
            raise ValueError(f"experiment fingerprint mismatch: {name}")
    policy = RLlibMAPPO.load(output / "policy.pt")
    offers_protocol = read_json(output / "offer_protocol.json")
    rows = []
    with (output / "traces.jsonl").open(encoding="utf-8") as stream:
        for world, line in zip(_worlds(config, corpus), stream, strict=True):
            trace = json.loads(line)
            if trace["world"] != world or any(trace[name] != value for name, value in _metadata(world).items()):
                raise ValueError("saved world differs from regenerated chaos inputs")
            scenario, dispatch, offers, fallback, alert = _offer_state(world, offers_protocol)
            if (trace["alerts"], trace["alert"], trace["offers"], trace["fallbacks"]) != (world["alerts"], alert, offers, fallback):
                raise ValueError("alerts or shared offer stage differ from replay")
            candidates = trace["candidates"]
            if list(candidates) != world["menu_order"]:
                raise ValueError("saved intervention ordering differs from the frozen protocol")
            if _core(scenario, dispatch, policy.choose) != candidates["continue_mappo"]:
                raise ValueError("frozen MAPPO replay differs")
            _certify_record(scenario, dispatch, candidates["route_repair"])
            if candidates["route_repair"]["result"]["metrics"] != trace["repair_receipt"]["certified_metrics"]:
                raise ValueError("route repair receipt does not certify the replayed result")
            menu = [asdict(candidate) for candidate in _menu(scenario, candidates)]
            if trace["menu"] != menu:
                raise ValueError("saved intervention summary differs from certified results")
            deterministic = trace["selections"]["deterministic_supervisor"]
            agentic = trace["selections"]["agentic_supervisor"]
            budget_ms = 1000 * config["supervisor_budget_s"]
            _verify_choice(deterministic, menu, trace["preselection_planning_ms"], budget_ms, None)
            _verify_choice(agentic, menu, trace["preselection_planning_ms"], budget_ms, trace["model_scores"])
            for arm, choice in zip(ARMS, (trace["mappo_alone"], agentic, deterministic), strict=True):
                rows.append(_row(world, scenario, candidates[choice["decision_id"]], arm, choice, offers))
    with (output / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("comparison columns differ from saved schema")
        for row, saved in zip(rows, reader, strict=True):
            if saved != {name: "" if value is None else str(value) for name, value in row.items()}:
                raise ValueError("comparison row differs from saved trace replay")
    summary = _summary(rows)
    if summary != read_json(output / "summary.json"):
        raise ValueError("summary differs from replayed outcomes")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("run")
    evaluate.add_argument("corpus", type=Path)
    evaluate.add_argument("training", type=Path)
    evaluate.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    evaluate.add_argument("--output", type=Path)
    verify_command = commands.add_parser("verify")
    verify_command.add_argument("corpus", type=Path)
    verify_command.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "run":
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            output = args.output or Path("runs") / f"m4_eval_{stamp}"
            summary = run(args.corpus, args.training, args.model, output)
        else:
            output = args.output
            summary = verify(args.corpus, output)
        print(output.resolve())
        print(f"{summary['matched_cases']} matched cases and {summary['episode_rows']} episode rows")
    except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
