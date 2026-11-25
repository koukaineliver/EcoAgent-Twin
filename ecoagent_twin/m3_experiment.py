"""Train and compare the integrated Month 3 routing and supervisor components."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter

from .m2_experiment import _rollout
from .m2_offer_experiment import DEFAULT_PROTOCOL as OFFER_PROTOCOL
from .m2_offer_experiment import _schedule, prepare_offer_episode
from .m2_offers import UtilityProfile
from .m2_rule import RuleBased
from .m2_simulator import OutageEnv
from .m3_cases import digest, load_worlds, read_json, verify as verify_corpus, write_json
from .model import Scenario


DEFAULT_MODEL = Path("artifacts/models/smollm2-135m-instruct")
FIELDS = (
    "case_id", "stage", "traffic", "traffic_seed", "outage_condition", "method", "demand", "served",
    "service_rate", "recovery_min", "restricted_recovery_min", "travel_wh", "charged_wh",
    "offer_budget_cents", "credit_spent_cents", "selection_status", "selector_ms", "selected_intervention",
)


def _environment_settings(config: dict) -> dict:
    keys = ("charger_rate_wh_per_min", "charger_capacity_wh")
    return {name: config[name] for name in keys}


def train(corpus: Path, output: Path) -> dict:
    from .m3_rllib import train as train_policy

    config, manifest, worlds = load_worlds(corpus)
    training = [world["snapshot"] for world in worlds if world["split"] == "training"]
    validation = [world["snapshot"] for world in worlds if world["split"] == "validation"]
    if len(training) != manifest["world_counts"]["training"] or len(validation) != manifest["world_counts"]["validation"]:
        raise ValueError("training or validation snapshot count differs from corpus manifest")
    output.mkdir(parents=True, exist_ok=False)
    policy, receipt = train_policy(training, validation, **_environment_settings(config), **config["training"])
    checkpoint = output / "policy.pt"
    policy.save(checkpoint)
    report = {
        "corpus_manifest_sha256": digest(corpus / "manifest.json"), "checkpoint_sha256": digest(checkpoint),
        "training_traffic_seeds": config["traffic_seeds"]["training"],
        "validation_traffic_seeds": config["traffic_seeds"]["validation"],
        "runtime": {"ray": version("ray"), "torch": version("torch")},
        "receipt": receipt,
    }
    write_json(output / "training.json", report)
    return report


def _core(scenario: Scenario, settings: dict, choose) -> dict:
    record = _rollout(partial(OutageEnv, scenario, **settings), choose, measure=False)
    return {field: record[field] for field in ("actions", "result")}


def _repair(scenario: Scenario, settings: dict, limits: dict) -> tuple[dict, dict]:
    from .m3_route_repair import repair

    started = perf_counter()
    program = repair(scenario, **settings, **limits)
    planning_ms = 1000 * (perf_counter() - started)
    record = _core(scenario, settings, program.choose)
    if record["result"]["metrics"] != program.receipt["certified_metrics"]:
        raise ValueError("route repair execution disagrees with its certification")
    return record, {**program.receipt, "planning_ms": planning_ms}


def _credit_commitments(scenario: Scenario, case: dict, protocol: dict) -> tuple[Scenario, int, dict, list]:
    profile = UtilityProfile(**protocol["profiles"][protocol["reference_profile"]])
    offers = _schedule(scenario, protocol, profile, "ortools")
    return prepare_offer_episode(scenario, protocol, case, profile, offers, "fixed_credit")


def _restricted(scenario: Scenario, metrics: dict) -> int:
    recovery = metrics["recovery_95_min"]
    return scenario.horizon_min - scenario.outage_min if recovery is None else recovery


def _menu(scenario: Scenario, candidates: dict) -> tuple:
    from .m3_supervisor import Candidate

    labels = {"continue_mappo": "Continue with the shared MAPPO policy", "route_repair": "Use certified OR-Tools route repair"}
    result = []
    for name, record in candidates.items():
        metric = record["result"]["metrics"]
        result.append(Candidate(name, labels[name], metric["served_parcels"], metric["travel_wh"],
                                metric["unserved_parcels"], _restricted(scenario, metric)))
    return tuple(result)


def _row(world: dict, scenario: Scenario, stage: str, method: str, record: dict,
         offers: dict | None = None, selector: dict | None = None) -> dict:
    metric = record["result"]["metrics"]
    offer = {} if offers is None else offers["metrics"]
    choice = {} if selector is None else selector
    return {
        "case_id": scenario.id, "stage": stage, "traffic": world["traffic"], "traffic_seed": world["traffic_seed"],
        "outage_condition": world["case"]["scenario"], "method": method,
        "demand": metric["total_parcels"], "served": metric["served_parcels"], "service_rate": metric["service_rate"],
        "recovery_min": metric["recovery_95_min"], "restricted_recovery_min": _restricted(scenario, metric),
        "travel_wh": metric["travel_wh"], "charged_wh": metric["charged_wh"],
        "offer_budget_cents": offer.get("budget_cents", 0), "credit_spent_cents": offer.get("spent_cents", 0),
        "selection_status": choice.get("status"), "selector_ms": choice.get("elapsed_ms"),
        "selected_intervention": choice.get("decision_id"),
    }


def _summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        for traffic in ("all", row["traffic"]):
            groups[row["stage"], traffic, row["method"]].append(row)
    summary = {}
    for (stage, traffic, method), group in sorted(groups.items()):
        target = summary.setdefault(stage, {}).setdefault(traffic, {})
        statuses = Counter(row["selection_status"] for row in group if row["selection_status"] is not None)
        choices = Counter(row["selected_intervention"] for row in group if row["selected_intervention"] is not None)
        timings = [row["selector_ms"] for row in group if row["selector_ms"] is not None]
        target[method] = {
            "episodes": len(group), "mean_service_rate": fmean(row["service_rate"] for row in group),
            "mean_travel_wh": fmean(row["travel_wh"] for row in group),
            "mean_restricted_recovery_min": fmean(row["restricted_recovery_min"] for row in group),
            "total_credit_spent_cents": sum(row["credit_spent_cents"] for row in group),
            "selection_statuses": dict(statuses), "interventions": dict(choices),
            "mean_selector_ms": fmean(timings) if timings else None,
        }
    by_case = defaultdict(dict)
    for row in rows:
        if row["stage"] == "supervisor":
            by_case[row["case_id"]][row["method"]] = row
    paired = Counter()
    for arms in by_case.values():
        model, rule = arms["local_model_supervisor"], arms["deterministic_supervisor"]
        if (model["offer_budget_cents"], model["credit_spent_cents"]) != (rule["offer_budget_cents"], rule["credit_spent_cents"]):
            raise ValueError("paired supervisors received different budgets or commitments")
        difference = model["served"] - rule["served"]
        paired["model_better" if difference > 0 else "deterministic_better" if difference < 0 else "service_tie"] += 1
    return {"data_basis": "synthetic SUMO-conditioned engineering pilot", "rows": len(rows),
            "by_stage_and_traffic": summary, "supervisor_case_comparison": dict(paired)}


def evaluate(corpus: Path, training_run: Path, model_path: Path, output: Path) -> dict:
    from .m3_local_model import LocalMenuModel
    from .m3_rllib import RLlibMAPPO
    from .m3_supervisor import DecisionView, supervise

    verify_corpus(corpus)
    config, _, worlds = load_worlds(corpus)
    trained = read_json(training_run / "training.json")
    checkpoint = training_run / "policy.pt"
    if (trained["corpus_manifest_sha256"] != digest(corpus / "manifest.json")
            or trained["checkpoint_sha256"] != digest(checkpoint)):
        raise ValueError("training checkpoint or corpus fingerprint differs from the training receipt")
    policy = RLlibMAPPO.load(checkpoint)
    model = LocalMenuModel(model_path)
    settings = _environment_settings(config)
    offer_protocol = {**read_json(OFFER_PROTOCOL), **settings}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "offer_protocol.json", offer_protocol)
    (output / "policy.pt").write_bytes(checkpoint.read_bytes())
    rows = []
    counts = Counter()
    with (output / "traces.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for world in worlds:
            if world["split"] != "evaluation":
                continue
            scenario = Scenario.from_dict(world["snapshot"])
            repaired, receipt = _repair(scenario, settings, config["repair"])
            direct = {"rule_based": _core(scenario, settings, RuleBased(scenario).choose),
                      "rllib_mappo": _core(scenario, settings, policy.choose), "ortools_repair": repaired}
            direct_rows = [_row(world, scenario, "routing", name, record) for name, record in direct.items()]

            routed, expiry, offers, fallback = _credit_commitments(scenario, world["case"], offer_protocol)
            dispatch = {**settings, "preassigned_slots": offers["assignments"], "dispatch_after_min": expiry}
            after_offer_repair, after_offer_receipt = _repair(routed, dispatch, config["repair"])
            candidates = {"continue_mappo": _core(routed, dispatch, policy.choose), "route_repair": after_offer_repair}
            menu = _menu(routed, candidates)
            alert = {"kind": "locker_outage", "locker_id": scenario.outage_locker,
                     "alert_min": scenario.outage_min, "dispatch_min": expiry,
                     "accepted_reassignments": offers["metrics"]["reassignments"],
                     "offer_budget_cents": offers["metrics"]["budget_cents"],
                     "credit_spent_cents": offers["metrics"]["spent_cents"]}
            selections = {
                "deterministic_supervisor": supervise(alert, menu, budget_s=config["supervisor_budget_s"]),
                "local_model_supervisor": supervise(alert, menu, model=model, budget_s=config["supervisor_budget_s"]),
            }
            selected_rows = [_row(world, routed, "supervisor", name, candidates[choice["decision_id"]], offers, choice)
                             for name, choice in selections.items()]
            trace = {"world": world, "direct": direct, "direct_repair_receipt": receipt,
                     "supervisor": {"alert": alert, "offers": offers, "fallbacks": fallback, "candidates": candidates,
                                    "repair_receipt": after_offer_receipt, "selections": selections,
                                    "model_scores": model.last_scores.copy()}}
            stream.write(json.dumps(trace, allow_nan=False) + "\n")
            rows.extend(direct_rows + selected_rows)
            counts["direct_" + receipt["strategy"]] += 1
            counts["supervisor_" + after_offer_receipt["strategy"]] += 1
    with (output / "comparison.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = _summarize(rows)
    summary["route_repair_execution_statuses"] = dict(counts)
    write_json(output / "summary.json", summary)
    source_names = ("m3_cases.py", "m3_rllib.py", "m3_local_model.py", "m3_supervisor.py", "m3_experiment.py",
                    "m3_route_repair.py", "m2_offer_experiment.py", "m2_simulator.py")
    write_json(output / "manifest.json", {
        "schema": 1, "kind": "month_3_mvp", "corpus_manifest_sha256": digest(corpus / "manifest.json"),
        "training_receipt_sha256": digest(training_run / "training.json"), "model": model.provenance,
        "dependencies": {name: version(name) for name in ("ray", "torch", "ortools", "langgraph", "mesa", "transformers")},
        "sha256": {name: digest(output / name) for name in ("policy.pt", "traces.jsonl", "comparison.csv", "summary.json", "offer_protocol.json")},
        "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in source_names},
    })
    return summary


def _certify_record(scenario: Scenario, settings: dict, record: dict) -> None:
    choices = iter(record["actions"])

    def select(decision) -> int:
        choice = next(choices, None)
        if choice is None or (choice["vehicle_id"], choice["time_min"]) != (decision.vehicle_id, decision.time_min):
            raise ValueError("saved action does not match the simulator decision epoch")
        return choice["action_index"]

    if _core(scenario, settings, select) != record or next(choices, None) is not None:
        raise ValueError("saved fleet actions or outcomes differ from certified replay")


def verify_evaluation(corpus: Path, output: Path) -> dict:
    from .m3_rllib import RLlibMAPPO

    manifest = read_json(output / "manifest.json")
    if manifest["schema"] != 1 or manifest["kind"] != "month_3_mvp" or digest(corpus / "manifest.json") != manifest["corpus_manifest_sha256"]:
        raise ValueError("saved evaluation has the wrong corpus or schema")
    for name, digest_value in manifest["sha256"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or digest(path) != digest_value:
            raise ValueError(f"saved evaluation fingerprint mismatch: {name}")
    config, _, worlds = load_worlds(corpus)
    settings = _environment_settings(config)
    policy = RLlibMAPPO.load(output / "policy.pt")
    offer_protocol = read_json(output / "offer_protocol.json")
    rows = []
    counts = Counter()
    with (output / "traces.jsonl").open(encoding="utf-8") as stream:
        evaluation = (world for world in worlds if world["split"] == "evaluation")
        for world, line in zip(evaluation, stream, strict=True):
            trace = json.loads(line)
            if trace["world"] != world:
                raise ValueError("saved trace uses a different evaluation snapshot")
            scenario = Scenario.from_dict(world["snapshot"])
            for name, record in trace["direct"].items():
                _certify_record(scenario, settings, record)
                rows.append(_row(world, scenario, "routing", name, record))
            if (_core(scenario, settings, policy.choose) != trace["direct"]["rllib_mappo"]
                    or _core(scenario, settings, RuleBased(scenario).choose) != trace["direct"]["rule_based"]):
                raise ValueError("stored direct policy decisions differ from fresh inference")
            saved = trace["supervisor"]
            routed, expiry, offers, fallback = _credit_commitments(scenario, world["case"], offer_protocol)
            if offers != saved["offers"] or fallback != saved["fallbacks"]:
                raise ValueError("saved customer responses differ from the matched offer replay")
            settings_with_offers = {**settings, "preassigned_slots": offers["assignments"], "dispatch_after_min": expiry}
            for record in saved["candidates"].values():
                _certify_record(routed, settings_with_offers, record)
            if _core(routed, settings_with_offers, policy.choose) != saved["candidates"]["continue_mappo"]:
                raise ValueError("stored supervisor MAPPO option differs from fresh inference")
            for name, selection in saved["selections"].items():
                rows.append(_row(world, routed, "supervisor", name,
                                 saved["candidates"][selection["decision_id"]], offers, selection))
            counts["direct_" + trace["direct_repair_receipt"]["strategy"]] += 1
            counts["supervisor_" + saved["repair_receipt"]["strategy"]] += 1
    with (output / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("comparison columns differ from saved schema")
        for row, saved in zip(rows, reader, strict=True):
            if {name: "" if value is None else str(value) for name, value in row.items()} != saved:
                raise ValueError("comparison row differs from certified replay")
    summary = _summarize(rows)
    summary["route_repair_execution_statuses"] = dict(counts)
    if summary != read_json(output / "summary.json"):
        raise ValueError("comparison summary differs from certified replay")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    learner = commands.add_parser("train")
    learner.add_argument("corpus", type=Path)
    learner.add_argument("--output", type=Path)
    pilot = commands.add_parser("evaluate")
    pilot.add_argument("corpus", type=Path)
    pilot.add_argument("training", type=Path)
    pilot.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    pilot.add_argument("--output", type=Path)
    check = commands.add_parser("verify")
    check.add_argument("corpus", type=Path)
    check.add_argument("output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        if args.command == "train":
            output = args.output or Path("runs") / f"m3_train_{stamp}"
            train(args.corpus, output)
        elif args.command == "evaluate":
            output = args.output or Path("runs") / f"m3_eval_{stamp}"
            evaluate(args.corpus, args.training, args.model, output)
        else:
            output = args.output
            verify_evaluation(args.corpus, output)
        print(output.resolve())
    except (ImportError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
