"""Run a matched fresh-seed ablation and a crossed workload scalability panel."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import json
from math import isclose, isfinite
from pathlib import Path
from random import Random
from statistics import fmean
from time import perf_counter

from .m2_offer_experiment import DEFAULT_PROTOCOL as OFFER_PROTOCOL
from .m3_cases import digest, read_json, write_json
from .m3_experiment import _certify_record, _core, _menu, _repair
from .m3_supervisor import DecisionView, _freeze, _graph
from .m4_corpus import verify as verify_corpus
from .m4_experiment import ARMS, DEFAULT_MODEL, FIELDS as M4_FIELDS
from .m4_experiment import _metadata, _offer_state, _row, _take_decision, _worlds as chaos_worlds
from .m5_deadline import DeadlineModel
from .m5_local_model import OrderInvariantMenuModel
from .m5_scale import FLEET_SIZES, scale_workload


FIELDS = (*M4_FIELDS, "base_case_id", "fleet_size", "locker_count", "open_locker_slots", "scale_panel",
          "offer_ms", "preselection_ms", "selector_ms", "worker_status", "over_budget")
ARTIFACTS = ("policy.pt", "training.json", "offer_protocol.json", "traces.jsonl", "comparison.csv", "summary.json")
SOURCE_FILES = (
    "m5_experiment.py", "m5_local_model.py", "m5_deadline.py", "m5_scale.py",
    "m4_experiment.py", "m4_corpus.py", "m4_chaos.py", "m1_benchmark.py",
    "m2_offer_experiment.py", "m2_simulator.py", "m3_experiment.py",
    "m3_route_repair.py", "m3_rllib.py", "m3_supervisor.py", "m3_local_model.py",
)


def _protocol(corpus: Path, training: dict) -> dict:
    config = verify_corpus(corpus)
    seeds, scale_seeds = config["traffic_seeds"], config["scale_traffic_seeds"]
    used = {*training["training_traffic_seeds"], *training["validation_traffic_seeds"]}
    if used.intersection(seeds):
        raise ValueError("the evaluation uses a frozen training or validation traffic seed")
    if not scale_seeds or len(set(scale_seeds)) != len(scale_seeds) or not set(scale_seeds) <= set(seeds):
        raise ValueError("the scalability panel must select distinct seeds from the ablation")
    if tuple(config["fleet_sizes"]) != FLEET_SIZES:
        raise ValueError("expected the registered fleet sizes in ascending order")
    reserve, budget = config["model_fallback_reserve_s"], config["supervisor_budget_s"]
    if not isfinite(budget) or not 0 < reserve < budget:
        raise ValueError("the model fallback reserve must be within the decision budget")
    if type(config["model_torch_threads"]) is not int or config["model_torch_threads"] < 1:
        raise ValueError("model thread count must be a positive integer")
    return config


def _worlds(config: dict, corpus: Path):
    scale_seeds = set(config["scale_traffic_seeds"])
    for index, base in enumerate(chaos_worlds(config, corpus)):
        panel = base["traffic_seed"] in scale_seeds
        sizes = config["fleet_sizes"] if panel else config["fleet_sizes"][:1]
        # Rotate the within-snapshot execution sequence to distribute order and warm-cache effects.
        start = index % len(sizes)
        for size in (*sizes[start:], *sizes[:start]):
            snapshot, scaling = scale_workload(base["snapshot"], size)
            case_id = f"{base['case_id']}-v{size}"
            snapshot["id"] = case_id
            yield {**base, "case_id": case_id, "base_case_id": base["case_id"], "scale_panel": panel,
                   "fleet_size": size, "snapshot": snapshot, "scaling": scaling}


def _state(world: dict, protocol: dict):
    factor = world["fleet_size"] // FLEET_SIZES[0]
    pickup = protocol["locker_pickup"]
    pickup_at_banks = {node: pickup[origin] for node, origin in world["scaling"]["node_origins"].items()
                       if origin in pickup}
    settings = {**protocol, "budget_cents": protocol["budget_cents"] * factor, "locker_pickup": pickup_at_banks}
    scenario, dispatch, offers, fallbacks, alert = _offer_state(world, settings)
    alert = {**alert, "fleet_size": world["fleet_size"], "open_lockers": world["scaling"]["open_locker_count"]}
    return scenario, dispatch, offers, fallbacks, alert


def _rows(world, scenario, offers, candidates, solo, selections, timing, worker_status, budget_ms):
    choices = solo, selections["agentic_supervisor"], selections["deterministic_supervisor"]
    rows = []
    for arm, choice in zip(ARMS, choices, strict=True):
        rows.append({
            **_row(world, scenario, candidates[choice["decision_id"]], arm, choice, offers),
            "base_case_id": world["base_case_id"], "fleet_size": world["fleet_size"],
            "locker_count": world["scaling"]["locker_count"], "open_locker_slots": world["scaling"]["open_slots"],
            "scale_panel": world["scale_panel"], "offer_ms": timing["offer_ms"],
            "preselection_ms": timing["solo_ms"] if arm == ARMS[0] else timing["planning_ms"],
            "selector_ms": choice.get("elapsed_ms", 0.0),
            "worker_status": worker_status if arm == ARMS[1] else "",
            "over_budget": choice["end_to_end_decision_ms"] >= budget_ms,
        })
    return rows


def _evaluate(world, config, offer_protocol, policy, model):
    started = perf_counter()
    scenario, dispatch, offers, fallbacks, alert = _state(world, offer_protocol)
    offer_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    baseline = _core(scenario, dispatch, policy.choose)
    solo_ms = (perf_counter() - started) * 1000
    repaired, repair_receipt = _repair(scenario, dispatch, config["repair"])
    plans = {"continue_mappo": baseline, "route_repair": repaired}
    candidates = {name: plans[name] for name in world["menu_order"]}
    menu = _menu(scenario, candidates)
    planning_ms = (perf_counter() - started) * 1000
    budget = config["supervisor_budget_s"]
    deterministic = _take_decision(alert, menu, budget, planning_ms)
    agentic = _take_decision(alert, menu, budget, planning_ms, model=model)
    worker_status = "not_called" if agentic["status"] == "planning_timeout" else model.last_status
    scores = model.accepted_scores(agentic)
    solo = {"status": "selected", "decision_id": "continue_mappo", "end_to_end_decision_ms": solo_ms}
    timing = {"offer_ms": offer_ms, "solo_ms": solo_ms, "planning_ms": planning_ms}
    selections = {"deterministic_supervisor": deterministic, "agentic_supervisor": agentic}

    trace = {**_metadata(world), "world": world, "alerts": world["alerts"], "alert": alert,
             "offers": offers, "fallbacks": fallbacks, "candidates": candidates, "repair_receipt": repair_receipt,
             "menu": [asdict(item) for item in menu], "mappo_alone": solo, "selections": selections,
             "model_scores": scores, "worker_status": worker_status, "timing": timing,
             "preselection_planning_ms": planning_ms}
    return trace, _rows(world, scenario, offers, candidates, solo, selections, timing, worker_status, budget * 1000)


def _latencies(values):
    ordered = sorted(values)
    return {"mean": fmean(ordered), **{f"p{p}": ordered[(len(ordered) * p + 99) // 100 - 1]
                                     for p in (50, 95, 99)}}


def _arm_stats(rows):
    return {
        "cases": len(rows), "mean_on_time_service": fmean(r["service_rate"] for r in rows),
        "mean_travel_wh": fmean(r["travel_wh"] for r in rows), "mean_charged_wh": fmean(r["charged_wh"] for r in rows),
        "reached_95_percent": sum(r["recovery_min"] is not None for r in rows),
        "mean_restricted_recovery_min": fmean(r["restricted_recovery_min"] for r in rows),
        "offer_credit_cents": sum(r["credit_spent_cents"] for r in rows),
        "decision_ms": _latencies(r["end_to_end_decision_ms"] for r in rows),
        "offer_ms": _latencies(r["offer_ms"] for r in rows), "statuses": dict(Counter(r["status"] for r in rows)),
        "interventions": dict(Counter(r["selected_intervention"] for r in rows)),
        "over_budget": sum(r["over_budget"] for r in rows),
        "worker_statuses": dict(Counter(r["worker_status"] for r in rows if r["worker_status"])),
    }


def _paired(cases, test, control, *, interval=False):
    signs = Counter()
    deltas = defaultdict(list)
    for arms in cases.values():
        selected, comparison = arms[test], arms[control]
        difference = selected["served"] - comparison["served"]
        signs["better" if difference > 0 else "worse" if difference < 0 else "tie"] += 1
        deltas[selected["traffic_seed"]].append(100 * difference / comparison["demand"])
    mean = fmean(delta for cluster in deltas.values() for delta in cluster)
    result = {"service_change_pp": mean, **{key: signs[key] for key in ("better", "tie", "worse")}}
    if interval:
        rng = Random(20261125)
        seed_means = [fmean(cluster) for cluster in deltas.values()]
        samples = sorted(fmean(rng.choices(seed_means, k=len(seed_means))) for _ in range(3000))
        result["traffic_seed_cluster_bootstrap_95_pp"] = [samples[74], samples[2924]]
    return result


def _group(cases, *, interval=False):
    return {
        "cases_per_arm": len(cases), "traffic_seeds": len({a[ARMS[0]]["traffic_seed"] for a in cases.values()}),
        "by_arm": {arm: _arm_stats([matched[arm] for matched in cases.values()]) for arm in ARMS},
        "paired": {f"{label}_vs_{comparison}": _paired(cases, test, control, interval=interval)
                   for label, comparison, test, control in (
                       ("agentic", "mappo", ARMS[1], ARMS[0]),
                       ("deterministic", "mappo", ARMS[2], ARMS[0]),
                       ("agentic", "deterministic", ARMS[1], ARMS[2]),
                   )},
    }


def _summary(rows, repair_statuses):
    cases = defaultdict(dict)
    for row in rows:
        matched = cases[row["case_id"]]
        if row["arm"] in matched:
            raise ValueError("a matched case contains a duplicate arm")
        matched[row["arm"]] = row
    baseline, scaling, families = {}, defaultdict(dict), defaultdict(dict)
    for case_id, matched in cases.items():
        if matched.keys() != set(ARMS):
            raise ValueError("a matched case does not contain all registered arms")
        shared = ("demand", "credit_spent_cents", "budget_cents", "traffic_seed", "fleet_size", "locker_count")
        first = matched[ARMS[0]]
        if any(first[k] != row[k] for k in shared for row in matched.values()):
            raise ValueError("a controller did not share the registered case inputs")
        if first["fleet_size"] == FLEET_SIZES[0]:
            baseline[case_id] = matched
            families[first["chaos_kind"]][case_id] = matched
        if first["scale_panel"]:
            scaling[first["fleet_size"]][case_id] = matched
    panels = [{a[ARMS[0]]["base_case_id"] for a in scale.values()} for scale in scaling.values()]
    if set(scaling) != set(FLEET_SIZES) or not all(panel == panels[0] for panel in panels):
        raise ValueError("the scalability panel is missing a matched base workload")
    return {
        "matched_cases": len(cases), "episode_rows": len(rows),
        "ablation": _group(baseline, interval=True),
        "by_chaos_kind": {kind: _group(group) for kind, group in families.items()},
        "scalability": {str(size): _group(group) for size, group in sorted(scaling.items())},
        "repair_receipts_by_fleet": {str(size): dict(counts) for size, counts in sorted(repair_statuses.items())},
        "latency_boundary": "post-offer MAPPO rollout, optional certified menu construction, and selector",
    }


def _save_rows(output, rows):
    with (output / "comparison.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run(corpus: Path, training: Path, model_path: Path, output: Path) -> dict:
    from .m3_rllib import RLlibMAPPO

    training_receipt = read_json(training / "training.json")
    checkpoint = training / "policy.pt"
    config = _protocol(corpus, training_receipt)
    if digest(checkpoint) != training_receipt["checkpoint_sha256"]:
        raise ValueError("the MAPPO checkpoint does not match its training receipt")
    offer_protocol = read_json(OFFER_PROTOCOL)
    policy = RLlibMAPPO.load(checkpoint)
    factory = partial(OrderInvariantMenuModel, model_path, threads=config["model_torch_threads"])
    model = DeadlineModel(factory, reserve_s=config["model_fallback_reserve_s"])
    rows, repair_statuses = [], defaultdict(Counter)
    try:
        model.warmup(timeout_s=90)
        _graph()
        first = next(_worlds(config, corpus))
        scenario, dispatch, _, _, alert = _state(first, offer_protocol)
        baseline = _core(scenario, dispatch, policy.choose)
        menu = _menu(scenario, {"continue_mappo": baseline, "route_repair": baseline})
        model(DecisionView(_freeze(alert), menu), timeout_s=20)
        output.mkdir(parents=True, exist_ok=False)
        for filename in ("policy.pt", "training.json"):
            (output / filename).write_bytes((training / filename).read_bytes())
        write_json(output / "offer_protocol.json", offer_protocol)
        with (output / "traces.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for index, world in enumerate(_worlds(config, corpus), 1):
                trace, matched = _evaluate(world, config, offer_protocol, policy, model)
                _verify_selections(trace, config)
                stream.write(json.dumps(trace, allow_nan=False) + "\n")
                rows.extend(matched)
                receipt = trace["repair_receipt"]
                repair_statuses[world["fleet_size"]][f"{receipt['strategy']}:{receipt['solver_status']}"] += 1
                if index % 48 == 0:
                    stream.flush()
                    print(f"{index} matched cases completed", flush=True)
    finally:
        model.close()
    _save_rows(output, rows)
    summary = _summary(rows, repair_statuses)
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", {
        "schema": 1, "kind": "month_5_matched_ablation", "corpus_manifest_sha256": digest(corpus / "manifest.json"),
        "model": read_json(model_path / "source.json"), "sha256": {name: digest(output / name) for name in ARTIFACTS},
        "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in SOURCE_FILES},
    })
    return summary


def _verify_selections(trace, config):
    menu, scores, timing, solo = (trace[key] for key in ("menu", "model_scores", "timing", "mappo_alone"))
    budget_ms = config["supervisor_budget_s"] * 1000
    if (set(timing) != {"offer_ms", "solo_ms", "planning_ms"} or
            not all(isfinite(x) and x >= 0 for x in timing.values()) or timing["solo_ms"] > timing["planning_ms"] or
            trace["preselection_planning_ms"] != timing["planning_ms"] or
            solo != {"status": "selected", "decision_id": "continue_mappo", "end_to_end_decision_ms": timing["solo_ms"]}):
        raise ValueError("saved MAPPO timing or choice differs from the recorded planning boundary")
    rank = min(menu, key=lambda c: (-c["served"], c["travel_wh"], c["restricted_recovery_min"], c["id"]))["id"]
    agentic = trace["selections"]["agentic_supervisor"]
    worker_status = trace["worker_status"]
    if agentic["status"] == "selected":
        if worker_status != "completed" or scores.keys() != {c["id"] for c in menu} or not all(
            isfinite(value) for value in scores.values()
        ):
            raise ValueError("accepted model proposal has invalid worker status or scores")
        if min(scores, key=lambda key: (-scores[key], key)) != agentic["decision_id"]:
            raise ValueError("model proposal does not maximize saved order-balanced scores")
    elif scores or (worker_status == "not_called") != (agentic["status"] == "planning_timeout"):
        raise ValueError("rejected model request has accepted scores or inconsistent worker status")
    for name, choice in trace["selections"].items():
        elapsed, status = choice["elapsed_ms"], choice["status"]
        total = choice["end_to_end_decision_ms"]
        if (not isfinite(elapsed) or elapsed < 0 or not isclose(total, timing["planning_ms"] + elapsed) or
                choice["fallback_used"] != (status != "selected") or
                status not in {"selected", "planning_timeout", "timeout", "backend_error", "invalid_proposal"}):
            raise ValueError("saved supervisor selection has inconsistent timing or fallback")
        if status == "planning_timeout":
            expected = "continue_mappo"
            if timing["planning_ms"] < budget_ms or elapsed:
                raise ValueError("saved planning timeout differs from the decision budget")
        else:
            expected = agentic["decision_id"] if status == "selected" and name == "agentic_supervisor" else rank
            if status == "selected" and (total >= budget_ms or choice["proposed_id"] != choice["decision_id"]):
                raise ValueError("saved selected proposal violated the acceptance budget")
        if choice["decision_id"] != expected:
            raise ValueError("saved choice differs from the approved routing policy")


def verify(corpus: Path, output: Path) -> dict:
    from .m3_rllib import RLlibMAPPO

    manifest = read_json(output / "manifest.json")
    if (manifest["schema"] != 1 or manifest["kind"] != "month_5_matched_ablation" or
            manifest["corpus_manifest_sha256"] != digest(corpus / "manifest.json")):
        raise ValueError("saved evaluation uses a different corpus or schema")
    for filename in ARTIFACTS:
        if digest(output / filename) != manifest["sha256"][filename]:
            raise ValueError(f"experiment fingerprint differs for {filename}")
    for filename in SOURCE_FILES:
        if digest(Path(__file__).with_name(filename)) != manifest["source_sha256"][filename]:
            raise ValueError(f"experiment source fingerprint differs for {filename}")
    training = read_json(output / "training.json")
    checkpoint = output / "policy.pt"
    if digest(checkpoint) != training["checkpoint_sha256"]:
        raise ValueError("saved checkpoint differs from the frozen training receipt")
    config = _protocol(corpus, training)
    offer_protocol = read_json(output / "offer_protocol.json")
    policy = RLlibMAPPO.load(checkpoint)
    rows, repair_statuses = [], defaultdict(Counter)
    with (output / "traces.jsonl").open(encoding="utf-8") as stream:
        for world, line in zip(_worlds(config, corpus), stream, strict=True):
            trace = json.loads(line)
            scenario, dispatch, offers, fallbacks, alert = _state(world, offer_protocol)
            if (trace["world"] != world or any(trace[key] != value for key, value in _metadata(world).items()) or
                    (trace["alerts"], trace["alert"], trace["offers"], trace["fallbacks"]) !=
                    (world["alerts"], alert, offers, fallbacks)):
                raise ValueError("stored chaos input or shared commitments differ from regenerated case")
            candidates = trace["candidates"]
            if list(candidates) != world["menu_order"] or _core(scenario, dispatch, policy.choose) != candidates["continue_mappo"]:
                raise ValueError("saved menu order or frozen policy rollout differs during replay")
            _certify_record(scenario, dispatch, candidates["route_repair"])
            receipt = trace["repair_receipt"]
            if (receipt["certified_metrics"] != candidates["route_repair"]["result"]["metrics"] or
                    trace["menu"] != [asdict(item) for item in _menu(scenario, candidates)]):
                raise ValueError("saved menu or solver receipt differs from certified outcomes")
            _verify_selections(trace, config)
            repair_statuses[world["fleet_size"]][f"{receipt['strategy']}:{receipt['solver_status']}"] += 1
            rows.extend(_rows(world, scenario, offers, candidates, trace["mappo_alone"], trace["selections"],
                              trace["timing"], trace["worker_status"], config["supervisor_budget_s"] * 1000))
    with (output / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("saved comparison has a different column schema")
        for saved, rebuilt in zip(reader, rows, strict=True):
            if saved != {key: "" if value is None else str(value) for key, value in rebuilt.items()}:
                raise ValueError("saved comparison differs from replay")
    summary = _summary(rows, repair_statuses)
    if summary != read_json(output / "summary.json"):
        raise ValueError("saved summary differs from replay")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("run")
    evaluate.add_argument("corpus", type=Path)
    evaluate.add_argument("training", type=Path)
    evaluate.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    evaluate.add_argument("--output", type=Path)
    check = commands.add_parser("verify")
    check.add_argument("corpus", type=Path)
    check.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "run":
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            output = args.output or Path("runs") / f"m5_eval_{stamp}"
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
