"""Run matched canonical EcoAgent-Twin controllers on an OSMnx Los Angeles road graph."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
import json
from math import fsum, isfinite
from pathlib import Path
from statistics import fmean
from time import perf_counter

from .m1_benchmark import _perturb
from .m2_offer_experiment import DEFAULT_PROTOCOL as OFFER_PROTOCOL
from .m2_rule import RuleBased
from .m2_simulator import OutageEnv
from .m3_cases import digest, read_json, write_json
from .m3_experiment import DEFAULT_MODEL, _certify_record, _core, _menu, _repair
from .m3_local_model import LocalMenuModel
from .m3_supervisor import DecisionView, _freeze, _graph
from .m4_chaos import inject
from .m4_experiment import ARMS, SEVERITIES, _metadata, _row, _take_decision, _verify_choice
from .m5_deadline import DeadlineModel
from .m5_experiment import FIELDS as M5_FIELDS, SOURCE_FILES as M5_SOURCES
from .m5_experiment import _arm_stats, _rows, _state, _summary, _verify_selections
from .m5_local_model import OrderInvariantMenuModel
from .m5_scale import FLEET_SIZES, scale_workload
from .model import Scenario
from .replay import DEFAULT_SCENARIO


PROFILES = ("light_traffic", "heavy_traffic")
WORKLOAD_SEEDS = (4011, 4012, 4013, 4014)
SCALE_SEEDS = (4011, 4014)
RULE_ARM = "rule_based"
M4_ARM = "mappo_m4_agentic_supervisor"
FIELDS = (*M5_FIELDS, "input_sha256", "network_distance_m")
ARTIFACTS = ("policy.pt", "training.json", "offer_protocol.json", "protocol.json", "traces.jsonl",
             "comparison.csv", "summary.json")


def _json_digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _protocol(training, layout):
    config = read_json(DEFAULT_SCENARIO.with_name("m5_campaign.json"))
    heldout = set(WORKLOAD_SEEDS) - {*training["training_traffic_seeds"], *training["validation_traffic_seeds"]}
    if heldout != set(WORKLOAD_SEEDS):
        raise ValueError("LA case-generation seeds overlap frozen policy training or validation")
    return {**config, "id": "la-osmnx-frozen-transfer-v1", "traffic_seeds": list(WORKLOAD_SEEDS),
            "scale_traffic_seeds": list(SCALE_SEEDS), "geometry_layout": layout,
            "population": "synthetic parcels, recipients, shocks and speed profiles, same real OSM road graph",
            "policy_scope": "frozen validated SUMO-trained MAPPO evaluated out of domain; weights never updated",
            "ablation_scope": "canonical M4 positional and M5 balanced local-model selectors share certified menu",
            "scalability_scope": (
                "canonical M5 logical banks co-located at original OSM nodes" if layout == "colocated" else
                "replicated M5 parcel relationships with separately modeled physical origins and locker sites"
            )}


def _profile(network, raw, name):
    profile = network["profiles"][name]
    nodes = {raw["depot"], *raw["locker_free_capacity"], *(row["node"] for row in raw["vehicles"])}
    for field in ("travel_minutes", "travel_wh", "distance_m", "routes"):
        table = profile[field]
        for start in sorted(nodes):
            for end in sorted(nodes):
                if start not in table or end not in table[start]:
                    raise ValueError(f"profile {name} misses required {field} {start} -> {end}")
                value = table[start][end]
                if field == "distance_m" and (type(value) not in (int, float) or not isfinite(value)
                                               or value < 0 or (start == end and value != 0)):
                    raise ValueError("profile contains an invalid distance")
                if field == "routes" and (not isinstance(value, list) or (start != end and len(value) < 2)):
                    raise ValueError("profile contains an invalid graph path")
    return profile, nodes


def _apply_profile(network, raw, name):
    profile, nodes = _profile(network, raw, name)
    return {**raw, **{field: {start: {end: profile[field][start][end] for end in sorted(nodes)}
                             for start in sorted(nodes)} for field in ("travel_minutes", "travel_wh")}}


def _worlds(network, config):
    base = read_json(DEFAULT_SCENARIO)
    if set(network["profiles"]) != set(PROFILES):
        raise ValueError("expected the two registered modeled OSMnx speed profiles")
    index = 0
    for workload in config["traffic_seeds"]:
        for traffic_index, profile in enumerate(PROFILES):
            base_profile = _apply_profile(network, base, profile)
            for severity_index, severity in enumerate(SEVERITIES):
                seed = config["case_seeds"][severity]
                common = _perturb(base_profile, config, workload, seed)
                for kind_index, kind in enumerate(config["chaos_kinds"]):
                    raw, charger, alerts = inject(
                        common, kind, severity, workload * 1000 + seed,
                        charger_rate_wh_per_min=config["charger_rate_wh_per_min"],
                        charger_capacity_wh=config["charger_capacity_wh"],
                    )
                    case_id = f"la-{workload}-{profile}-{kind}-{severity}"
                    options = ["continue_mappo", "route_repair"]
                    if (index // (len(PROFILES) * len(SEVERITIES) * len(config["chaos_kinds"])) +
                            traffic_index + severity_index + kind_index) % 2:
                        options.reverse()
                    panel = workload in config["scale_traffic_seeds"]
                    sizes = FLEET_SIZES if panel else FLEET_SIZES[:1]
                    offset = index % len(sizes)
                    index += 1
                    for size in (*sizes[offset:], *sizes[:offset]):
                        snapshot, scaling = scale_workload(raw, size)
                        road_nodes = scaling["node_origins"]
                        if config["geometry_layout"] == "distinct":
                            snapshot, alerts_at_scale, road_nodes = _distinct_world(
                                network, profile, common, snapshot, scaling, deepcopy(alerts),
                            )
                            charger_at_scale = charger
                        else:
                            charger_at_scale, alerts_at_scale = charger, alerts
                        identifier = f"{case_id}-v{size}"
                        snapshot["id"] = identifier
                        yield {"case_id": identifier, "base_case_id": case_id, "fleet_size": size,
                               "traffic": profile, "traffic_seed": workload, "chaos_kind": kind, "severity": severity,
                               "case": {"day": workload, "seed": seed, "scenario": f"{kind}:{severity}"},
                               "snapshot": snapshot, "charger": charger_at_scale, "alerts": alerts_at_scale,
                               "menu_order": options, "scale_panel": panel, "scaling": scaling,
                               "road_nodes": road_nodes}


def _distinct_world(network, profile, raw, snapshot, scaling, alerts):
    # Ensure two-EV worlds exactly match the canonical ablation. For larger
    # worlds retain replicated workloads, replacing their path costs with the
    # physical coordinates supplied for each additional origin and locker.
    if scaling["fleet_size"] == FLEET_SIZES[0]:
        return snapshot, alerts, scaling["node_origins"]
    original_wh = {vehicle["id"]: vehicle["battery_wh"] for vehicle in raw["vehicles"]}
    origins = {new: old for clone in scaling["replicas"] for old, new in clone["vehicles"].items()}
    for vehicle in snapshot["vehicles"]:
        vehicle["battery_wh"] = original_wh[origins[vehicle["id"]]]
    snapshot = _apply_profile(network, snapshot, profile)
    chargers = next((event for event in alerts if event["kind"] == "charger_outage"), None)
    if chargers:
        changes = {item["vehicle_id"]: item for item in chargers["payload"]["battery_changes"]}
        restored = []
        for vehicle in snapshot["vehicles"]:
            old = changes[origins[vehicle["id"]]]
            margin = old["battery_after_wh"] - old["direct_return_and_reserve_wh"]
            direct = vehicle["reserve_wh"] + snapshot["travel_wh"][vehicle["node"]][snapshot["depot"]]
            before = vehicle["battery_wh"]
            vehicle["battery_wh"] = min(before, direct + max(0, margin))
            restored.append({"vehicle_id": vehicle["id"], "battery_before_wh": before,
                             "battery_after_wh": vehicle["battery_wh"], "direct_return_and_reserve_wh": direct})
        chargers["payload"]["battery_changes"] = restored
    vehicle_nodes = {v["id"]: v["node"] for v in snapshot["vehicles"]}
    for event in alerts:
        if event["kind"] != "road_blockage":
            continue
        replacements = []
        for row in event["payload"]["replacements"]:
            for clone in scaling["replicas"]:
                mappings = {**clone["lockers"], raw["depot"]: raw["depot"]}
                mappings.update({item["node"]: vehicle_nodes[clone["vehicles"][item["id"]]]
                                 for item in raw["vehicles"]})
                origin, via, destination = (mappings[row[label]] for label in ("origin", "via", "destination"))
                times, energy = snapshot["travel_minutes"], snapshot["travel_wh"]
                before = times[origin][destination]
                changed = times[origin][via] + times[via][destination]
                if changed <= before:
                    continue
                before_wh = energy[origin][destination]
                changed_wh = energy[origin][via] + energy[via][destination]
                times[origin][destination] = changed
                energy[origin][destination] = changed_wh
                replacements.append({"origin": origin, "via": via, "destination": destination,
                                     "travel_before_min": before, "travel_after_min": changed,
                                     "traction_before_wh": before_wh, "traction_after_wh": changed_wh})
        event["payload"]["replacements"] = replacements
    return snapshot, alerts, {node: node for node in snapshot["travel_minutes"]}


def _distance_receipt(record, scenario, dispatch, world, profile):
    """Link simulator actions to the matching directed OSM route or modeled detour."""
    environment = OutageEnv(scenario, **dispatch)
    locations = {vehicle.id: vehicle.node for vehicle in scenario.vehicles}
    direct = profile["distance_m"]
    alias = world["road_nodes"]
    detours = {(patch["origin"], patch["destination"]): patch["via"]
               for alert in world["alerts"] if alert["kind"] == "road_blockage"
               for patch in alert["payload"]["replacements"]}
    legs = []
    for saved in record["actions"]:
        decision = environment.observation()
        if (decision.vehicle_id, decision.time_min) != (saved["vehicle_id"], saved["time_min"]):
            raise ValueError("distance action differs from the canonical replay")
        action = decision.actions[saved["action_index"]]
        vehicle = decision.vehicle_id
        start = locations[vehicle]
        end = action.locker_id if action.kind == "deposit" else scenario.depot
        if (start, end) not in detours:
            # M5 copies affected costs between logical banks at the same sites.
            via = detours.get((alias[start], alias[end]))
        else:
            via = detours[start, end]
        nodes = [alias[start], alias[end]] if via is None else [alias[start], alias[via], alias[end]]
        meters = fsum(direct[a][b] for a, b in zip(nodes, nodes[1:]))
        legs.append({"vehicle_id": vehicle, "time_min": decision.time_min, "origin": start, "destination": end,
                     "action": asdict(action), "osm_route_nodes": nodes, "network_distance_m": meters})
        locations[vehicle] = end
        environment.step(saved["action_index"])
    if environment.result() != record["result"]:
        raise ValueError("distance receipt action replay differs from stored outcome")
    return {"legs": legs, "network_distance_m": fsum(leg["network_distance_m"] for leg in legs)}


def _input_hash(world, dispatch, offers):
    return _json_digest({"snapshot": world["snapshot"], "case": world["case"], "dispatch": dispatch, "offers": offers})


def _choice(label, ms):
    return {"status": "selected", "decision_id": label, "end_to_end_decision_ms": ms}


def _case_rows(world, scenario, trace, config):
    timing, offers, scope = trace["timing"], trace["offers"], trace["route_distances"]
    budget = config["supervisor_budget_s"] * 1000
    matched = _rows(world, scenario, offers, trace["candidates"], trace["mappo_alone"], trace["selections"],
                    timing, trace["worker_status"], budget)
    visits = [(row, row["selected_intervention"]) for row in matched]
    extra = [(RULE_ARM, _choice(RULE_ARM, trace["rule_ms"]), trace["rule_rollout"], trace["rule_ms"])]
    if trace["m4_agentic"] is not None:
        m4 = trace["m4_agentic"]
        extra.append((M4_ARM, m4, trace["candidates"][m4["decision_id"]], timing["planning_ms"]))
    for arm, choice, route, planning_ms in extra:
        row = {**_row(world, scenario, route, arm, choice, offers),
               "base_case_id": world["base_case_id"], "fleet_size": world["fleet_size"],
               "locker_count": world["scaling"]["locker_count"], "open_locker_slots": world["scaling"]["open_slots"],
               "scale_panel": world["scale_panel"], "offer_ms": timing["offer_ms"], "preselection_ms": planning_ms,
               "selector_ms": choice.get("elapsed_ms", 0.0), "worker_status": "",
               "over_budget": choice["end_to_end_decision_ms"] >= budget}
        visits.append((row, choice["decision_id"]))
    return [{**row, "input_sha256": trace["input_sha256"], "network_distance_m": scope[label]["network_distance_m"]}
            for row, label in visits]


def _evaluate(world, network, config, protocol, policy, m4_model, m5_model):
    started = perf_counter()
    scenario, dispatch, offers, fallbacks, alert = _state(world, protocol)
    offer_ms = 1000 * (perf_counter() - started)
    started = perf_counter()
    rule = _core(scenario, dispatch, RuleBased(scenario).choose)
    rule_ms = 1000 * (perf_counter() - started)
    started = perf_counter()
    baseline = _core(scenario, dispatch, policy.choose)
    solo_ms = 1000 * (perf_counter() - started)
    repaired, receipt = _repair(scenario, dispatch, config["repair"])
    options = {"continue_mappo": baseline, "route_repair": repaired}
    candidates = {name: options[name] for name in world["menu_order"]}
    menu = _menu(scenario, candidates)
    planning_ms = 1000 * (perf_counter() - started)
    budget = config["supervisor_budget_s"]
    deterministic = _take_decision(alert, menu, budget, planning_ms)
    m5 = _take_decision(alert, menu, budget, planning_ms, model=m5_model)
    worker_status = "not_called" if m5["status"] == "planning_timeout" else m5_model.last_status
    m4, m4_scores = None, None
    if world["fleet_size"] == FLEET_SIZES[0]:
        m4_model.last_scores = {}
        m4 = _take_decision(alert, menu, budget, planning_ms, model=m4_model)
        m4_scores = m4_model.last_scores.copy() if m4["status"] == "selected" else {}
    roads = network["profiles"][world["traffic"]]
    distances = {name: _distance_receipt(record, scenario, dispatch, world, roads)
                 for name, record in {RULE_ARM: rule, **candidates}.items()}
    trace = {**_metadata(world), "world": world, "alert": alert, "dispatch": dispatch, "offers": offers,
             "fallbacks": fallbacks, "input_sha256": _input_hash(world, dispatch, offers),
             "rule_rollout": rule, "rule_ms": rule_ms, "candidates": candidates, "repair_receipt": receipt,
             "menu": [asdict(item) for item in menu], "mappo_alone": _choice("continue_mappo", solo_ms),
             "selections": {"deterministic_supervisor": deterministic, "agentic_supervisor": m5},
             "m4_agentic": m4, "m4_model_scores": m4_scores, "model_scores": m5_model.accepted_scores(m5),
             "worker_status": worker_status, "timing": {"offer_ms": offer_ms, "solo_ms": solo_ms, "planning_ms": planning_ms},
             "preselection_planning_ms": planning_ms, "route_distances": distances}
    _verify_model(trace, config)
    return trace, _case_rows(world, scenario, trace, config)


def _verify_model(trace, config):
    _verify_selections(trace, config)
    m4 = trace["m4_agentic"]
    if m4 is not None:
        planning = trace["timing"]["planning_ms"]
        _verify_choice(m4, trace["menu"], planning, config["supervisor_budget_s"] * 1000,
                       trace["m4_model_scores"])


def _report(rows, statuses, config):
    panel = defaultdict(dict)
    for row in rows:
        episode = panel[row["case_id"]]
        if row["arm"] in episode:
            raise ValueError("duplicate arm in the matched LA case")
        episode[row["arm"]] = row
    for episode in panel.values():
        representative = episode[RULE_ARM]
        names = {RULE_ARM, *ARMS, *([M4_ARM] if representative["fleet_size"] == FLEET_SIZES[0] else [])}
        if set(episode) != names or len({row["input_sha256"] for row in episode.values()}) != 1:
            raise ValueError("LA controller selector arms did not share one input")
    m5_rows = [row for row in rows if row["arm"] in ARMS]
    rules = [row for row in rows if row["arm"] == RULE_ARM]
    m4 = [row for row in rows if row["arm"] == M4_ARM]

    def rates(group):
        return {**_arm_stats(group), "mean_network_distance_m": fmean(row["network_distance_m"] for row in group)}

    def pair(reference, test):
        values = [episode[test]["served"] - episode[reference]["served"] for episode in panel.values() if test in episode]
        return {"cases": len(values), "mean_served_difference": fmean(values),
                "wins": sum(n > 0 for n in values), "ties": sum(n == 0 for n in values), "losses": sum(n < 0 for n in values)}

    sizes = {str(size): {arm: rates([r for r in rows if r["scale_panel"] and r["fleet_size"] == size and r["arm"] == arm])
                         for arm in (RULE_ARM, *ARMS)} for size in FLEET_SIZES}
    return {"population": config["population"], "policy_scope": config["policy_scope"],
            "scalability_scope": config["scalability_scope"], "matched_cases": len(panel), "episode_rows": len(rows),
            "all_arms_share_snapshot_and_offers": True,
            "two_ev_ablation": {arm: rates([r for r in rows if r["fleet_size"] == 2 and r["arm"] == arm])
                                for arm in (RULE_ARM, *ARMS, M4_ARM)},
            "paired_all_fleets": {"mappo_minus_rule": pair(RULE_ARM, ARMS[0]),
                                  "deterministic_repair_supervisor_minus_mappo": pair(ARMS[0], ARMS[2]),
                                  "m5_agentic_minus_deterministic": pair(ARMS[2], ARMS[1]),
                                  "m4_agentic_minus_deterministic_two_ev": pair(ARMS[2], M4_ARM)},
            "scalability": sizes, "m5_canonical_summary": _summary(m5_rows, statuses),
            "rule_rows": len(rules), "m4_additional_rows": len(m4),
            "latency_scope": "local warm model, matched offer generation separate, planning and selector included"}


def _sources():
    names = {*M5_SOURCES, "la_controller_experiment.py", "m2_rule.py", "m2_customer_mesa.py", "m2_offers.py",
             "m2_offer_allocator.py", "model.py"}
    return {name: digest(Path(__file__).with_name(name)) for name in sorted(names)}


def _append(rows, statuses, trace, result):
    rows.extend(result)
    receipt = trace["repair_receipt"]
    statuses[trace["world"]["fleet_size"]][f"{receipt['strategy']}:{receipt['solver_status']}"] += 1


def run(network_path, training_path, model_path, output, layout):
    from .m3_rllib import RLlibMAPPO

    network = read_json(network_path)
    training = read_json(training_path / "training.json")
    checkpoint = training_path / "policy.pt"
    if digest(checkpoint) != training["checkpoint_sha256"]:
        raise ValueError("frozen MAPPO checkpoint fingerprint differs from training")
    policy = RLlibMAPPO.load(checkpoint)
    config = _protocol(training, layout)
    protocol = read_json(OFFER_PROTOCOL)
    trials = list(_worlds(network, config))
    if len(trials) != 192 or len({world["case_id"] for world in trials}) != len(trials):
        raise ValueError("LA crossed design is incomplete")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", config)
    write_json(output / "offer_protocol.json", protocol)
    for file in ("policy.pt", "training.json"):
        (output / file).write_bytes((training_path / file).read_bytes())
    model_path = Path(model_path)
    m4_model = LocalMenuModel(model_path, threads=config["model_torch_threads"])
    m5_model = DeadlineModel(partial(OrderInvariantMenuModel, model_path, threads=config["model_torch_threads"]),
                             reserve_s=config["model_fallback_reserve_s"])
    rows, statuses = [], defaultdict(Counter)
    try:
        m5_model.warmup(timeout_s=90)
        _graph()
        scenario, dispatch, _, _, alert = _state(trials[0], protocol)
        baseline = _core(scenario, dispatch, policy.choose)
        view = DecisionView(_freeze(alert), _menu(scenario, {"continue_mappo": baseline, "route_repair": baseline}))
        m4_model(view, timeout_s=30)
        m5_model(view, timeout_s=30)
        with (output / "traces.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for i, world in enumerate(trials, 1):
                trace, matched = _evaluate(world, network, config, protocol, policy, m4_model, m5_model)
                stream.write(json.dumps(trace, allow_nan=False) + "\n")
                _append(rows, statuses, trace, matched)
                if i % 24 == 0:
                    stream.flush()
                    print(f"{i} matched Los Angeles controller cases completed", flush=True)
    finally:
        m5_model.close()
    with (output / "comparison.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = _report(rows, statuses, config)
    write_json(output / "summary.json", report)
    write_json(output / "manifest.json", {
        "schema": 1, "kind": "la_osmnx_frozen_controller_transfer", "created_utc": datetime.now(timezone.utc).isoformat(),
        "road_network_sha256": digest(network_path), "base_synthetic_scenario_sha256": digest(DEFAULT_SCENARIO),
        "model": read_json(model_path / "source.json"), "source_sha256": _sources(),
        "sha256": {name: digest(output / name) for name in ARTIFACTS},
    })
    return report


def verify(network_path, output):
    from .m3_rllib import RLlibMAPPO

    manifest = read_json(output / "manifest.json")
    if (manifest["kind"] != "la_osmnx_frozen_controller_transfer" or manifest["road_network_sha256"] != digest(network_path)
            or manifest["base_synthetic_scenario_sha256"] != digest(DEFAULT_SCENARIO)
            or manifest["source_sha256"] != _sources()
            or any(manifest["sha256"][name] != digest(output / name) for name in ARTIFACTS)):
        raise ValueError("LA run source or artifact fingerprint differs")
    network = read_json(network_path)
    config = read_json(output / "protocol.json")
    protocol = read_json(output / "offer_protocol.json")
    training = read_json(output / "training.json")
    if config != _protocol(training, config["geometry_layout"]) or digest(output / "policy.pt") != training["checkpoint_sha256"]:
        raise ValueError("LA experiment config or frozen checkpoint differs")
    policy = RLlibMAPPO.load(output / "policy.pt")
    rows, statuses = [], defaultdict(Counter)
    with (output / "traces.jsonl").open(encoding="utf-8") as stream:
        for world, line in zip(_worlds(network, config), stream, strict=True):
            trace = json.loads(line)
            scenario, dispatch, offers, fallbacks, alert = _state(world, protocol)
            if (trace["world"], trace["dispatch"], trace["offers"], trace["fallbacks"], trace["alert"], trace["input_sha256"]) != (
                world, dispatch, offers, fallbacks, alert, _input_hash(world, dispatch, offers)
            ):
                raise ValueError("LA shared case inputs or offer commitments differ from replay")
            candidates = trace["candidates"]
            if (_core(scenario, dispatch, policy.choose) != candidates["continue_mappo"]
                    or _core(scenario, dispatch, RuleBased(scenario).choose) != trace["rule_rollout"]):
                raise ValueError("LA frozen policy or rule prediction differs from replay")
            route = candidates["route_repair"]
            _certify_record(scenario, dispatch, route)
            if (trace["menu"] != [asdict(item) for item in _menu(scenario, candidates)]
                    or trace["repair_receipt"]["certified_metrics"] != route["result"]["metrics"]
                    or trace["repair_receipt"]["certified_unserved_parcel_ids"] != route["result"]["unserved_parcel_ids"]):
                raise ValueError("LA repair certificate or projected menu differs from replay")
            roads = network["profiles"][world["traffic"]]
            distances = {name: _distance_receipt(record, scenario, dispatch, world, roads)
                         for name, record in {RULE_ARM: trace["rule_rollout"], **candidates}.items()}
            if distances != trace["route_distances"]:
                raise ValueError("OSM network route distances differ from replay")
            _verify_model(trace, config)
            _append(rows, statuses, trace, _case_rows(world, scenario, trace, config))
    report = _report(rows, statuses, config)
    if report != read_json(output / "summary.json"):
        raise ValueError("LA results differ from verified rows")
    with (output / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        for stored, recomputed in zip(csv.DictReader(stream), rows, strict=True):
            if stored != {key: "" if value is None else str(value) for key, value in recomputed.items()}:
                raise ValueError("LA comparison rows differ from replay")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "verify"):
        sub = subs.add_parser(name)
        sub.add_argument("network", type=Path)
        sub.add_argument("output", type=Path)
        if name == "run":
            sub.add_argument("training", type=Path)
            sub.add_argument("--model", type=Path, default=DEFAULT_MODEL)
            sub.add_argument("--layout", choices=("colocated", "distinct"), default="colocated")
    args = parser.parse_args()
    report = (run(args.network, args.training, args.model, args.output, args.layout) if args.command == "run" else
              verify(args.network, args.output))
    print(json.dumps({name: report[name] for name in ("matched_cases", "episode_rows", "paired_all_fleets")}, indent=2))


if __name__ == "__main__":
    main()
