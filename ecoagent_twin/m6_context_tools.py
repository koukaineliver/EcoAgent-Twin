"""Run read-only context tools before certified selection in a separate synthetic case."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import ceil, isfinite
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import TypedDict
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
from xml.etree import ElementTree

from .m2_sumo_bridge import apply_travel_matrix


DEFAULT_CASE = "301-light_traffic-locker_outage-moderate-v2"
WEATHER_VARIABLES = ("temperature_2m", "precipitation", "weather_code", "wind_speed_10m")
PUBLIC_WEATHER_URL = "https://api.open-meteo.com/v1/forecast?" + urlencode({
    "latitude": 34.0522,
    "longitude": -118.2437,
    "current": ",".join(WEATHER_VARIABLES),
    "timezone": "GMT",
    "timeformat": "unixtime",
})
SUMO_VERSION = "SUMO 1.28.0"
ADDED_DELAY_AUX_W = 350


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _verified(path: Path, fingerprint: str) -> None:
    if _sha(path) != fingerprint:
        raise ValueError(f"{path.name} does not match its archived fingerprint")


def _positive(value) -> bool:
    return type(value) in (int, float) and isfinite(value) and value > 0


def _weather_receipt(raw: bytes, fetched_at: datetime) -> dict:
    weather = json.loads(raw)
    reading = weather["current"]
    valid_at = reading["time"]
    if type(valid_at) is not int or not -60 <= fetched_at.timestamp() - valid_at <= 45 * 60:
        raise ValueError("weather model timestamp is outside the current-data window")
    units = weather["current_units"]
    values = {variable: reading[variable] for variable in WEATHER_VARIABLES}
    if (
        weather["utc_offset_seconds"] != 0
        or units["time"] != "unixtime"
        or units["temperature_2m"] != "°C"
        or units["precipitation"] != "mm"
        or units["weather_code"] != "wmo code"
        or units["wind_speed_10m"] != "km/h"
        or any(type(value) not in (float, int) or not isfinite(value) for value in values.values())
        or min(values["precipitation"], values["wind_speed_10m"], values["weather_code"]) < 0
        or type(values["weather_code"]) is not int
    ):
        raise ValueError("unexpected weather values, units, or time base")
    return {
        "status": "available",
        "role": "advisory_only",
        "source": "Open-Meteo current model conditions at a fixed public Los Angeles reference point",
        "request_url": PUBLIC_WEATHER_URL,
        "response_sha256": sha256(raw).hexdigest(),
        "response_json": raw.decode("utf-8"),
        "fetched_utc": fetched_at.isoformat(),
        "valid_utc": datetime.fromtimestamp(valid_at, tz=timezone.utc).isoformat(),
        "values": values,
        "units": {variable: units[variable] for variable in WEATHER_VARIABLES},
    }


def query_weather() -> dict:
    request = Request(PUBLIC_WEATHER_URL, headers={"User-Agent": "EcoAgent-Twin research demo"})
    with build_opener(_NoRedirect).open(request, timeout=8) as response:
        raw = response.read(65_537)
    if len(raw) > 65_536:
        raise ValueError("weather response exceeds the bounded size")
    return _weather_receipt(raw, datetime.now(timezone.utc))


def _road_index(network: Path) -> dict[str, tuple[str, str]]:
    return {
        edge.attrib["id"]: (edge.attrib["from"], edge.attrib["to"])
        for edge in ElementTree.parse(network).getroot().findall("edge") if "from" in edge.attrib
    }


def _sumo_worker(network: Path, routes: Path, seed: int, second: int, nodes: tuple[str, ...]) -> dict:
    from .m2_sumo_runner import _binaries

    _, binary, home = _binaries()
    if home is not None:
        sys.path.insert(0, str(home / "tools"))
    import traci

    traci.start([
        str(binary), "--net-file", str(network), "--route-files", str(routes), "--seed", str(seed),
        "--no-step-log", "true", "--no-warnings", "true",
    ], label="context", verbose=False, numRetries=4, stdout=subprocess.DEVNULL)
    connection = traci.getConnection("context")
    try:
        connection.simulationStep(second)
        observed_at = connection.simulation.getTime()
        api, version = connection.getVersion()
        edges = {
            edge: {
                "vehicle_count": connection.edge.getLastStepVehicleNumber(edge),
                "mean_speed_m_s": connection.edge.getLastStepMeanSpeed(edge),
                "travel_time_s": connection.edge.getTraveltime(edge),
            }
            for edge in _road_index(network)
        }
        arcs = []
        for origin in nodes:
            for destination in nodes:
                if origin == destination:
                    continue
                route = connection.simulation.findRoute(
                    f"access_{origin}_out", f"access_{destination}_in", "probe_ev",
                )
                arcs.append({
                    "origin": origin, "destination": destination,
                    "edges": list(route.edges), "sumo_route_time_s": route.travelTime,
                })
        if connection.simulation.getTime() != observed_at:
            raise ValueError("SUMO advanced while reading the snapshot")
        return {
            "status": "available", "source": "Local SUMO TraCI simulation",
            "api_version": api, "version": version, "seed": seed,
            "simulation_second": observed_at, "edges": edges, "routes": arcs,
        }
    finally:
        connection.close()


def query_traffic(
    python: Path, network: Path, routes: Path, seed: int, second: int, nodes: tuple[str, ...],
) -> dict:
    worker = subprocess.run([
        str(python.resolve(strict=True)), "-B", "-m", "ecoagent_twin.m6_context_tools", "_sumo",
        "--network", str(network), "--routes", str(routes),
        "--seed", str(seed), "--second", str(second), "--nodes", *nodes,
    ], capture_output=True, encoding="utf-8", timeout=25,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if worker.returncode != 0:
        raise ValueError("local SUMO TraCI worker did not finish successfully")
    rows = [line for line in worker.stdout.splitlines() if line.startswith("SUMO_RECEIPT ")]
    if len(rows) != 1:
        raise ValueError("SUMO TraCI worker did not return exactly one receipt")
    return json.loads(rows[0].removeprefix("SUMO_RECEIPT "))


def _updated_world(
    raw: dict, traffic: dict, roads: dict[str, tuple[str, str]], seed: int, second: int,
) -> tuple[dict, list[dict]]:
    if (
        traffic["version"] != SUMO_VERSION or type(traffic["seed"]) is not int or traffic["seed"] != seed
        or not _positive(traffic["simulation_second"]) or traffic["simulation_second"] != second
    ):
        raise ValueError("SUMO version, seed or simulation clock differs from the requested snapshot")
    edges = traffic["edges"]
    if edges.keys() != roads.keys() or any(
        not _positive(row["travel_time_s"])
        or type(row["vehicle_count"]) is not int or row["vehicle_count"] < 0
        or type(row["mean_speed_m_s"]) not in (int, float)
        or not isfinite(row["mean_speed_m_s"]) or row["mean_speed_m_s"] < 0
        for row in edges.values()
    ):
        raise ValueError("SUMO edge measurements do not cover the verified network")
    nodes = raw["travel_minutes"].keys()
    past_window = raw["horizon_min"] - raw["outage"]["time_min"] + 1
    verified = set()
    updated, changes = [], []
    for row in traffic["routes"]:
        origin, destination, route = row["origin"], row["destination"], row["edges"]
        arc = origin, destination
        if (
            origin not in nodes or destination not in nodes or origin == destination or arc in verified
            or not route or len(route) != len(set(route)) or not _positive(row["sumo_route_time_s"])
            or route[0] != f"access_{origin}_out" or route[-1] != f"access_{destination}_in"
            or not all(edge in roads for edge in route)
            or any(roads[left][1] != roads[right][0] for left, right in zip(route, route[1:]))
        ):
            raise ValueError("invalid or duplicate directed route in the SUMO receipt")
        verified.add(arc)
        old_min = raw["travel_minutes"][origin][destination]
        old_wh = raw["travel_wh"][origin][destination]
        snapshot_seconds = max(row["sumo_route_time_s"], sum(edges[edge]["travel_time_s"] for edge in route))
        if not _positive(snapshot_seconds):
            raise ValueError("nonfinite directed estimate in the SUMO receipt")
        snapshot_min = ceil(snapshot_seconds / 60)
        minutes = max(old_min, min(snapshot_min, past_window))
        energy = old_wh + ceil(ADDED_DELAY_AUX_W * (minutes - old_min) / 60)
        updated.append({"origin": origin, "destination": destination, "travel_min": minutes, "travel_wh": energy})
        if minutes != old_min:
            changes.append({**updated[-1], "archived_min": old_min, "archived_wh": old_wh,
                            "snapshot_travel_estimate_s": snapshot_seconds,
                            "outside_dispatch_window": snapshot_min >= past_window})
    return apply_travel_matrix(raw, updated), changes


class _State(TypedDict, total=False):
    tools: list[dict]
    weather: dict
    traffic: dict
    plan: dict
    decision: dict


def _run_graph(weather, traffic, planning, selection) -> _State:
    """Invoke only the fixed tool sequence and stop before choice on verification failure."""
    from langgraph.graph import END, START, StateGraph
    from langsmith import tracing_context

    def call(name, field, invoke):
        def run(state: _State) -> _State:
            started = perf_counter()
            try:
                result = invoke(state)
            except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
                status = "unavailable" if field == "weather" else "held"
                result = {"status": status, "failure_type": type(error).__name__}
                if field == "weather":
                    result.update(role="advisory_only", request_url=PUBLIC_WEATHER_URL)
            return {
                **{field: result},
                "tools": [*state["tools"], {"tool": name, "status": result["status"],
                                          "elapsed_ms": round((perf_counter() - started) * 1000, 3)}],
            }
        return run

    graph = StateGraph(_State)
    graph.add_node("get_weather", call("get_weather", "weather", lambda _: weather()))
    graph.add_node("query_traffic_twin", call("query_traffic_twin", "traffic", lambda _: traffic()))
    graph.add_node("certify_route_repair", call("certify_route_repair", "plan", planning))
    graph.add_node("choose_approved_route", call("choose_approved_route", "decision", selection))
    graph.add_edge(START, "get_weather")
    graph.add_edge("get_weather", "query_traffic_twin")
    graph.add_conditional_edges(
        "query_traffic_twin", lambda state: state["traffic"]["status"] == "available",
        {True: "certify_route_repair", False: END},
    )
    graph.add_conditional_edges(
        "certify_route_repair", lambda state: state["plan"]["status"] == "certified",
        {True: "choose_approved_route", False: END},
    )
    graph.add_edge("choose_approved_route", END)
    with tracing_context(enabled=False):
        return graph.compile().invoke({"tools": []})


def run_context(
    archive: Path, corpus: Path, sumo_python: Path, *, case_id: str = DEFAULT_CASE,
    snapshot_second: int = 180, model_dir: Path | None = None,
) -> dict:
    from .m3_experiment import _certify_record, _core, _menu, _repair
    from .m3_rllib import RLlibMAPPO
    from .m3_supervisor import Candidate, supervise
    from .m4_xai import _load_case
    from .m5_deadline import DeadlineModel
    from .m5_experiment import _state
    from .m5_local_model import OrderInvariantMenuModel
    from .model import Scenario

    if type(snapshot_second) is not int or not 1 <= snapshot_second <= 720:
        raise ValueError("the local SUMO snapshot must be an integer second in [1, 720]")
    archive, corpus = archive.resolve(strict=True), corpus.resolve(strict=True)
    archive_manifest = json.loads((archive / "manifest.json").read_bytes())
    if archive_manifest["kind"] != "month_5_matched_ablation":
        raise ValueError("runtime smoke requires an archived Month 5 matched synthetic case")
    _verified(corpus / "manifest.json", archive_manifest["corpus_manifest_sha256"])
    corpus_manifest = json.loads((corpus / "manifest.json").read_bytes())
    _verified(corpus / "protocol.json", corpus_manifest["sha256"]["protocol.json"])
    _verified(archive / "offer_protocol.json", archive_manifest["sha256"]["offer_protocol.json"])
    _verified(archive / "policy.pt", archive_manifest["sha256"]["policy.pt"])
    prior, source = _load_case(archive, case_id)
    world = prior["world"]
    if world["fleet_size"] != 2 or world["chaos_kind"] != "locker_outage":
        raise ValueError("choose an ordinary two-EV locker-outage case")
    protocol = json.loads((corpus / "protocol.json").read_bytes())
    offer_protocol = json.loads((archive / "offer_protocol.json").read_bytes())
    scenario, dispatch, offers, fallbacks, alert = _state(world, offer_protocol)
    if offers != prior["offers"] or fallbacks != prior["fallbacks"] or alert != prior["alert"]:
        raise ValueError("archived customer commitments did not reproduce")
    seed, condition = world["traffic_seed"], world["traffic"]
    if condition not in ("light_traffic", "heavy_traffic") or type(seed) is not int:
        raise ValueError("unexpected SUMO profile in the archived case")
    measurement = corpus / "sumo" / str(seed)
    _verified(measurement / "manifest.json", corpus_manifest["measurements_sha256"][str(seed)])
    measurement_manifest = json.loads((measurement / "manifest.json").read_bytes())
    filenames = ("network.net.xml", "traffic.rou.xml")
    paths = tuple(measurement / condition / name for name in filenames)
    for path in paths:
        _verified(path, measurement_manifest["sha256"][f"{condition}/{path.name}"])
    roads = _road_index(paths[0])
    raw = world["snapshot"]
    nodes = tuple(sorted(raw["travel_minutes"]))
    if len(nodes) != 7 or set(nodes) != {raw["depot"], *raw["locker_free_capacity"], *(v["node"] for v in raw["vehicles"])}:
        raise ValueError("runtime smoke expects the seven unreplicated physical SUMO sites")
    policy = RLlibMAPPO.load(archive / "policy.pt")

    def plan(state: _State) -> dict:
        updated, changes = _updated_world(raw, state["traffic"], roads, seed, snapshot_second)
        costs = Scenario.from_dict(updated)
        active = replace(scenario, travel_time=costs.travel_time, travel_energy=costs.travel_energy)
        continued = _core(active, dispatch, policy.choose)
        repaired, repair = _repair(active, dispatch, protocol["repair"])
        records = {"continue_mappo": continued, "route_repair": repaired}
        for record in records.values():
            _certify_record(active, dispatch, record)
        if repair["fallback_reason"] is not None:
            raise ValueError("route repair did not produce its own certified plan")
        return {
            "status": "certified", "matrix_sha256": sha256(json.dumps({
                field: updated[field] for field in ("travel_minutes", "travel_wh")
            }, sort_keys=True).encode()).hexdigest(),
            "changed_arcs": changes, "solver_receipt": repair,
            "menu": [asdict(entry) for entry in _menu(active, records)], "candidates": records,
        }

    model = None
    if model_dir is not None:
        model = DeadlineModel(lambda: OrderInvariantMenuModel(model_dir, threads=2))
    try:
        if model is not None:
            model.warmup(timeout_s=45)
        result = _run_graph(
            query_weather,
            lambda: query_traffic(sumo_python, *paths, seed, snapshot_second, nodes),
            plan,
            lambda state: supervise(
                alert, tuple(Candidate(**entry) for entry in state["plan"]["menu"]),
                model=model, budget_s=protocol["supervisor_budget_s"],
            ),
        )
        scores = model.accepted_scores(result.get("decision", {})) if model is not None else {}
    finally:
        if model is not None:
            model.close()
    return {
        "schema": 1,
        "scope": "Runtime tool-use smoke on one synthetic scene, separate from historical evaluations",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "source": {**source, "corpus_manifest_sha256": archive_manifest["corpus_manifest_sha256"],
                   "sumo_input_sha256": {path.name: _sha(path) for path in paths}},
        "travel_update": (
            "Max of archived minutes, SUMO route estimate and summed current edge estimates, rounded up. "
            "A current estimate beyond the full response window is truncated to one minute above that "
            "window so the arc is infeasible within the certified plan. "
            f"Added minutes incur {ADDED_DELAY_AUX_W} W modeled moving auxiliary energy."
        ),
        "authority": "Fixed graph calls tools. The chooser selects a certified menu ID. No fleet action is applied.",
        "weather_policy": "Current public weather is advisory and has no effect on costs, constraints or scores.",
        "model_scores": scores,
        **result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True, dest="operation")
    run = commands.add_parser("run")
    run.add_argument("archive", type=Path)
    run.add_argument("corpus", type=Path)
    run.add_argument("--sumo-python", required=True, type=Path)
    run.add_argument("--case", default=DEFAULT_CASE)
    run.add_argument("--second", default=180, type=int)
    run.add_argument("--model", type=Path)
    run.add_argument("--output", required=True, type=Path)
    worker = commands.add_parser("_sumo")
    worker.add_argument("--network", required=True, type=Path)
    worker.add_argument("--routes", required=True, type=Path)
    worker.add_argument("--seed", required=True, type=int)
    worker.add_argument("--second", required=True, type=int)
    worker.add_argument("--nodes", required=True, nargs="+")
    args = parser.parse_args()
    if args.operation == "_sumo":
        print("SUMO_RECEIPT " + json.dumps(
            _sumo_worker(args.network, args.routes, args.seed, args.second, tuple(args.nodes)), allow_nan=False,
        ))
        return
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(path.resolve()) for path in (args.archive, args.corpus)):
        parser.error("output must be a new file outside the existing evidence directories")
    receipt = run_context(args.archive, args.corpus, args.sumo_python, case_id=args.case,
                          snapshot_second=args.second, model_dir=args.model)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "tools": receipt["tools"], "decision": receipt.get("decision")}, indent=2))


if __name__ == "__main__":
    main()
