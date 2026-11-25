"""Import the University Park OSM snapshot and verify SUMO electric route probes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import file_digest
from heapq import heappop, heappush
from itertools import pairwise
import json
from math import asin, cos, radians, sin, sqrt
import os
from pathlib import Path
from statistics import fmean, median
from xml.etree import ElementTree as ET

from .m2_sumo_runner import _binaries, _invoke, _version, read_trips


SOURCE = Path("runs/osmnx_la_university_park/osm_small.osm")
ROADS = Path("runs/osmnx_la_university_park/road_network.json")
PROFILES = ("light_traffic", "heavy_traffic")
EV_CLASS = "delivery"
EV_SPEED_M_S = 25.0
SEED = 131
GAP_SECONDS = 300
COPYRIGHT = "https://www.openstreetmap.org/copyright"
ODBL = "https://opendatacommons.org/licenses/odbl/1-0/"
FILES = ("network.net.xml", "netconvert.log", "netconvert.version.log", "sumo.version.log", "probe_routes.rou.xml",
         "routes.json", "sumo.log", "tripinfo.xml", "comparison.csv", "summary.json")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def _write(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def _allows(element: ET.Element) -> bool:
    allowed = set(element.get("allow", "all").split())
    rejected = set(element.get("disallow", "").split())
    return ("all" in allowed or EV_CLASS in allowed) and not ({"all", EV_CLASS} & rejected)


def _road_edges(network: Path) -> tuple[dict, dict, dict]:
    root = ET.parse(network).getroot()
    edges, incoming, outgoing, lanes, successors = {}, defaultdict(list), defaultdict(list), {}, defaultdict(set)
    for edge in root.findall("edge"):
        by_index = {lane.attrib["index"]: lane for lane in edge.findall("lane") if _allows(lane)}
        if not by_index or "from" not in edge.attrib or edge.get("function") == "internal":
            continue
        name = edge.attrib["id"]
        origin, destination = edge.attrib["from"], edge.attrib["to"]
        lengths = [float(lane.attrib["length"]) for lane in by_index.values()]
        speeds = [float(lane.attrib["speed"]) for lane in by_index.values()]
        edges[name] = {"from": origin, "to": destination, "length_m": min(lengths),
                       "seconds": min(lengths) / min(max(speeds), EV_SPEED_M_S)}
        lanes[name] = by_index
        incoming[destination].append(name)
        outgoing[origin].append(name)
    for connection in root.findall("connection"):
        origin, destination = connection.attrib["from"], connection.attrib["to"]
        if (origin in edges and destination in edges and _allows(connection)
                and connection.get("fromLane") in lanes[origin] and connection.get("toLane") in lanes[destination]):
            successors[origin].add(destination)
    return edges, {k: sorted(v) for k, v in successors.items()}, {k: sorted(v) for k, v in outgoing.items() if incoming[k]}


def _osm_features(path: Path) -> tuple[dict, dict, dict, dict]:
    root = ET.parse(path).getroot()
    locations, coordinates = defaultdict(set), {}
    for node in root.findall("node"):
        coordinate = tuple(round(float(node.attrib[axis]), 7) for axis in ("lon", "lat"))
        locations[coordinate].add(node.attrib["id"])
        coordinates[node.attrib["id"]] = coordinate
    followed, ways = {}, {}
    for way in root.findall("way"):
        nodes = [point.attrib["ref"] for point in way.findall("nd")]
        ways[way.attrib["id"]] = nodes
        followed[way.attrib["id"]] = {frozenset((first, second)) for first, second in pairwise(nodes)}
    return locations, coordinates, followed, ways


def _distance(a: tuple, b: tuple) -> float:
    lon_a, lat_a = map(radians, a)
    lon_b, lat_b = map(radians, b)
    ratio = sin((lat_b - lat_a) / 2) ** 2 + cos(lat_a) * cos(lat_b) * sin((lon_b - lon_a) / 2) ** 2
    return 2 * 6371009 * asin(sqrt(ratio))


def _way_id(edge: str) -> str:
    return edge.lstrip("-").split("#", 1)[0]


def _interior_positions(node: str, edges: dict, ways: dict, coordinates: dict) -> dict:
    options = {}
    for name, fields in edges.items():
        sequence = ways.get(_way_id(name), ())
        if node not in sequence or fields["from"] not in sequence or fields["to"] not in sequence:
            continue
        start, end = sequence.index(fields["from"]), sequence.index(fields["to"])
        index = sequence.index(node)
        if not min(start, end) < index < max(start, end):
            continue
        shape = [coordinates[sequence[i]] for i in range(start, end + (1 if end > start else -1),
                                                        1 if end > start else -1)]
        split = abs(index - start)
        first = sum(_distance(a, b) for a, b in pairwise(shape[:split + 1]))
        total = sum(_distance(a, b) for a, b in pairwise(shape))
        position = round(fields["length_m"] * first / total, 4)
        if 0 < position < fields["length_m"]:
            options[name] = {"position_m": position, "prefix": shape[:split], "suffix": shape[split + 1:]}
    return options


def _same_way(edge: str, coordinate_segment: list, locations: dict, followed: dict) -> bool:
    way = followed.get(_way_id(edge), set())
    return all(any(frozenset((first, second)) in way
                   for first in locations[tuple(a)] for second in locations[tuple(b)])
               for a, b in pairwise(coordinate_segment))


def _pinned_path(geometry: list, locations: dict, followed: dict, edges: dict,
                 links: dict, out: dict, origin: str, destination: str) -> tuple[str, ...] | None:
    passing = []
    for i, point in enumerate(geometry):
        candidates = locations[tuple(point)] & out.keys()
        if len(candidates) > 1:
            return None
        if candidates:
            passing.append((i, next(iter(candidates))))
    if not passing or passing[0] != (0, origin) or passing[-1] != (len(geometry) - 1, destination):
        return None
    options = {None: ()}
    for (start, first), (end, second) in pairwise(passing):
        possible = [edge for edge in out[first] if edges[edge]["to"] == second
                    and _same_way(edge, geometry[start:end + 1], locations, followed)]
        options = {edge: path for edge in possible if (path := min(
            (prefix + (edge,) for prior, prefix in options.items() if prior is None or edge in links.get(prior, ())),
            default=None,
        )) is not None}
        if not options:
            return None
    return min(options.values())


def _shortest_legal(starts: dict, destinations: dict, edges: dict, links: dict) -> tuple[str, ...]:
    queue = []
    costs = {}
    for name in starts:
        priority = (edges[name]["seconds"], (name,))
        costs[name] = priority
        heappush(queue, priority)
    while queue:
        seconds, path = heappop(queue)
        name = path[-1]
        if costs[name] != (seconds, path):
            continue
        if name in destinations:
            return path
        for next_edge in links.get(name, ()):
            candidate = (seconds + edges[next_edge]["seconds"], (*path, next_edge))
            if next_edge not in costs or candidate < costs[next_edge]:
                costs[next_edge] = candidate
                heappush(queue, candidate)
    raise ValueError("No SUMO path with delivery lane and connection permissions between access edges")


def _access(site: dict, edges: dict, out: dict, *, origin: bool) -> tuple[dict, list[tuple]]:
    node = site["osm_node_id"]
    interior = site["interior"]
    if interior:
        side, padding = ("from", "prefix") if origin else ("to", "suffix")
        return ({edge: row["position_m"] for edge, row in interior.items()},
                [(edge, edges[edge][side], row[padding]) for edge, row in interior.items()])
    options = out[node] if origin else [edge for edge, item in edges.items() if item["to"] == node]
    return dict.fromkeys(options, 0 if origin else "max"), [(None, node, [])]


def _aligned_path(origin: dict, destination: dict, shape: list, locations: dict, followed: dict,
                  edges: dict, links: dict, outgoing: dict) -> tuple[tuple, bool, dict, dict]:
    starts, first = _access(origin, edges, outgoing, origin=True)
    ends, last = _access(destination, edges, outgoing, origin=False)
    paths = []
    for departure, start_node, prefix in first:
        for arrival, end_node, suffix in last:
            route = _pinned_path(prefix + shape + suffix, locations, followed, edges, links, outgoing,
                                 start_node, end_node)
            if (route and route[0] in starts and route[-1] in ends
                    and (departure is None or route[0] == departure)
                    and (arrival is None or route[-1] == arrival)):
                paths.append(route)
    path = min(paths) if paths else _shortest_legal(starts, ends, edges, links)
    return path, bool(paths), starts, ends


def _routes(source: Path, road_path: Path, network: Path) -> dict:
    road = _read(road_path)
    if road["provenance"]["source"]["sha256"] != _sha(source):
        raise ValueError("The raw OSM source differs from the OSMnx route artifact")
    edges, links, outgoing = _road_edges(network)
    locations, coordinates, followed, ways = _osm_features(source)
    sites = {}
    for name, point in road["sites"]["coordinates"].items():
        raw = locations[tuple(point)]
        junctions = raw & outgoing.keys()
        if len(junctions) == 1:
            sites[name] = {"osm_node_id": next(iter(junctions)), "placement": "junction", "interior": {}}
            continue
        if len(raw) != 1:
            raise ValueError(f"Site {name} does not map unambiguously to an original OSM point")
        node = next(iter(raw))
        interior = _interior_positions(node, edges, ways, coordinates)
        if not interior:
            raise ValueError(f"Site {name} lacks an imported SUMO edge interior")
        sites[name] = {"osm_node_id": node, "placement": "edge_interior", "interior": interior}
    rows = {}
    for profile in PROFILES:
        data = road["profiles"][profile]
        for origin, destinations in data["routes"].items():
            for destination, shape in destinations.items():
                if origin == destination:
                    continue
                path, exact, starts, ends = _aligned_path(sites[origin], sites[destination], shape, locations,
                                                          followed, edges, links, outgoing)
                depart, arrival = starts[path[0]], ends[path[-1]]
                arrival_trim = 0 if arrival == "max" else edges[path[-1]]["length_m"] - arrival
                distance = sum(edges[name]["length_m"] for name in path) - depart - arrival_trim
                identifier = f"probe_{len(rows) + 1:03}"
                rows[identifier] = {
                    "profile": profile, "origin": origin, "destination": destination,
                    "sumo_edges": list(path), "sumo_same_osm_way_geometry": exact,
                    "departure_lane_position_m": depart, "arrival_lane_position_m": arrival,
                    "sumo_regular_edge_distance_m": round(distance, 6),
                    "sumo_static_seconds": round(sum(edges[name]["seconds"] for name in path), 6),
                    "osmnx_distance_m": data["distance_m"][origin][destination],
                    "osmnx_modeled_seconds": data["travel_seconds"][origin][destination],
                    "osmnx_modeled_wh": data["travel_wh"][origin][destination],
                }
    public_sites = {name: {"osm_node_id": data["osm_node_id"], "placement": data["placement"],
                           "edge_positions_m": {edge: row["position_m"] for edge, row in data["interior"].items()}}
                    for name, data in sites.items()}
    return {"sites": public_sites, "delivery_edges": len(edges), "rows": rows}


def _traffic(path: Path, mapping: dict) -> None:
    root = ET.Element("routes")
    probe = ET.SubElement(root, "vType", {"id": "probe_ev", "vClass": EV_CLASS, "emissionClass": "Energy/unknown",
                                         "mass": "1900", "length": "5", "minGap": "2.5", "accel": "1.4",
                                         "decel": "3.0", "sigma": "0", "maxSpeed": str(EV_SPEED_M_S), "speedFactor": "1"})
    parameters = {"has.battery.device": "true", "device.battery.capacity": "35000",
                  "constantPowerIntake": "350", "frontSurfaceArea": "3", "airDragCoefficient": "0.36",
                  "rollDragCoefficient": "0.01", "propulsionEfficiency": "0.90", "recuperationEfficiency": "0.72"}
    for key, value in parameters.items():
        ET.SubElement(probe, "param", {"key": key, "value": value})
    for index, (identifier, row) in enumerate(mapping["rows"].items()):
        vehicle = ET.SubElement(root, "vehicle", {"id": identifier, "type": "probe_ev",
                                                   "depart": str(index * GAP_SECONDS), "departSpeed": "0",
                                                   "departPos": str(row["departure_lane_position_m"]),
                                                   "arrivalPos": str(row["arrival_lane_position_m"])})
        ET.SubElement(vehicle, "route", {"edges": " ".join(row["sumo_edges"])})
        ET.SubElement(vehicle, "param", {"key": "device.battery.chargeLevel", "value": "26000"})
    ET.indent(root, space="  ")
    with path.open("xb") as handle:
        ET.ElementTree(root).write(handle, encoding="utf-8", xml_declaration=True)


def _results(directory: Path, mapping: dict) -> tuple[dict, list[dict]]:
    shared = mapping["rows"]
    measured = read_trips(directory / "tripinfo.xml", {
        name: (record["origin"], record["destination"]) for name, record in shared.items()
    })
    rows = []
    for trip in measured:
        info = shared[trip["probe_id"]]
        row = {"profile": info["profile"], "origin": info["origin"], "destination": info["destination"],
               "same_osm_way_geometry": info["sumo_same_osm_way_geometry"],
               "sumo_duration_s": trip["duration_s"], "sumo_departure_delay_s": trip["depart_delay_s"],
               "sumo_distance_m": trip["distance_m"], "sumo_consumed_wh": trip["consumed_wh"],
               "sumo_regenerated_wh": trip["regenerated_wh"],
               "sumo_static_seconds": info["sumo_static_seconds"],
               "osmnx_distance_m": info["osmnx_distance_m"],
               "osmnx_modeled_seconds": info["osmnx_modeled_seconds"], "osmnx_modeled_wh": info["osmnx_modeled_wh"]}
        rows.append(row)
    summary = {"completed_electric_probes": len(rows), "osm_sites": mapping["sites"],
               "imported_delivery_edges": mapping["delivery_edges"],
               "route_geometry_count": dict(Counter("same_osm_way" if r["same_osm_way_geometry"] else "sumo_legal_alternative"
                                                    for r in rows)), "profiles": {}}
    duplicate_geometry = sum(shared[name]["sumo_edges"] == shared[f"probe_{i+43:03}"]["sumo_edges"]
                             for i, name in enumerate(shared) if i < 42)
    summary["identical_sumo_route_pairs_across_profiles"] = duplicate_geometry
    for profile in PROFILES:
        records = [row for row in rows if row["profile"] == profile]
        if len(records) != 42:
            raise ValueError(f"Expected 42 completed directed probes for {profile}")
        summary["profiles"][profile] = {
            "complete": len(records), "same_osm_way": sum(row["same_osm_way_geometry"] for row in records),
            "mean_sumo_seconds": fmean(row["sumo_duration_s"] for row in records),
            "mean_osmnx_seconds": fmean(row["osmnx_modeled_seconds"] for row in records),
            "median_abs_distance_difference_m": median(abs(row["sumo_distance_m"] - row["osmnx_distance_m"])
                                                        for row in records),
            "mean_sumo_consumed_wh": fmean(row["sumo_consumed_wh"] for row in records),
            "mean_osmnx_modeled_wh": fmean(row["osmnx_modeled_wh"] for row in records),
        }
    return summary, rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(source: Path, road_path: Path, directory: Path) -> dict:
    source, road_path = source.resolve(strict=True), road_path.resolve(strict=True)
    directory.mkdir(parents=True, exist_ok=False)
    executable, engine, home = _binaries()
    environment = os.environ.copy()
    if home is not None:
        environment["SUMO_HOME"] = str(home)
    launch = lambda command, name: _invoke(directory, command, name, environment)
    netconvert_version = _version(launch([str(executable), "--version"], "netconvert.version.log"))
    sumo_version = _version(launch([str(engine), "--version"], "sumo.version.log"))
    launch([str(executable), "--osm-files", str(source), "--geometry.remove", "false", "--junctions.join", "false",
            "--keep-edges.by-vclass", EV_CLASS, "--output.original-names", "true", "-o", "network.net.xml"], "netconvert.log")
    mapping = _routes(source, road_path, directory / "network.net.xml")
    _write(directory / "routes.json", mapping)
    _traffic(directory / "probe_routes.rou.xml", mapping)
    launch([str(engine), "-n", "network.net.xml", "-r", "probe_routes.rou.xml", "--tripinfo-output", "tripinfo.xml",
            "--tripinfo-output.write-unfinished", "true", "--tripinfo-output.write-undeparted", "true",
            "--time-to-teleport", "-1", "--max-num-teleports", "0", "--ignore-route-errors", "false",
            "--seed", str(SEED), "--end", str((len(mapping["rows"]) + 3) * GAP_SECONDS),
            "--no-step-log", "true"], "sumo.log")
    summary, rows = _results(directory, mapping)
    _write_rows(directory / "comparison.csv", rows)
    _write(directory / "summary.json", summary)
    _write(directory / "manifest.json", {
        "schema": 1, "seed": SEED, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_osm_sha256": _sha(source), "osmnx_road_artifact_sha256": _sha(road_path),
        "sumo_version": sumo_version, "netconvert_version": netconvert_version,
        "attribution": "© OpenStreetMap contributors", "copyright": COPYRIGHT,
        "license": "ODbL 1.0", "license_url": ODBL,
        "scope": "84 electric-van probes on the native imported University Park street network. Both route sets use "
                 "the same signalized simulator and scheduled sparse departures, with no background demand.",
        "route_matching": "Common OSM access points with a centerline-position fraction mapped onto the SUMO "
                          "depot lane. Preserve each OSMnx way geometry where the SUMO "
                          "delivery network retains a valid sequence. Remaining routes minimize native SUMO free-flow "
                          "whole-edge cost among permitted connections between the same access points.",
        "limitations": "Light and heavy label the upstream OSMnx route choices. SUMO probes share the same sparse "
                       "traffic conditions. SUMO lane endpoints, junction offsets and movement differ from OSMnx "
                       "road centerlines. Some OSM restrictions are incomplete in this bounded import. Energy reflects "
                       "a simulated electric van. Six sites are matching junctions and the OSM depot shape vertex "
                       "is represented by its proportional lane position.",
        "sha256": {name: _sha(directory / name) for name in FILES},
    })
    return summary


def verify(source: Path, road_path: Path, directory: Path) -> dict:
    receipt = _read(directory / "manifest.json")
    if receipt["source_osm_sha256"] != _sha(source) or receipt["osmnx_road_artifact_sha256"] != _sha(road_path):
        raise ValueError("SUMO inputs differ from the source fingerprints")
    for name in FILES:
        if receipt["sha256"][name] != _sha(directory / name):
            raise ValueError(f"SUMO artifact differs from its manifest: {name}")
    mapping = _routes(source, road_path, directory / "network.net.xml")
    if mapping != _read(directory / "routes.json"):
        raise ValueError("Recorded route geometry is inconsistent with the imported delivery network")
    summary, rows = _results(directory, mapping)
    if summary != _read(directory / "summary.json"):
        raise ValueError("SUMO trip metrics differ from the saved summary")
    with (directory / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        for saved, expected in zip(csv.DictReader(stream), rows, strict=True):
            if saved != {key: str(value) for key, value in expected.items()}:
                raise ValueError("Comparison rows differ from the SUMO trip output")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--roads", type=Path, default=ROADS)
    args = parser.parse_args()
    operation = run if args.command == "run" else verify
    print(json.dumps(operation(args.source, args.roads, args.directory), sort_keys=True))


if __name__ == "__main__":
    main()
