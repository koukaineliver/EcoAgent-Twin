"""Build a forward-verified Los Angeles street and route artifact from OpenStreetMap."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from itertools import product
import json
from math import ceil, hypot, isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

import osmnx as ox
from pyproj import Transformer


BOUNDS = (-118.294, 34.015, -118.278, 34.035)
OSM_MAP = "https://api.openstreetmap.org/api/0.6/map"
COPYRIGHT = "https://www.openstreetmap.org/copyright"
LICENSE = "https://opendatacommons.org/licenses/odbl/1-0/"
ROOT = Path("runs/osmnx_la_university_park")
FREE_FLOW_KPH = {
    "motorway": 85, "motorway_link": 45, "trunk": 65, "trunk_link": 45,
    "primary": 50, "primary_link": 35, "secondary": 45, "secondary_link": 35,
    "tertiary": 40, "tertiary_link": 30, "unclassified": 32, "residential": 32,
    "living_street": 16,
}
ARTERIAL = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
            "secondary", "secondary_link"}
ENERGY_WH_KM, AUX_W = 220, 300
BASE_SITES = {
    "depot": (-118.28420, 34.02165, "University Park depot"),
    "central": (-118.2852155, 34.0256334, "USC parcel locker"),
    "east": (-118.28055, 34.02560, "East candidate locker"),
    "west": (-118.28900, 34.02580, "West candidate locker"),
    "north": (-118.28540, 34.03080, "North candidate locker"),
    "start-1": (-118.28503, 34.02185, "EV 1 starting point"),
    "start-2": (-118.28305, 34.02250, "EV 2 starting point"),
}
SCALE_SITES = {
    "rep02-east": (-118.27940, 34.02270), "rep02-north": (-118.28190, 34.02920),
    "rep02-west": (-118.28260, 34.02080), "rep02-start-1": (-118.28470, 34.02050),
    "rep02-start-2": (-118.28250, 34.02390), "rep03-east": (-118.28740, 34.02850),
    "rep03-north": (-118.28860, 34.03220), "rep03-west": (-118.29120, 34.02670),
    "rep03-start-1": (-118.29010, 34.02270), "rep03-start-2": (-118.29100, 34.03010),
    "rep04-east": (-118.27920, 34.03050), "rep04-north": (-118.28230, 34.03315),
    "rep04-west": (-118.28730, 34.03280), "rep04-start-1": (-118.28340, 34.03210),
    "rep04-start-2": (-118.28600, 34.03330),
}


def _json(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False, separators=(",", ":"))
            + "\n").encode("utf-8")


def acquire(source: Path, receipt: Path) -> None:
    if source.exists() or receipt.exists():
        raise FileExistsError("Use a new path to acquire a fresh OpenStreetMap snapshot")
    request_url = OSM_MAP + "?bbox=" + ",".join(map(str, BOUNDS))
    request = Request(request_url, headers={"User-Agent": "EcoAgent-Twin University Park research demonstrator/1.0"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
        root = ET.fromstring(payload)
        if root.tag != "osm" or not root.findall("way"):
            raise ValueError("The OpenStreetMap endpoint returned no street-source ways")
        metadata = {"url": request_url, "sha256": sha256(payload).hexdigest(), "bytes": len(payload),
                    "bbox": BOUNDS, "date": response.headers.get("Date"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "acquired_utc": datetime.now(timezone.utc).isoformat()}
    source.parent.mkdir(parents=True, exist_ok=True)
    with source.open("xb") as stream:
        stream.write(payload)
    with receipt.open("xb") as stream:
        stream.write(_json(metadata))


def _source(source: Path, receipt: Path) -> tuple[ET.Element, dict]:
    metadata = json.loads(receipt.read_bytes())
    raw = source.read_bytes()
    if (metadata["sha256"] != sha256(raw).hexdigest() or metadata["bytes"] != len(raw)
            or tuple(metadata["bbox"]) != BOUNDS):
        raise ValueError("OpenStreetMap XML differs from its saved geographic acquisition receipt")
    original = ET.fromstring(raw)
    if original.tag != "osm" or not original.findall("way"):
        raise ValueError("Expected raw OpenStreetMap XML containing ways")
    return original, metadata


def _tags(element: ET.Element) -> dict:
    return {tag.attrib["k"]: tag.attrib["v"] for tag in element.findall("tag")}


def _driving(tags: dict) -> bool:
    restricted = {"no", "private", "customers", "delivery", "destination", "permit", "agricultural", "forestry"}
    return (tags.get("highway") in FREE_FLOW_KPH and tags.get("area") != "yes"
            and all(tags.get(key) not in restricted for key in ("access", "vehicle", "motor_vehicle", "motorcar")))


def _graph(original: ET.Element):
    ways = [way for way in original.findall("way") if _driving(_tags(way))]
    wanted = {tag.attrib["ref"] for way in ways for tag in way.findall("nd")}
    filtered = ET.Element("osm", {"version": "0.6", "generator": "EcoAgent-Twin public motor road extraction"})
    present = set()
    for node in original.findall("node"):
        if node.attrib["id"] in wanted:
            ET.SubElement(filtered, "node", {key: node.attrib[key] for key in ("id", "lat", "lon")})
            present.add(node.attrib["id"])
    if present != wanted or not ways:
        raise ValueError("The OpenStreetMap extract omits nodes needed by the motor road ways")
    for way in ways:
        clean = ET.SubElement(filtered, "way", {"id": way.attrib["id"]})
        clean.extend(way.findall("nd"))
        clean.extend(tag for tag in way.findall("tag") if tag.attrib["k"] in ox.settings.useful_tags_way)
    with TemporaryDirectory() as temporary:
        road_xml = Path(temporary) / "motor_roads.osm"
        ET.ElementTree(filtered).write(road_xml, encoding="utf-8", xml_declaration=True)
        graph = ox.graph_from_xml(road_xml, simplify=False, retain_all=True, bidirectional=False)
    graph = ox.truncate.truncate_graph_bbox(graph, BOUNDS)
    graph = ox.truncate.largest_component(graph, strongly=True)
    graph = ox.routing.add_edge_speeds(graph, hwy_speeds=FREE_FLOW_KPH, fallback=32)
    graph = ox.routing.add_edge_travel_times(graph)
    for _, _, fields in graph.edges(data=True):
        fields["heavy_seconds"] = fields["travel_time"] * (1.50 if fields["highway"] in ARTERIAL else 1.25)
    return graph


def _point(graph, node) -> list[float]:
    return [round(graph.nodes[node][key], 7) for key in ("x", "y")]


def _sites(graph, original, expanded: bool) -> tuple[dict, dict[str, int]]:
    locker = original.find("node[@id='5237388100']")
    if locker is None or _tags(locker).get("amenity") != "parcel_locker":
        raise ValueError("The source area must contain the project’s mapped University Park parcel locker")
    real = float(locker.attrib["lon"]), float(locker.attrib["lat"])
    if max(abs(a - b) for a, b in zip(real, BASE_SITES["central"][:2], strict=True)) > 0.0001:
        raise ValueError("The mapped University Park parcel locker moved from the documented anchor")
    requested = BASE_SITES | {"central": (*real, BASE_SITES["central"][2])}
    if expanded:
        requested.update({key: (*position, key) for key, position in SCALE_SITES.items()})
    projected = ox.project_graph(graph)
    transform = Transformer.from_crs("EPSG:4326", projected.graph["crs"], always_xy=True)
    nodes, coordinates, anchors, labels, snap = {}, {}, {}, {}, {}
    used = set()
    for key, (lon, lat, label) in requested.items():
        x, y = transform.transform(lon, lat)
        node = int(ox.distance.nearest_nodes(projected, x, y))

        def offset(identifier):
            data = projected.nodes[identifier]
            return hypot(data["x"] - x, data["y"] - y)

        if node in used:
            node = min((int(candidate) for candidate in projected if candidate not in used),
                       key=lambda identifier: (offset(identifier), identifier))
        distance = offset(node)
        if distance > 350:
            raise ValueError(f"Road access for {key} lies more than 350 m from its stated anchor")
        used.add(node)
        nodes[key], coordinates[key] = node, _point(graph, node)
        labels[key], anchors[key], snap[key] = label, [lon, lat], round(distance, 1)
    roads = sorted({tuple(sorted((tuple(_point(graph, origin)), tuple(_point(graph, destination)))))
                    for origin, destination in graph.edges() if origin != destination})
    result = {"coordinates": coordinates, "roads": roads, "bounds": BOUNDS, "labels": labels,
              "anchors": anchors, "snap_distance_m": snap,
              "mapped_osm_node_ids": {"central": 5237388100}}
    return result, nodes


def _profile(graph, sites: dict[str, int], weight: str) -> dict:
    fields = ("travel_seconds", "travel_minutes", "distance_m", "travel_wh", "routes")
    rows = {field: {name: {} for name in sites} for field in fields}
    pairs = list(product(sites, repeat=2))
    routes = ox.routing.shortest_path(graph, [sites[a] for a, _ in pairs], [sites[b] for _, b in pairs],
                                    weight=weight, cpus=1)
    for (origin, destination), route in zip(pairs, routes, strict=True):
        if not route or route[0] != sites[origin] or route[-1] != sites[destination]:
            raise ValueError(f"No directed motor route from {origin} to {destination}")
        geometry = [_point(graph, node) for node in route]
        if origin == destination:
            distance, seconds = 0.0, 0.0
        else:
            edges = ox.routing.route_to_gdf(graph, route, weight=weight)
            distance, seconds = float(edges["length"].sum()), float(edges[weight].sum())
            if not all(isfinite(value) and value > 0 for value in (distance, seconds)):
                raise ValueError(f"Invalid road distance or time for {origin} to {destination}")
        values = (round(seconds, 6), ceil(seconds / 60), round(distance, 3),
                  ceil(distance * ENERGY_WH_KM / 1000 + seconds * AUX_W / 3600), geometry)
        for field, value in zip(fields, values, strict=True):
            rows[field][origin][destination] = value
    if any(rows["travel_wh"][name]["depot"] > 300 for name in ("start-1", "start-2")):
        raise ValueError("A baseline EV cannot return to the depot at its minimum 40% starting charge")
    return rows


def build(source: Path, receipt: Path, *, expanded: bool = False) -> dict:
    original, origin = _source(source, receipt)
    graph = _graph(original)
    sites, nodes = _sites(graph, original, expanded)
    profiles = {"light_traffic": _profile(graph, nodes, "travel_time"),
                "heavy_traffic": _profile(graph, nodes, "heavy_seconds")}
    provenance = {
        "study_area": "University Park, Los Angeles, California",
        "spatial_reference": "EPSG:4326, longitude then latitude",
        "source": origin, "attribution": "© OpenStreetMap contributors", "copyright_url": COPYRIGHT,
        "license": "ODbL 1.0", "license_url": LICENSE, "osmnx_version": version("osmnx"),
        "graph": {"nodes": len(graph), "directed_edges": graph.number_of_edges(),
                  "scope": "largest strongly connected public motor-road component inside the source bounds",
                  "routing": "OSMnx shortest path by profile travel time on directed OSM ways"},
        "speed": {"basis": "OSM maxspeed where usable, with OSMnx edge speed assignment and class defaults",
                  "class_defaults_kph": FREE_FLOW_KPH, "heavy_arterial_time_multiplier": 1.50,
                  "heavy_other_time_multiplier": 1.25, "heavy_arterial_highways": sorted(ARTERIAL)},
        "ev_energy": {"distance_wh_per_km": ENERGY_WH_KM, "moving_auxiliary_w": AUX_W,
                      "basis": "modeled distance energy plus travel-time auxiliary energy, rounded up to Wh"},
        "site_basis": "Central uses the street access nearest the mapped USC parcel locker. Other sites are scenario curb positions.",
        "measurement_basis": "Current road geometry with modeled light and heavy travel times and electric energy."
    }
    return {"schema": 1, "sites": sites, "profiles": profiles, "provenance": provenance}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("acquire", "build", "verify"))
    parser.add_argument("--source", type=Path, default=ROOT / "osm_small.osm")
    parser.add_argument("--receipt", type=Path, default=ROOT / "osm_small.source.json")
    parser.add_argument("--output", type=Path, default=ROOT / "road_network.json")
    parser.add_argument("--expanded", action="store_true", help="add 15 distinct curb locations for the scale experiment")
    args = parser.parse_args()
    if args.command == "acquire":
        acquire(args.source, args.receipt)
        return
    payload = build(args.source, args.receipt, expanded=args.expanded)
    data = _json(payload)
    if args.command == "verify":
        if args.output.read_bytes() != data:
            raise ValueError("Saved street map or OD routes differ from the OSMnx reconstruction")
    else:
        with args.output.open("xb") as stream:
            stream.write(data)
    print(json.dumps({"output": str(args.output), "sha256": sha256(data).hexdigest(),
                      "sites": len(payload["sites"]["coordinates"]), **payload["provenance"]["graph"]}))


if __name__ == "__main__":
    main()
