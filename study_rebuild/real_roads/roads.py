"""Build directional routing profiles from the archived Southern California road extract."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
from pathlib import Path
import re

import networkx as nx
import numpy as np
import osmium
import osmnx as ox
import pandas as pd
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree

from ..future_transport import data as ft_data


SOURCE = "https://download.geofabrik.de/north-america/us/california/socal/socal-190101.osm.pbf"
BOUNDS = (-119.15, 33.12, -116.60, 34.77)
SITE_SNAP_LIMIT_M = 1_500
DETOUR_SNAP_LIMIT_M = 4_000
ARC_SAMPLES = 21
EARTH_RADIUS_M = 6_371_009
TRAFFIC_KPH = {
    "motorway": 100, "motorway_link": 45, "trunk": 80, "trunk_link": 45, "primary": 65,
    "primary_link": 40, "secondary": 55, "secondary_link": 35, "tertiary": 45, "tertiary_link": 30,
    "unclassified": 35, "residential": 35, "living_street": 15, "service": 20, "road": 25,
}
DENIED = {"no", "private", "agricultural", "forestry", "emergency", "permit"}
PROJECT = Transformer.from_crs("EPSG:4326", "EPSG:3310", always_xy=True)


@dataclass(frozen=True)
class Site:
    key: str
    longitude: float
    latitude: float
    snap_limit_m: float = SITE_SNAP_LIMIT_M


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, result: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_profile(path: Path, result: dict) -> None:
    packed = json.dumps(result, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
    with path.open("xb") as handle:
        with gzip.GzipFile(fileobj=handle, mode="wb", filename="", mtime=0) as compressed:
            compressed.write(packed)


def read_profile(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return json.load(handle)


def _vehicle_speed(tags, highway: str) -> float:
    source = tags.get("maxspeed", "")
    posted = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(mph|km/h|kph)?\s*", source)
    if posted:
        speed = float(posted[1]) * (1.609344 if posted[2] == "mph" else 1)
        if 5 <= speed <= 140:
            return speed
    return float(TRAFFIC_KPH[highway])


def _directions(tags, highway: str) -> tuple[bool, bool]:
    directed = next((tags[field] for field in ("oneway:motorcar", "oneway:motor_vehicle", "oneway") if field in tags), "")
    if directed in {"-1", "reverse"}:
        return False, True
    if directed in {"yes", "1", "true"}:
        return True, False
    if directed in {"no", "0", "false"}:
        return True, True
    forward_only = highway in {"motorway", "motorway_link"} or tags.get("junction") in {"roundabout", "circular"}
    return True, not forward_only


def build_graph(source: Path, bounds=BOUNDS) -> tuple[nx.MultiDiGraph, dict]:
    """Load local Public OSM ways, preserving motor access and one-way tags."""
    source = source.resolve(strict=True)
    with osmium.io.Reader(str(source)) as reader:
        header = reader.header()
        snapshot = header.get("osmosis_replication_timestamp") or None
        generator = header.get("generator") or None
    west, south, east, north = bounds
    g = nx.MultiDiGraph(crs="epsg:4326", simplified=False)
    kept_ways = set()
    source_graph = osmium.FileProcessor(str(source)).with_locations("flex_mem")
    source_graph.with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
    for way in source_graph:
        tags = way.tags
        highway = tags.get("highway")
        access = next((tags[field] for field in ("motorcar", "motor_vehicle", "vehicle", "access") if field in tags), "")
        if highway not in TRAFFIC_KPH or access in DENIED or tags.get("area") == "yes":
            continue
        if tags.get("service") in {"parking_aisle", "driveway", "emergency_access", "drive-through"}:
            continue
        forward, backward = _directions(tags, highway)
        speed = _vehicle_speed(tags, highway)
        previous = None
        for node in way.nodes:
            current = None
            if node.location.valid():
                x, y = node.location.lon, node.location.lat
                if west <= x <= east and south <= y <= north:
                    current = node.ref, x, y
            if previous is not None and current is not None and previous[0] != current[0] and previous[1:] != current[1:]:
                u, ux, uy = previous
                v, vx, vy = current
                if u not in g:
                    g.add_node(u, x=ux, y=uy)
                if v not in g:
                    g.add_node(v, x=vx, y=vy)
                if forward:
                    g.add_edge(u, v, osmid=way.id, speed_kph=speed)
                if backward:
                    g.add_edge(v, u, osmid=way.id, speed_kph=speed)
                kept_ways.add(way.id)
            previous = current
    if not len(g):
        raise ValueError("the source extract supplies no drivable edges within the declared clipping bounds")
    total_nodes, total_edges = len(g), g.number_of_edges()
    component = max(nx.strongly_connected_components(g), key=len)
    g.remove_nodes_from(set(g) - component)
    ox.distance.add_edge_lengths(g)
    return g, {
        "source_url": SOURCE, "source_file": source.name, "source_sha256": digest(source),
        "historical_extract_nominal_date": "2019-01-01", "header_replication_timestamp": snapshot,
        "header_generator": generator, "bounding_box_wsen": list(bounds), "input_ways": len(kept_ways),
        "all_graph_nodes": total_nodes, "all_graph_edges": total_edges,
        "connected_nodes": len(g), "connected_edges": g.number_of_edges(),
        "topology": "Directed drivable OSM ways, strongest connected component; no turn-restriction relations",
        "road_time": "Free flow from posted numeric speed or declared road-class speeds",
        "road_class_default_kph": TRAFFIC_KPH, "osmnx": version("osmnx"), "osmium": version("osmium"),
        "attribution": "© OpenStreetMap contributors via Geofabrik. Open Database License (ODbL).",
        "license_url": "https://www.openstreetmap.org/copyright",
    }


@dataclass(frozen=True)
class _GraphArrays:
    nodes: np.ndarray
    coords: np.ndarray
    distance_network: csr_matrix
    travel_network: csr_matrix
    duration_on_distance: np.ndarray
    distance_on_duration: np.ndarray
    way_on_distance: np.ndarray
    way_on_duration: np.ndarray


def _arrays(graph: nx.MultiDiGraph) -> _GraphArrays:
    node_order = np.asarray(sorted(graph), dtype=np.int64)
    point = np.asarray([(graph.nodes[node]["x"], graph.nodes[node]["y"]) for node in node_order])
    node_index = {node: index for index, node in enumerate(node_order)}
    columns = np.empty(graph.number_of_edges(), dtype=np.int32)
    starts = np.empty(len(node_order) + 1, dtype=np.int32)
    values = np.empty((graph.number_of_edges(), 4), dtype=np.float64)
    osmids = np.empty((graph.number_of_edges(), 2), dtype=np.int64)
    cursor = 0
    for index, node in enumerate(node_order):
        starts[index] = cursor
        for neighbor, choices in sorted(graph[node].items()):
            if node == neighbor:
                continue
            possibilities = []
            for edge in choices.values():
                length = float(edge["length"])
                if length <= 0 or not np.isfinite(length):
                    raise ValueError("the drivable graph contains an invalid edge length")
                duration = length * 3.6 / edge["speed_kph"]
                possibilities.append((length, duration, edge["osmid"]))
            shortest = min(possibilities)
            fastest = min(possibilities, key=lambda row: (row[1], row[0], row[2]))
            columns[cursor] = node_index[neighbor]
            values[cursor] = [shortest[0], fastest[1], shortest[1], fastest[0]]
            osmids[cursor] = [shortest[2], fastest[2]]
            cursor += 1
    starts[-1] = cursor
    cols = columns[:cursor].copy()
    data = values[:cursor].T.copy()
    ways = osmids[:cursor].T.copy()
    shape = (len(node_order), len(node_order))
    return _GraphArrays(
        node_order, point, csr_matrix((data[0], cols, starts), shape=shape),
        csr_matrix((data[1], cols, starts), shape=shape), data[2], data[3], ways[0], ways[1],
    )


def save_network(path: Path, graph: _GraphArrays) -> None:
    """Save the sparse route graph without Python-object serialization."""
    with path.open("xb") as handle:
        np.savez_compressed(
            handle, nodes=graph.nodes, coordinates=graph.coords,
            indices=graph.distance_network.indices, indptr=graph.distance_network.indptr,
            distance_cost=graph.distance_network.data, time_cost=graph.travel_network.data,
            duration_on_distance=graph.duration_on_distance, distance_on_duration=graph.distance_on_duration,
            way_on_distance=graph.way_on_distance, way_on_duration=graph.way_on_duration,
        )


def load_network(path: Path) -> _GraphArrays:
    """Load the saved directed OSMnx-derived graph for independent routed studies."""
    with np.load(path, allow_pickle=False) as archive:
        nodes = archive["nodes"]
        layout = archive["indices"], archive["indptr"]
        count = len(nodes)
        distance = csr_matrix((archive["distance_cost"], *layout), shape=(count, count))
        duration = csr_matrix((archive["time_cost"], *layout), shape=(count, count))
        return _GraphArrays(
            nodes, archive["coordinates"], distance, duration,
            archive["duration_on_distance"], archive["distance_on_duration"],
            archive["way_on_distance"], archive["way_on_duration"],
        )


def _snap(graph: nx.MultiDiGraph, sites: list[Site]) -> list[dict]:
    if len({site.key for site in sites}) != len(sites) or not sites:
        raise ValueError("routing profile requires unique site IDs")
    vertices, offsets = ox.distance.nearest_nodes(
        graph, X=[site.longitude for site in sites], Y=[site.latitude for site in sites], return_dist=True,
    )
    return [
        {"site_id": site.key, "osm_node": int(vertex), "offset_m": float(offset), "limit_m": site.snap_limit_m,
         "covered": bool(offset <= site.snap_limit_m), "longitude": site.longitude, "latitude": site.latitude,
         "road_longitude": float(graph.nodes[vertex]["x"]), "road_latitude": float(graph.nodes[vertex]["y"])}
        for site, vertex, offset in zip(sites, vertices, offsets, strict=True)
    ]


def route_sites(graph: _GraphArrays, sites: list[Site], *, mode: str = "travel_time", cells: pd.DataFrame | None = None) -> dict:
    """Snap new sites locally and compute the complete directed path-cost profile."""
    if len({site.key for site in sites}) != len(sites) or not sites:
        raise ValueError("routing profile requires unique site IDs")
    query = np.deg2rad([[site.latitude, site.longitude] for site in sites])
    if not np.isfinite(query).all():
        raise ValueError("road graph snapping requires finite WGS84 coordinates")
    offsets, indexes = BallTree(np.deg2rad(graph.coords[:, ::-1]), metric="haversine").query(query, k=1)
    snaps = []
    for site, index, offset in zip(sites, indexes[:, 0], offsets[:, 0] * EARTH_RADIUS_M, strict=True):
        road_longitude, road_latitude = graph.coords[index]
        snaps.append({"site_id": site.key, "osm_node": int(graph.nodes[index]), "offset_m": float(offset),
                      "limit_m": site.snap_limit_m, "covered": bool(offset <= site.snap_limit_m),
                      "longitude": site.longitude, "latitude": site.latitude,
                      "road_longitude": float(road_longitude), "road_latitude": float(road_latitude)})
    uncovered = [row["site_id"] for row in snaps if not row["covered"]]
    if uncovered:
        raise ValueError("graph coverage insufficient for requested sites: " + ", ".join(uncovered))
    return _route_profile(graph, snaps, mode, cells=cells)


def _route_indices(network, predecessors, origin: int, destination: int) -> tuple[np.ndarray, np.ndarray]:
    if origin == destination:
        return np.asarray([origin]), np.empty(0, dtype=np.int64)
    nodes, arcs = [destination], []
    while destination != origin:
        start = int(predecessors[destination])
        if start < 0 or len(nodes) > network.shape[0]:
            raise ValueError("one directed origin–destination route is disconnected")
        lo, hi = network.indptr[start:start + 2]
        offset = lo + int(np.searchsorted(network.indices[lo:hi], destination))
        if offset >= hi or network.indices[offset] != destination:
            raise ValueError("a reconstructed path step is absent from the OSM graph")
        arcs.append(offset)
        nodes.append(start)
        destination = start
    return np.asarray(nodes[::-1]), np.asarray(arcs[::-1])


def _samples(points: np.ndarray, lengths: np.ndarray, center_tree: cKDTree) -> list[int]:
    if not len(lengths):
        return [int(center_tree.query(points[0])[1])] * ARC_SAMPLES
    distance = np.concatenate([[0.0], np.cumsum(lengths)])
    target = np.linspace(0, distance[-1], ARC_SAMPLES)
    interpolated = np.column_stack([np.interp(target, distance, points[:, dim]) for dim in range(2)])
    return center_tree.query(interpolated)[1].tolist()


def _route_profile(graph: _GraphArrays, snaps: list[dict], mode: str, *, cells: pd.DataFrame | None = None) -> dict:
    retained = [row["osm_node"] for row in snaps]
    sites = [row["site_id"] for row in snaps]
    indices = np.searchsorted(graph.nodes, retained)
    distances, durations, fingerprints, sample_cells = [], [], [], []
    if mode == "distance":
        network, distance, duration, way = graph.distance_network, graph.distance_network.data, graph.duration_on_distance, graph.way_on_distance
    elif mode == "travel_time":
        network, distance, duration, way = graph.travel_network, graph.distance_on_duration, graph.travel_network.data, graph.way_on_duration
    else:
        raise ValueError(f"unrecognized route objective: {mode}")
    projected, tree = None, None
    if cells is not None:
        projected = np.column_stack(PROJECT.transform(graph.coords[:, 0], graph.coords[:, 1]))
        tree = cKDTree(cells[["x_m", "y_m"]].to_numpy(dtype=float))
    computed = {}
    for origin in indices:
        if origin not in computed:
            cost, predecessors = dijkstra(network, directed=True, indices=origin, return_predecessors=True)
            dist_row, time_row, hash_row, samples_row = [], [], [], []
            for destination in indices:
                path, edges = _route_indices(network, predecessors, int(origin), int(destination))
                if not np.isclose(float(network.data[edges].sum()), cost[destination], atol=1e-7, rtol=1e-11):
                    raise ValueError("reconstructed route cost disagrees with directed Dijkstra result")
                dist_row.append(float(distance[edges].sum()) / 1000)
                time_row.append(float(duration[edges].sum()) / 60)
                payload = graph.nodes[path].astype("<i8").tobytes() + way[edges].astype("<i8").tobytes()
                hash_row.append(sha256(payload).hexdigest())
                if projected is not None:
                    samples_row.append(_samples(projected[path], distance[edges], tree))
            computed[origin] = dist_row, time_row, hash_row, samples_row
        dist_row, time_row, hash_row, samples_row = computed[origin]
        distances.append(dist_row)
        durations.append(time_row)
        fingerprints.append(hash_row)
        if projected is not None:
            sample_cells.append(samples_row)
    result = {
        "schema": 1, "objective": mode, "sites": sites, "snaps": snaps, "distance_km": distances,
        "travel_min": durations, "ordered_osm_node_and_way_sha256": fingerprints,
    }
    if cells is not None:
        result["risk_cell_ids"] = cells["cell_id"].tolist()
        result["sample_cell_indices"] = sample_cells
    return result


def _required_sites(computation: dict, tables: dict) -> tuple[list[Site], list[Site], pd.DataFrame]:
    computational = [Site(str(index), lon, lat) for index, (lon, lat) in enumerate(computation["coordinates"])]
    future = []
    for basis, key, prefix, limit in (
        ("candidates", "candidate_id", "candidate", SITE_SNAP_LIMIT_M),
        ("depots", "station_code", "depot", SITE_SNAP_LIMIT_M),
        ("cells", "cell_id", "cell", DETOUR_SNAP_LIMIT_M),
    ):
        frame = tables[basis].sort_values(key)
        future.extend(Site(f"{prefix}:{getattr(row, key)}", row.longitude, row.latitude, limit) for row in frame.itertuples())
    return computational, future, tables["cells"].sort_values("cell_id")


def build(source: Path, computation: Path, prepared: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    comp = json.loads(computation.read_bytes())
    _, tables = ft_data.prepared(prepared)
    comp_sites, ft_sites, cells = _required_sites(comp, tables)
    graph, metrics = build_graph(source)
    print(f"Constructed directed OSMnx graph with {metrics['connected_nodes']:,} connected nodes.", flush=True)
    sites = {"computation": _snap(graph, comp_sites), "future_transport": _snap(graph, ft_sites)}
    graph_arrays = _arrays(graph)
    del graph
    output.mkdir(parents=True, exist_ok=False)
    save_network(output / "road_graph.npz", graph_arrays)
    print("Saved the reusable sparse OSM road graph.", flush=True)
    results = {
        "graph": metrics, "computation_data_sha256": digest(computation),
        "future_transport_prepared_sha256": digest(prepared / "manifest.json"),
        "snapping": "Nearest OSM node in the directed connected component; access connectors excluded from distance and time",
        "coverage": {},
    }
    for study, snaps in sites.items():
        missing = [record for record in snaps if not record["covered"]]
        results["coverage"][study] = {
            "snap_count": len(snaps), "covered": len(snaps) - len(missing), "max_snap_m": max(r["offset_m"] for r in snaps),
            "uncovered": missing,
        }
        if missing:
            continue
        profile = _route_profile(
            graph_arrays, snaps, "travel_time" if study == "computation" else "distance",
            cells=cells if study == "future_transport" else None,
        )
        write_profile(output / f"{study}_matrix.json.gz", profile)
        print(f"Saved {study}: {len(snaps) * (len(snaps) - 1):,} directed routes.", flush=True)
    results["files_sha256"] = {file.name: digest(file) for file in sorted(output.iterdir()) if file.is_file()}
    write_json(output / "manifest.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--computation", type=Path, required=True)
    parser.add_argument("--future-transport", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.source, args.computation, args.future_transport, args.output)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
