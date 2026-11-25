"""Bind OSMnx regional road paths to the Amazon benchmark and M6 forecast sites."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

from .empirical_amazon import file_sha256, read_records
from .m6_empirical_benchmark import route_window


def _matrix_rows(profile: dict, *, include_depot: bool):
    for i, start in enumerate(profile["sites"]):
        for j, end in enumerate(profile["sites"]):
            if include_depot or "depot" not in (start, end):
                yield (start, end, profile["distance_km"][i][j], profile["travel_min"][i][j], "osmnx", True)


def build(graph_path: Path, amazon: Path, processed: Path, output: Path, *,
          graph_sha256: str | None = None, acquisition: Path | None = None) -> None:
    from study_rebuild.real_roads.roads import Site, load_network, route_sites, write_json

    graph_digest = file_sha256(graph_path)
    manifest_path = graph_path.with_name("manifest.json")
    if graph_sha256 is not None and acquisition is not None:
        graph_source = json.loads(acquisition.read_bytes())
        expected = graph_sha256
    else:
        graph_manifest = json.loads(manifest_path.read_bytes())
        graph_source = graph_manifest["graph"]
        expected = graph_manifest["files_sha256"][graph_path.name]
    if expected != graph_digest:
        raise ValueError("Regional OSMnx graph checksum mismatch")
    graph = load_network(graph_path)
    provenance = {"graph_file_sha256": graph_digest, "graph": graph_source,
                  "objective": "shortest directed road length between snapped access nodes",
                  "turn_relations": "Road direction and vehicle access are modeled. Turn restrictions are not modeled."}
    output.mkdir(parents=True, exist_ok=False)
    sites_path = processed / "la_phase5_top30_candidate_lockers.csv"
    with sites_path.open(encoding="utf-8-sig", newline="") as stream:
        sites = [Site(row["locker_id"], float(row["lng"]), float(row["lat"])) for row in csv.DictReader(stream)]
    if len(sites) != 30:
        raise ValueError("Expected 30 retained forecast candidates")
    sites.insert(0, Site("depot", fmean(site.longitude for site in sites), fmean(site.latitude for site in sites)))
    forecast = route_sites(graph, sites, mode="travel_time")
    write_json(output / "forecast_road_profile.json", {"profile": forecast, "provenance": {
        **provenance, "objective": "least modeled road travel time, with distance from the same path",
        "depot": "centroid of the 30 retained candidate sites", "site_csv_sha256": file_sha256(sites_path),
    }})
    for name in ("la_locker_demand_daily.csv", sites_path.name):
        (output / name).write_bytes((processed / name).read_bytes())
    for filename, include_depot in (("la_locker_road_time_matrix_api.csv", False),
                                    ("la_depot_locker_road_matrix_api.csv", True)):
        with (output / filename).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("from_locker_id", "to_locker_id", "distance_km", "travel_time_min", "matrix_source", "true_road_matrix"))
            writer.writerows(_matrix_rows(forecast, include_depot=include_depot))
    by_route = {}
    for index, record in enumerate(read_records(amazon), 1):
        source_stops = route_window(record)[0]
        stops = record["route"]["stops"]
        nodes = [Site(site, stops[stop]["lng"], stops[stop]["lat"]) for site, stop in source_stops.items()]
        profile = route_sites(graph, nodes, mode="distance")
        by_route[record["route_id"]] = {"source_stops": source_stops, "snaps": profile["snaps"],
                                       "path_sha256": profile["ordered_osm_node_and_way_sha256"],
                                       "distance_m": {start: {end: meters * 1000 for end, meters in zip(
                                           profile["sites"], row, strict=True)} for start, row in zip(
                                           profile["sites"], profile["distance_km"], strict=True)}}
        if index % 10 == 0:
            print(f"{index} Amazon windows routed", flush=True)
    write_json(output / "amazon_osmnx_distances.json", {"schema": 1, "source": "osmnx", "provenance": provenance,
               "source_records_sha256": file_sha256(amazon / "records.jsonl"), "by_route": by_route})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("graph", "amazon", "processed", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--graph-sha256", help="verified fingerprint when the regional experiment manifest is still being generated")
    parser.add_argument("--acquisition", type=Path, help="source acquisition receipt accompanying --graph-sha256")
    args = parser.parse_args()
    if (args.graph_sha256 is None) != (args.acquisition is None):
        parser.error("--graph-sha256 and --acquisition must be supplied together")
    build(args.graph, args.amazon, args.processed, args.output, graph_sha256=args.graph_sha256, acquisition=args.acquisition)


if __name__ == "__main__":
    main()
