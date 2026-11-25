"""Construct the published analytical supports directly from traceable source records."""

from __future__ import annotations

from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from threadpoolctl import threadpool_limits


SEED = 42
DEMAND_FIRST = "2018-06-01"
DEMAND_VALIDATION = "2018-08-27"
DEMAND_TEST = "2018-09-14"
RISK_FIRST = "2018-01-01"
RISK_VALIDATION = "2018-07-01"
RISK_TEST = "2018-07-23"
TEST_END = "2018-10-02"
DATA_END = "2018-12-31"
SOURCE_FILES = ("amazon_delivery_stops.csv.gz", "weather_2018.csv", "collisions_2018.csv")
TABLES = ("candidates", "depots", "cells", "weather", "demand_panel", "risk_panel", "mapped_lockers")
WEATHER_FIELDS = ("temperature_2m", "precipitation", "rain", "wind_speed_10m")
PROJECTION = Transformer.from_crs("EPSG:4326", "EPSG:3310", always_xy=True)
INVERSE = Transformer.from_crs("EPSG:3310", "EPSG:4326", always_xy=True)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, table: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        table.to_csv(stream, index=False, lineterminator="\n", float_format="%.17g")


def calendar(start: str, end: str) -> pd.Index:
    return pd.date_range(start, end, freq="D").strftime("%Y-%m-%d").rename("date")


def split(dates, validation: str, test: str) -> np.ndarray:
    return np.where(dates < validation, "train", np.where(dates < test, "validation", "test"))


def _coordinates(frame: pd.DataFrame) -> np.ndarray:
    coords = frame[["longitude", "latitude"]].to_numpy(dtype=float)
    if not np.isfinite(coords).all() or (np.abs(coords) > [180, 90]).any():
        raise ValueError("invalid WGS84 coordinates in source table")
    return coords


def _project(frame: pd.DataFrame) -> pd.DataFrame:
    positions = _coordinates(frame)
    x, y = PROJECTION.transform(positions[:, 0], positions[:, 1])
    return frame.assign(x_m=x, y_m=y)


def _daily_weather(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["time", *WEATHER_FIELDS])
    frame["date"] = pd.to_datetime(frame["time"], errors="raise").dt.strftime("%Y-%m-%d")
    if len(frame) != 8760 or frame["time"].duplicated().any():
        raise ValueError("expected one complete 2018 hourly weather archive")
    values = frame[list(WEATHER_FIELDS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, 1:] < 0).any():
        raise ValueError("weather archive contains missing or invalid values")
    grouped = frame.groupby("date", sort=True)
    if not (grouped.size() == 24).all() or not grouped.size().index.equals(calendar(RISK_FIRST, DATA_END)):
        raise ValueError("weather observations do not span the complete local 2018 calendar")
    return grouped.agg(
        temperature_2m=("temperature_2m", "mean"), precipitation=("precipitation", "sum"),
        rain=("rain", "sum"), wind_speed_10m=("wind_speed_10m", "mean"),
    ).reset_index()


def _demand(path: Path, clusters: int, starts: int, threads: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    ids = ["source_split", "route_id", "stop_id"]
    frame = pd.read_csv(
        path, keep_default_na=False,
        dtype={column: str for column in [*ids, "station_code", "service_date", "stop_type"]},
    )
    if frame.duplicated(ids).any() or (frame[ids + ["station_code"]] == "").any().any():
        raise ValueError("duplicate or empty route/stop key")
    dates = pd.to_datetime(frame["service_date"], format="%Y-%m-%d", errors="raise")
    if not dates.between(DEMAND_FIRST, TEST_END).all():
        raise ValueError("stop records do not match the June–October demand chronology")
    parcels = pd.to_numeric(frame["packages"], errors="raise").to_numpy()
    if (parcels < 0).any() or not np.isfinite(parcels).all() or (parcels % 1).any():
        raise ValueError("stop package counts must be finite nonnegative integers")
    frame["packages"] = parcels.astype(int)
    route_keys = ["source_split", "route_id"]
    station_fields = ["station_code", "station_latitude", "station_longitude"]
    unique_routes = frame[[*route_keys, *station_fields, "service_date"]].drop_duplicates()
    if unique_routes.duplicated(route_keys).any():
        raise ValueError("a route has conflicting station, coordinates or service date")
    depots = _project(
        unique_routes.groupby("station_code", as_index=False, sort=True)[station_fields[1:]]
        .median().rename(columns={"station_latitude": "latitude", "station_longitude": "longitude"})
    )
    delivery = frame.loc[~frame["stop_type"].str.casefold().eq("station")].copy()
    if delivery.empty or len(unique_routes) == 0:
        raise ValueError("the joined source contains no delivery stops")
    geographic = _coordinates(delivery)
    x, y = PROJECTION.transform(geographic[:, 0], geographic[:, 1])
    points, point_indices, frequencies = np.unique(
        np.column_stack([x, y]), axis=0, return_inverse=True, return_counts=True,
    )
    if len(points) < clusters:
        raise ValueError("candidate count exceeds the number of distinct delivery coordinates")
    model = KMeans(n_clusters=clusters, n_init=starts, max_iter=300, tol=1e-4, random_state=SEED)
    with threadpool_limits(limits=threads):
        model.fit(points, sample_weight=frequencies)
        centers = model.cluster_centers_
        centers = centers[np.lexsort((centers[:, 1], centers[:, 0]))]
        indices = pairwise_distances_argmin(points, centers)[point_indices]
        nearest = pairwise_distances_argmin(centers, depots[["x_m", "y_m"]].to_numpy())
    longitude, latitude = INVERSE.transform(centers[:, 0], centers[:, 1])
    candidate_ids = np.asarray([f"C{number:02d}" for number in range(1, clusters + 1)])
    candidates = pd.DataFrame({
        "candidate_id": candidate_ids, "latitude": latitude, "longitude": longitude,
        "x_m": centers[:, 0], "y_m": centers[:, 1],
        "station_code": depots.iloc[nearest]["station_code"].to_numpy(),
    })
    delivery["candidate_id"] = candidate_ids[indices]
    actual = delivery.groupby(["service_date", "candidate_id"])["packages"].sum()
    grid = pd.MultiIndex.from_product([calendar(DEMAND_FIRST, TEST_END), candidate_ids],
                                      names=["service_date", "candidate_id"])
    complete = actual.reindex(grid, fill_value=0).rename("actual").reset_index().rename(columns={"service_date": "date"})
    complete["split"] = split(complete["date"], DEMAND_VALIDATION, DEMAND_TEST)
    assigned = complete.groupby("date")["actual"].sum()
    dated = delivery.groupby("service_date")["packages"].sum().reindex(assigned.index, fill_value=0)
    if not np.array_equal(assigned, dated):
        raise ValueError("candidate assignment did not conserve linked package records by date")
    if assigned.sum() <= 0:
        raise ValueError("the stop cohort contains no packages for forecasting or routing")
    return candidates, depots, complete, {
        "routes": len(unique_routes), "source_stops": len(frame), "delivery_stops": len(delivery),
        "assigned_packages": int(assigned.sum()), "station_packages_outside_target": int(frame["packages"].sum() - assigned.sum()),
        "service_dates": int((assigned > 0).sum()), "vehicle_capacity": max(1, round(float(assigned.sum()) / len(unique_routes))),
        "candidate_count": clusters, "kmeans_starts": starts, "kmeans_weighted_distinct_points": len(points),
        "kmeans_inertia_projected_m2": float(model.inertia_), "depot_count": len(depots),
    }


def _risk(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frame = pd.read_csv(path, usecols=["dr_no", "date_occ", "latitude", "longitude"], dtype={"dr_no": str})
    if frame["dr_no"].isna().any() or frame["dr_no"].duplicated().any():
        raise ValueError("collision event identifiers must be unique and nonempty")
    dates = pd.to_datetime(frame["date_occ"], errors="raise").dt.strftime("%Y-%m-%d")
    if not dates.between(RISK_FIRST, DATA_END).all():
        raise ValueError("collision records do not match the 2018 source window")
    valid = frame[["longitude", "latitude"]].notna().all(axis=1) & frame["longitude"].ne(0) & frame["latitude"].ne(0)
    points = _coordinates(frame.loc[valid])
    if not len(points):
        raise ValueError("no geocoded collision records")
    grid = np.floor(np.round(points * 20, decimals=10)).astype(int)
    cells, indexes = np.unique(grid, axis=0, return_inverse=True)
    ids = np.asarray([f"G{x}_{y}" for x, y in cells])
    coordinates = _project(pd.DataFrame({
        "cell_id": ids, "longitude": (cells[:, 0] + 0.5) / 20, "latitude": (cells[:, 1] + 0.5) / 20,
        "grid_longitude_index": cells[:, 0], "grid_latitude_index": cells[:, 1],
    }))
    events = pd.DataFrame({"date": dates.loc[valid].to_numpy(), "cell_id": ids[indexes]}).value_counts()
    complete = pd.MultiIndex.from_product([calendar(RISK_FIRST, DATA_END), ids], names=["date", "cell_id"])
    panel = events.reindex(complete, fill_value=0).rename("collision_count").reset_index()
    panel["actual"] = panel["collision_count"].gt(0).astype(int)
    panel["split"] = np.where(panel["date"] <= TEST_END, split(panel["date"], RISK_VALIDATION, RISK_TEST), "outside")
    if panel["collision_count"].sum() != len(points):
        raise ValueError("collision target does not conserve geocoded event counts")
    return coordinates, panel, {
        "raw_events": len(frame), "geocoded_events": len(points), "invalid_location_events": int((~valid).sum()),
        "cell_count": len(cells), "grid_cell_degree": 0.05,
        "support": "All cells containing a geocoded event in the full 2018 retrospective source",
    }


def prepare(source: Path, output: Path, *, smoke: bool = False, threads: int = 4) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    paths = {name: source / name for name in SOURCE_FILES}
    optional = source / "osm_locker_nodes.csv"
    if optional.is_file():
        paths[optional.name] = optional
    fingerprints = {name: digest(path) for name, path in paths.items()}
    if smoke:
        receipt_hashes = {"manifest.json": digest(source / "manifest.json")}
    else:
        verifier = source / "source_verification.json"
        authority = json.loads(verifier.read_bytes())
        receipt_hashes = {"source_verification.json": digest(verifier)}
        for name in ("amazon_demand_manifest.json", "public_context_manifest.json"):
            receipt_hashes[name] = digest(source / name)
            if receipt_hashes[name] != authority["input_manifest_sha256"][name]:
                raise ValueError(f"input acquisition receipt fingerprint changed: {name}")
        amazon = json.loads((source / "amazon_demand_manifest.json").read_bytes())
        context = json.loads((source / "public_context_manifest.json").read_bytes())
        recorded = [amazon["output"], *(context[group]["output"] for group in ("weather", "collisions", "osm"))]
        if fingerprints != {entry["name"]: entry["sha256"] for entry in recorded}:
            raise ValueError("input records differ from the independent acquisition receipts")
    counts = _demand(paths[SOURCE_FILES[0]], 4 if smoke else 50, 3 if smoke else 50, threads)
    candidates, depots, demand, demand_receipt = counts
    cells, risk, risk_receipt = _risk(paths[SOURCE_FILES[2]])
    weather = _daily_weather(paths[SOURCE_FILES[1]])
    mapped = _project(pd.read_csv(optional, dtype={"osm_id": str})) if optional.name in paths else pd.DataFrame(
        columns=["osm_id", "latitude", "longitude", "x_m", "y_m"]
    )
    if mapped["osm_id"].duplicated().any():
        raise ValueError("duplicate mapped locker reference")
    output.mkdir(parents=True, exist_ok=False)
    tables = dict(zip(TABLES, (candidates, depots, cells, weather, demand, risk, mapped), strict=True))
    for name, table in tables.items():
        write_csv(output / f"{name}.csv", table)
    manifest = {
        "schema": 1, "kind": "future_transport_prepared", "smoke": smoke, "seed": SEED,
        "source_sha256": fingerprints, "source_receipt_sha256": receipt_hashes,
        "sources": {"amazon": "https://registry.opendata.aws/amazon-last-mile-challenges/",
                    "weather": "https://open-meteo.com/en/docs/historical-weather-api",
                    "collisions": "https://data.lacity.org/resource/d5tf-ez2w.json"},
        "demand": demand_receipt, "risk": risk_receipt, "mapped_reference_points": len(mapped),
        "candidate_geometry": "Retrospective full-period deliveries and frequency-weighted EPSG:3310 k-means",
        "weather": "Retrospective LA hourly weather aggregated to local calendar dates. Models use preceding-day observations only.",
        "files_sha256": {f"{name}.csv": digest(output / f"{name}.csv") for name in TABLES},
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def prepared(folder: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    manifest = json.loads((folder / "manifest.json").read_bytes())
    if manifest.get("kind") != "future_transport_prepared" or manifest.get("schema") != 1:
        raise ValueError("expected a new version 1 Future Transportation prepared corpus")
    tables = {}
    for name in TABLES:
        filename = f"{name}.csv"
        path = folder / filename
        if manifest["files_sha256"][filename] != digest(path):
            raise ValueError(f"prepared table fingerprint changed: {filename}")
        tables[name] = pd.read_csv(path)
    return manifest, tables
