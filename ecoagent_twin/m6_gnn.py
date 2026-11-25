"""Train a new next-day graph demand predictor on aligned historical locker aggregates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import date, datetime, timedelta, timezone
from hashlib import file_digest
import json
from math import floor, isclose, isfinite, pi
from pathlib import Path
from statistics import median

import numpy as np
import torch
from torch import nn


LAGS = 7
HIDDEN = 32
METHODS = ("gnn", "local_mlp", "lag_one", "lag_week", "trailing_mean")


class DemandGNN(nn.Module):
    """Use directed road-neighbor messages as well as a node's own history."""

    def __init__(self) -> None:
        super().__init__()
        self.local = nn.Linear(LAGS + 3, HIDDEN)
        self.neighbors = nn.Linear(LAGS + 3, HIDDEN, bias=False)
        self.local_output = nn.Linear(HIDDEN, 1)
        self.neighbor_output = nn.Linear(HIDDEN, 1, bias=False)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        latent = torch.relu(self.local(x) + self.neighbors(adjacency @ x))
        return nn.functional.softplus(
            self.local_output(latent) + self.neighbor_output(adjacency @ latent)
        ).squeeze(-1)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _road_graph(path: Path, neighbors: int) -> tuple[tuple[str, ...], np.ndarray, str]:
    times = {}
    sources = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            arc = row["from_locker_id"], row["to_locker_id"]
            minutes = float(row["travel_time_min"])
            source = row["matrix_source"]
            if (arc in times or not isfinite(minutes) or minutes < 0
                    or source not in {"osrm_table_api", "osmnx"}
                    or row["true_road_matrix"].lower() != "true"):
                raise ValueError(f"Invalid road matrix row for {arc}")
            times[arc] = minutes
            sources.add(source)
    ids = tuple(sorted({node for arc in times for node in arc}))
    if len(sources) != 1 or not 1 <= neighbors < len(ids) or len(times) != len(ids) ** 2:
        raise ValueError("Expected a complete directed road matrix from one source and a valid neighborhood size")
    graph = np.zeros((len(ids), len(ids)), dtype=np.float32)
    for row, origin in enumerate(ids):
        nearest = sorted(
            (col for col, target in enumerate(ids) if target != origin),
            key=lambda col: (times[origin, ids[col]], ids[col]),
        )[:neighbors]
        weights = np.array([1 / max(1.0, times[origin, ids[col]]) for col in nearest])
        graph[row, nearest] = weights / weights.sum()
    return ids, graph, sources.pop()


def _panel(path: Path, ids: tuple[str, ...]) -> tuple[list[date], np.ndarray]:
    indexed = {node: index for index, node in enumerate(ids)}
    counts = defaultdict(lambda: np.zeros(len(ids), dtype=np.float32))
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            node = row["locker_id"]
            if node not in indexed:
                continue
            day, packages = date.fromisoformat(row["date"]), int(row["packages_assigned"])
            if packages < 0 or (day, node) in seen:
                raise ValueError(f"Invalid or duplicated locker-day aggregate for {(day, node)}")
            seen.add((day, node))
            counts[day][indexed[node]] = packages
    if {node for _, node in seen} != set(ids):
        raise ValueError("Historical records and directed matrix must cover the same locker IDs")
    first, last = min(counts), max(counts)
    days = [first + timedelta(days=offset) for offset in range((last - first).days + 1)]
    if len(days) < 30:
        raise ValueError("At least 30 calendar days are needed for a chronological evaluation")
    return days, np.stack([counts[day] for day in days])


def _day_features(day: date, lag: np.ndarray, means: np.ndarray, scale: float) -> np.ndarray:
    angle = 2 * pi * day.weekday() / 7
    days_of_week = np.tile([np.sin(angle), np.cos(angle)], (len(means), 1))
    return np.column_stack((lag.T / scale, means, days_of_week))


def predict(checkpoint: Path, target_day: date, history: dict[date, dict[str, int]]) -> dict:
    """Predict one day using exactly its preceding seven complete aligned histories."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["schema"] != 1:
        raise ValueError("Unsupported GNN checkpoint schema")
    ids = state["locker_ids"]
    prior = [history[target_day - timedelta(days=offset)] for offset in range(LAGS, 0, -1)]
    if any(set(day) != set(ids) for day in prior):
        raise ValueError("History locker IDs do not align with the checkpoint graph")
    if any(type(value) is not int or value < 0 for day in prior for value in day.values()):
        raise ValueError("History requires nonnegative integer counts")
    lag = np.array([[day[node] for node in ids] for day in prior])
    means = state["training_node_means_scaled"].numpy()
    scale = state["training_scale"]
    features = torch.tensor(_day_features(target_day, lag, means, scale), dtype=torch.float32)
    model = DemandGNN()
    model.load_state_dict(state["state_dict"])
    model.eval()
    with torch.inference_mode():
        values = (model(features, state["adjacency"]) * scale).tolist()
    return {
        "issue_date": (target_day - timedelta(days=1)).isoformat(),
        "target_date": target_day.isoformat(),
        "forecast_by_locker": dict(zip(ids, values, strict=True)),
    }


def _learn(x, y, graph, train_count: int, validation_end: int, seed: int, epochs: int):
    torch.manual_seed(seed)
    model = DemandGNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    best = (float("inf"), 0, {})
    for epoch in range(1, epochs + 1):
        predicted = model(x[:train_count], graph)
        loss = nn.functional.smooth_l1_loss(predicted, y[:train_count])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.inference_mode():
            error = torch.mean(torch.abs(model(x[train_count:validation_end], graph)
                                         - y[train_count:validation_end])).item()
        if error < best[0]:
            best = (error, epoch, {
                key: value.detach().clone() for key, value in model.state_dict().items()
            })
    model.load_state_dict(best[2])
    return model.eval(), best[1]


def staging_allocations(forecast: dict[str, float], available_slots: int) -> dict[str, int]:
    """Assign a fixed assumed slot budget by predicted demand and largest remainders."""
    if (type(available_slots) is not int or available_slots < 0 or not forecast
            or any(not isfinite(value) or value < 0 for value in forecast.values())):
        raise ValueError("Slot budget and forecasts must be finite and nonnegative")
    ids = sorted(forecast)
    total = sum(forecast.values())
    quota = {
        node: available_slots * (forecast[node] / total if total else 1 / len(ids))
        for node in ids
    }
    assignments = {node: floor(quota[node]) for node in ids}
    residual = available_slots - sum(assignments.values())
    for node in sorted(ids, key=lambda node: (assignments[node] - quota[node], node))[:residual]:
        assignments[node] += 1
    return assignments


def forecast_experiment(
    demand: Path, matrix: Path, output: Path, *, seed: int = 631, epochs: int = 120, neighbors: int = 5,
) -> dict:
    """Fit on the first 60% of dates and choose epochs on the next 20% only."""
    if type(epochs) is not int or epochs < 1:
        raise ValueError("Epoch count must be positive")
    ids, graph, road_source = _road_graph(matrix, neighbors)
    days, panel = _panel(demand, ids)
    before = {str(path.resolve()): _digest(path) for path in (demand, matrix)}
    train_end, validation_end = (len(days) * percent // 100 for percent in (60, 80))
    train_count, validation_count = train_end - LAGS, validation_end - LAGS
    scale = max(1.0, float(panel[:train_end].mean()))
    means = panel[:train_end].mean(axis=0) / scale
    x = torch.tensor(np.stack([
        _day_features(days[index], panel[index - LAGS:index], means, scale)
        for index in range(LAGS, len(days))
    ]), dtype=torch.float32)
    y = torch.tensor(panel[LAGS:] / scale)
    adjacency = torch.tensor(graph)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gnn, gnn_epoch = _learn(x, y, adjacency, train_count, validation_count, seed, epochs)
    no_edges = torch.zeros_like(adjacency)
    local, local_epoch = _learn(x, y, no_edges, train_count, validation_count, seed, epochs)
    with torch.inference_mode():
        forecasts = {
            "gnn": gnn(x[validation_count:], adjacency).numpy() * scale,
            "local_mlp": local(x[validation_count:], no_edges).numpy() * scale,
            "lag_one": panel[validation_end - 1:-1],
            "lag_week": panel[validation_end - 7:-7],
            "trailing_mean": np.stack([
                panel[index - LAGS:index].mean(axis=0)
                for index in range(validation_end, len(days))
            ]),
        }
    observed = panel[validation_end:]
    slots = int(0.6 * median(panel[:train_end].sum(axis=1)))
    records = []
    covered = defaultdict(int)
    changed = 0
    for index, day in enumerate(days[validation_end:]):
        options = {method: dict(zip(ids, map(float, prediction[index]), strict=True))
                   for method, prediction in forecasts.items()}
        allocations = {method: staging_allocations(values, slots) for method, values in options.items()}
        actual = dict(zip(ids, map(int, observed[index]), strict=True))
        for method, allocation in allocations.items():
            covered[method] += sum(min(actual[node], allocation[node]) for node in ids)
        changed += allocations["gnn"] != allocations["trailing_mean"]
        records.append({
            "target_date": day.isoformat(), "issue_date": (day - timedelta(days=1)).isoformat(),
            "observed_by_locker": actual, "forecast_by_locker": options,
            "staged_slots_by_locker": allocations,
        })
    denominator = float(observed.sum())
    scores = {}
    for method, prediction in forecasts.items():
        error = prediction - observed
        scores[method] = {
            "mae_packages": float(np.abs(error).mean()),
            "rmse_packages": float(np.sqrt((error ** 2).mean())),
            "wape": float(np.abs(error).sum() / denominator) if denominator else None,
            "historical_demand_covered_under_assumed_slots": covered[method],
        }
    if before != {str(path.resolve()): _digest(path) for path in (demand, matrix)}:
        raise ValueError("An input changed while the forecasting experiment was running")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "gnn.pt"
    torch.save({"schema": 1, "state_dict": gnn.state_dict(), "adjacency": adjacency,
                "locker_ids": ids, "training_scale": scale,
                "training_node_means_scaled": torch.tensor(means)}, checkpoint)
    with (output / "forecasts.json").open("x", encoding="utf-8") as stream:
        json.dump(records, stream, indent=2, allow_nan=False)
    geometry = "OSMnx" if road_source == "osmnx" else "OSRM"
    report = {
        "scope": f"New demand GNN extension using archived analytical locker-day aggregates and {geometry} geometry",
        "historical_checkpoint_replay": False,
        "inputs_sha256": before, "checkpoint_sha256": _digest(checkpoint),
        "forecast_sha256": _digest(output / "forecasts.json"),
        "locker_count": len(ids), "directed_neighbors": neighbors, "seed": seed,
        "lag_calendar_days": LAGS, "epoch_limit": epochs,
        "selected_epochs": {"gnn": gnn_epoch, "local_mlp": local_epoch},
        "dates": {
            "train_through": days[train_end - 1].isoformat(),
            "validation_from": days[train_end].isoformat(),
            "validation_through": days[validation_end - 1].isoformat(),
            "test_from": days[validation_end].isoformat(),
            "test_through": days[-1].isoformat(),
            "holdout_days": len(records),
        },
        "observation": "Absent archived node-day rows count as zero recorded assignments; forecasts are conditional on fixed retrospective candidate IDs.",
        "scores": scores, "assumed_staging_slots_per_day": slots,
        "gnn_allocation_differs_from_trailing_mean_days": changed,
        "planning_scope": "Offline slot-staging calculation evaluated on holdout demand; physical locker inventory, fleet delivery, and road-risk prediction are outside this lane.",
    }
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def verify(demand: Path, matrix: Path, output: Path) -> dict:
    """Check saved predictions and slot calculations using checkpoint inference."""
    report = json.loads((output / "summary.json").read_bytes())
    checkpoint, forecasts = output / "gnn.pt", output / "forecasts.json"
    if ({_digest(path) for path in (demand, matrix)} != set(report["inputs_sha256"].values())
            or _digest(checkpoint) != report["checkpoint_sha256"]
            or _digest(forecasts) != report["forecast_sha256"]):
        raise ValueError("The historical inputs or saved forecast fingerprints differ")
    ids, _, _ = _road_graph(matrix, report["directed_neighbors"])
    days, panel = _panel(demand, ids)
    daily = {day: dict(zip(ids, map(int, values), strict=True))
             for day, values in zip(days, panel, strict=True)}
    records = json.loads(forecasts.read_bytes())
    test_days = [day for day in days if day >= date.fromisoformat(report["dates"]["test_from"])]
    maximum = 0.0
    covered = 0
    torch.set_num_threads(1)
    for day, row in zip(test_days, records, strict=True):
        replay = predict(checkpoint, day, daily)
        if (replay["target_date"] != row["target_date"] or replay["issue_date"] != row["issue_date"]
                or row["observed_by_locker"] != daily[day]):
            raise ValueError("Forecast issue, target or observation differs from the held-out calendar")
        published = row["forecast_by_locker"]["gnn"]
        for node, value in replay["forecast_by_locker"].items():
            maximum = max(maximum, abs(published[node] - value))
            if not isclose(published[node], value, rel_tol=2e-6, abs_tol=5e-5):
                raise ValueError(f"Checkpoint inference did not reproduce {node} on {day}")
        allocation = staging_allocations(published, report["assumed_staging_slots_per_day"])
        if allocation != row["staged_slots_by_locker"]["gnn"]:
            raise ValueError(f"Saved staging plan did not reproduce on {day}")
        covered += sum(min(daily[day][node], allocation[node]) for node in ids)
    if covered != report["scores"]["gnn"]["historical_demand_covered_under_assumed_slots"]:
        raise ValueError("Saved GNN planning coverage differs from replay")
    return {"checked_days": len(records), "maximum_prediction_difference": maximum,
            "covered_historical_assignments": covered}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demand", type=Path, help="Processed LA locker-day demand CSV")
    parser.add_argument("matrix", type=Path, help="Aligned directed OSRM CSV")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--output", type=Path)
    modes.add_argument("--verify", type=Path, help="Verify an existing run without refitting")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path("runs") / f"m6_gnn_{stamp}"
    try:
        if args.verify is not None:
            print(json.dumps(verify(args.demand, args.matrix, args.verify)))
            return
        report = forecast_experiment(args.demand, args.matrix, output)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(output.resolve())
    print(f"{report['locker_count']} aligned nodes and {report['dates']['holdout_days']} holdout days")


if __name__ == "__main__":
    main()
