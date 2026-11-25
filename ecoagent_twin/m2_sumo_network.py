"""Generate an abstract SUMO road grid and controlled electric OD probes."""

from __future__ import annotations

from itertools import permutations, product
from pathlib import Path
from random import Random
from xml.etree import ElementTree as ET


CONDITIONS = ("light_traffic", "heavy_traffic")
SITES = {
    "depot": (-85, 0, (0, 0)),
    "central": (648, 658, (1, 1)),
    "east": (1285, 600, (2, 1)),
    "west": (-85, 600, (0, 1)),
    "north": (600, 1285, (1, 2)),
    "start-1": (1285, 1200, (2, 2)),
    "start-2": (0, 1285, (0, 2)),
}
GRID_SPACING_M = 600
MAIN_ROAD_SPEED_M_S = 10
SITE_ACCESS_SPEED_M_S = 6
PROBE_BATTERY_CAPACITY_WH = 35_000
PROBE_INITIAL_BATTERY_WH = 26_000


def _junction(x: int, y: int) -> str:
    return f"j{x}{y}"


def _write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="  ")
    with path.open("xb") as stream:
        ET.ElementTree(root).write(stream, encoding="utf-8", xml_declaration=True)


def write_network(directory: Path) -> None:
    """Write a strongly connected 3-by-3 network and seven bidirectional site links."""
    nodes = ET.Element("nodes")
    edges = ET.Element("edges")
    junction_cells = list(product(range(3), repeat=2))
    for x, y in junction_cells:
        ET.SubElement(nodes, "node", {
            "id": _junction(x, y),
            "x": str(x * GRID_SPACING_M),
            "y": str(y * GRID_SPACING_M),
            "type": "traffic_light" if (x, y) == (1, 1) else "priority",
        })
    for x, y in junction_cells:
        for dx, dy in ((0, 1), (1, 0)):
            neighbor = x + dx, y + dy
            if neighbor in junction_cells:
                first, second = _junction(x, y), _junction(*neighbor)
                for start, end in ((first, second), (second, first)):
                    ET.SubElement(edges, "edge", {
                        "id": f"road_{start}_{end}", "from": start, "to": end,
                        "numLanes": "1", "speed": str(MAIN_ROAD_SPEED_M_S), "priority": "2",
                    })
    for site, (x, y, cell) in SITES.items():
        ET.SubElement(nodes, "node", {"id": site, "x": str(x), "y": str(y), "type": "priority"})
        junction = _junction(*cell)
        for direction, start, end in (("out", site, junction), ("in", junction, site)):
            ET.SubElement(edges, "edge", {
                "id": f"access_{site}_{direction}", "from": start, "to": end,
                "numLanes": "1", "speed": str(SITE_ACCESS_SPEED_M_S), "priority": "1",
            })
    _write_xml(nodes, directory / "roads.nod.xml")
    _write_xml(edges, directory / "roads.edg.xml")


def _background_trips(condition: str, seed: int) -> list[tuple[int, str, str, str]]:
    """Keep sparse traffic in both arms and add an overlapping demand pulse in heavy."""
    sparse_corridors = (
        ("west", "east"), ("east", "west"), ("depot", "north"),
        ("north", "depot"), ("start-1", "start-2"), ("start-2", "start-1"),
    )
    # High-rate corridors overlap the signalized center. Departures stop while
    # probes are still active, allowing all routes to clear without teleports.
    pulse_corridors = (
        ("west", "east", 6), ("east", "west", 6), ("depot", "north", 10),
        ("north", "depot", 10), ("start-1", "west", 14), ("start-2", "east", 14),
    )
    rng = Random(seed)
    trips = []
    for index, (origin, destination) in enumerate(sparse_corridors):
        for time in range(5 + 3 * index, 390, 95):
            depart = time + rng.randint(0, 3)
            trips.append((depart, f"ambient_{len(trips):04d}", origin, destination))
    if condition == "heavy_traffic":
        for index, (origin, destination, interval) in enumerate(pulse_corridors):
            for time in range(15 + index, 395, interval):
                depart = time + rng.randint(0, 2)
                trips.append((depart, f"ambient_{len(trips):04d}", origin, destination))
    return trips


def write_demand(directory: Path, condition: str, seed: int) -> dict[str, tuple[str, str]]:
    """Write one battery-equipped probe on each ordered distinct site pair."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown traffic condition: {condition}")
    if type(seed) is not int:
        raise ValueError("the traffic seed must be an integer")

    routes = ET.Element("routes")
    probe_type = ET.SubElement(routes, "vType", {
        "id": "probe_ev", "vClass": "delivery", "emissionClass": "Energy/unknown",
        "mass": "1900", "length": "5", "minGap": "2.5", "accel": "1.4",
        "decel": "3.0", "sigma": "0", "maxSpeed": "10", "speedFactor": "1",
    })
    # Abstract electric van inputs. The probe battery supplies the diagnostic
    # trip and is independent of the M2 decision environment's energy budget.
    for key, value in (
        ("has.battery.device", "true"),
        ("device.battery.capacity", str(PROBE_BATTERY_CAPACITY_WH)),
        ("constantPowerIntake", "350"),
        ("frontSurfaceArea", "3"),
        ("airDragCoefficient", "0.36"),
        ("rollDragCoefficient", "0.01"),
        ("propulsionEfficiency", "0.90"),
        ("recuperationEfficiency", "0.72"),
    ):
        ET.SubElement(probe_type, "param", {"key": key, "value": value})
    ET.SubElement(routes, "vType", {
        "id": "ambient_car", "vClass": "passenger", "length": "4.5", "minGap": "2.5",
        "accel": "2.0", "decel": "4.5", "sigma": "0", "maxSpeed": "10", "speedFactor": "1",
    })

    probes = {}
    departures = _background_trips(condition, seed)
    for index, (origin, destination) in enumerate(permutations(SITES, 2)):
        probe_id = f"probe_{origin}__{destination}"
        probes[probe_id] = origin, destination
        departures.append((90 + 6 * index, probe_id, origin, destination))
    for depart, trip_id, origin, destination in sorted(departures):
        is_probe = trip_id in probes
        trip = ET.SubElement(routes, "trip", {
            "id": trip_id, "type": "probe_ev" if is_probe else "ambient_car",
            "depart": str(depart), "from": f"access_{origin}_out", "to": f"access_{destination}_in",
        })
        if is_probe:
            ET.SubElement(trip, "param", {
                "key": "device.battery.chargeLevel", "value": str(PROBE_INITIAL_BATTERY_WH),
            })
    _write_xml(routes, directory / "traffic.rou.xml")
    return probes
