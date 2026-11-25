"""Transparent EV and operational carbon accounting for the Amazon route windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


EPA_EGRID_2018 = "https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P1015YKH.TXT"
EPA_GASOLINE_CO2 = "https://www.epa.gov/greenvehicles/greenhouse-gas-rating"
CAMX_CO2E_LB_MWH = 498.7
CAMX_GRID_LOSS = 0.048
GRID_KG_CO2E_KWH = CAMX_CO2E_LB_MWH * 0.45359237 / 1000
ROAD_CIRCUITY = 1.25
CAR_KG_CO2_PER_KM = 8.887 / (25 * 1.609344)
CUSTOMER_CAR_SHARE = 0.5


@dataclass(frozen=True)
class EVProfile:
    id: str
    nominal_pack_wh: int
    movement_wh_per_km: float
    travel_auxiliary_w: int
    charging_efficiency: float


PROFILES = (
    EVProfile("standard_van", 68_000, 240, 450, 0.90),
    EVProfile("heavy_van", 85_000, 330, 700, 0.88),
)


def road_proxy_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Apply stated circuity to the obfuscated stop coordinate distance."""
    a, b = map(math.radians, first)
    c, d = map(math.radians, second)
    haversine = math.sin((c - a) / 2) ** 2 + math.cos(a) * math.cos(c) * math.sin((d - b) / 2) ** 2
    return ROAD_CIRCUITY * 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(max(0.0, haversine))))


def leg_wh(distance_km: float, seconds: float, profile: EVProfile) -> int:
    return math.ceil(distance_km * profile.movement_wh_per_km + seconds * profile.travel_auxiliary_w / 3600)


def carbon_receipt(
    *, traction_wh: int, fallback_wh: int, charged_wh: int, initial_wh: int, terminal_wh: int,
    all_car_pickup_km: float, profile: EVProfile,
) -> dict:
    """Return average-grid carbon including fallback and battery inventory equalization."""
    if initial_wh + charged_wh - traction_wh != terminal_wh:
        raise ValueError("battery energy does not reconcile")
    delivery_factor = profile.charging_efficiency * (1 - CAMX_GRID_LOSS)
    in_service_grid = charged_wh / 1000 / delivery_factor
    terminal_and_fallback_grid = (initial_wh - terminal_wh + fallback_wh) / 1000 / delivery_factor
    grid_total = (traction_wh + fallback_wh) / 1000 / delivery_factor
    carrier = grid_total * GRID_KG_CO2E_KWH
    customer_all_car = all_car_pickup_km * CAR_KG_CO2_PER_KM
    return {
        "initial_battery_wh": initial_wh,
        "terminal_battery_wh": terminal_wh,
        "onboard_recharge_wh": charged_wh,
        "carrier_fallback_traction_wh": fallback_wh,
        "in_window_grid_generation_kwh": in_service_grid,
        "inventory_adjustment_and_fallback_grid_generation_kwh": terminal_and_fallback_grid,
        "total_grid_generation_kwh": grid_total,
        "carrier_operational_kg_co2e": carrier,
        "incremental_pickup_km_if_all_car": all_car_pickup_km,
        "customer_pickup_kg_co2_at_assumed_share": CUSTOMER_CAR_SHARE * customer_all_car,
        "operational_kg_co2e_at_assumed_share": carrier + CUSTOMER_CAR_SHARE * customer_all_car,
        "operational_kg_co2e_no_car": carrier,
        "operational_kg_co2e_all_car": carrier + customer_all_car,
    }


def assumptions() -> dict:
    return {
        "profiles": [asdict(profile) for profile in PROFILES],
        "profile_status": "modeled sensitivity values; Amazon has no observed EV energy/charging data",
        "movement_model": "Wh per km of 1.25× obfuscated-coordinate great-circle distance plus travel auxiliary Wh",
        "grid": {
            "scope": "EPA eGRID 2018 CAMX annual total-output generation; constant over each route window",
            "source": EPA_EGRID_2018,
            "lb_co2e_per_mwh": CAMX_CO2E_LB_MWH,
            "kg_co2e_per_kwh_generated": GRID_KG_CO2E_KWH,
            "grid_gross_loss": CAMX_GRID_LOSS,
        },
        "pickup": {
            "scope": "one round-trip per original stop/alternate locker pair, 0/50/100% dedicated car sensitivity",
            "assumed_mpg": 25,
            "gasoline_kg_co2_per_gallon": 8.887,
            "source": EPA_GASOLINE_CO2,
        },
        "boundary": (
            "Carrier traction and moving auxiliary electricity, grid and charging loss, modeled deferred carrier "
            "delivery of all missed parcels, battery inventory equalization, and incremental customer tailpipe "
            "pickup CO2. Excludes manufacture, infrastructure, upstream gasoline, and stationary auxiliary load."
        ),
    }
