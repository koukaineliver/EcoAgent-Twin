"""Explain one fingerprinted Month 4 simulation trace without rerunning a policy."""

from __future__ import annotations

import argparse
from hashlib import sha256
from html import escape
import json
from math import isclose, isfinite
from pathlib import Path
import re


_ARM_NAMES = ("deterministic_supervisor", "agentic_supervisor")
_ARM_LABELS = ("Deterministic", "Agentic")


def _load_case(run_dir: Path, case_id: str | None) -> tuple[dict, dict]:
    manifest = json.loads((run_dir / "manifest.json").read_bytes())
    try:
        fingerprint = manifest["sha256"]["traces.jsonl"]
    except (KeyError, TypeError) as error:
        raise ValueError("manifest does not identify the trace fingerprint") from error
    trace_path = (run_dir / "traces.jsonl").resolve()
    if not trace_path.is_relative_to(run_dir):
        raise ValueError("trace path is outside the run directory")
    payload = trace_path.read_bytes()
    if sha256(payload).hexdigest() != fingerprint:
        raise ValueError("trace fingerprint differs from the manifest")

    selected = None
    seen = set()
    for line_number, raw_line in enumerate(payload.splitlines(), 1):
        row = json.loads(raw_line)
        key = row.get("case_id") if isinstance(row, dict) else None
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError(f"missing or duplicate case ID at trace line {line_number}")
        seen.add(key)
        if (case_id is None and selected is None) or key == case_id:
            selected = row, {
                "trace_path": str(trace_path), "trace_line": line_number, "trace_sha256": fingerprint,
            }
    if selected is None:
        raise ValueError(f"no trace case found for {case_id!r}")
    return selected


def _nonnegative(value) -> bool:
    return type(value) in (int, float) and isfinite(value) and value >= 0


def _offers(case: dict) -> dict:
    ledger = case["offers"]
    metrics, assignments = ledger["metrics"], ledger["assignments"]
    accepted = [event for event in ledger["events"] if event["kind"] == "offer_accepted"]
    paid = sum(event["paid_cents"] for event in accepted)
    expected_assignments = {event["parcel_id"]: event["locker_id"] for event in accepted}
    counts = {
        "issued": "offer_issued", "declined": "offer_declined", "expired": "offer_expired",
    }
    for field, kind in counts.items():
        if metrics[field] != sum(event["kind"] == kind for event in ledger["events"]):
            raise ValueError(f"offer event count disagrees with ledger metric {field}")
    if (
        len(accepted) != len(expected_assignments)
        or assignments != expected_assignments
        or not len(accepted) <= metrics["accepted"] <= metrics["issued"]
        or metrics["reassignments"] != len(assignments)
        or paid != metrics["spent_cents"]
        or not 0 <= paid <= metrics["budget_cents"]
        or any(type(event["paid_cents"]) is not int or event["paid_cents"] < 0 for event in accepted)
    ):
        raise ValueError("offer commitments, payments, or budget disagree with the ledger")
    for alert_field, actual in (
        ("accepted_reassignments", len(assignments)),
        ("offer_budget_cents", metrics["budget_cents"]),
        ("credit_spent_cents", paid),
    ):
        if case["alert"].get(alert_field, actual) != actual:
            raise ValueError(f"alert disagrees with offer ledger metric {alert_field}")
    return {
        "issued": metrics["issued"], "declined": metrics["declined"], "expired": metrics["expired"],
        "accept_responses": metrics["accepted"],
        "budget_cents": metrics["budget_cents"], "spent_cents": paid,
        "committed_slots": [
            {key: event[key] for key in ("parcel_id", "locker_id", "time_min", "paid_cents")}
            for event in accepted
        ],
    }


def _candidate(menu_entry: dict, record: dict, offers: dict, alert: dict) -> dict:
    candidate_id = menu_entry["id"]
    metrics = record["result"]["metrics"]
    events = record["result"]["events"]
    unserved = record["result"]["unserved_parcel_ids"]
    deposits = [event for event in events if event["kind"] == "parcel_deposited"]
    deposited_ids = {event["parcel_id"] for event in deposits}
    served, total = metrics["served_parcels"], metrics["total_parcels"]
    times = [event["time_min"] for event in events]
    if (
        len(deposited_ids) != len(deposits)
        or len(set(unserved)) != len(unserved)
        or deposited_ids.intersection(unserved)
        or served != len(deposits)
        or metrics["unserved_parcels"] != len(unserved)
        or total != served + len(unserved)
        or total <= 0
        or not isclose(metrics["service_rate"], served / total)
        or times != sorted(times)
    ):
        raise ValueError(f"result events and service metrics disagree for {candidate_id}")

    assignments = {item["parcel_id"]: item["locker_id"] for item in offers["committed_slots"]}
    slots = set(assignments.items())
    routing_slots = [
        (event["parcel_id"], event["locker_id"])
        for event in events if event["kind"] == "offer_slot_preassigned"
    ]
    if slots != set(routing_slots) or len(routing_slots) != len(slots):
        raise ValueError(f"committed offer slots disagree with routing events for {candidate_id}")
    if any(event["locker_id"] != assignments[event["parcel_id"]] for event in deposits
           if event["parcel_id"] in assignments):
        raise ValueError(f"a deposited parcel violated its committed slot for {candidate_id}")
    if metrics["charged_wh"] != sum(
        event["charged_wh"] for event in events if event["kind"] == "charging_completed"
    ):
        raise ValueError(f"charging events disagree with energy metrics for {candidate_id}")
    for field, actual in (
        ("served", served), ("unserved", len(unserved)), ("travel_wh", metrics["travel_wh"]),
    ):
        if menu_entry[field] != actual or not _nonnegative(actual):
            raise ValueError(f"menu disagrees with result metric {field} for {candidate_id}")

    restricted, recovery = menu_entry["restricted_recovery_min"], metrics["recovery_95_min"]
    threshold = (95 * total + 99) // 100
    if (
        not _nonnegative(restricted)
        or (recovery is None) != (served < threshold)
        or (recovery is not None and recovery != restricted)
    ):
        raise ValueError(f"recovery metrics disagree for {candidate_id}")
    dispatch = alert.get("dispatch_min")
    if dispatch is not None:
        routing_started = [event["time_min"] for event in events if event["kind"] == "routing_started"]
        if routing_started != [dispatch]:
            raise ValueError(f"dispatch time disagrees with routing events for {candidate_id}")
    if "alert_min" in alert:
        if recovery is not None and recovery != deposits[threshold - 1]["time_min"] - alert["alert_min"]:
            raise ValueError(f"recovery does not match the logged threshold deposit for {candidate_id}")
        if recovery is None and "horizon_min" in alert:
            if restricted != alert["horizon_min"] - alert["alert_min"]:
                raise ValueError(f"restricted recovery does not match the configured horizon for {candidate_id}")
    return {
        **menu_entry, "total_parcels": total, "service_rate": metrics["service_rate"],
        "charged_wh": metrics["charged_wh"], "travel_min": metrics["travel_min"],
        "recovery_95_min": recovery, "deposited_parcel_ids": sorted(deposited_ids),
        "unserved_parcel_ids": unserved,
    }


def _selections(case: dict, candidates: dict) -> dict:
    selections = {}
    ranked = min(candidates.values(), key=lambda item: (
        -item["served"], item["travel_wh"], item["restricted_recovery_min"], item["id"],
    ))
    for arm in _ARM_NAMES:
        decision = case["selections"][arm]
        selected, status = decision["decision_id"], decision["status"]
        fallback, elapsed = decision["fallback_used"], decision["elapsed_ms"]
        expected_fallback = "continue_mappo" if status == "planning_timeout" else ranked["id"]
        if (
            selected not in candidates or type(fallback) is not bool or not _nonnegative(elapsed)
            or status not in {"selected", "invalid_proposal", "timeout", "backend_error", "planning_timeout"}
            or fallback != (status != "selected")
            or ((arm == _ARM_NAMES[0] or fallback) and selected != expected_fallback)
            or (status == "selected" and decision.get("deadline_exceeded", False))
            or (status == "selected" and decision.get("proposed_id", selected) != selected)
        ):
            raise ValueError(f"invalid recorded selection for {arm}")
        end_to_end = decision.get("end_to_end_decision_ms")
        if end_to_end is not None and (
            not _nonnegative(end_to_end)
            or end_to_end < elapsed
            or not isclose(end_to_end, case.get("preselection_planning_ms", end_to_end - elapsed) + elapsed)
        ):
            raise ValueError(f"invalid recorded end-to-end time for {arm}")
        selections[arm] = {
            "decision_id": selected, "status": status, "fallback_used": fallback,
            "elapsed_ms": elapsed, "budget_ms": decision.get("budget_ms"),
            "proposed_id": decision.get("proposed_id"), "end_to_end_decision_ms": end_to_end,
        }
    return selections


def _delta(focal: dict, comparison: dict) -> dict:
    return {
        field: focal[field] - comparison[field]
        for field in ("served", "travel_wh", "charged_wh", "restricted_recovery_min")
    }


def explain(run_dir: Path, case_id: str | None = None) -> dict:
    """Verify one saved case and return only evidence grounded in its trace."""
    case, source = _load_case(Path(run_dir).resolve(), case_id)
    try:
        alert, alerts, fallbacks = (case[key] for key in ("alert", "alerts", "fallbacks"))
        if (
            not isinstance(alert, dict) or not isinstance(alerts, list)
            or not isinstance(fallbacks, list)
            or any(not isinstance(event, dict) for event in (*alerts, *fallbacks))
        ):
            raise ValueError("alert and fallback events must use the trace schema")
        offers = _offers(case)
        records = case["candidates"]
        menu = {item["id"]: item for item in case["menu"]}
        if not menu or len(menu) != len(case["menu"]) or menu.keys() != records.keys():
            raise ValueError("candidate menu and saved alternatives disagree")
        candidates = {
            key: _candidate(entry, records[key], offers, alert) for key, entry in menu.items()
        }
        cohorts = {
            frozenset(result["unserved_parcel_ids"] + result["deposited_parcel_ids"])
            for result in candidates.values()
        }
        if len(cohorts) != 1:
            raise ValueError("candidate outcomes do not describe the same parcel cohort")
        receipt = case["repair_receipt"]
        if receipt["certified_metrics"] != records["route_repair"]["result"]["metrics"]:
            raise ValueError("repair receipt disagrees with its saved certified outcomes")
        scores = case["model_scores"]
        if not isinstance(scores, dict) or any(
            key not in menu or type(value) not in (int, float) or not isfinite(value)
            for key, value in scores.items()
        ):
            raise ValueError("model scores must be finite and refer to approved candidate IDs")
        selections = _selections(case, candidates)
        deterministic, agentic = (
            candidates[selections[arm]["decision_id"]] for arm in _ARM_NAMES
        )
        mappo = case.get("mappo_alone")
        if mappo is not None and (
            mappo["decision_id"] != "continue_mappo" or mappo["status"] != "selected"
            or not _nonnegative(mappo["end_to_end_decision_ms"])
        ):
            raise ValueError("invalid recorded standalone MAPPO selection")
        return {
            "source": source,
            "case": {key: case[key] for key in ("case_id", "traffic", "severity", "chaos_kind")},
            "alert": alert, "alerts": alerts, "offers": offers, "fallbacks": fallbacks,
            "repair": {key: receipt[key] for key in (
                "engine", "strategy", "solver_status", "fallback_reason", "columns_per_vehicle_limit",
                "column_capped_vehicles", "all_objectives_optimal_within_columns",
            ) if key in receipt},
            "candidates": candidates, "selections": selections,
            "model_scores": scores,
            "agentic_minus_deterministic": _delta(agentic, deterministic),
            "mappo_alone": mappo,
            "agentic_minus_mappo": None if mappo is None else _delta(agentic, candidates["continue_mappo"]),
            "scope": "Saved counterfactual route replays in one synthetic case.",
        }
    except (KeyError, TypeError, AttributeError, ZeroDivisionError) as error:
        raise ValueError("selected case does not match the Month 4 trace schema") from error


def _text(value) -> str:
    return re.sub(r"([\\`*_{}\[\]()|#])", r"\\\1", escape(str(value), quote=False)).replace("\n", " ")


def _alert_description(event: dict) -> str:
    kind, payload = event.get("kind", "recorded disruption"), event.get("payload", {})
    if kind == "locker_outage" and "locker_id" in payload:
        detail = f"Locker {_text(payload['locker_id'])} became unavailable."
    elif kind == "locker_saturation":
        changes = [
            f"{_text(row['locker_id'])} went from {_text(row['free_slots_before'])} to "
            f"{_text(row['free_slots_after'])} free slots" for row in payload["capacity_changes"]
        ]
        detail = ", ".join(changes) or "No free-slot change was recorded."
    elif kind == "charger_outage":
        detail = (
            f"Charging rate changed from {_text(payload['charger_rate_before_wh_per_min'])} to "
            f"{_text(payload['charger_rate_after_wh_per_min'])} Wh/min."
        )
    elif kind == "road_blockage":
        detours = [
            f"{_text(row['origin'])} to {_text(row['destination'])} via {_text(row['via'])} "
            f"(from {_text(row['travel_before_min'])} to {_text(row['travel_after_min'])} min)"
            for row in payload["replacements"]
        ]
        detail = ", ".join(detours) or "No directed detour was recorded."
    elif kind == "demand_surge":
        detail = (f"The onboard cohort rose from {_text(payload['onboard_before'])} to "
                  f"{_text(payload['onboard_after'])} parcels.")
    elif kind == "compound":
        detail = "Components were " + ", ".join(_text(x) for x in payload["components"]) + "."
    else:
        detail = "The event is preserved in the source trace."
    at = f" at minute {_text(event['time_min'])}" if "time_min" in event else ""
    return f"**{_text(kind)}**{at}. {detail}"


def _delta_description(delta: dict) -> str:
    return (
        f"**{delta['served']:+g}** on-time parcels, **{delta['travel_wh']:+g} Wh** traction, "
        f"**{delta['charged_wh']:+g} Wh** charged, and "
        f"**{delta['restricted_recovery_min']:+g} min** restricted recovery."
    )


def render_markdown(explanation: dict) -> str:
    """Render the same verified evidence as a readable, standalone case report."""
    case, alert, offers = (explanation[key] for key in ("case", "alert", "offers"))
    source = explanation["source"]
    lines = [
        f"# EcoAgent-Twin case {_text(case['case_id'])}", "",
        explanation["scope"], "",
        "## Alert and dispatch", "",
        f"Traffic **{_text(case['traffic'])}**, severity **{_text(case['severity'])}**, "
        f"disruption **{_text(case['chaos_kind'])}**.",
    ]
    if "alert_min" in alert and "dispatch_min" in alert:
        lines.append(f"Alert at minute {_text(alert['alert_min'])}. Routing began at minute "
                     f"{_text(alert['dispatch_min'])}.")
    for event in explanation["alerts"]:
        lines.extend(("", f"- {_alert_description(event)}"))

    lines.extend(("", "## Commitments and constraints", "",
                  f"{offers['issued']} offers issued, {len(offers['committed_slots'])} committed, "
                  f"{offers['declined']} declined, and {offers['expired']} expired. "
                  f"Paid **{offers['spent_cents']}** of **{offers['budget_cents']} cents**."))
    for item in offers["committed_slots"]:
        lines.extend(("", f"- Parcel {_text(item['parcel_id'])} held a slot at "
                      f"{_text(item['locker_id'])} at minute {item['time_min']}. "
                      f"The ledger paid {item['paid_cents']} cents."))
    for event in explanation["fallbacks"]:
        lines.extend(("", f"- Operational fallback for parcel {_text(event['parcel_id'])} "
                      f"was {_text(event['mode'])}, available at minute {_text(event['available_min'])}."))
    repair = explanation["repair"]
    if repair:
        lines.extend(("", f"Repair strategy **{_text(repair.get('strategy', 'recorded'))}** "
                      f"with solver status **{_text(repair.get('solver_status', 'unreported'))}**."))
        if repair.get("fallback_reason"):
            lines.append(f"Recorded reason for the repair fallback was {_text(repair['fallback_reason'])}.")
        if "columns_per_vehicle_limit" in repair:
            lines.append(f"The search allowed {_text(repair['columns_per_vehicle_limit'])} columns per vehicle.")

    lines.extend(("", "## Saved route alternatives", "",
                  "| Candidate | On-time parcels | Traction Wh | Charged Wh | Restricted recovery min |",
                  "| --- | ---: | ---: | ---: | ---: |"))
    for candidate in explanation["candidates"].values():
        lines.append(
            f"| {_text(candidate['label'])} | {candidate['served']}/{candidate['total_parcels']} "
            f"| {candidate['travel_wh']} | {candidate['charged_wh']} "
            f"| {candidate['restricted_recovery_min']} |"
        )
    lines.extend(("", "A restricted recovery value equals the observation window when the "
                  "95% service threshold is unobserved.", "", "## Selections and difference", ""))
    for arm, label in zip(_ARM_NAMES, _ARM_LABELS, strict=True):
        outcome = explanation["selections"][arm]
        suffix = ", deterministic fallback" if outcome["fallback_used"] else ""
        if outcome["status"] == "planning_timeout":
            suffix = ", MAPPO continuation fallback"
        elapsed = f"{outcome['elapsed_ms']:.2f} ms selection"
        if outcome["end_to_end_decision_ms"] is not None:
            elapsed += f", {outcome['end_to_end_decision_ms']:.2f} ms end to end"
        lines.append(f"- **{label}** selected {_text(outcome['decision_id'])} "
                     f"({_text(outcome['status'])}{suffix}, {elapsed}).")
        saved = explanation["candidates"][outcome["decision_id"]]
        committed_unserved = [
            _text(item["parcel_id"]) for item in offers["committed_slots"]
            if item["parcel_id"] in saved["unserved_parcel_ids"]
        ]
        if committed_unserved:
            lines.append("  Paid commitments without a deposit by the deadline were "
                         + ", ".join(committed_unserved) + ".")
    if (mappo := explanation["mappo_alone"]) is not None:
        lines.append(f"- **MAPPO alone** continued its policy ({mappo['end_to_end_decision_ms']:.2f} ms).")
    lines.extend(("", "Agentic minus deterministic in these saved routes was "
                  + _delta_description(explanation["agentic_minus_deterministic"])))
    if (mappo_delta := explanation["agentic_minus_mappo"]) is not None:
        lines.extend(("", "Agentic minus MAPPO alone was " + _delta_description(mappo_delta)))
    scores = explanation["model_scores"]
    if scores:
        visible_scores = ", ".join(
            f"{_text(key)} = {value:.4g}" for key, value in scores.items() if type(value) in (int, float)
        )
        if visible_scores:
            lines.extend(("", f"Recorded model menu scores were {visible_scores}."))
            if explanation["selections"]["agentic_supervisor"]["status"] == "timeout":
                lines.append("The score arrived after the acceptance deadline. The fallback controlled dispatch.")
    lines.extend(("", "## Evidence", "",
                  f"Verified trace {_text(source['trace_path'])}, line {source['trace_line']}.",
                  f"SHA-256 `{source['trace_sha256']}`.", ""))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="run directory containing the manifest and trace")
    parser.add_argument("--case", help="case ID (defaults to the first trace case)")
    parser.add_argument("--format", choices=("markdown", "json"))
    parser.add_argument("--output", type=Path, help="new output file (defaults to stdout)")
    args = parser.parse_args()
    output_format = args.format or (
        "json" if args.output and args.output.suffix.lower() == ".json" else "markdown"
    )
    try:
        evidence = explain(args.run_dir, args.case)
        content = (
            json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            if output_format == "json" else render_markdown(evidence)
        )
        if args.output is None:
            print(content, end="")
        else:
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
