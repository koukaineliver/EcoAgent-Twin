"""Export an offline interactive case explorer from a saved Month 4 run."""

from __future__ import annotations

import argparse
import json
from math import fsum, isclose, isfinite
from pathlib import Path

from .m4_xai import explain, render_markdown


def _finite(value) -> bool:
    return type(value) in (int, float) and isfinite(value)


def _validate_attribution(receipt: dict, evidence: dict) -> dict:
    try:
        attribution = receipt["attribution"]
        values = attribution["shapley_values"]
        baseline, full = attribution["baseline_margin"], attribution["full_margin"]
        order_check = receipt["reversed_menu"]
        ids = set(evidence["candidates"])
        recorded = evidence["selections"]["agentic_supervisor"]["decision_id"]
        if (
            receipt["case_id"] != evidence["case"]["case_id"]
            or receipt["source"]["trace_sha256"] != evidence["source"]["trace_sha256"]
            or not isinstance(values, dict) or not values
            or not all(_finite(value) for value in (*values.values(), baseline, full))
            or not isclose(fsum(values.values()), full - baseline, abs_tol=1e-8)
            or order_check["original"]["decision_id"] != recorded
        ):
            raise ValueError("attribution receipt does not match the selected run or reconstruct its margin")
        for label in ("original", "reversed"):
            item = order_check[label]
            if (
                item["decision_id"] not in ids or not isinstance(item["scores"], dict)
                or item["scores"].keys() != ids or not all(_finite(x) for x in item["scores"].values())
            ):
                raise ValueError("menu-order audit does not match the approved candidate IDs")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("attribution receipt has an invalid schema") from error
    return {
        "shapley_values": values, "baseline_margin": baseline, "full_margin": full,
        "reversed_menu": order_check,
    }


def export_viewer(run_dir: Path, output: Path, attributions: tuple[Path, ...] = ()) -> int:
    """Create a self-contained report at a new path and return its case count."""
    run_dir = run_dir.resolve(strict=True)
    with (run_dir / "traces.jsonl").open(encoding="utf-8") as stream:
        case_ids = [json.loads(line)["case_id"] for line in stream]
    if not case_ids:
        raise ValueError("run contains no cases")
    cases = {}
    for case_id in case_ids:
        evidence = explain(run_dir, case_id)
        cases[case_id] = {"evidence": evidence, "report": render_markdown(evidence)}
    for path in attributions:
        receipt = json.loads(path.read_bytes())
        case_id = receipt.get("case_id") if isinstance(receipt, dict) else None
        if not isinstance(case_id, str) or case_id not in cases:
            raise ValueError(f"attribution does not identify a case in this run: {path}")
        case = cases[case_id]
        if "attribution" in case:
            raise ValueError(f"more than one attribution receipt for case {case_id}")
        case["attribution"] = _validate_attribution(receipt, case["evidence"])

    # Keep every embedded value inside the data script, even if a label contains </script>.
    payload = (json.dumps(cases, ensure_ascii=True, allow_nan=False)
               .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_PAGE.replace("__CASE_DATA__", payload, 1))
    return len(cases)


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy"
 content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>EcoAgent-Twin case explorer</title>
<style>
:root { color-scheme: light; --paper: #f5f3ed; --panel: #fffdf8; --ink: #172824; --muted: #435951;
        --border: #cbd4ca; --teal: #096558; --red: #a64536; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink); font: 1rem/1.55 system-ui, sans-serif; }
main { max-width: 68rem; padding: 2rem 1rem 4rem; margin: auto; }
h1, h2 { line-height: 1.15; margin: 0 0 .6rem; }
h1 { font-size: clamp(1.9rem, 5vw, 3rem); letter-spacing: -.035em; }
h2 { font-size: 1.25rem; }
p { margin: .4rem 0; }
.muted { color: var(--muted); }
.case-picker, section, details { background: var(--panel); border: 1px solid var(--border);
                                border-radius: .85rem; padding: 1rem 1.15rem; }
.case-picker { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: .4rem 1rem;
               margin: 1.4rem 0 1rem; align-items: end; }
label { display: block; font-weight: 700; margin-bottom: .35rem; }
select { font: inherit; color: inherit; width: 100%; background: white; border: 1px solid #82978c;
         border-radius: .45rem; padding: .55rem; }
:focus-visible { outline: 3px solid var(--teal); outline-offset: 3px; }
.count { font-size: .9rem; color: var(--muted); padding: 0 0 .5rem; white-space: nowrap; }
.tags { display: flex; gap: .4rem; flex-wrap: wrap; margin: .7rem 0 1rem; }
.tags span { background: #e6efe8; border-radius: 999px; padding: .15rem .6rem; font-size: .9rem; }
.panel-grid { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); gap: 1rem; }
.scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
caption { text-align: left; color: var(--muted); }
th, td { padding: .55rem .45rem; border-bottom: 1px solid var(--border); text-align: left; }
th { font-weight: 650; }
td.number, th.number { text-align: right; font-variant-numeric: tabular-nums; }
td.number { white-space: nowrap; }
ul { padding-left: 1.2rem; margin: .5rem 0 0; }
li + li { margin-top: .35rem; }
.attribution, details { margin-top: 1rem; }
.bar-list { list-style: none; padding: 0; }
.bar-list li { display: grid; grid-template-columns: minmax(7rem, 1fr) minmax(4rem, 2fr) 5.5rem;
               align-items: center; gap: .6rem; }
.track { height: .8rem; position: relative; background: #edf0e9; border-radius: .25rem; }
.track::after { content: ""; position: absolute; width: 1px; inset: 0 auto 0 50%; background: #50645c; }
.bar { position: absolute; height: 100%; top: 0; background: var(--teal); }
.bar.negative { background: var(--red); }
.bar-number { font-variant-numeric: tabular-nums; text-align: right; }
summary { cursor: pointer; font-weight: 700; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; font: .85rem/1.55 ui-monospace, monospace; }
@media (max-width: 55rem) { .panel-grid { grid-template-columns: minmax(0, 1fr); } }
@media (max-width: 30rem) { main { padding-top: 1.25rem; } .case-picker { grid-template-columns: 1fr; }
                          .count { padding: 0; } .bar-list li { gap: .35rem; } }
</style>
</head>
<body>
<main>
  <h1>EcoAgent-Twin case explorer</h1>
  <p class="muted">Compare saved routing decisions from the same synthetic outage snapshot.</p>
  <div class="case-picker">
    <div><label for="case">Case</label><select id="case" aria-describedby="case-count"></select></div>
    <span id="case-count" class="count"></span>
  </div>
  <p id="case-status" aria-live="polite"></p>
  <div class="tags" id="tags"></div>
  <div class="panel-grid">
    <section aria-labelledby="outcome-title">
      <h2 id="outcome-title">Routing outcomes</h2>
      <div class="scroll">
        <table>
          <caption>Restricted recovery uses the observation window when 95% service was unobserved.</caption>
          <thead><tr><th scope="col">Arm and intervention</th><th scope="col" class="number">Served</th>
            <th scope="col" class="number">Traction Wh</th><th scope="col" class="number">Recovery min</th>
          </tr></thead>
          <tbody id="outcomes"></tbody>
        </table>
      </div>
      <p class="muted" id="difference"></p>
    </section>
    <section aria-labelledby="offers-title">
      <h2 id="offers-title">Offers and fallback</h2>
      <p id="offer-total"></p>
      <ul id="offer-lines"></ul>
    </section>
  </div>
  <section class="attribution" id="attribution" aria-labelledby="attribution-title" hidden>
    <h2 id="attribution-title">Model-score attribution</h2>
    <p class="muted">Shapley contributions to the local model’s score margin for the saved candidate pair.</p>
    <p id="margin"></p>
    <ul class="bar-list" id="bars"></ul>
    <p id="order-check"></p>
  </section>
  <details>
    <summary>Verified case report and trace fingerprint</summary>
    <pre id="report"></pre>
  </details>
  <noscript>Enable JavaScript to select a saved case in this offline report.</noscript>
</main>
<script id="case-data" type="application/json">__CASE_DATA__</script>
<script>
"use strict";
const byId = (id) => document.getElementById(id);
const cases = JSON.parse(byId("case-data").textContent);
const select = byId("case");
for (const [id, item] of Object.entries(cases)) {
    select.add(new Option(item.attribution ? id + " (attribution)" : id, id));
}
byId("case-count").textContent = select.options.length + " verified cases";

function element(tag, text, className = "") {
    const node = document.createElement(tag);
    node.textContent = text;
    if (className) node.className = className;
    return node;
}

const readable = (value) => String(value).replaceAll("_", " ");
const signed = (value) => (value >= 0 ? "+" : "") + value.toPrecision(4);

function loadAttribution(receipt) {
    byId("attribution").hidden = !receipt;
    const bars = byId("bars");
    bars.replaceChildren();
    if (!receipt) return;
    const entries = Object.entries(receipt.shapley_values);
    const scale = Math.max(1e-9, ...entries.map(([, value]) => Math.abs(value)));
    byId("margin").textContent = "Baseline margin " + signed(receipt.baseline_margin) +
        " · Full margin " + signed(receipt.full_margin);
    for (const [label, value] of entries) {
        const row = document.createElement("li");
        const track = element("span", "", "track");
        track.setAttribute("aria-hidden", "true");
        const fill = element("span", "", value < 0 ? "bar negative" : "bar");
        const width = Math.abs(value) / scale * 50;
        fill.style.width = width + "%";
        fill.style.left = (value < 0 ? 50 - width : 50) + "%";
        track.append(fill);
        row.append(element("span", readable(label)), track, element("span", signed(value), "bar-number"));
        bars.append(row);
    }
    const audit = receipt.reversed_menu;
    const scores = (order) => Object.entries(order.scores)
        .map(([id, value]) => readable(id) + " " + signed(value)).join(", ");
    byId("order-check").textContent = "Menu-order check chose " + readable(audit.original.decision_id) +
        " in the saved order (" + scores(audit.original) + ") and " +
        readable(audit.reversed.decision_id) + " when reversed (" + scores(audit.reversed) + ").";
}

function update() {
    const item = cases[select.value];
    const e = item.evidence;
    const meta = e.case;
    byId("case-status").textContent = "Alert at minute " + e.alert.alert_min +
        " · Routing at minute " + e.alert.dispatch_min;
    byId("tags").replaceChildren(...[meta.traffic, meta.severity, meta.chaos_kind]
        .map((value) => element("span", readable(value))));
    const arms = [
        ["MAPPO alone", e.mappo_alone],
        ["Deterministic", e.selections.deterministic_supervisor],
        ["Agentic", e.selections.agentic_supervisor],
    ];
    const rows = arms.filter(([, decision]) => decision).map(([name, decision]) => {
        const result = e.candidates[decision.decision_id];
        const row = document.createElement("tr");
        const label = element("th", name + " · " + result.label +
            (decision.fallback_used ? " (fallback)" : ""));
        label.scope = "row";
        row.append(label, element("td", result.served + "/" + result.total_parcels, "number"),
            element("td", result.travel_wh, "number"),
            element("td", result.restricted_recovery_min, "number"));
        return row;
    });
    byId("outcomes").replaceChildren(...rows);
    const delta = e.agentic_minus_mappo;
    byId("difference").textContent = delta ? "Agentic minus MAPPO alone · " +
        (delta.served >= 0 ? "+" : "") + delta.served + " parcels · " +
        (delta.travel_wh >= 0 ? "+" : "") + delta.travel_wh + " traction Wh" : "";

    const offers = e.offers;
    byId("offer-total").textContent = offers.issued + " issued · " + offers.committed_slots.length +
        " committed · " + offers.spent_cents + " of " + offers.budget_cents + " cents paid";
    const lines = offers.committed_slots.map((slot) => element("li",
        slot.parcel_id + " held " + slot.locker_id + " · " + slot.paid_cents + " cents paid"));
    for (const fallback of e.fallbacks) {
        lines.push(element("li", fallback.parcel_id + " · " + readable(fallback.mode) +
            " at minute " + fallback.available_min));
    }
    byId("offer-lines").replaceChildren(...lines);
    byId("report").textContent = item.report;
    loadAttribution(item.attribution);
}

select.addEventListener("change", update);
update();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attribution", type=Path, action="append", default=[])
    args = parser.parse_args()
    try:
        count = export_viewer(args.run_dir, args.output, tuple(args.attribution))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(f"Saved {count} verified cases to {args.output.resolve()}")


if __name__ == "__main__":
    main()
