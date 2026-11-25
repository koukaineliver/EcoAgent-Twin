# EcoAgent-Twin

EcoAgent-Twin evaluates routing recovery after parcel-locker disruption. A shared
event simulator certifies frozen MAPPO continuation and OR-Tools repair. The
comparison evaluates MAPPO alone, a deterministic supervisor, and an order-balanced
local language-model supervisor under identical customer commitments.

## Quick start

Use Python 3.11 from the extracted directory. The first example needs only the
standard library. Commands create new run directories.

    python -B -m ecoagent_twin run
    python -B -m unittest discover -s tests -p test_outage_replay.py

## Reproduce the synthetic experiment

Install CPU PyTorch matching the platform, requirements-m3.txt in the evaluation
environment and requirements-m2-sumo.txt in a separate SUMO environment. Use the
SUMO environment for corpus preparation and the evaluation environment for the
remaining commands. Supply the new run paths printed by each command.

    python -B -m ecoagent_twin.m3_cases prepare
    python -B -m ecoagent_twin.m3_experiment train <m3-corpus>
    python -B -m ecoagent_twin.m3_model_assets
    python -B -m ecoagent_twin.m4_corpus prepare --protocol ecoagent_twin/scenarios/m5_campaign.json
    python -B -m ecoagent_twin.m5_experiment run <m5-corpus> <m3-training>
    python -B -m ecoagent_twin.m5_experiment verify <m5-corpus> <m5-evaluation>

The Hugging Face adapter fetches and validates one pinned public model revision.
The table and comparison records under evidence/ report the earlier 576-case
synthetic campaign. Reproduction records new checkpoints and latency on the
running machine. The synthetic road network is distinct from field conditions.

## Source distribution

See LICENSE for the candidate source-code license and CITATION.cff for citation.
The archive embeds synthetic results and source modules. External data, software
dependencies and fetched model assets carry their respective terms. See
THIRD_PARTY.md.

## Route-based validation and additional software

Build the frozen M3 policy and fetch the local model using the commands above.
The Amazon module independently obtains records from the official evaluation
corpus under its published noncommercial terms. Its route adaptation uses
modeled EV, locker, customer, and emissions inputs.

    python -B -m ecoagent_twin.empirical_amazon --output <amazon-data> --count 40
    python -B -m ecoagent_twin.m6_empirical_benchmark run <amazon-data> <m3-training>/policy.pt --output <amazon-evaluation>
    python -B -m ecoagent_twin.m6_empirical_benchmark verify <amazon-data> <amazon-evaluation>
    python -B -m ecoagent_twin.m6_dynamic_grid download <ciso-workbook>
    python -B -m ecoagent_twin.m6_dynamic_grid run <amazon-evaluation> <ciso-workbook> <hourly-sensitivity>

The m6_gnn and m6_predictive_routing commands take a compatible archived demand
table and directed OSRM table. See their --help arguments. The archive supplies
the code while the corresponding data remain under their source terms. The
m6_incentive_experiment and m6_solver_experiment commands record and replay an
adaptive offer comparison and a common-certification comparison of CP-SAT with
independent Python search.

## Decision explanations and recipient replies

Explain decisions from a completed case run using its archived menus. The audit
checks the full model scores against the trace and uses rule preference for an
executed rule or fallback. M4 uses the original menu order. M5 and M6 score both
orders. Use the same model revision as the saved run.

    python -B -m ecoagent_twin.m6_decision_xai run M5 <m5-evaluation> --output <audit.jsonl> --prefix-batch
    python -B -m ecoagent_twin.m6_decision_xai verify M5 <m5-evaluation> --output <audit.jsonl>

In a fictional local negotiation, the operator can accept, decline, or skip
each offer. A timely decline may produce one revision. The ledger certifies the
remaining budget and available locker space before committing a reply and gives
the resulting assignments to the outage simulator.

    python -B -m ecoagent_twin.m6_recipient_dialogue demo <dialogue.json>
    python -B -m ecoagent_twin.m6_recipient_dialogue verify <dialogue.json>

## Charging and runtime tools

The hub-charging experiment compares certified schedules with and without an
eligible locker charger. The carbon objective uses a week-lagged EIA intensity
proxy before execution. The replay evaluates charging and terminal battery
replenishment against the realized hourly rates.

    python -B -m ecoagent_twin.m6_hub_charging_experiment run <amazon-evaluation> <m5-evaluation> <ciso-workbook> <hub-experiment.json>
    python -B -m ecoagent_twin.m6_hub_charging_experiment verify <amazon-evaluation> <m5-evaluation> <ciso-workbook> <hub-experiment.json>

The separate runtime smoke test queries SUMO through TraCI, fetches advisory
weather for a fixed public Los Angeles point, and certifies the regenerated
route candidates before selection. Use the SUMO environment's Python path for
TraCI. The weather request sends no study inputs.

    python -B -m ecoagent_twin.m6_context_tools run <m5-evaluation> <m5-corpus> --sumo-python <sumo-python> --model <local-model> --output <context-receipt.json>

## Los Angeles street network and controller tests

`runs/osmnx_la_university_park/` contains a pinned University Park OpenStreetMap
snapshot, two routed OSMnx tables, and the acquisition receipt. Consult the
data license before redistributing the road extracts. OSM way direction and
motor vehicle access are modeled. Time and energy are modeled on the same road
paths. The USC locker is mapped. Parcel workloads, remaining curb sites,
traffic profiles, and recipient decisions were designed for comparison.

    pip install -r requirements-la-roads.txt
    python -B -m ecoagent_twin.la_osmnx verify
    python -B -m ecoagent_twin.la_policy_training m1 runs/osmnx_la_university_park/road_network.json runs/la-m1-new
    python -B -m ecoagent_twin.la_controller_experiment run runs/osmnx_la_university_park/road_network_scale.json runs/la-new <m3-training> --layout distinct
    python -B -m ecoagent_twin.la_controller_experiment verify runs/osmnx_la_university_park/road_network_scale.json runs/la-new
    python -B -m ecoagent_twin.m6_incentive_experiment run runs/osmnx_la_university_park/road_network.json --osmnx --output runs/la-incentive-new

The aggregate University Park results and comparative traces under `evidence/`
share cases within each paired analysis. The road graph contains OpenStreetMap
data. The accompanying license file covers those database artifacts.
