"""Audit saved supervisory decisions using exact prompt and rule Shapley games."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
from itertools import takewhile
import json
from math import isclose, isfinite
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

from .m3_local_model import LocalMenuModel
from .m3_supervisor import Candidate, DecisionView, _freeze
from .m4_model_xai import attribute
from .m5_local_model import OrderInvariantMenuModel, _compact_prompt


TRACE_FILES = {"M4": "traces.jsonl", "M5": "traces.jsonl", "M6": "observations.jsonl"}
ARMS = ("deterministic_supervisor", "agentic_supervisor")
EMPIRICAL_METHODS = {"preserve_observed_order", "frozen_mappo", *ARMS}
MODEL_ERROR_ABS = 1e-4
LOGITS = "model_logit_margin"
RULE = "lexicographic_preference"
FIELDS = (("service", "served"), ("energy", "travel_wh"), ("recovery", "restricted_recovery_min"))
MODEL_DIR = Path("artifacts/models/smollm2-135m-instruct")


def _read_bound(run: Path, name: str, manifest: dict) -> tuple[bytes, str]:
    path = (run / name).resolve(strict=True)
    if not path.is_relative_to(run):
        raise ValueError(f"{name} is outside the trace directory")
    data = path.read_bytes()
    digest = sha256(data).hexdigest()
    if digest != manifest["sha256"][name]:
        raise ValueError(f"{name} fingerprint differs from the source manifest")
    return data, digest


def _source_cases(campaign: str, run: Path, limit: int | None) -> tuple[dict, list]:
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    manifest_path = (run / "manifest.json").resolve(strict=True)
    if not manifest_path.is_relative_to(run):
        raise ValueError("manifest is outside the trace directory")
    manifest_data = manifest_path.read_bytes()
    manifest = json.loads(manifest_data)
    name = TRACE_FILES[campaign]
    trace, trace_hash = _read_bound(run, name, manifest)
    rows = trace.splitlines()
    if not rows:
        raise ValueError("empty source trace")
    worlds = None
    if campaign == "M6":
        world_data, _ = _read_bound(run, "worlds.jsonl", manifest)
        worlds = world_data.splitlines()
        if len(worlds) != len(rows):
            raise ValueError("empirical observations and worlds have different counts")
    cases = []
    seen = set()
    manifest_hash = sha256(manifest_data).hexdigest()
    for number, raw in enumerate(rows, 1):
        row = json.loads(raw)
        if worlds is None:
            case_id = row["case_id"]
        else:
            case_id = json.loads(worlds[number - 1])["snapshot"]["id"]
            episodes = row["rows"]
            if (len(episodes) != len(EMPIRICAL_METHODS)
                    or {entry["method"] for entry in episodes} != EMPIRICAL_METHODS
                    or {entry["case_id"] for entry in episodes} != {case_id}):
                raise ValueError(f"unmatched empirical case ID at line {number}")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"invalid or duplicate case ID at line {number}")
        seen.add(case_id)
        if limit is None or number <= limit:
            source = {"trace_file": name, "trace_line": number, "trace_sha256": trace_hash,
                      "row_sha256": sha256(raw).hexdigest(), "manifest_sha256": manifest_hash}
            cases.append((case_id, source, row))
    return manifest, cases


class _CachedModel:
    """Reuse identical inputs to the same pinned scorer across coalitions."""

    def __init__(self, model: LocalMenuModel, campaign: str) -> None:
        self.model = model
        self.campaign = campaign
        self.cache: dict = {}
        self.last_scores: dict = {}

    def __call__(self, view, *, timeout_s: float) -> None:
        if self.campaign == "M4":
            key = json.dumps(view.to_dict(), sort_keys=True, ensure_ascii=True)
        else:
            first, second = sorted(view.candidates, key=lambda candidate: candidate.id)
            key = (first.id, second.id, _compact_prompt(view.alert, first, second),
                   _compact_prompt(view.alert, second, first))
        if key not in self.cache:
            self.model(view, timeout_s=timeout_s)
            self.cache[key] = self.model.last_scores.copy()
        self.last_scores = self.cache[key]


class _PrefixModel:
    """Batch distinct prompt suffixes after one same-case causal prefix prefill."""

    def __init__(self, model: LocalMenuModel, campaign: str) -> None:
        self.model = model
        self.campaign = campaign
        self.cache: dict = {}
        self.last_scores: dict = {}
        self.label_ids = [model._tokenizer.encode(letter, add_special_tokens=False) for letter in "AB"]
        if any(len(ids) != 1 for ids in self.label_ids) or self.label_ids[0] == self.label_ids[1]:
            raise ValueError("model labels must be distinct single tokens")
        self.label_ids = [ids[0] for ids in self.label_ids]

    def _prompts(self, view: DecisionView) -> tuple[tuple[str, str] | tuple[str], tuple[str, str]]:
        if self.campaign == "M4":
            visible = view.to_dict()
            entries = visible["candidates"]
            # Reproduce the original M3/M4 scorer template pinned by the run.
            prompt = "\n".join([
                "Choose one certified action after a parcel-locker outage.",
                "First maximize parcels served by deadline. If tied, minimize travel Wh, then recovery minutes.",
                f"Alert: {json.dumps(visible['alert'], ensure_ascii=True, sort_keys=True)}",
                *(f"{letter}: {json.dumps(entry, ensure_ascii=True, sort_keys=True)}"
                  for letter, entry in zip("AB", entries, strict=True)),
                "Answer with the single letter of the best action.",
            ])
            return (prompt,), (entries[0]["id"], entries[1]["id"])
        first, second = sorted(view.candidates, key=lambda item: item.id)
        return (_compact_prompt(view.alert, first, second), _compact_prompt(view.alert, second, first)), (
            first.id, second.id,
        )

    def _score(self, prompts: list[str]) -> None:
        torch, tokenizer, model = self.model._torch, self.model._tokenizer, self.model._model
        tokens = [tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
        ) for prompt in prompts]
        columns = zip(*tokens)
        common = sum(1 for _ in takewhile(lambda column: len(set(column)) == 1, columns))
        prefix = min(common, min(map(len, tokens)) - 1)
        suffixes = [row[prefix:] for row in tokens]
        last = torch.tensor([len(row) - 1 for row in suffixes])
        width = max(map(len, suffixes))
        padded = torch.tensor([row + [tokenizer.pad_token_id] * (width - len(row)) for row in suffixes])
        mask = torch.arange(prefix + width)[None, :] <= (last + prefix)[:, None]
        with torch.inference_mode():
            cache = None
            if prefix > 0:
                cache = model.model(torch.tensor([tokens[0][:prefix]]), use_cache=True).past_key_values
                cache.batch_repeat_interleave(len(tokens))
            hidden = model.model(padded, attention_mask=mask, past_key_values=cache,
                                 use_cache=True).last_hidden_state
            logits = model.lm_head(hidden[torch.arange(len(tokens)), last])[:, self.label_ids].tolist()
        self.cache.update(zip(prompts, logits, strict=True))

    def prepare(self, alert: dict, menu: tuple[Candidate, Candidate]) -> None:
        # Reuse the same neutral values and paired three-group interventions as attribute().
        cohort = menu[0].served + menu[0].unserved
        neutral = ({"served": 0, "unserved": cohort}, {"travel_wh": 0}, {"restricted_recovery_min": 0})
        frozen = _freeze(alert)
        needed = {}
        for mask in range(8):
            fields = {key: value for bit, group in enumerate(neutral) if not mask & (1 << bit)
                      for key, value in group.items()}
            view = DecisionView(frozen, tuple(replace(candidate, **fields) for candidate in menu))
            prompts, _ = self._prompts(view)
            needed.update((prompt, None) for prompt in prompts if prompt not in self.cache)
        if needed:
            self._score(list(needed))

    def __call__(self, view, *, timeout_s: float) -> None:
        prompts, (first, second) = self._prompts(view)
        a, b = self.cache[prompts[0]]
        if len(prompts) == 2:
            swap_a, swap_b = self.cache[prompts[1]]
            a, b = (a + swap_b) / 2, (b + swap_a) / 2
        self.last_scores = {first: a, second: b}


class _RuleScorer:
    """Express the rule's pairwise preference as winner 1, loser 0."""

    def __init__(self) -> None:
        self.last_scores: dict = {}

    def __call__(self, view, *, timeout_s: float) -> None:
        best = min(view.candidates, key=lambda item: (
            -item.served, item.travel_wh, item.restricted_recovery_min, item.id,
        ))
        self.last_scores = {candidate.id: int(candidate.id == best.id) for candidate in view.candidates}


class _ReceiptScorer:
    """Replay a stored coalition table for structural and arithmetic verification."""

    def __init__(self, attribution: dict) -> None:
        self.selected = attribution["selected_id"]
        self.alternative = attribution["alternative_id"]
        self.rows = iter(attribution["coalitions"])
        self.last_scores: dict = {}

    def __call__(self, view, *, timeout_s: float) -> None:
        row = next(self.rows, None)
        if row is None:
            raise ValueError("missing saved coalition scores")
        self.last_scores = {self.selected: row["selected_score"], self.alternative: row["alternative_score"]}


def _game(alert: dict, menu: tuple[Candidate, Candidate], selected: str, alternative: str,
          model, unit: str) -> dict:
    game = attribute(alert, [asdict(entry) for entry in menu], selected, alternative, model, timeout_s=60)
    del game["elapsed_ms"]
    game["unit"] = unit
    if unit == RULE:
        game["method"] = "exact Shapley attribution of lexicographic pairwise preference"
        game["score_definition"] = "selected candidate's binary win indicator minus the alternative's"
        game["interpretation"] = (
            "Three metric groups are restored together for both candidates. The stable ID tie-break "
            "remains in every coalition, including the neutral reference. Values explain a unitless "
            "change in binary rule preference and do not measure parcels, energy, time, or model logits."
        )
        game["full_score_verified"] = None
    return game


def _short_rule_reason(selected: Candidate, alternative: Candidate) -> tuple[str, str]:
    for label, field in FIELDS:
        selected_value = getattr(selected, field)
        alternative_value = getattr(alternative, field)
        if selected_value != alternative_value:
            messages = {
                "service": f"It served more parcels by their deadlines ({selected_value} versus {alternative_value}).",
                "energy": f"Service tied and it used less traction energy ({selected_value} versus {alternative_value} Wh).",
                "recovery": (f"Service and energy tied and it had lower restricted recovery "
                             f"({selected_value} versus {alternative_value} min)."),
            }
            return label, messages[label]
    return "candidate_id", "All three metrics tied, so the stable candidate ID broke the tie."


def _model_scores(record: dict, ids: set[str]) -> dict:
    scores = record["model_scores"]
    if set(scores) != ids or any(type(value) not in (int, float) or not isfinite(value) for value in scores.values()):
        raise ValueError("saved model scores are missing, nonfinite, or concern a different menu")
    return scores


def _verify_full_scores(game: dict, archive: dict, selected: str, alternative: str) -> None:
    full = game["coalitions"][-1]
    if not all(isclose(a, b, abs_tol=MODEL_ERROR_ABS, rel_tol=0) for a, b in (
        (full["selected_score"], archive[selected]), (full["alternative_score"], archive[alternative]),
        (game["full_margin"], archive[selected] - archive[alternative]),
    )):
        raise ValueError("recomputed full coalition disagrees with archived model logits")
    game["archived_scores"] = archive
    game["full_score_verified"] = True
    game["full_score_abs_tolerance"] = MODEL_ERROR_ABS


def _describe_order_balanced(game: dict) -> None:
    game["score_definition"] = (
        "selected candidate's mean next-token logit across both positions minus the alternative's mean"
    )
    game["interpretation"] = (
        "Both candidate-label orders are scored and averaged within every coalition. Attributions depend on "
        "the alert, pinned model, compact prompt and declared neutral values. The interventions change prompt "
        "metrics for attribution while leaving the recorded route and offer outcomes as executed."
    )


def _case_receipt(campaign: str, case: tuple, model=None, stored: dict | None = None) -> dict:
    case_id, source, trace = case
    menu = tuple(Candidate(**entry) for entry in trace["menu"])
    by_id = {entry.id: entry for entry in menu}
    if len(menu) != 2 or len(by_id) != 2:
        raise ValueError("a critical supervisor decision must contain two distinct certified candidates")
    decisions = {}
    for arm in ARMS:
        decision = trace["selections"][arm]
        selected, status = decision["decision_id"], decision["status"]
        if selected not in by_id or status not in (
            "selected", "timeout", "invalid_proposal", "backend_error", "planning_timeout",
        ) or decision["fallback_used"] is not (status != "selected"):
            raise ValueError(f"invalid recorded decision for {case_id}/{arm}")
        alternative = next(key for key in by_id if key != selected)
        outcome = {"selected_id": selected, "alternative_id": alternative, "status": status,
                   "fallback_used": decision["fallback_used"], "archived_reason": decision["reason"]}
        if status == "planning_timeout":
            if selected != "continue_mappo":
                raise ValueError("planning fallback did not continue MAPPO")
            outcome.update(attribution=None, decisive_factor="planning_deadline", explanation=(
                "Planning exceeded the shared decision budget. The supervisor continued with the certified MAPPO plan."
            ))
        elif arm == "agentic_supervisor" and status == "selected":
            archive = _model_scores(trace, set(by_id))
            if campaign == "M4":
                expected = max(menu, key=lambda item: archive[item.id]).id
            else:
                expected = min(by_id, key=lambda key: (-archive[key], key))
            if expected != selected:
                raise ValueError("archived scores do not select the recorded intervention")
            scorer = model
            if stored is not None:
                scorer = _ReceiptScorer(stored["decisions"][arm]["attribution"])
            if scorer is None:
                raise ValueError("accepted model decisions require a real scorer or a saved coalition receipt")
            if isinstance(scorer, _PrefixModel):
                scorer.prepare(trace["alert"], menu)
            game = _game(trace["alert"], menu, selected, alternative, scorer, LOGITS)
            _verify_full_scores(game, archive, selected, alternative)
            if campaign != "M4":
                _describe_order_balanced(game)
            method = "the saved prompt order" if campaign == "M4" else "both candidate positions"
            outcome.update(attribution=game, decisive_factor="model_score", explanation=(
                f"The model selected {selected} using next-token scores from {method}. The bars show "
                "how the visible prompt metrics changed its selected-minus-alternative logit margin."
            ))
        else:
            if arm == "deterministic_supervisor" and status != "selected":
                raise ValueError("deterministic choices may only select or hit the shared planning deadline")
            game = _game(trace["alert"], menu, selected, alternative, _RuleScorer(), RULE)
            if game["full_margin"] != 1:
                raise ValueError("the full rule coalition does not prefer the recorded decision")
            reason, narrative = _short_rule_reason(by_id[selected], by_id[alternative])
            prefix = "The recorded model proposal did not control execution. " if status != "selected" else ""
            outcome.update(attribution=game, decisive_factor=reason, explanation=(
                prefix + f"The deterministic rule selected {selected}. " + narrative
            ))
        decisions[arm] = outcome
    return {"schema": 1, "campaign": campaign, "case_id": case_id, "source": source, "decisions": decisions}


def _count(counts: Counter, receipt: dict) -> None:
    counts["cases"] += 1
    for decision in receipt["decisions"].values():
        game = decision["attribution"]
        counts[game["unit"] if game else "planning_deadline"] += 1


def _check_scorer_sources(campaign: str, manifest: dict) -> None:
    files = {"m3_supervisor.py", "m3_local_model.py", "m5_local_model.py"}
    recorded = manifest["source_sha256"]
    expected = "m3_local_model.py" if campaign == "M4" else "m5_local_model.py"
    if expected not in recorded:
        raise ValueError("archive does not identify the model scorer implementation")
    for name in files.intersection(recorded):
        digest = sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        if digest != recorded[name]:
            raise ValueError(f"local scorer source differs from the archive: {name}")


def _validated_prefix(campaign: str, cases: list, path: Path | None) -> list[dict]:
    if path is None:
        return []
    with path.open("rb") as stream:
        rows = []
        for line in stream:
            if not line.endswith(b"\n"):
                break  # An interrupted writer may have left an incomplete last line.
            if len(rows) >= len(cases):
                raise ValueError("resume input exceeds the requested case set")
            saved = json.loads(line)
            reproduced = _case_receipt(campaign, cases[len(rows)], stored=saved)
            if reproduced != saved:
                raise ValueError(f"resume receipt differs for {reproduced['case_id']}")
            rows.append(saved)
        return rows


def run(campaign: str, run_dir: Path, output: Path, *, model_dir: Path = MODEL_DIR,
        limit: int | None = None, threads: int = 6, prefix_batch: bool = False,
        resume: Path | None = None) -> dict:
    run_dir = run_dir.resolve(strict=True)
    output = output.resolve()
    if output.exists() or output.is_relative_to(run_dir):
        raise ValueError("the output must be a new file outside the archived run directory")
    if type(threads) is not int or threads <= 0:
        raise ValueError("threads must be a positive integer")
    manifest, cases = _source_cases(campaign, run_dir, limit)
    _check_scorer_sources(campaign, manifest)
    earlier = _validated_prefix(campaign, cases, resume)
    model_class = LocalMenuModel if campaign == "M4" else OrderInvariantMenuModel
    model = model_class(model_dir, threads=threads)
    if model.provenance != manifest["model"]:
        raise ValueError("local model provenance differs from the archived model")
    scorer = _PrefixModel(model, campaign) if prefix_batch else _CachedModel(model, campaign)
    counts: Counter = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for index, case in enumerate(cases):
                receipt = earlier[index] if index < len(earlier) else _case_receipt(campaign, case, scorer)
                stream.write(json.dumps(receipt, ensure_ascii=True, allow_nan=False) + "\n")
                _count(counts, receipt)
                if counts["cases"] % 32 == 0:
                    print(f"{campaign} {counts['cases']}/{len(cases)} cases", file=sys.stderr, flush=True)
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return dict(counts)


def verify(campaign: str, run_dir: Path, output: Path, *, limit: int | None = None) -> dict:
    """Verify bound traces, stored score parity, selection pathways, and Shapley arithmetic."""
    _, cases = _source_cases(campaign, run_dir.resolve(strict=True), limit)
    counts: Counter = Counter()
    with output.open(encoding="utf-8") as stream:
        for case, line in zip(cases, stream, strict=True):
            saved = json.loads(line)
            regenerated = _case_receipt(campaign, case, stored=saved)
            if regenerated != saved:
                raise ValueError(f"saved attribution receipt differs for {case[0]}")
            _count(counts, regenerated)
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "verify"):
        command = commands.add_parser(name)
        command.add_argument("campaign", choices=TRACE_FILES)
        command.add_argument("run_dir", type=Path)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--limit", type=int, help="limit cases for an explicitly partial smoke test")
        if name == "run":
            command.add_argument("--model", type=Path, default=MODEL_DIR)
            command.add_argument("--threads", type=int, default=6)
            command.add_argument("--prefix-batch", action="store_true",
                                 help="batch shared token prefixes and check full scores against the archive")
            command.add_argument("--resume", type=Path, help="validate and reuse a prefix of existing receipts")
    args = parser.parse_args()
    try:
        if args.command == "run":
            counts = run(args.campaign, args.run_dir, args.output, model_dir=args.model,
                         limit=args.limit, threads=args.threads, prefix_batch=args.prefix_batch, resume=args.resume)
        else:
            counts = verify(args.campaign, args.run_dir, args.output, limit=args.limit)
    except (ImportError, OSError, RuntimeError, TimeoutError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    print(json.dumps({"output": str(args.output.resolve()), **counts}, sort_keys=True))


if __name__ == "__main__":
    main()
