#!/usr/bin/env python3
"""Offline routing regression benchmark. Does not measure answer quality or spend."""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gateway_policy import GatewayPolicy


def _load_cases(cases_path: Path) -> list[dict]:
    cases = [json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
    if not cases:
        raise ValueError("Benchmark contains no cases")
    return cases


def _prepare_config(
    base_config: dict,
    *,
    use_local: bool | None = None,
    min_confidence: float | None = None,
    min_margin: float | None = None,
) -> dict:
    config = copy.deepcopy(base_config)
    for provider in config["providers"].values():
        if provider.get("transport") == "vertex":
            provider.setdefault("project", "offline-benchmark")
            provider.setdefault("region", "global")
            provider.setdefault("base_url", "https://aiplatform.googleapis.com")
    clf = dict(config.get("classifier") or {})
    if use_local is not None:
        clf["use_local"] = use_local
    if min_confidence is not None:
        clf["min_confidence"] = float(min_confidence)
    if min_margin is not None:
        clf["min_margin"] = float(min_margin)
    if clf:
        config["classifier"] = clf
    return config


def _macro_f1(expected: list[str], actual: list[str], profiles: tuple[str, ...]) -> tuple[float, dict]:
    per_profile: dict[str, dict] = {}
    f1s: list[float] = []
    for prof in sorted(profiles):
        tp = sum(1 for e, a in zip(expected, actual) if e == prof and a == prof)
        fp = sum(1 for e, a in zip(expected, actual) if e != prof and a == prof)
        fn = sum(1 for e, a in zip(expected, actual) if e == prof and a != prof)
        support = tp + fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / support if support > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_profile[prof] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": support}
        if support > 0:
            f1s.append(f1)
    return round(sum(f1s) / max(1, len(f1s)), 4), per_profile


def _evaluate_policy(config: dict, cases: list[dict]) -> tuple[dict, dict[str, str]]:
    policy = GatewayPolicy(config)
    failures = []
    sources: Counter = Counter()
    expected_list: list[str] = []
    actual_list: list[str] = []
    pred_by_id: dict[str, str] = {}
    start = time.perf_counter()

    for case in cases:
        if case["profile"] not in policy.profiles:
            raise ValueError("Unknown expected profile")
        result = policy.classify({"model": policy.alias, "messages": [{"role": "user", "content": case["task"]}]}, {}, "offline")
        expected_list.append(case["profile"])
        actual_list.append(result.profile)
        pred_by_id[str(case["id"])] = result.profile
        sources[result.explanation["source"]] += 1
        if result.profile != case["profile"]:
            failures.append({"id": case["id"], "expected": case["profile"], "actual": result.profile})

    correct = sum(1 for e, a in zip(expected_list, actual_list) if e == a)
    macro_f1, per_profile = _macro_f1(expected_list, actual_list, tuple(policy.profiles.keys()))
    report = {
        "mode": "offline_synthetic_routing",
        "cases": len(cases),
        "correct": correct,
        "profile_accuracy": correct / len(cases),
        "macro_f1": macro_f1,
        "per_profile": per_profile,
        "sources": dict(sources),
        "failures": failures,
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
        "answer_quality_measured": False,
        "cost_savings_measured": False,
    }
    return report, pred_by_id


def _activate_onnx_dir(onnx_dir: Path | None) -> None:
    if onnx_dir is None:
        return
    os.environ["NEXUS_ROUTER_LOCAL_CLASSIFIER_DIR"] = str(onnx_dir)
    from src import gateway_classifier, local_classifier

    local_classifier._LOCAL_CLASSIFIER = None
    gateway_classifier._local_model.cache_clear()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "src/gateway.example.yaml")
    parser.add_argument("--cases", type=Path, default=ROOT / "tests/fixtures/gateway_tasks.jsonl")
    parser.add_argument("--outcomes", type=Path, help="Compare paired caller-evaluated baseline/routed JSONL outcomes")
    parser.add_argument("--compare-local", action="store_true", help="Compare use_local=false vs use_local=true on --cases")
    parser.add_argument("--min-confidence", type=float, default=None, help="Override classifier.min_confidence")
    parser.add_argument("--min-margin", type=float, default=None, help="Override classifier.min_margin")
    parser.add_argument("--onnx-dir", type=Path, default=None, help="Override NEXUS_ROUTER_LOCAL_CLASSIFIER_DIR")
    parser.add_argument("--strict", action="store_true", help="In --compare-local mode, exit non-zero if local fails any case")
    args = parser.parse_args(argv)
    if args.outcomes:
        from src.gateway_benchmark import compare_outcomes

        records = [json.loads(line) for line in args.outcomes.read_text().splitlines() if line.strip()]
        print(json.dumps(compare_outcomes(records), indent=2))
        return 0

    _activate_onnx_dir(args.onnx_dir)
    base_config = yaml.safe_load(args.config.read_text())
    cases = _load_cases(args.cases)

    if args.compare_local:
        rules_cfg = _prepare_config(base_config, use_local=False, min_confidence=args.min_confidence, min_margin=args.min_margin)
        local_cfg = _prepare_config(base_config, use_local=True, min_confidence=args.min_confidence, min_margin=args.min_margin)
        rules_report, rules_preds = _evaluate_policy(rules_cfg, cases)
        local_report, local_preds = _evaluate_policy(local_cfg, cases)
        disagreements = [
            {
                "id": case["id"],
                "expected": case["profile"],
                "rules": rules_preds[str(case["id"])],
                "with_local": local_preds[str(case["id"])],
                "task_preview": str(case["task"])[:120],
            }
            for case in cases
            if rules_preds[str(case["id"])] != local_preds[str(case["id"])]
            or local_preds[str(case["id"])] != case["profile"]
        ][:15]
        comparison = {
            "mode": "offline_gateway_comparison",
            "cases": len(cases),
            "rules_only": rules_report,
            "with_local_model": local_report,
            "accuracy_delta": round(local_report["profile_accuracy"] - rules_report["profile_accuracy"], 4),
            "macro_f1_delta": round(local_report["macro_f1"] - rules_report["macro_f1"], 4),
            "disagreements": disagreements,
        }
        print(json.dumps(comparison, indent=2))
        return int(bool(args.strict and local_report["failures"]))

    single_cfg = _prepare_config(base_config, min_confidence=args.min_confidence, min_margin=args.min_margin)
    report, _preds = _evaluate_policy(single_cfg, cases)
    print(json.dumps(report, indent=2))
    return int(bool(report["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
