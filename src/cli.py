#!/usr/bin/env python3
"""
cli.py — Nexus Router CLI

Usage:
  python -m src.cli route --task coding --complexity high
  python -m src.cli route --task vision --has-image
  python -m src.cli route --task summarization --long-context
  python -m src.cli route --task eco
  python -m src.cli explain
  python -m src.cli stats
"""

import argparse
import json
import sys

from .router import Router
from .types import ClassifierOutput, PreSignals
from .quota_sync import load_snapshots_from_path, sync_snapshots


def cmd_route(args):
    router = Router(persist=False)

    classifier = ClassifierOutput(
        task_type=args.task,
        subtype=args.subtype,
        complexity=args.complexity,
        needs_tools=not args.no_tools,
        needs_vision=args.has_image,
        needs_long_context=args.long_context,
        cost_profile=args.cost_profile,
        confidence=args.confidence,
        detected_language=args.language,
    )

    pre_signals = PreSignals(
        has_image=args.has_image,
        has_code=args.has_code,
        has_diff=args.has_diff,
        has_logs=args.has_logs,
        estimated_tokens=args.estimated_tokens,
    )

    decision = router.route(classifier, pre_signals)

    if args.json:
        print(json.dumps({
            "task_type": decision.task_type,
            "selected_model": decision.selected_model,
            "fallbacks": decision.fallbacks,
            "score": decision.score,
            "reason": decision.reason,
        }, indent=2))
    else:
        print(router.explain(decision))
        print()
        print("Top 5 scored models:")
        for s in [x for x in decision.all_scores if not x.excluded][:5]:
            print(f"  {s.model_id:55s}  total={s.total_score:.3f}  "
                  f"fit={s.task_fit:.2f}  health={s.health:.2f}  "
                  f"pref={s.preference:.2f}  learned={s.learned:.2f}")


def cmd_stats(args):
    from .db import load_model_stats
    stats = load_model_stats()
    if not stats:
        print("No stats yet. Stats are recorded after routing decisions with outcomes.")
        return
    print(f"{'Model':<55} {'Total':>6} {'Success%':>9} {'Overrides':>10} {'Avg Latency':>12}")
    print("-" * 100)
    for model, s in sorted(stats.items(), key=lambda x: -(x[1].get("success_rate") or 0)):
        sr  = s.get("success_rate")
        lat = s.get("avg_latency_ms")
        print(f"{model:<55} {s['total_selected']:>6} "
              f"{f'{sr*100:.1f}%' if sr else 'n/a':>9} "
              f"{s.get('total_override', 0):>10} "
              f"{f'{lat:.0f}ms' if lat else 'n/a':>12}")


def cmd_quota_sync(args):
    snapshots = load_snapshots_from_path(args.input, skip_missing=args.skip_missing)
    results = sync_snapshots(snapshots)
    if args.json:
        print(json.dumps({"synced": results}, indent=2))
    else:
        for row in results:
            ratio = row.get("quota_remaining_ratio")
            ratio_text = f"{ratio:.3f}" if isinstance(ratio, (int, float)) else "n/a"
            print(
                f"{row['provider']}: auth={row.get('auth', 'unknown')} quota={row.get('quota', 'unknown')} "
                f"ratio={ratio_text}"
            )


def main():
    parser = argparse.ArgumentParser(description="Nexus Router CLI")
    sub = parser.add_subparsers(dest="command")

    # route subcommand
    p_route = sub.add_parser("route", help="Route a request to the best model")
    p_route.add_argument("--task", default="general_chat",
                         choices=["coding","code_review","reasoning","summarization",
                                  "fast_utility","long_context","vision","general_chat","eco"])
    p_route.add_argument("--subtype", default=None)
    p_route.add_argument("--complexity", default="medium", choices=["low","medium","high"])
    p_route.add_argument("--cost-profile", default="balanced", choices=["cheap","balanced","premium"])
    p_route.add_argument("--confidence", type=float, default=0.80)
    p_route.add_argument("--language", default=None)
    p_route.add_argument("--no-tools", action="store_true")
    p_route.add_argument("--has-image", action="store_true")
    p_route.add_argument("--has-code", action="store_true")
    p_route.add_argument("--has-diff", action="store_true")
    p_route.add_argument("--has-logs", action="store_true")
    p_route.add_argument("--long-context", action="store_true")
    p_route.add_argument("--estimated-tokens", type=int, default=0)
    p_route.add_argument("--json", action="store_true", help="Output JSON")
    p_route.set_defaults(func=cmd_route)

    # stats subcommand
    p_stats = sub.add_parser("stats", help="Show learned model stats")
    p_stats.set_defaults(func=cmd_stats)

    # quota-sync subcommand
    p_quota = sub.add_parser("quota-sync", help="Sync provider quota snapshots into runtime health")
    p_quota.add_argument("--input", default="-", help="JSON file path or - for stdin")
    p_quota.add_argument("--skip-missing", action="store_true", help="Exit 0 if the input file is missing")
    p_quota.add_argument("--json", action="store_true", help="Output JSON")
    p_quota.set_defaults(func=cmd_quota_sync)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
