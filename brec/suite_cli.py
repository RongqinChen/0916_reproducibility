"""BREC commands dispatched by the repository's reproduce.py entry point."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .suite_runtime import ROOT, STAGES, atomic_json, digest, make_jobs, protocol, run, worker

PLAN_FORMAT = "exact_brec_suite_v2"


def parse_pairs(value):
    result = []
    for part in value.split(","):
        if "-" in part:
            first, last = map(int, part.split("-"))
            if last < first:
                raise argparse.ArgumentTypeError("Pair ranges must increase")
            result.extend(range(first, last+1))
        else:
            result.append(int(part))
    if not result or min(result) < 0 or max(result) >= 400 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Select distinct pair IDs in 0..399")
    return sorted(result)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python reproduce.py", description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("brec-plan", "brec-run"):
        sub = commands.add_parser(name)
        sub.add_argument("--root", "--out", dest="out", type=Path, required=True)
        sub.add_argument("--brec-file", "--data", dest="data", type=Path,
                         required=name == "brec-plan")
        sub.add_argument("--config", type=Path)
        sub.add_argument("--stage", choices=["all", *STAGES])
        sub.add_argument("--pairs", type=parse_pairs)
        sub.add_argument("--dry-run", action="store_true", help="describe the plan without writing or executing")
        sub.add_argument("--trust-pickle", action="store_true", help=argparse.SUPPRESS)
        if name == "brec-run":
            sub.add_argument("--workers", type=int, default=44)
            sub.add_argument("--memory-gib", type=float, default=100)
            sub.add_argument("--reserve-gib", type=float, default=8)
            sub.add_argument("--parent-memory-gib", type=float, default=4)
            sub.add_argument("--job-timeout", type=float, default=21600)
            sub.add_argument("--job-memory-scale", type=float, default=1)
            sub.add_argument("--admission-timeout", type=float, default=300)
            sub.add_argument("--resume", action="store_true", help="also enabled automatically for an existing manifest")
            sub.add_argument("--retry-incomplete", action="store_true")
    sub = commands.add_parser("brec-report")
    sub.add_argument("--root", "--out", dest="out", type=Path, required=True)
    sub.add_argument("--verify-artifacts", action="store_true")
    sub.add_argument("--strict", action="store_true")
    sub.add_argument("--dry-run", action="store_true")
    sub = commands.add_parser("brec-worker", help="internal frozen-source worker")
    sub.add_argument("job_file", type=Path)
    sub = commands.add_parser("brec-rescore")
    sub.add_argument("embeddings", type=Path)
    args = parser.parse_args(argv)
    if args.action == "brec-worker":
        worker(args.job_file)
        return
    if args.action == "brec-rescore":
        import numpy as np
        from .suite_neural import rpc_score
        with np.load(args.embeddings, allow_pickle=False) as values:
            print(json.dumps({k: rpc_score(values[k]) for k in ("main", "reliability")}, indent=2))
        return
    args.out = args.out.resolve()
    if args.action == "brec-report":
        if args.dry_run:
            print(f"Report source: {args.out / 'manifest.json'}")
            return
        from .suite_report import summarize
        summary = summarize(args.out, args.verify_artifacts)
        print(json.dumps({k: v for k, v in summary.items() if k != "issues"} |
                         {"issue_count": len(summary["issues"]), "issues_file": str(args.out/"reports"/"issues.json")}, indent=2))
        if args.strict and not summary["safe_to_use_as_complete_results"]:
            raise SystemExit(2)
        return
    if args.trust_pickle:
        parser.error("The exact suite accepts fixed-width graph6 strings and does not load pickle/object arrays")
    plan_path = args.out/"plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    if plan and plan.get("format") != PLAN_FORMAT:
        parser.error("This is a legacy BREC plan. Use a new --root for the exact suite; the legacy frozen scripts remain in that output directory")
    if plan:
        scientific = plan["scientific"]
        if plan.get("plan_sha256") != digest(scientific):
            parser.error("Plan checksum mismatch")
        config = scientific["config"]
        if args.config and json.loads(args.config.read_text()) != config:
            parser.error("The supplied configuration differs from the saved plan")
        if args.data and args.data.resolve() != Path(scientific["data"]):
            parser.error("The supplied dataset path differs from the saved plan")
        args.data = Path(scientific["data"])
        args.pairs = args.pairs if args.pairs is not None else scientific["pairs"]
        args.stage = args.stage or scientific["stage"]
    else:
        config = json.loads((args.config or ROOT/"configs"/"brec_suite.json").read_text())
        args.pairs = args.pairs if args.pairs is not None else list(range(400))
        args.stage = args.stage or "all"
    config = protocol(config)
    if args.data is None:
        parser.error("Supply --brec-file or create a plan with brec-plan first")
    args.data = args.data.resolve()
    jobs = make_jobs(config, args.pairs, STAGES if args.stage == "all" else [args.stage])
    description = {"entry_point": "reproduce.py", "root": str(args.out), "data": str(args.data),
                   "stage": args.stage, "pairs": len(args.pairs), "jobs": dict(Counter(j["stage"] for j in jobs)),
                   "total_jobs": len(jobs), "default_workers": 44, "server_memory_gib": 100}
    if args.dry_run:
        print(json.dumps(description, indent=2))
        return
    if args.action == "brec-plan":
        if args.out.exists() and any(args.out.iterdir()):
            parser.error("Planning requires an empty output root; use brec-run to resume a saved plan")
        scientific = {"config": config, "data": str(args.data), "pairs": args.pairs, "stage": args.stage}
        atomic_json(plan_path, {"format": PLAN_FORMAT, "scientific": scientific,
                                "plan_sha256": digest(scientific), "summary": description})
        print(json.dumps(description, indent=2))
        return
    numeric = (args.memory_gib, args.reserve_gib, args.parent_memory_gib, args.job_timeout,
               args.job_memory_scale, args.admission_timeout)
    if (args.workers < 1 or any(not math.isfinite(v) for v in numeric) or args.reserve_gib < 0
            or args.parent_memory_gib <= 0 or args.memory_gib <= args.reserve_gib+args.parent_memory_gib
            or args.job_timeout < 0 or args.job_memory_scale <= 0 or args.admission_timeout < 0):
        parser.error("Invalid CPU, memory, or time budgets")
    args.resume = args.resume or (args.out/"manifest.json").exists()
    out = run(args, config)
    from .suite_report import summarize
    summary = summarize(out)
    print(json.dumps({"coverage": summary["coverage"], "issue_count": len(summary["issues"]),
                      "all_requested_jobs_complete": summary["all_requested_jobs_complete"],
                      "reports": str(out/"reports")}, indent=2))
    if not summary["safe_to_use_as_complete_results"]:
        raise SystemExit(2)
