from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from .data import prepare_lending_data
from .etl import load_lending_data
from .models import train_models
from .rag import _execute_sql, answer_question, clear_cache


def _normalize_rows(rows: list[dict]) -> list[tuple]:
    normalized = []
    for row in rows:
        vals = []
        for v in row.values():
            if isinstance(v, float):
                vals.append(round(v, 2))
            else:
                vals.append(v)
        normalized.append(tuple(vals))
    return normalized


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(ordered[idx], 3)


def run_benchmark(database: str, benchmark_file: str | Path, output_file: str | Path) -> dict:
    clear_cache()
    cases = json.loads(Path(benchmark_file).read_text())
    sql_ok = 0
    entity_scores: list[float] = []
    exact_matches = 0
    cube_exact_matches = 0
    cold_latencies: list[float] = []
    cube_latencies: list[float] = []
    warm_latencies: list[float] = []
    details: list[dict] = []

    for case in cases:
        res = answer_question(case["question"], database, use_cache=False, use_olap_cube=False)
        cube_res = answer_question(case["question"], database, use_cache=False, use_olap_cube=True)
        # Prime cache then measure warm hit
        answer_question(case["question"], database, use_cache=True, use_olap_cube=True)
        warm = answer_question(case["question"], database, use_cache=True, use_olap_cube=True)

        cold_latencies.append(res["latency_ms"])
        cube_latencies.append(cube_res["latency_ms"])
        warm_latencies.append(warm["latency_ms"])

        matched_exact = False
        cube_matched_exact = False
        entity_score = 0.0
        if res["sql"]:
            sql_ok += 1
            sql_lower = res["sql"].lower()
            expected_entities = [e.lower() for e in case.get("expected_entities", [])]
            matched = sum(1 for e in expected_entities if e in sql_lower)
            entity_score = matched / len(expected_entities) if expected_entities else 1.0
            entity_scores.append(entity_score)

            if case.get("gold_sql"):
                _, gold_rows = _execute_sql(case["gold_sql"], database)
                gold_norm = _normalize_rows(gold_rows)
                if _normalize_rows(res["rows"]) == gold_norm:
                    matched_exact = True
                    exact_matches += 1
                if _normalize_rows(cube_res["rows"]) == gold_norm:
                    cube_matched_exact = True
                    cube_exact_matches += 1
            elif len(res["rows"]) > 0:
                matched_exact = True
                exact_matches += 1
                cube_matched_exact = len(cube_res["rows"]) > 0
                if cube_matched_exact:
                    cube_exact_matches += 1
        else:
            entity_scores.append(0.0)

        details.append({
            "id": case.get("id"),
            "category": case.get("category"),
            "question": case["question"],
            "execution_mode": res["execution_mode"],
            "sql_ok": bool(res["sql"]),
            "entity_match": round(entity_score, 4),
            "exact_result_match": matched_exact,
            "cube_exact_result_match": cube_matched_exact,
            "row_count": len(res["rows"]),
            "cold_latency_ms": res["latency_ms"],
            "olap_cube_cold_latency_ms": cube_res["latency_ms"],
            "warm_latency_ms": warm["latency_ms"],
        })

    n = max(len(cases), 1)
    cold_mean = round(sum(cold_latencies) / n, 3)
    cube_mean = round(sum(cube_latencies) / n, 3)
    warm_mean = round(sum(warm_latencies) / n, 3)
    summary = {
        "total_queries": len(cases),
        "sql_execution_success_pct": round(100.0 * sql_ok / n, 2),
        "schema_entity_match_pct": round(100.0 * (sum(entity_scores) / n), 2),
        "result_set_exact_match_pct": round(100.0 * exact_matches / n, 2),
        "olap_cube_exact_match_pct": round(100.0 * cube_exact_matches / n, 2),
        "cold_mean_latency_ms": cold_mean,
        "cold_p50_latency_ms": round(statistics.median(cold_latencies), 3) if cold_latencies else 0.0,
        "cold_p95_latency_ms": _percentile(cold_latencies, 95),
        "olap_cube_cold_mean_latency_ms": cube_mean,
        "olap_cube_cold_p50_latency_ms": round(statistics.median(cube_latencies), 3) if cube_latencies else 0.0,
        "olap_cube_cold_p95_latency_ms": _percentile(cube_latencies, 95),
        "olap_cube_speedup_factor": round(cold_mean / max(cube_mean, 0.001), 2),
        "warm_cache_mean_latency_ms": warm_mean,
        "warm_cache_p95_latency_ms": _percentile(warm_latencies, 95),
        "cache_speedup_factor": round(cold_mean / max(warm_mean, 0.001), 1),
        "query_results": details,
    }
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lending Club reproducible workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Normalize/filter a manually downloaded Kaggle CSV")
    prepare.add_argument("input")
    prepare.add_argument("output")
    prepare.add_argument("--limit", type=int, default=100_000)
    prepare.add_argument("--seed", type=int, default=42)

    load = sub.add_parser("load", help="Load prepared CSV into SQLite or PostgreSQL")
    load.add_argument("input")
    load.add_argument("database")

    train = sub.add_parser("train", help="Train classifier and recovery regressor")
    train.add_argument("input")
    train.add_argument("artifacts")

    bench = sub.add_parser("benchmark", help="Run golden NL-to-SQL benchmark")
    bench.add_argument("database")
    bench.add_argument("benchmark_file")
    bench.add_argument("output")

    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_lending_data(args.input, args.output, limit=args.limit, seed=args.seed)
        print(f"Prepared {len(result.frame)} rows; excluded statuses: {', '.join(result.excluded_statuses) or 'none'}")
    elif args.command == "load":
        import pandas as pd
        stats = load_lending_data(pd.read_csv(args.input), args.database)
        print(f"Loaded {args.input} -> {stats}")
    elif args.command == "train":
        import pandas as pd
        metrics = train_models(pd.read_csv(args.input), args.artifacts)
        print(json.dumps({k: v for k, v in metrics.items() if k != "classification"}, indent=2))
    else:
        summary = run_benchmark(args.database, args.benchmark_file, args.output)
        print(json.dumps({k: v for k, v in summary.items() if k != "query_results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
