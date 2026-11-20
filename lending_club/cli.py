from __future__ import annotations

import argparse
from pathlib import Path

from .data import prepare_lending_data
from .etl import load_lending_data
from .models import train_models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lending Club reproducible workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Normalize/filter a manually downloaded Kaggle CSV")
    prepare.add_argument("input")
    prepare.add_argument("output")
    prepare.add_argument("--limit", type=int, default=100_000)
    prepare.add_argument("--seed", type=int, default=42)
    load = sub.add_parser("load", help="Load prepared CSV into SQLite")
    load.add_argument("input")
    load.add_argument("database")
    train = sub.add_parser("train", help="Train classifier and loan amount regressor")
    train.add_argument("input")
    train.add_argument("artifacts")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_lending_data(args.input, args.output, limit=args.limit, seed=args.seed)
        print(f"Prepared {len(result.frame)} rows; excluded statuses: {', '.join(result.excluded_statuses) or 'none'}")
    elif args.command == "load":
        import pandas as pd
        load_lending_data(pd.read_csv(args.input), args.database)
        print(f"Loaded {args.input} into {args.database}")
    else:
        import pandas as pd
        metrics = train_models(pd.read_csv(args.input), args.artifacts)
        print(metrics)
    return 0

