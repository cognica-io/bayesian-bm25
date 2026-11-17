#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Merge disjoint dataset runs with identical experimental metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_results(payloads: list[dict]) -> dict:
    if not payloads:
        raise ValueError("at least one result payload is required")
    metadata = {k: v for k, v in payloads[0].items() if k != "results"}
    datasets = {}
    for payload in payloads:
        if {k: v for k, v in payload.items() if k != "results"} != metadata:
            raise ValueError("experimental metadata differs between result files")
        overlap = set(datasets) & set(payload["results"])
        if overlap:
            raise ValueError(f"dataset results overlap: {sorted(overlap)}")
        datasets.update(payload["results"])
    return dict(metadata, results=datasets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    payload = merge_results([json.loads(Path(p).read_text(encoding="ascii")) for p in args.inputs])
    temporary = Path(args.output + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True), encoding="ascii")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
