#!/usr/bin/env python3
"""Compare HF reference embeddings to llama.cpp embeddings (cosine similarity).

Usage:
  python3 scripts/compare-granite-embeddings.py <ref.npy> <llama.tsv>
"""
import re
import sys

import numpy as np


def parse_llama_tsv(path: str) -> np.ndarray:
    rows = []
    for raw in open(path):
        line = raw.strip()
        if not line:
            continue
        # llama-embedding --embd-output-format array prints e.g. "[[ 0.12, -0.34, ... ]]"
        line = line.strip("[]")
        nums = [float(x) for x in re.split(r"[\s,]+", line) if x]
        if nums:
            rows.append(nums)
    return np.asarray(rows, dtype=np.float64)


def main(ref_path: str, got_path: str) -> int:
    ref = np.load(ref_path).astype(np.float64)
    got = parse_llama_tsv(got_path)
    if ref.shape != got.shape:
        print(f"[cmp] shape mismatch: ref={ref.shape} got={got.shape}", file=sys.stderr)
        return 2

    ref /= np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12
    got /= np.linalg.norm(got, axis=1, keepdims=True) + 1e-12
    cos = (ref * got).sum(axis=1)
    for i, c in enumerate(cos):
        print(f"  [{i}] cos={c:.6f}")
    print(f"min={cos.min():.6f}  mean={cos.mean():.6f}  max={cos.max():.6f}")
    return 0 if cos.min() > 0.99 else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1], sys.argv[2]))
