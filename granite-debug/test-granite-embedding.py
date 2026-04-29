#!/usr/bin/env python3
"""HF reference for IBM Granite ModernBert embedding models.

Writes:
  inputs.json            — list[str] of test inputs
  final_embeddings.npy   — (N, hidden_size) L2-normalized CLS embeddings via sentence-transformers
  hidden_states.npz      — input_ids, attention_mask, h0..h{n_layer}, last
                           per-layer hidden states from AutoModel(output_hidden_states=True)

Usage:
  python3 scripts/test-granite-embedding.py <model_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


INPUTS = [
    "Hello world.",
    "What is the capital of France?",
    "Granite embedding models are produced by IBM Research.",
    "The quick brown fox jumps over the lazy dog.",
]


def main(model_dir: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "inputs.json").write_text(json.dumps(INPUTS))

    print(f"[ref] loading sentence-transformer from {model_dir}")
    st = SentenceTransformer(model_dir, trust_remote_code=True, device="cpu")
    embs = st.encode(INPUTS, normalize_embeddings=True, convert_to_numpy=True)
    np.save(out / "final_embeddings.npy", embs)
    print(f"[ref] final_embeddings shape={embs.shape} norm[0]={np.linalg.norm(embs[0]):.4f}")

    print(f"[ref] loading AutoModel from {model_dir}")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(
        model_dir,
        output_hidden_states=True,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).eval()

    with torch.no_grad():
        enc = tok(INPUTS, padding=True, return_tensors="pt")
        out_hf = model(**enc)
        hidden = [h.cpu().numpy() for h in out_hf.hidden_states]
        last = out_hf.last_hidden_state.cpu().numpy()

    payload = {
        "input_ids": enc["input_ids"].numpy(),
        "attention_mask": enc["attention_mask"].numpy(),
        "last": last,
    }
    for i, h in enumerate(hidden):
        payload[f"h{i}"] = h
    np.savez(out / "hidden_states.npz", **payload)

    print(f"[ref] hidden_states: {len(hidden)} layers (incl. embeddings); shape={hidden[0].shape}")
    for i, ids in enumerate(enc["input_ids"].tolist()):
        n = int(enc["attention_mask"][i].sum().item())
        print(f"[ref] [{i}] cls_id={ids[0]} eos_or_last_id={ids[n-1]} len={n}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
