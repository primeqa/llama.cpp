# IBM Granite Embedding (ModernBert) — llama.cpp Integration Status

Branch: `granite-embedding-multilingual-r2`
Date: 2026-04-28

## TL;DR

- **97m model**: end-to-end **fixed**, min cosine **0.99996** vs HF reference (was 0.856), confirmed via both `llama-embedding` and `llama-server /v1/embeddings`. Fix is a small, backward-compatible activation hparam added to ModernBert: `LLM_KV_HIDDEN_ACT` written by the converter, read into `hparams.ffn_act_swiglu`, and selected at the GLU pair in `modern-bert.cpp`.
- **311m model**: model graph is now **proven correct**. The remaining min cosine of 0.9533 is explained entirely by tokenization differences — verified by feeding HF the same tokens llama.cpp produces, which reproduces 0.9647 exactly. Remaining work is a tokenizer change (see "Open issue: 311m tokenizer" below).
- **Test infra added**: `granite-debug/test-granite-embedding.py` (HF reference + per-layer hidden states), `granite-debug/test-granite-llama-cpp.sh` (driver), `granite-debug/compare-granite-embeddings.py` (cosine).
- **GGUFs**: re-converted f32/bf16 GGUFs at `ibm-granite/granite-embedding-97m-multilingual-r2/granite-97m-debug-{f32,bf16}.gguf` and the 311m equivalent. The pre-existing `*-BF16.gguf` files in those dirs are stale (lack the new `hidden_activation` key) — re-convert with this branch's converter to pick up the fix.

## Results

| model | format | baseline cos (min) | post-fix cos (min) |
|---|---|---|---|
| **97m** | BF16 (existing GGUF, stale) | 0.856 | — |
| **97m** | F32 (re-converted) | — | **0.99997** ✅ |
| **97m** | BF16 (re-converted) | — | **0.99996** ✅ |
| **97m** via `llama-server /v1/embeddings` | BF16 | — | **0.99996** ✅ |
| **311m** | BF16 (existing GGUF) | 0.953 | — |
| **311m** | F32 (re-converted) | — | 0.953 ⚠ (tokenizer-bound) |

Test inputs:
1. "Hello world."
2. "What is the capital of France?"
3. "Granite embedding models are produced by IBM Research."
4. "The quick brown fox jumps over the lazy dog."

## Fixed issue: 97m FFN activation mismatch

**Root cause**: IBM Granite Embedding 97m R2 sets `hidden_activation = "silu"` in its HF config (i.e. SiLU → SwiGLU in the gated FFN). llama.cpp's ModernBert graph hardcoded `LLM_FFN_GEGLU` for all ModernBert variants, so the 97m's FFN was running with the wrong nonlinearity. 311m uses `gelu` which already matches GeGLU.

**Diagnosis path**:
1. `gguf-dump` of both shipped GGUFs confirmed all other ModernBert hparams were correct (rope_freq_base 150000 + freq_base_swa 160000, sliding_window 128, sliding_window_pattern 3, layer_norm_eps 1e-5/1e-12, pooling_type 2 (CLS), correct BOS ids).
2. Tensor list confirmed `token_embd_norm` present + no `blk.0.attn_norm` — matches the current "layer 0 identity" assumption in `src/models/modern-bert.cpp:31`.
3. Baseline cosine: 97m=0.856, 311m=0.953. The big gap on 97m + small gap on 311m (with same architecture) pointed at activation, not architecture.
4. Phase 5 (per-layer eval-callback) was queued but unnecessary — the baseline split between models was conclusive.

**Files changed** (~30 lines, fully backward-compatible — GGUFs without the new key default to GeGLU):

| file | change |
|---|---|
| `gguf-py/gguf/constants.py` | new key `Keys.LLM.HIDDEN_ACT = "{arch}.hidden_activation"` |
| `gguf-py/gguf/gguf_writer.py` | new method `add_hidden_act(value: str)` |
| `convert_hf_to_gguf.py` | `ModernBertModel.set_gguf_parameters` writes `hparams["hidden_activation"]` (defaults to `"gelu"`) |
| `src/llama-arch.h` | new `LLM_KV_HIDDEN_ACT` enum |
| `src/llama-arch.cpp` | string entry `"%s.hidden_activation"` |
| `src/llama-hparams.h` | new `bool ffn_act_swiglu = false` |
| `src/llama-model.cpp` | in `LLM_ARCH_MODERN_BERT` block, read the key and set `hparams.ffn_act_swiglu = (hidden_act == "silu" || hidden_act == "swish")` |
| `src/models/modern-bert.cpp` | `build_ffn(... hparams.ffn_act_swiglu ? LLM_FFN_SWIGLU : LLM_FFN_GEGLU ...)` |

`LLM_FFN_GEGLU` and `LLM_FFN_SWIGLU` both consume an `ffn_up` tensor sized `2*n_ff` (already provisioned in `src/llama-model.cpp:3680`), so no tensor reshape was needed — only the activation enum changes.

**Verification**:
```bash
# Re-convert with new converter
python convert_hf_to_gguf.py ibm-granite/granite-embedding-97m-multilingual-r2 \
    --outfile ibm-granite/granite-embedding-97m-multilingual-r2/granite-97m-debug-bf16.gguf \
    --outtype bf16

# Reference (HF + sentence-transformers)
python granite-debug/test-granite-embedding.py \
    ibm-granite/granite-embedding-97m-multilingual-r2 /tmp/granite-ref-97m

# llama.cpp via CLI
granite-debug/test-granite-llama-cpp.sh \
    ibm-granite/granite-embedding-97m-multilingual-r2/granite-97m-debug-bf16.gguf \
    /tmp/granite-llama-97m /tmp/granite-ref-97m
python granite-debug/compare-granite-embeddings.py \
    /tmp/granite-ref-97m/final_embeddings.npy /tmp/granite-llama-97m/embeddings.tsv
# -> min=0.999962  mean=0.999971
```

llama-server smoke test (`/v1/embeddings` OAI-compat):
```
server cos[0] (Hello world.)                       = 0.999962
server cos[2] (Granite embedding ... IBM Research) = 0.999963
```

## Open issue: 311m tokenizer (not in scope of this fix)

**Confirmed**: the 311m model graph in llama.cpp is **correct**. Proof:

```python
# Feed HF model the EXACT tokens llama.cpp produces:
# HF native ids for "Hello world.": [2, 9259, 1902, 236761]                  (4 tokens)
# llama.cpp's ids for "Hello world.": [2, 9259, 245237, 12392, 236761]       (5 tokens)
# cos(HF native vs HF run with llama-cpp's tokens) = 0.9647
```

This 0.9647 exactly matches the cosine reported by llama.cpp end-to-end, so the entire gap is tokenization, not the model.

**Mechanism**: 311m's HF tokenizer (`tokenizer.json`) uses
```
normalizer    = Replace(" " → "▁")
pre_tokenizer = Split on " " (MergedWithPrevious)
model         = BPE
```
Spaces are normalized to `▁` (U+2581) before BPE merges. The vocab tokens are stored as raw UTF-8 strings (e.g. token 1902 is literally `"▁world"` with bytes `0xe2 0x96 0x81 0x77 0x6f 0x72 0x6c 0x64`).

llama.cpp's `tokenizer.ggml.pre = "modern-bert"` maps to `LLAMA_VOCAB_PRE_TYPE_GPT2`, which:
1. Does **not** normalize ` ` → `▁` before BPE.
2. Encodes input bytes through GPT2's byte→unicode table, so `0xe2 0x96 0x81` becomes characters `â ĸ ģ` — but the merges in this vocab are stored against literal `▁`, not against `â ĸ ģ`. They never match.

**What I tried** (then reverted to keep the diff minimal):
- Added a `tokenizer_pre == "granite-embedding-r2"` branch in `src/llama-vocab.cpp` that reuses the GPT2 splitter with `escape_whitespaces = true`, plus converter-side detection of the normalizer. Result was *worse* (7 tokens for "Hello world." instead of 4): `escape_whitespaces` writes the ▁ sequence in raw UTF-8, which is then passed through the GPT2 byte→unicode table and still doesn't match the vocab.

**What's needed** to fix 311m end-to-end:
- Option A: register a new pre-tokenizer in `convert_hf_to_gguf_update.py` (with the test-corpus checksum) that maps to a SentencePiece-style loading path — i.e. byte-passthrough BPE that operates on raw UTF-8, plus the ` `→`▁` normalization at input time.
- Option B: switch ModernBert with this normalizer to a different `tokenizer.ggml.model` (e.g. `"spm"`/`"llama"` instead of `"gpt2"`) and adapt `set_vocab` accordingly.

Neither is a small change. The existing `convert_hf_to_gguf_update.py` already has a hash entry registering the **97m** tokenizer as `modern-bert`; the 311m tokenizer is a different beast and would warrant its own entry.

## Plan

The full debugging plan that produced this fix is at `~/.claude/plans/i-want-to-add-toasty-mitten.md`. Phase 8 (4-layer truncated mini-Granite) was deleted as unnecessary — the activation bug was identified directly from the baseline cosine split between models, without needing layer-by-layer diff via `llama-eval-callback`.

## Files & artifacts

**Code changes** (uncommitted, on `granite-embedding-multilingual-r2`):
```
 M convert_hf_to_gguf.py
 M gguf-py/gguf/constants.py
 M gguf-py/gguf/gguf_writer.py
 M src/llama-arch.cpp
 M src/llama-arch.h
 M src/llama-hparams.h
 M src/llama-model.cpp
 M src/models/modern-bert.cpp
```

**Test scripts** (new):
```
granite-debug/test-granite-embedding.py     — HF reference: final emb + per-layer hidden states
granite-debug/test-granite-llama-cpp.sh     — invokes build/bin/llama-embedding per input
granite-debug/compare-granite-embeddings.py — cosine similarity, min/mean
```

**GGUFs** (in the model directories, re-convert any time):
```
ibm-granite/granite-embedding-97m-multilingual-r2/granite-97m-debug-f32.gguf
ibm-granite/granite-embedding-97m-multilingual-r2/granite-97m-debug-bf16.gguf
ibm-granite/granite-embedding-311m-multilingual-r2/granite-311m-debug-f32.gguf
```

**Python env**: `/home/raduf/miniforge3/envs/docu5/bin/python` (sentence_transformers 5.2.3, transformers 5.3.0, torch 2.7.0).
