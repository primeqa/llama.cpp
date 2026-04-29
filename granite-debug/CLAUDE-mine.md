# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## Build Commands

```bash
# Standard CPU build
cmake -B build
cmake --build build --config Release -j $(nproc)

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j $(nproc)

# CUDA build
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)

# Vulkan build
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j $(nproc)
```

Build output goes to `build/bin/`. The Makefile is a stub that errors and redirects to CMake.

## Testing

```bash
# Run all tests (from build directory)
cd build && ctest --output-on-failure -L main

# Run a single test binary
./build/bin/test-tokenizer-0 /path/to/model.gguf

# Run backend ops test (required when modifying ggml operators)
./build/bin/test-backend-ops

# Run CI locally
bash ./ci/run.sh ./tmp/results ./tmp/mnt
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

## Python / Conversion Scripts

```bash
# Type checking
mypy convert_hf_to_gguf.py

# Convert a HuggingFace model to GGUF
python convert_hf_to_gguf.py /path/to/hf-model --outfile model.gguf

# Update convert_hf_to_gguf.py metadata from HF
python convert_hf_to_gguf_update.py
```

Python dependencies are managed with `poetry` (see `pyproject.toml`) or `pip install -r requirements.txt`.

## Code Style

- **Indentation**: 4 spaces, brackets on same line
- **Pointer/reference spacing**: `void * ptr`, `int & a`
- **Integer types**: use `int32_t`, `size_t`, etc. in public API; avoid bare `int` in interfaces
- **Naming**: `snake_case` everywhere; enum values `UPPER_CASE` prefixed with enum name; optimize for longest common prefix (e.g., `number_small` not `small_number`)
- **Structs**: `struct foo {}` not `typedef struct foo {} foo`; omit `struct`/`enum` keywords in C++ where not required
- **Naming pattern**: `<class>_<action>_<noun>` — use `init`/`free` for constructor/destructor; omit `get_`; use `_t` suffix for opaque types
- **Filenames**: lowercase with dashes for C/C++ (`.h`/`.c`/`.cpp`); lowercase with underscores for Python
- **Formatting**: use `clang-format` (v15+) for C/C++ when in doubt; `.clang-format` and `.clang-tidy` are present
- **Tensors**: stored row-major; dimension 0 = columns, 1 = rows, 2 = matrices. Matrix multiply `C = ggml_mul_mat(ctx, A, B)` means $C = B A^T$
- Avoid STL templates and fancy modern constructs; prefer simple `for` loops; keep vertical alignment for readability
- Avoid adding third-party dependencies

## Repository Architecture

### Core layers

```
ggml/           — low-level tensor library (CPU + accelerator backends)
  include/      — public ggml headers (ggml.h, ggml-backend.h, gguf.h, ...)
  src/          — backend implementations: ggml-cpu/, ggml-cuda/, ggml-metal/, ggml-vulkan/, etc.

include/        — public llama.cpp API (llama.h)
src/            — llama library implementation
  llama.cpp     — main entry point, ties everything together
  llama-model.cpp/h       — model loading, graph building per architecture
  llama-arch.cpp/h        — architecture enum + tensor name registry
  llama-context.cpp/h     — inference context, KV cache management
  llama-vocab.cpp/h       — tokenizer
  llama-sampler.cpp/h     — sampling strategies
  llama-model-loader.cpp  — GGUF file loading
  llama-graph.cpp/h       — compute graph helpers

common/         — shared utilities used by tools and examples
  common.cpp/h  — argument parsing, model init helpers
  chat.cpp/h    — chat template handling
  sampling.cpp/h — high-level sampling wrappers
  peg-parser.cpp/h        — PEG parser for model output parsing
  chat-peg-parser.cpp/h   — chat-specific PEG parsers
  jinja/        — Jinja2 template engine (for chat templates)

gguf-py/gguf/   — Python library for reading/writing GGUF files
  constants.py  — MODEL_ARCH enum, MODEL_TENSORS, tensor name definitions
  tensor_mapping.py — HF → GGUF tensor name mappings
  gguf_writer.py — GGUF file writer

tools/          — main user-facing programs
  server/       — OpenAI-compatible HTTP inference server (most complex tool)
  cli/          — interactive chat CLI
  quantize/     — model quantization
  llama-bench/  — performance benchmarking
  perplexity/   — perplexity evaluation
  imatrix/      — importance matrix for quantization
  mtmd/         — multimodal (vision) support library

tests/          — C++ and Python test files; run via ctest
```

### Data flow for inference

1. Model file (GGUF) → `llama-model-loader` → `llama_model` (weights in memory)
2. `llama_context` wraps a model, owns KV cache and compute buffers
3. `llama_batch` feeds token sequences into the context
4. `llama_model::build_graph` constructs a `ggml_cgraph` for the architecture
5. `ggml_backend` executes the graph on CPU/GPU
6. Logits → sampler chain → next token

### Adding a new model architecture

Full guide: [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md)

Summary:
1. **Python conversion** (`convert_hf_to_gguf.py`): subclass `TextModel` or `MmprojModel`, register with `@ModelBase.register("ArchForCausalLM")`, set `model_arch`
2. **gguf-py constants** (`gguf-py/gguf/constants.py`): add `MODEL_ARCH` enum entry, name in `MODEL_ARCH_NAMES`, tensor list in `MODEL_TENSORS`
3. **Tensor mapping** (`gguf-py/gguf/tensor_mapping.py`): map HF tensor names → GGUF names using `{bid}` for block index
4. **C++ arch definition** (`src/llama-arch.h/cpp`): add `llm_arch` enum value, arch name, tensor names
5. **Graph implementation** (`src/llama-model.cpp`): create struct inheriting `llm_graph_context`, implement constructor, add case in `llama_model::build_graph`
6. **Initial PR**: CPU-only; GPU backends in follow-up PRs

### Server architecture

`tools/server/` implements an OpenAI-compatible HTTP API. Key components:
- `server_context`: holds primary inference state and active slots
- `server_slot`: one parallel inference sequence
- `server_queue` / `server_response`: thread-safe work queues between HTTP workers and inference loop
- `server_routes`: JSON parsing/formatting middleware

Features that require external API call loops, model-specific behavior, or exposing internal model state are explicitly out of scope. See [tools/server/README-dev.md](tools/server/README-dev.md) before adding server features.

### PEG parser / auto-parser

Use the PEG parser (`common/peg-parser.h`) instead of regex for parsing model output — it supports partial/streaming input and JSON. The auto-parser (`common/chat-auto-parser.h`) wraps PEG with automatic detection of model-specific features. See [docs/development/parsing.md](docs/development/parsing.md) and [docs/autoparser.md](docs/autoparser.md).

## Key Resources

- [CONTRIBUTING.md](CONTRIBUTING.md) — coding and naming guidelines, PR process
- [docs/build.md](docs/build.md) — full build options for all backends
- [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md) — step-by-step model addition guide
- [tools/server/README.md](tools/server/README.md) — server usage
- [tools/server/README-dev.md](tools/server/README-dev.md) — server development and scope
- [docs/development/parsing.md](docs/development/parsing.md) — PEG parser usage
- [docs/autoparser.md](docs/autoparser.md) — auto-parser
- [common/jinja/README.md](common/jinja/README.md) — Jinja engine
