# IREE Tokenizer

A high-performance streaming tokenizer library for ML inference and training,
written in C. Loads HuggingFace `tokenizer.json` files and produces identical
output to the HuggingFace tokenizers library, with full support for BPE,
WordPiece, and Unigram models.

## What This Is

The IREE tokenizer is a format-agnostic tokenization engine with a pull-based
streaming architecture. It processes arbitrarily long input with fixed memory,
produces tokens incrementally, and tracks byte-exact offsets through the full
normalization pipeline for training use cases.

The tokenizer is thread-safe and allocation-free in the hot path: all buffers
are caller-provided, all state is in fixed-size structs, and the tokenizer
object itself is immutable after construction. This makes it suitable for
high-throughput inference servers (pool state objects, share the tokenizer
across threads) as well as embedded systems (statically allocate everything).

### Key Properties

- **Streaming-first**: `encode(text)` is just `initialize() -> feed(text) ->
  finalize()`. Chunked input produces identical output regardless of chunk
  boundaries.
- **Bounded memory**: Fixed-size state (~1-2KB) plus a caller-provided
  transform buffer. No hidden allocations during encode/decode. Processes
  infinite-length I/O with constant memory.
- **Offset tracking**: Every token maps back to its exact byte range in the
  original input, propagated forward through normalization (NFC, lowercase,
  etc.) using run-length encoded offset maps. Essential for training data
  provenance and RAG chunking.
- **HuggingFace compatible**: Bit-exact match against the HuggingFace
  tokenizers library across all supported normalizers, pre-tokenizers, models,
  and decoders.
- **Safe for untrusted input**: All arithmetic uses checked overflow helpers.
  No undefined behavior on malformed UTF-8, pathological regex patterns, or
  adversarial input.

## Performance

Throughput in MiB/s (one-shot encode and decode, pool rotation with 64
copies defeating cache warming, Clang `-O3 -march=native` with ThinLTO,
AMD EPYC 5.4 GHz). Corpus: 593KB ASCII (Sherlock Holmes), 1.8MB CJK
(Chinese classic), 1.4MB Code (concatenated C source):

| Model | Algo | Vocab | ASCII Enc | CJK Enc | Code Enc | ASCII Dec | CJK Dec | Code Dec |
|-------|------|------:|----------:|--------:|---------:|----------:|--------:|---------:|
| GPT-2             | BPE       |  50K |  28 |  15 |  44 |   587 | 1,588 |   630 |
| Llama 3           | BPE       | 128K |  42 |   9 |  54 |   709 | 1,236 |   781 |
| Gemma 2B          | BPE       | 256K |  40 |  67 |  15 |   213 |   366 |   215 |
| Qwen 2.5          | BPE       | 151K |  24 |   8 |  44 |   736 | 1,270 |   835 |
| BLOOM             | BPE       | 250K |  24 |   7 |   2 |   621 |   984 |   762 |
| Mistral NeMo      | BPE       | 131K |  39 |  10 |  39 |   601 | 1,221 |   780 |
| DeepSeek V3       | BPE       | 129K |  13 |  10 |  16 |   658 | 1,667 | 2,125 |
| Whisper           | BPE       |  51K |  26 |  11 |  43 |   588 | 1,439 |   596 |
| BERT              | WordPiece |  30K |  44 |  36 |  40 |   814 | 3,061 |   792 |
| T5                | Unigram   |  32K |  18 |  34 |   3 |   646 | 1,808 |   556 |

Streaming encode produces bit-exact output regardless of buffer size and
tracks one-shot throughput closely. DeepSeek V3 streaming encode on ASCII
(same settings as above, pool rotation, Sherlock Holmes corpus):

| Buffer Size | Throughput | Tokens | vs One-Shot |
|------------:|-----------:|-------:|------------:|
|        1 KB |  10.7 MiB/s | 143,053 | 0.82x |
|        4 KB |  10.7 MiB/s | 143,053 | 0.82x |
|       16 KB |  11.6 MiB/s | 143,053 | 0.89x |
|       64 KB |  11.6 MiB/s | 143,053 | 0.89x |
|    One-Shot |  13.1 MiB/s | 143,053 | 1.00x |

Throughput is flat across all buffer sizes. The 0.82-0.89x streaming ratio is
the cost of the pass-through probe: when a segmenter child finds no matches
(e.g., Numbers split on English text), the pipeline probes the final child's
DFA to determine the natural segment boundary before expanding through the
full child chain. This avoids forcing word splits at arbitrary buffer
boundaries.

### Running Benchmarks

```bash
# Single model:
iree-bazel-run --copt=-O3 --copt=-march=native --features=thin_lto \
  //runtime/src/iree/tokenizer/tools:comprehensive_benchmark -- \
  --tokenizer_json=path/to/tokenizer.json --benchmark_min_time=1s

# All 10 models (downloads tokenizers automatically):
runtime/src/iree/tokenizer/tools/run_benchmarks.sh --benchmark_min_time=1s
```

The `--rotate` flag allocates 64 copies of each corpus at separate heap
addresses and cycles through them. The total working set exceeds L3 cache,
defeating cache warming. Off by default.

## Architecture

### Streaming Encode Pipeline

```
Input ──┐
(bytes) │
        v
 ┌─────────────┐       (by config)
 │ UTF-8       │  OR  ┌─────────────┐
 │ Decoder     │      │ ByteLevel   │
 └──────┬──────┘      │ Encoder     │
        │             └──────┬──────┘
        └───────┬────────────┘
                │ codepoints (batched)
                v
 ┌──────────────────────────────┐
 │ Normalizer Chain             │
 │ (NFC, lowercase, strip, ...) │
 └──────────────┬───────────────┘
                │ codepoints (batched)
                v
         ┌─────────────┐
         │ UTF-8       │
         │ Encoder     │
         └──────┬──────┘
                │ bytes
                v
 ┌───────────────────────────────────┐
 │ Transform Buffer (ring buffer)    │
 │ [normalized bytes]                │
 └────────────────┬──────────────────┘
                  │
                  v
 ┌──────────────────────────────┐
 │ Segmenter                    │
 │ (Metaspace, ByteLevel, BERT, │
 │  whitespace, regex split)    │
 └──────────────┬───────────────┘
                │ segments (byte ranges)
                v
 ┌──────────────────────────────┐
 │ BPE / WordPiece / Unigram    │
 │ (subword tokenization)       │
 └──────────────┬───────────────┘
                │
                v
        Token IDs + Offsets
```

The pipeline is **pull-based**: output buffer capacity drives processing. When
the caller calls `feed()`, the encoder pulls data through each stage only as
needed to fill the output. Each stage maintains a small batch buffer (default
64 codepoints) to amortize indirect call overhead. For 1M codepoints through
4 pipeline stages, this means ~63K indirect calls instead of 4M.

### Streaming Decode Pipeline

```
Token IDs ──┐
            v
 ┌─────────────────────────────┐
 │ Vocabulary Lookup           │
 │ (id -> token string)        │
 └──────────────┬──────────────┘
                │ token strings
                v
         ┌─────────────┐
         │ UTF-8       │
         │ Decoder     │
         └──────┬──────┘
                │ codepoints (batched)
                v
 ┌──────────────────────────────┐
 │ Decoder Chain                │
 │ (ByteFallback, Metaspace,    │
 │  WordPiece strip, CTC, ...)  │
 └──────────────┬───────────────┘
                │ codepoints (batched)
                v
         ┌─────────────┐
         │ UTF-8       │
         │ Encoder     │
         └──────┬──────┘
                │
                v
           Output bytes
```

### Data Flow

```
User                    Tokenizer                 Pipeline Stages
 │                          │                            │
 │  initialize(state,       │                            │
 │    buffer, capacity)     │                            │
 │─────────────────────────>│                            │
 │                          │                            │
 │  feed(state, chunk1)     │                            │
 │─────────────────────────>│  pull codepoints           │
 │                          │───────────────────────────>│
 │                          │  batched codepoints        │
 │                          │<───────────────────────────│
 │                          │  [transform, split, BPE]   │
 │  (partial tokens)        │                            │
 │<─────────────────────────│                            │
 │                          │                            │
 │  feed(state, chunk2)     │                            │
 │─────────────────────────>│            ...             │
 │  (more tokens)           │                            │
 │<─────────────────────────│                            │
 │                          │                            │
 │  finalize(state)         │  flush remaining           │
 │─────────────────────────>│───────────────────────────>│
 │  (final tokens)          │<───────────────────────────│
 │<─────────────────────────│                            │
 │                          │                            │
 │  deinitialize(state)     │                            │
 │─────────────────────────>│                            │
```

## Design Choices

### Cache and Locality Efficiency

The pipeline is organized to keep working data in L1/L2 cache:

- **Batched pull model**: Each stage processes 64 items at a time from a small
  internal buffer (~256 bytes), which fits in L1. The upstream indirect call
  happens once per batch, not once per item.
- **Ring buffer transform**: Normalized bytes flow through a power-of-two ring
  buffer. The segmenter reads from one region while the normalizer writes to
  another, keeping both hot in cache.
- **Compact vocabulary**: Token strings are stored contiguously in a single
  allocation with offset indexing. Vocabulary lookup is a single indexed load,
  not a hash table chase.
- **Merge hash table**: BPE merge lookups use an open-addressing hash table
  with linear probing, keeping probe sequences cache-line-local.

### Algorithmic Efficiency

- **BPE**: Priority-queue based merge with a fixed-size heap. Processes one
  segment at a time with O(n log n) merges where n is the segment length
  (typically 5-20 bytes, not the full input length). The heap is embedded in
  the encode state, not heap-allocated.
- **Trie-based vocabulary**: Token lookup during encoding uses a trie for
  longest-prefix matching, giving O(k) lookup where k is the max token length
  (bounded, typically < 50 bytes).
- **DFA regex**: The regex engine compiles patterns to DFA at tokenizer load
  time. Matching is O(n) in input length with no backtracking, making it safe
  for untrusted input patterns.

### Memory Consumption

The tokenizer uses a fixed, small amount of memory for streaming state:

| Component | Size | Notes |
|-----------|------|-------|
| Encode state | ~1-2 KB | Pipeline stage batch buffers + state |
| Transform buffer | 4-64 KB | Caller-provided, tunable |
| Decode state | ~1 KB | Vocab string batch buffer + decoder state |
| Tokenizer (shared) | Varies | Vocabulary + merge table + regex DFA |

The encode and decode state sizes are independent of input length. A 1-byte
input and a 1-GB input use the same state allocation. This makes the tokenizer
suitable for streaming/speculative decoding where you process tokens
incrementally without ever needing to re-decode previous output.

### Composability with Streaming Workflows

Because the tokenizer processes input incrementally with bounded state:

- **Speculative decoding**: Decode tentative tokens, discard on misprediction,
  continue from the same state. No re-decode of accepted prefix.
- **Streaming chat**: Feed each LLM-generated token to the decoder as it
  arrives. UTF-8 output is produced as soon as complete characters are
  available.
- **Chunked encoding**: Feed input in arbitrary-sized chunks (network packets,
  file read buffers). The tokenizer handles chunk boundaries transparently,
  including mid-codepoint splits.
- **Multi-turn context**: Control the `AT_INPUT_START` flag to handle
  Metaspace prepend behavior correctly across conversation turns without
  re-encoding history.

## Features

### Tokenization Models

- **BPE** (Byte-Pair Encoding): GPT-2, Llama, Gemma, Qwen, Mistral,
  DeepSeek, BLOOM, Whisper, etc.
- **WordPiece**: BERT and variants (DistilBERT, RoBERTa, DeBERTa, etc.)
- **Unigram**: T5, XLM-RoBERTa, CamemBERT, etc.

### Normalizers

NFC, NFD, NFKD, lowercase, strip, strip accents, BERT normalization,
precompiled (HuggingFace format), prepend, replace, regex replace, sequences
of the above.

### Segmenters (Pre-tokenizers)

ByteLevel (GPT-2), Metaspace (SentencePiece), BERT (whitespace + punctuation +
CJK), whitespace, digits, punctuation, regex split, sequences of the above.
All five HuggingFace split behaviors (Removed, Isolated, MergedWithPrevious,
MergedWithNext, Contiguous) are supported.

### Decoders

ByteLevel, ByteFallback, Metaspace, WordPiece (## strip), CTC, Replace, Strip,
sequences of the above.

### Post-processing

Template-based special token insertion (BOS, EOS, CLS, SEP) with
HuggingFace-compatible template pair support and type ID assignment.

### Offset Tracking

Forward-propagated byte offsets through the full normalization pipeline.
Offsets are stored as run-length encoded maps (O(discontinuities) memory, not
O(input_length)), making them practical for arbitrarily long inputs.

### Regex Engine

Self-contained regex compiler and DFA execution engine. Compiles patterns at
tokenizer load time; no runtime regex dependency. O(n) matching with no
backtracking.

## Directory Layout

```
tokenizer/
  tokenizer.h/c          Core API: encode, decode, streaming state
  types.h                Shared types (token_id_t, offset_t, etc.)
  model.h/c              Model vtable interface
  normalizer.h/c         Normalizer vtable interface
  segmenter.h/c          Segmenter vtable interface
  decoder.h/c            Decoder vtable interface
  postprocessor.h/c      Template-based special token insertion
  special_tokens.h/c     Added token matching (pre/post normalization)

  model/                 Tokenization algorithms
    bpe.h/c              BPE with priority-queue merge
    wordpiece.h/c        WordPiece greedy longest-match
    unigram.h/c          Unigram (Viterbi) model

  normalizer/            Codepoint transforms
    nfc.c, nfd.c, nfkd.c, lowercase.c, strip.c, strip_accents.c,
    bert.c, precompiled.c, prepend.c, replace.c, regex_replace.c,
    sequence.c, passthrough.c

  segmenter/             Pre-tokenization (word boundaries)
    bert.c, whitespace.c, metaspace.c, digits.c, punctuation.c,
    split.c, sequence.c, passthrough.c

  decoder/               Token-to-text transforms
    byte_level.c, byte_fallback.c, metaspace.c, wordpiece.c,
    ctc.c, replace.c, strip.c, sequence.c, passthrough.c

  vocab/                 Vocabulary management
    vocab.h/c            Vocab lookup (id->string, string->id)
    vocab_builder.h/c    Incremental vocab construction
    vocab_hash.h/c       String-to-ID hash table
    vocab_merge_hash.h/c BPE merge pair hash table
    vocab_trie.h/c       Trie for longest-prefix matching

  regex/                 Self-contained regex engine
    compile.h/c          Pattern -> NFA -> DFA compilation
    exec.h/c             DFA execution
    internal/            Lexer, parser, NFA, DFA internals

  format/                Format loaders (tokenizer core is format-agnostic)
    huggingface/         HuggingFace tokenizer.json loader
      tokenizer_json.h/c   Top-level JSON parser
      model_json.h/c       BPE/WordPiece/Unigram JSON
      normalizer_json.h/c  Normalizer chain JSON
      segmenter_json.h/c   Pre-tokenizer JSON
      decoder_json.h/c     Decoder chain JSON
      added_tokens_json.h/c  Special token JSON
      postprocessor_json.h/c Post-processor template JSON

  tools/                 Benchmarks and test infrastructure
    comprehensive_benchmark.cc  Google Benchmark suite
    run_benchmarks.sh           Downloads models, runs full suite
    huggingface_smoketest.py    Correctness tests against HuggingFace
    run_smoketest.sh            Wrapper for smoketest dependencies

  testing/               Test utilities
    scoped_resource.h    RAII wrappers for test cleanup
```

The `format/` directory contains only loaders: they parse a specific file
format and populate the tokenizer builder. The core tokenizer has no knowledge
of any file format. Future formats (SentencePiece protobuf, a custom mmap
format for zero-copy loading) will be additional subdirectories under
`format/`.

## Quick Start

```c
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"

// Load from HuggingFace JSON.
iree_tokenizer_t* tokenizer = NULL;
iree_status_t status = iree_tokenizer_from_huggingface_json(
    json_contents, iree_allocator_system(), &tokenizer);

// One-shot encode.
iree_tokenizer_token_id_t tokens[1024];
iree_host_size_t count = 0;
status = iree_tokenizer_encode(
    tokenizer, iree_make_cstring_view("Hello, world!"),
    IREE_TOKENIZER_ENCODE_FLAG_NONE,
    iree_tokenizer_make_token_output(tokens, NULL, NULL, 1024),
    iree_allocator_system(), &count);

// One-shot decode.
char text[4096];
iree_host_size_t text_length = 0;
status = iree_tokenizer_decode(
    tokenizer, iree_tokenizer_make_token_id_list(tokens, count),
    IREE_TOKENIZER_DECODE_FLAG_NONE,
    iree_make_mutable_string_view(text, sizeof(text)),
    iree_allocator_system(), &text_length);

iree_tokenizer_free(tokenizer);
```

### Streaming Encode

```c
// Calculate required state size.
iree_host_size_t state_size = 0;
iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size);

// Allocate state + transform buffer (caller controls placement).
uint8_t* state_storage = malloc(state_size);
size_t buffer_size = iree_tokenizer_transform_buffer_recommended_size(4096);
uint8_t* transform_buffer = malloc(buffer_size);

// Initialize streaming state.
iree_tokenizer_encode_state_t* state = NULL;
iree_tokenizer_encode_state_initialize(
    tokenizer,
    iree_make_byte_span(state_storage, state_size),
    iree_make_byte_span(transform_buffer, buffer_size),
    iree_tokenizer_offset_run_list_empty(),
    IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START,
    &state);

// Feed chunks as they arrive (network, file, etc.).
while (has_more_input()) {
    iree_string_view_t chunk = get_next_chunk();
    while (chunk.size > 0) {
        iree_host_size_t bytes_consumed = 0, tokens_written = 0;
        iree_tokenizer_encode_state_feed(
            state, chunk, output, &bytes_consumed, &tokens_written);
        chunk.data += bytes_consumed;
        chunk.size -= bytes_consumed;
        process_tokens(output, tokens_written);
    }
}

// Flush remaining tokens.
iree_host_size_t final_count = 0;
iree_tokenizer_encode_state_finalize(state, output, &final_count);
process_tokens(output, final_count);

iree_tokenizer_encode_state_deinitialize(state);
free(transform_buffer);
free(state_storage);
```

### Streaming Decode (Token-at-a-Time)

```c
iree_tokenizer_decode_state_t* state = NULL;
iree_tokenizer_decode_state_initialize(
    tokenizer, IREE_TOKENIZER_DECODE_FLAG_NONE,
    iree_make_byte_span(state_storage, state_size), &state);

// Feed each token as the LLM generates it.
while (llm_has_more_tokens()) {
    iree_tokenizer_token_id_t token = llm_sample_next();
    char text[64];
    iree_host_size_t consumed = 0, written = 0;
    iree_tokenizer_decode_state_feed(
        state, iree_tokenizer_make_token_id_list(&token, 1),
        iree_make_mutable_string_view(text, sizeof(text)),
        &consumed, &written);
    if (written > 0) {
        stream_to_user(text, written);  // UTF-8 safe for display
    }
}

iree_host_size_t final_bytes = 0;
iree_tokenizer_decode_state_finalize(
    state, iree_make_mutable_string_view(text, sizeof(text)), &final_bytes);
iree_tokenizer_decode_state_deinitialize(state);
```
