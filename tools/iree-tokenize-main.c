// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tokenizes text using HuggingFace tokenizer.json files.
//
// Example encoding text to token IDs (default: comma-separated):
//   iree-tokenize --tokenizer=tokenizer.json "Hello, world!"
//   # Output: 101,7592,1010,2088,999,102
//
// Example JSON output:
//   iree-tokenize --tokenizer=tokenizer.json --json "Hello, world!"
//   # Output: {"ids":[101,7592,1010,2088,999,102]}
//
// Example encoding without special tokens:
//   iree-tokenize --tokenizer=tokenizer.json --special=false "Hello, world!"
//   # Output: 7592,1010,2088,999
//
// Example with offset tracking:
//   iree-tokenize --tokenizer=tokenizer.json --offsets "Hello, world!"
//   # Output: 7592[0:5],1010[5:6],2088[7:12],999[12:13]
//
// Example decoding token IDs to text:
//   iree-tokenize --tokenizer=tokenizer.json --decode
//   "101,7592,1010,2088,999,102"
//   # Output: Hello, world!
//
// Example batch mode (one line per input):
//   echo -e "Hello\nWorld" | iree-tokenize --tokenizer=tokenizer.json --batch
//   # Output: 101,7592,...
//   #         101,2088,...
//
// Example showing tokenizer info (always JSON):
//   iree-tokenize --tokenizer=tokenizer.json --info
//   # Output: {"vocab_size":30522,"model_type":"BPE",...}
//
// Example benchmarking:
//   iree-tokenize --tokenizer=tokenizer.json --benchmark=oneshot "Hello,
//   world!" # Output: timing stats to stderr, token IDs to stdout

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/json.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/vocab/vocab.h"

//===----------------------------------------------------------------------===//
// Flags
//===----------------------------------------------------------------------===//

IREE_FLAG(bool, decode, false, "Decode mode: input is comma-separated IDs.");
IREE_FLAG(bool, decode_special, false,
          "Include special tokens (BOS/EOS) in decode output.");
IREE_FLAG(bool, special, true, "Add special tokens (BOS/EOS, CLS/SEP).");
IREE_FLAG(bool, batch, false, "Batch mode: read lines from stdin.");
IREE_FLAG(bool, stream, false, "Stream stdin continuously (not line-by-line).");
IREE_FLAG(int32_t, max_length, 0, "Max output length (0 = unlimited).");
IREE_FLAG(bool, info, false, "Show tokenizer info instead of encoding.");
IREE_FLAG(bool, json, false,
          "Output JSON format (default: comma-separated IDs).");
IREE_FLAG(bool, json_string, false,
          "Input is a JSON-encoded string (handles \\uXXXX escapes).");
IREE_FLAG(string, tokenizer, "", "Path to HuggingFace tokenizer.json file.");
IREE_FLAG(bool, offsets, false, "Show token-to-byte offset mappings.");
IREE_FLAG(string, benchmark, "",
          "Benchmark mode: oneshot, batch, stream, or decode.");
IREE_FLAG(int32_t, benchmark_iterations, 100,
          "Number of timed iterations for benchmarking.");
IREE_FLAG(int32_t, benchmark_warmup, 5,
          "Number of warmup iterations before timing.");
IREE_FLAG(int32_t, benchmark_chunk_size, 4096,
          "Chunk size in bytes for stream benchmark.");
//===----------------------------------------------------------------------===//
// Output Helpers
//===----------------------------------------------------------------------===//

// Prints a token ID with optional offset annotation.
static void iree_tooling_print_token(iree_tokenizer_token_id_t token_id,
                                     const iree_tokenizer_offset_t* offset,
                                     bool first) {
  if (!first) fputc(',', stdout);
  fprintf(stdout, "%" PRId32, token_id);
  if (offset) {
    fprintf(stdout, "[%zu:%zu]", (size_t)offset->start, (size_t)offset->end);
  }
}

// Prints a token sequence with optional offsets.
static void iree_tooling_print_tokens(
    const iree_tokenizer_token_id_t* token_ids,
    const iree_tokenizer_offset_t* offsets, iree_host_size_t count,
    bool* first) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_tooling_print_token(token_ids[i], offsets ? &offsets[i] : NULL,
                             *first);
    *first = false;
  }
}

// Prints a JSON array of token IDs with optional offsets.
static void iree_tooling_print_json_tokens(
    const iree_tokenizer_token_id_t* token_ids,
    const iree_tokenizer_offset_t* offsets, iree_host_size_t count) {
  fputs("{\"ids\":[", stdout);
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (i > 0) fputc(',', stdout);
    fprintf(stdout, "%" PRId32, token_ids[i]);
  }
  fputc(']', stdout);
  if (offsets) {
    fputs(",\"offsets\":[", stdout);
    for (iree_host_size_t i = 0; i < count; ++i) {
      if (i > 0) fputc(',', stdout);
      fprintf(stdout, "[%zu,%zu]", (size_t)offsets[i].start,
              (size_t)offsets[i].end);
    }
    fputc(']', stdout);
  }
  fputs("}\n", stdout);
}

// Writes decoded text to stdout with JSON escaping if needed.
static void iree_tooling_print_text(const char* data, iree_host_size_t length,
                                    bool json_escape) {
  if (!json_escape) {
    fwrite(data, 1, length, stdout);
    return;
  }
  for (iree_host_size_t i = 0; i < length; ++i) {
    char c = data[i];
    switch (c) {
      case '"':
        fputs("\\\"", stdout);
        break;
      case '\\':
        fputs("\\\\", stdout);
        break;
      case '\b':
        fputs("\\b", stdout);
        break;
      case '\f':
        fputs("\\f", stdout);
        break;
      case '\n':
        fputs("\\n", stdout);
        break;
      case '\r':
        fputs("\\r", stdout);
        break;
      case '\t':
        fputs("\\t", stdout);
        break;
      default:
        if ((unsigned char)c < 0x20) {
          fprintf(stdout, "\\u%04x", (unsigned char)c);
        } else {
          fputc(c, stdout);
        }
        break;
    }
  }
}

//===----------------------------------------------------------------------===//
// Encode (one-shot with retry)
//===----------------------------------------------------------------------===//

// Builds encode flags from CLI flags.
static iree_tokenizer_encode_flags_t iree_tooling_encode_flags(void) {
  iree_tokenizer_encode_flags_t flags =
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START;
  if (FLAG_special) flags |= IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS;
  if (FLAG_offsets) flags |= IREE_TOKENIZER_ENCODE_FLAG_TRACK_OFFSETS;
  return flags;
}

static iree_status_t iree_tooling_tokenize_encode(
    const iree_tokenizer_t* tokenizer, iree_string_view_t text,
    iree_allocator_t allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tokenizer_encode_flags_t flags = iree_tooling_encode_flags();

  // Allocate combined output buffer. Use fixed capacity - the streaming encode
  // API operates in bounded memory and should never need retries.
  iree_host_size_t capacity = 8192;
  iree_host_size_t total_size = 0;
  iree_host_size_t token_ids_offset = 0;
  iree_host_size_t offsets_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              0, &total_size,
              IREE_STRUCT_FIELD(capacity, iree_tokenizer_token_id_t,
                                &token_ids_offset),
              IREE_STRUCT_FIELD(FLAG_offsets ? capacity : 0,
                                iree_tokenizer_offset_t, &offsets_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&storage));

  iree_tokenizer_token_id_t* token_ids =
      (iree_tokenizer_token_id_t*)(storage + token_ids_offset);
  iree_tokenizer_offset_t* offsets =
      FLAG_offsets ? (iree_tokenizer_offset_t*)(storage + offsets_offset)
                   : NULL;
  iree_host_size_t token_count = 0;

  iree_tokenizer_token_output_t output =
      iree_tokenizer_make_token_output(token_ids, offsets, NULL, capacity);
  iree_status_t status = iree_tokenizer_encode(tokenizer, text, flags, output,
                                               allocator, &token_count);

  if (iree_status_is_ok(status)) {
    // Apply max_length truncation.
    if (FLAG_max_length > 0 &&
        token_count > (iree_host_size_t)FLAG_max_length) {
      token_count = (iree_host_size_t)FLAG_max_length;
    }

    if (FLAG_json) {
      iree_tooling_print_json_tokens(token_ids, offsets, token_count);
    } else {
      bool first = true;
      iree_tooling_print_tokens(token_ids, offsets, token_count, &first);
      fputc('\n', stdout);
    }
  }

  iree_allocator_free(allocator, storage);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Decode (one-shot with retry)
//===----------------------------------------------------------------------===//

// Parses comma-separated IDs from a string.
static iree_status_t iree_tooling_parse_ids(iree_string_view_t text,
                                            iree_tokenizer_token_id_t* out_ids,
                                            iree_host_size_t max_ids,
                                            iree_host_size_t* out_count) {
  *out_count = 0;
  if (text.size == 0) return iree_ok_status();

  iree_host_size_t position = 0;
  while (position < text.size) {
    // Skip whitespace.
    while (position < text.size &&
           (text.data[position] == ' ' || text.data[position] == '\t')) {
      ++position;
    }
    if (position >= text.size) break;

    // Parse number.
    bool negative = false;
    if (text.data[position] == '-') {
      negative = true;
      ++position;
    }
    int32_t value = 0;
    bool found_digit = false;
    while (position < text.size && text.data[position] >= '0' &&
           text.data[position] <= '9') {
      value = value * 10 + (text.data[position] - '0');
      found_digit = true;
      ++position;
    }
    if (!found_digit) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected number at position %zu", position);
    }
    if (negative) value = -value;

    if (*out_count >= max_ids) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "too many IDs (max %zu)", max_ids);
    }
    out_ids[(*out_count)++] = value;

    // Skip comma.
    while (position < text.size &&
           (text.data[position] == ' ' || text.data[position] == '\t')) {
      ++position;
    }
    if (position < text.size && text.data[position] == ',') {
      ++position;
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_tooling_tokenize_decode(
    const iree_tokenizer_t* tokenizer, iree_string_view_t input,
    iree_allocator_t allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Parse IDs into stack buffer.
  iree_tokenizer_token_id_t ids[8192];
  iree_host_size_t id_count = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tooling_parse_ids(input, ids, IREE_ARRAYSIZE(ids), &id_count));

  iree_tokenizer_token_id_list_t tokens =
      iree_tokenizer_make_token_id_list(ids, id_count);

  // Decode with retry on RESOURCE_EXHAUSTED.
  iree_host_size_t text_capacity = 65536;
  char* text_buffer = NULL;
  iree_host_size_t text_length = 0;
  iree_status_t status = iree_ok_status();

  for (;;) {
    status =
        iree_allocator_malloc(allocator, text_capacity, (void**)&text_buffer);
    if (!iree_status_is_ok(status)) break;

    iree_mutable_string_view_t text_output = {text_buffer, text_capacity};
    iree_tokenizer_decode_flags_t decode_flags =
        FLAG_decode_special ? IREE_TOKENIZER_DECODE_FLAG_NONE
                            : IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS;
    status = iree_tokenizer_decode(tokenizer, tokens, decode_flags, text_output,
                                   allocator, &text_length);
    if (iree_status_is_resource_exhausted(status)) {
      iree_status_ignore(status);
      iree_allocator_free(allocator, text_buffer);
      text_buffer = NULL;
      text_capacity *= 2;
      continue;
    }
    break;
  }

  if (iree_status_is_ok(status)) {
    if (FLAG_json) {
      fputs("{\"text\":\"", stdout);
      iree_tooling_print_text(text_buffer, text_length, /*json_escape=*/true);
      fputs("\"}\n", stdout);
    } else {
      iree_tooling_print_text(text_buffer, text_length, /*json_escape=*/false);
      fputc('\n', stdout);
    }
  }

  iree_allocator_free(allocator, text_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Info
//===----------------------------------------------------------------------===//

static iree_status_t iree_tooling_tokenize_info(
    const iree_tokenizer_t* tokenizer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_tokenizer_vocab_t* vocab = iree_tokenizer_vocab(tokenizer);
  iree_host_size_t vocab_size = iree_tokenizer_vocab_capacity(vocab);
  iree_host_size_t merge_count = iree_tokenizer_vocab_merge_count(vocab);
  iree_tokenizer_special_ids_t special =
      iree_tokenizer_vocab_special_ids(vocab);
  iree_string_view_t model_type = iree_tokenizer_model_type_name(tokenizer);

  fprintf(stdout, "{\"vocab_size\":%zu,\"model_type\":\"%.*s\"",
          (size_t)vocab_size, (int)model_type.size, model_type.data);

  if (merge_count > 0) {
    fprintf(stdout, ",\"merge_count\":%zu", (size_t)merge_count);
  }

  // Special tokens.
  if (special.bos >= 0) fprintf(stdout, ",\"bos_id\":%" PRId32, special.bos);
  if (special.eos >= 0) fprintf(stdout, ",\"eos_id\":%" PRId32, special.eos);
  if (special.unk >= 0) fprintf(stdout, ",\"unk_id\":%" PRId32, special.unk);
  if (special.pad >= 0) fprintf(stdout, ",\"pad_id\":%" PRId32, special.pad);
  if (special.cls >= 0) fprintf(stdout, ",\"cls_id\":%" PRId32, special.cls);
  if (special.sep >= 0) fprintf(stdout, ",\"sep_id\":%" PRId32, special.sep);
  if (special.mask >= 0) fprintf(stdout, ",\"mask_id\":%" PRId32, special.mask);

  fputs("}\n", stdout);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Streaming Stdin Mode
//===----------------------------------------------------------------------===//

// Streams stdin, reading chunks and emitting tokens incrementally.
// Uses the pull-based streaming encode API which handles all boundary
// conditions:
// - Incomplete UTF-8 sequences at chunk boundaries
// - Literals (added_tokens) that span chunks
// - Transform segments that span chunks
// - BOS/EOS token emission (via postprocessor)
static iree_status_t iree_tooling_tokenize_stdin_streaming(
    const iree_tokenizer_t* tokenizer, iree_allocator_t allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tokenizer_encode_flags_t flags = iree_tooling_encode_flags();

  // Calculate state storage requirements.
  iree_host_size_t state_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));

  // Allocate combined state and transform buffer.
  iree_host_size_t transform_size =
      iree_tokenizer_transform_buffer_recommended_size(8192);
  iree_host_size_t total_size = 0;
  iree_host_size_t state_offset = 0;
  iree_host_size_t transform_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      IREE_STRUCT_LAYOUT(
          0, &total_size, IREE_STRUCT_FIELD(state_size, uint8_t, &state_offset),
          IREE_STRUCT_FIELD(transform_size, uint8_t, &transform_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&storage));

  iree_byte_span_t state_span = {storage + state_offset, state_size};
  iree_byte_span_t transform_span = {storage + transform_offset,
                                     transform_size};

  // Initialize streaming state.
  iree_tokenizer_encode_state_t* state = NULL;
  iree_status_t status = iree_tokenizer_encode_state_initialize(
      tokenizer, state_span, transform_span,
      iree_tokenizer_offset_run_list_empty(), flags, &state);

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, storage);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Token output buffer (reused each feed call).
  iree_tokenizer_token_id_t token_buffer[1024];
  iree_tokenizer_token_output_t output = iree_tokenizer_make_token_output(
      token_buffer, NULL, NULL, IREE_ARRAYSIZE(token_buffer));

  // Start output.
  if (FLAG_json) fputs("{\"ids\":[", stdout);
  bool first_token = true;

  // Read and feed chunks until EOF.
  char read_buffer[8192];
  size_t bytes_read;
  while (iree_status_is_ok(status) &&
         (bytes_read = fread(read_buffer, 1, sizeof(read_buffer), stdin)) > 0) {
    iree_string_view_t chunk = iree_make_string_view(read_buffer, bytes_read);
    while (chunk.size > 0 && iree_status_is_ok(status)) {
      iree_host_size_t bytes_consumed = 0;
      iree_host_size_t token_count = 0;
      status = iree_tokenizer_encode_state_feed(state, chunk, output,
                                                &bytes_consumed, &token_count);
      if (iree_status_is_ok(status)) {
        iree_tooling_print_tokens(token_buffer, NULL, token_count,
                                  &first_token);
        chunk.data += bytes_consumed;
        chunk.size -= bytes_consumed;
      }
    }
  }

  // Finalize: flush any pending state.
  if (iree_status_is_ok(status)) {
    iree_host_size_t token_count = 0;
    status = iree_tokenizer_encode_state_finalize(state, output, &token_count);
    if (iree_status_is_ok(status)) {
      iree_tooling_print_tokens(token_buffer, NULL, token_count, &first_token);
    }
  }

  // Close output.
  if (FLAG_json) {
    fputs("]}\n", stdout);
  } else {
    fputc('\n', stdout);
  }

  iree_tokenizer_encode_state_deinitialize(state);
  iree_allocator_free(allocator, storage);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Batch Mode (line-by-line)
//===----------------------------------------------------------------------===//

// Strips the trailing LF line terminator from a string view.
// Only strips \n - does NOT strip \r, which could be content.
// The batch protocol uses \n as delimiter (via Python's "\n".join()), so any
// \r before the \n is content that must be preserved.
static iree_string_view_t iree_tooling_string_view_strip_trailing_newline(
    iree_string_view_t text) {
  if (text.size > 0 && text.data[text.size - 1] == '\n') {
    --text.size;
  }
  return text;
}

// Portable getline implementation that dynamically grows the buffer.
// Returns the line length (excluding null terminator), or -1 on EOF/error.
// The caller must free *line_ptr using the same allocator when done.
static intptr_t iree_tooling_getline(char** line_ptr,
                                     iree_host_size_t* capacity_ptr,
                                     FILE* stream, iree_allocator_t allocator) {
  if (*line_ptr == NULL || *capacity_ptr == 0) {
    *capacity_ptr = 256;
    iree_status_t status =
        iree_allocator_malloc(allocator, *capacity_ptr, (void**)line_ptr);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return -1;
    }
  }

  iree_host_size_t position = 0;
  int character;
  while ((character = fgetc(stream)) != EOF) {
    // Grow buffer if needed (leaving room for null terminator).
    if (position + 1 >= *capacity_ptr) {
      iree_host_size_t new_capacity = *capacity_ptr * 2;
      iree_status_t status =
          iree_allocator_realloc(allocator, new_capacity, (void**)line_ptr);
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        return -1;
      }
      *capacity_ptr = new_capacity;
    }

    (*line_ptr)[position++] = (char)character;
    if (character == '\n') break;
  }

  if (position == 0 && character == EOF) return -1;

  (*line_ptr)[position] = '\0';
  return (intptr_t)position;
}

static iree_status_t iree_tooling_tokenize_batch(
    const iree_tokenizer_t* tokenizer, iree_allocator_t allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  char* line = NULL;
  iree_host_size_t line_capacity = 0;
  intptr_t line_length;

  while ((line_length = iree_tooling_getline(&line, &line_capacity, stdin,
                                             allocator)) != -1) {
    iree_string_view_t text = iree_tooling_string_view_strip_trailing_newline(
        iree_make_string_view(line, (iree_host_size_t)line_length));
    iree_status_t status;
    if (FLAG_decode) {
      status = iree_tooling_tokenize_decode(tokenizer, text, allocator);
    } else {
      status = iree_tooling_tokenize_encode(tokenizer, text, allocator);
    }
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(allocator, line);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  iree_allocator_free(allocator, line);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Benchmark Mode
//===----------------------------------------------------------------------===//

typedef struct {
  iree_time_t min_ns;
  iree_time_t max_ns;
  iree_time_t total_ns;
  int32_t iterations;
  iree_host_size_t total_input_bytes;
  iree_host_size_t total_tokens;
  iree_host_size_t peak_memory;
} iree_tooling_benchmark_stats_t;

static void iree_tooling_benchmark_stats_initialize(
    iree_tooling_benchmark_stats_t* stats) {
  memset(stats, 0, sizeof(*stats));
  stats->min_ns = INT64_MAX;
}

static void iree_tooling_benchmark_stats_record(
    iree_tooling_benchmark_stats_t* stats, iree_time_t elapsed_ns,
    iree_host_size_t input_bytes, iree_host_size_t tokens) {
  if (elapsed_ns < stats->min_ns) stats->min_ns = elapsed_ns;
  if (elapsed_ns > stats->max_ns) stats->max_ns = elapsed_ns;
  stats->total_ns += elapsed_ns;
  stats->iterations++;
  stats->total_input_bytes += input_bytes;
  stats->total_tokens += tokens;
}

static void iree_tooling_benchmark_stats_print(
    const iree_tooling_benchmark_stats_t* stats, const char* mode) {
  iree_time_t average_ns = stats->total_ns / stats->iterations;
  double tokens_per_sec =
      (double)stats->total_tokens / ((double)stats->total_ns / 1e9);
  double mb_per_sec =
      (double)stats->total_input_bytes / ((double)stats->total_ns / 1e9) / 1e6;

  if (FLAG_json) {
    fprintf(stdout,
            "{\"mode\":\"%s\",\"iterations\":%d,"
            "\"total_input_bytes\":%zu,\"total_tokens\":%zu,"
            "\"min_ns\":%" PRId64 ",\"avg_ns\":%" PRId64 ",\"max_ns\":%" PRId64
            ","
            "\"tokens_per_sec\":%.1f,\"mb_per_sec\":%.3f,"
            "\"peak_memory_bytes\":%zu}\n",
            mode, stats->iterations, (size_t)stats->total_input_bytes,
            (size_t)stats->total_tokens, stats->min_ns, average_ns,
            stats->max_ns, tokens_per_sec, mb_per_sec,
            (size_t)stats->peak_memory);
  } else {
    fprintf(stderr,
            "Benchmark: %s\n"
            "  Iterations:    %d\n"
            "  Input bytes:   %zu total\n"
            "  Tokens:        %zu total\n"
            "  Latency (ns):  min=%" PRId64 " avg=%" PRId64 " max=%" PRId64
            "\n"
            "  Throughput:    %.1f tokens/sec, %.3f MB/sec\n"
            "  Peak memory:   %zu bytes\n",
            mode, stats->iterations, (size_t)stats->total_input_bytes,
            (size_t)stats->total_tokens, stats->min_ns, average_ns,
            stats->max_ns, tokens_per_sec, mb_per_sec,
            (size_t)stats->peak_memory);
  }
}

static iree_status_t iree_tooling_benchmark_oneshot(
    const iree_tokenizer_t* tokenizer, iree_string_view_t text,
    iree_allocator_t allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tokenizer_encode_flags_t flags = iree_tooling_encode_flags();

  // Allocate output buffer sized to text length (generous).
  iree_host_size_t capacity = iree_max(text.size, (iree_host_size_t)8192);
  iree_tokenizer_token_id_t* token_ids = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator,
                                capacity * sizeof(iree_tokenizer_token_id_t),
                                (void**)&token_ids));

  iree_tokenizer_token_output_t output =
      iree_tokenizer_make_token_output(token_ids, NULL, NULL, capacity);

  iree_tooling_benchmark_stats_t stats;
  iree_tooling_benchmark_stats_initialize(&stats);
  stats.peak_memory = capacity * sizeof(iree_tokenizer_token_id_t);

  // Warmup.
  for (int32_t i = 0; i < FLAG_benchmark_warmup; ++i) {
    iree_host_size_t token_count = 0;
    iree_status_t status = iree_tokenizer_encode(tokenizer, text, flags, output,
                                                 allocator, &token_count);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(allocator, token_ids);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  // Timed iterations.
  for (int32_t i = 0; i < FLAG_benchmark_iterations; ++i) {
    iree_host_size_t token_count = 0;
    iree_time_t start = iree_time_now();
    iree_status_t status = iree_tokenizer_encode(tokenizer, text, flags, output,
                                                 allocator, &token_count);
    iree_time_t end = iree_time_now();
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(allocator, token_ids);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    iree_tooling_benchmark_stats_record(&stats, end - start, text.size,
                                        token_count);
  }

  iree_tooling_benchmark_stats_print(&stats, "oneshot");
  iree_allocator_free(allocator, token_ids);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_tooling_benchmark_stream(
    const iree_tokenizer_t* tokenizer, iree_string_view_t text,
    iree_allocator_t allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tokenizer_encode_flags_t flags = iree_tooling_encode_flags();
  iree_host_size_t chunk_size = (iree_host_size_t)FLAG_benchmark_chunk_size;

  // Allocate combined state and transform buffer.
  iree_host_size_t state_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  iree_host_size_t transform_size =
      iree_tokenizer_transform_buffer_recommended_size(chunk_size);
  iree_host_size_t total_size = 0;
  iree_host_size_t state_offset = 0;
  iree_host_size_t transform_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      IREE_STRUCT_LAYOUT(
          0, &total_size, IREE_STRUCT_FIELD(state_size, uint8_t, &state_offset),
          IREE_STRUCT_FIELD(transform_size, uint8_t, &transform_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&storage));

  // Token output buffer.
  iree_tokenizer_token_id_t token_buffer[1024];
  iree_tokenizer_token_output_t output = iree_tokenizer_make_token_output(
      token_buffer, NULL, NULL, IREE_ARRAYSIZE(token_buffer));

  iree_byte_span_t state_span = {storage + state_offset, state_size};
  iree_byte_span_t transform_span = {storage + transform_offset,
                                     transform_size};

  iree_tooling_benchmark_stats_t stats;
  iree_tooling_benchmark_stats_initialize(&stats);
  stats.peak_memory = total_size + sizeof(token_buffer);

  iree_status_t status = iree_ok_status();
  int32_t total_iterations = FLAG_benchmark_warmup + FLAG_benchmark_iterations;
  for (int32_t iteration = 0;
       iteration < total_iterations && iree_status_is_ok(status); ++iteration) {
    bool is_warmup = (iteration < FLAG_benchmark_warmup);
    iree_time_t start = iree_time_now();
    iree_host_size_t iteration_tokens = 0;

    // Initialize state for this iteration.
    iree_tokenizer_encode_state_t* state = NULL;
    status = iree_tokenizer_encode_state_initialize(
        tokenizer, state_span, transform_span,
        iree_tokenizer_offset_run_list_empty(), flags, &state);
    if (!iree_status_is_ok(status)) break;

    // Feed text in chunks.
    iree_host_size_t text_position = 0;
    while (text_position < text.size && iree_status_is_ok(status)) {
      iree_host_size_t remaining = text.size - text_position;
      iree_host_size_t this_chunk = iree_min(remaining, chunk_size);
      iree_string_view_t chunk =
          iree_make_string_view(text.data + text_position, this_chunk);
      while (chunk.size > 0 && iree_status_is_ok(status)) {
        iree_host_size_t bytes_consumed = 0;
        iree_host_size_t token_count = 0;
        status = iree_tokenizer_encode_state_feed(
            state, chunk, output, &bytes_consumed, &token_count);
        if (iree_status_is_ok(status)) {
          iteration_tokens += token_count;
          chunk.data += bytes_consumed;
          chunk.size -= bytes_consumed;
        }
      }
      text_position += this_chunk;
    }

    // Finalize.
    if (iree_status_is_ok(status)) {
      iree_host_size_t token_count = 0;
      status =
          iree_tokenizer_encode_state_finalize(state, output, &token_count);
      if (iree_status_is_ok(status)) {
        iteration_tokens += token_count;
      }
    }

    iree_tokenizer_encode_state_deinitialize(state);

    if (iree_status_is_ok(status) && !is_warmup) {
      iree_time_t end = iree_time_now();
      iree_tooling_benchmark_stats_record(&stats, end - start, text.size,
                                          iteration_tokens);
    }
  }

  if (iree_status_is_ok(status)) {
    iree_tooling_benchmark_stats_print(&stats, "stream");
  }

  iree_allocator_free(allocator, storage);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_tooling_benchmark_decode(
    const iree_tokenizer_t* tokenizer, iree_string_view_t text,
    iree_allocator_t allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // First, encode the text to get tokens.
  iree_tokenizer_encode_flags_t flags = iree_tooling_encode_flags();
  iree_host_size_t capacity = iree_max(text.size, (iree_host_size_t)8192);
  iree_tokenizer_token_id_t* token_ids = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator,
                                capacity * sizeof(iree_tokenizer_token_id_t),
                                (void**)&token_ids));

  iree_tokenizer_token_output_t output =
      iree_tokenizer_make_token_output(token_ids, NULL, NULL, capacity);
  iree_host_size_t token_count = 0;
  iree_status_t status = iree_tokenizer_encode(tokenizer, text, flags, output,
                                               allocator, &token_count);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, token_ids);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_tokenizer_token_id_list_t tokens =
      iree_tokenizer_make_token_id_list(token_ids, token_count);

  // Allocate decode output buffer.
  iree_host_size_t text_capacity = 65536;
  char* text_buffer = NULL;
  status =
      iree_allocator_malloc(allocator, text_capacity, (void**)&text_buffer);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, token_ids);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_mutable_string_view_t text_output = {text_buffer, text_capacity};

  iree_tooling_benchmark_stats_t stats;
  iree_tooling_benchmark_stats_initialize(&stats);
  stats.peak_memory =
      capacity * sizeof(iree_tokenizer_token_id_t) + text_capacity;

  // Decode flags for benchmark iterations.
  iree_tokenizer_decode_flags_t decode_flags =
      FLAG_decode_special ? IREE_TOKENIZER_DECODE_FLAG_NONE
                          : IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS;

  // Warmup.
  for (int32_t i = 0; i < FLAG_benchmark_warmup && iree_status_is_ok(status);
       ++i) {
    iree_host_size_t text_length = 0;
    status = iree_tokenizer_decode(tokenizer, tokens, decode_flags, text_output,
                                   allocator, &text_length);
  }

  // Timed iterations.
  for (int32_t i = 0;
       i < FLAG_benchmark_iterations && iree_status_is_ok(status); ++i) {
    iree_host_size_t text_length = 0;
    iree_time_t start = iree_time_now();
    status = iree_tokenizer_decode(tokenizer, tokens, decode_flags, text_output,
                                   allocator, &text_length);
    iree_time_t end = iree_time_now();
    if (iree_status_is_ok(status)) {
      iree_tooling_benchmark_stats_record(&stats, end - start, text_length,
                                          token_count);
    }
  }

  if (iree_status_is_ok(status)) {
    iree_tooling_benchmark_stats_print(&stats, "decode");
  }

  iree_allocator_free(allocator, text_buffer);
  iree_allocator_free(allocator, token_ids);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_tooling_tokenize_benchmark(
    const iree_tokenizer_t* tokenizer, iree_string_view_t text,
    iree_allocator_t allocator) {
  iree_string_view_t mode = iree_make_cstring_view(FLAG_benchmark);

  if (iree_string_view_equal(mode, IREE_SV("oneshot"))) {
    return iree_tooling_benchmark_oneshot(tokenizer, text, allocator);
  } else if (iree_string_view_equal(mode, IREE_SV("stream"))) {
    return iree_tooling_benchmark_stream(tokenizer, text, allocator);
  } else if (iree_string_view_equal(mode, IREE_SV("decode"))) {
    return iree_tooling_benchmark_decode(tokenizer, text, allocator);
  }

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown benchmark mode '%.*s' "
                          "(expected: oneshot, stream, decode)",
                          (int)mode.size, mode.data);
}

//===----------------------------------------------------------------------===//
// JSON String Processing
//===----------------------------------------------------------------------===//

// Decodes a JSON-escaped string into raw UTF-8 bytes.
static iree_status_t iree_tooling_decode_json_string(
    iree_string_view_t raw_input, iree_allocator_t allocator, char** out_buffer,
    iree_string_view_t* out_text) {
  // Strip surrounding quotes if present.
  iree_string_view_t escaped = raw_input;
  if (escaped.size >= 2 && escaped.data[0] == '"' &&
      escaped.data[escaped.size - 1] == '"') {
    escaped = iree_string_view_substr(escaped, 1, escaped.size - 2);
  }

  // First pass: compute required size.
  iree_host_size_t decoded_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_json_unescape_string(escaped, 0, NULL, &decoded_length));

  // Allocate and decode.
  char* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, decoded_length + 1, (void**)&buffer));
  iree_status_t status = iree_json_unescape_string(escaped, decoded_length + 1,
                                                   buffer, &decoded_length);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, buffer);
    return status;
  }
  buffer[decoded_length] = '\0';

  *out_buffer = buffer;
  *out_text = iree_make_string_view(buffer, decoded_length);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = iree_allocator_system();
  int exit_code = EXIT_SUCCESS;

  iree_flags_set_usage(
      "iree-tokenize",
      "Tokenizes text using HuggingFace tokenizer.json files.\n"
      "Outputs comma-separated token IDs (use --json for JSON format).\n"
      "\n"
      "Usage:\n"
      "  iree-tokenize --tokenizer=<file> [flags] <text>\n"
      "\n"
      "Examples:\n"
      "\n"
      "  Encode text to token IDs (default: comma-separated):\n"
      "    iree-tokenize --tokenizer=tokenizer.json \"hello, world!\"\n"
      "    101,7592,1010,2088,999,102\n"
      "\n"
      "  JSON output:\n"
      "    iree-tokenize --tokenizer=tokenizer.json --json \"hello, world!\"\n"
      "    {\"ids\":[101,7592,1010,2088,999,102]}\n"
      "\n"
      "  Use with iree-run-module:\n"
      "    iree-run-module --module=model.vmfb \\\n"
      "      --input=\"6xi32=$(iree-tokenize --tokenizer=tokenizer.json "
      "'hello')\"\n"
      "\n"
      "  Encode without special tokens (no [CLS]/[SEP] or BOS/EOS):\n"
      "    iree-tokenize --tokenizer=tokenizer.json --special=false \"hello "
      "world\"\n"
      "    7592,2088\n"
      "\n"
      "  Show token-to-byte offset mappings:\n"
      "    iree-tokenize --tokenizer=tokenizer.json --offsets \"hello world\"\n"
      "    7592[0:5],2088[6:11]\n"
      "\n"
      "  Decode token IDs back to text:\n"
      "    iree-tokenize --tokenizer=tokenizer.json --decode "
      "\"101,7592,2088,102\"\n"
      "    [CLS]helloworld[SEP]\n"
      "\n"
      "  Show tokenizer info (always JSON):\n"
      "    iree-tokenize --tokenizer=tokenizer.json --info\n"
      "    {\"vocab_size\":30522,\"model_type\":\"BPE\",\"unk_id\":100,"
      "\"cls_id\":101,\"sep_id\":102}\n"
      "\n"
      "  Batch mode - encode one line per input from stdin:\n"
      "    echo -e \"hello\\nworld\" | iree-tokenize "
      "--tokenizer=tokenizer.json --batch\n"
      "    101,7592,102\n"
      "    101,2088,102\n"
      "\n"
      "  Stream mode - continuous stdin encoding (no line buffering):\n"
      "    cat large_file.txt | iree-tokenize --tokenizer=tokenizer.json "
      "--stream\n"
      "    101,7592,...\n"
      "\n"
      "  Truncate output to max length:\n"
      "    iree-tokenize --tokenizer=tokenizer.json --max_length=5 \"hello "
      "world foo\"\n"
      "    101,7592,2088,29379,102\n"
      "\n"
      "  Benchmark encode throughput:\n"
      "    iree-tokenize --tokenizer=tokenizer.json --benchmark=oneshot "
      "\"hello world\"\n"
      "\n"
      "  JSON output with jq:\n"
      "    iree-tokenize --tokenizer=tokenizer.json --json \"hello\" | jq "
      "'.ids'\n"
      "    [101,7592,102]\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  if (FLAG_tokenizer[0] == '\0') {
    fprintf(stderr,
            "Error: missing --tokenizer=<file> flag\n"
            "Usage: iree-tokenize --tokenizer=<file> [flags] <text>\n"
            "Run with --help for more information.\n");
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(EXIT_FAILURE);
    return EXIT_FAILURE;
  }

  const char* tokenizer_path = FLAG_tokenizer;

  // Load tokenizer.json file.
  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status = iree_io_file_contents_map(
      iree_make_cstring_view(tokenizer_path), IREE_IO_FILE_ACCESS_READ,
      host_allocator, &file_contents);

  // Create tokenizer.
  iree_tokenizer_t* tokenizer = NULL;
  if (iree_status_is_ok(status)) {
    iree_host_size_t json_length =
        strnlen((const char*)file_contents->const_buffer.data,
                file_contents->const_buffer.data_length);
    iree_string_view_t json = iree_make_string_view(
        (const char*)file_contents->const_buffer.data, json_length);
    status =
        iree_tokenizer_from_huggingface_json(json, host_allocator, &tokenizer);
  }

  // Validate flag combinations.
  if (iree_status_is_ok(status) && FLAG_json_string) {
    if (FLAG_batch || FLAG_stream) {
      fprintf(stderr,
              "Error: --json_string is not supported with --batch/--stream\n"
              "(file/stdin input preserves UTF-8; use --json_string only for "
              "command-line arguments)\n");
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--json_string requires single input mode");
    } else if (FLAG_decode) {
      fprintf(stderr,
              "Error: --json_string is not supported with --decode\n"
              "(decode takes numeric IDs, not text)\n");
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--json_string requires encode mode");
    }
  }

  // Process based on mode.
  if (iree_status_is_ok(status)) {
    if (FLAG_info) {
      status = iree_tooling_tokenize_info(tokenizer);
    } else if (FLAG_benchmark[0] != '\0') {
      // Benchmark mode requires text input.
      if (argc < 2) {
        fprintf(stderr, "Error: --benchmark requires input text argument\n");
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "--benchmark requires input text");
      } else {
        iree_string_view_t input = iree_make_cstring_view(argv[1]);
        char* decoded_buffer = NULL;
        if (FLAG_json_string) {
          status = iree_tooling_decode_json_string(input, host_allocator,
                                                   &decoded_buffer, &input);
        }
        if (iree_status_is_ok(status)) {
          status =
              iree_tooling_tokenize_benchmark(tokenizer, input, host_allocator);
        }
        iree_allocator_free(host_allocator, decoded_buffer);
      }
    } else if (FLAG_stream) {
      if (FLAG_decode) {
        fprintf(stderr, "Error: --stream is not supported with --decode\n");
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "--stream requires encode mode");
      } else {
        status =
            iree_tooling_tokenize_stdin_streaming(tokenizer, host_allocator);
      }
    } else if (FLAG_batch) {
      status = iree_tooling_tokenize_batch(tokenizer, host_allocator);
    } else if (argc < 2) {
      fprintf(stderr,
              "Error: missing input text\n"
              "Usage: iree-tokenize --tokenizer=<file> [flags] <text>\n");
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing input");
    } else {
      iree_string_view_t input = iree_make_cstring_view(argv[1]);
      char* decoded_buffer = NULL;
      if (FLAG_json_string) {
        status = iree_tooling_decode_json_string(input, host_allocator,
                                                 &decoded_buffer, &input);
      }
      if (iree_status_is_ok(status)) {
        if (FLAG_decode) {
          status =
              iree_tooling_tokenize_decode(tokenizer, input, host_allocator);
        } else {
          status =
              iree_tooling_tokenize_encode(tokenizer, input, host_allocator);
        }
      }
      iree_allocator_free(host_allocator, decoded_buffer);
    }
  }

  // Cleanup.
  if (tokenizer) iree_tokenizer_free(tokenizer);
  if (file_contents) iree_io_file_contents_free(file_contents);

  fflush(stdout);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }
  fflush(stderr);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
