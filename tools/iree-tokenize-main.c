// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tokenizes text using HuggingFace tokenizer.json files.
//
// Example encoding text to token IDs:
//   iree-tokenize tokenizer.json "Hello, world!"
//   # Output: {"ids":[101,7592,1010,2088,999,102]}
//
// Example encoding without special tokens:
//   iree-tokenize tokenizer.json --no_special "Hello, world!"
//   # Output: {"ids":[7592,1010,2088,999]}
//
// Example decoding token IDs to text:
//   iree-tokenize tokenizer.json --decode "101,7592,1010,2088,999,102"
//   # Output: {"text":"Hello, world!"}
//
// Example batch mode (one line per input):
//   echo -e "Hello\nWorld" | iree-tokenize tokenizer.json --batch
//   # Output: {"ids":[...]}
//   #         {"ids":[...]}
//
// Example showing tokenizer info:
//   iree-tokenize tokenizer.json --info
//   # Output: {"vocab_size":30522,"model_type":"BPE",...}
//
// Example raw output for iree-run-module integration:
//   iree-tokenize tokenizer.json --raw "Hello, world!"
//   # Output: 101,7592,1010,2088,999,102

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/base/internal/json.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"

//===----------------------------------------------------------------------===//
// Flags
//===----------------------------------------------------------------------===//

IREE_FLAG(bool, decode, false, "Decode mode: input is comma-separated IDs.");
IREE_FLAG(bool, no_special, false, "Don't add special tokens (BOS/EOS).");
IREE_FLAG(bool, batch, false, "Batch mode: read lines from stdin.");
IREE_FLAG(bool, stream, false, "Stream stdin continuously (not line-by-line).");
IREE_FLAG(int32_t, max_length, 0, "Max output length (0 = unlimited).");
IREE_FLAG(bool, info, false, "Show tokenizer info instead of encoding.");
IREE_FLAG(bool, raw, false,
          "Output comma-separated IDs only (no JSON), for iree-run-module.");
IREE_FLAG(bool, json_string, false,
          "Input is a JSON-encoded string (handles \\uXXXX escapes).");

//===----------------------------------------------------------------------===//
// Input Processing Flags
//===----------------------------------------------------------------------===//

typedef uint32_t iree_tooling_encode_flags_t;
enum iree_tooling_encode_flag_bits_t {
  IREE_TOOLING_ENCODE_FLAG_DEFAULT = 0,
  // Input is a JSON-encoded string with \uXXXX escapes.
  IREE_TOOLING_ENCODE_FLAG_JSON_STRING = 1u << 0,
};

//===----------------------------------------------------------------------===//
// JSON Output Helpers
//===----------------------------------------------------------------------===//

// Prints a JSON array of int32 IDs.
static void iree_tooling_print_json_ids(const int32_t* ids,
                                        iree_host_size_t count) {
  fputc('[', stdout);
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (i > 0) fputc(',', stdout);
    fprintf(stdout, "%" PRId32, ids[i]);
  }
  fputc(']', stdout);
}

//===----------------------------------------------------------------------===//
// Encode (streaming)
//===----------------------------------------------------------------------===//

// Context for streaming encode output.
typedef struct {
  bool first_token;
} iree_tooling_encode_context_t;

// Callback that prints token IDs as they arrive.
static iree_status_t iree_tooling_encode_callback(
    void* user_data, iree_tokenizer_id_list_t ids) {
  iree_tooling_encode_context_t* ctx = user_data;
  for (iree_host_size_t i = 0; i < ids.count; ++i) {
    if (!ctx->first_token) fputc(',', stdout);
    ctx->first_token = false;
    fprintf(stdout, "%" PRId32, ids.values[i]);
  }
  return iree_ok_status();
}

static iree_status_t iree_tooling_tokenize_encode(iree_tokenizer_t* tokenizer,
                                                  iree_string_view_t text) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tokenizer_encode_flags_t flags =
      FLAG_no_special ? IREE_TOKENIZER_ENCODE_FLAG_DEFAULT
                      : IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS;

  // Use buffer-based API when truncation is requested (streaming ignores it).
  if (FLAG_max_length > 0) {
    iree_tokenizer_encode_options_t options = {
        .flags = flags,
        .max_length = (iree_host_size_t)FLAG_max_length,
    };
    int32_t ids[8192];
    iree_host_size_t count = 0;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_tokenizer_encode(tokenizer, text, options, ids,
                                  IREE_ARRAYSIZE(ids), &count));
    if (FLAG_raw) {
      // Raw mode: comma-separated IDs only.
      for (iree_host_size_t i = 0; i < count; ++i) {
        if (i > 0) fputc(',', stdout);
        fprintf(stdout, "%" PRId32, ids[i]);
      }
      fputc('\n', stdout);
    } else {
      fputs("{\"ids\":", stdout);
      iree_tooling_print_json_ids(ids, count);
      fputs("}\n", stdout);
    }
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Stream tokens directly to stdout (no output buffer limit).
  if (!FLAG_raw) fputs("{\"ids\":[", stdout);
  iree_tooling_encode_context_t ctx = {.first_token = true};
  iree_status_t status = iree_tokenizer_encode_streaming(
      tokenizer, text, flags, iree_tooling_encode_callback, &ctx);
  if (FLAG_raw) {
    fputc('\n', stdout);
  } else {
    fputs("]}\n", stdout);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Decode (streaming)
//===----------------------------------------------------------------------===//

// Parses comma-separated IDs from a string.
static iree_status_t iree_tooling_parse_ids(iree_string_view_t text,
                                            int32_t* out_ids,
                                            iree_host_size_t max_ids,
                                            iree_host_size_t* out_count) {
  *out_count = 0;
  if (text.size == 0) return iree_ok_status();

  iree_host_size_t pos = 0;
  while (pos < text.size) {
    // Skip whitespace.
    while (pos < text.size &&
           (text.data[pos] == ' ' || text.data[pos] == '\t')) {
      ++pos;
    }
    if (pos >= text.size) break;

    // Parse number.
    bool negative = false;
    if (text.data[pos] == '-') {
      negative = true;
      ++pos;
    }
    int32_t value = 0;
    bool found_digit = false;
    while (pos < text.size && text.data[pos] >= '0' && text.data[pos] <= '9') {
      value = value * 10 + (text.data[pos] - '0');
      found_digit = true;
      ++pos;
    }
    if (!found_digit) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected number at position %zu", pos);
    }
    if (negative) value = -value;

    if (*out_count >= max_ids) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "too many IDs (max %zu)", max_ids);
    }
    out_ids[(*out_count)++] = value;

    // Skip comma.
    while (pos < text.size &&
           (text.data[pos] == ' ' || text.data[pos] == '\t')) {
      ++pos;
    }
    if (pos < text.size && text.data[pos] == ',') {
      ++pos;
    }
  }
  return iree_ok_status();
}

// Callback that prints decoded text as JSON-escaped strings.
static iree_status_t iree_tooling_decode_callback(
    void* user_data, iree_string_view_list_t strings) {
  (void)user_data;
  for (iree_host_size_t i = 0; i < strings.count; ++i) {
    iree_string_view_t text = strings.values[i];
    for (iree_host_size_t j = 0; j < text.size; ++j) {
      char c = text.data[j];
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
  return iree_ok_status();
}

static iree_status_t iree_tooling_tokenize_decode(iree_tokenizer_t* tokenizer,
                                                  iree_string_view_t input) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Parse IDs into stack buffer.
  int32_t ids[8192];
  iree_host_size_t id_count = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tooling_parse_ids(input, ids, IREE_ARRAYSIZE(ids), &id_count));

  iree_tokenizer_decode_flags_t decode_flags =
      FLAG_no_special ? IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS
                      : IREE_TOKENIZER_DECODE_FLAG_DEFAULT;

  // Start JSON output.
  fputs("{\"text\":\"", stdout);

  // Stream decoded text directly to stdout (no output buffer limit).
  iree_status_t status =
      iree_tokenizer_decode_streaming(tokenizer, ids, id_count, decode_flags,
                                      iree_tooling_decode_callback, NULL);

  // Close JSON.
  fputs("\"}\n", stdout);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Info
//===----------------------------------------------------------------------===//

static iree_status_t iree_tooling_tokenize_info(iree_tokenizer_t* tokenizer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_tokenizer_vocab_t* vocab = iree_tokenizer_vocab(tokenizer);
  iree_host_size_t vocab_size = iree_tokenizer_vocab_capacity(vocab);
  iree_host_size_t merge_count = iree_tokenizer_vocab_merge_count(vocab);
  iree_tokenizer_special_ids_t special =
      iree_tokenizer_vocab_special_ids(vocab);

  // Determine model type.
  const char* model_type = merge_count > 0 ? "BPE" : "WordPiece";

  fprintf(stdout, "{\"vocab_size\":%zu,\"model_type\":\"%s\"",
          (size_t)vocab_size, model_type);

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
// Uses the streaming encode API which handles all boundary conditions:
// - Incomplete UTF-8 sequences at chunk boundaries
// - Literals (added_tokens) that span chunks
// - Transform segments that span chunks
// - BOS/EOS token emission
static iree_status_t iree_tooling_tokenize_stdin_streaming(
    iree_tokenizer_t* tokenizer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tokenizer_encode_flags_t flags =
      FLAG_no_special ? IREE_TOKENIZER_ENCODE_FLAG_DEFAULT
                      : IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS;

  // Start output (JSON wrapper or raw).
  if (!FLAG_raw) fputs("{\"ids\":[", stdout);
  iree_tooling_encode_context_t ctx = {.first_token = true};

  // Initialize streaming state (~8.5KB, stack allocated).
  iree_tokenizer_encode_stream_state_t state;
  iree_tokenizer_encode_stream_initialize(&state, tokenizer, flags);

  // Read and feed chunks until EOF.
  char buffer[8192];
  size_t bytes_read;
  while ((bytes_read = fread(buffer, 1, sizeof(buffer), stdin)) > 0) {
    iree_string_view_t chunk = iree_make_string_view(buffer, bytes_read);
    iree_status_t status = iree_tokenizer_encode_stream_feed(
        &state, chunk, iree_tooling_encode_callback, &ctx);
    if (!iree_status_is_ok(status)) {
      if (FLAG_raw) {
        fputc('\n', stdout);
      } else {
        fputs("]}\n", stdout);
      }
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  // Finalize: flush any pending state and emit EOS if configured.
  iree_status_t status = iree_tokenizer_encode_stream_finalize(
      &state, iree_tooling_encode_callback, &ctx);
  if (!iree_status_is_ok(status)) {
    if (FLAG_raw) {
      fputc('\n', stdout);
    } else {
      fputs("]}\n", stdout);
    }
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Close output.
  if (FLAG_raw) {
    fputc('\n', stdout);
  } else {
    fputs("]}\n", stdout);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Batch Mode (line-by-line)
//===----------------------------------------------------------------------===//

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
    iree_tokenizer_t* tokenizer, iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  char* line = NULL;
  iree_host_size_t line_capacity = 0;
  intptr_t line_length;

  while ((line_length = iree_tooling_getline(&line, &line_capacity, stdin,
                                             host_allocator)) != -1) {
    // Remove trailing newline.
    while (line_length > 0 &&
           (line[line_length - 1] == '\n' || line[line_length - 1] == '\r')) {
      line[--line_length] = '\0';
    }

    iree_string_view_t text =
        iree_make_string_view(line, (iree_host_size_t)line_length);
    iree_status_t status;
    if (FLAG_decode) {
      status = iree_tooling_tokenize_decode(tokenizer, text);
    } else {
      status = iree_tooling_tokenize_encode(tokenizer, text);
    }
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(host_allocator, line);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  iree_allocator_free(host_allocator, line);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Single Input Processing
//===----------------------------------------------------------------------===//

// Processes text input for encoding, handling JSON escapes if requested.
static iree_status_t iree_tooling_process_encode(
    iree_tokenizer_t* tokenizer, iree_string_view_t raw_input,
    iree_tooling_encode_flags_t flags, iree_allocator_t allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_string_view_t input = raw_input;
  char* decoded_buffer = NULL;

  // Decode JSON escapes if requested.
  if (iree_any_bit_set(flags, IREE_TOOLING_ENCODE_FLAG_JSON_STRING)) {
    // Strip surrounding quotes if present.
    iree_string_view_t escaped = raw_input;
    if (escaped.size >= 2 && escaped.data[0] == '"' &&
        escaped.data[escaped.size - 1] == '"') {
      escaped = iree_string_view_substr(escaped, 1, escaped.size - 2);
    }

    // First pass: compute required size.
    iree_host_size_t decoded_length = 0;
    iree_status_t status =
        iree_json_unescape_string(escaped, 0, NULL, &decoded_length);
    if (!iree_status_is_ok(status)) {
      IREE_TRACE_ZONE_END(z0);
      return status;
    }

    // Allocate and decode.
    status = iree_allocator_malloc(allocator, decoded_length + 1,
                                   (void**)&decoded_buffer);
    if (!iree_status_is_ok(status)) {
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    status = iree_json_unescape_string(escaped, decoded_length + 1,
                                       decoded_buffer, &decoded_length);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(allocator, decoded_buffer);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    decoded_buffer[decoded_length] = '\0';
    input = iree_make_string_view(decoded_buffer, decoded_length);
  }

  iree_status_t status = iree_tooling_tokenize_encode(tokenizer, input);

  iree_allocator_free(allocator, decoded_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Processes comma-separated IDs for decoding back to text.
static iree_status_t iree_tooling_process_decode(iree_tokenizer_t* tokenizer,
                                                 iree_string_view_t input) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_tooling_tokenize_decode(tokenizer, input);
  IREE_TRACE_ZONE_END(z0);
  return status;
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
      "Outputs JSON for easy scripting and pipeline integration.\n"
      "\n"
      "Usage:\n"
      "  iree-tokenize <tokenizer.json> [flags] <text>\n"
      "\n"
      "Examples:\n"
      "\n"
      "  Encode text to token IDs (default mode):\n"
      "    iree-tokenize tokenizer.json \"hello, world!\"\n"
      "    {\"ids\":[101,7592,1010,2088,999,102]}\n"
      "\n"
      "  Raw output for iree-run-module integration:\n"
      "    iree-tokenize tokenizer.json --raw \"hello, world!\"\n"
      "    101,7592,1010,2088,999,102\n"
      "\n"
      "  Use with iree-run-module:\n"
      "    iree-run-module --module=model.vmfb \\\n"
      "      --input=\"6xi32=$(iree-tokenize tokenizer.json --raw 'hello')\"\n"
      "\n"
      "  Encode without special tokens (no [CLS]/[SEP] or BOS/EOS):\n"
      "    iree-tokenize tokenizer.json --no_special \"hello world\"\n"
      "    {\"ids\":[7592,2088]}\n"
      "\n"
      "  Decode token IDs back to text:\n"
      "    iree-tokenize tokenizer.json --decode \"101,7592,2088,102\"\n"
      "    {\"text\":\"[CLS]helloworld[SEP]\"}\n"
      "\n"
      "  Show tokenizer info (vocab size, model type, special tokens):\n"
      "    iree-tokenize tokenizer.json --info\n"
      "    {\"vocab_size\":30522,\"model_type\":\"WordPiece\",\"unk_id\":100,"
      "\"cls_id\":101,\"sep_id\":102}\n"
      "\n"
      "  Batch mode - encode one line per input from stdin:\n"
      "    echo -e \"hello\\nworld\" | iree-tokenize tokenizer.json --batch\n"
      "    {\"ids\":[101,7592,102]}\n"
      "    {\"ids\":[101,2088,102]}\n"
      "\n"
      "  Stream mode - continuous stdin encoding (no line buffering):\n"
      "    cat large_file.txt | iree-tokenize tokenizer.json --stream\n"
      "    {\"ids\":[...]}\n"
      "\n"
      "  Truncate output to max length:\n"
      "    iree-tokenize tokenizer.json --max_length=5 \"hello world foo\"\n"
      "    {\"ids\":[101,7592,2088,29379,102]}\n"
      "\n"
      "  Pipeline with jq to extract just the IDs:\n"
      "    iree-tokenize tokenizer.json \"hello\" | jq '.ids'\n"
      "    [101,7592,102]\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  if (argc < 2) {
    fprintf(stderr,
            "Error: missing tokenizer.json path\n"
            "Usage: iree-tokenize <tokenizer.json> [flags] <text>\n"
            "Run with --help for more information.\n");
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(EXIT_FAILURE);
    return EXIT_FAILURE;
  }

  const char* tokenizer_path = argv[1];

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
    } else if (FLAG_stream) {
      // Continuous streaming from stdin (encode only).
      if (FLAG_decode) {
        fprintf(stderr, "Error: --stream is not supported with --decode\n");
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "--stream requires encode mode");
      } else {
        status = iree_tooling_tokenize_stdin_streaming(tokenizer);
      }
    } else if (FLAG_batch) {
      status = iree_tooling_tokenize_batch(tokenizer, host_allocator);
    } else if (argc < 3) {
      fprintf(stderr,
              "Error: missing input text\n"
              "Usage: iree-tokenize <tokenizer.json> [flags] <text>\n");
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing input");
    } else {
      iree_string_view_t input = iree_make_cstring_view(argv[2]);
      if (FLAG_decode) {
        status = iree_tooling_process_decode(tokenizer, input);
      } else {
        iree_tooling_encode_flags_t flags = IREE_TOOLING_ENCODE_FLAG_DEFAULT;
        if (FLAG_json_string) {
          flags |= IREE_TOOLING_ENCODE_FLAG_JSON_STRING;
        }
        status = iree_tooling_process_encode(tokenizer, input, flags,
                                             host_allocator);
      }
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
