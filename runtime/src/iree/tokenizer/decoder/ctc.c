// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder/ctc.h"

#include <stdbool.h>
#include <string.h>

//===----------------------------------------------------------------------===//
// CTC Decoder Implementation
//===----------------------------------------------------------------------===//

// Cleanup rules from HuggingFace tokenizers/src/decoders/wordpiece.rs.
// Applied in order. All rules are shrinking or equal-size.
typedef struct iree_tokenizer_decoder_ctc_cleanup_rule_t {
  const char* pattern;
  iree_host_size_t pattern_length;
  const char* replacement;
  iree_host_size_t replacement_length;
} iree_tokenizer_decoder_ctc_cleanup_rule_t;

static const iree_tokenizer_decoder_ctc_cleanup_rule_t kCleanupRules[] = {
    {" .", 2, ".", 1},     {" ?", 2, "?", 1},           {" !", 2, "!", 1},
    {" ,", 2, ",", 1},     {" ' ", 3, "'", 1},          {" n't", 4, "n't", 3},
    {" 'm", 3, "'m", 2},   {" do not", 7, " don't", 6}, {" 's", 3, "'s", 2},
    {" 've", 4, "'ve", 3}, {" 're", 4, "'re", 3},
};
static const iree_host_size_t kCleanupRuleCount =
    sizeof(kCleanupRules) / sizeof(kCleanupRules[0]);

typedef struct iree_tokenizer_decoder_ctc_t {
  iree_tokenizer_decoder_t base;
  iree_allocator_t allocator;
  uint8_t pad_token[IREE_TOKENIZER_DECODER_CTC_MAX_TOKEN_SIZE];
  iree_host_size_t pad_token_length;
  uint8_t word_delimiter[IREE_TOKENIZER_DECODER_CTC_MAX_TOKEN_SIZE];
  iree_host_size_t word_delimiter_length;
  bool cleanup;
} iree_tokenizer_decoder_ctc_t;

typedef struct iree_tokenizer_decoder_ctc_state_t {
  iree_tokenizer_decoder_state_t base;

  // Previous token for deduplication (raw, before filter_map).
  uint8_t prev_token[IREE_TOKENIZER_DECODER_CTC_MAX_TOKEN_SIZE];
  // 0 = none (start or after pad).
  iree_host_size_t prev_token_length;

  // Pending output (transformed token waiting to be written).
  uint8_t pending_output[IREE_TOKENIZER_DECODER_CTC_MAX_TOKEN_SIZE];
  iree_host_size_t pending_length;
  iree_host_size_t pending_written;

  // Cached config for hot-path (avoid pointer chase to decoder).
  uint8_t pad_token[IREE_TOKENIZER_DECODER_CTC_MAX_TOKEN_SIZE];
  iree_host_size_t pad_token_length;
  uint8_t word_delimiter[IREE_TOKENIZER_DECODER_CTC_MAX_TOKEN_SIZE];
  iree_host_size_t word_delimiter_length;
  bool cleanup;
} iree_tokenizer_decoder_ctc_state_t;

static const iree_tokenizer_decoder_vtable_t iree_tokenizer_decoder_ctc_vtable;

//===----------------------------------------------------------------------===//
// Internal Helpers
//===----------------------------------------------------------------------===//

// Applies a single pattern replacement in-place.
// Returns the new length after replacement.
static iree_host_size_t iree_tokenizer_decoder_ctc_replace_one(
    uint8_t* data, iree_host_size_t length, const char* pattern,
    iree_host_size_t pattern_length, const char* replacement,
    iree_host_size_t replacement_length) {
  if (pattern_length > length) return length;

  iree_host_size_t read_pos = 0;
  iree_host_size_t write_pos = 0;

  while (read_pos <= length - pattern_length) {
    if (memcmp(data + read_pos, pattern, pattern_length) == 0) {
      // Always copy replacement - content may differ even if length matches.
      memcpy(data + write_pos, replacement, replacement_length);
      write_pos += replacement_length;
      read_pos += pattern_length;
    } else {
      if (write_pos != read_pos) {
        data[write_pos] = data[read_pos];
      }
      write_pos++;
      read_pos++;
    }
  }

  while (read_pos < length) {
    if (write_pos != read_pos) {
      data[write_pos] = data[read_pos];
    }
    write_pos++;
    read_pos++;
  }

  return write_pos;
}

// Removes all occurrences of |substr| from |data|.
static iree_host_size_t iree_tokenizer_decoder_ctc_remove_substr(
    uint8_t* data, iree_host_size_t length, const uint8_t* substr,
    iree_host_size_t substr_length) {
  if (substr_length == 0 || substr_length > length) return length;

  iree_host_size_t read_pos = 0;
  iree_host_size_t write_pos = 0;

  while (read_pos <= length - substr_length) {
    if (memcmp(data + read_pos, substr, substr_length) == 0) {
      read_pos += substr_length;
    } else {
      if (write_pos != read_pos) {
        data[write_pos] = data[read_pos];
      }
      write_pos++;
      read_pos++;
    }
  }

  while (read_pos < length) {
    if (write_pos != read_pos) {
      data[write_pos] = data[read_pos];
    }
    write_pos++;
    read_pos++;
  }

  return write_pos;
}

// Applies the HF CTC filter_map transformation to a token.
// Returns the transformed length (may be 0 if token became empty).
static iree_host_size_t iree_tokenizer_decoder_ctc_filter_map(
    const iree_tokenizer_decoder_ctc_state_t* state, const uint8_t* token,
    iree_host_size_t token_length, uint8_t* output) {
  memcpy(output, token, token_length);
  iree_host_size_t length = token_length;

  // Step 1: Remove pad_token substring.
  length = iree_tokenizer_decoder_ctc_remove_substr(
      output, length, state->pad_token, state->pad_token_length);

  if (length == 0) return 0;
  if (!state->cleanup) return length;

  // Step 2a: Apply wordpiece cleanup rules.
  for (iree_host_size_t i = 0; i < kCleanupRuleCount; ++i) {
    const iree_tokenizer_decoder_ctc_cleanup_rule_t* rule = &kCleanupRules[i];
    length = iree_tokenizer_decoder_ctc_replace_one(
        output, length, rule->pattern, rule->pattern_length, rule->replacement,
        rule->replacement_length);
  }

  // Step 2b: Replace word delimiter with space.
  length = iree_tokenizer_decoder_ctc_replace_one(
      output, length, (const char*)state->word_delimiter,
      state->word_delimiter_length, " ", 1);

  return length;
}

// Tries to write |data| to |output|. Updates output in-place.
// Returns bytes written.
static iree_host_size_t iree_tokenizer_decoder_ctc_try_write(
    const uint8_t* data, iree_host_size_t length,
    iree_mutable_string_view_t* output) {
  iree_host_size_t to_write = length < output->size ? length : output->size;
  if (to_write > 0) {
    memmove(output->data, data, to_write);
    output->data += to_write;
    output->size -= to_write;
  }
  return to_write;
}

// Drains pending output. Returns bytes written.
static iree_host_size_t iree_tokenizer_decoder_ctc_drain_pending(
    iree_tokenizer_decoder_ctc_state_t* state,
    iree_mutable_string_view_t* output) {
  iree_host_size_t remaining = state->pending_length - state->pending_written;
  iree_host_size_t written = iree_tokenizer_decoder_ctc_try_write(
      state->pending_output + state->pending_written, remaining, output);
  state->pending_written += written;
  if (state->pending_written >= state->pending_length) {
    state->pending_length = 0;
    state->pending_written = 0;
  }
  return written;
}

// Transforms prev_token via filter_map and writes to output.
// Buffers remainder in pending if output is too small.
// Clears prev_token if fully written. Returns bytes written.
static iree_host_size_t iree_tokenizer_decoder_ctc_emit_prev_token(
    iree_tokenizer_decoder_ctc_state_t* state,
    iree_mutable_string_view_t* output) {
  if (state->prev_token_length == 0) return 0;

  uint8_t transformed[IREE_TOKENIZER_DECODER_CTC_MAX_TOKEN_SIZE];
  iree_host_size_t transformed_length = iree_tokenizer_decoder_ctc_filter_map(
      state, state->prev_token, state->prev_token_length, transformed);

  if (transformed_length == 0) {
    state->prev_token_length = 0;
    return 0;
  }

  iree_host_size_t written = iree_tokenizer_decoder_ctc_try_write(
      transformed, transformed_length, output);

  if (written < transformed_length) {
    // Buffer remainder.
    iree_host_size_t remainder = transformed_length - written;
    memcpy(state->pending_output, transformed + written, remainder);
    state->pending_length = remainder;
    state->pending_written = 0;
  }

  state->prev_token_length = 0;
  return written;
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_decoder_ctc_allocate(
    iree_string_view_t pad_token, iree_string_view_t word_delimiter_token,
    bool cleanup, iree_allocator_t allocator,
    iree_tokenizer_decoder_t** out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  *out_decoder = NULL;

  if (pad_token.size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pad_token cannot be empty");
  }
  if (pad_token.size > IREE_TOKENIZER_DECODER_CTC_MAX_TOKEN_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pad_token exceeds max size %" PRIhsz,
                            (iree_host_size_t)pad_token.size);
  }
  if (word_delimiter_token.size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "word_delimiter_token cannot be empty");
  }
  if (word_delimiter_token.size > IREE_TOKENIZER_DECODER_CTC_MAX_TOKEN_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "word_delimiter_token exceeds max size %" PRIhsz,
                            (iree_host_size_t)word_delimiter_token.size);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tokenizer_decoder_ctc_t* decoder = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*decoder), (void**)&decoder));

  iree_tokenizer_decoder_initialize(&decoder->base,
                                    &iree_tokenizer_decoder_ctc_vtable,
                                    sizeof(iree_tokenizer_decoder_ctc_state_t),
                                    IREE_TOKENIZER_DECODER_CAPABILITY_NONE);
  decoder->allocator = allocator;

  memcpy(decoder->pad_token, pad_token.data, pad_token.size);
  decoder->pad_token_length = pad_token.size;
  memcpy(decoder->word_delimiter, word_delimiter_token.data,
         word_delimiter_token.size);
  decoder->word_delimiter_length = word_delimiter_token.size;
  decoder->cleanup = cleanup;

  *out_decoder = &decoder->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_decoder_ctc_destroy(
    iree_tokenizer_decoder_t* decoder) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_decoder_ctc_t* self = (iree_tokenizer_decoder_ctc_t*)decoder;
  iree_allocator_t allocator = self->allocator;
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_decoder_ctc_state_initialize(
    const iree_tokenizer_decoder_t* decoder, void* storage,
    iree_tokenizer_decoder_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_tokenizer_decoder_ctc_t* self =
      (const iree_tokenizer_decoder_ctc_t*)decoder;

  iree_tokenizer_decoder_ctc_state_t* state =
      (iree_tokenizer_decoder_ctc_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.decoder = decoder;

  // Cache config for hot-path.
  memcpy(state->pad_token, self->pad_token, self->pad_token_length);
  state->pad_token_length = self->pad_token_length;
  memcpy(state->word_delimiter, self->word_delimiter,
         self->word_delimiter_length);
  state->word_delimiter_length = self->word_delimiter_length;
  state->cleanup = self->cleanup;

  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_decoder_ctc_state_deinitialize(
    iree_tokenizer_decoder_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Processing
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_decoder_ctc_state_process(
    iree_tokenizer_decoder_state_t* base_state,
    iree_tokenizer_string_list_t token_strings,
    iree_mutable_string_view_t output, iree_host_size_t* out_strings_consumed,
    iree_host_size_t* out_bytes_written) {
  iree_tokenizer_decoder_ctc_state_t* state =
      (iree_tokenizer_decoder_ctc_state_t*)base_state;

  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;

  // Drain any pending output from previous call.
  if (state->pending_length > state->pending_written) {
    bytes_written += iree_tokenizer_decoder_ctc_drain_pending(state, &output);
    if (state->pending_length > state->pending_written) {
      *out_strings_consumed = 0;
      *out_bytes_written = bytes_written;
      return iree_ok_status();
    }
  }

  for (iree_host_size_t i = 0; i < token_strings.count; ++i) {
    iree_string_view_t token = token_strings.values[i];

    if (token.size > IREE_TOKENIZER_DECODER_CTC_MAX_TOKEN_SIZE) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "token exceeds max size %" PRIhsz, token.size);
    }

    // Check for dedup.
    if (state->prev_token_length > 0 &&
        state->prev_token_length == token.size &&
        memcmp(state->prev_token, token.data, token.size) == 0) {
      strings_consumed++;
      continue;
    }

    // Different token - emit prev_token first.
    bytes_written += iree_tokenizer_decoder_ctc_emit_prev_token(state, &output);

    // Update prev_token with current.
    if (token.size == state->pad_token_length &&
        memcmp(token.data, state->pad_token, token.size) == 0) {
      // Pad token - resets dedup state.
      state->prev_token_length = 0;
    } else {
      memcpy(state->prev_token, token.data, token.size);
      state->prev_token_length = token.size;
    }

    strings_consumed++;

    // If emit buffered data in pending, stop to let caller drain.
    if (state->pending_length > 0) {
      break;
    }
  }

  *out_strings_consumed = strings_consumed;
  *out_bytes_written = bytes_written;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_decoder_ctc_state_finalize(
    iree_tokenizer_decoder_state_t* base_state,
    iree_mutable_string_view_t output, iree_host_size_t* out_written) {
  iree_tokenizer_decoder_ctc_state_t* state =
      (iree_tokenizer_decoder_ctc_state_t*)base_state;

  iree_host_size_t bytes_written = 0;

  // Drain pending output.
  if (state->pending_length > state->pending_written) {
    bytes_written += iree_tokenizer_decoder_ctc_drain_pending(state, &output);
    if (state->pending_length > state->pending_written) {
      *out_written = bytes_written;
      return iree_ok_status();
    }
  }

  // Emit final prev_token.
  bytes_written += iree_tokenizer_decoder_ctc_emit_prev_token(state, &output);

  *out_written = bytes_written;
  return iree_ok_status();
}

static bool iree_tokenizer_decoder_ctc_state_has_pending(
    const iree_tokenizer_decoder_state_t* base_state) {
  const iree_tokenizer_decoder_ctc_state_t* state =
      (const iree_tokenizer_decoder_ctc_state_t*)base_state;
  return (state->pending_length > state->pending_written) ||
         (state->prev_token_length > 0);
}

//===----------------------------------------------------------------------===//
// VTable
//===----------------------------------------------------------------------===//

static const iree_tokenizer_decoder_vtable_t iree_tokenizer_decoder_ctc_vtable =
    {
        .destroy = iree_tokenizer_decoder_ctc_destroy,
        .state_initialize = iree_tokenizer_decoder_ctc_state_initialize,
        .state_deinitialize = iree_tokenizer_decoder_ctc_state_deinitialize,
        .state_process = iree_tokenizer_decoder_ctc_state_process,
        .state_finalize = iree_tokenizer_decoder_ctc_state_finalize,
        .state_has_pending = iree_tokenizer_decoder_ctc_state_has_pending,
        .flags = IREE_TOKENIZER_DECODER_FLAG_SHRINKING,
};
