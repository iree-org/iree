// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/tokenizer.h"

//===----------------------------------------------------------------------===//
// Tokenizer Interface Implementation
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_tokenizer_initialize(
    iree_tokenizer_t* tokenizer, const iree_tokenizer_vtable_t* vtable,
    iree_allocator_t allocator, const iree_tokenizer_vocab_t* vocab,
    const iree_tokenizer_text_transform_t* transform,
    const iree_tokenizer_decoder_t* decoder,
    const iree_tokenizer_postprocessor_t* postprocessor) {
  tokenizer->vtable = vtable;
  tokenizer->allocator = allocator;
  tokenizer->vocab = vocab;
  if (transform) {
    tokenizer->transform = *transform;
  } else {
    iree_tokenizer_text_transform_initialize_none(&tokenizer->transform);
  }
  if (decoder) {
    tokenizer->decoder = *decoder;
  } else {
    iree_tokenizer_decoder_initialize_none(&tokenizer->decoder);
  }
  if (postprocessor) {
    tokenizer->postprocessor = *postprocessor;
  } else {
    iree_tokenizer_postprocessor_initialize_none(&tokenizer->postprocessor);
  }
  // Literals are initialized empty; populated by algorithm-specific factories.
  iree_tokenizer_literals_initialize(allocator, &tokenizer->literals);
}

IREE_API_EXPORT void iree_tokenizer_free(iree_tokenizer_t* tokenizer) {
  if (!tokenizer) return;
  // Call the algorithm-specific destroy first (may need base resources).
  if (tokenizer->vtable && tokenizer->vtable->destroy) {
    tokenizer->vtable->destroy(tokenizer);
  }
  // Deinitialize the transform (may free heap allocations for Sequence).
  iree_tokenizer_text_transform_deinitialize(&tokenizer->transform);
  // Deinitialize the decoder (currently a no-op, but future-proofs cleanup).
  iree_tokenizer_decoder_deinitialize(&tokenizer->decoder);
  // Deinitialize the postprocessor (may free template arrays, sequence
  // children).
  iree_tokenizer_postprocessor_deinitialize(&tokenizer->postprocessor);
  // Deinitialize literals (frees entries, string storage, match order).
  iree_tokenizer_literals_deinitialize(&tokenizer->literals);
  // Free the vocab (base owns it).
  if (tokenizer->vocab) {
    iree_tokenizer_vocab_free((iree_tokenizer_vocab_t*)tokenizer->vocab);
  }
  // Free the tokenizer struct itself.
  iree_allocator_free(tokenizer->allocator, tokenizer);
}

IREE_API_EXPORT const iree_tokenizer_vocab_t* iree_tokenizer_vocab(
    const iree_tokenizer_t* tokenizer) {
  return tokenizer ? tokenizer->vocab : NULL;
}

//===----------------------------------------------------------------------===//
// Streaming Encode (one-shot wrapper)
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_tokenizer_encode_streaming(
    const iree_tokenizer_t* tokenizer, iree_string_view_t text,
    iree_tokenizer_encode_flags_t flags,
    iree_tokenizer_token_callback_fn_t callback, void* user_data) {
  IREE_ASSERT_ARGUMENT(tokenizer);
  IREE_ASSERT_ARGUMENT(callback);

  // Use the chunk-based streaming API for consistent behavior between
  // one-shot and incremental encoding. The stream API handles:
  // - BOS/EOS emission based on postprocessor configuration
  // - Literal token interception
  // - Transform segmentation
  // - Token batching and callback emission
  iree_tokenizer_encode_stream_state_t state;
  iree_tokenizer_encode_stream_initialize(&state, tokenizer, flags);
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_encode_stream_feed(&state, text, callback, user_data));
  return iree_tokenizer_encode_stream_finalize(&state, callback, user_data);
}

//===----------------------------------------------------------------------===//
// Buffer-based Encode (wrapper around streaming)
//===----------------------------------------------------------------------===//

// Context for buffer-based encoding callback.
typedef struct {
  int32_t* out_ids;
  iree_host_size_t max_ids;
  iree_host_size_t total_tokens;
} iree_tokenizer_encode_buffer_context_t;

// Callback that collects tokens into a buffer.
static iree_status_t iree_tokenizer_encode_buffer_callback(
    void* user_data, iree_tokenizer_id_list_t ids) {
  iree_tokenizer_encode_buffer_context_t* ctx = user_data;
  if (ctx->total_tokens + ids.count > ctx->max_ids) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "output buffer too small for tokens");
  }
  memcpy(ctx->out_ids + ctx->total_tokens, ids.values,
         ids.count * sizeof(int32_t));
  ctx->total_tokens += ids.count;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_tokenizer_encode(
    const iree_tokenizer_t* tokenizer, iree_string_view_t text,
    iree_tokenizer_encode_options_t options, int32_t* out_ids,
    iree_host_size_t max_ids, iree_host_size_t* out_count) {
  IREE_ASSERT_ARGUMENT(tokenizer);
  IREE_ASSERT_ARGUMENT(out_ids);
  IREE_ASSERT_ARGUMENT(out_count);
  *out_count = 0;

  // Use streaming encode with buffer collection callback.
  iree_tokenizer_encode_buffer_context_t ctx = {
      .out_ids = out_ids,
      .max_ids = max_ids,
      .total_tokens = 0,
  };
  IREE_RETURN_IF_ERROR(iree_tokenizer_encode_streaming(
      tokenizer, text, options.flags, iree_tokenizer_encode_buffer_callback,
      &ctx));

  // Apply truncation if requested (post-hoc on collected tokens).
  iree_tokenizer_truncate(out_ids, ctx.total_tokens, options.max_length,
                          options.flags, out_count);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Streaming Decode
//===----------------------------------------------------------------------===//

// Context for streaming decode - collects decoded text into bounded buffer,
// applies inverse transform, and emits chunks via callback.
typedef struct {
  const iree_tokenizer_text_transform_t* transform;
  bool skip_transform_decode;  // True when decoder already handles inverse.
  iree_tokenizer_string_callback_fn_t callback;
  void* user_data;
  char buffer[IREE_TOKENIZER_DATA_BATCH_CAPACITY];
  iree_host_size_t position;
} iree_tokenizer_decode_streaming_context_t;

// Flushes the decode buffer - applies inverse transform and emits via callback.
static iree_status_t iree_tokenizer_decode_streaming_flush(
    iree_tokenizer_decode_streaming_context_t* ctx) {
  if (ctx->position == 0) return iree_ok_status();

  iree_host_size_t out_size = ctx->position;
  if (!ctx->skip_transform_decode) {
    // Apply inverse transform in-place.
    iree_string_view_t input =
        iree_make_string_view(ctx->buffer, ctx->position);
    IREE_RETURN_IF_ERROR(iree_tokenizer_text_transform_decode(
        ctx->transform, input, ctx->buffer, sizeof(ctx->buffer), &out_size));
  }

  // Emit the transformed text.
  if (out_size > 0) {
    iree_string_view_t output = iree_make_string_view(ctx->buffer, out_size);
    iree_string_view_list_t list = {.count = 1, .values = &output};
    IREE_RETURN_IF_ERROR(ctx->callback(ctx->user_data, list));
  }

  ctx->position = 0;
  return iree_ok_status();
}

// Callback that accumulates decoded text, flushing when buffer fills.
static iree_status_t iree_tokenizer_decode_streaming_callback(
    void* user_data, iree_string_view_list_t strings) {
  iree_tokenizer_decode_streaming_context_t* ctx = user_data;
  for (iree_host_size_t i = 0; i < strings.count; ++i) {
    iree_string_view_t text = strings.values[i];
    iree_host_size_t remaining = text.size;
    const char* src = text.data;

    while (remaining > 0) {
      iree_host_size_t space = sizeof(ctx->buffer) - ctx->position;
      iree_host_size_t copy_size = remaining < space ? remaining : space;

      memcpy(ctx->buffer + ctx->position, src, copy_size);
      ctx->position += copy_size;
      src += copy_size;
      remaining -= copy_size;

      // Flush if buffer is full.
      if (ctx->position >= sizeof(ctx->buffer)) {
        IREE_RETURN_IF_ERROR(iree_tokenizer_decode_streaming_flush(ctx));
      }
    }
  }
  return iree_ok_status();
}

// Returns true if decoder handles the same inverse operation as transform.
// When both ByteLevel decoder and ByteLevel transform are present (or both
// Metaspace), they would apply the same inverse mapping twice, corrupting
// output. In this case, skip the transform decode since decoder handles it.
// Recursively checks Sequence children to handle Sequence[ByteLevel] etc.
static bool iree_tokenizer_decoder_handles_transform_inverse(
    iree_tokenizer_decoder_type_t decoder_type,
    const iree_tokenizer_text_transform_t* transform) {
  // Direct match: ByteLevel decoder + ByteLevel transform.
  if (decoder_type == IREE_TOKENIZER_DECODER_BYTE_LEVEL &&
      transform->type == IREE_TOKENIZER_TEXT_TRANSFORM_BYTE_LEVEL) {
    return true;
  }
  // Direct match: Metaspace decoder + Metaspace transform.
  if (decoder_type == IREE_TOKENIZER_DECODER_METASPACE &&
      transform->type == IREE_TOKENIZER_TEXT_TRANSFORM_METASPACE) {
    return true;
  }
  // Recurse into Sequence children.
  if (transform->type == IREE_TOKENIZER_TEXT_TRANSFORM_SEQUENCE) {
    const iree_tokenizer_sequence_config_t* seq = &transform->config.sequence;
    for (iree_host_size_t i = 0; i < seq->count; ++i) {
      if (iree_tokenizer_decoder_handles_transform_inverse(decoder_type,
                                                           &seq->children[i])) {
        return true;
      }
    }
  }
  return false;
}

IREE_API_EXPORT iree_status_t iree_tokenizer_decode_streaming(
    const iree_tokenizer_t* tokenizer, const int32_t* ids,
    iree_host_size_t id_count, iree_tokenizer_decode_flags_t flags,
    iree_tokenizer_string_callback_fn_t callback, void* user_data) {
  IREE_ASSERT_ARGUMENT(tokenizer);
  IREE_ASSERT_ARGUMENT(callback);

  // Validate ids pointer when count > 0.
  if (id_count > 0 && !ids) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ids must not be NULL when id_count > 0");
  }

  const iree_tokenizer_vocab_t* vocab = tokenizer->vocab;
  iree_host_size_t vocab_size = iree_tokenizer_vocab_capacity(vocab);

  // Initialize decoder state and streaming context.
  iree_tokenizer_decoder_state_t decoder_state;
  iree_tokenizer_decoder_begin(&tokenizer->decoder, &decoder_state);

  // Check if decoder already handles transform inverse (ByteLevel/Metaspace).
  // Also checks recursively into Sequence children.
  bool skip_transform = iree_tokenizer_decoder_handles_transform_inverse(
      tokenizer->decoder.type, &tokenizer->transform);

  iree_tokenizer_decode_streaming_context_t ctx = {
      .transform = &tokenizer->transform,
      .skip_transform_decode = skip_transform,
      .callback = callback,
      .user_data = user_data,
      .position = 0,
  };

  // Process tokens in batches.
  iree_string_view_t token_batch[IREE_TOKENIZER_STRING_BATCH_CAPACITY];
  iree_host_size_t batch_count = 0;

  for (iree_host_size_t i = 0; i < id_count; ++i) {
    int32_t id = ids[i];

    // Validate token ID is in range.
    if (id < 0 || (iree_host_size_t)id >= vocab_size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid token ID %d at position %" PRIhsz
                              " (vocab size: %" PRIhsz ")",
                              id, i, vocab_size);
    }

    // Skip special tokens if requested.
    if (iree_any_bit_set(flags,
                         IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS)) {
      iree_tokenizer_token_attr_t attrs =
          iree_tokenizer_vocab_token_attrs(vocab, id);
      if (iree_any_bit_set(attrs, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL)) {
        continue;
      }
    }

    // Get token text and add to batch.
    token_batch[batch_count++] = iree_tokenizer_vocab_token_text(vocab, id);

    // Flush batch if full.
    if (batch_count == IREE_TOKENIZER_STRING_BATCH_CAPACITY) {
      iree_string_view_list_t batch = {batch_count, token_batch};
      IREE_RETURN_IF_ERROR(iree_tokenizer_decoder_decode(
          &tokenizer->decoder, &decoder_state, batch,
          iree_tokenizer_decode_streaming_callback, &ctx));
      batch_count = 0;
    }
  }

  // Flush remaining tokens.
  if (batch_count > 0) {
    iree_string_view_list_t batch = {batch_count, token_batch};
    IREE_RETURN_IF_ERROR(iree_tokenizer_decoder_decode(
        &tokenizer->decoder, &decoder_state, batch,
        iree_tokenizer_decode_streaming_callback, &ctx));
  }

  // Flush any remaining buffered text.
  IREE_RETURN_IF_ERROR(iree_tokenizer_decode_streaming_flush(&ctx));

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Buffer-based Decode (wrapper around streaming)
//===----------------------------------------------------------------------===//

// Context for buffer-based decoding callback.
typedef struct {
  char* out_text;
  iree_host_size_t max_text;
  iree_host_size_t position;
} iree_tokenizer_decode_buffer_context_t;

// Callback that collects text into a buffer.
static iree_status_t iree_tokenizer_decode_buffer_callback(
    void* user_data, iree_string_view_list_t strings) {
  iree_tokenizer_decode_buffer_context_t* ctx = user_data;
  for (iree_host_size_t i = 0; i < strings.count; ++i) {
    iree_string_view_t text = strings.values[i];
    // Leave room for null terminator.
    if (ctx->position + text.size >= ctx->max_text) {
      ctx->out_text[ctx->position] = '\0';
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "output buffer too small for decoded text");
    }
    memcpy(ctx->out_text + ctx->position, text.data, text.size);
    ctx->position += text.size;
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_tokenizer_decode(
    const iree_tokenizer_t* tokenizer, const int32_t* ids,
    iree_host_size_t id_count, iree_tokenizer_decode_flags_t flags,
    char* out_text, iree_host_size_t max_text, iree_host_size_t* out_length) {
  IREE_ASSERT_ARGUMENT(tokenizer);
  IREE_ASSERT_ARGUMENT(out_text);
  IREE_ASSERT_ARGUMENT(out_length);
  *out_length = 0;

  if (max_text == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "output buffer size must be > 0");
  }

  // Use streaming decode with buffer collection callback.
  iree_tokenizer_decode_buffer_context_t ctx = {
      .out_text = out_text,
      .max_text = max_text,
      .position = 0,
  };
  IREE_RETURN_IF_ERROR(iree_tokenizer_decode_streaming(
      tokenizer, ids, id_count, flags, iree_tokenizer_decode_buffer_callback,
      &ctx));

  out_text[ctx.position] = '\0';
  *out_length = ctx.position;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Truncation Utility
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_tokenizer_truncate(
    int32_t* ids, iree_host_size_t count, iree_host_size_t max_length,
    iree_tokenizer_encode_flags_t flags, iree_host_size_t* out_count) {
  if (!ids || !out_count || max_length == 0) {
    if (out_count) *out_count = count;
    return;
  }

  // No truncation needed.
  if (count <= max_length) {
    *out_count = count;
    return;
  }

  // Truncate from left (keep end) or right (keep start).
  if (iree_any_bit_set(flags, IREE_TOKENIZER_ENCODE_FLAG_TRUNCATE_LEFT)) {
    iree_host_size_t shift = count - max_length;
    memmove(ids, ids + shift, max_length * sizeof(int32_t));
  }
  *out_count = max_length;
}

//===----------------------------------------------------------------------===//
// Chunk-based Streaming Encode API
//===----------------------------------------------------------------------===//

// Returns the length of the longest literal prefix that matches the end of
// text. This is used to determine how much to hold back in the literal
// lookahead buffer. We only need to hold back if the trailing bytes could be a
// prefix of a literal.
//
// For example, if literals are ["[UNK]", "<mask>"] and text ends with "<ma",
// we return 3 because "<ma" is a prefix of "<mask>".
// If text ends with "hello", we return 0 because no literal starts with 'o'.
//
// Uses the cached first-byte bitmask for fast rejection: if a suffix's first
// byte doesn't match any literal's first byte, we skip it without checking
// all literals. This makes the common case (no prefix match)
// O(max_literal_length) bitmask checks instead of O(max_literal_length *
// literal_count) memcmps.
static iree_host_size_t iree_tokenizer_literal_prefix_length(
    const iree_tokenizer_encode_stream_state_t* state, const char* text,
    iree_host_size_t text_size) {
  const iree_tokenizer_literals_t* literals = &state->tokenizer->literals;
  if (literals->count == 0 || text_size == 0) return 0;

  // Check each possible suffix of the text to see if it's a prefix of any
  // literal. Start with longest possible suffix and work down. Limit to max
  // literal length since longer suffixes can't possibly be prefixes of any
  // literal.
  iree_host_size_t max_check = text_size;
  if (max_check > state->literal_max_length) {
    max_check = state->literal_max_length;
  }

  for (iree_host_size_t suffix_length = max_check; suffix_length > 0;
       --suffix_length) {
    const char* suffix = text + text_size - suffix_length;
    uint8_t first_byte = (uint8_t)suffix[0];

    // Fast rejection: if this byte can't start any literal, skip.
    if ((state->literal_first_byte_mask[first_byte / 8] &
         (1u << (first_byte % 8))) == 0) {
      continue;
    }

    // Check if this suffix is a prefix of any literal.
    for (iree_host_size_t i = 0; i < literals->count; ++i) {
      iree_string_view_t content = literals->entries[i].content;
      if (content.size >= suffix_length &&
          memcmp(content.data, suffix, suffix_length) == 0) {
        // Found a literal that starts with this suffix.
        return suffix_length;
      }
    }
  }

  return 0;
}

// Returns the number of trailing bytes that form an incomplete UTF-8 sequence.
// Used to detect multi-byte UTF-8 characters split across chunk boundaries.
//
// UTF-8 lead byte patterns and expected sequence lengths:
//   0xxxxxxx → 1 byte (ASCII, always complete)
//   110xxxxx → 2 bytes
//   1110xxxx → 3 bytes
//   11110xxx → 4 bytes
//   10xxxxxx → continuation byte (not a lead byte)
//
// Algorithm: Scan backwards from end (max 3 bytes) to find lead byte,
// then compare expected length against available bytes.
static iree_host_size_t iree_tokenizer_utf8_incomplete_tail_length(
    const char* data, iree_host_size_t size) {
  if (size == 0) return 0;

  // Scan backwards to find a lead byte (max 3 bytes back since longest
  // sequence is 4 bytes and we need at least 1 byte present).
  iree_host_size_t scan_limit = size < 4 ? size : 4;
  for (iree_host_size_t i = 1; i <= scan_limit; ++i) {
    uint8_t byte = (uint8_t)data[size - i];

    // Continuation bytes (10xxxxxx) are not lead bytes - keep scanning.
    if ((byte & 0xC0) == 0x80) continue;

    // Found a lead byte. Determine expected sequence length.
    iree_host_size_t expected_length;
    if ((byte & 0x80) == 0x00) {
      // ASCII (0xxxxxxx) - 1 byte, always complete.
      expected_length = 1;
    } else if ((byte & 0xE0) == 0xC0) {
      // 2-byte sequence (110xxxxx).
      expected_length = 2;
    } else if ((byte & 0xF0) == 0xE0) {
      // 3-byte sequence (1110xxxx).
      expected_length = 3;
    } else if ((byte & 0xF8) == 0xF0) {
      // 4-byte sequence (11110xxx).
      expected_length = 4;
    } else {
      // Invalid lead byte (0xFF, 0xFE, or overlong) - treat as complete.
      return 0;
    }

    // Available bytes from lead to end of buffer.
    iree_host_size_t available = i;
    if (available < expected_length) {
      // Incomplete - return number of bytes in the partial sequence.
      return available;
    }
    // Complete sequence found - no incomplete tail.
    return 0;
  }

  // Only continuation bytes found - treat as invalid/complete.
  return 0;
}

// Helper to flush the token buffer in streaming state.
static iree_status_t iree_tokenizer_encode_stream_flush_tokens(
    iree_tokenizer_encode_stream_state_t* state,
    iree_tokenizer_token_callback_fn_t callback, void* user_data) {
  if (state->token_count == 0 || !callback) return iree_ok_status();
  iree_tokenizer_id_list_t ids = {
      .count = state->token_count,
      .values = state->token_buffer,
  };
  iree_status_t status = callback(user_data, ids);
  state->token_count = 0;
  return status;
}

// Helper to accumulate tokens in streaming state, flushing when full.
static iree_status_t iree_tokenizer_encode_stream_accumulate(
    iree_tokenizer_encode_stream_state_t* state, const int32_t* ids,
    iree_host_size_t count, iree_tokenizer_token_callback_fn_t callback,
    void* user_data) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    state->token_buffer[state->token_count++] = ids[i];
    if (state->token_count >= IREE_TOKENIZER_TOKEN_BATCH_CAPACITY) {
      IREE_RETURN_IF_ERROR(iree_tokenizer_encode_stream_flush_tokens(
          state, callback, user_data));
    }
  }
  return iree_ok_status();
}

// Context for stream-based encoding, wrapping the stream state.
typedef struct {
  iree_tokenizer_encode_stream_state_t* state;
  iree_tokenizer_token_callback_fn_t callback;
  void* user_data;
} iree_tokenizer_stream_encode_context_t;

// Callback that encodes each segment for streaming.
static iree_status_t iree_tokenizer_stream_segment_callback(
    void* user_data, iree_string_view_list_t segments) {
  iree_tokenizer_stream_encode_context_t* ctx = user_data;
  int32_t word_ids[IREE_TOKENIZER_TOKEN_BATCH_CAPACITY];
  for (iree_host_size_t i = 0; i < segments.count; ++i) {
    iree_host_size_t word_count = 0;
    IREE_RETURN_IF_ERROR(ctx->state->tokenizer->vtable->encode_word(
        ctx->state->tokenizer, segments.values[i], word_ids,
        IREE_ARRAYSIZE(word_ids), &word_count));
    IREE_RETURN_IF_ERROR(iree_tokenizer_encode_stream_accumulate(
        ctx->state, word_ids, word_count, ctx->callback, ctx->user_data));
  }
  return iree_ok_status();
}

// Callback for literal interception in streaming: encodes text through
// transform.
static iree_status_t iree_tokenizer_stream_literals_text_callback(
    void* user_data, iree_string_view_list_t segments) {
  iree_tokenizer_stream_encode_context_t* ctx = user_data;
  for (iree_host_size_t i = 0; i < segments.count; ++i) {
    if (segments.values[i].size == 0) continue;
    IREE_RETURN_IF_ERROR(iree_tokenizer_text_transform_encode(
        NULL, &ctx->state->tokenizer->transform, segments.values[i],
        iree_tokenizer_stream_segment_callback, ctx));
  }
  return iree_ok_status();
}

// Callback for literal interception in streaming: emits matched token IDs.
static iree_status_t iree_tokenizer_stream_literals_token_callback(
    void* user_data, iree_tokenizer_id_list_t ids) {
  iree_tokenizer_stream_encode_context_t* ctx = user_data;
  return iree_tokenizer_encode_stream_accumulate(
      ctx->state, ids.values, ids.count, ctx->callback, ctx->user_data);
}

// Encodes text through the pipeline (literal intercept -> transform -> word
// encode).
static iree_status_t iree_tokenizer_stream_encode_text(
    iree_tokenizer_encode_stream_state_t* state, iree_string_view_t text,
    iree_tokenizer_token_callback_fn_t callback, void* user_data) {
  if (text.size == 0) return iree_ok_status();

  iree_tokenizer_stream_encode_context_t ctx = {
      .state = state,
      .callback = callback,
      .user_data = user_data,
  };

  return iree_tokenizer_literals_intercept(
      &state->tokenizer->literals, text,
      iree_tokenizer_stream_literals_text_callback,
      iree_tokenizer_stream_literals_token_callback, &ctx);
}

IREE_API_EXPORT void iree_tokenizer_encode_stream_initialize(
    iree_tokenizer_encode_stream_state_t* state,
    const iree_tokenizer_t* tokenizer, iree_tokenizer_encode_flags_t flags) {
  memset(state, 0, sizeof(*state));
  state->tokenizer = tokenizer;
  state->flags = flags;

  // Cache the maximum literal length and first-byte bitmask for lookahead.
  // The bitmask allows O(1) rejection of suffixes that can't possibly match
  // any literal, making the per-chunk prefix check efficient.
  // NOTE: max_literal is clamped to lookahead buffer capacity to prevent
  // overflow when literals span chunk boundaries.
  uint8_t max_literal = 0;
  for (iree_host_size_t i = 0; i < tokenizer->literals.count; ++i) {
    iree_string_view_t content = tokenizer->literals.entries[i].content;
    if (content.size > max_literal &&
        content.size <= IREE_TOKENIZER_LITERAL_LOOKAHEAD_CAPACITY) {
      max_literal = (uint8_t)content.size;
    }
    // Set bit for first byte of this literal.
    if (content.size > 0) {
      uint8_t first_byte = (uint8_t)content.data[0];
      state->literal_first_byte_mask[first_byte / 8] |=
          (1u << (first_byte % 8));
    }
  }
  state->literal_max_length = max_literal;
}

//===----------------------------------------------------------------------===//
// Stream Feed Helper: UTF-8 Partial Sequence Handling
//===----------------------------------------------------------------------===//

// Handles UTF-8 sequences split across chunk boundaries.
//
// Multi-byte UTF-8 characters can be split across chunks. We buffer incomplete
// sequences and prepend them to the next chunk.
//
// Example: "Hello" + 3-byte UTF-8 char (E2 96 81 = ▁)
//   Chunk 1: "Hello\xE2"    -> process "Hello", save \xE2
//   Chunk 2: "\x96\x81foo"  -> prepend \xE2, process "▁foo"
//
// Returns true if the entire chunk was consumed (still waiting for more bytes).
// Updates |working_chunk| to point at the UTF-8-safe portion.
static iree_status_t iree_tokenizer_stream_handle_utf8(
    iree_tokenizer_encode_stream_state_t* state, iree_string_view_t chunk,
    char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* working_chunk, bool* consumed_all,
    iree_tokenizer_token_callback_fn_t callback, void* user_data) {
  *consumed_all = false;
  *working_chunk = chunk;

  // Prepend any buffered partial UTF-8 bytes from previous chunk.
  if (state->utf8_partial_length > 0) {
    iree_host_size_t partial_length = state->utf8_partial_length;

    // Check if chunk provides enough bytes to complete the sequence.
    char utf8_check_buffer[8];
    memcpy(utf8_check_buffer, state->utf8_partial, partial_length);
    iree_host_size_t copy_for_check = 4 - partial_length;
    if (copy_for_check > chunk.size) copy_for_check = chunk.size;
    memcpy(utf8_check_buffer + partial_length, chunk.data, copy_for_check);

    iree_host_size_t check_length = partial_length + copy_for_check;
    iree_host_size_t still_incomplete =
        iree_tokenizer_utf8_incomplete_tail_length(utf8_check_buffer,
                                                   check_length);

    if (still_incomplete > 0 && still_incomplete == check_length) {
      // Still incomplete - buffer all and wait for more data.
      memcpy(state->utf8_partial, utf8_check_buffer, check_length);
      state->utf8_partial_length = (uint8_t)check_length;
      *consumed_all = true;
      return iree_ok_status();
    }

    // Complete sequence found. Build: [completed UTF-8] + [remaining chunk].
    iree_host_size_t complete_length = check_length - still_incomplete;
    iree_host_size_t chunk_consumed = copy_for_check - still_incomplete;
    iree_host_size_t remaining_length = chunk.size - chunk_consumed;

    if (complete_length + remaining_length <= buffer_capacity) {
      memcpy(buffer, utf8_check_buffer, complete_length);
      if (remaining_length > 0) {
        memcpy(buffer + complete_length, chunk.data + chunk_consumed,
               remaining_length);
      }
      *working_chunk =
          iree_make_string_view(buffer, complete_length + remaining_length);
    } else {
      // Too large - encode completed UTF-8 directly, continue with rest.
      iree_string_view_t complete =
          iree_make_string_view(utf8_check_buffer, complete_length);
      IREE_RETURN_IF_ERROR(iree_tokenizer_stream_encode_text(
          state, complete, callback, user_data));
      working_chunk->data += chunk_consumed;
      working_chunk->size = remaining_length;
    }
    state->utf8_partial_length = 0;
  }

  // Check for incomplete UTF-8 at end of this chunk.
  iree_host_size_t incomplete_tail = iree_tokenizer_utf8_incomplete_tail_length(
      working_chunk->data, working_chunk->size);
  if (incomplete_tail > 0) {
    iree_host_size_t tail_start = working_chunk->size - incomplete_tail;
    memcpy(state->utf8_partial, working_chunk->data + tail_start,
           incomplete_tail);
    state->utf8_partial_length = (uint8_t)incomplete_tail;
    working_chunk->size = tail_start;
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Stream Feed Helper: Literal Token Lookahead
//===----------------------------------------------------------------------===//

// Handles literal tokens (e.g., "<mask>", "[UNK]") that can span chunk
// boundaries.
//
// We hold back trailing bytes that could be a prefix of a literal token,
// waiting for the next chunk to determine if it completes a literal.
//
// Updates |working_chunk| to exclude any held-back literal prefix bytes.
static iree_status_t iree_tokenizer_stream_handle_literal_lookahead(
    iree_tokenizer_encode_stream_state_t* state, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* working_chunk,
    iree_tokenizer_token_callback_fn_t callback, void* user_data) {
  if (state->literal_max_length == 0) return iree_ok_status();

  iree_host_size_t lookahead_length = state->literal_lookahead_length;
  iree_host_size_t combined_length = lookahead_length + working_chunk->size;

  if (combined_length <= buffer_capacity) {
    // Combine pending lookahead with chunk.
    // IMPORTANT: Move chunk data FIRST (may overlap with buffer from UTF-8).
    memmove(buffer + lookahead_length, working_chunk->data,
            working_chunk->size);
    if (lookahead_length > 0) {
      memcpy(buffer, state->literal_lookahead, lookahead_length);
    }

    // Check if trailing bytes could be a literal prefix.
    iree_host_size_t holdback =
        iree_tokenizer_literal_prefix_length(state, buffer, combined_length);

    if (holdback > 0) {
      memcpy(state->literal_lookahead, buffer + combined_length - holdback,
             holdback);
    }
    state->literal_lookahead_length = (uint8_t)holdback;

    *working_chunk = iree_make_string_view(buffer, combined_length - holdback);
  } else {
    // Large chunk: process pending lookahead first.
    if (lookahead_length > 0) {
      iree_host_size_t chunk_for_lookahead = state->literal_max_length;
      if (chunk_for_lookahead > working_chunk->size) {
        chunk_for_lookahead = working_chunk->size;
      }
      memcpy(buffer, state->literal_lookahead, lookahead_length);
      memcpy(buffer + lookahead_length, working_chunk->data,
             chunk_for_lookahead);
      iree_host_size_t first_part_length =
          lookahead_length + chunk_for_lookahead;

      iree_host_size_t holdback = iree_tokenizer_literal_prefix_length(
          state, buffer, first_part_length);
      iree_host_size_t emit_length = first_part_length - holdback;

      if (emit_length > 0) {
        iree_string_view_t emit = iree_make_string_view(buffer, emit_length);
        IREE_RETURN_IF_ERROR(iree_tokenizer_stream_encode_text(
            state, emit, callback, user_data));
      }

      if (holdback > 0) {
        memcpy(state->literal_lookahead, buffer + first_part_length - holdback,
               holdback);
      }
      state->literal_lookahead_length = (uint8_t)holdback;

      working_chunk->data += chunk_for_lookahead;
      working_chunk->size -= chunk_for_lookahead;
    }

    // Hold back potential literal prefixes from remaining chunk.
    if (working_chunk->size > 0) {
      iree_host_size_t holdback = iree_tokenizer_literal_prefix_length(
          state, working_chunk->data, working_chunk->size);

      if (holdback > 0) {
        memcpy(state->literal_lookahead,
               working_chunk->data + working_chunk->size - holdback, holdback);
      }
      state->literal_lookahead_length = (uint8_t)holdback;
      working_chunk->size -= holdback;
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Stream Feed Helper: Word Boundary Carryover
//===----------------------------------------------------------------------===//

// Handles word boundaries for transforms that split on whitespace (WHITESPACE,
// BERT).
//
// These transforms need complete words to produce correct segments. If a chunk
// ends mid-word (no trailing whitespace), we buffer the partial word and
// prepend it to the next chunk.
//
// Transforms that handle their own boundaries (SEQUENCE with Split, SPLIT with
// regex, BYTE_LEVEL, METASPACE) don't need this.
static iree_status_t iree_tokenizer_stream_handle_word_carryover(
    iree_tokenizer_encode_stream_state_t* state,
    iree_string_view_t working_chunk,
    iree_tokenizer_token_callback_fn_t callback, void* user_data) {
  const iree_tokenizer_t* tokenizer = state->tokenizer;

  bool needs_carryover =
      tokenizer->transform.type == IREE_TOKENIZER_TEXT_TRANSFORM_WHITESPACE ||
      tokenizer->transform.type == IREE_TOKENIZER_TEXT_TRANSFORM_BERT;

  if (!needs_carryover || working_chunk.size == 0) {
    // No carryover needed - encode directly.
    if (working_chunk.size > 0) {
      return iree_tokenizer_stream_encode_text(state, working_chunk, callback,
                                               user_data);
    }
    return iree_ok_status();
  }

  // Find last whitespace - text up to whitespace is safe to encode.
  iree_host_size_t safe_length = 0;
  for (iree_host_size_t i = working_chunk.size; i > 0; --i) {
    char c = working_chunk.data[i - 1];
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      safe_length = i;
      break;
    }
  }

  iree_host_size_t carryover_length = state->transform_carryover_length;

  if (safe_length == 0) {
    // No whitespace - buffer entire chunk.
    iree_host_size_t total = carryover_length + working_chunk.size;
    if (total <= IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
      memcpy(state->transform_carryover + carryover_length, working_chunk.data,
             working_chunk.size);
      state->transform_carryover_length = total;
    } else {
      // Overflow - flush carryover and start fresh.
      if (carryover_length > 0) {
        iree_string_view_t carryover =
            iree_make_string_view(state->transform_carryover, carryover_length);
        IREE_RETURN_IF_ERROR(iree_tokenizer_stream_encode_text(
            state, carryover, callback, user_data));
      }
      if (working_chunk.size <= IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
        memcpy(state->transform_carryover, working_chunk.data,
               working_chunk.size);
        state->transform_carryover_length = working_chunk.size;
      } else {
        IREE_RETURN_IF_ERROR(iree_tokenizer_stream_encode_text(
            state, working_chunk, callback, user_data));
        state->transform_carryover_length = 0;
      }
    }
    return iree_ok_status();
  }

  // Found whitespace - encode carryover + safe portion.
  iree_host_size_t total = carryover_length + safe_length;
  if (carryover_length > 0 && total <= IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
    memcpy(state->transform_carryover + carryover_length, working_chunk.data,
           safe_length);
    iree_string_view_t combined =
        iree_make_string_view(state->transform_carryover, total);
    IREE_RETURN_IF_ERROR(iree_tokenizer_stream_encode_text(
        state, combined, callback, user_data));
  } else {
    if (carryover_length > 0) {
      iree_string_view_t carryover =
          iree_make_string_view(state->transform_carryover, carryover_length);
      IREE_RETURN_IF_ERROR(iree_tokenizer_stream_encode_text(
          state, carryover, callback, user_data));
    }
    iree_string_view_t safe =
        iree_make_string_view(working_chunk.data, safe_length);
    IREE_RETURN_IF_ERROR(
        iree_tokenizer_stream_encode_text(state, safe, callback, user_data));
  }

  // Save trailing non-whitespace for next chunk.
  iree_host_size_t trailing = working_chunk.size - safe_length;
  if (trailing > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
    // Word fragment exceeds carryover buffer - fail explicitly rather than
    // setting length without copying data (which would cause heap overflow).
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "word fragment %" PRIhsz
                            " bytes exceeds carryover buffer capacity",
                            trailing);
  }
  if (trailing > 0) {
    memcpy(state->transform_carryover, working_chunk.data + safe_length,
           trailing);
    state->transform_carryover_length = trailing;
  } else {
    state->transform_carryover_length = 0;
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Stream Feed Main Entry Point
//===----------------------------------------------------------------------===//

// Adapter context for postprocessor emit callbacks.
typedef struct {
  iree_tokenizer_encode_stream_state_t* state;
  iree_tokenizer_token_callback_fn_t callback;
  void* user_data;
} iree_tokenizer_emit_adapter_context_t;

// Adapter callback that converts emit_token_fn calls to stream accumulation.
static iree_status_t iree_tokenizer_emit_adapter_callback(void* user_data,
                                                          int32_t token_id) {
  iree_tokenizer_emit_adapter_context_t* ctx = user_data;
  return iree_tokenizer_encode_stream_accumulate(ctx->state, &token_id, 1,
                                                 ctx->callback, ctx->user_data);
}

IREE_API_EXPORT iree_status_t iree_tokenizer_encode_stream_feed(
    iree_tokenizer_encode_stream_state_t* state, iree_string_view_t chunk,
    iree_tokenizer_token_callback_fn_t callback, void* user_data) {
  if (!state) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "state is NULL");
  }
  if (chunk.size == 0) {
    return iree_ok_status();
  }

  const iree_tokenizer_t* tokenizer = state->tokenizer;

  // Emit prefix tokens (BOS + any additional prefix specials) on first text.
  if (iree_any_bit_set(state->flags,
                       IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS) &&
      !state->bos_emitted) {
    iree_tokenizer_emit_adapter_context_t emit_ctx = {
        .state = state,
        .callback = callback,
        .user_data = user_data,
    };
    IREE_RETURN_IF_ERROR(iree_tokenizer_postprocessor_emit_prefix(
        &tokenizer->postprocessor, iree_tokenizer_emit_adapter_callback,
        &emit_ctx));
    state->bos_emitted = true;
  }
  state->text_started = true;

  // Shared buffer for UTF-8 and literal lookahead handling.
  char buffer[IREE_TOKENIZER_LITERAL_LOOKAHEAD_CAPACITY +
              IREE_TOKENIZER_DATA_BATCH_CAPACITY];
  const iree_host_size_t buffer_capacity = sizeof(buffer);

  // Step 1: Handle UTF-8 partial sequences spanning chunk boundaries.
  iree_string_view_t working_chunk;
  bool consumed_all = false;
  IREE_RETURN_IF_ERROR(iree_tokenizer_stream_handle_utf8(
      state, chunk, buffer, buffer_capacity, &working_chunk, &consumed_all,
      callback, user_data));
  if (consumed_all) {
    state->total_bytes_processed += chunk.size;
    return iree_ok_status();
  }

  // Step 2: Handle literal tokens spanning chunk boundaries.
  IREE_RETURN_IF_ERROR(iree_tokenizer_stream_handle_literal_lookahead(
      state, buffer, buffer_capacity, &working_chunk, callback, user_data));

  // Step 3: Handle word boundaries for WHITESPACE/BERT transforms.
  IREE_RETURN_IF_ERROR(iree_tokenizer_stream_handle_word_carryover(
      state, working_chunk, callback, user_data));

  state->total_bytes_processed += chunk.size;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_tokenizer_encode_stream_finalize(
    iree_tokenizer_encode_stream_state_t* state,
    iree_tokenizer_token_callback_fn_t callback, void* user_data) {
  if (!state) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "state is NULL");
  }

  const iree_tokenizer_t* tokenizer = state->tokenizer;
  bool add_special = iree_any_bit_set(
      state->flags, IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS);

  // Adapter context for postprocessor emit callbacks.
  iree_tokenizer_emit_adapter_context_t emit_ctx = {
      .state = state,
      .callback = callback,
      .user_data = user_data,
  };

  // For empty input with ADD_SPECIAL_TOKENS, we still need to emit prefix.
  // This handles the case where feed() was never called or only called with
  // empty chunks. The output should be [prefix..., suffix...] for empty input.
  if (add_special && !state->bos_emitted) {
    IREE_RETURN_IF_ERROR(iree_tokenizer_postprocessor_emit_prefix(
        &tokenizer->postprocessor, iree_tokenizer_emit_adapter_callback,
        &emit_ctx));
    state->bos_emitted = true;
  }

  // Handle incomplete UTF-8 at stream end.
  // Invalid/incomplete UTF-8 sequences are treated as replacement characters
  // (U+FFFD). We emit one replacement character per incomplete sequence.
  if (state->utf8_partial_length > 0) {
    // U+FFFD in UTF-8 is: EF BF BD (3 bytes).
    static const char kReplacementChar[] = "\xEF\xBF\xBD";
    iree_string_view_t replacement = iree_make_string_view(kReplacementChar, 3);
    IREE_RETURN_IF_ERROR(iree_tokenizer_stream_encode_text(
        state, replacement, callback, user_data));
    state->utf8_partial_length = 0;
  }

  // Process any remaining carryover.
  if (state->transform_carryover_length > 0) {
    iree_string_view_t remaining = iree_make_string_view(
        state->transform_carryover, state->transform_carryover_length);
    IREE_RETURN_IF_ERROR(iree_tokenizer_stream_encode_text(
        state, remaining, callback, user_data));
    state->transform_carryover_length = 0;
  }

  // Process any remaining literal lookahead.
  if (state->literal_lookahead_length > 0) {
    iree_string_view_t remaining = iree_make_string_view(
        state->literal_lookahead, state->literal_lookahead_length);
    IREE_RETURN_IF_ERROR(iree_tokenizer_stream_encode_text(
        state, remaining, callback, user_data));
    state->literal_lookahead_length = 0;
  }

  // Emit suffix tokens (EOS + any additional suffix specials).
  if (add_special) {
    IREE_RETURN_IF_ERROR(iree_tokenizer_postprocessor_emit_suffix(
        &tokenizer->postprocessor, iree_tokenizer_emit_adapter_callback,
        &emit_ctx));
  }

  // Final flush of any remaining tokens.
  return iree_tokenizer_encode_stream_flush_tokens(state, callback, user_data);
}
