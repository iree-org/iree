// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/transforms/sequence.h"

//===----------------------------------------------------------------------===//
// Sequence Encode
//===----------------------------------------------------------------------===//

// Context for chaining transform callbacks.
typedef struct {
  const iree_tokenizer_sequence_config_t* config;
  iree_host_size_t next_transform_index;
  iree_tokenizer_string_callback_fn_t final_callback;
  void* final_user_data;
} iree_tokenizer_sequence_chain_context_t;

// Callback that chains to the next transform in the sequence.
static iree_status_t iree_tokenizer_sequence_chain_callback(
    void* user_data, iree_string_view_list_t segments) {
  iree_tokenizer_sequence_chain_context_t* context = user_data;

  // If all transforms have been applied, pass through to final callback.
  if (context->next_transform_index >= context->config->count) {
    return context->final_callback(context->final_user_data, segments);
  }

  // Apply the next transform to each segment in the batch.
  // Pass NULL for normalizer since first transform already normalized the text.
  for (iree_host_size_t i = 0; i < segments.count; ++i) {
    iree_tokenizer_sequence_chain_context_t next_context = {
        .config = context->config,
        .next_transform_index = context->next_transform_index + 1,
        .final_callback = context->final_callback,
        .final_user_data = context->final_user_data,
    };
    IREE_RETURN_IF_ERROR(iree_tokenizer_text_transform_encode(
        NULL, &context->config->children[context->next_transform_index],
        segments.values[i], iree_tokenizer_sequence_chain_callback,
        &next_context));
  }

  return iree_ok_status();
}

// Sequence encoding: chains transforms via recursive callbacks.
// For Sequence[A], this is zero overhead - we delegate directly to A.
// For Sequence[A, B, ...], A's output feeds into B's input via callbacks.
// The normalizer is only passed to the first child - subsequent children
// receive already-normalized text.
iree_status_t iree_tokenizer_sequence_encode(
    const iree_tokenizer_normalizer_t* normalizer,
    const iree_tokenizer_sequence_config_t* config, iree_string_view_t text,
    iree_tokenizer_string_callback_fn_t callback, void* user_data) {
  IREE_ASSERT_ARGUMENT(config);

  if (config->count == 0) {
    // Empty sequence: passthrough entire text as single segment.
    if (text.size == 0) return iree_ok_status();
    iree_string_view_list_t list = {1, &text};
    return callback(user_data, list);
  }

  if (config->count == 1) {
    // Single transform: direct delegation with zero overhead.
    return iree_tokenizer_text_transform_encode(
        normalizer, &config->children[0], text, callback, user_data);
  }

  // Multiple transforms: chain callbacks.
  // Transform 0 runs first with the normalizer. Its output (already normalized)
  // feeds into transform 1, etc. Subsequent transforms receive NULL normalizer.
  iree_tokenizer_sequence_chain_context_t context = {
      .config = config,
      .next_transform_index = 1,
      .final_callback = callback,
      .final_user_data = user_data,
  };
  return iree_tokenizer_text_transform_encode(
      normalizer, &config->children[0], text,
      iree_tokenizer_sequence_chain_callback, &context);
}

//===----------------------------------------------------------------------===//
// Sequence Decode
//===----------------------------------------------------------------------===//

// Sequence decoding: applies transforms in reverse order.
// Each transform writes to the same output buffer, progressively shrinking
// or maintaining the data size (all decodes are size-reducing or neutral).
iree_status_t iree_tokenizer_sequence_decode(
    const iree_tokenizer_sequence_config_t* config, iree_string_view_t text,
    char* out_buffer, iree_host_size_t max_size, iree_host_size_t* out_size) {
  IREE_ASSERT_ARGUMENT(config);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_ASSERT_ARGUMENT(out_size);
  *out_size = 0;

  if (config->count == 0) {
    // Empty sequence: passthrough.
    if (text.size > max_size) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "output buffer too small");
    }
    memmove(out_buffer, text.data, text.size);
    *out_size = text.size;
    return iree_ok_status();
  }

  // Copy input to output buffer if not already overlapping.
  // This allows in-place transformation in subsequent steps.
  if (text.data != out_buffer) {
    if (text.size > max_size) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "output buffer too small");
    }
    memmove(out_buffer, text.data, text.size);
  }
  iree_host_size_t current_size = text.size;

  // Apply transforms in reverse order (last to first).
  // Each transform writes directly to out_buffer and shrinks or maintains size.
  for (iree_host_size_t i = config->count; i > 0; --i) {
    iree_host_size_t new_size = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_text_transform_decode(
        &config->children[i - 1],
        iree_make_string_view(out_buffer, current_size), out_buffer, max_size,
        &new_size));
    current_size = new_size;
  }

  *out_size = current_size;
  return iree_ok_status();
}
