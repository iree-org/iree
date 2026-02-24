// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder/passthrough.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Passthrough Decoder Implementation
//===----------------------------------------------------------------------===//

// Passthrough decoder: no additional config beyond base.
typedef struct iree_tokenizer_decoder_passthrough_t {
  iree_tokenizer_decoder_t base;
  iree_allocator_t allocator;
} iree_tokenizer_decoder_passthrough_t;

// Passthrough state: just the base (no buffering needed).
typedef struct iree_tokenizer_decoder_passthrough_state_t {
  iree_tokenizer_decoder_state_t base;
} iree_tokenizer_decoder_passthrough_state_t;

static const iree_tokenizer_decoder_vtable_t
    iree_tokenizer_decoder_passthrough_vtable;

iree_status_t iree_tokenizer_decoder_passthrough_allocate(
    iree_allocator_t allocator, iree_tokenizer_decoder_t** out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  *out_decoder = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tokenizer_decoder_passthrough_t* decoder = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*decoder), (void**)&decoder));

  iree_tokenizer_decoder_initialize(
      &decoder->base, &iree_tokenizer_decoder_passthrough_vtable,
      sizeof(iree_tokenizer_decoder_passthrough_state_t),
      IREE_TOKENIZER_DECODER_CAPABILITY_STATELESS);
  decoder->allocator = allocator;

  *out_decoder = &decoder->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_decoder_passthrough_destroy(
    iree_tokenizer_decoder_t* decoder) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_decoder_passthrough_t* self =
      (iree_tokenizer_decoder_passthrough_t*)decoder;
  iree_allocator_t allocator = self->allocator;
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_decoder_passthrough_state_initialize(
    const iree_tokenizer_decoder_t* decoder, void* storage,
    iree_tokenizer_decoder_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_decoder_passthrough_state_t* state =
      (iree_tokenizer_decoder_passthrough_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.decoder = decoder;
  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_decoder_passthrough_state_deinitialize(
    iree_tokenizer_decoder_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_decoder_passthrough_state_process(
    iree_tokenizer_decoder_state_t* state,
    iree_tokenizer_string_list_t token_strings,
    iree_mutable_string_view_t output, iree_host_size_t* out_strings_consumed,
    iree_host_size_t* out_bytes_written) {
  // Concatenate token strings into output buffer.
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;

  for (iree_host_size_t i = 0; i < token_strings.count; ++i) {
    iree_string_view_t token = token_strings.values[i];

    // Check if we have room for this token.
    if (bytes_written + token.size > output.size) {
      // No room - stop here. Only count tokens that fit completely.
      break;
    }

    // Copy token to output. Use memmove - source and destination may overlap
    // when used in sequence decoder (same buffer for input and output).
    if (token.size > 0) {
      memmove(output.data + bytes_written, token.data, token.size);
    }
    bytes_written += token.size;
    strings_consumed++;
  }

  *out_strings_consumed = strings_consumed;
  *out_bytes_written = bytes_written;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_decoder_passthrough_state_finalize(
    iree_tokenizer_decoder_state_t* state, iree_mutable_string_view_t output,
    iree_host_size_t* out_written) {
  // No buffering, nothing to flush.
  *out_written = 0;
  return iree_ok_status();
}

static bool iree_tokenizer_decoder_passthrough_state_has_pending(
    const iree_tokenizer_decoder_state_t* state) {
  // Never has pending data (no buffering).
  return false;
}

static const iree_tokenizer_decoder_vtable_t
    iree_tokenizer_decoder_passthrough_vtable = {
        .destroy = iree_tokenizer_decoder_passthrough_destroy,
        .state_initialize = iree_tokenizer_decoder_passthrough_state_initialize,
        .state_deinitialize =
            iree_tokenizer_decoder_passthrough_state_deinitialize,
        .state_process = iree_tokenizer_decoder_passthrough_state_process,
        .state_finalize = iree_tokenizer_decoder_passthrough_state_finalize,
        .state_has_pending =
            iree_tokenizer_decoder_passthrough_state_has_pending,
        .flags = IREE_TOKENIZER_DECODER_FLAG_SHRINKING,
};
