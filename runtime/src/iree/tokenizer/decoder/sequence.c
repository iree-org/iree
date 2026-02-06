// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder/sequence.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Sequence Decoder Implementation
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_decoder_sequence_t {
  iree_tokenizer_decoder_t base;
  iree_allocator_t allocator;
  iree_host_size_t child_count;
  iree_tokenizer_decoder_t* children[IREE_TOKENIZER_DECODER_SEQUENCE_MAX_DEPTH];
  // Cached state offsets for fast child state lookup.
  iree_host_size_t
      child_state_offsets[IREE_TOKENIZER_DECODER_SEQUENCE_MAX_DEPTH];
} iree_tokenizer_decoder_sequence_t;

typedef struct iree_tokenizer_decoder_sequence_state_t {
  iree_tokenizer_decoder_state_t base;
  iree_host_size_t child_count;
  // Child states stored contiguously after this struct.
  // Access via: (uint8_t*)state + state->child_state_offsets[i]
} iree_tokenizer_decoder_sequence_state_t;

static const iree_tokenizer_decoder_vtable_t
    iree_tokenizer_decoder_sequence_vtable;

iree_status_t iree_tokenizer_decoder_sequence_allocate(
    iree_tokenizer_decoder_t* const* children, iree_host_size_t child_count,
    iree_allocator_t allocator, iree_tokenizer_decoder_t** out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  *out_decoder = NULL;

  // Validate child count: require at least 2 children.
  // Sequences with 0 or 1 children should not be created - the JSON parser
  // handles these cases by returning NULL (empty) or the single child directly.
  if (child_count < 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "sequence decoder requires at least 2 children, got %" PRIhsz
        "; use NULL for empty or pass single child directly",
        child_count);
  }
  if (child_count > IREE_TOKENIZER_DECODER_SEQUENCE_MAX_DEPTH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "sequence child count %" PRIhsz " exceeds maximum %d", child_count,
        IREE_TOKENIZER_DECODER_SEQUENCE_MAX_DEPTH);
  }

  // Validate all children are shrinking.
  for (iree_host_size_t i = 0; i < child_count; ++i) {
    if (!children[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "sequence child %" PRIhsz " is NULL", i);
    }
    if (!iree_any_bit_set(children[i]->vtable->flags,
                          IREE_TOKENIZER_DECODER_FLAG_SHRINKING)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "sequence child %" PRIhsz
          " lacks SHRINKING flag; only shrinking decoders are supported",
          i);
    }
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // Calculate total state size: base state + all child states.
  iree_host_size_t total_state_size =
      sizeof(iree_tokenizer_decoder_sequence_state_t);
  for (iree_host_size_t i = 0; i < child_count; ++i) {
    total_state_size += iree_tokenizer_decoder_state_size(children[i]);
  }

  // Allocate decoder.
  iree_tokenizer_decoder_sequence_t* decoder = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*decoder), (void**)&decoder));

  // Compute capabilities from children:
  //   STATELESS = intersection (all children must be stateless).
  //   STATELESS_EXCEPT_BYTE_TOKENS = promoted from any child that has it,
  //     provided all other children are STATELESS. At most one child may have
  //     this capability (ByteFallback); if any child has NONE, the result is
  //     NONE.
  //   POSITION_SENSITIVE = union (any child position-sensitive).
  iree_tokenizer_decoder_capability_t capabilities =
      IREE_TOKENIZER_DECODER_CAPABILITY_STATELESS;
  for (iree_host_size_t i = 0; i < child_count; ++i) {
    iree_tokenizer_decoder_capability_t child_capabilities =
        children[i]->capabilities;
    if (iree_any_bit_set(child_capabilities,
                         IREE_TOKENIZER_DECODER_CAPABILITY_STATELESS)) {
      // Fully stateless child — compatible with everything.
    } else if (
        iree_any_bit_set(
            child_capabilities,
            IREE_TOKENIZER_DECODER_CAPABILITY_STATELESS_EXCEPT_BYTE_TOKENS)) {
      // Promote sequence to STATELESS_EXCEPT_BYTE_TOKENS. This replaces
      // STATELESS (non-byte tokens still pre-decodable) and is only valid
      // if no other child already contributed this capability.
      capabilities &= ~IREE_TOKENIZER_DECODER_CAPABILITY_STATELESS;
      capabilities |=
          IREE_TOKENIZER_DECODER_CAPABILITY_STATELESS_EXCEPT_BYTE_TOKENS;
    } else {
      // Child has NONE — entire sequence is not pre-decodable.
      capabilities = IREE_TOKENIZER_DECODER_CAPABILITY_NONE;
      break;
    }
    if (iree_any_bit_set(
            child_capabilities,
            IREE_TOKENIZER_DECODER_CAPABILITY_POSITION_SENSITIVE)) {
      capabilities |= IREE_TOKENIZER_DECODER_CAPABILITY_POSITION_SENSITIVE;
    }
  }

  iree_tokenizer_decoder_initialize(&decoder->base,
                                    &iree_tokenizer_decoder_sequence_vtable,
                                    total_state_size, capabilities);
  decoder->allocator = allocator;
  decoder->child_count = child_count;

  // Copy child pointers and calculate state offsets.
  iree_host_size_t offset = sizeof(iree_tokenizer_decoder_sequence_state_t);
  for (iree_host_size_t i = 0; i < child_count; ++i) {
    decoder->children[i] = children[i];
    decoder->child_state_offsets[i] = offset;
    offset += iree_tokenizer_decoder_state_size(children[i]);
  }

  *out_decoder = &decoder->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_decoder_sequence_destroy(
    iree_tokenizer_decoder_t* base_decoder) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_decoder_sequence_t* decoder =
      (iree_tokenizer_decoder_sequence_t*)base_decoder;
  iree_allocator_t allocator = decoder->allocator;

  // Free all child decoders (sequence takes ownership).
  for (iree_host_size_t i = 0; i < decoder->child_count; ++i) {
    if (decoder->children[i]) {
      iree_tokenizer_decoder_free(decoder->children[i]);
    }
  }

  iree_allocator_free(allocator, decoder);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_decoder_sequence_state_initialize(
    const iree_tokenizer_decoder_t* base_decoder, void* storage,
    iree_tokenizer_decoder_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_tokenizer_decoder_sequence_t* decoder =
      (const iree_tokenizer_decoder_sequence_t*)base_decoder;
  iree_tokenizer_decoder_sequence_state_t* state =
      (iree_tokenizer_decoder_sequence_state_t*)storage;

  memset(state, 0, sizeof(*state));
  state->base.decoder = base_decoder;
  state->child_count = decoder->child_count;

  // Initialize each child state in the contiguous storage.
  for (iree_host_size_t i = 0; i < decoder->child_count; ++i) {
    void* child_storage = (uint8_t*)storage + decoder->child_state_offsets[i];
    iree_tokenizer_decoder_state_t* child_state = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_tokenizer_decoder_state_initialize(
                decoder->children[i], child_storage, &child_state));
  }

  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_decoder_sequence_state_deinitialize(
    iree_tokenizer_decoder_state_t* base_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_decoder_sequence_state_t* state =
      (iree_tokenizer_decoder_sequence_state_t*)base_state;
  const iree_tokenizer_decoder_sequence_t* decoder =
      (const iree_tokenizer_decoder_sequence_t*)base_state->decoder;
  // Deinitialize child states in reverse order.
  for (iree_host_size_t i = state->child_count; i > 0; --i) {
    iree_tokenizer_decoder_state_t* child_state =
        (iree_tokenizer_decoder_state_t*)((uint8_t*)state +
                                          decoder->child_state_offsets[i - 1]);
    iree_tokenizer_decoder_state_deinitialize(child_state);
  }
  IREE_TRACE_ZONE_END(z0);
}

// Gets the child state at index |i|.
static inline iree_tokenizer_decoder_state_t*
iree_tokenizer_decoder_sequence_get_child_state(
    iree_tokenizer_decoder_sequence_state_t* state,
    const iree_tokenizer_decoder_sequence_t* decoder, iree_host_size_t i) {
  return (iree_tokenizer_decoder_state_t*)((uint8_t*)state +
                                           decoder->child_state_offsets[i]);
}

static iree_status_t iree_tokenizer_decoder_sequence_state_process(
    iree_tokenizer_decoder_state_t* base_state,
    iree_tokenizer_string_list_t token_strings,
    iree_mutable_string_view_t output, iree_host_size_t* out_strings_consumed,
    iree_host_size_t* out_bytes_written) {
  iree_tokenizer_decoder_sequence_state_t* state =
      (iree_tokenizer_decoder_sequence_state_t*)base_state;
  const iree_tokenizer_decoder_sequence_t* decoder =
      (const iree_tokenizer_decoder_sequence_t*)base_state->decoder;

  IREE_ASSERT(decoder->child_count >= 2);

  // Step 1: First decoder processes full token batch -> output buffer.
  iree_tokenizer_decoder_state_t* first_state =
      iree_tokenizer_decoder_sequence_get_child_state(state, decoder, 0);
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_decoder_state_process(
      first_state, token_strings, output, &consumed, &written));

  // Step 2: Subsequent decoders transform output buffer in-place.
  // Each sees the ENTIRE previous output as a single string.
  for (iree_host_size_t i = 1; i < decoder->child_count; ++i) {
    iree_tokenizer_decoder_state_t* child_state =
        iree_tokenizer_decoder_sequence_get_child_state(state, decoder, i);

    // View current output as single-string input.
    iree_string_view_t intermediate =
        iree_make_string_view(output.data, written);
    iree_tokenizer_string_list_t input = {1, &intermediate};

    // Transform in-place. Since all decoders are shrinking, new_written <=
    // written.
    iree_host_size_t dummy_consumed = 0;
    iree_host_size_t new_written = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_decoder_state_process(
        child_state, input, output, &dummy_consumed, &new_written));
    written = new_written;
  }

  *out_strings_consumed = consumed;  // From first decoder only.
  *out_bytes_written = written;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_decoder_sequence_state_finalize(
    iree_tokenizer_decoder_state_t* base_state,
    iree_mutable_string_view_t output, iree_host_size_t* out_written) {
  iree_tokenizer_decoder_sequence_state_t* state =
      (iree_tokenizer_decoder_sequence_state_t*)base_state;
  const iree_tokenizer_decoder_sequence_t* decoder =
      (const iree_tokenizer_decoder_sequence_t*)base_state->decoder;

  // Finalize first decoder to get any pending data.
  if (decoder->child_count == 0) {
    *out_written = 0;
    return iree_ok_status();
  }

  iree_tokenizer_decoder_state_t* first_state =
      iree_tokenizer_decoder_sequence_get_child_state(state, decoder, 0);
  iree_host_size_t written = 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_decoder_state_finalize(first_state, output, &written));

  // Pass finalized output through remaining decoders.
  for (iree_host_size_t i = 1; i < decoder->child_count; ++i) {
    iree_tokenizer_decoder_state_t* child_state =
        iree_tokenizer_decoder_sequence_get_child_state(state, decoder, i);

    // First, finalize this child to flush its pending data.
    iree_host_size_t child_finalize_written = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_decoder_state_finalize(
        child_state,
        iree_make_mutable_string_view(output.data + written,
                                      output.size - written),
        &child_finalize_written));
    written += child_finalize_written;
  }

  *out_written = written;
  return iree_ok_status();
}

static bool iree_tokenizer_decoder_sequence_state_has_pending(
    const iree_tokenizer_decoder_state_t* base_state) {
  const iree_tokenizer_decoder_sequence_state_t* state =
      (const iree_tokenizer_decoder_sequence_state_t*)base_state;
  const iree_tokenizer_decoder_sequence_t* decoder =
      (const iree_tokenizer_decoder_sequence_t*)base_state->decoder;

  // Any child with pending data means the sequence has pending data.
  for (iree_host_size_t i = 0; i < decoder->child_count; ++i) {
    const iree_tokenizer_decoder_state_t* child_state =
        (const iree_tokenizer_decoder_state_t*)((const uint8_t*)state +
                                                decoder
                                                    ->child_state_offsets[i]);
    if (iree_tokenizer_decoder_state_has_pending(child_state)) {
      return true;
    }
  }
  return false;
}

static const iree_tokenizer_decoder_vtable_t
    iree_tokenizer_decoder_sequence_vtable = {
        .destroy = iree_tokenizer_decoder_sequence_destroy,
        .state_initialize = iree_tokenizer_decoder_sequence_state_initialize,
        .state_deinitialize =
            iree_tokenizer_decoder_sequence_state_deinitialize,
        .state_process = iree_tokenizer_decoder_sequence_state_process,
        .state_finalize = iree_tokenizer_decoder_sequence_state_finalize,
        .state_has_pending = iree_tokenizer_decoder_sequence_state_has_pending,
        .flags = IREE_TOKENIZER_DECODER_FLAG_SHRINKING,
};
