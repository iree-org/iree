// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/deferred.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Deferred Normalizer Implementation
//===----------------------------------------------------------------------===//

// Deferred normalizer: buffers input during process(), emits during finalize().
typedef struct iree_tokenizer_normalizer_deferred_t {
  iree_tokenizer_normalizer_t base;
  iree_allocator_t allocator;
} iree_tokenizer_normalizer_deferred_t;

// Deferred state: internal buffer for accumulating input.
typedef struct iree_tokenizer_normalizer_deferred_state_t {
  iree_tokenizer_normalizer_state_t base;
  uint8_t buffer[IREE_TOKENIZER_NORMALIZER_DEFERRED_BUFFER_SIZE];
  iree_host_size_t stored_count;   // Bytes accumulated during process().
  iree_host_size_t emitted_count;  // Bytes emitted during finalize().
} iree_tokenizer_normalizer_deferred_state_t;

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_deferred_vtable;

iree_status_t iree_tokenizer_normalizer_deferred_allocate(
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_normalizer);
  *out_normalizer = NULL;

  iree_tokenizer_normalizer_deferred_t* normalizer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*normalizer),
                                (void**)&normalizer));

  iree_tokenizer_normalizer_initialize(
      &normalizer->base, &iree_tokenizer_normalizer_deferred_vtable,
      sizeof(iree_tokenizer_normalizer_deferred_state_t));
  normalizer->allocator = allocator;

  *out_normalizer = &normalizer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_deferred_destroy(
    iree_tokenizer_normalizer_t* normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_deferred_t* self =
      (iree_tokenizer_normalizer_deferred_t*)normalizer;
  iree_allocator_t allocator = self->allocator;
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_normalizer_deferred_state_initialize(
    const iree_tokenizer_normalizer_t* normalizer, void* storage,
    iree_tokenizer_normalizer_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_deferred_state_t* state =
      (iree_tokenizer_normalizer_deferred_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.normalizer = normalizer;
  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_deferred_state_deinitialize(
    iree_tokenizer_normalizer_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_normalizer_deferred_state_process(
    iree_tokenizer_normalizer_state_t* base_state, iree_string_view_t input,
    iree_mutable_string_view_t output, iree_tokenizer_normalizer_flags_t flags,
    iree_host_size_t* IREE_RESTRICT out_consumed,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_tokenizer_normalizer_deferred_state_t* state =
      (iree_tokenizer_normalizer_deferred_state_t*)base_state;
  (void)output;  // We never write during process().
  (void)flags;   // No flag handling needed.

  // Consume as much as fits in our buffer.
  iree_host_size_t available =
      IREE_TOKENIZER_NORMALIZER_DEFERRED_BUFFER_SIZE - state->stored_count;
  iree_host_size_t to_consume = iree_min(input.size, available);

  if (to_consume > 0) {
    memcpy(state->buffer + state->stored_count, input.data, to_consume);
    state->stored_count += to_consume;
  }

  *out_consumed = to_consume;
  *out_written = 0;  // Never produce output during process().
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_normalizer_deferred_state_finalize(
    iree_tokenizer_normalizer_state_t* base_state,
    iree_mutable_string_view_t output,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_tokenizer_normalizer_deferred_state_t* state =
      (iree_tokenizer_normalizer_deferred_state_t*)base_state;

  // Emit as much buffered data as output capacity allows.
  iree_host_size_t remaining = state->stored_count - state->emitted_count;
  iree_host_size_t to_emit = iree_min(remaining, output.size);

  if (to_emit > 0) {
    memcpy(output.data, state->buffer + state->emitted_count, to_emit);
    state->emitted_count += to_emit;
  }

  *out_written = to_emit;
  return iree_ok_status();
}

static bool iree_tokenizer_normalizer_deferred_state_has_pending(
    const iree_tokenizer_normalizer_state_t* base_state) {
  const iree_tokenizer_normalizer_deferred_state_t* state =
      (const iree_tokenizer_normalizer_deferred_state_t*)base_state;
  return state->stored_count > state->emitted_count;
}

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_deferred_vtable = {
        .destroy = iree_tokenizer_normalizer_deferred_destroy,
        .state_initialize = iree_tokenizer_normalizer_deferred_state_initialize,
        .state_deinitialize =
            iree_tokenizer_normalizer_deferred_state_deinitialize,
        .state_process = iree_tokenizer_normalizer_deferred_state_process,
        .state_finalize = iree_tokenizer_normalizer_deferred_state_finalize,
        .state_has_pending =
            iree_tokenizer_normalizer_deferred_state_has_pending,
};
