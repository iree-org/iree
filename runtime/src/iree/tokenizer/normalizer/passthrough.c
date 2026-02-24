// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/passthrough.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Passthrough Normalizer Implementation
//===----------------------------------------------------------------------===//

// Passthrough normalizer: no additional config beyond base.
typedef struct iree_tokenizer_normalizer_passthrough_t {
  iree_tokenizer_normalizer_t base;
  iree_allocator_t allocator;
} iree_tokenizer_normalizer_passthrough_t;

// Passthrough state: just the base (no buffering needed).
typedef struct iree_tokenizer_normalizer_passthrough_state_t {
  iree_tokenizer_normalizer_state_t base;
} iree_tokenizer_normalizer_passthrough_state_t;

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_passthrough_vtable;

iree_status_t iree_tokenizer_normalizer_passthrough_allocate(
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_normalizer);
  *out_normalizer = NULL;

  iree_tokenizer_normalizer_passthrough_t* normalizer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*normalizer),
                                (void**)&normalizer));

  iree_tokenizer_normalizer_initialize(
      &normalizer->base, &iree_tokenizer_normalizer_passthrough_vtable,
      sizeof(iree_tokenizer_normalizer_passthrough_state_t));
  normalizer->allocator = allocator;

  *out_normalizer = &normalizer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_passthrough_destroy(
    iree_tokenizer_normalizer_t* normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_passthrough_t* self =
      (iree_tokenizer_normalizer_passthrough_t*)normalizer;
  iree_allocator_t allocator = self->allocator;
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_normalizer_passthrough_state_initialize(
    const iree_tokenizer_normalizer_t* normalizer, void* storage,
    iree_tokenizer_normalizer_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_passthrough_state_t* state =
      (iree_tokenizer_normalizer_passthrough_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.normalizer = normalizer;
  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_passthrough_state_deinitialize(
    iree_tokenizer_normalizer_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_normalizer_passthrough_state_process(
    iree_tokenizer_normalizer_state_t* state, iree_string_view_t input,
    iree_mutable_string_view_t output, iree_tokenizer_normalizer_flags_t flags,
    iree_host_size_t* IREE_RESTRICT out_consumed,
    iree_host_size_t* IREE_RESTRICT out_written) {
  (void)flags;  // Passthrough has no state to reset on segment boundaries.

  // Copy as much as we can fit.
  iree_host_size_t to_copy =
      input.size < output.size ? input.size : output.size;
  if (to_copy > 0) {
    memcpy(output.data, input.data, to_copy);
  }
  *out_consumed = to_copy;
  *out_written = to_copy;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_normalizer_passthrough_state_finalize(
    iree_tokenizer_normalizer_state_t* state, iree_mutable_string_view_t output,
    iree_host_size_t* IREE_RESTRICT out_written) {
  // No buffering, nothing to flush.
  *out_written = 0;
  return iree_ok_status();
}

static bool iree_tokenizer_normalizer_passthrough_state_has_pending(
    const iree_tokenizer_normalizer_state_t* state) {
  // Never has pending data (no buffering).
  return false;
}

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_passthrough_vtable = {
        .destroy = iree_tokenizer_normalizer_passthrough_destroy,
        .state_initialize =
            iree_tokenizer_normalizer_passthrough_state_initialize,
        .state_deinitialize =
            iree_tokenizer_normalizer_passthrough_state_deinitialize,
        .state_process = iree_tokenizer_normalizer_passthrough_state_process,
        .state_finalize = iree_tokenizer_normalizer_passthrough_state_finalize,
        .state_has_pending =
            iree_tokenizer_normalizer_passthrough_state_has_pending,
};
