// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/segmenter/passthrough.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Passthrough Segmenter Implementation
//===----------------------------------------------------------------------===//

// Passthrough segmenter: no additional config beyond base.
typedef struct iree_tokenizer_segmenter_passthrough_t {
  iree_tokenizer_segmenter_t base;
  iree_allocator_t allocator;
} iree_tokenizer_segmenter_passthrough_t;

// Passthrough state: just the base (no buffering needed).
typedef struct iree_tokenizer_segmenter_passthrough_state_t {
  iree_tokenizer_segmenter_state_t base;
} iree_tokenizer_segmenter_passthrough_state_t;

static const iree_tokenizer_segmenter_vtable_t
    iree_tokenizer_segmenter_passthrough_vtable;

iree_status_t iree_tokenizer_segmenter_passthrough_allocate(
    iree_allocator_t allocator, iree_tokenizer_segmenter_t** out_segmenter) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_segmenter);
  *out_segmenter = NULL;

  iree_tokenizer_segmenter_passthrough_t* segmenter = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(allocator, sizeof(*segmenter), (void**)&segmenter));

  iree_tokenizer_segmenter_initialize(
      &segmenter->base, &iree_tokenizer_segmenter_passthrough_vtable,
      sizeof(iree_tokenizer_segmenter_passthrough_state_t));
  segmenter->allocator = allocator;

  *out_segmenter = &segmenter->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_segmenter_passthrough_destroy(
    iree_tokenizer_segmenter_t* segmenter) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_segmenter_passthrough_t* self =
      (iree_tokenizer_segmenter_passthrough_t*)segmenter;
  iree_allocator_t allocator = self->allocator;
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_segmenter_passthrough_state_initialize(
    const iree_tokenizer_segmenter_t* segmenter, void* storage,
    iree_tokenizer_segmenter_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_segmenter_passthrough_state_t* state =
      (iree_tokenizer_segmenter_passthrough_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.segmenter = segmenter;
  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_segmenter_passthrough_state_deinitialize(
    iree_tokenizer_segmenter_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_segmenter_passthrough_state_process(
    iree_tokenizer_segmenter_state_t* state, iree_string_view_t input,
    iree_tokenizer_segment_output_t output, iree_host_size_t* out_consumed,
    iree_host_size_t* out_segment_count) {
  // If no input, nothing to do.
  if (input.size == 0) {
    *out_consumed = 0;
    *out_segment_count = 0;
    return iree_ok_status();
  }

  // If no capacity for segments, consume nothing.
  if (output.capacity == 0) {
    *out_consumed = 0;
    *out_segment_count = 0;
    return iree_ok_status();
  }

  // Emit entire input as one segment.
  // Offsets are relative to input start (per offset contract in segmenter.h).
  output.values[0].start = 0;
  output.values[0].end = input.size;
  *out_consumed = input.size;
  *out_segment_count = 1;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_segmenter_passthrough_state_finalize(
    iree_tokenizer_segmenter_state_t* state, iree_string_view_t remaining_input,
    iree_tokenizer_segment_output_t output,
    iree_host_size_t* out_segment_count) {
  (void)remaining_input;  // Passthrough doesn't buffer, so nothing to flush.
  *out_segment_count = 0;
  return iree_ok_status();
}

static bool iree_tokenizer_segmenter_passthrough_state_has_pending(
    const iree_tokenizer_segmenter_state_t* state) {
  // Never has pending data (no buffering).
  return false;
}

static const iree_tokenizer_segmenter_vtable_t
    iree_tokenizer_segmenter_passthrough_vtable = {
        .destroy = iree_tokenizer_segmenter_passthrough_destroy,
        .state_initialize =
            iree_tokenizer_segmenter_passthrough_state_initialize,
        .state_deinitialize =
            iree_tokenizer_segmenter_passthrough_state_deinitialize,
        .state_process = iree_tokenizer_segmenter_passthrough_state_process,
        .state_finalize = iree_tokenizer_segmenter_passthrough_state_finalize,
        .state_has_pending =
            iree_tokenizer_segmenter_passthrough_state_has_pending,
};
