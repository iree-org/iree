// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/string_builder.h"

#include "iree/base/alignment.h"
#include "iree/base/printf.h"

// Minimum alignment for storage buffer allocations.
#define IREE_STRING_BUILDER_ALIGNMENT 128

IREE_API_EXPORT void iree_string_builder_initialize(
    iree_allocator_t allocator, iree_string_builder_t* out_builder) {
  memset(out_builder, 0, sizeof(*out_builder));
  out_builder->allocator = allocator;
}

IREE_API_EXPORT void iree_string_builder_initialize_with_storage(
    char* buffer, iree_host_size_t buffer_capacity,
    iree_string_builder_t* out_builder) {
  iree_string_builder_initialize(iree_allocator_null(), out_builder);
  out_builder->buffer = buffer;
  out_builder->capacity = buffer_capacity;
}

IREE_API_EXPORT void iree_string_builder_deinitialize(
    iree_string_builder_t* builder) {
  if (builder->buffer != NULL) {
    iree_allocator_free(builder->allocator, builder->buffer);
  }
  memset(builder, 0, sizeof(*builder));
}

static bool iree_string_builder_is_calculating_size(
    const iree_string_builder_t* builder) {
  return iree_allocator_is_null(builder->allocator) && builder->buffer == NULL;
}

IREE_API_EXPORT const char* iree_string_builder_buffer(
    const iree_string_builder_t* builder) {
  return builder->buffer;
}

IREE_API_EXPORT iree_host_size_t
iree_string_builder_size(const iree_string_builder_t* builder) {
  return builder->size;
}

IREE_API_EXPORT iree_host_size_t
iree_string_builder_capacity(const iree_string_builder_t* builder) {
  return builder->capacity;
}

IREE_API_EXPORT iree_string_view_t
iree_string_builder_view(const iree_string_builder_t* builder) {
  return iree_make_string_view(iree_string_builder_buffer(builder),
                               iree_string_builder_size(builder));
}

IREE_API_EXPORT char* iree_string_builder_take_storage(
    iree_string_builder_t* builder) {
  char* buffer = builder->buffer;
  if (builder->size == 0) {
    // In empty cases we return NULL and need to clean up inline as the user is
    // expecting to be able to discard the builder after this returns.
    if (builder->buffer != NULL) {
      iree_allocator_free(builder->allocator, builder->buffer);
      builder->buffer = NULL;
    }
    buffer = NULL;
  }
  builder->size = 0;
  builder->capacity = 0;
  builder->buffer = NULL;
  return buffer;
}

IREE_API_EXPORT iree_status_t iree_string_builder_reserve(
    iree_string_builder_t* builder, iree_host_size_t minimum_capacity) {
  if (IREE_LIKELY(builder->capacity >= minimum_capacity)) {
    // Already at/above the requested minimum capacity.
    return iree_ok_status();
  }

  // If no allocator was provided the builder cannot grow.
  if (iree_allocator_is_null(builder->allocator)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "non-growable builder capacity exceeded (capacity=%" PRIhsz
        "; requested>=%" PRIhsz ")",
        builder->capacity, minimum_capacity);
  }

  // Grow by 2x. Note that the current capacity may be zero.
  iree_host_size_t doubled_capacity = 0;
  if (!iree_host_size_checked_mul(builder->capacity, 2, &doubled_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "capacity overflow");
  }
  iree_host_size_t aligned_minimum = 0;
  if (!iree_host_size_checked_align(
          minimum_capacity, IREE_STRING_BUILDER_ALIGNMENT, &aligned_minimum)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "capacity overflow");
  }
  iree_host_size_t new_capacity = iree_max(doubled_capacity, aligned_minimum);
  IREE_RETURN_IF_ERROR(iree_allocator_realloc(builder->allocator, new_capacity,
                                              (void**)&builder->buffer));
  builder->buffer[builder->size] = 0;
  builder->capacity = new_capacity;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_string_builder_reserve_for_append(
    iree_string_builder_t* builder,
    iree_host_size_t minimum_additional_capacity, char** out_buffer,
    iree_host_size_t* out_capacity) {
  iree_host_size_t new_capacity = 0;
  if (!iree_host_size_checked_add(builder->size, minimum_additional_capacity,
                                  &new_capacity) ||
      !iree_host_size_checked_add(new_capacity, /*NUL=*/1, &new_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "capacity overflow");
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_reserve(builder, new_capacity));
  *out_buffer = builder->buffer + builder->size;
  *out_capacity = builder->capacity - builder->size - /*NUL=*/1;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_string_builder_commit_append(
    iree_string_builder_t* builder, iree_host_size_t append_size) {
  builder->size += append_size;
  builder->buffer[builder->size] = 0;
}

IREE_API_EXPORT void iree_string_builder_reset(iree_string_builder_t* builder) {
  builder->size = 0;
}

IREE_API_EXPORT iree_status_t iree_string_builder_append_inline(
    iree_string_builder_t* builder, iree_host_size_t count, char** out_head) {
  *out_head = NULL;
  if (!iree_string_builder_is_calculating_size(builder)) {
    iree_host_size_t required_capacity = 0;
    if (!iree_host_size_checked_add(builder->size, count, &required_capacity) ||
        !iree_host_size_checked_add(required_capacity, /*NUL=*/1,
                                    &required_capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "capacity overflow");
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_reserve(builder, required_capacity));
    *out_head = &builder->buffer[builder->size];
  }
  builder->size += count;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_string_builder_append_string(
    iree_string_builder_t* builder, iree_string_view_t value) {
  // Ensure capacity for the value + NUL terminator.
  if (!iree_string_builder_is_calculating_size(builder)) {
    iree_host_size_t required_capacity = 0;
    if (!iree_host_size_checked_add(builder->size, value.size,
                                    &required_capacity) ||
        !iree_host_size_checked_add(required_capacity, /*NUL=*/1,
                                    &required_capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "capacity overflow");
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_reserve(builder, required_capacity));
    // Only copy the bytes if we are not doing a size calculation.
    memcpy(builder->buffer + builder->size, value.data, value.size);
    builder->buffer[builder->size + value.size] = 0;  // NUL
  }
  builder->size += value.size;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_string_builder_append_cstring(
    iree_string_builder_t* builder, const char* value) {
  return iree_string_builder_append_string(builder,
                                           iree_make_cstring_view(value));
}

// State threaded through the fctprintf callback during formatting.
typedef struct iree_string_builder_printf_state_t {
  iree_string_builder_t* builder;
  iree_status_t status;
} iree_string_builder_printf_state_t;

// Per-character callback from iree_vfctprintf that appends to the builder.
// Handles both size-calculation mode (no buffer, just counting) and normal mode
// (writing characters and growing the buffer as needed).
static void iree_string_builder_printf_out(char character, void* user_data) {
  iree_string_builder_printf_state_t* state =
      (iree_string_builder_printf_state_t*)user_data;
  iree_string_builder_t* builder = state->builder;

  // Size-calculation mode: just count characters, no buffer operations.
  if (iree_string_builder_is_calculating_size(builder)) {
    builder->size++;
    return;
  }

  // Bail if a previous callback already failed (allocation error).
  if (!iree_status_is_ok(state->status)) return;

  // Ensure capacity for this character + NUL terminator.
  if (builder->size + 1 >= builder->capacity) {
    state->status = iree_string_builder_reserve(builder, builder->size + 2);
    if (!iree_status_is_ok(state->status)) return;
  }
  builder->buffer[builder->size++] = character;
}

IREE_API_EXPORT iree_status_t IREE_PRINTF_ATTRIBUTE(2, 3)
    iree_string_builder_append_format(iree_string_builder_t* builder,
                                      const char* format, ...) {
  va_list varargs;
  va_start(varargs, format);
  iree_string_builder_printf_state_t state = {builder, iree_ok_status()};
  iree_vfctprintf(iree_string_builder_printf_out, &state, format, varargs);
  va_end(varargs);
  // NUL-terminate the buffer (fctprintf does not emit a terminator).
  if (!iree_string_builder_is_calculating_size(builder) && builder->buffer) {
    builder->buffer[builder->size] = '\0';
  }
  return state.status;
}

// Callback for iree_string_builder_append_status that appends chunks to a
// string builder.
static bool iree_string_builder_append_status_output(iree_string_view_t chunk,
                                                     void* user_data) {
  iree_string_builder_t* builder = (iree_string_builder_t*)user_data;
  iree_status_t status = iree_string_builder_append_string(builder, chunk);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return false;
  }
  return true;
}

IREE_API_EXPORT iree_status_t iree_string_builder_append_status(
    iree_string_builder_t* builder, iree_status_t status) {
  if (!iree_status_format_to(status, iree_string_builder_append_status_output,
                             builder)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to append formatted status to builder");
  }
  return iree_ok_status();
}

IREE_API_EXPORT void iree_string_pair_builder_initialize(
    iree_allocator_t allocator, iree_string_pair_builder_t* out_builder) {
  memset(out_builder, 0, sizeof(*out_builder));
  out_builder->allocator = allocator;
}

IREE_API_EXPORT void iree_string_pair_builder_deinitialize(
    iree_string_pair_builder_t* builder) {
  for (iree_host_size_t i = 0; i < builder->temp_strings_size; ++i) {
    iree_allocator_free(builder->allocator, builder->temp_strings[i]);
  }
  iree_allocator_free(builder->allocator, builder->pairs);
  iree_allocator_free(builder->allocator, builder->temp_strings);
}

IREE_API_EXPORT iree_status_t iree_string_pair_builder_add(
    iree_string_pair_builder_t* builder, iree_string_pair_t pair) {
  if (builder->pairs_size == builder->pairs_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        builder->allocator, 8, sizeof(builder->pairs[0]),
        &builder->pairs_capacity, (void**)&builder->pairs));
  }
  builder->pairs[builder->pairs_size++] = pair;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_string_pair_builder_add_int32(iree_string_pair_builder_t* builder,
                                   iree_string_view_t key, int32_t value) {
  char temp[32];
  iree_snprintf(temp, sizeof(temp), "%d", value);
  iree_string_view_t value_string = iree_make_cstring_view(temp);
  IREE_RETURN_IF_ERROR(
      iree_string_pair_builder_emplace_string(builder, &value_string));
  return iree_string_pair_builder_add(builder,
                                      iree_make_string_pair(key, value_string));
}

IREE_API_EXPORT iree_status_t iree_string_pair_builder_emplace_string(
    iree_string_pair_builder_t* builder, iree_string_view_t* inout_string) {
  if (builder->temp_strings_size == builder->temp_strings_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        builder->allocator, 8, sizeof(builder->temp_strings[0]),
        &builder->temp_strings_capacity, (void**)&builder->temp_strings));
  }

  char* alloced = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      builder->allocator, inout_string->size + 1, (void**)&alloced));
  memcpy(alloced, inout_string->data, inout_string->size);
  alloced[inout_string->size] = 0;
  builder->temp_strings[builder->temp_strings_size++] = alloced;
  inout_string->data = alloced;
  return iree_ok_status();
}
