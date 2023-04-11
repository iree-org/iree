// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/hal/utils/buffer_diagnostics.h"

#include <stdio.h>
#include <string.h>

//===----------------------------------------------------------------------===//
// iree_hal_buffer_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_modules_buffer_assert(
    iree_vm_ref_t buffer_ref, iree_vm_ref_t message_ref,
    iree_device_size_t minimum_length,
    iree_hal_memory_type_t required_memory_types,
    iree_hal_buffer_usage_t required_buffer_usage) {
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(buffer_ref, &buffer));
  iree_vm_buffer_t* message = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(message_ref, &message));
  iree_string_view_t message_str IREE_ATTRIBUTE_UNUSED =
      iree_vm_buffer_as_string(message);

  // Ensure we have enough bytes in the buffer for the encoding we have.
  // Note that having more bytes is fine:
  //   assert(expected_length <= actual_length);
  iree_device_size_t actual_length = iree_hal_buffer_byte_length(buffer);
  if (actual_length < minimum_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s buffer byte length %" PRIdsz
                            " less than expected minimum %" PRIdsz,
                            (int)message_str.size, message_str.data,
                            actual_length, minimum_length);
  }

  // All memory type bits expected (indicating where the program intends to use
  // the buffer data) must be set in the buffer while the buffer is allowed to
  // have more bits.
  iree_hal_memory_type_t actual_memory_type =
      iree_hal_buffer_memory_type(buffer);
  if (!iree_all_bits_set(actual_memory_type, required_memory_types)) {
#if ((IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) != 0) && \
    IREE_HAL_MODULE_STRING_UTIL_ENABLE
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t actual_memory_type_str =
        iree_hal_memory_type_format(actual_memory_type, &temp0);
    iree_string_view_t expected_memory_type_str =
        iree_hal_memory_type_format(required_memory_types, &temp1);
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "%.*s buffer memory type is not compatible; buffer has %.*s, operation "
        "requires %.*s",
        (int)message_str.size, message_str.data,
        (int)actual_memory_type_str.size, actual_memory_type_str.data,
        (int)expected_memory_type_str.size, expected_memory_type_str.data);
#else
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "%.*s buffer memory type is not compatible; buffer has %08X, operation "
        "requires %08X",
        (int)message_str.size, message_str.data, actual_memory_type,
        expected_memory_type);
#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE
  }

  // All usage bits expected (indicating what the program intends to use the
  // buffer for) must be set in the buffer while the buffer is allowed to have
  // more bits.
  iree_hal_buffer_usage_t actual_buffer_usage =
      iree_hal_buffer_allowed_usage(buffer);
  if (!iree_all_bits_set(actual_buffer_usage, required_buffer_usage)) {
#if ((IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) != 0) && \
    IREE_HAL_MODULE_STRING_UTIL_ENABLE
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t allowed_usage_str =
        iree_hal_buffer_usage_format(actual_buffer_usage, &temp0);
    iree_string_view_t required_usage_str =
        iree_hal_buffer_usage_format(required_buffer_usage, &temp1);
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "%.*s requested usage was not specified when the buffer was allocated; "
        "buffer allows %.*s, operation requires %.*s",
        (int)message_str.size, message_str.data, (int)allowed_usage_str.size,
        allowed_usage_str.data, (int)required_usage_str.size,
        required_usage_str.data);
#else
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "%.*s requested usage was not specified when the buffer was allocated; "
        "buffer allows %08X, operation requires %08X",
        (int)message_str.size, message_str.data, allowed_buffer_usage,
        required_buffer_usage);
#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_t
//===----------------------------------------------------------------------===//

// Returns true if the |expected_type| can be satisfied with |actual_type|.
// This allows for basic type widening and bypassing instead of requiring an
// exact match in all cases.
static bool iree_hal_element_types_are_compatible(
    iree_hal_element_type_t actual_type,
    iree_hal_element_type_t expected_type) {
  if (iree_hal_element_numerical_type_is_opaque(actual_type)) {
    // If the provided type is opaque it can map to anything. This allows
    // applications to bypass the checks when they are treating all the data as
    // opaque, such as when carrying around buffer data in binary blobs.
    return true;
  }

  if (iree_hal_element_numerical_type_is_integer(actual_type) &&
      iree_hal_element_numerical_type_is_integer(expected_type) &&
      iree_hal_element_bit_count(actual_type) ==
          iree_hal_element_bit_count(expected_type)) {
    // Integer types of the same bit width are allowed to be cast.
    // This allows users or the compiler to treat data as signless while still
    // allowing signedness. For example, tensor<1xi32> can successfully match
    // a tensor<1xui32> expectation.
    return true;
  }

  // Otherwise we require an exact match. This may be overly conservative but
  // in most cases is a useful error message. Users can pass in OPAQUE types if
  // hitting this to bypass.
  return actual_type == expected_type;
}

iree_status_t iree_hal_modules_buffer_view_assert(
    iree_vm_ref_t buffer_view_ref, iree_vm_ref_t message_ref,
    iree_hal_element_type_t expected_element_type,
    iree_hal_encoding_type_t expected_encoding_type,
    iree_host_size_t expected_shape_rank,
    const iree_hal_dim_t* expected_shape_dims) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(buffer_view_ref, &buffer_view));
  iree_vm_buffer_t* message = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(message_ref, &message));
  iree_string_view_t message_str IREE_ATTRIBUTE_UNUSED =
      iree_vm_buffer_as_string(message);

  // Check encoding first; getting the encoding wrong is worse than the shape.
  // If the actual encoding is opaque we allow it to pass through - this lets
  // users override the assertion in the case where they are just passing data
  // around and don't care about the contents.
  iree_hal_encoding_type_t actual_encoding_type =
      iree_hal_buffer_view_encoding_type(buffer_view);
  if (actual_encoding_type != IREE_HAL_ENCODING_TYPE_OPAQUE &&
      actual_encoding_type != expected_encoding_type) {
    // TODO(benvanik): string formatting of encodings.
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s encoding mismatch; expected %08X but have %08X",
        (int)message_str.size, message_str.data, expected_encoding_type,
        actual_encoding_type);
  }

  // Element types determine the storage requirements.
  // If the actual element type is opaque we allow it to pass through.
  iree_hal_element_type_t actual_element_type =
      iree_hal_buffer_view_element_type(buffer_view);
  if (!iree_hal_element_types_are_compatible(actual_element_type,
                                             expected_element_type)) {
#if ((IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) != 0) && \
    IREE_HAL_MODULE_STRING_UTIL_ENABLE
    char actual_element_type_str[32];
    iree_host_size_t actual_element_type_str_length = 0;
    char expected_element_type_str[32];
    iree_host_size_t expected_element_type_str_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_format_element_type(
        actual_element_type, sizeof(actual_element_type_str),
        actual_element_type_str, &actual_element_type_str_length));
    IREE_RETURN_IF_ERROR(iree_hal_format_element_type(
        expected_element_type, sizeof(expected_element_type_str),
        expected_element_type_str, &expected_element_type_str_length));
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s element type mismatch; expected %.*s (%08X) but have %.*s (%08X)",
        (int)message_str.size, message_str.data,
        (int)expected_element_type_str_length, expected_element_type_str,
        expected_element_type, (int)actual_element_type_str_length,
        actual_element_type_str, actual_element_type);
#else
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s element type mismatch; expected %08X but have %08X",
        (int)message_str.size, message_str.data, expected_element_type,
        actual_element_type);
#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE
  }

  // Rank check before the individual shape dimensions.
  iree_host_size_t actual_shape_rank =
      iree_hal_buffer_view_shape_rank(buffer_view);
  const iree_hal_dim_t* actual_shape_dims =
      iree_hal_buffer_view_shape_dims(buffer_view);
  iree_status_t shape_status = iree_ok_status();
  if (actual_shape_rank != expected_shape_rank) {
    shape_status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s shape rank mismatch; expected %" PRIhsz "%s but have %" PRIhsz
        "%s",
        (int)message_str.size, message_str.data, expected_shape_rank,
        expected_shape_rank == 0 ? " (scalar)" : "", actual_shape_rank,
        actual_shape_rank == 0 ? " (scalar)" : "");
  }
  if (iree_status_is_ok(shape_status)) {
    for (iree_host_size_t i = 0; i < actual_shape_rank; ++i) {
      if (actual_shape_dims[i] == expected_shape_dims[i]) continue;
      // Dimension mismatch.
      shape_status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "%.*s shape dimension %" PRIhsz
                           " mismatch; expected %" PRIdim " but have %" PRIdim,
                           (int)message_str.size, message_str.data, i,
                           expected_shape_dims[i], actual_shape_dims[i]);
      break;
    }
  }

#if ((IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) != 0) && \
    IREE_HAL_MODULE_STRING_UTIL_ENABLE
  if (!iree_status_is_ok(shape_status)) {
    char actual_shape_str[32];
    iree_host_size_t actual_shape_str_length = 0;
    char expected_shape_str[32];
    iree_host_size_t expected_shape_str_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_format_shape(
        actual_shape_rank, actual_shape_dims, sizeof(actual_shape_str),
        actual_shape_str, &actual_shape_str_length));
    IREE_RETURN_IF_ERROR(iree_hal_format_shape(
        expected_shape_rank, expected_shape_dims, sizeof(expected_shape_str),
        expected_shape_str, &expected_shape_str_length));
    shape_status = iree_status_annotate_f(
        shape_status, "expected shape `%.*s`, actual shape `%.*s`",
        (int)expected_shape_str_length, expected_shape_str,
        (int)actual_shape_str_length, actual_shape_str);
  }
#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE

  return shape_status;
}

iree_status_t iree_hal_modules_buffer_view_trace(
    iree_vm_ref_t key_ref, iree_vm_size_t buffer_view_count,
    iree_vm_abi_r_t* buffer_view_refs, iree_allocator_t host_allocator) {
#if IREE_HAL_MODULE_STRING_UTIL_ENABLE

  iree_vm_buffer_t* key = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(key_ref, &key));
  iree_string_view_t key_str = iree_vm_buffer_as_string(key);

  fprintf(stderr, "=== %.*s ===\n", (int)key_str.size, key_str.data);
  for (iree_host_size_t i = 0; i < buffer_view_count; ++i) {
    iree_hal_buffer_view_t* buffer_view = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_check_deref(buffer_view_refs[i].r0, &buffer_view));

    // NOTE: this export is for debugging only and a no-op in min-size builds.
    // We heap-alloc here because at the point this export is used performance
    // is not a concern.

    // Query total length (excluding NUL terminator).
    iree_host_size_t result_length = 0;
    iree_status_t status = iree_hal_buffer_view_format(buffer_view, SIZE_MAX, 0,
                                                       NULL, &result_length);
    if (!iree_status_is_out_of_range(status)) {
      return status;
    }
    ++result_length;  // include NUL

    // Allocate scratch heap memory to contain the result and format into it.
    char* result_str = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, result_length,
                                               (void**)&result_str));
    status = iree_hal_buffer_view_format(buffer_view, SIZE_MAX, result_length,
                                         result_str, &result_length);
    if (iree_status_is_ok(status)) {
      fprintf(stderr, "%.*s\n", (int)result_length, result_str);
    }
    iree_allocator_free(host_allocator, result_str);
    IREE_RETURN_IF_ERROR(status);
  }
  fprintf(stderr, "\n");

#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE
  return iree_ok_status();
}
