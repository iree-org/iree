// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/comparison.h"

#include <cstdint>
#include <cstdio>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/tooling/buffer_view_matchers.h"
#include "iree/vm/api.h"

using namespace iree;

IREE_FLAG(float, expected_f16_threshold, 0.001f,
          "Threshold under which two f16 values are considered equal.");
IREE_FLAG(float, expected_f32_threshold, 0.0001f,
          "Threshold under which two f32 values are considered equal.");
IREE_FLAG(double, expected_f64_threshold, 0.0001,
          "Threshold under which two f64 values are considered equal.");

static iree_hal_buffer_equality_t iree_tooling_equality_from_flags(void) {
  iree_hal_buffer_equality_t equality;
  equality.mode = IREE_HAL_BUFFER_EQUALITY_APPROXIMATE_ABSOLUTE;
  equality.f16_threshold = FLAG_expected_f16_threshold;
  equality.f32_threshold = FLAG_expected_f32_threshold;
  equality.f64_threshold = FLAG_expected_f64_threshold;
  return equality;
}

static iree_status_t iree_vm_append_variant_type_string(
    iree_vm_variant_t variant, iree_string_builder_t* builder) {
  if (iree_vm_variant_is_empty(variant)) {
    return iree_string_builder_append_string(builder, IREE_SV("empty"));
  } else if (iree_vm_variant_is_value(variant)) {
    const char* type = NULL;
    switch (iree_vm_type_def_as_value(variant.type)) {
      case IREE_VM_VALUE_TYPE_I8:
        type = "i8";
        break;
      case IREE_VM_VALUE_TYPE_I16:
        type = "i16";
        break;
      case IREE_VM_VALUE_TYPE_I32:
        type = "i32";
        break;
      case IREE_VM_VALUE_TYPE_I64:
        type = "i64";
        break;
      case IREE_VM_VALUE_TYPE_F32:
        type = "f32";
        break;
      case IREE_VM_VALUE_TYPE_F64:
        type = "f64";
        break;
      default:
        type = "?";
        break;
    }
    return iree_string_builder_append_cstring(builder, type);
  } else if (iree_vm_variant_is_ref(variant)) {
    return iree_string_builder_append_string(
        builder, iree_vm_ref_type_name(iree_vm_type_def_as_ref(variant.type)));
  } else {
    return iree_string_builder_append_string(builder, IREE_SV("unknown"));
  }
}

static bool iree_tooling_compare_values(int result_index,
                                        iree_vm_variant_t expected_variant,
                                        iree_vm_variant_t actual_variant,
                                        iree_string_builder_t* builder) {
  IREE_ASSERT_TRUE(
      iree_vm_type_def_equal(expected_variant.type, actual_variant.type));
  switch (iree_vm_type_def_as_value(expected_variant.type)) {
    case IREE_VM_VALUE_TYPE_I8:
      if (expected_variant.i8 != actual_variant.i8) {
        IREE_CHECK_OK(iree_string_builder_append_format(
            builder,
            "[FAILED] result[%d]: i8 values differ\n  expected: %" PRIi8
            "\n  actual: %" PRIi8 "\n",
            result_index, expected_variant.i8, actual_variant.i8));
        return false;
      }
      return true;
    case IREE_VM_VALUE_TYPE_I16:
      if (expected_variant.i16 != actual_variant.i16) {
        IREE_CHECK_OK(iree_string_builder_append_format(
            builder,
            "[FAILED] result[%d]: i16 values differ\n  expected: %" PRIi16
            "\n  actual: %" PRIi16 "\n",
            result_index, expected_variant.i16, actual_variant.i16));
        return false;
      }
      return true;
    case IREE_VM_VALUE_TYPE_I32:
      if (expected_variant.i32 != actual_variant.i32) {
        IREE_CHECK_OK(iree_string_builder_append_format(
            builder,
            "[FAILED] result[%d]: i32 values differ\n  expected: %" PRIi32
            "\n  actual: %" PRIi32 "\n",
            result_index, expected_variant.i32, actual_variant.i32));
        return false;
      }
      return true;
    case IREE_VM_VALUE_TYPE_I64:
      if (expected_variant.i64 != actual_variant.i64) {
        IREE_CHECK_OK(iree_string_builder_append_format(
            builder,
            "[FAILED] result[%d]: i64 values differ\n  expected: %" PRIi64
            "\n  actual: %" PRIi64 "\n",
            result_index, expected_variant.i64, actual_variant.i64));
        return false;
      }
      return true;
    case IREE_VM_VALUE_TYPE_F32:
      // TODO(benvanik): use tolerance flag.
      if (expected_variant.f32 != actual_variant.f32) {
        IREE_CHECK_OK(iree_string_builder_append_format(
            builder,
            "[FAILED] result[%d]: f32 values differ\n  expected: %G\n  actual: "
            "%G\n",
            result_index, expected_variant.f32, actual_variant.f32));
        return false;
      }
      return true;
    case IREE_VM_VALUE_TYPE_F64:
      // TODO(benvanik): use tolerance flag.
      if (expected_variant.f64 != actual_variant.f64) {
        IREE_CHECK_OK(iree_string_builder_append_format(
            builder,
            "[FAILED] result[%d]: f64 values differ\n  expected: %G\n  actual: "
            "%G\n",
            result_index, expected_variant.f64, actual_variant.f64));
        return false;
      }
      return true;
    default:
      IREE_CHECK_OK(iree_string_builder_append_format(
          builder, "[FAILED] result[%d]: unknown value type, cannot match\n",
          result_index));
      return false;
  }
}

static bool iree_tooling_compare_buffer_views(
    int result_index, iree_hal_buffer_view_t* expected_view,
    iree_hal_buffer_view_t* actual_view, iree_allocator_t host_allocator,
    iree_host_size_t max_element_count, iree_string_builder_t* builder) {
  iree_string_builder_t subbuilder;
  iree_string_builder_initialize(host_allocator, &subbuilder);

  iree_hal_buffer_equality_t equality = iree_tooling_equality_from_flags();
  bool did_match = false;
  IREE_CHECK_OK(iree_hal_buffer_view_match_equal(
      equality, expected_view, actual_view, &subbuilder, &did_match));
  if (did_match) {
    iree_string_builder_deinitialize(&subbuilder);
    return true;
  }
  IREE_CHECK_OK(iree_string_builder_append_format(
      builder, "[FAILED] result[%d]: ", result_index));
  IREE_CHECK_OK(iree_string_builder_append_string(
      builder, iree_string_builder_view(&subbuilder)));
  iree_string_builder_deinitialize(&subbuilder);

  IREE_CHECK_OK(
      iree_string_builder_append_string(builder, IREE_SV("\n  expected:\n")));
  IREE_CHECK_OK(iree_hal_buffer_view_append_to_builder(
      expected_view, max_element_count, builder));
  IREE_CHECK_OK(
      iree_string_builder_append_string(builder, IREE_SV("\n  actual:\n")));
  IREE_CHECK_OK(iree_hal_buffer_view_append_to_builder(
      actual_view, max_element_count, builder));
  IREE_CHECK_OK(iree_string_builder_append_string(builder, IREE_SV("\n")));

  return false;
}

static bool iree_tooling_compare_variants(int result_index,
                                          iree_vm_variant_t expected_variant,
                                          iree_vm_variant_t actual_variant,
                                          iree_allocator_t host_allocator,
                                          iree_host_size_t max_element_count,
                                          iree_string_builder_t* builder) {
  IREE_TRACE_SCOPE();

  if (iree_vm_variant_is_empty(expected_variant)) {
    return true;  // expected empty is sentinel for (ignored)
  } else if (iree_vm_variant_is_empty(actual_variant) &&
             iree_vm_variant_is_empty(expected_variant)) {
    return true;  // both empty
  } else if (iree_vm_variant_is_value(actual_variant) &&
             iree_vm_variant_is_value(expected_variant)) {
    if (!iree_vm_type_def_equal(expected_variant.type, actual_variant.type)) {
      return iree_tooling_compare_values(result_index, expected_variant,
                                         actual_variant, builder);
    }
  } else if (iree_vm_variant_is_ref(actual_variant) &&
             iree_vm_variant_is_ref(expected_variant)) {
    if (iree_hal_buffer_view_isa(actual_variant.ref) &&
        iree_hal_buffer_view_isa(expected_variant.ref)) {
      return iree_tooling_compare_buffer_views(
          result_index, iree_hal_buffer_view_deref(expected_variant.ref),
          iree_hal_buffer_view_deref(actual_variant.ref), host_allocator,
          max_element_count, builder);
    }
  }

  IREE_CHECK_OK(iree_string_builder_append_format(
      builder, "[FAILED] result[%d]: ", result_index));
  IREE_CHECK_OK(iree_string_builder_append_string(
      builder, IREE_SV("variant types mismatch; expected ")));
  IREE_CHECK_OK(iree_vm_append_variant_type_string(expected_variant, builder));
  IREE_CHECK_OK(
      iree_string_builder_append_string(builder, IREE_SV(" but got ")));
  IREE_CHECK_OK(iree_vm_append_variant_type_string(actual_variant, builder));
  IREE_CHECK_OK(iree_string_builder_append_string(builder, IREE_SV("\n")));

  return false;
}

bool iree_tooling_compare_variant_lists_and_append(
    iree_vm_list_t* expected_list, iree_vm_list_t* actual_list,
    iree_allocator_t host_allocator, iree_string_builder_t* builder) {
  IREE_TRACE_SCOPE();

  if (iree_vm_list_size(expected_list) != iree_vm_list_size(actual_list)) {
    IREE_CHECK_OK(iree_string_builder_append_format(
        builder,
        "[FAILED] expected %" PRIhsz " list elements but %" PRIhsz
        " provided\n",
        iree_vm_list_size(expected_list), iree_vm_list_size(actual_list)));
    return false;
  }

  bool all_match = true;
  for (iree_host_size_t i = 0; i < iree_vm_list_size(expected_list); ++i) {
    iree_vm_variant_t expected_variant = iree_vm_variant_empty();
    IREE_CHECK_OK(
        iree_vm_list_get_variant_assign(expected_list, i, &expected_variant));
    iree_vm_variant_t actual_variant = iree_vm_variant_empty();
    IREE_CHECK_OK(
        iree_vm_list_get_variant_assign(actual_list, i, &actual_variant));
    bool did_match = iree_tooling_compare_variants(
        (int)i, expected_variant, actual_variant, host_allocator,
        /*max_element_count=*/1024, builder);
    if (!did_match) all_match = false;
  }

  return all_match;
}

bool iree_tooling_compare_variant_lists(iree_vm_list_t* expected_list,
                                        iree_vm_list_t* actual_list,
                                        iree_allocator_t host_allocator,
                                        FILE* file) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);
  bool all_match = iree_tooling_compare_variant_lists_and_append(
      expected_list, actual_list, host_allocator, &builder);
  fwrite(iree_string_builder_buffer(&builder), 1,
         iree_string_builder_size(&builder), file);
  iree_string_builder_deinitialize(&builder);
  return all_match;
}
