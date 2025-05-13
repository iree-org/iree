// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tools/testing/e2e/test_utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/cpu.h"
#include "iree/base/internal/flags.h"
#include "iree/base/internal/math.h"
#include "iree/base/internal/path.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/device_util.h"
#include "iree/vm/api.h"

IREE_FLAG(bool, require_exact_results, true,
          "Requires floating point result elements to match exactly.");

bool iree_test_utils_require_exact_results(void) {
  return FLAG_require_exact_results;
}

IREE_FLAG(
    float, acceptable_fp_delta, 1e-5f,
    "Maximum absolute difference allowed with inexact floating point results.");

float iree_test_utils_acceptable_fb_delta(void) {
  return FLAG_acceptable_fp_delta;
}

IREE_FLAG(
    int32_t, max_elements_to_check, 10000,
    "Maximum number of tensor elements to check for the given test. For larger "
    "buffers, only every n-th element will be checked for some n chosed to "
    "stay just under that threshold and to avoid being a divisor of the inner "
    "dimension size to avoid special patterns. As the check uses a slow "
    "reference implementation, this is a trade-off between test latency and "
    "coverage. The value 0 means check all elements.");

int32_t iree_test_utils_max_elements_to_check(void) {
  return FLAG_max_elements_to_check;
}

const char* iree_test_utils_emoji(bool good) { return good ? "🦄" : "🎃"; }

int iree_test_utils_calculate_check_every(iree_hal_dim_t tot_elements,
                                          iree_hal_dim_t no_div_of) {
  int check_every = 1;
  if (iree_test_utils_max_elements_to_check()) {
    check_every =
        ((tot_elements) + iree_test_utils_max_elements_to_check() - 1) /
        iree_test_utils_max_elements_to_check();
    if (check_every < 1) check_every = 1;
    if (check_every > 1)
      while ((no_div_of % check_every) == 0) ++check_every;
  }
  return check_every;
}

iree_test_utils_e2e_value_t iree_test_utils_value_make_none() {
  iree_test_utils_e2e_value_t result;
  result.type = IREE_TEST_UTILS_VALUE_TYPE_NONE;
  return result;
}

iree_test_utils_e2e_value_t iree_test_utils_value_make_i8(int8_t value) {
  iree_test_utils_e2e_value_t result;
  result.type = IREE_TEST_UTILS_VALUE_TYPE_I8;
  result.i8 = value;
  return result;
}

iree_test_utils_e2e_value_t iree_test_utils_value_make_i16(int16_t value) {
  iree_test_utils_e2e_value_t result;
  result.type = IREE_TEST_UTILS_VALUE_TYPE_I16;
  result.i16 = value;
  return result;
}

iree_test_utils_e2e_value_t iree_test_utils_value_make_i32(int32_t value) {
  iree_test_utils_e2e_value_t result;
  result.type = IREE_TEST_UTILS_VALUE_TYPE_I32;
  result.i32 = value;
  return result;
}

iree_test_utils_e2e_value_t iree_test_utils_value_make_f8E5M2(uint8_t value) {
  iree_test_utils_e2e_value_t result;
  result.type = IREE_TEST_UTILS_VALUE_TYPE_F8E5M2;
  result.f8_u8 = value;
  return result;
}

iree_test_utils_e2e_value_t iree_test_utils_value_make_f8E4M3FN(uint8_t value) {
  iree_test_utils_e2e_value_t result;
  result.type = IREE_TEST_UTILS_VALUE_TYPE_F8E4M3FN;
  result.f8_u8 = value;
  return result;
}

iree_test_utils_e2e_value_t iree_test_utils_value_make_f8E5M2FNUZ(
    uint16_t value) {
  iree_test_utils_e2e_value_t result;
  result.type = IREE_TEST_UTILS_VALUE_TYPE_F8E5M2FNUZ;
  result.f8_u8 = value;
  return result;
}

iree_test_utils_e2e_value_t iree_test_utils_value_make_f8E4M3FNUZ(
    uint16_t value) {
  iree_test_utils_e2e_value_t result;
  result.type = IREE_TEST_UTILS_VALUE_TYPE_F8E4M3FNUZ;
  result.f8_u8 = value;
  return result;
}

iree_test_utils_e2e_value_t iree_test_utils_value_make_f8E8M0FNU(
    uint8_t value) {
  iree_test_utils_e2e_value_t result;
  result.type = IREE_TEST_UTILS_VALUE_TYPE_F8E8M0FNU;
  result.f8_u8 = value;
  return result;
}

iree_test_utils_e2e_value_t iree_test_utils_value_make_f16(uint16_t value) {
  iree_test_utils_e2e_value_t result;
  result.type = IREE_TEST_UTILS_VALUE_TYPE_F16;
  result.f16_u16 = value;
  return result;
}

iree_test_utils_e2e_value_t iree_test_utils_value_make_bf16(uint16_t value) {
  iree_test_utils_e2e_value_t result;
  result.type = IREE_TEST_UTILS_VALUE_TYPE_BF16;
  result.bf16_u16 = value;
  return result;
}

iree_test_utils_e2e_value_t iree_test_utils_value_make_f32(float value) {
  iree_test_utils_e2e_value_t result;
  result.type = IREE_TEST_UTILS_VALUE_TYPE_F32;
  result.f32 = value;
  return result;
}

iree_test_utils_e2e_value_t iree_test_utils_value_make_f64(float value) {
  iree_test_utils_e2e_value_t result;
  result.type = IREE_TEST_UTILS_VALUE_TYPE_F64;
  result.f64 = value;
  return result;
}

iree_test_utils_e2e_value_t iree_test_utils_read_buffer_element(
    iree_hal_dim_t index, iree_hal_element_type_t result_type,
    const void* data) {
  if (iree_hal_element_type_is_integer(result_type, 8)) {
    return iree_test_utils_value_make_i8(((int8_t*)data)[index]);
  } else if (iree_hal_element_type_is_integer(result_type, 16)) {
    return iree_test_utils_value_make_i16(((int16_t*)data)[index]);
  } else if (iree_hal_element_type_is_integer(result_type, 32)) {
    return iree_test_utils_value_make_i32(((int32_t*)data)[index]);
  } else if (result_type == IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2) {
    return iree_test_utils_value_make_f8E5M2(((uint8_t*)data)[index]);
  } else if (result_type == IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FN) {
    return iree_test_utils_value_make_f8E4M3FN(((uint8_t*)data)[index]);
  } else if (result_type == IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2_FNUZ) {
    return iree_test_utils_value_make_f8E5M2FNUZ(((uint8_t*)data)[index]);
  } else if (result_type == IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FNUZ) {
    return iree_test_utils_value_make_f8E4M3FNUZ(((uint8_t*)data)[index]);
  } else if (result_type == IREE_HAL_ELEMENT_TYPE_FLOAT_8_E8M0_FNU) {
    return iree_test_utils_value_make_f8E8M0FNU(((uint8_t*)data)[index]);
  } else if (result_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16) {
    return iree_test_utils_value_make_f16(((uint16_t*)data)[index]);
  } else if (result_type == IREE_HAL_ELEMENT_TYPE_BFLOAT_16) {
    return iree_test_utils_value_make_bf16(((uint16_t*)data)[index]);
  } else if (result_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
    return iree_test_utils_value_make_f32(((float*)data)[index]);
  } else if (result_type == IREE_HAL_ELEMENT_TYPE_FLOAT_64) {
    return iree_test_utils_value_make_f64(((double*)data)[index]);
  }
  iree_status_abort(iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                     "unhandled matmul result type"));
  return iree_test_utils_value_make_none();
}

// Important: print all floating point values to FULL precision.
// The audience is debugging low-level numerical bugs.
// Since the values used in most tests are small and integral, these will
// normally print just as concisely, while the extra precision requested here
// will only kick in when it's needed, when there is a numerical bug.
int iree_test_utils_snprintf_value(char* buf, size_t bufsize,
                                   iree_test_utils_e2e_value_t value) {
  switch (value.type) {
    case IREE_TEST_UTILS_VALUE_TYPE_I8:
      return snprintf(buf, bufsize, "%" PRIi8, value.i8);
    case IREE_TEST_UTILS_VALUE_TYPE_I16:
      return snprintf(buf, bufsize, "%" PRIi16, value.i16);
    case IREE_TEST_UTILS_VALUE_TYPE_I32:
      return snprintf(buf, bufsize, "%" PRIi32, value.i32);
    case IREE_TEST_UTILS_VALUE_TYPE_I64:
      return snprintf(buf, bufsize, "%" PRIi64, value.i64);
    case IREE_TEST_UTILS_VALUE_TYPE_F8E5M2:
      return snprintf(buf, bufsize, "%.3g",
                      iree_math_f8e5m2_to_f32(value.f8_u8));
    case IREE_TEST_UTILS_VALUE_TYPE_F8E4M3FN:
      return snprintf(buf, bufsize, "%.3g",
                      iree_math_f8e4m3fn_to_f32(value.f8_u8));
    case IREE_TEST_UTILS_VALUE_TYPE_F8E5M2FNUZ:
      return snprintf(buf, bufsize, "%.3g",
                      iree_math_f8e5m2fnuz_to_f32(value.f8_u8));
    case IREE_TEST_UTILS_VALUE_TYPE_F8E4M3FNUZ:
      return snprintf(buf, bufsize, "%.3g",
                      iree_math_f8e4m3fnuz_to_f32(value.f8_u8));
    case IREE_TEST_UTILS_VALUE_TYPE_F16:
      return snprintf(buf, bufsize, "%.5g",
                      iree_math_f16_to_f32(value.f16_u16));
    case IREE_TEST_UTILS_VALUE_TYPE_BF16:
      return snprintf(buf, bufsize, "%.5g",
                      iree_math_bf16_to_f32(value.bf16_u16));
    case IREE_TEST_UTILS_VALUE_TYPE_F32:
      return snprintf(buf, bufsize, "%.8g", value.f32);
    case IREE_TEST_UTILS_VALUE_TYPE_F64:
      return snprintf(buf, bufsize, "%.16g", value.f64);
    default:
      iree_status_abort(iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                         "unhandled value type"));
      return 0;
  }
}

bool iree_test_utils_result_elements_agree(iree_test_utils_e2e_value_t expected,
                                           iree_test_utils_e2e_value_t actual) {
  float acceptable_fp_delta = iree_test_utils_acceptable_fb_delta();
  if (expected.type != actual.type) {
    iree_status_abort(
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "mismatched types"));
    return false;
  }

  if (acceptable_fp_delta < 0.0f) {
    iree_status_abort(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "negative tolerance (acceptable_fp_delta=%.8g)", acceptable_fp_delta));
    return false;
  }

  switch (expected.type) {
    case IREE_TEST_UTILS_VALUE_TYPE_I32:
      return actual.i32 == expected.i32;
    // Since we fill buffers with small integers for floating point GEMMs
    // functional testing, we can test for bit-exactness on the actual and
    // expected values. Inexact results are only permitted when the
    // `require_exact_results` flag is set to `false`.
    case IREE_TEST_UTILS_VALUE_TYPE_F16:
      if (actual.f16_u16 == expected.f16_u16) return true;
      if (iree_test_utils_max_elements_to_check()) return false;
      return fabsf(iree_math_f16_to_f32(actual.f16_u16) -
                   iree_math_f16_to_f32(expected.f16_u16)) <
             acceptable_fp_delta;
    case IREE_TEST_UTILS_VALUE_TYPE_BF16: {
      if (actual.bf16_u16 == expected.bf16_u16) return true;
      // This is the rare case where the accumulator itself (not just LHS/RHS)
      // is bf16. This doesn't really happen in practice and is mostly just in
      // some CPU tests for completeness. Accumulators grow outside of the
      // narrow range of bf16 exact representation of integers, forcing fuzzy
      // compares.
      float actual_f32 = iree_math_bf16_to_f32(actual.bf16_u16);
      float expected_f32 = iree_math_bf16_to_f32(expected.bf16_u16);
      if (fabsf(actual_f32) > 127.0f || fabsf(expected_f32) > 127.0f) {
        if (fabsf(actual_f32 - expected_f32) < 10.0f) {
          return true;
        }
      }
      if (iree_test_utils_require_exact_results()) return false;
      return fabsf(actual_f32 - expected_f32) < acceptable_fp_delta;
    }
    case IREE_TEST_UTILS_VALUE_TYPE_F32:
      if (actual.f32 == expected.f32) return true;
      if (iree_test_utils_require_exact_results()) return false;
      return fabsf(actual.f32 - expected.f32) < acceptable_fp_delta;
    case IREE_TEST_UTILS_VALUE_TYPE_F64:
      if (actual.f64 == expected.f64) return true;
      if (iree_test_utils_require_exact_results()) return false;
      return fabs(actual.f64 - expected.f64) < acceptable_fp_delta;
    default:
      iree_status_abort(iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                         "unhandled value type"));
      return false;
  }
}

//===----------------------------------------------------------------------===//
// RNG utilities
//===----------------------------------------------------------------------===//

void iree_test_utils_write_element(iree_hal_element_type_t element_type,
                                   int32_t value, void* dst) {
#define WRITE_ELEMENT_CASE(ETYPE, CTYPE) \
  case IREE_HAL_ELEMENT_TYPE_##ETYPE:    \
    *(CTYPE*)dst = (CTYPE)value;         \
    break;

  switch (element_type) {
    WRITE_ELEMENT_CASE(INT_8, int8_t)
    WRITE_ELEMENT_CASE(INT_16, int16_t)
    WRITE_ELEMENT_CASE(INT_32, int32_t)
    WRITE_ELEMENT_CASE(INT_64, int64_t)
    WRITE_ELEMENT_CASE(SINT_8, int8_t)
    WRITE_ELEMENT_CASE(SINT_16, int16_t)
    WRITE_ELEMENT_CASE(SINT_32, int32_t)
    WRITE_ELEMENT_CASE(SINT_64, int64_t)
    WRITE_ELEMENT_CASE(UINT_8, uint8_t)
    WRITE_ELEMENT_CASE(UINT_16, uint16_t)
    WRITE_ELEMENT_CASE(UINT_32, uint32_t)
    WRITE_ELEMENT_CASE(UINT_64, uint64_t)
      // clang-format off
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      *(uint16_t*)dst = iree_math_f32_to_f16((float)value);
      break;
    case IREE_HAL_ELEMENT_TYPE_BFLOAT_16:
      *(uint16_t*)dst = iree_math_f32_to_bf16((float)value);
      break;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2:
      *(uint8_t*)dst = iree_math_f32_to_f8e5m2((float)value);
      break;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FN:
      *(uint8_t*)dst = iree_math_f32_to_f8e4m3fn((float)value);
      break;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2_FNUZ:
      *(uint8_t*)dst = iree_math_f32_to_f8e5m2fnuz((float)value);
      break;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FNUZ:
      *(uint8_t*)dst = iree_math_f32_to_f8e4m3fnuz((float)value);
      break;
    WRITE_ELEMENT_CASE(FLOAT_32, float)
    WRITE_ELEMENT_CASE(FLOAT_64, double)
    // clang-format on
    default:
      IREE_ASSERT(false, "unhandled element type");
      break;
  }

#undef WRITE_ELEMENT_CASE
}

uint32_t iree_test_utils_pseudorandom_uint32(uint32_t* state) {
  *state = (*state * IREE_PRNG_MULTIPLIER) % IREE_PRNG_MODULUS;
  return *state;
}

uint32_t iree_test_utils_pseudorandom_range(uint32_t* state, uint32_t range) {
  return iree_test_utils_pseudorandom_uint32(state) % range;
}

void iree_test_utils_get_min_max_for_element_type(
    iree_hal_element_type_t element_type, int32_t* min, int32_t* max) {
  switch (element_type) {
    case IREE_HAL_ELEMENT_TYPE_INT_8:
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
      *min = -2;
      *max = +2;
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
      *min = 0;
      *max = +2;
      break;
    case IREE_HAL_ELEMENT_TYPE_INT_16:
    case IREE_HAL_ELEMENT_TYPE_SINT_16:
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      *min = -4;
      *max = +4;
      break;
    case IREE_HAL_ELEMENT_TYPE_BFLOAT_16:
    case IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2:
    case IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FN:
    case IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2_FNUZ:
    case IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FNUZ:
      *min = -2;
      *max = +2;
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_16:
      *min = 0;
      *max = +4;
      break;
    case IREE_HAL_ELEMENT_TYPE_INT_32:
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      *min = -8;
      *max = +8;
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
      *min = 0;
      *max = +8;
      break;
    case IREE_HAL_ELEMENT_TYPE_INT_64:
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      *min = -16;
      *max = +16;
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_64:
      *min = 0;
      *max = +16;
      break;
    default:
      IREE_ASSERT(false, "unhandled element type");
      break;
  }
}

//===----------------------------------------------------------------------===//
// Test runner
//===----------------------------------------------------------------------===//

iree_status_t iree_test_utils_check_test_function(iree_vm_function_t function,
                                                  bool* out_is_valid) {
  *out_is_valid = true;

  iree_string_view_t function_name = iree_vm_function_name(&function);
  if (iree_string_view_starts_with(function_name,
                                   iree_make_cstring_view("__"))) {
    // Internal compiler/runtime support function.
    *out_is_valid = false;
  }

  iree_vm_function_signature_t function_signature =
      iree_vm_function_signature(&function);
  iree_host_size_t argument_count = 0;
  iree_host_size_t result_count = 0;
  IREE_RETURN_IF_ERROR(iree_vm_function_call_count_arguments_and_results(
      &function_signature, &argument_count, &result_count));
  if (argument_count || result_count) {
    // Takes args or has results we don't expect.
    *out_is_valid = false;
  }

  return iree_ok_status();
}

iree_status_t iree_test_utils_run_test_function(
    iree_vm_context_t* context, iree_vm_function_t function,
    iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_string_view_t function_name = iree_vm_function_name(&function);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, function_name.data, function_name.size);
  fprintf(stderr, "--- TEST[%.*s] ---\n", (int)function_name.size,
          function_name.data);
  iree_string_view_t function_desc =
      iree_vm_function_lookup_attr_by_name(&function, IREE_SV("description"));
  if (!iree_string_view_is_empty(function_desc)) {
    fprintf(stderr, "%.*s\n", (int)function_desc.size, function_desc.data);
  }
  iree_status_t status = iree_vm_invoke(
      context, function, IREE_VM_INVOCATION_FLAG_NONE, /*policy=*/NULL,
      /*inputs=*/NULL, /*outputs=*/NULL, host_allocator);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_test_utils_run_all_test_functions(
    iree_vm_context_t* context, iree_vm_module_t* test_module,
    iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Walk all functions and find the ones we can run (no args, non-internal).
  const iree_vm_module_signature_t module_signature =
      iree_vm_module_signature(test_module);
  for (iree_host_size_t i = 0; i < module_signature.export_function_count;
       ++i) {
    // Get the function and filter to just the public user exports.
    iree_vm_function_t function;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_vm_module_lookup_function_by_ordinal(
                test_module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &function));
    bool is_valid = false;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_test_utils_check_test_function(function, &is_valid));
    if (is_valid) {
      // Try to run the function and fail on mismatch.
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0,
          iree_test_utils_run_test_function(context, function, host_allocator));
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_test_utils_check_module_requirements(
    iree_vm_module_t* module) {
  iree_string_view_t target_features =
      iree_vm_module_lookup_attr_by_name(module, IREE_SV("target_features"));
  while (!iree_string_view_is_empty(target_features)) {
    iree_string_view_t required_feature;
    iree_string_view_split(target_features, ',', &required_feature,
                           &target_features);
    if (iree_string_view_is_empty(required_feature)) continue;
    int64_t feature_is_supported = 0;
    IREE_RETURN_IF_ERROR(
        iree_cpu_lookup_data_by_key(required_feature, &feature_is_supported));
    if (!feature_is_supported) {
      return iree_make_status(
          // The error status matters. We distinguish "feature not supported"
          // which is a normal thing to happen from actual errors.
          IREE_STATUS_NOT_FOUND,
          "target device does not have the required feature '%.*s'",
          (int)required_feature.size, required_feature.data);
    }
  }
  return iree_ok_status();
}

iree_status_t iree_test_utils_load_and_run_e2e_tests(
    iree_allocator_t host_allocator,
    iree_status_t (*test_module_create)(iree_vm_instance_t*, iree_allocator_t,
                                        iree_vm_module_t**)) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_cpu_initialize(host_allocator);

  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tooling_create_instance(host_allocator, &instance));

  iree_tooling_module_list_t module_list;
  iree_tooling_module_list_initialize(&module_list);

  // Create the test module providing helper functions used by test programs.
  iree_vm_module_t* custom_test_module = NULL;
  iree_status_t status =
      test_module_create(instance, host_allocator, &custom_test_module);
  if (iree_status_is_ok(status)) {
    status =
        iree_tooling_module_list_push_back(&module_list, custom_test_module);
  }
  iree_vm_module_release(custom_test_module);

  // Load all modules specified by --module= flags.
  if (iree_status_is_ok(status)) {
    status = iree_tooling_load_modules_from_flags(instance, host_allocator,
                                                  &module_list);
  }
  iree_vm_module_t* test_module = iree_tooling_module_list_back(&module_list);

  // Create the context with our support module and all --module= flags.
  iree_vm_context_t* context = NULL;
  iree_hal_device_t* device = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tooling_create_context_from_flags(
        instance, module_list.count, module_list.values,
        /*default_device_uri=*/iree_string_view_empty(), host_allocator,
        &context, &device, /*out_device_allocator=*/NULL);
  }

  // Ensure the test module is possible to run.
  if (iree_status_is_ok(status)) {
    status = iree_test_utils_check_module_requirements(test_module);
  }
  iree_tooling_module_list_reset(&module_list);

  // Begin profiling (if enabled).
  if (iree_status_is_ok(status)) {
    status = iree_hal_begin_profiling_from_flags(device);
  }

  // Run all of the tests in the test module.
  if (iree_status_is_ok(status)) {
    status = iree_test_utils_run_all_test_functions(context, test_module,
                                                    host_allocator);
  }

  // End profiling (if enabled).
  if (iree_status_is_ok(status)) {
    status = iree_hal_end_profiling_from_flags(device);
  }

  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
