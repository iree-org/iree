// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLS_TESTING_E2E_TEST_UTILS_H_
#define IREE_TOOLS_TESTING_E2E_TEST_UTILS_H_
#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

bool iree_test_utils_require_exact_results(void);

float iree_test_utils_acceptable_fb_delta(void);

int32_t iree_test_utils_max_elements_to_check(void);

const char* iree_test_utils_emoji(bool good);

int iree_test_utils_calculate_check_every(iree_hal_dim_t tot_elements,
                                          iree_hal_dim_t no_div_of);

// Defines the type of a primitive value.
typedef enum iree_test_utils_value_type_e {
  // Not a value type.
  IREE_TEST_UTILS_VALUE_TYPE_NONE = 0,
  // int8_t.
  IREE_TEST_UTILS_VALUE_TYPE_I8 = 1,
  // int16_t.
  IREE_TEST_UTILS_VALUE_TYPE_I16 = 2,
  // int32_t.
  IREE_TEST_UTILS_VALUE_TYPE_I32 = 3,
  // int64_t.
  IREE_TEST_UTILS_VALUE_TYPE_I64 = 4,
  // halft_t.
  IREE_TEST_UTILS_VALUE_TYPE_F16 = 5,
  // float.
  IREE_TEST_UTILS_VALUE_TYPE_F32 = 6,
  // double.
  IREE_TEST_UTILS_VALUE_TYPE_F64 = 7,
  // bfloat16
  IREE_TEST_UTILS_VALUE_TYPE_BF16 = 8,
  // float8
  IREE_TEST_UTILS_VALUE_TYPE_F8 = 9,
} iree_test_utils_value_type_t;

// Maximum size, in bytes, of any value type we can represent.
#define IREE_E2E_TEST_VALUE_STORAGE_SIZE 8

// A variant value type.
typedef struct iree_test_utils_value_t {
  iree_test_utils_value_type_t type;
  union {
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    float f32;
    uint8_t f8_u8;
    uint16_t f16_u16;
    uint16_t bf16_u16;
    double f64;
    uint8_t value_storage[IREE_E2E_TEST_VALUE_STORAGE_SIZE];  // max size of all
                                                              // value types
  };
} iree_test_utils_e2e_value_t;

// Enum controlling how many decimals to print floats with.
typedef enum iree_test_utils_precision_e {
  PRECISION_LOW,
  PRECISION_HIGH,
} precision_t;

// Reads an element from a buffer given index.
iree_test_utils_e2e_value_t iree_test_utils_read_buffer_element(
    iree_hal_dim_t index, iree_hal_element_type_t result_type,
    const void* data);

// Prints a iree_e2e_test_value_t to a string buffer. Returns the number of
// characters written. Like snprintf.
int iree_test_utils_snprintf_value(char* buf, size_t bufsize,
                                   iree_test_utils_e2e_value_t value,
                                   precision_t precision);

// Returns true if |expected| and |actual| agree to tolerable accuracy.
bool iree_test_utils_result_elements_agree(iree_test_utils_e2e_value_t expected,
                                           iree_test_utils_e2e_value_t actual);

//===----------------------------------------------------------------------===//
// RNG utilities
//===----------------------------------------------------------------------===//

// Parameter for locally defined lcg similar to std::minstd_rand.
#define IREE_PRNG_MULTIPLIER 48271
#define IREE_PRNG_MODULUS 2147483647

// Simple deterministic pseudorandom generator.
// This function is same as C++'s std::minstd_rand.
uint32_t iree_test_utils_pseudorandom_uint32(uint32_t* state);

// Returns a random uint32_t in the range [0, range).
uint32_t iree_test_utils_pseudorandom_range(uint32_t* state, uint32_t range);

// Writes an element of the given |element_type| with the given integral |value|
// to |dst|.
void iree_test_utils_write_element(iree_hal_element_type_t element_type,
                                   int32_t value, void* dst);

// Get minimum and maximum for integer-valued uniform distribution.
void iree_test_utils_get_min_max_for_element_type(
    iree_hal_element_type_t element_type, int32_t* min, int32_t* max);

// Returns true if the |function| is a supported callable test function.
// We only support functions that are publicly exported, not an internal
// compiler/runtime function (__ prefixed), and take/return no args/results.
iree_status_t iree_test_utils_check_test_function(iree_vm_function_t function,
                                                  bool* out_is_valid);

// Synchronous runs a test |function|.
// If the test fails then the failure status is returned to the caller.
iree_status_t iree_test_utils_run_test_function(
    iree_vm_context_t* context, iree_vm_function_t function,
    iree_allocator_t host_allocator);

// Runs all test functions in |test_module|.
iree_status_t iree_test_utils_run_all_test_functions(
    iree_vm_context_t* context, iree_vm_module_t* test_module,
    iree_allocator_t host_allocator);

// Returns OK if there are declared requirements on |module| and they are all
// met and otherwise UNAVAILABLE indicating that the module should not be run.
iree_status_t iree_test_utils_check_module_requirements(
    iree_vm_module_t* module);

iree_status_t iree_test_utils_load_and_run_e2e_tests(
    iree_allocator_t host_allocator,
    iree_status_t (*test_module_create)(iree_vm_instance_t*, iree_allocator_t,
                                        iree_vm_module_t**));
#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLS_TESTING_E2E_TEST_UTILS_H_
