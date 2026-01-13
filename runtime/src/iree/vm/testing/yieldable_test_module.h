// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A native module for testing vm.call.yieldable to imports.
// Exports a single function that yields N times before returning.
//
// This module provides a controlled way to test async/yield behavior:
//   yield_n(arg: i32, yield_count: i32) -> i32
//   Returns arg + yield_count after yielding yield_count times.
//
// NOTE: This module stores coroutine state (yield_count, accumulator) in module
// state, which means it is not reentrant. Concurrent or interleaved calls on
// the same module instance through the same VM context are not supported. This
// is consistent with IREE's threading model where modules are thread-compatible
// (safe for sequential use) but not thread-safe (no concurrent access).

#ifndef IREE_VM_TESTING_YIELDABLE_TEST_MODULE_H_
#define IREE_VM_TESTING_YIELDABLE_TEST_MODULE_H_

#include "iree/base/api.h"
#include "iree/vm/native_module.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// yieldable_test_module
//===----------------------------------------------------------------------===//

typedef struct yieldable_test_module_state_t {
  iree_allocator_t allocator;
  // Yield counter: decremented on each resume until 0.
  int32_t yield_count;
  // Accumulator for the result.
  int32_t accumulator;
} yieldable_test_module_state_t;

// Shim for yield_variadic_sum: (i32..., i32 yield_count) -> i32
// Sums all variadic i32 args, yields yield_count times, returns sum +
// yield_count.
static iree_status_t yieldable_test_module_yield_variadic_sum_shim(
    iree_vm_stack_t* stack, iree_vm_native_function_flags_t flags,
    iree_byte_span_t args_storage, iree_byte_span_t rets_storage,
    void* target_fn, void* module, void* module_state_ptr) {
  yieldable_test_module_state_t* state =
      (yieldable_test_module_state_t*)module_state_ptr;

  if (flags == IREE_VM_NATIVE_FUNCTION_CALL_BEGIN) {
    // Parse variadic arguments.
    // Layout: [segment_count: i32] [values: i32 * segment_count] [yield_count:
    // i32]

    // Validate minimum size for segment_count field.
    if (args_storage.data_length < sizeof(int32_t)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "argument buffer too small for segment_count; have %" PRIhsz
          " bytes, need at least %" PRIhsz,
          args_storage.data_length, sizeof(int32_t));
    }

    const uint8_t* p = args_storage.data;
    int32_t segment_count;
    memcpy(&segment_count, p, sizeof(int32_t));
    p += sizeof(int32_t);

    // Validate segment_count is non-negative and buffer has sufficient space.
    // Required size: segment_count (1) + values (segment_count) + yield_count
    // (1).
    if (segment_count < 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "segment_count must be non-negative; got %d",
                              segment_count);
    }
    iree_host_size_t required_size =
        (iree_host_size_t)(segment_count + 2) * sizeof(int32_t);
    if (args_storage.data_length < required_size) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "argument buffer too small for %d variadic args; have %" PRIhsz
          " bytes, need %" PRIhsz,
          segment_count, args_storage.data_length, required_size);
    }

    // Sum all variadic values.
    int32_t sum = 0;
    for (int32_t i = 0; i < segment_count; ++i) {
      int32_t value;
      memcpy(&value, p, sizeof(int32_t));
      sum += value;
      p += sizeof(int32_t);
    }

    int32_t yield_count;
    memcpy(&yield_count, p, sizeof(int32_t));

    // Initialize state.
    state->yield_count = yield_count;
    state->accumulator = sum;

    if (state->yield_count > 0) {
      state->accumulator += 1;
      state->yield_count -= 1;
      return iree_status_from_code(IREE_STATUS_DEFERRED);
    }
    // yield_count == 0: return immediately.
    typedef struct {
      int32_t ret;
    } results_t;
    results_t* results = (results_t*)rets_storage.data;
    results->ret = state->accumulator;
    return iree_ok_status();
  }

  // RESUME path.
  if (state->yield_count > 0) {
    state->accumulator += 1;
    state->yield_count -= 1;
    return iree_status_from_code(IREE_STATUS_DEFERRED);
  }

  // Done yielding, return result.
  typedef struct {
    int32_t ret;
  } results_t;
  results_t* results = (results_t*)rets_storage.data;
  results->ret = state->accumulator;
  return iree_ok_status();
}

// Shim for yield_n: (i32 arg, i32 yield_count) -> i32
// Yields yield_count times, returns arg + yield_count.
static iree_status_t yieldable_test_module_yield_n_shim(
    iree_vm_stack_t* stack, iree_vm_native_function_flags_t flags,
    iree_byte_span_t args_storage, iree_byte_span_t rets_storage,
    void* target_fn, void* module, void* module_state_ptr) {
  yieldable_test_module_state_t* state =
      (yieldable_test_module_state_t*)module_state_ptr;

  if (flags == IREE_VM_NATIVE_FUNCTION_CALL_BEGIN) {
    // Parse arguments.
    typedef struct {
      int32_t arg;
      int32_t yield_count;
    } args_t;

    // Validate buffer size.
    if (args_storage.data_length < sizeof(args_t)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "argument buffer too small; have %" PRIhsz
                              " bytes, need %" PRIhsz,
                              args_storage.data_length, sizeof(args_t));
    }

    // Use memcpy for alignment-safe access.
    args_t args;
    memcpy(&args, args_storage.data, sizeof(args_t));

    // Initialize state for coroutine.
    state->yield_count = args.yield_count;
    state->accumulator = args.arg;

    if (state->yield_count > 0) {
      state->accumulator += 1;
      state->yield_count -= 1;
      return iree_status_from_code(IREE_STATUS_DEFERRED);
    }
    // yield_count == 0: return immediately.
    typedef struct {
      int32_t ret;
    } results_t;
    results_t* results = (results_t*)rets_storage.data;
    results->ret = state->accumulator;
    return iree_ok_status();
  }

  // RESUME path.
  if (state->yield_count > 0) {
    state->accumulator += 1;
    state->yield_count -= 1;
    return iree_status_from_code(IREE_STATUS_DEFERRED);
  }

  // Done yielding, return result.
  typedef struct {
    int32_t ret;
  } results_t;
  results_t* results = (results_t*)rets_storage.data;
  results->ret = state->accumulator;
  return iree_ok_status();
}

static iree_status_t IREE_API_PTR
yieldable_test_module_alloc_state(void* self, iree_allocator_t allocator,
                                  iree_vm_module_state_t** out_module_state) {
  yieldable_test_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->allocator = allocator;
  *out_module_state = (iree_vm_module_state_t*)state;
  return iree_ok_status();
}

static void IREE_API_PTR yieldable_test_module_free_state(
    void* self, iree_vm_module_state_t* module_state) {
  yieldable_test_module_state_t* state =
      (yieldable_test_module_state_t*)module_state;
  iree_allocator_free(state->allocator, state);
}

static const iree_vm_native_export_descriptor_t
    yieldable_test_module_exports_[] = {
        // yield_n(arg: i32, yield_count: i32) -> i32
        // Signature: "0ii_i" = version 0, (i32, i32) -> i32
        {IREE_SV("yield_n"), IREE_SV("0ii_i"), 0, NULL},
        // yield_variadic_sum(args: i32..., yield_count: i32) -> i32
        // Signature: "0CiDi_i" = version 0, variadic i32, i32, returns i32
        {IREE_SV("yield_variadic_sum"), IREE_SV("0CiDi_i"), 0, NULL},
};
static const iree_vm_native_function_ptr_t yieldable_test_module_funcs_[] = {
    {(iree_vm_native_function_shim_t)yieldable_test_module_yield_n_shim, NULL},
    {(iree_vm_native_function_shim_t)
         yieldable_test_module_yield_variadic_sum_shim,
     NULL},
};
static_assert(IREE_ARRAYSIZE(yieldable_test_module_funcs_) ==
                  IREE_ARRAYSIZE(yieldable_test_module_exports_),
              "function pointer table must be 1:1 with exports");

static const iree_vm_native_module_descriptor_t
    yieldable_test_module_descriptor_ = {
        /*name=*/IREE_SV("yieldable_test"),
        /*version=*/0,
        /*attr_count=*/0,
        /*attrs=*/NULL,
        /*dependency_count=*/0,
        /*dependencies=*/NULL,
        /*import_count=*/0,
        /*imports=*/NULL,
        /*export_count=*/IREE_ARRAYSIZE(yieldable_test_module_exports_),
        /*exports=*/yieldable_test_module_exports_,
        /*function_count=*/IREE_ARRAYSIZE(yieldable_test_module_funcs_),
        /*functions=*/yieldable_test_module_funcs_,
};

static iree_status_t yieldable_test_module_create(
    iree_vm_instance_t* instance, iree_allocator_t allocator,
    iree_vm_module_t** out_module) {
  iree_vm_module_t interface;
  IREE_RETURN_IF_ERROR(iree_vm_module_initialize(&interface, NULL));
  interface.alloc_state = yieldable_test_module_alloc_state;
  interface.free_state = yieldable_test_module_free_state;
  return iree_vm_native_module_create(&interface,
                                      &yieldable_test_module_descriptor_,
                                      instance, allocator, out_module);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_VM_TESTING_YIELDABLE_TEST_MODULE_H_
