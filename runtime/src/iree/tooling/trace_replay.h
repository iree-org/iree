// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_TRACE_REPLAY_H_
#define IREE_TOOLING_TRACE_REPLAY_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/tooling/yaml_util.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

enum iree_trace_replay_shutdown_flag_bits_e {
  IREE_TRACE_REPLAY_SHUTDOWN_QUIET = 0u,
  IREE_TRACE_REPLAY_SHUTDOWN_PRINT_STATISTICS = 1 << 0u,
};
typedef uint32_t iree_trace_replay_shutdown_flags_t;

typedef struct iree_trace_replay_t {
  iree_allocator_t host_allocator;
  iree_string_view_t root_path;

  iree_vm_instance_t* instance;
  iree_vm_context_flags_t context_flags;

  iree_hal_driver_registry_t* driver_registry;
  iree_host_size_t device_uri_count;
  const iree_string_view_t* device_uris;

  iree_vm_context_t* context;
  iree_hal_device_t* device;
} iree_trace_replay_t;

// Initializes a trace replay context.
// Relative paths will be joined with |root_path| to form a fully-qualified
// path (may be cwd, may be file source, etc).
iree_status_t iree_trace_replay_initialize(
    iree_string_view_t root_path, iree_vm_instance_t* instance,
    iree_vm_context_flags_t context_flags,
    iree_hal_driver_registry_t* driver_registry,
    iree_allocator_t host_allocator, iree_trace_replay_t* out_replay);

// Deinitializes a trace replay context and releases all resources.
void iree_trace_replay_deinitialize(iree_trace_replay_t* replay,
                                    iree_trace_replay_shutdown_flags_t flags);

// TODO(#5724): remove this and instead provide a device set on initialize.
// Overrides the HAL driver used in the trace with the given |driver|.
// |device_uris| must remain valid for the lifetime of the replay instance.
void iree_trace_replay_set_hal_devices_override(
    iree_trace_replay_t* replay, iree_host_size_t device_uri_count,
    const iree_string_view_t* device_uris);

// Replays the given |event_node| against the replay context.
// Automatically switches between the default iree_trace_replay_event_* methods.
iree_status_t iree_trace_replay_event(iree_trace_replay_t* replay,
                                      yaml_document_t* document,
                                      yaml_node_t* event_node);

// Replays a `context_load` event against the replay context.
iree_status_t iree_trace_replay_event_context_load(iree_trace_replay_t* replay,
                                                   yaml_document_t* document,
                                                   yaml_node_t* event_node);

// Replays a `module_load` event against the replay context.
iree_status_t iree_trace_replay_event_module_load(iree_trace_replay_t* replay,
                                                  yaml_document_t* document,
                                                  yaml_node_t* event_node);

// Prepares to replay a `call` event.
// |out_function| will contain the function to invoke and |out_input_list| will
// contain the caller-owned inputs to the invocation.
iree_status_t iree_trace_replay_event_call_prepare(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* event_node, iree_vm_function_t* out_function,
    iree_vm_list_t** out_input_list);

// Replays a `call` event against the replay context.
// Optionally |out_output_list| can be populated with a caller-owned set of
// outputs from the call.
iree_status_t iree_trace_replay_event_call(iree_trace_replay_t* replay,
                                           yaml_document_t* document,
                                           yaml_node_t* event_node,
                                           iree_vm_list_t** out_output_list);

// Defines the type of a primitive value.
typedef enum iree_e2e_test_value_type_e {
  // Not a value type.
  IREE_E2E_TEST_VALUE_TYPE_NONE = 0,
  // int8_t.
  IREE_E2E_TEST_VALUE_TYPE_I8 = 1,
  // int16_t.
  IREE_E2E_TEST_VALUE_TYPE_I16 = 2,
  // int32_t.
  IREE_E2E_TEST_VALUE_TYPE_I32 = 3,
  // int64_t.
  IREE_E2E_TEST_VALUE_TYPE_I64 = 4,
  // halft_t.
  IREE_E2E_TEST_VALUE_TYPE_F16 = 5,
  // float.
  IREE_E2E_TEST_VALUE_TYPE_F32 = 6,
  // double.
  IREE_E2E_TEST_VALUE_TYPE_F64 = 7,
} iree_e2e_test_value_type_t;

// Maximum size, in bytes, of any value type we can represent.
#define IREE_E2E_TEST_VALUE_STORAGE_SIZE 8

// A variant value type.
typedef struct iree_e2e_test_value_t {
  iree_e2e_test_value_type_t type;
  union {
    uint8_t value_storage[IREE_E2E_TEST_VALUE_STORAGE_SIZE];  // max size of all
                                                              // value types
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    float f32;
    uint16_t f16_u16;
    double f64;
  };
} iree_e2e_test_value_t;

static inline iree_e2e_test_value_t iree_e2e_test_value_make_none() {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_NONE;
  return result;
}

static inline iree_e2e_test_value_t iree_e2e_test_value_make_i8(int8_t value) {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_I8;
  result.i8 = value;
  return result;
}

static inline iree_e2e_test_value_t iree_e2e_test_value_make_i16(
    int16_t value) {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_I16;
  result.i16 = value;
  return result;
}

static inline iree_e2e_test_value_t iree_e2e_test_value_make_i32(
    int32_t value) {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_I32;
  result.i32 = value;
  return result;
}

// TODO(#5542): check the value type before accessing the union.
static inline int32_t iree_e2e_test_value_get_i32(
    iree_e2e_test_value_t* value) {
  return value->i32;
}

static inline iree_e2e_test_value_t iree_e2e_test_value_make_i64(
    int64_t value) {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_I64;
  result.i64 = value;
  return result;
}

// TODO(#5542): check the value type before accessing the union.
static inline int64_t iree_e2e_test_value_get_i64(
    iree_e2e_test_value_t* value) {
  return value->i64;
}

static inline iree_e2e_test_value_t iree_e2e_test_value_make_f16(
    uint16_t value) {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_F16;
  result.f16_u16 = value;
  return result;
}

static inline iree_e2e_test_value_t iree_e2e_test_value_make_f32(float value) {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_F32;
  result.f32 = value;
  return result;
}

// TODO(#5542): check the value type before accessing the union.
static inline float iree_e2e_test_value_get_f32(iree_e2e_test_value_t* value) {
  return value->f32;
}

// TODO(#5542): check the value type before accessing the union.
static inline uint16_t iree_e2e_test_value_get_f16(
    iree_e2e_test_value_t* value) {
  return value->f16_u16;
}

static inline iree_e2e_test_value_t iree_e2e_test_value_make_f64(double value) {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_F64;
  result.f64 = value;
  return result;
}

// TODO(#5542): check the value type before accessing the union.
static inline double iree_e2e_test_value_get_f64(iree_e2e_test_value_t* value) {
  return value->f64;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_TRACE_REPLAY_H_
