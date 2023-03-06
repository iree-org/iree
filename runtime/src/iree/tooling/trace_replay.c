// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/trace_replay.h"

#include <ctype.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/internal/file_io.h"
#include "iree/base/internal/math.h"
#include "iree/base/internal/path.h"
#include "iree/base/tracing.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/bytecode/module.h"

//===----------------------------------------------------------------------===//
// iree_trace_replay_t
//===----------------------------------------------------------------------===//

iree_status_t iree_trace_replay_initialize(
    iree_string_view_t root_path, iree_vm_instance_t* instance,
    iree_vm_context_flags_t context_flags,
    iree_hal_driver_registry_t* driver_registry,
    iree_allocator_t host_allocator, iree_trace_replay_t* out_replay) {
  memset(out_replay, 0, sizeof(*out_replay));

  IREE_RETURN_IF_ERROR(iree_hal_module_register_all_types(instance));

  out_replay->host_allocator = host_allocator;
  out_replay->root_path = root_path;

  out_replay->instance = instance;
  iree_vm_instance_retain(out_replay->instance);
  out_replay->context_flags = context_flags;

  out_replay->driver_registry = driver_registry;

  return iree_ok_status();
}

void iree_trace_replay_deinitialize(iree_trace_replay_t* replay,
                                    iree_trace_replay_shutdown_flags_t flags) {
  iree_vm_context_release(replay->context);
  iree_vm_instance_release(replay->instance);

  if (iree_all_bits_set(flags, IREE_TRACE_REPLAY_SHUTDOWN_PRINT_STATISTICS)) {
    IREE_IGNORE_ERROR(iree_hal_allocator_statistics_fprint(
        stderr, iree_hal_device_allocator(replay->device)));
  }
  iree_hal_device_release(replay->device);

  memset(replay, 0, sizeof(*replay));
}

void iree_trace_replay_set_hal_devices_override(
    iree_trace_replay_t* replay, iree_host_size_t device_uri_count,
    const iree_string_view_t* device_uris) {
  replay->device_uri_count = device_uri_count;
  replay->device_uris = device_uris;
}

//===----------------------------------------------------------------------===//
// type: context_load
//===----------------------------------------------------------------------===//

iree_status_t iree_trace_replay_event_context_load(iree_trace_replay_t* replay,
                                                   yaml_document_t* document,
                                                   yaml_node_t* event_node) {
  // Cleanup previous state.
  iree_hal_device_release(replay->device);
  replay->device = NULL;
  iree_vm_context_release(replay->context);
  replay->context = NULL;

  // Create new context.
  // TODO(benvanik): allow setting flags from the trace files.
  return iree_vm_context_create(replay->instance, replay->context_flags,
                                replay->host_allocator, &replay->context);
}

//===----------------------------------------------------------------------===//
// type: module_load
//===----------------------------------------------------------------------===//

// TODO(benvanik): rework this to allow for multiple devices from a device set.
static iree_status_t iree_trace_replay_create_device(
    iree_trace_replay_t* replay, yaml_node_t* device_node,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  // Use the provided driver name or override with the --device= flag.
  iree_string_view_t device_uri = iree_yaml_node_as_string(device_node);
  if (iree_string_view_is_empty(device_uri)) {
    if (replay->device_uri_count != 1) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exactly one device must be specified when none "
                              "is present in the trace file");
    }
    device_uri = replay->device_uris[0];
  }

  // Try to create the device.
  return iree_hal_create_device(replay->driver_registry, device_uri,
                                host_allocator, out_device);
}

static iree_status_t iree_trace_replay_load_builtin_module(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* module_node) {
  iree_vm_module_t* module = NULL;

  yaml_node_t* name_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(document, module_node,
                                              IREE_SV("name"), &name_node));
  if (iree_yaml_string_equal(name_node, IREE_SV("hal"))) {
    yaml_node_t* driver_node = NULL;
    IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
        document, module_node, IREE_SV("driver"), &driver_node));
    IREE_RETURN_IF_ERROR(iree_trace_replay_create_device(
        replay, driver_node, replay->host_allocator, &replay->device));
    IREE_RETURN_IF_ERROR(iree_hal_module_create(
        replay->instance, replay->device, IREE_HAL_MODULE_FLAG_NONE,
        replay->host_allocator, &module));
  }
  if (!module) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND, "builtin module '%.*s' not registered",
        (int)name_node->data.scalar.length, name_node->data.scalar.value);
  }

  iree_status_t status = iree_vm_context_register_modules(
      replay->context, /*module_count=*/1, /*modules=*/&module);
  iree_vm_module_release(module);
  return status;
}

static iree_status_t iree_trace_replay_load_bytecode_module(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* module_node) {
  yaml_node_t* path_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(document, module_node,
                                              IREE_SV("path"), &path_node));

  // Load bytecode file (or stdin) contents into memory.
  iree_file_contents_t* flatbuffer_contents = NULL;
  iree_status_t status = iree_ok_status();
  if (iree_yaml_string_equal(path_node, IREE_SV("<stdin>"))) {
    fprintf(stdout, "Reading bytecode contents from stdin...\n");
    status =
        iree_stdin_read_contents(replay->host_allocator, &flatbuffer_contents);
  } else {
    char* full_path = NULL;
    IREE_RETURN_IF_ERROR(iree_file_path_join(
        replay->root_path, iree_yaml_node_as_string(path_node),
        replay->host_allocator, &full_path));
    status = iree_file_read_contents(full_path, replay->host_allocator,
                                     &flatbuffer_contents);
    iree_allocator_free(replay->host_allocator, full_path);
  }

  // Load and verify the bytecode module.
  iree_vm_module_t* module = NULL;
  if (iree_status_is_ok(status)) {
    iree_allocator_t flatbuffer_deallocator =
        iree_file_contents_deallocator(flatbuffer_contents);
    status = iree_vm_bytecode_module_create(
        replay->instance, flatbuffer_contents->const_buffer,
        flatbuffer_deallocator, replay->host_allocator, &module);
    if (!iree_status_is_ok(status)) {
      iree_file_contents_free(flatbuffer_contents);
    }
  }

  // Register the bytecode module with the context.
  if (iree_status_is_ok(status)) {
    status = iree_vm_context_register_modules(
        replay->context, /*module_count=*/1, /*modules=*/&module);
  }

  iree_vm_module_release(module);
  return status;
}

iree_status_t iree_trace_replay_event_module_load(iree_trace_replay_t* replay,
                                                  yaml_document_t* document,
                                                  yaml_node_t* event_node) {
  yaml_node_t* module_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(document, event_node,
                                              IREE_SV("module"), &module_node));

  yaml_node_t* type_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(document, module_node,
                                              IREE_SV("type"), &type_node));
  iree_string_view_t type = iree_yaml_node_as_string(type_node);

  if (iree_string_view_equal(type, IREE_SV("builtin"))) {
    return iree_trace_replay_load_builtin_module(replay, document, module_node);
  } else if (iree_string_view_equal(type, IREE_SV("bytecode"))) {
    return iree_trace_replay_load_bytecode_module(replay, document,
                                                  module_node);
  }

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "module type '%.*s' not recognized", (int)type.size,
                          type.data);
}

//===----------------------------------------------------------------------===//
// RNG utilities
//===----------------------------------------------------------------------===//
// TODO(benvanik): move these out to another file.

// Parameter for locally defined lcg similar to std::minstd_rand.
#define IREE_PRNG_MULTIPLIER 48271
#define IREE_PRNG_MODULUS 2147483647

// Writes an element of the given |element_type| with the given integral |value|
// to |dst|.
static void iree_trace_replay_write_element(
    iree_hal_element_type_t element_type, int32_t value, void* dst) {
#define IREE_TRACE_REPLAY_WRITE_ELEMENT_CASE(ETYPE, CTYPE) \
  case IREE_HAL_ELEMENT_TYPE_##ETYPE:                      \
    *(CTYPE*)dst = (CTYPE)value;                           \
    break;

  switch (element_type) {
    IREE_TRACE_REPLAY_WRITE_ELEMENT_CASE(INT_8, int8_t)
    IREE_TRACE_REPLAY_WRITE_ELEMENT_CASE(INT_16, int16_t)
    IREE_TRACE_REPLAY_WRITE_ELEMENT_CASE(INT_32, int32_t)
    IREE_TRACE_REPLAY_WRITE_ELEMENT_CASE(INT_64, int64_t)
    IREE_TRACE_REPLAY_WRITE_ELEMENT_CASE(SINT_8, int8_t)
    IREE_TRACE_REPLAY_WRITE_ELEMENT_CASE(SINT_16, int16_t)
    IREE_TRACE_REPLAY_WRITE_ELEMENT_CASE(SINT_32, int32_t)
    IREE_TRACE_REPLAY_WRITE_ELEMENT_CASE(SINT_64, int64_t)
    IREE_TRACE_REPLAY_WRITE_ELEMENT_CASE(UINT_8, uint8_t)
    IREE_TRACE_REPLAY_WRITE_ELEMENT_CASE(UINT_16, uint16_t)
    IREE_TRACE_REPLAY_WRITE_ELEMENT_CASE(UINT_32, uint32_t)
    IREE_TRACE_REPLAY_WRITE_ELEMENT_CASE(UINT_64, uint64_t)
    // clang-format off
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      *(uint16_t*)dst = iree_math_f32_to_f16((float)value);
      break;
    IREE_TRACE_REPLAY_WRITE_ELEMENT_CASE(FLOAT_32, float)
    IREE_TRACE_REPLAY_WRITE_ELEMENT_CASE(FLOAT_64, double)
    // clang-format on
    default:
      IREE_ASSERT(false, "unhandled element type");
      break;
  }

#undef IREE_TRACE_REPLAY_WRITE_ELEMENT_CASE
}

// Simple deterministic pseudorandom generator.
// This function is same as C++'s std::minstd_rand.
static uint32_t iree_tree_replay_pseudorandom_uint32(uint32_t* state) {
  *state = (*state * IREE_PRNG_MULTIPLIER) % IREE_PRNG_MODULUS;
  return *state;
}

// Returns a random uint8_t in the range of [0, UCHAR_MAX].
static uint8_t iree_trace_replay_pseudorandom_uint8(uint32_t* state) {
  // return the second-least-signicant out of the 4 bytes of state. it avoids
  // some mild issues with the least-significant and most-significant bytes.
  return iree_tree_replay_pseudorandom_uint32(state) >> 8;
}

// Returns a random uint32_t in the range [0, range).
static inline uint32_t iree_trace_replay_pseudorandom_range(uint32_t* state,
                                                            uint32_t range) {
  return iree_tree_replay_pseudorandom_uint32(state) % range;
}

// Returns a random double in the range of [0, 1.0).
static double iree_trace_replay_pseudorandom_double(uint32_t* state) {
  const double inv_modulus = 1.0 / IREE_PRNG_MODULUS;
  return iree_tree_replay_pseudorandom_uint32(state) * inv_modulus;
}

// Get minimum and maximum for integer-valued uniform distribution.
static void iree_trace_replay_get_min_max_for_element_type(
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
      *min = +16;
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

// Fills the destination span with pseudorandom values of the given
// |element_type|. The given |seed| is passed to the pseudorandom generator.
// The pseudorandom values are reproducible both across runs and across
// machines.
static void iree_trace_replay_generate_fully_specified_pseudorandom_buffer(
    iree_hal_element_type_t element_type, iree_byte_span_t span,
    uint32_t seed) {
  iree_host_size_t element_byte_count =
      iree_hal_element_dense_byte_count(element_type);
  uint8_t* data_end = span.data + span.data_length;
  uint32_t state = seed;
  uint32_t range;
  int32_t min, max;
  iree_trace_replay_get_min_max_for_element_type(element_type, &min, &max);
  range = (max - min + 1);
  for (uint8_t* data = span.data; data < data_end; data += element_byte_count) {
    // Generate "uniform" integer-valued numbers in the range [min, max].
    int32_t value =
        (int32_t)iree_trace_replay_pseudorandom_range(&state, range) + min;
    iree_trace_replay_write_element(element_type, value, data);
  }
}

//===----------------------------------------------------------------------===//
// Input
//===----------------------------------------------------------------------===//

static iree_status_t iree_trace_replay_parse_item(iree_trace_replay_t* replay,
                                                  yaml_document_t* document,
                                                  yaml_node_t* value_node,
                                                  iree_vm_list_t* target_list);
static iree_status_t iree_trace_replay_parse_item_sequence(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* sequence_node, iree_vm_list_t* target_list);

// Parses a scalar value and appends it to |target_list|.
//
// ```yaml
// i8: 7
// ```
static iree_status_t iree_trace_replay_parse_scalar(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* value_node, iree_vm_list_t* target_list) {
  yaml_node_t* data_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(document, value_node,
                                                  IREE_SV("i8"), &data_node));
  if (data_node) {
    int32_t value = 0;
    if (!iree_string_view_atoi_int32(iree_yaml_node_as_string(data_node),
                                     &value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "failed to parse i8 value: '%.*s'",
          (int)data_node->data.scalar.length, data_node->data.scalar.value);
    }
    iree_vm_variant_t variant = iree_vm_variant_empty();
    variant.type.value_type = IREE_VM_VALUE_TYPE_I8;
    variant.i8 = (int8_t)value;
    return iree_vm_list_push_variant(target_list, &variant);
  }
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(document, value_node,
                                                  IREE_SV("i16"), &data_node));
  if (data_node) {
    int32_t value = 0;
    if (!iree_string_view_atoi_int32(iree_yaml_node_as_string(data_node),
                                     &value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "failed to parse i16 value: '%.*s'",
          (int)data_node->data.scalar.length, data_node->data.scalar.value);
    }
    iree_vm_variant_t variant = iree_vm_variant_empty();
    variant.type.value_type = IREE_VM_VALUE_TYPE_I16;
    variant.i16 = (int16_t)value;
    return iree_vm_list_push_variant(target_list, &variant);
  }
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(document, value_node,
                                                  IREE_SV("i32"), &data_node));
  if (data_node) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    variant.type.value_type = IREE_VM_VALUE_TYPE_I32;
    if (!iree_string_view_atoi_int32(iree_yaml_node_as_string(data_node),
                                     &variant.i32)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "failed to parse i32 value: '%.*s'",
          (int)data_node->data.scalar.length, data_node->data.scalar.value);
    }
    return iree_vm_list_push_variant(target_list, &variant);
  }
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(document, value_node,
                                                  IREE_SV("i64"), &data_node));
  if (data_node) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    variant.type.value_type = IREE_VM_VALUE_TYPE_I64;
    if (!iree_string_view_atoi_int64(iree_yaml_node_as_string(data_node),
                                     &variant.i64)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "failed to parse i64 value: '%.*s'",
          (int)data_node->data.scalar.length, data_node->data.scalar.value);
    }
    return iree_vm_list_push_variant(target_list, &variant);
  }
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(document, value_node,
                                                  IREE_SV("f32"), &data_node));
  if (data_node) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    variant.type.value_type = IREE_VM_VALUE_TYPE_F32;
    if (!iree_string_view_atof(iree_yaml_node_as_string(data_node),
                               &variant.f32)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "failed to parse f32 value: '%.*s'",
          (int)data_node->data.scalar.length, data_node->data.scalar.value);
    }
    return iree_vm_list_push_variant(target_list, &variant);
  }
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(document, value_node,
                                                  IREE_SV("f64"), &data_node));
  if (data_node) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    variant.type.value_type = IREE_VM_VALUE_TYPE_F64;
    if (!iree_string_view_atod(iree_yaml_node_as_string(data_node),
                               &variant.f64)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "failed to parse f64 value: '%.*s'",
          (int)data_node->data.scalar.length, data_node->data.scalar.value);
    }
    return iree_vm_list_push_variant(target_list, &variant);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "(%zu): unimplemented scalar type parser",
                          value_node->start_mark.line);
}

// Parses a !vm.list and appends it to |target_list|.
//
// ```yaml
// items:
// - type: value
//   i8: 7
// - type: vm.list
//   items: ...
// ```
static iree_status_t iree_trace_replay_parse_vm_list(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* value_node, iree_vm_list_t* target_list) {
  if (value_node->type != YAML_MAPPING_NODE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "(%zu): expected sequence node for type",
                            value_node->start_mark.line);
  }
  yaml_node_t* items_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
      document, value_node, IREE_SV("items"), &items_node));

  iree_vm_list_t* list = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(/*element_type=*/NULL,
                                           /*initial_capacity=*/8,
                                           replay->host_allocator, &list));

  iree_status_t status = iree_ok_status();
  if (items_node) {
    status = iree_trace_replay_parse_item_sequence(replay, document, items_node,
                                                   list);
  }

  if (iree_status_is_ok(status)) {
    iree_vm_ref_t list_ref = iree_vm_list_move_ref(list);
    status = iree_vm_list_push_ref_move(target_list, &list_ref);
  }
  if (!iree_status_is_ok(status)) {
    iree_vm_list_release(list);
  }
  return status;
}

// Parses a shape sequence.
//
// ```yaml
// shape:
// - 1
// - 2
// - 3
// ```
// or
// ```yaml
// shape: 1x2x3
// ```
static iree_status_t iree_trace_replay_parse_hal_shape(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* shape_node, iree_host_size_t shape_capacity,
    iree_hal_dim_t* shape, iree_host_size_t* out_shape_rank) {
  iree_host_size_t shape_rank = 0;
  *out_shape_rank = shape_rank;
  if (!shape_node) return iree_ok_status();

  if (shape_node->type == YAML_SCALAR_NODE) {
    // Short-hand using the canonical shape parser (4x8).
    return iree_hal_parse_shape(iree_yaml_node_as_string(shape_node),
                                shape_capacity, out_shape_rank, shape);
  } else if (shape_node->type != YAML_SEQUENCE_NODE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "(%zu): expected scalar or sequence node for shape",
                            shape_node->start_mark.line);
  }

  // Shape dimension list:
  for (yaml_node_item_t* item = shape_node->data.sequence.items.start;
       item != shape_node->data.sequence.items.top; ++item) {
    yaml_node_t* dim_node = yaml_document_get_node(document, *item);
    if (dim_node->type != YAML_SCALAR_NODE) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "(%zu): expected integer shape dimension",
                              dim_node->start_mark.line);
    }
    int64_t dim = 0;
    if (!iree_string_view_atoi_int64(iree_yaml_node_as_string(dim_node),
                                     &dim)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "(%zu): invalid shape dimension '%.*s'",
          dim_node->start_mark.line, (int)dim_node->data.scalar.length,
          dim_node->data.scalar.value);
    }
    if (shape_rank >= shape_capacity) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "(%zu): shape rank overflow (>%zu)",
                              shape_node->start_mark.line, shape_capacity);
    }
    shape[shape_rank++] = (iree_hal_dim_t)dim;
  }
  *out_shape_rank = shape_rank;
  return iree_ok_status();
}

// Parses an element type.
//
// ```yaml
// element_type: 553648160
// ```
// or
// ```yaml
// element_type: f32
// ```
static iree_status_t iree_trace_replay_parse_hal_element_type(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* element_type_node, iree_hal_element_type_t* out_element_type) {
  *out_element_type = IREE_HAL_ELEMENT_TYPE_NONE;

  iree_string_view_t element_type_str =
      iree_yaml_node_as_string(element_type_node);
  if (iree_string_view_is_empty(element_type_str)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "(%zu): element type missing",
                            element_type_node->start_mark.line);
  }

  // If the first character is a digit then interpret as a %d type.
  if (isdigit(element_type_str.data[0])) {
    static_assert(sizeof(*out_element_type) == sizeof(uint32_t), "4 bytes");
    if (!iree_string_view_atoi_uint32(element_type_str, out_element_type)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "(%zu): invalid element type",
                              element_type_node->start_mark.line);
    }
    return iree_ok_status();
  }

  // Parse as a canonical element type.
  return iree_hal_parse_element_type(element_type_str, out_element_type);
}

// Parses an encoding type.
//
// ```yaml
// encoding_type: 553648160
// ```
static iree_status_t iree_trace_replay_parse_hal_encoding_type(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* encoding_type_node,
    iree_hal_encoding_type_t* out_encoding_type) {
  *out_encoding_type = IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR;
  if (!encoding_type_node) {
    return iree_ok_status();
  }

  if (!encoding_type_node) return iree_ok_status();
  iree_string_view_t encoding_type_str =
      iree_yaml_node_as_string(encoding_type_node);
  if (iree_string_view_is_empty(encoding_type_str)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "(%zu): encoding type missing",
                            encoding_type_node->start_mark.line);
  }

  // If the first character is a digit then interpret as a %d type.
  if (isdigit(encoding_type_str.data[0])) {
    static_assert(sizeof(*out_encoding_type) == sizeof(uint32_t), "4 bytes");
    if (!iree_string_view_atoi_uint32(encoding_type_str, out_encoding_type)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "(%zu): invalid encoding type",
                              encoding_type_node->start_mark.line);
    }
    return iree_ok_status();
  }

  // Parse as a canonical encoding type.
  // TODO(#6762): implement iree_hal_parse_element_type.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "iree_hal_parse_encoding_type not implemented");
}

// Parses a serialized !hal.buffer into |buffer|.
//
// ```yaml
// contents: !!binary |
//   AACAPwAAAEAAAEBAAACAQA==
// ```
static iree_status_t iree_trace_replay_parse_hal_buffer_contents(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* contents_node, iree_hal_element_type_t element_type,
    iree_hal_buffer_mapping_t* mapping) {
  if (!contents_node) {
    // Empty contents = zero fill.
    return iree_ok_status();
  } else if (contents_node->type != YAML_SCALAR_NODE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "(%zu): expected scalar node for buffer contents",
                            contents_node->start_mark.line);
  }
  iree_string_view_t value =
      iree_string_view_trim(iree_yaml_node_as_string(contents_node));

  if (strcmp(contents_node->tag, "tag:yaml.org,2002:binary") == 0) {
    return iree_yaml_base64_decode(value, mapping->contents);
  } else if (strcmp(contents_node->tag, "tag:yaml.org,2002:str") == 0) {
    return iree_hal_parse_buffer_elements(value, element_type,
                                          mapping->contents);
  } else {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "(%zu): unimplemented buffer encoding '%s'",
                            contents_node->start_mark.line, contents_node->tag);
  }
}

// Generates the destination |buffer| using the generator specified by
// |generator_node|.
static iree_status_t iree_trace_replay_generate_hal_buffer(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* generator_node, iree_hal_element_type_t element_type,
    const iree_hal_dim_t* shape, iree_host_size_t shape_rank,
    iree_hal_buffer_mapping_t* mapping) {
  if (generator_node->type != YAML_SCALAR_NODE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "(%zu): expected scalar node for buffer contents_generator",
        generator_node->start_mark.line);
  } else if (strcmp(generator_node->tag,
                    "!tag:iree:fully_specified_pseudorandom") == 0) {
    // To enable pseudorandom tests that are both reproducible and invariant
    // under reordering and filtering testcases, the seed is explicitly
    // passed as argument in the contents_generator tag.
    iree_string_view_t seed_str =
        iree_string_view_trim(iree_yaml_node_as_string(generator_node));
    uint32_t seed = 0;
    if (iree_string_view_atoi_uint32(seed_str, &seed)) {
      iree_trace_replay_generate_fully_specified_pseudorandom_buffer(
          element_type, mapping->contents, seed);
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "could not parse the seed argument ('%s') of "
                              "the fully_specified_pseudorandom tag",
                              seed_str.data);
    }
  } else {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED, "(%zu): unimplemented buffer generator '%s'",
        generator_node->start_mark.line, generator_node->tag);
  }
  return iree_ok_status();
}

typedef struct iree_trace_replay_generation_params_t {
  iree_trace_replay_t* replay;
  yaml_document_t* document;
  yaml_node_t* contents_node;
  yaml_node_t* generator_node;
  iree_hal_element_type_t element_type;
  const iree_hal_dim_t* shape;
  iree_host_size_t shape_rank;
} iree_trace_replay_generation_params_t;

static iree_status_t iree_trace_replay_generate_hal_buffer_callback(
    iree_hal_buffer_mapping_t* mapping, void* user_data) {
  iree_trace_replay_generation_params_t* params =
      (iree_trace_replay_generation_params_t*)user_data;
  if (params->contents_node) {
    return iree_trace_replay_parse_hal_buffer_contents(
        params->replay, params->document, params->contents_node,
        params->element_type, mapping);
  } else {
    return iree_trace_replay_generate_hal_buffer(
        params->replay, params->document, params->generator_node,
        params->element_type, params->shape, params->shape_rank, mapping);
  }
}

// Parses a !hal.buffer and appends it to |target_list|.
//
// ```yaml
// shape:
// - 4
// element_type: 553648160
// ```
static iree_status_t iree_trace_replay_parse_hal_buffer(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* value_node, iree_vm_list_t* target_list) {
  yaml_node_t* shape_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
      document, value_node, IREE_SV("shape"), &shape_node));
  iree_hal_dim_t shape[16];
  iree_host_size_t shape_rank = 0;
  IREE_RETURN_IF_ERROR(iree_trace_replay_parse_hal_shape(
      replay, document, shape_node, IREE_ARRAYSIZE(shape), shape, &shape_rank));

  yaml_node_t* element_type_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(
      document, value_node, IREE_SV("element_type"), &element_type_node));
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  IREE_RETURN_IF_ERROR(iree_trace_replay_parse_hal_element_type(
      replay, document, element_type_node, &element_type));

  yaml_node_t* encoding_type_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
      document, value_node, IREE_SV("encoding_type"), &encoding_type_node));
  iree_hal_encoding_type_t encoding_type =
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR;
  IREE_RETURN_IF_ERROR(iree_trace_replay_parse_hal_encoding_type(
      replay, document, encoding_type_node, &encoding_type));

  iree_device_size_t allocation_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_compute_view_size(
      shape_rank, shape, element_type, encoding_type, &allocation_size));

  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(replay->device),
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      allocation_size, iree_const_byte_span_empty(), &buffer));

  iree_vm_ref_t buffer_ref = iree_hal_buffer_move_ref(buffer);
  iree_status_t status = iree_vm_list_push_ref_move(target_list, &buffer_ref);
  iree_vm_ref_release(&buffer_ref);
  return status;
}

// Parses a !hal.buffer_view and appends it to |target_list|.
//
// ```yaml
// shape:
// - 4
// element_type: 553648160
// contents: !!binary |
//   AACAPwAAAEAAAEBAAACAQA==
// ```
static iree_status_t iree_trace_replay_parse_hal_buffer_view(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* value_node, iree_vm_list_t* target_list) {
  yaml_node_t* shape_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
      document, value_node, IREE_SV("shape"), &shape_node));
  iree_hal_dim_t shape[16];
  iree_host_size_t shape_rank = 0;
  IREE_RETURN_IF_ERROR(iree_trace_replay_parse_hal_shape(
      replay, document, shape_node, IREE_ARRAYSIZE(shape), shape, &shape_rank));

  yaml_node_t* element_type_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(
      document, value_node, IREE_SV("element_type"), &element_type_node));
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  IREE_RETURN_IF_ERROR(iree_trace_replay_parse_hal_element_type(
      replay, document, element_type_node, &element_type));

  yaml_node_t* encoding_type_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
      document, value_node, IREE_SV("encoding_type"), &encoding_type_node));
  iree_hal_encoding_type_t encoding_type =
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR;
  IREE_RETURN_IF_ERROR(iree_trace_replay_parse_hal_encoding_type(
      replay, document, encoding_type_node, &encoding_type));

  yaml_node_t* contents_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
      document, value_node, IREE_SV("contents"), &contents_node));

  yaml_node_t* generator_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
      document, value_node, IREE_SV("contents_generator"), &generator_node));

  iree_hal_buffer_view_t* buffer_view = NULL;
  if (contents_node && generator_node) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "(%zu): cannot have both contents and contents_generator",
        generator_node->start_mark.line);
  } else if (contents_node || generator_node) {
    iree_trace_replay_generation_params_t params = {
        .replay = replay,
        .document = document,
        .contents_node = contents_node,
        .generator_node = generator_node,
        .element_type = element_type,
        .shape = shape,
        .shape_rank = shape_rank,
    };
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_generate_buffer(
        iree_hal_device_allocator(replay->device), shape_rank, shape,
        element_type, encoding_type,
        (iree_hal_buffer_params_t){
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        },
        iree_trace_replay_generate_hal_buffer_callback, &params, &buffer_view));
  } else {
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer(
        iree_hal_device_allocator(replay->device), shape_rank, shape,
        element_type, encoding_type,
        (iree_hal_buffer_params_t){
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        },
        iree_const_byte_span_empty(), &buffer_view));
  }

  iree_vm_ref_t buffer_view_ref = iree_hal_buffer_view_move_ref(buffer_view);
  iree_status_t status =
      iree_vm_list_push_ref_move(target_list, &buffer_view_ref);
  iree_vm_ref_release(&buffer_view_ref);
  return status;
}

// Parses a !hal.buffer in tensor form and appends it to |target_list|.
// The tensor form is used to size and initialize the buffer but then the
// metadata is thrown away.
//
// ```yaml
// !hal.buffer 4xf32=[0 1 2 3]
// ```
static iree_status_t iree_trace_replay_parse_inline_hal_buffer(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* value_node, iree_vm_list_t* target_list) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_parse(
      iree_yaml_node_as_string(value_node),
      iree_hal_device_allocator(replay->device), &buffer_view));
  iree_vm_ref_t buffer_ref =
      iree_hal_buffer_retain_ref(iree_hal_buffer_view_buffer(buffer_view));
  iree_status_t status = iree_vm_list_push_ref_move(target_list, &buffer_ref);
  iree_hal_buffer_view_release(buffer_view);
  return status;
}

// Parses a !hal.buffer_view in tensor form and appends it to |target_list|.
//
// ```yaml
// !hal.buffer_view 4xf32=[0 1 2 3]
// ```
static iree_status_t iree_trace_replay_parse_inline_hal_buffer_view(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* value_node, iree_vm_list_t* target_list) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_parse(
      iree_yaml_node_as_string(value_node),
      iree_hal_device_allocator(replay->device), &buffer_view));
  iree_vm_ref_t buffer_view_ref = iree_hal_buffer_view_move_ref(buffer_view);
  return iree_vm_list_push_ref_move(target_list, &buffer_view_ref);
}

// Parses a typed item from |value_node| and appends it to |target_list|.
//
// ```yaml
// type: vm.list
// items:
// - type: value
//   i8: 7
// ```
// or
// ```yaml
// !hal.buffer_view 4xf32=[0 1 2 3]
// ```
static iree_status_t iree_trace_replay_parse_item(iree_trace_replay_t* replay,
                                                  yaml_document_t* document,
                                                  yaml_node_t* value_node,
                                                  iree_vm_list_t* target_list) {
  if (strcmp(value_node->tag, "!hal.buffer") == 0) {
    return iree_trace_replay_parse_inline_hal_buffer(replay, document,
                                                     value_node, target_list);
  } else if (strcmp(value_node->tag, "!hal.buffer_view") == 0) {
    return iree_trace_replay_parse_inline_hal_buffer_view(
        replay, document, value_node, target_list);
  }

  yaml_node_t* type_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(document, value_node,
                                              IREE_SV("type"), &type_node));
  iree_string_view_t type = iree_yaml_node_as_string(type_node);
  if (iree_string_view_equal(type, IREE_SV("null"))) {
    iree_vm_variant_t null_value = iree_vm_variant_empty();
    return iree_vm_list_push_variant(target_list, &null_value);
  } else if (iree_string_view_equal(type, IREE_SV("value"))) {
    return iree_trace_replay_parse_scalar(replay, document, value_node,
                                          target_list);
  } else if (iree_string_view_equal(type, IREE_SV("vm.list"))) {
    return iree_trace_replay_parse_vm_list(replay, document, value_node,
                                           target_list);
  } else if (iree_string_view_equal(type, IREE_SV("hal.buffer"))) {
    return iree_trace_replay_parse_hal_buffer(replay, document, value_node,
                                              target_list);
  } else if (iree_string_view_equal(type, IREE_SV("hal.buffer_view"))) {
    return iree_trace_replay_parse_hal_buffer_view(replay, document, value_node,
                                                   target_list);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "unimplemented type parser: '%.*s'", (int)type.size,
                          type.data);
}

// Parses a sequence of items appending each to |target_list|.
static iree_status_t iree_trace_replay_parse_item_sequence(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* sequence_node, iree_vm_list_t* target_list) {
  for (yaml_node_item_t* item = sequence_node->data.sequence.items.start;
       item != sequence_node->data.sequence.items.top; ++item) {
    yaml_node_t* item_node = yaml_document_get_node(document, *item);
    IREE_RETURN_IF_ERROR(
        iree_trace_replay_parse_item(replay, document, item_node, target_list));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Output
//===----------------------------------------------------------------------===//

static iree_status_t iree_trace_replay_print_item(
    iree_vm_variant_t* value, iree_allocator_t host_allocator);

static iree_status_t iree_trace_replay_print_scalar(iree_vm_variant_t* value) {
  switch (value->type.value_type) {
    case IREE_VM_VALUE_TYPE_I8:
      fprintf(stdout, "i8=%" PRIi8, value->i8);
      break;
    case IREE_VM_VALUE_TYPE_I16:
      fprintf(stdout, "i16=%" PRIi16, value->i16);
      break;
    case IREE_VM_VALUE_TYPE_I32:
      fprintf(stdout, "i32=%" PRIi32, value->i32);
      break;
    case IREE_VM_VALUE_TYPE_I64:
      fprintf(stdout, "i64=%" PRIi64, value->i64);
      break;
    case IREE_VM_VALUE_TYPE_F32:
      fprintf(stdout, "f32=%G", value->f32);
      break;
    case IREE_VM_VALUE_TYPE_F64:
      fprintf(stdout, "f64=%G", value->f64);
      break;
    default:
      fprintf(stdout, "?");
      break;
  }
  return iree_ok_status();
}

static iree_status_t iree_trace_replay_print_vm_list(
    iree_vm_list_t* list, iree_allocator_t host_allocator) {
  for (iree_host_size_t i = 0; i < iree_vm_list_size(list); ++i) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_RETURN_IF_ERROR(iree_vm_list_get_variant(list, i, &variant),
                         "variant %zu not present", i);
    IREE_RETURN_IF_ERROR(
        iree_trace_replay_print_item(&variant, host_allocator));
    fprintf(stdout, "\n");
  }
  return iree_ok_status();
}

static iree_status_t iree_trace_replay_print_item(
    iree_vm_variant_t* value, iree_allocator_t host_allocator) {
  if (iree_vm_variant_is_value(*value)) {
    IREE_RETURN_IF_ERROR(iree_trace_replay_print_scalar(value));
  } else if (iree_vm_variant_is_ref(*value)) {
    if (iree_hal_buffer_view_isa(value->ref)) {
      iree_hal_buffer_view_t* buffer_view =
          iree_hal_buffer_view_deref(value->ref);
      IREE_RETURN_IF_ERROR(iree_hal_buffer_view_fprint(
          stdout, buffer_view,
          /*max_element_count=*/1024, host_allocator));
    } else if (iree_vm_list_isa(value->ref)) {
      iree_vm_list_t* list = iree_vm_list_deref(value->ref);
      IREE_RETURN_IF_ERROR(
          iree_trace_replay_print_vm_list(list, host_allocator));
    } else {
      // TODO(benvanik): a way for ref types to describe themselves.
      fprintf(stdout, "(no printer)");
    }
  } else {
    fprintf(stdout, "(null)");
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// type: call
//===----------------------------------------------------------------------===//

iree_status_t iree_trace_replay_event_call_prepare(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* event_node, iree_vm_function_t* out_function,
    iree_vm_list_t** out_input_list) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_function, 0, sizeof(*out_function));
  *out_input_list = NULL;

  // Resolve the function ('module.function') within the context.
  yaml_node_t* function_node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_yaml_mapping_find(document, event_node, IREE_SV("function"),
                                 &function_node));
  iree_vm_function_t function;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_vm_context_resolve_function(
          replay->context, iree_yaml_node_as_string(function_node), &function));

  // Parse function inputs.
  yaml_node_t* args_node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_yaml_mapping_try_find(document, event_node, IREE_SV("args"),
                                     &args_node));
  iree_vm_list_t* input_list = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vm_list_create(/*element_type=*/NULL, /*initial_capacity=*/8,
                              replay->host_allocator, &input_list));
  iree_status_t status = iree_trace_replay_parse_item_sequence(
      replay, document, args_node, input_list);
  if (iree_status_is_ok(status)) {
    *out_function = function;
    *out_input_list = input_list;
  } else {
    iree_vm_list_release(input_list);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_trace_replay_event_call(iree_trace_replay_t* replay,
                                           yaml_document_t* document,
                                           yaml_node_t* event_node,
                                           iree_vm_list_t** out_output_list) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (out_output_list) *out_output_list = NULL;

  iree_vm_function_t function;
  iree_vm_list_t* input_list = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_trace_replay_event_call_prepare(replay, document, event_node,
                                               &function, &input_list));

  // Invoke the function to produce outputs.
  iree_vm_list_t* output_list = NULL;
  iree_status_t status =
      iree_vm_list_create(/*element_type=*/NULL, /*initial_capacity=*/8,
                          replay->host_allocator, &output_list);
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(replay->context, function,
                            IREE_VM_INVOCATION_FLAG_NONE, /*policy=*/NULL,
                            input_list, output_list, replay->host_allocator);
  }
  iree_vm_list_release(input_list);

  if (iree_status_is_ok(status) && out_output_list) {
    *out_output_list = output_list;
  } else {
    iree_vm_list_release(output_list);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_trace_replay_event_call_stdout(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* event_node) {
  yaml_node_t* function_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(
      document, event_node, IREE_SV("function"), &function_node));
  iree_string_view_t function_name = iree_yaml_node_as_string(function_node);
  fprintf(stdout, "--- CALL[%.*s] ---\n", (int)function_name.size,
          function_name.data);

  // Prepare to call the function.
  iree_vm_function_t function;
  iree_vm_list_t* input_list = NULL;
  IREE_RETURN_IF_ERROR(iree_trace_replay_event_call_prepare(
      replay, document, event_node, &function, &input_list));

  // Invoke the function to produce outputs.
  iree_vm_list_t* output_list = NULL;
  iree_status_t status =
      iree_vm_list_create(/*element_type=*/NULL, /*initial_capacity=*/8,
                          replay->host_allocator, &output_list);
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(replay->context, function,
                            IREE_VM_INVOCATION_FLAG_NONE, /*policy=*/NULL,
                            input_list, output_list, replay->host_allocator);
  }
  iree_vm_list_release(input_list);

  // Print the outputs.
  if (iree_status_is_ok(status)) {
    status =
        iree_trace_replay_print_vm_list(output_list, replay->host_allocator);
  }
  iree_vm_list_release(output_list);

  return status;
}

//===----------------------------------------------------------------------===//
// Event dispatch
//===----------------------------------------------------------------------===//

iree_status_t iree_trace_replay_event(iree_trace_replay_t* replay,
                                      yaml_document_t* document,
                                      yaml_node_t* event_node) {
  if (event_node->type != YAML_MAPPING_NODE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "(%zu): expected mapping node",
                            event_node->start_mark.line);
  }
  yaml_node_t* type_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(document, event_node,
                                              IREE_SV("type"), &type_node));
  if (iree_yaml_string_equal(type_node, IREE_SV("context_load"))) {
    return iree_trace_replay_event_context_load(replay, document, event_node);
  } else if (iree_yaml_string_equal(type_node, IREE_SV("module_load"))) {
    return iree_trace_replay_event_module_load(replay, document, event_node);
  } else if (iree_yaml_string_equal(type_node, IREE_SV("call"))) {
    return iree_trace_replay_event_call_stdout(replay, document, event_node);
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED, "(%zu): unhandled type '%.*s'",
      event_node->start_mark.line, (int)type_node->data.scalar.length,
      type_node->data.scalar.value);
}
