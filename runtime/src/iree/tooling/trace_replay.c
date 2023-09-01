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
#include "iree/modules/hal/module.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/numpy_io.h"
#include "iree/vm/bytecode/module.h"

//===----------------------------------------------------------------------===//
// iree_trace_replay_t
//===----------------------------------------------------------------------===//

iree_status_t iree_trace_replay_initialize(
    iree_string_view_t root_path, iree_vm_instance_t* instance,
    iree_trace_replay_flags_t replay_flags,
    iree_vm_context_flags_t context_flags,
    iree_hal_driver_registry_t* driver_registry,
    iree_allocator_t host_allocator, iree_trace_replay_t* out_replay) {
  memset(out_replay, 0, sizeof(*out_replay));

  IREE_RETURN_IF_ERROR(iree_hal_module_register_all_types(instance));

  out_replay->host_allocator = host_allocator;
  out_replay->root_path = root_path;

  out_replay->replay_flags = replay_flags;

  out_replay->instance = instance;
  iree_vm_instance_retain(out_replay->instance);
  out_replay->context_flags = context_flags;

  out_replay->driver_registry = driver_registry;

  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 8u,
                                 host_allocator, &out_replay->inputs);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 8u,
                                 host_allocator, &out_replay->outputs);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 8u,
                                 host_allocator, &out_replay->blackboard);
  }

  if (!iree_status_is_ok(status)) {
    iree_trace_replay_deinitialize(out_replay);
  }
  return status;
}

void iree_trace_replay_deinitialize(iree_trace_replay_t* replay) {
  iree_vm_list_release(replay->inputs);
  iree_vm_list_release(replay->outputs);
  iree_vm_list_release(replay->blackboard);

  iree_vm_context_release(replay->context);
  for (iree_host_size_t i = 0; i < replay->module_count; ++i) {
    iree_vm_module_release(replay->modules[i]);
  }
  replay->module_count = 0;
  iree_vm_instance_release(replay->instance);

  if (iree_all_bits_set(replay->replay_flags,
                        IREE_TRACE_REPLAY_FLAG_PRINT_STATISTICS)) {
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

void iree_trace_replay_reset(iree_trace_replay_t* replay) {
  iree_vm_list_clear(replay->inputs);
  iree_vm_list_clear(replay->outputs);
  iree_vm_list_clear(replay->blackboard);
}

//===----------------------------------------------------------------------===//
// type: context_load
//===----------------------------------------------------------------------===//

iree_status_t iree_trace_replay_event_context_load(iree_trace_replay_t* replay,
                                                   yaml_document_t* document,
                                                   yaml_node_t* event_node) {
  // Cleanup previous state.
  if (!iree_all_bits_set(replay->replay_flags,
                         IREE_TRACE_REPLAY_FLAG_REUSE_DEVICES)) {
    iree_hal_device_release(replay->device);
    replay->device = NULL;
  }
  iree_vm_context_release(replay->context);
  if (!iree_all_bits_set(replay->replay_flags,
                         IREE_TRACE_REPLAY_FLAG_REUSE_MODULES)) {
    for (iree_host_size_t i = 0; i < replay->module_count; ++i) {
      iree_vm_module_release(replay->modules[i]);
    }
    replay->module_count = 0;
  }
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
  // If there's already a device then reuse that if allowed.
  if (*out_device && iree_all_bits_set(replay->replay_flags,
                                       IREE_TRACE_REPLAY_FLAG_REUSE_DEVICES)) {
    return iree_ok_status();
  }
  iree_hal_device_release(*out_device);

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
  return iree_hal_create_device_from_flags(replay->driver_registry, device_uri,
                                           host_allocator, out_device);
}

static iree_status_t iree_trace_replay_load_builtin_module(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* module_node, iree_vm_module_t** out_module) {
  *out_module = NULL;
  iree_vm_module_t* module = NULL;

  yaml_node_t* name_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(document, module_node,
                                              IREE_SV("name"), &name_node));
  if (iree_yaml_string_equal(name_node, IREE_SV("hal"))) {
    yaml_node_t* device_node = NULL;
    IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
        document, module_node, IREE_SV("device"), &device_node));
    IREE_RETURN_IF_ERROR(iree_trace_replay_create_device(
        replay, device_node, replay->host_allocator, &replay->device));
    IREE_RETURN_IF_ERROR(iree_hal_module_create(
        replay->instance, replay->device, IREE_HAL_MODULE_FLAG_NONE,
        replay->host_allocator, &module));
  }
  if (!module) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND, "builtin module '%.*s' not registered",
        (int)name_node->data.scalar.length, name_node->data.scalar.value);
  }

  *out_module = module;
  return iree_ok_status();
}

static iree_status_t iree_trace_replay_load_bytecode_module(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* module_node, iree_vm_module_t** out_module) {
  *out_module = NULL;

  yaml_node_t* path_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(document, module_node,
                                              IREE_SV("path"), &path_node));
  yaml_node_t* mmap_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(document, module_node,
                                                  IREE_SV("mmap"), &mmap_node));

  // Special case sourcing from stdin, which is useful for fast iteration and
  // tests where iree-compile output is piped directly into the replay tool.
  bool from_stdin = iree_yaml_string_equal(path_node, IREE_SV("<stdin>"));
  if (from_stdin && !iree_const_byte_span_is_empty(replay->stdin_contents)) {
    // A copy of stdin was injected and we can use that instead of going to the
    // system. The data is not owned but guaranteed to be live for as long as
    // we'll need it.
    return iree_vm_bytecode_module_create(
        replay->instance, replay->stdin_contents, iree_allocator_null(),
        replay->host_allocator, out_module);
  }

  // Load bytecode file (or stdin) contents into memory.
  iree_file_contents_t* flatbuffer_contents = NULL;
  iree_status_t status = iree_ok_status();
  if (from_stdin) {
    fprintf(stdout, "Reading bytecode contents from stdin...\n");
    status =
        iree_stdin_read_contents(replay->host_allocator, &flatbuffer_contents);
  } else {
    char* full_path = NULL;
    IREE_RETURN_IF_ERROR(iree_file_path_join(
        replay->root_path, iree_yaml_node_as_string(path_node),
        replay->host_allocator, &full_path));
    status = iree_file_read_contents(
        full_path,
        mmap_node ? IREE_FILE_READ_FLAG_MMAP : IREE_FILE_READ_FLAG_PRELOAD,
        replay->host_allocator, &flatbuffer_contents);
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

  *out_module = module;
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

  // Check the cache to see if we've already loaded the module.
  // We only do this with named modules.
  yaml_node_t* name_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(document, module_node,
                                                  IREE_SV("name"), &name_node));
  if (name_node) {
    iree_string_view_t name = iree_yaml_node_as_string(name_node);
    for (iree_host_size_t i = 0; i < replay->module_count; ++i) {
      if (iree_string_view_equal(iree_vm_module_name(replay->modules[i]),
                                 name)) {
        return iree_vm_context_register_modules(
            replay->context, /*module_count=*/1,
            /*modules=*/&replay->modules[i]);
      }
    }
  }

  // Ensure the cache has room for the module. We can make this a growable list
  // if we end up with lots of modules, but most programs today have 2-3.
  if (replay->module_count + 1 > IREE_ARRAYSIZE(replay->modules)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "maximum unique module count hit; ensure modules "
                            "have consistent names");
  }

  // Load the module.
  iree_vm_module_t* module = NULL;
  if (iree_string_view_equal(type, IREE_SV("builtin"))) {
    IREE_RETURN_IF_ERROR(iree_trace_replay_load_builtin_module(
        replay, document, module_node, &module));
  } else if (iree_string_view_equal(type, IREE_SV("bytecode"))) {
    IREE_RETURN_IF_ERROR(iree_trace_replay_load_bytecode_module(
        replay, document, module_node, &module));
  } else {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "module type '%.*s' not recognized", (int)type.size,
                            type.data);
  }

  // Insert the module into the cache.
  // Note that we ensured there was room above.
  // Note that the list retains the module.
  replay->modules[replay->module_count++] = module;
  iree_vm_module_retain(module);

  // Register the loaded module with the context.
  iree_status_t status = iree_vm_context_register_modules(
      replay->context, /*module_count=*/1, /*modules=*/&module);

  iree_vm_module_release(module);
  return status;
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
  int32_t min = 0;
  int32_t max = 0;
  iree_trace_replay_get_min_max_for_element_type(element_type, &min, &max);
  uint32_t range = (max - min + 1);
  iree_host_size_t element_byte_count =
      iree_hal_element_dense_byte_count(element_type);
  uint8_t* data_end = span.data + span.data_length;
  uint32_t state = seed;
  for (uint8_t* data = span.data; data < data_end; data += element_byte_count) {
    // Generate "uniform" integer-valued numbers in the range [min, max].
    int32_t value =
        (int32_t)iree_trace_replay_pseudorandom_range(&state, range) + min;
    iree_trace_replay_write_element(element_type, value, data);
  }
}

//===----------------------------------------------------------------------===//
// List I/O macros
//===----------------------------------------------------------------------===//

// Parses an I/O macro referencing the replay-global inputs/outputs.
// If |is_move| is true then the value is consumed from |list| and the original
// value in the list is reset to NULL.
//
// ```yaml
// !input.get 0
// !input.take 1
// !output.get 2
// ```
static iree_status_t iree_trace_replay_parse_list_get_macro(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* value_node, iree_vm_list_t* list, bool is_move,
    iree_vm_variant_t* out_result) {
  iree_string_view_t value_str = iree_yaml_node_as_string(value_node);
  int32_t ordinal = 0;
  if (!iree_string_view_atoi_int32(value_str, &ordinal)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "failed to parse I/O ordinal from `%.*s`",
                            (int)value_str.size, value_str.data);
  }
  iree_vm_variant_t variant = iree_vm_variant_empty();
  if (is_move) {
    IREE_RETURN_IF_ERROR(
        iree_vm_list_get_variant_move(list, ordinal, &variant));
  } else {
    IREE_RETURN_IF_ERROR(
        iree_vm_list_get_variant_retain(list, ordinal, &variant));
  }
  *out_result = variant;
  return iree_ok_status();
}

// Parses a list load macro referencing a replay-global |list|.
//
// ```yaml
// # gets |variant| at index 2, leaving it in the list for future use
// [!input.]get 2
// # takes |variant| at index 2, clearing the entry in the list
// [!input.]take 2
// ```
static iree_status_t iree_trace_replay_parse_list_load_macro(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* value_node, iree_string_view_t name, iree_vm_list_t* list,
    iree_vm_variant_t* out_result) {
  if (iree_string_view_equal(name, IREE_SV("get"))) {
    return iree_trace_replay_parse_list_get_macro(
        replay, document, value_node, list,
        /*is_move=*/false, out_result);
  } else if (iree_string_view_equal(name, IREE_SV("take"))) {
    return iree_trace_replay_parse_list_get_macro(replay, document, value_node,
                                                  list,
                                                  /*is_move=*/true, out_result);
  } else if (iree_string_view_equal(name, IREE_SV("pop"))) {
    iree_host_size_t i = iree_vm_list_size(list) - 1;
    IREE_RETURN_IF_ERROR(iree_vm_list_get_variant_move(list, i, out_result));
    return iree_vm_list_resize(list, i);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "unsupported list load macro: `%.*s`", (int)name.size,
                          name.data);
}

// Parses an output-set macro referencing the replay-global outputs.
// The provided |variant| value is set at the specified index.
//
// ```yaml
// !output.set 2
// ```
static iree_status_t iree_trace_replay_parse_list_set_macro(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* value_node, iree_vm_list_t* list, iree_vm_variant_t variant) {
  iree_string_view_t value_str = iree_yaml_node_as_string(value_node);
  int32_t ordinal = 0;
  if (!iree_string_view_atoi_int32(value_str, &ordinal)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "failed to parse I/O ordinal from `%.*s`",
                            (int)value_str.size, value_str.data);
  }
  if (iree_vm_list_size(list) <= ordinal) {
    IREE_RETURN_IF_ERROR(iree_vm_list_resize(list, ordinal + 1));
  }
  return iree_vm_list_set_variant_retain(list, ordinal, &variant);
}

// Parses a list store macro referencing a replay-global |list|.
//
// ```yaml
// # sets |variant| at index 2 in the output list
// [!output.]set 2
// # pushes |variant| to the end of the output list
// [!output.]push
// ```
static iree_status_t iree_trace_replay_parse_list_store_macro(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* value_node, iree_string_view_t name, iree_vm_list_t* list,
    iree_vm_variant_t variant) {
  if (iree_string_view_equal(name, IREE_SV("set"))) {
    return iree_trace_replay_parse_list_set_macro(replay, document, value_node,
                                                  list, variant);
  } else if (iree_string_view_equal(name, IREE_SV("push"))) {
    return iree_vm_list_push_variant_retain(list, &variant);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "unsupported list store macro: `%.*s`",
                          (int)name.size, name.data);
}

//===----------------------------------------------------------------------===//
// YAML value parsing
//===----------------------------------------------------------------------===//

static iree_status_t iree_trace_replay_parse_item(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* value_node, iree_vm_variant_t* out_result);
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
    yaml_node_t* value_node, iree_vm_variant_t* out_result) {
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
    *out_result =
        iree_vm_make_variant_value(iree_vm_value_make_i8((int8_t)value));
    return iree_ok_status();
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
    *out_result =
        iree_vm_make_variant_value(iree_vm_value_make_i16((int16_t)value));
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(document, value_node,
                                                  IREE_SV("i32"), &data_node));
  if (data_node) {
    int32_t value = 0;
    if (!iree_string_view_atoi_int32(iree_yaml_node_as_string(data_node),
                                     &value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "failed to parse i32 value: '%.*s'",
          (int)data_node->data.scalar.length, data_node->data.scalar.value);
    }
    *out_result = iree_vm_make_variant_value(iree_vm_value_make_i32(value));
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(document, value_node,
                                                  IREE_SV("i64"), &data_node));
  if (data_node) {
    int64_t value = 0;
    if (!iree_string_view_atoi_int64(iree_yaml_node_as_string(data_node),
                                     &value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "failed to parse i64 value: '%.*s'",
          (int)data_node->data.scalar.length, data_node->data.scalar.value);
    }
    *out_result = iree_vm_make_variant_value(iree_vm_value_make_i64(value));
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(document, value_node,
                                                  IREE_SV("f32"), &data_node));
  if (data_node) {
    float value = 0.0f;
    if (!iree_string_view_atof(iree_yaml_node_as_string(data_node), &value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "failed to parse f32 value: '%.*s'",
          (int)data_node->data.scalar.length, data_node->data.scalar.value);
    }
    *out_result = iree_vm_make_variant_value(iree_vm_value_make_f32(value));
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(document, value_node,
                                                  IREE_SV("f64"), &data_node));
  if (data_node) {
    double value = 0.0;
    if (!iree_string_view_atod(iree_yaml_node_as_string(data_node), &value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "failed to parse f64 value: '%.*s'",
          (int)data_node->data.scalar.length, data_node->data.scalar.value);
    }
    *out_result = iree_vm_make_variant_value(iree_vm_value_make_f64(value));
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "(%zu): unimplemented scalar type parser",
                          value_node->start_mark.line);
}

// Parses a !vm.list into |out_result|.
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
    yaml_node_t* value_node, iree_vm_variant_t* out_result) {
  if (value_node->type != YAML_MAPPING_NODE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "(%zu): expected sequence node for type",
                            value_node->start_mark.line);
  }
  yaml_node_t* items_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
      document, value_node, IREE_SV("items"), &items_node));

  iree_vm_list_t* list = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                           /*initial_capacity=*/8,
                                           replay->host_allocator, &list));

  iree_status_t status = iree_ok_status();
  if (items_node) {
    status = iree_trace_replay_parse_item_sequence(replay, document, items_node,
                                                   list);
  }

  if (iree_status_is_ok(status)) {
    *out_result = iree_vm_make_variant_ref_assign(iree_vm_list_move_ref(list));
  } else {
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
                              "(%zu): shape rank overflow (>%" PRIhsz ")",
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

// Parses a !hal.buffer into |out_result|.
//
// ```yaml
// shape:
// - 4
// element_type: 553648160
// ```
static iree_status_t iree_trace_replay_parse_hal_buffer(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* value_node, iree_vm_variant_t* out_result) {
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
      allocation_size, &buffer));

  *out_result =
      iree_vm_make_variant_ref_assign(iree_hal_buffer_move_ref(buffer));
  return iree_ok_status();
}

// Parses a !hal.buffer_view into |out_result|.
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
    yaml_node_t* value_node, iree_vm_variant_t* out_result) {
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
        replay->device, iree_hal_device_allocator(replay->device), shape_rank,
        shape, element_type, encoding_type,
        (iree_hal_buffer_params_t){
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        },
        iree_trace_replay_generate_hal_buffer_callback, &params, &buffer_view));
  } else {
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
        replay->device, iree_hal_device_allocator(replay->device), shape_rank,
        shape, element_type, encoding_type,
        (iree_hal_buffer_params_t){
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        },
        iree_const_byte_span_empty(), &buffer_view));
  }

  *out_result = iree_vm_make_variant_ref_assign(
      iree_hal_buffer_view_move_ref(buffer_view));
  return iree_ok_status();
}

// Parses a !hal.buffer in tensor form into |out_result|.
// The tensor form is used to size and initialize the buffer but then the
// metadata is thrown away.
//
// ```yaml
// !hal.buffer 4xf32=[0 1 2 3]
// ```
static iree_status_t iree_trace_replay_parse_inline_hal_buffer(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* value_node, iree_vm_variant_t* out_result) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_parse(
      iree_yaml_node_as_string(value_node), replay->device,
      iree_hal_device_allocator(replay->device), &buffer_view));
  *out_result = iree_vm_make_variant_ref_assign(
      iree_hal_buffer_retain_ref(iree_hal_buffer_view_buffer(buffer_view)));
  iree_hal_buffer_view_release(buffer_view);
  return iree_ok_status();
}

// Parses a !hal.buffer_view in tensor form into |out_result|.
//
// ```yaml
// !hal.buffer_view 4xf32=[0 1 2 3]
// ```
static iree_status_t iree_trace_replay_parse_inline_hal_buffer_view(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* value_node, iree_vm_variant_t* out_result) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_parse(
      iree_yaml_node_as_string(value_node), replay->device,
      iree_hal_device_allocator(replay->device), &buffer_view));
  *out_result = iree_vm_make_variant_ref_assign(
      iree_hal_buffer_view_move_ref(buffer_view));
  return iree_ok_status();
}

// Parses a typed item from |value_node| into |out_result|.
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
static iree_status_t iree_trace_replay_parse_item(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* value_node, iree_vm_variant_t* out_result) {
  iree_string_view_t tag = iree_make_cstring_view(value_node->tag);
  if (iree_string_view_consume_prefix(&tag, IREE_SV("!input."))) {
    return iree_trace_replay_parse_list_load_macro(
        replay, document, value_node, tag, replay->inputs, out_result);
  } else if (iree_string_view_consume_prefix(&tag, IREE_SV("!output."))) {
    return iree_trace_replay_parse_list_load_macro(
        replay, document, value_node, tag, replay->outputs, out_result);
  } else if (iree_string_view_consume_prefix(&tag, IREE_SV("!blackboard."))) {
    return iree_trace_replay_parse_list_load_macro(
        replay, document, value_node, tag, replay->blackboard, out_result);
  } else if (strcmp(value_node->tag, "!hal.buffer") == 0) {
    return iree_trace_replay_parse_inline_hal_buffer(replay, document,
                                                     value_node, out_result);
  } else if (strcmp(value_node->tag, "!hal.buffer_view") == 0) {
    return iree_trace_replay_parse_inline_hal_buffer_view(
        replay, document, value_node, out_result);
  }

  yaml_node_t* type_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(document, value_node,
                                              IREE_SV("type"), &type_node));
  iree_string_view_t type = iree_yaml_node_as_string(type_node);
  if (iree_string_view_equal(type, IREE_SV("null"))) {
    *out_result = iree_vm_variant_empty();
    return iree_ok_status();
  } else if (iree_string_view_equal(type, IREE_SV("value"))) {
    return iree_trace_replay_parse_scalar(replay, document, value_node,
                                          out_result);
  } else if (iree_string_view_equal(type, IREE_SV("vm.list"))) {
    return iree_trace_replay_parse_vm_list(replay, document, value_node,
                                           out_result);
  } else if (iree_string_view_equal(type, IREE_SV("hal.buffer"))) {
    return iree_trace_replay_parse_hal_buffer(replay, document, value_node,
                                              out_result);
  } else if (iree_string_view_equal(type, IREE_SV("hal.buffer_view"))) {
    return iree_trace_replay_parse_hal_buffer_view(replay, document, value_node,
                                                   out_result);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "unimplemented type parser: '%.*s'", (int)type.size,
                          type.data);
}

// Parses a sequence of items appending each to |target_list|.
static iree_status_t iree_trace_replay_parse_item_sequence(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* sequence_node, iree_vm_list_t* target_list) {
  if (!sequence_node) return iree_ok_status();
  for (yaml_node_item_t* item = sequence_node->data.sequence.items.start;
       item != sequence_node->data.sequence.items.top; ++item) {
    yaml_node_t* item_node = yaml_document_get_node(document, *item);
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_RETURN_IF_ERROR(
        iree_trace_replay_parse_item(replay, document, item_node, &variant));
    iree_status_t status =
        iree_vm_list_push_variant_move(target_list, &variant);
    iree_vm_variant_reset(&variant);
    IREE_RETURN_IF_ERROR(status);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Output
//===----------------------------------------------------------------------===//

// Parses a single item.
static iree_status_t iree_trace_replay_parse_result_item(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* item_node, iree_vm_variant_t variant) {
  iree_string_view_t tag = iree_make_cstring_view(item_node->tag);
  if (iree_string_view_consume_prefix(&tag, IREE_SV("!output."))) {
    return iree_trace_replay_parse_list_store_macro(
        replay, document, item_node, tag, replay->outputs, variant);
  } else if (iree_string_view_consume_prefix(&tag, IREE_SV("!blackboard."))) {
    return iree_trace_replay_parse_list_store_macro(
        replay, document, item_node, tag, replay->blackboard, variant);
  }
  // NOTE: we ignore other types currently; we could parse them and compare
  // against the |source_list| values or something.
  return iree_ok_status();
}

// Parses a sequence of items and checks each against |source_list|.
static iree_status_t iree_trace_replay_parse_result_item_sequence(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* sequence_node, iree_vm_list_t* source_list) {
  if (!sequence_node) return iree_ok_status();
  iree_host_size_t i = 0;
  for (yaml_node_item_t* item = sequence_node->data.sequence.items.start;
       item != sequence_node->data.sequence.items.top; ++item, ++i) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_RETURN_IF_ERROR(
        iree_vm_list_get_variant_assign(source_list, i, &variant));
    yaml_node_t* item_node = yaml_document_get_node(document, *item);
    IREE_RETURN_IF_ERROR(iree_trace_replay_parse_result_item(
        replay, document, item_node, variant));
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
      z0, iree_vm_list_create(iree_vm_make_undefined_type_def(),
                              /*initial_capacity=*/8, replay->host_allocator,
                              &input_list));
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

iree_status_t iree_trace_replay_event_call_finish(iree_trace_replay_t* replay,
                                                  yaml_document_t* document,
                                                  yaml_node_t* event_node,
                                                  iree_vm_function_t function,
                                                  iree_vm_list_t* output_list) {
  IREE_TRACE_ZONE_BEGIN(z0);

  yaml_node_t* results_node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_yaml_mapping_try_find(document, event_node, IREE_SV("results"),
                                     &results_node));
  if (results_node) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_trace_replay_parse_result_item_sequence(
                replay, document, results_node, output_list));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_trace_replay_event_call(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* event_node, const iree_trace_replay_call_hooks_t* hooks) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vm_function_t function;
  iree_vm_list_t* input_list = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_trace_replay_event_call_prepare(replay, document, event_node,
                                               &function, &input_list));

  iree_vm_list_t* output_list = NULL;
  iree_status_t status = iree_vm_list_create(
      iree_vm_make_undefined_type_def(), /*initial_capacity=*/8,
      replay->host_allocator, &output_list);

  if (iree_status_is_ok(status) && hooks && hooks->before) {
    status = hooks->before(hooks->user_data, replay, document, event_node,
                           function, input_list);
  }

  // Invoke the function to produce outputs.
  iree_status_t call_status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    call_status = iree_vm_invoke(
        replay->context, function, IREE_VM_INVOCATION_FLAG_NONE,
        /*policy=*/NULL, input_list, output_list, replay->host_allocator);
  }

  if (!iree_status_is_ok(call_status)) {
    if (hooks && hooks->error) {
      status = hooks->error(hooks->user_data, replay, document, event_node,
                            function, call_status);
    } else {
      status = call_status;
    }
  } else if (hooks && hooks->after) {
    status = hooks->after(hooks->user_data, replay, document, event_node,
                          function, output_list);
  }

  // NOTE: we could release this prior to the hooks but if the hooks are timing
  // or capturing performance data we want them to be able to scope themselves
  // to just the call. The downside is that if the hook allocates memory it
  // won't be able to reuse what we're holding on to here.
  iree_vm_list_release(input_list);

  if (iree_status_is_ok(status)) {
    status = iree_trace_replay_event_call_finish(replay, document, event_node,
                                                 function, output_list);
  }
  iree_vm_list_release(output_list);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Blackboard management
//===----------------------------------------------------------------------===//

static iree_status_t iree_trace_replay_event_blackboard_clear(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* event_node) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_vm_list_clear(replay->blackboard);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_trace_replay_event_blackboard_assign(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* event_node) {
  IREE_TRACE_ZONE_BEGIN(z0);

  yaml_node_t* from_node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_yaml_mapping_find(document, event_node, IREE_SV("from"),
                                 &from_node));
  yaml_node_t* to_node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_yaml_mapping_find(document, event_node, IREE_SV("to"), &to_node));

  iree_vm_list_t* list = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vm_list_create(iree_vm_make_undefined_type_def(), 8u,
                              replay->host_allocator, &list));

  iree_status_t status =
      iree_trace_replay_parse_item_sequence(replay, document, from_node, list);
  if (iree_status_is_ok(status)) {
    status = iree_trace_replay_parse_result_item_sequence(replay, document,
                                                          to_node, list);
  }

  iree_vm_list_release(list);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Numpy ndarray management
//===----------------------------------------------------------------------===//

// Loads one or more ndarrays from a .npy file.
//
// Example:
// ```yaml
// type: numpy_load
// path: three_ndarrays.npy
// arrays:
// - !blackboard.set 2
// - !blackboard.set 3
// - !output.set 4
// ```
//
// NOTE: this currently reads things into new buffers; we could try mapping from
// disk and other fancy things where possible.
static iree_status_t iree_trace_replay_event_numpy_load(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* event_node) {
  if (!replay->device) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL module must be loaded before loading numpy arrays");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  yaml_node_t* path_node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_yaml_mapping_find(document, event_node, IREE_SV("path"),
                                 &path_node));
  iree_string_view_t path_str = iree_yaml_node_as_string(path_node);

  yaml_node_t* arrays_node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_yaml_mapping_find(document, event_node, IREE_SV("arrays"),
                                 &arrays_node));

  char* full_path = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_file_path_join(replay->root_path, path_str,
                              replay->host_allocator, &full_path));
  FILE* file = fopen(full_path, "rb");
  iree_allocator_free(replay->host_allocator, full_path);
  if (!file) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to open file `%.*s` for read",
                            (int)path_str.size, path_str.data);
  }

  uint64_t file_length = 0;
  iree_status_t status = iree_file_query_length(file, &file_length);

  iree_hal_buffer_params_t buffer_params = {0};
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_READ;
  buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  if (iree_status_is_ok(status)) {
    for (yaml_node_item_t* item = arrays_node->data.sequence.items.start;
         item != arrays_node->data.sequence.items.top; ++item) {
      // Parse the next array in the file. Note that we may be at the end!
      if (iree_file_is_at(file, file_length)) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "file ended before all arrays were decoded");
        break;
      }
      iree_hal_buffer_view_t* buffer_view = NULL;
      status = iree_numpy_npy_load_ndarray(
          file, IREE_NUMPY_NPY_LOAD_OPTION_DEFAULT, buffer_params,
          replay->device, iree_hal_device_allocator(replay->device),
          &buffer_view);
      if (!iree_status_is_ok(status)) break;

      // Route the loaded value to its destination.
      iree_vm_variant_t variant = iree_vm_make_variant_ref_assign(
          iree_hal_buffer_view_move_ref(buffer_view));
      yaml_node_t* item_node = yaml_document_get_node(document, *item);
      status = iree_trace_replay_parse_result_item(replay, document, item_node,
                                                   variant);
      iree_vm_variant_reset(&variant);
      if (!iree_status_is_ok(status)) break;
    }
  }

  fclose(file);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Saves one or more ndarrays to a .npy file.
//
// Example:
// ```yaml
// type: numpy_save
// path: three_ndarrays.npy
// append: true
// arrays:
// - !blackboard.get 2
// - !blackboard.take 3
// - !input.get 4
// - !hal.buffer_view 4xf32=0,1,2,3
// ```
static iree_status_t iree_trace_replay_event_numpy_save(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* event_node) {
  IREE_TRACE_ZONE_BEGIN(z0);

  yaml_node_t* path_node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_yaml_mapping_find(document, event_node, IREE_SV("path"),
                                 &path_node));
  iree_string_view_t path_str = iree_yaml_node_as_string(path_node);

  yaml_node_t* append_node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_yaml_mapping_try_find(document, event_node, IREE_SV("append"),
                                     &append_node));

  yaml_node_t* arrays_node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_yaml_mapping_find(document, event_node, IREE_SV("arrays"),
                                 &arrays_node));

  char* full_path = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_file_path_join(replay->root_path, path_str,
                              replay->host_allocator, &full_path));
  const char* mode = append_node ? "ab" : "wb";
  FILE* file = fopen(full_path, mode);
  iree_allocator_free(replay->host_allocator, full_path);
  if (!file) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to open file `%.*s` for write",
                            (int)path_str.size, path_str.data);
  }

  iree_status_t status = iree_ok_status();
  for (yaml_node_item_t* item = arrays_node->data.sequence.items.start;
       item != arrays_node->data.sequence.items.top; ++item) {
    yaml_node_t* item_node = yaml_document_get_node(document, *item);
    iree_vm_variant_t variant = iree_vm_variant_empty();
    status =
        iree_trace_replay_parse_item(replay, document, item_node, &variant);
    if (!iree_status_is_ok(status)) break;
    if (!iree_vm_variant_is_ref(variant) ||
        !iree_hal_buffer_view_isa(variant.ref)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "only buffer views can be saved to numpy files");
      break;
    }
    iree_hal_buffer_view_t* buffer_view =
        iree_hal_buffer_view_deref(variant.ref);
    status =
        iree_numpy_npy_save_ndarray(file, IREE_NUMPY_NPY_SAVE_OPTION_DEFAULT,
                                    buffer_view, replay->host_allocator);
    iree_vm_variant_reset(&variant);
  }

  fclose(file);

  IREE_TRACE_ZONE_END(z0);
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
  } else if (iree_yaml_string_equal(type_node, IREE_SV("blackboard_clear"))) {
    return iree_trace_replay_event_blackboard_clear(replay, document,
                                                    event_node);
  } else if (iree_yaml_string_equal(type_node, IREE_SV("assign"))) {
    return iree_trace_replay_event_blackboard_assign(replay, document,
                                                     event_node);
  } else if (iree_yaml_string_equal(type_node, IREE_SV("numpy_load"))) {
    return iree_trace_replay_event_numpy_load(replay, document, event_node);
  } else if (iree_yaml_string_equal(type_node, IREE_SV("numpy_save"))) {
    return iree_trace_replay_event_numpy_save(replay, document, event_node);
  } else if (iree_yaml_string_equal(type_node, IREE_SV("call"))) {
    return iree_trace_replay_event_call(replay, document, event_node,
                                        &replay->call_hooks);
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED, "(%zu): unhandled type '%.*s'",
      event_node->start_mark.line, (int)type_node->data.scalar.length,
      type_node->data.scalar.value);
}
