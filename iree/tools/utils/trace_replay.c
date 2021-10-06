// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tools/utils/trace_replay.h"

#include <ctype.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/internal/file_io.h"
#include "iree/base/internal/file_path.h"
#include "iree/base/tracing.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/bytecode_module.h"

iree_status_t iree_trace_replay_initialize(
    iree_string_view_t root_path, iree_vm_instance_t* instance,
    iree_vm_context_flags_t context_flags, iree_allocator_t host_allocator,
    iree_trace_replay_t* out_replay) {
  memset(out_replay, 0, sizeof(*out_replay));

  IREE_RETURN_IF_ERROR(iree_hal_module_register_types());

  out_replay->root_path = root_path;
  out_replay->instance = instance;
  out_replay->context_flags = context_flags;
  out_replay->host_allocator = host_allocator;
  iree_vm_instance_retain(out_replay->instance);
  return iree_ok_status();
}

void iree_trace_replay_deinitialize(iree_trace_replay_t* replay) {
  iree_hal_device_release(replay->device);
  iree_vm_context_release(replay->context);
  iree_vm_instance_release(replay->instance);
  memset(replay, 0, sizeof(*replay));
}

void iree_trace_replay_set_hal_driver_override(iree_trace_replay_t* replay,
                                               iree_string_view_t driver) {
  replay->driver = driver;
}

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

static iree_status_t iree_trace_replay_create_device(
    iree_trace_replay_t* replay, yaml_node_t* driver_node,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  // Use the provided driver name or override with the --driver= flag.
  iree_string_view_t driver_name = iree_yaml_node_as_string(driver_node);
  if (iree_string_view_is_empty(driver_name)) {
    driver_name = replay->driver;
  }

  // Try to create a device from the driver.
  iree_hal_driver_t* driver = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_driver_registry_try_create_by_name(
      iree_hal_driver_registry_default(), driver_name, host_allocator,
      &driver));
  iree_status_t status =
      iree_hal_driver_create_default_device(driver, host_allocator, out_device);
  iree_hal_driver_release(driver);

  return status;
}

static iree_status_t iree_trace_replay_load_builtin_module(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* module_node) {
  iree_vm_module_t* module = NULL;

  yaml_node_t* name_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(
      document, module_node, iree_make_cstring_view("name"), &name_node));
  if (iree_yaml_string_equal(name_node, iree_make_cstring_view("hal"))) {
    yaml_node_t* driver_node = NULL;
    IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
        document, module_node, iree_make_cstring_view("driver"), &driver_node));
    IREE_RETURN_IF_ERROR(iree_trace_replay_create_device(
        replay, driver_node, replay->host_allocator, &replay->device));
    IREE_RETURN_IF_ERROR(iree_hal_module_create(
        replay->device, replay->host_allocator, &module));
  }
  if (!module) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND, "builtin module '%.*s' not registered",
        (int)name_node->data.scalar.length, name_node->data.scalar.value);
  }

  iree_status_t status =
      iree_vm_context_register_modules(replay->context, &module, 1);
  iree_vm_module_release(module);
  return status;
}

static iree_status_t iree_trace_replay_load_bytecode_module(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* module_node) {
  yaml_node_t* path_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(
      document, module_node, iree_make_cstring_view("path"), &path_node));

  // Load bytecode file (or stdin) contents into memory.
  iree_byte_span_t flatbuffer_data;
  iree_status_t status = iree_ok_status();
  if (iree_yaml_string_equal(path_node, iree_make_cstring_view("<stdin>"))) {
    status = iree_stdin_read_contents(replay->host_allocator, &flatbuffer_data);
  } else {
    char* full_path = NULL;
    IREE_RETURN_IF_ERROR(iree_file_path_join(
        replay->root_path, iree_yaml_node_as_string(path_node),
        replay->host_allocator, &full_path));
    status = iree_file_read_contents(full_path, replay->host_allocator,
                                     &flatbuffer_data);
    iree_allocator_free(replay->host_allocator, full_path);
  }

  // Load and verify the bytecode module.
  iree_vm_module_t* module = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_module_create(
        iree_make_const_byte_span(flatbuffer_data.data,
                                  flatbuffer_data.data_length),
        replay->host_allocator, replay->host_allocator, &module);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(replay->host_allocator, flatbuffer_data.data);
    }
  }

  // Register the bytecode module with the context.
  if (iree_status_is_ok(status)) {
    status = iree_vm_context_register_modules(replay->context, &module, 1);
  }

  iree_vm_module_release(module);
  return status;
}

iree_status_t iree_trace_replay_event_module_load(iree_trace_replay_t* replay,
                                                  yaml_document_t* document,
                                                  yaml_node_t* event_node) {
  yaml_node_t* module_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(
      document, event_node, iree_make_cstring_view("module"), &module_node));

  yaml_node_t* type_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(
      document, module_node, iree_make_cstring_view("type"), &type_node));
  iree_string_view_t type = iree_yaml_node_as_string(type_node);

  if (iree_string_view_equal(type, iree_make_cstring_view("builtin"))) {
    return iree_trace_replay_load_builtin_module(replay, document, module_node);
  } else if (iree_string_view_equal(type, iree_make_cstring_view("bytecode"))) {
    return iree_trace_replay_load_bytecode_module(replay, document,
                                                  module_node);
  }

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "module type '%.*s' not recognized", (int)type.size,
                          type.data);
}

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
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
      document, value_node, iree_make_cstring_view("i8"), &data_node));
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
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
      document, value_node, iree_make_cstring_view("i16"), &data_node));
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
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
      document, value_node, iree_make_cstring_view("i32"), &data_node));
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
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
      document, value_node, iree_make_cstring_view("i64"), &data_node));
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
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
      document, value_node, iree_make_cstring_view("f32"), &data_node));
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
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
      document, value_node, iree_make_cstring_view("f64"), &data_node));
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
      document, value_node, iree_make_cstring_view("items"), &items_node));

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
                                shape_capacity, shape, out_shape_rank);
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
// element_type: 50331680
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
// encoding_type: 50331680
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
static iree_status_t iree_trace_replay_parse_hal_buffer(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* contents_node, iree_hal_element_type_t element_type,
    iree_hal_buffer_t* buffer) {
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

  iree_hal_buffer_mapping_t mapping;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_map_range(buffer, IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, 0,
                                IREE_WHOLE_BUFFER, &mapping));
  iree_status_t status = iree_ok_status();
  if (strcmp(contents_node->tag, "tag:yaml.org,2002:binary") == 0) {
    status = iree_yaml_base64_decode(value, mapping.contents);
  } else if (strcmp(contents_node->tag, "tag:yaml.org,2002:str") == 0) {
    status =
        iree_hal_parse_buffer_elements(value, element_type, mapping.contents);
  } else {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED, "(%zu): unimplemented buffer encoding '%s'",
        contents_node->start_mark.line, contents_node->tag);
  }
  iree_hal_buffer_unmap_range(&mapping);
  return status;
}

// Parses a !hal.buffer_view and appends it to |target_list|.
//
// ```yaml
// shape:
// - 4
// element_type: 50331680
// contents: !!binary |
//   AACAPwAAAEAAAEBAAACAQA==
// ```
static iree_status_t iree_trace_replay_parse_hal_buffer_view(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* value_node, iree_vm_list_t* target_list) {
  yaml_node_t* shape_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
      document, value_node, iree_make_cstring_view("shape"), &shape_node));
  iree_hal_dim_t shape[16];
  iree_host_size_t shape_rank = 0;
  IREE_RETURN_IF_ERROR(iree_trace_replay_parse_hal_shape(
      replay, document, shape_node, IREE_ARRAYSIZE(shape), shape, &shape_rank));

  yaml_node_t* element_type_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(
      document, value_node, iree_make_cstring_view("element_type"),
      &element_type_node));
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  IREE_RETURN_IF_ERROR(iree_trace_replay_parse_hal_element_type(
      replay, document, element_type_node, &element_type));

  yaml_node_t* encoding_type_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
      document, value_node, iree_make_cstring_view("encoding_type"),
      &encoding_type_node));
  iree_hal_encoding_type_t encoding_type =
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR;
  IREE_RETURN_IF_ERROR(iree_trace_replay_parse_hal_encoding_type(
      replay, document, encoding_type_node, &encoding_type));

  yaml_node_t* contents_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
      document, value_node, iree_make_cstring_view("contents"),
      &contents_node));

  iree_device_size_t allocation_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_compute_view_size(
      shape, shape_rank, element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      &allocation_size));

  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(replay->device),
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      IREE_HAL_BUFFER_USAGE_ALL, allocation_size, &buffer));
  iree_status_t status = iree_trace_replay_parse_hal_buffer(
      replay, document, contents_node, element_type, buffer);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(buffer);
    return status;
  }

  iree_hal_buffer_view_t* buffer_view = NULL;
  status = iree_hal_buffer_view_create(buffer, shape, shape_rank, element_type,
                                       encoding_type, &buffer_view);
  iree_hal_buffer_release(buffer);
  IREE_RETURN_IF_ERROR(status);

  iree_vm_ref_t buffer_view_ref = iree_hal_buffer_view_move_ref(buffer_view);
  status = iree_vm_list_push_ref_move(target_list, &buffer_view_ref);
  iree_vm_ref_release(&buffer_view_ref);
  return status;
}

// Parses a !hal.buffer_view in tensor form and appends it to |target_list|.
//
// ```yaml
// !tensor 4xf32=[0 1 2 3]
// ```
static iree_status_t iree_trace_replay_parse_inline_hal_buffer_view(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* value_node, iree_vm_list_t* target_list) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_parse(
      iree_yaml_node_as_string(value_node),
      iree_hal_device_allocator(replay->device), &buffer_view));
  iree_vm_ref_t buffer_view_ref = iree_hal_buffer_view_move_ref(buffer_view);
  iree_status_t status =
      iree_vm_list_push_ref_move(target_list, &buffer_view_ref);
  iree_vm_ref_release(&buffer_view_ref);
  return status;
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
  if (strcmp(value_node->tag, "!hal.buffer_view") == 0) {
    return iree_trace_replay_parse_inline_hal_buffer_view(
        replay, document, value_node, target_list);
  }

  yaml_node_t* type_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(
      document, value_node, iree_make_cstring_view("type"), &type_node));
  iree_string_view_t type = iree_yaml_node_as_string(type_node);
  if (iree_string_view_equal(type, iree_make_cstring_view("null"))) {
    iree_vm_variant_t null_value = iree_vm_variant_empty();
    return iree_vm_list_push_variant(target_list, &null_value);
  } else if (iree_string_view_equal(type, iree_make_cstring_view("value"))) {
    return iree_trace_replay_parse_scalar(replay, document, value_node,
                                          target_list);
  } else if (iree_string_view_equal(type, iree_make_cstring_view("vm.list"))) {
    return iree_trace_replay_parse_vm_list(replay, document, value_node,
                                           target_list);
  } else if (iree_string_view_equal(
                 type, iree_make_cstring_view("hal.buffer_view"))) {
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

static iree_status_t iree_trace_replay_print_item(iree_vm_variant_t* value);

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

static iree_status_t iree_trace_replay_print_vm_list(iree_vm_list_t* list) {
  for (iree_host_size_t i = 0; i < iree_vm_list_size(list); ++i) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_RETURN_IF_ERROR(iree_vm_list_get_variant(list, i, &variant),
                         "variant %zu not present", i);
    IREE_RETURN_IF_ERROR(iree_trace_replay_print_item(&variant));
    fprintf(stdout, "\n");
  }
  return iree_ok_status();
}

static iree_status_t iree_trace_replay_print_hal_buffer_view(
    iree_hal_buffer_view_t* buffer_view) {
  return iree_hal_buffer_view_fprint(stdout, buffer_view,
                                     /*max_element_count=*/1024);
}

static iree_status_t iree_trace_replay_print_item(iree_vm_variant_t* value) {
  if (iree_vm_variant_is_value(*value)) {
    IREE_RETURN_IF_ERROR(iree_trace_replay_print_scalar(value));
  } else if (iree_vm_variant_is_ref(*value)) {
    if (iree_hal_buffer_view_isa(value->ref)) {
      iree_hal_buffer_view_t* buffer_view =
          iree_hal_buffer_view_deref(value->ref);
      IREE_RETURN_IF_ERROR(
          iree_trace_replay_print_hal_buffer_view(buffer_view));
    } else if (iree_vm_list_isa(value->ref)) {
      iree_vm_list_t* list = iree_vm_list_deref(value->ref);
      IREE_RETURN_IF_ERROR(iree_trace_replay_print_vm_list(list));
    } else {
      // TODO(benvanik): a way for ref types to describe themselves.
      fprintf(stdout, "(no printer)");
    }
  } else {
    fprintf(stdout, "(null)");
  }
  return iree_ok_status();
}

iree_status_t iree_trace_replay_event_call_prepare(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* event_node, iree_vm_function_t* out_function,
    iree_vm_list_t** out_input_list) {
  memset(out_function, 0, sizeof(*out_function));
  *out_input_list = NULL;

  // Resolve the function ('module.function') within the context.
  yaml_node_t* function_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(
      document, event_node, iree_make_cstring_view("function"),
      &function_node));
  iree_vm_function_t function;
  IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(
      replay->context, iree_yaml_node_as_string(function_node), &function));

  // Parse function inputs.
  yaml_node_t* args_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_try_find(
      document, event_node, iree_make_cstring_view("args"), &args_node));
  iree_vm_list_t* input_list = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_list_create(/*element_type=*/NULL, /*initial_capacity=*/8,
                          replay->host_allocator, &input_list));
  iree_status_t status = iree_trace_replay_parse_item_sequence(
      replay, document, args_node, input_list);

  if (iree_status_is_ok(status)) {
    *out_function = function;
    *out_input_list = input_list;
  } else {
    iree_vm_list_release(input_list);
  }
  return status;
}

iree_status_t iree_trace_replay_event_call(iree_trace_replay_t* replay,
                                           yaml_document_t* document,
                                           yaml_node_t* event_node,
                                           iree_vm_list_t** out_output_list) {
  if (out_output_list) *out_output_list = NULL;

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

  if (iree_status_is_ok(status) && out_output_list) {
    *out_output_list = output_list;
  } else {
    iree_vm_list_release(output_list);
  }
  return status;
}

static iree_status_t iree_trace_replay_event_call_stdout(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* event_node) {
  yaml_node_t* function_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(
      document, event_node, iree_make_cstring_view("function"),
      &function_node));
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
    status = iree_trace_replay_print_vm_list(output_list);
  }
  iree_vm_list_release(output_list);

  return status;
}

iree_status_t iree_trace_replay_event(iree_trace_replay_t* replay,
                                      yaml_document_t* document,
                                      yaml_node_t* event_node) {
  if (event_node->type != YAML_MAPPING_NODE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "(%zu): expected mapping node",
                            event_node->start_mark.line);
  }
  yaml_node_t* type_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(
      document, event_node, iree_make_cstring_view("type"), &type_node));
  if (iree_yaml_string_equal(type_node,
                             iree_make_cstring_view("context_load"))) {
    return iree_trace_replay_event_context_load(replay, document, event_node);
  } else if (iree_yaml_string_equal(type_node,
                                    iree_make_cstring_view("module_load"))) {
    return iree_trace_replay_event_module_load(replay, document, event_node);
  } else if (iree_yaml_string_equal(type_node,
                                    iree_make_cstring_view("call"))) {
    return iree_trace_replay_event_call_stdout(replay, document, event_node);
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED, "(%zu): unhandled type '%.*s'",
      event_node->start_mark.line, (int)type_node->data.scalar.length,
      type_node->data.scalar.value);
}
