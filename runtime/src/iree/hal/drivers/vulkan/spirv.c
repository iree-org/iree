// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/spirv.h"

#include <string.h>

enum {
  IREE_HAL_VULKAN_SPIRV_MAGIC = 0x07230203u,
  IREE_HAL_VULKAN_SPIRV_HEADER_WORD_COUNT = 5u,
  IREE_HAL_VULKAN_SPIRV_OP_MEMORY_MODEL = 14u,
  IREE_HAL_VULKAN_SPIRV_OP_ENTRY_POINT = 15u,
  IREE_HAL_VULKAN_SPIRV_OP_EXECUTION_MODE = 16u,
  IREE_HAL_VULKAN_SPIRV_OP_CAPABILITY = 17u,
  IREE_HAL_VULKAN_SPIRV_OP_TYPE_STRUCT = 30u,
  IREE_HAL_VULKAN_SPIRV_OP_TYPE_POINTER = 32u,
  IREE_HAL_VULKAN_SPIRV_OP_VARIABLE = 59u,
  IREE_HAL_VULKAN_SPIRV_OP_DECORATE = 71u,
  IREE_HAL_VULKAN_SPIRV_OP_MEMBER_DECORATE = 72u,
  IREE_HAL_VULKAN_SPIRV_OP_MODULE_PROCESSED = 330u,
  IREE_HAL_VULKAN_SPIRV_CAPABILITY_PHYSICAL_STORAGE_BUFFER_ADDRESSES = 5347u,
  IREE_HAL_VULKAN_SPIRV_ADDRESSING_MODEL_PHYSICAL_STORAGE_BUFFER64 = 5348u,
  IREE_HAL_VULKAN_SPIRV_MEMORY_MODEL_GLSL450 = 1u,
  IREE_HAL_VULKAN_SPIRV_EXECUTION_MODEL_GL_COMPUTE = 5u,
  IREE_HAL_VULKAN_SPIRV_EXECUTION_MODE_LOCAL_SIZE = 17u,
  IREE_HAL_VULKAN_SPIRV_STORAGE_CLASS_UNIFORM_CONSTANT = 0u,
  IREE_HAL_VULKAN_SPIRV_STORAGE_CLASS_UNIFORM = 2u,
  IREE_HAL_VULKAN_SPIRV_STORAGE_CLASS_PUSH_CONSTANT = 9u,
  IREE_HAL_VULKAN_SPIRV_STORAGE_CLASS_ATOMIC_COUNTER = 10u,
  IREE_HAL_VULKAN_SPIRV_STORAGE_CLASS_IMAGE = 11u,
  IREE_HAL_VULKAN_SPIRV_STORAGE_CLASS_STORAGE_BUFFER = 12u,
  IREE_HAL_VULKAN_SPIRV_DECORATION_BLOCK = 2u,
  IREE_HAL_VULKAN_SPIRV_DECORATION_BINDING = 33u,
  IREE_HAL_VULKAN_SPIRV_DECORATION_DESCRIPTOR_SET = 34u,
  IREE_HAL_VULKAN_SPIRV_DECORATION_OFFSET = 35u,
  IREE_HAL_VULKAN_SPIRV_BDA_ROOT_MEMBER_COUNT = 6u,
  IREE_HAL_VULKAN_SPIRV_BDA_ROOT_BYTE_LENGTH = 32u,
};

static const uint32_t iree_hal_vulkan_spirv_bda_root_member_offsets[] = {
    0u, 8u, 16u, 20u, 24u, 28u,
};

static iree_status_t iree_hal_vulkan_spirv_next_instruction(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_host_size_t* inout_word_offset, uint16_t* out_opcode,
    uint16_t* out_word_count, const uint32_t** out_operands) {
  if (*inout_word_offset >= spirv_word_count) {
    *out_opcode = 0;
    *out_word_count = 0;
    *out_operands = NULL;
    return iree_ok_status();
  }
  const uint32_t header = spirv_words[*inout_word_offset];
  const uint16_t word_count = (uint16_t)(header >> 16);
  const uint16_t opcode = (uint16_t)(header & 0xFFFFu);
  if (word_count == 0 || *inout_word_offset > spirv_word_count - word_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V instruction at word %" PRIhsz
                            " exceeds module length",
                            *inout_word_offset);
  }
  *inout_word_offset += word_count;
  *out_opcode = opcode;
  *out_word_count = word_count;
  *out_operands = &spirv_words[*inout_word_offset - word_count + 1];
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_spirv_string_operand(
    const uint32_t* words, uint16_t word_count, const char* description,
    iree_string_view_t* out_value) {
  const char* name = (const char*)words;
  const iree_host_size_t name_capacity = word_count * sizeof(uint32_t);
  iree_host_size_t name_length = 0;
  while (name_length < name_capacity && name[name_length] != 0) {
    ++name_length;
  }
  if (name_length == name_capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s is not NUL-terminated", description);
  }
  *out_value = iree_make_string_view(name, name_length);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_spirv_entry_point_name(
    const uint32_t* operands, uint16_t operand_word_count,
    iree_string_view_t* out_name) {
  if (operand_word_count < 3) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V entry point instruction is truncated");
  }
  return iree_hal_vulkan_spirv_string_operand(
      &operands[2], (uint16_t)(operand_word_count - 2),
      "SPIR-V entry point name", out_name);
}

static iree_status_t iree_hal_vulkan_spirv_module_processed_string(
    const uint32_t* operands, uint16_t word_count,
    iree_string_view_t* out_string) {
  if (word_count < 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SPIR-V OpModuleProcessed instruction is truncated");
  }
  return iree_hal_vulkan_spirv_string_operand(
      operands, (uint16_t)(word_count - 1), "SPIR-V OpModuleProcessed string",
      out_string);
}

static iree_status_t iree_hal_vulkan_spirv_parse_metadata_uint64(
    iree_string_view_t value, uint64_t max_value, const char* field_name,
    uint64_t* out_value) {
  if (iree_string_view_is_empty(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan BDA metadata field '%s' is empty",
                            field_name);
  }
  uint64_t parsed_value = 0;
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    const char c = value.data[i];
    if (c < '0' || c > '9') {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Vulkan BDA metadata field '%s' has non-decimal value '%.*s'",
          field_name, (int)value.size, value.data);
    }
    const uint64_t digit = (uint64_t)(c - '0');
    if (parsed_value > (max_value - digit) / 10u) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan BDA metadata field '%s' value '%.*s' overflows", field_name,
          (int)value.size, value.data);
    }
    parsed_value = parsed_value * 10u + digit;
  }
  *out_value = parsed_value;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_spirv_parse_metadata_uint32(
    iree_string_view_t value, uint32_t max_value, const char* field_name,
    uint32_t* out_value) {
  uint64_t parsed_value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_parse_metadata_uint64(
      value, max_value, field_name, &parsed_value));
  *out_value = (uint32_t)parsed_value;
  return iree_ok_status();
}

static bool iree_hal_vulkan_spirv_storage_class_is_descriptor_backed(
    uint32_t storage_class) {
  switch (storage_class) {
    case IREE_HAL_VULKAN_SPIRV_STORAGE_CLASS_UNIFORM_CONSTANT:
    case IREE_HAL_VULKAN_SPIRV_STORAGE_CLASS_UNIFORM:
    case IREE_HAL_VULKAN_SPIRV_STORAGE_CLASS_ATOMIC_COUNTER:
    case IREE_HAL_VULKAN_SPIRV_STORAGE_CLASS_IMAGE:
    case IREE_HAL_VULKAN_SPIRV_STORAGE_CLASS_STORAGE_BUFFER:
      return true;
    default:
      return false;
  }
}

static iree_status_t iree_hal_vulkan_spirv_verify_module_header(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count) {
  if (IREE_UNLIKELY(!spirv_words && spirv_word_count != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V module word storage is NULL");
  }
  if (spirv_word_count < IREE_HAL_VULKAN_SPIRV_HEADER_WORD_COUNT ||
      spirv_words[0] != IREE_HAL_VULKAN_SPIRV_MAGIC) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module is not SPIR-V");
  }
  return iree_ok_status();
}

typedef struct iree_hal_vulkan_spirv_scan_state_t {
  // Optional module-wide fact output.
  iree_hal_vulkan_spirv_module_analysis_t* analysis;

  // Optional table receiving compute entry point records.
  iree_hal_vulkan_spirv_compute_entry_point_t* entry_points;

  // Capacity of |entry_points|.
  iree_host_size_t entry_point_capacity;

  // Number of compute entry point records captured.
  iree_host_size_t entry_point_count;

  // Whether |lookup_entry_point| should be matched while scanning.
  bool lookup_entry_point_enabled;

  // Compute entry point name to look up.
  iree_string_view_t lookup_entry_point;

  // Whether |lookup_entry_point| was found.
  bool lookup_entry_point_found;

  // SPIR-V result id of |lookup_entry_point| when found.
  uint32_t lookup_entry_point_id;

  // Optional 3-element LocalSize output for |lookup_entry_point|.
  uint32_t* lookup_workgroup_size;

  // Whether |lookup_workgroup_size| has been populated.
  bool lookup_workgroup_size_found;
} iree_hal_vulkan_spirv_scan_state_t;

static iree_hal_vulkan_spirv_compute_entry_point_t*
iree_hal_vulkan_spirv_find_compute_entry_point_by_id(
    iree_host_size_t entry_point_count,
    iree_hal_vulkan_spirv_compute_entry_point_t* entry_points,
    uint32_t entry_point_id) {
  for (iree_host_size_t i = 0; i < entry_point_count; ++i) {
    if (entry_points[i].id == entry_point_id) return &entry_points[i];
  }
  return NULL;
}

static bool iree_hal_vulkan_spirv_workgroup_size_is_specified(
    const uint32_t workgroup_size[3]) {
  return workgroup_size[0] != 0 || workgroup_size[1] != 0 ||
         workgroup_size[2] != 0;
}

static iree_status_t iree_hal_vulkan_spirv_scan_compute_entry_point(
    const uint32_t* operands, uint16_t word_count,
    iree_hal_vulkan_spirv_scan_state_t* state) {
  if (word_count < 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V OpEntryPoint instruction is truncated");
  }
  if (operands[0] != IREE_HAL_VULKAN_SPIRV_EXECUTION_MODEL_GL_COMPUTE) {
    return iree_ok_status();
  }

  iree_string_view_t name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_entry_point_name(
      operands, (uint16_t)(word_count - 1), &name));

  if (state->analysis) {
    iree_host_size_t name_storage_size = 0;
    if (!iree_host_size_checked_add(name.size, /*NUL=*/1, &name_storage_size) ||
        !iree_host_size_checked_add(
            state->analysis->compute_entry_point_name_storage_size,
            name_storage_size,
            &state->analysis->compute_entry_point_name_storage_size)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "SPIR-V compute entry point name storage overflows");
    }
    ++state->analysis->compute_entry_point_count;
  }

  if (state->entry_points) {
    if (state->entry_point_count >= state->entry_point_capacity) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "SPIR-V compute entry point table overflowed");
    }
    for (iree_host_size_t i = 0; i < state->entry_point_count; ++i) {
      if (iree_string_view_equal(state->entry_points[i].name, name)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "SPIR-V module has duplicate compute entry "
                                "point '%.*s'",
                                (int)name.size, name.data);
      }
    }
    iree_hal_vulkan_spirv_compute_entry_point_t* entry_point =
        &state->entry_points[state->entry_point_count++];
    memset(entry_point, 0, sizeof(*entry_point));
    entry_point->name = name;
    entry_point->id = operands[1];
  }

  if (state->lookup_entry_point_enabled &&
      iree_string_view_equal(name, state->lookup_entry_point)) {
    if (state->lookup_entry_point_found) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "SPIR-V module has duplicate compute entry "
                              "point '%.*s'",
                              (int)name.size, name.data);
    }
    state->lookup_entry_point_found = true;
    state->lookup_entry_point_id = operands[1];
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_spirv_scan_compute_execution_mode(
    const uint32_t* operands, uint16_t word_count,
    iree_hal_vulkan_spirv_scan_state_t* state) {
  // Valid SPIR-V declares OpEntryPoint before OpExecutionMode, so LocalSize
  // reflection can attach directly to the entry point table during one scan.
  if (!state->entry_points && !state->lookup_workgroup_size) {
    return iree_ok_status();
  }
  if (word_count < 3) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V OpExecutionMode instruction is truncated");
  }
  if (operands[1] != IREE_HAL_VULKAN_SPIRV_EXECUTION_MODE_LOCAL_SIZE) {
    return iree_ok_status();
  }
  if (word_count < 6) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V LocalSize execution mode is truncated");
  }
  const uint32_t workgroup_size[3] = {operands[2], operands[3], operands[4]};
  if (workgroup_size[0] == 0 || workgroup_size[1] == 0 ||
      workgroup_size[2] == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SPIR-V LocalSize execution mode must have non-zero dimensions");
  }

  if (state->entry_points) {
    iree_hal_vulkan_spirv_compute_entry_point_t* entry_point =
        iree_hal_vulkan_spirv_find_compute_entry_point_by_id(
            state->entry_point_count, state->entry_points, operands[0]);
    if (entry_point) {
      if (iree_hal_vulkan_spirv_workgroup_size_is_specified(
              entry_point->workgroup_size)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "SPIR-V module has duplicate LocalSize execution modes for "
            "compute entry point '%.*s'",
            (int)entry_point->name.size, entry_point->name.data);
      }
      memcpy(entry_point->workgroup_size, workgroup_size,
             sizeof(entry_point->workgroup_size));
    }
  }

  if (state->lookup_workgroup_size && state->lookup_entry_point_found &&
      operands[0] == state->lookup_entry_point_id) {
    if (state->lookup_workgroup_size_found) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "SPIR-V module has duplicate LocalSize "
                              "execution modes for compute entry point '%.*s'",
                              (int)state->lookup_entry_point.size,
                              state->lookup_entry_point.data);
    }
    memcpy(state->lookup_workgroup_size, workgroup_size, sizeof(uint32_t) * 3);
    state->lookup_workgroup_size_found = true;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_spirv_scan_module(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_hal_vulkan_spirv_scan_state_t* state) {
  if (state->analysis) memset(state->analysis, 0, sizeof(*state->analysis));
  state->entry_point_count = 0;
  state->lookup_entry_point_found = false;
  state->lookup_entry_point_id = 0;
  state->lookup_workgroup_size_found = false;
  if (state->lookup_workgroup_size) {
    memset(state->lookup_workgroup_size, 0, sizeof(uint32_t) * 3);
  }
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_verify_module_header(
      spirv_words, spirv_word_count));

  iree_host_size_t word_offset = IREE_HAL_VULKAN_SPIRV_HEADER_WORD_COUNT;
  while (word_offset < spirv_word_count) {
    uint16_t opcode = 0;
    uint16_t word_count = 0;
    const uint32_t* operands = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_next_instruction(
        spirv_words, spirv_word_count, &word_offset, &opcode, &word_count,
        &operands));
    if (opcode == IREE_HAL_VULKAN_SPIRV_OP_ENTRY_POINT) {
      IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_scan_compute_entry_point(
          operands, word_count, state));
      continue;
    }
    if (opcode == IREE_HAL_VULKAN_SPIRV_OP_EXECUTION_MODE) {
      IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_scan_compute_execution_mode(
          operands, word_count, state));
      if (!state->analysis) continue;
    }
    if (!state->analysis) continue;
    switch (opcode) {
      case IREE_HAL_VULKAN_SPIRV_OP_MEMORY_MODEL:
        if (word_count < 3) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "SPIR-V OpMemoryModel instruction is truncated");
        }
        state->analysis->uses_physical_storage_buffer64_glsl450 =
            operands[0] ==
                IREE_HAL_VULKAN_SPIRV_ADDRESSING_MODEL_PHYSICAL_STORAGE_BUFFER64 &&
            operands[1] == IREE_HAL_VULKAN_SPIRV_MEMORY_MODEL_GLSL450;
        break;
      case IREE_HAL_VULKAN_SPIRV_OP_CAPABILITY:
        if (word_count != 2) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "SPIR-V OpCapability instruction is malformed");
        }
        if (operands[0] ==
            IREE_HAL_VULKAN_SPIRV_CAPABILITY_PHYSICAL_STORAGE_BUFFER_ADDRESSES) {
          state->analysis->has_physical_storage_buffer_addresses_capability =
              true;
        }
        break;
      case IREE_HAL_VULKAN_SPIRV_OP_DECORATE:
        if (word_count < 3) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "SPIR-V OpDecorate instruction is truncated");
        }
        if (operands[1] == IREE_HAL_VULKAN_SPIRV_DECORATION_BINDING ||
            operands[1] == IREE_HAL_VULKAN_SPIRV_DECORATION_DESCRIPTOR_SET) {
          state->analysis->has_descriptor_binding_decorations = true;
        }
        break;
      case IREE_HAL_VULKAN_SPIRV_OP_VARIABLE:
        if (word_count < 4) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "SPIR-V OpVariable instruction is truncated");
        }
        if (operands[2] == IREE_HAL_VULKAN_SPIRV_STORAGE_CLASS_PUSH_CONSTANT) {
          ++state->analysis->push_constant_variable_count;
          if (state->analysis->push_constant_variable_count == 1) {
            state->analysis->single_push_constant_pointer_type_id = operands[0];
          } else {
            state->analysis->single_push_constant_pointer_type_id = 0;
          }
        }
        if (iree_hal_vulkan_spirv_storage_class_is_descriptor_backed(
                operands[2])) {
          state->analysis->has_descriptor_storage_class_variables = true;
        }
        break;
      default:
        break;
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_spirv_verify_module(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count) {
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_verify_module_header(
      spirv_words, spirv_word_count));

  iree_host_size_t word_offset = IREE_HAL_VULKAN_SPIRV_HEADER_WORD_COUNT;
  while (word_offset < spirv_word_count) {
    uint16_t opcode = 0;
    uint16_t word_count = 0;
    const uint32_t* operands = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_next_instruction(
        spirv_words, spirv_word_count, &word_offset, &opcode, &word_count,
        &operands));
  }
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_spirv_analyze_module(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_hal_vulkan_spirv_module_analysis_t* out_analysis) {
  IREE_ASSERT_ARGUMENT(out_analysis);
  iree_hal_vulkan_spirv_scan_state_t state = {
      .analysis = out_analysis,
  };
  return iree_hal_vulkan_spirv_scan_module(spirv_words, spirv_word_count,
                                           &state);
}

static void iree_hal_vulkan_spirv_bda_dispatch_metadata_initialize(
    iree_hal_vulkan_spirv_bda_dispatch_metadata_t* out_metadata) {
  memset(out_metadata, 0, sizeof(*out_metadata));
  out_metadata->root_push_constant_offset = 0;
  out_metadata->root_push_constant_length =
      IREE_HAL_VULKAN_SPIRV_BDA_ROOT_BYTE_LENGTH;
  out_metadata->constant_push_constant_offset =
      IREE_HAL_VULKAN_SPIRV_BDA_ROOT_BYTE_LENGTH;
}

void iree_hal_vulkan_spirv_bda_dispatch_metadata_deinitialize(
    iree_hal_vulkan_spirv_bda_dispatch_metadata_t* metadata,
    iree_allocator_t host_allocator) {
  if (!metadata) return;
  iree_allocator_free(host_allocator, metadata->binding_requirements);
  iree_hal_vulkan_spirv_bda_dispatch_metadata_initialize(metadata);
}

typedef struct iree_hal_vulkan_spirv_bda_metadata_parse_state_t {
  // Reflected metadata receiving parsed scalar fields.
  iree_hal_vulkan_spirv_bda_dispatch_metadata_t* metadata;

  // Whether root layout metadata was already parsed.
  bool has_root_layout;

  // Whether constant offset metadata was already parsed.
  bool has_constant_offset;

  // Whether constant count metadata was already parsed.
  bool has_constant_count;

  // Whether binding count metadata was already parsed.
  bool has_binding_count;

  // Whether any per-binding requirement metadata was found.
  bool has_binding_requirements;

  // Whether this pass should populate |metadata->binding_requirements|.
  bool populate_binding_requirements;
} iree_hal_vulkan_spirv_bda_metadata_parse_state_t;

static iree_status_t iree_hal_vulkan_spirv_parse_metadata_pair_u32_u32(
    iree_string_view_t value, const char* field_name, uint32_t* out_lhs,
    uint32_t* out_rhs) {
  iree_string_view_t lhs = iree_string_view_empty();
  iree_string_view_t rhs = iree_string_view_empty();
  if (iree_string_view_split(value, ',', &lhs, &rhs) < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan BDA metadata field '%s' must be '<uint32>,<uint32>'",
        field_name);
  }
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_parse_metadata_uint32(
      lhs, UINT32_MAX, field_name, out_lhs));
  return iree_hal_vulkan_spirv_parse_metadata_uint32(rhs, UINT32_MAX,
                                                     field_name, out_rhs);
}

static iree_status_t iree_hal_vulkan_spirv_parse_metadata_pair_u32_u64(
    iree_string_view_t value, const char* field_name, uint32_t* out_lhs,
    uint64_t* out_rhs) {
  iree_string_view_t lhs = iree_string_view_empty();
  iree_string_view_t rhs = iree_string_view_empty();
  if (iree_string_view_split(value, ',', &lhs, &rhs) < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan BDA metadata field '%s' must be '<uint32>,<uint64>'",
        field_name);
  }
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_parse_metadata_uint32(
      lhs, UINT32_MAX, field_name, out_lhs));
  return iree_hal_vulkan_spirv_parse_metadata_uint64(rhs, UINT64_MAX,
                                                     field_name, out_rhs);
}

static iree_status_t iree_hal_vulkan_spirv_parse_bda_metadata_string(
    iree_string_view_t value,
    iree_hal_vulkan_spirv_bda_metadata_parse_state_t* state) {
  static const iree_string_view_t prefix =
      iree_string_view_literal("iree.vulkan.bda.v1");
  if (!iree_string_view_starts_with(value, prefix)) return iree_ok_status();

  value = iree_string_view_remove_prefix(value, prefix.size);
  iree_hal_vulkan_spirv_bda_dispatch_metadata_t* metadata = state->metadata;
  if (iree_string_view_is_empty(value)) {
    metadata->is_present = true;
    return iree_ok_status();
  }
  if (!iree_string_view_consume_prefix(&value, IREE_SV("."))) {
    return iree_ok_status();
  }
  metadata->is_present = true;
  if (state->populate_binding_requirements &&
      !iree_string_view_starts_with(value, IREE_SV("binding."))) {
    return iree_ok_status();
  }

  if (iree_string_view_consume_prefix(&value, IREE_SV("root="))) {
    if (state->has_root_layout) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Vulkan BDA metadata root layout is duplicated");
    }
    state->has_root_layout = true;
    return iree_hal_vulkan_spirv_parse_metadata_pair_u32_u32(
        value, "root", &metadata->root_push_constant_offset,
        &metadata->root_push_constant_length);
  }

  if (iree_string_view_consume_prefix(&value, IREE_SV("constant_offset="))) {
    if (state->has_constant_offset) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Vulkan BDA metadata constant offset is "
                              "duplicated");
    }
    state->has_constant_offset = true;
    uint32_t constant_offset = 0;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_parse_metadata_uint32(
        value, UINT32_MAX, "constant_offset", &constant_offset));
    metadata->constant_push_constant_offset = constant_offset;
    return iree_ok_status();
  }

  if (iree_string_view_consume_prefix(&value, IREE_SV("constants="))) {
    if (state->has_constant_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Vulkan BDA metadata constant count is "
                              "duplicated");
    }
    state->has_constant_count = true;
    uint32_t constant_count = 0;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_parse_metadata_uint32(
        value, UINT16_MAX, "constants", &constant_count));
    metadata->constant_count = (uint16_t)constant_count;
    return iree_ok_status();
  }

  if (iree_string_view_consume_prefix(&value, IREE_SV("bindings="))) {
    if (state->has_binding_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Vulkan BDA metadata binding count is "
                              "duplicated");
    }
    state->has_binding_count = true;
    uint32_t binding_count = 0;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_parse_metadata_uint32(
        value, UINT16_MAX, "bindings", &binding_count));
    metadata->binding_count = (uint16_t)binding_count;
    return iree_ok_status();
  }

  if (iree_string_view_consume_prefix(&value, IREE_SV("binding."))) {
    iree_string_view_t ordinal_text = iree_string_view_empty();
    iree_string_view_t requirement_text = iree_string_view_empty();
    if (iree_string_view_split(value, '=', &ordinal_text, &requirement_text) <
        0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Vulkan BDA binding metadata must be "
          "'binding.<ordinal>=<alignment>,<minimum_length>'");
    }
    uint32_t binding_ordinal = 0;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_parse_metadata_uint32(
        ordinal_text, UINT16_MAX, "binding ordinal", &binding_ordinal));
    uint32_t minimum_alignment = 0;
    uint64_t minimum_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_parse_metadata_pair_u32_u64(
        requirement_text, "binding requirement", &minimum_alignment,
        &minimum_length));
    if (minimum_alignment == 0 ||
        !iree_device_size_is_power_of_two(minimum_alignment)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Vulkan BDA binding metadata for ordinal %u has invalid alignment %u",
          binding_ordinal, minimum_alignment);
    }
    state->has_binding_requirements = true;
    if (!state->populate_binding_requirements) return iree_ok_status();
    if (binding_ordinal >= metadata->binding_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Vulkan BDA binding metadata ordinal %u exceeds binding count %u",
          binding_ordinal, metadata->binding_count);
    }
    iree_hal_vulkan_spirv_bda_binding_requirement_t* requirement =
        &metadata->binding_requirements[binding_ordinal];
    if (requirement->minimum_alignment != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Vulkan BDA binding metadata for ordinal %u is duplicated",
          binding_ordinal);
    }
    *requirement = (iree_hal_vulkan_spirv_bda_binding_requirement_t){
        .minimum_alignment = minimum_alignment,
        .minimum_length = minimum_length,
    };
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "Vulkan BDA metadata string has unknown field '%.*s'",
                          (int)value.size, value.data);
}

static iree_status_t iree_hal_vulkan_spirv_scan_bda_metadata_strings(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_hal_vulkan_spirv_bda_metadata_parse_state_t* state) {
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_verify_module_header(
      spirv_words, spirv_word_count));
  iree_host_size_t word_offset = IREE_HAL_VULKAN_SPIRV_HEADER_WORD_COUNT;
  while (word_offset < spirv_word_count) {
    uint16_t opcode = 0;
    uint16_t word_count = 0;
    const uint32_t* operands = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_next_instruction(
        spirv_words, spirv_word_count, &word_offset, &opcode, &word_count,
        &operands));
    if (opcode != IREE_HAL_VULKAN_SPIRV_OP_MODULE_PROCESSED) continue;
    iree_string_view_t value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_module_processed_string(
        operands, word_count, &value));
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_spirv_parse_bda_metadata_string(value, state));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_spirv_validate_bda_metadata(
    const iree_hal_vulkan_spirv_bda_metadata_parse_state_t* state) {
  const iree_hal_vulkan_spirv_bda_dispatch_metadata_t* metadata =
      state->metadata;
  if (!metadata->is_present) return iree_ok_status();
  if (metadata->root_push_constant_offset != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "raw Vulkan BDA metadata root push constant offset must be zero");
  }
  if (metadata->root_push_constant_length !=
      IREE_HAL_VULKAN_SPIRV_BDA_ROOT_BYTE_LENGTH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "raw Vulkan BDA metadata root push constant length %u does not match "
        "ABI v1 length %u",
        metadata->root_push_constant_length,
        IREE_HAL_VULKAN_SPIRV_BDA_ROOT_BYTE_LENGTH);
  }
  if ((metadata->constant_push_constant_offset % sizeof(uint32_t)) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "raw Vulkan BDA metadata constant push constant offset must be "
        "4-byte aligned");
  }
  const uint64_t constant_length =
      (uint64_t)metadata->constant_count * sizeof(uint32_t);
  const uint64_t constant_end =
      (uint64_t)metadata->constant_push_constant_offset + constant_length;
  if (constant_end > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "raw Vulkan BDA metadata constant push constant range overflows");
  }
  const uint64_t root_end = (uint64_t)metadata->root_push_constant_offset +
                            metadata->root_push_constant_length;
  if (constant_length != 0 &&
      metadata->root_push_constant_offset < constant_end &&
      metadata->constant_push_constant_offset < root_end) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "raw Vulkan BDA metadata inline constants overlap the hidden root");
  }
  if (state->has_binding_requirements && !state->has_binding_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "raw Vulkan BDA binding requirements require an "
                            "explicit binding count");
  }
  if (state->has_binding_requirements && metadata->binding_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "raw Vulkan BDA binding requirements require a non-zero binding count");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_spirv_parse_bda_dispatch_metadata(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_allocator_t host_allocator,
    iree_hal_vulkan_spirv_bda_dispatch_metadata_t* out_metadata) {
  IREE_ASSERT_ARGUMENT(out_metadata);
  iree_hal_vulkan_spirv_bda_dispatch_metadata_initialize(out_metadata);

  iree_hal_vulkan_spirv_bda_metadata_parse_state_t state = {
      .metadata = out_metadata,
  };
  iree_status_t status = iree_hal_vulkan_spirv_scan_bda_metadata_strings(
      spirv_words, spirv_word_count, &state);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_spirv_validate_bda_metadata(&state);
  }
  if (iree_status_is_ok(status) && out_metadata->is_present &&
      state.has_binding_requirements) {
    out_metadata->binding_requirement_count = out_metadata->binding_count;
    status = iree_allocator_malloc_array(
        host_allocator, out_metadata->binding_requirement_count,
        sizeof(out_metadata->binding_requirements[0]),
        (void**)&out_metadata->binding_requirements);
    if (iree_status_is_ok(status)) {
      memset(out_metadata->binding_requirements, 0,
             out_metadata->binding_requirement_count *
                 sizeof(out_metadata->binding_requirements[0]));
      state.populate_binding_requirements = true;
      status = iree_hal_vulkan_spirv_scan_bda_metadata_strings(
          spirv_words, spirv_word_count, &state);
    }
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < out_metadata->binding_requirement_count;
           ++i) {
        iree_hal_vulkan_spirv_bda_binding_requirement_t* requirement =
            &out_metadata->binding_requirements[i];
        if (requirement->minimum_alignment == 0) {
          requirement->minimum_alignment = 1;
          requirement->minimum_length = 0;
        }
      }
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_spirv_bda_dispatch_metadata_deinitialize(out_metadata,
                                                             host_allocator);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_spirv_find_push_constant_pointee_type(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    uint32_t pointer_type_id, uint32_t* out_pointee_type_id) {
  *out_pointee_type_id = 0;

  iree_host_size_t word_offset = IREE_HAL_VULKAN_SPIRV_HEADER_WORD_COUNT;
  while (word_offset < spirv_word_count) {
    uint16_t opcode = 0;
    uint16_t word_count = 0;
    const uint32_t* operands = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_next_instruction(
        spirv_words, spirv_word_count, &word_offset, &opcode, &word_count,
        &operands));
    if (opcode != IREE_HAL_VULKAN_SPIRV_OP_TYPE_POINTER) continue;
    if (word_count < 4) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "SPIR-V OpTypePointer instruction is truncated");
    }
    if (operands[0] != pointer_type_id) continue;
    if (operands[1] != IREE_HAL_VULKAN_SPIRV_STORAGE_CLASS_PUSH_CONSTANT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "raw Vulkan BDA root variable type is not a PushConstant pointer");
    }
    *out_pointee_type_id = operands[2];
    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "raw Vulkan BDA root variable pointer type is not declared");
}

static iree_status_t iree_hal_vulkan_spirv_verify_bda_root_struct_layout(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    uint32_t struct_type_id) {
  bool has_struct_type = false;
  bool has_block_decoration = false;
  bool has_member_offsets[IREE_HAL_VULKAN_SPIRV_BDA_ROOT_MEMBER_COUNT] = {0};
  uint32_t member_offsets[IREE_HAL_VULKAN_SPIRV_BDA_ROOT_MEMBER_COUNT] = {0};

  iree_host_size_t word_offset = IREE_HAL_VULKAN_SPIRV_HEADER_WORD_COUNT;
  while (word_offset < spirv_word_count) {
    uint16_t opcode = 0;
    uint16_t word_count = 0;
    const uint32_t* operands = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_next_instruction(
        spirv_words, spirv_word_count, &word_offset, &opcode, &word_count,
        &operands));
    switch (opcode) {
      case IREE_HAL_VULKAN_SPIRV_OP_TYPE_STRUCT:
        if (word_count < 2) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "SPIR-V OpTypeStruct instruction is truncated");
        }
        if (operands[0] != struct_type_id) break;
        has_struct_type = true;
        if (word_count - 2 != IREE_HAL_VULKAN_SPIRV_BDA_ROOT_MEMBER_COUNT) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "raw Vulkan BDA root has %u members but expected %u",
              (uint32_t)(word_count - 2),
              IREE_HAL_VULKAN_SPIRV_BDA_ROOT_MEMBER_COUNT);
        }
        break;
      case IREE_HAL_VULKAN_SPIRV_OP_DECORATE:
        if (word_count < 3) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "SPIR-V OpDecorate instruction is truncated");
        }
        if (operands[0] == struct_type_id &&
            operands[1] == IREE_HAL_VULKAN_SPIRV_DECORATION_BLOCK) {
          has_block_decoration = true;
        }
        break;
      case IREE_HAL_VULKAN_SPIRV_OP_MEMBER_DECORATE: {
        if (word_count < 4) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "SPIR-V OpMemberDecorate instruction is truncated");
        }
        if (operands[0] != struct_type_id ||
            operands[2] != IREE_HAL_VULKAN_SPIRV_DECORATION_OFFSET) {
          break;
        }
        if (word_count < 5) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "SPIR-V Offset member decoration instruction is truncated");
        }
        const uint32_t member_ordinal = operands[1];
        if (member_ordinal >= IREE_HAL_VULKAN_SPIRV_BDA_ROOT_MEMBER_COUNT) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "raw Vulkan BDA root has an unexpected member offset decoration");
        }
        has_member_offsets[member_ordinal] = true;
        member_offsets[member_ordinal] = operands[3];
        break;
      }
      default:
        break;
    }
  }

  if (!has_struct_type) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "raw Vulkan BDA root PushConstant struct type is not declared");
  }
  if (!has_block_decoration) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "raw Vulkan BDA root PushConstant struct must be Block-decorated");
  }

  for (uint32_t i = 0; i < IREE_HAL_VULKAN_SPIRV_BDA_ROOT_MEMBER_COUNT; ++i) {
    if (!has_member_offsets[i]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "raw Vulkan BDA root member %u has no Offset decoration", i);
    }
    if (member_offsets[i] != iree_hal_vulkan_spirv_bda_root_member_offsets[i]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "raw Vulkan BDA root member %u has offset %u but expected %u", i,
          member_offsets[i], iree_hal_vulkan_spirv_bda_root_member_offsets[i]);
    }
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_vulkan_spirv_verify_bda_root_push_constant_layout_from_analysis(
    const iree_hal_vulkan_spirv_module_analysis_t* analysis,
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count) {
  if (analysis->push_constant_variable_count != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "raw Vulkan BDA executable must declare exactly one PushConstant root "
        "variable; found %" PRIhsz,
        analysis->push_constant_variable_count);
  }

  uint32_t root_struct_type_id = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_find_push_constant_pointee_type(
      spirv_words, spirv_word_count,
      analysis->single_push_constant_pointer_type_id, &root_struct_type_id));

  return iree_hal_vulkan_spirv_verify_bda_root_struct_layout(
      spirv_words, spirv_word_count, root_struct_type_id);
}

iree_status_t iree_hal_vulkan_spirv_verify_bda_root_push_constant_layout(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count) {
  iree_hal_vulkan_spirv_module_analysis_t analysis;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_analyze_module(
      spirv_words, spirv_word_count, &analysis));
  return iree_hal_vulkan_spirv_verify_bda_root_push_constant_layout_from_analysis(
      &analysis, spirv_words, spirv_word_count);
}

iree_status_t iree_hal_vulkan_spirv_verify_bda_module_analysis(
    const iree_hal_vulkan_spirv_module_analysis_t* analysis,
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_hal_vulkan_spirv_bda_verification_flags_t verification_flags) {
  IREE_ASSERT_ARGUMENT(analysis);
  const iree_hal_vulkan_spirv_bda_verification_flags_t known_flags =
      IREE_HAL_VULKAN_SPIRV_BDA_VERIFICATION_FLAG_REQUIRE_PUSH_CONSTANT_ROOT;
  if (iree_any_bit_set(verification_flags, ~known_flags)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan BDA SPIR-V verification flags: 0x%" PRIx32,
        verification_flags);
  }

  if (!analysis->has_physical_storage_buffer_addresses_capability) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan BDA executable must declare PhysicalStorageBufferAddresses");
  }

  if (!analysis->uses_physical_storage_buffer64_glsl450) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan BDA executable must use PhysicalStorageBuffer64 GLSL450");
  }
  if (analysis->has_descriptor_binding_decorations) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan BDA executable must not declare descriptor set or binding "
        "decorations");
  }
  if (analysis->has_descriptor_storage_class_variables) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan BDA executable must not declare descriptor-backed variables");
  }

  if (iree_all_bits_set(
          verification_flags,
          IREE_HAL_VULKAN_SPIRV_BDA_VERIFICATION_FLAG_REQUIRE_PUSH_CONSTANT_ROOT)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_spirv_verify_bda_root_push_constant_layout_from_analysis(
            analysis, spirv_words, spirv_word_count));
  }

  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_spirv_verify_bda_module(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_hal_vulkan_spirv_bda_verification_flags_t verification_flags) {
  iree_hal_vulkan_spirv_module_analysis_t analysis;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_analyze_module(
      spirv_words, spirv_word_count, &analysis));
  return iree_hal_vulkan_spirv_verify_bda_module_analysis(
      &analysis, spirv_words, spirv_word_count, verification_flags);
}

iree_status_t iree_hal_vulkan_spirv_parse_compute_entry_points(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_host_size_t entry_point_capacity,
    iree_hal_vulkan_spirv_compute_entry_point_t* out_entry_points) {
  IREE_ASSERT_ARGUMENT(out_entry_points);
  iree_hal_vulkan_spirv_scan_state_t state = {
      .entry_points = out_entry_points,
      .entry_point_capacity = entry_point_capacity,
  };
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_spirv_scan_module(spirv_words, spirv_word_count, &state));
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_spirv_parse_compute_workgroup_size(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_string_view_t entry_point, bool* out_entry_point_found,
    uint32_t out_workgroup_size[3]) {
  IREE_ASSERT_ARGUMENT(out_workgroup_size);
  if (out_entry_point_found) *out_entry_point_found = false;
  memset(out_workgroup_size, 0, sizeof(uint32_t) * 3);
  iree_hal_vulkan_spirv_scan_state_t state = {
      .lookup_entry_point_enabled = true,
      .lookup_entry_point = entry_point,
      .lookup_workgroup_size = out_workgroup_size,
  };
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_spirv_scan_module(spirv_words, spirv_word_count, &state));
  if (out_entry_point_found) {
    *out_entry_point_found = state.lookup_entry_point_found;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_spirv_verify_bda_entry_point(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_string_view_t entry_point,
    iree_hal_vulkan_spirv_bda_verification_flags_t verification_flags,
    uint32_t out_workgroup_size[3]) {
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_verify_bda_module(
      spirv_words, spirv_word_count, verification_flags));

  bool entry_point_found = false;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_parse_compute_workgroup_size(
      spirv_words, spirv_word_count, entry_point, &entry_point_found,
      out_workgroup_size));
  if (!entry_point_found) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan BDA executable has no compute entry point '%.*s'",
        (int)entry_point.size, entry_point.data);
  }
  return iree_ok_status();
}
