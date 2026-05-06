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
  IREE_HAL_VULKAN_SPIRV_OP_VARIABLE = 59u,
  IREE_HAL_VULKAN_SPIRV_OP_DECORATE = 71u,
  IREE_HAL_VULKAN_SPIRV_CAPABILITY_PHYSICAL_STORAGE_BUFFER_ADDRESSES = 5347u,
  IREE_HAL_VULKAN_SPIRV_ADDRESSING_MODEL_PHYSICAL_STORAGE_BUFFER64 = 5348u,
  IREE_HAL_VULKAN_SPIRV_MEMORY_MODEL_GLSL450 = 1u,
  IREE_HAL_VULKAN_SPIRV_EXECUTION_MODEL_GL_COMPUTE = 5u,
  IREE_HAL_VULKAN_SPIRV_EXECUTION_MODE_LOCAL_SIZE = 17u,
  IREE_HAL_VULKAN_SPIRV_STORAGE_CLASS_PUSH_CONSTANT = 9u,
  IREE_HAL_VULKAN_SPIRV_DECORATION_BINDING = 33u,
  IREE_HAL_VULKAN_SPIRV_DECORATION_DESCRIPTOR_SET = 34u,
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

static bool iree_hal_vulkan_spirv_entry_point_name_matches(
    const uint32_t* operands, uint16_t operand_word_count,
    iree_string_view_t entry_point) {
  if (operand_word_count < 3) return false;
  const char* name = (const char*)&operands[2];
  const iree_host_size_t name_capacity = (operand_word_count - 2) * 4;
  iree_host_size_t name_length = 0;
  while (name_length < name_capacity && name[name_length] != 0) {
    ++name_length;
  }
  return name_length == entry_point.size &&
         memcmp(name, entry_point.data, name_length) == 0;
}

static iree_status_t iree_hal_vulkan_spirv_entry_point_name(
    const uint32_t* operands, uint16_t operand_word_count,
    iree_string_view_t* out_name) {
  if (operand_word_count < 3) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V entry point instruction is truncated");
  }
  const char* name = (const char*)&operands[2];
  const iree_host_size_t name_capacity = (operand_word_count - 2) * 4;
  iree_host_size_t name_length = 0;
  while (name_length < name_capacity && name[name_length] != 0) {
    ++name_length;
  }
  if (name_length == name_capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V entry point name is not NUL-terminated");
  }
  *out_name = iree_make_string_view(name, name_length);
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_spirv_verify_module(
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

iree_status_t iree_hal_vulkan_spirv_uses_physical_storage_buffer64_glsl450(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    bool* out_uses_memory_model) {
  IREE_ASSERT_ARGUMENT(out_uses_memory_model);
  *out_uses_memory_model = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_spirv_verify_module(spirv_words, spirv_word_count));

  iree_host_size_t word_offset = IREE_HAL_VULKAN_SPIRV_HEADER_WORD_COUNT;
  while (word_offset < spirv_word_count) {
    uint16_t opcode = 0;
    uint16_t word_count = 0;
    const uint32_t* operands = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_next_instruction(
        spirv_words, spirv_word_count, &word_offset, &opcode, &word_count,
        &operands));
    if (opcode != IREE_HAL_VULKAN_SPIRV_OP_MEMORY_MODEL || word_count < 3) {
      continue;
    }
    *out_uses_memory_model =
        operands[0] ==
            IREE_HAL_VULKAN_SPIRV_ADDRESSING_MODEL_PHYSICAL_STORAGE_BUFFER64 &&
        operands[1] == IREE_HAL_VULKAN_SPIRV_MEMORY_MODEL_GLSL450;
    return iree_ok_status();
  }
  return iree_ok_status();
}

iree_status_t
iree_hal_vulkan_spirv_has_physical_storage_buffer_addresses_capability(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    bool* out_has_capability) {
  IREE_ASSERT_ARGUMENT(out_has_capability);
  *out_has_capability = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_spirv_verify_module(spirv_words, spirv_word_count));

  iree_host_size_t word_offset = IREE_HAL_VULKAN_SPIRV_HEADER_WORD_COUNT;
  while (word_offset < spirv_word_count) {
    uint16_t opcode = 0;
    uint16_t word_count = 0;
    const uint32_t* operands = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_next_instruction(
        spirv_words, spirv_word_count, &word_offset, &opcode, &word_count,
        &operands));
    if (opcode != IREE_HAL_VULKAN_SPIRV_OP_CAPABILITY) continue;
    if (word_count != 2) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "SPIR-V OpCapability instruction is malformed");
    }
    if (operands[0] ==
        IREE_HAL_VULKAN_SPIRV_CAPABILITY_PHYSICAL_STORAGE_BUFFER_ADDRESSES) {
      *out_has_capability = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_spirv_has_descriptor_binding_decorations(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    bool* out_has_descriptor_binding_decorations) {
  IREE_ASSERT_ARGUMENT(out_has_descriptor_binding_decorations);
  *out_has_descriptor_binding_decorations = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_spirv_verify_module(spirv_words, spirv_word_count));

  iree_host_size_t word_offset = IREE_HAL_VULKAN_SPIRV_HEADER_WORD_COUNT;
  while (word_offset < spirv_word_count) {
    uint16_t opcode = 0;
    uint16_t word_count = 0;
    const uint32_t* operands = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_next_instruction(
        spirv_words, spirv_word_count, &word_offset, &opcode, &word_count,
        &operands));
    if (opcode != IREE_HAL_VULKAN_SPIRV_OP_DECORATE) continue;
    if (word_count < 3) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "SPIR-V OpDecorate instruction is truncated");
    }
    const uint32_t decoration = operands[1];
    if (decoration == IREE_HAL_VULKAN_SPIRV_DECORATION_BINDING ||
        decoration == IREE_HAL_VULKAN_SPIRV_DECORATION_DESCRIPTOR_SET) {
      *out_has_descriptor_binding_decorations = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_spirv_count_push_constant_variables(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_host_size_t* out_push_constant_variable_count) {
  IREE_ASSERT_ARGUMENT(out_push_constant_variable_count);
  *out_push_constant_variable_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_spirv_verify_module(spirv_words, spirv_word_count));

  iree_host_size_t word_offset = IREE_HAL_VULKAN_SPIRV_HEADER_WORD_COUNT;
  while (word_offset < spirv_word_count) {
    uint16_t opcode = 0;
    uint16_t word_count = 0;
    const uint32_t* operands = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_next_instruction(
        spirv_words, spirv_word_count, &word_offset, &opcode, &word_count,
        &operands));
    if (opcode != IREE_HAL_VULKAN_SPIRV_OP_VARIABLE) continue;
    if (word_count < 4) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "SPIR-V OpVariable instruction is truncated");
    }
    if (operands[2] == IREE_HAL_VULKAN_SPIRV_STORAGE_CLASS_PUSH_CONSTANT) {
      ++*out_push_constant_variable_count;
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_spirv_count_compute_entry_points(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_host_size_t* out_entry_point_count,
    iree_host_size_t* out_name_storage_size) {
  IREE_ASSERT_ARGUMENT(out_entry_point_count);
  IREE_ASSERT_ARGUMENT(out_name_storage_size);
  *out_entry_point_count = 0;
  *out_name_storage_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_spirv_verify_module(spirv_words, spirv_word_count));

  iree_host_size_t word_offset = IREE_HAL_VULKAN_SPIRV_HEADER_WORD_COUNT;
  while (word_offset < spirv_word_count) {
    uint16_t opcode = 0;
    uint16_t word_count = 0;
    const uint32_t* operands = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_next_instruction(
        spirv_words, spirv_word_count, &word_offset, &opcode, &word_count,
        &operands));
    if (opcode != IREE_HAL_VULKAN_SPIRV_OP_ENTRY_POINT || word_count < 4) {
      continue;
    }
    if (operands[0] != IREE_HAL_VULKAN_SPIRV_EXECUTION_MODEL_GL_COMPUTE) {
      continue;
    }

    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_entry_point_name(
        operands, (uint16_t)(word_count - 1), &name));
    iree_host_size_t entry_name_storage_size = 0;
    if (!iree_host_size_checked_add(name.size, /*NUL=*/1,
                                    &entry_name_storage_size) ||
        !iree_host_size_checked_add(*out_name_storage_size,
                                    entry_name_storage_size,
                                    out_name_storage_size)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "SPIR-V compute entry point name storage "
                              "overflows");
    }
    ++*out_entry_point_count;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_spirv_parse_compute_workgroup_size_by_id(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    uint32_t entry_point_id, iree_string_view_t entry_point,
    uint32_t out_workgroup_size[3]) {
  memset(out_workgroup_size, 0, sizeof(uint32_t) * 3);
  iree_host_size_t word_offset = IREE_HAL_VULKAN_SPIRV_HEADER_WORD_COUNT;
  while (word_offset < spirv_word_count) {
    uint16_t opcode = 0;
    uint16_t word_count = 0;
    const uint32_t* operands = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_next_instruction(
        spirv_words, spirv_word_count, &word_offset, &opcode, &word_count,
        &operands));
    if (opcode != IREE_HAL_VULKAN_SPIRV_OP_EXECUTION_MODE || word_count < 3) {
      continue;
    }
    if (operands[0] != entry_point_id ||
        operands[1] != IREE_HAL_VULKAN_SPIRV_EXECUTION_MODE_LOCAL_SIZE) {
      continue;
    }
    if (word_count < 6) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "SPIR-V LocalSize execution mode for '%.*s' is truncated",
          (int)entry_point.size, entry_point.data);
    }
    out_workgroup_size[0] = operands[2];
    out_workgroup_size[1] = operands[3];
    out_workgroup_size[2] = operands[4];
    return iree_ok_status();
  }
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_spirv_parse_compute_entry_points(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_host_size_t entry_point_capacity,
    iree_hal_vulkan_spirv_compute_entry_point_t* out_entry_points) {
  IREE_ASSERT_ARGUMENT(out_entry_points);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_spirv_verify_module(spirv_words, spirv_word_count));

  iree_host_size_t entry_point_count = 0;
  iree_host_size_t word_offset = IREE_HAL_VULKAN_SPIRV_HEADER_WORD_COUNT;
  while (word_offset < spirv_word_count) {
    uint16_t opcode = 0;
    uint16_t word_count = 0;
    const uint32_t* operands = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_next_instruction(
        spirv_words, spirv_word_count, &word_offset, &opcode, &word_count,
        &operands));
    if (opcode != IREE_HAL_VULKAN_SPIRV_OP_ENTRY_POINT || word_count < 4) {
      continue;
    }
    if (operands[0] != IREE_HAL_VULKAN_SPIRV_EXECUTION_MODEL_GL_COMPUTE) {
      continue;
    }
    if (entry_point_count >= entry_point_capacity) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "SPIR-V compute entry point table overflowed");
    }

    iree_hal_vulkan_spirv_compute_entry_point_t* entry_point =
        &out_entry_points[entry_point_count];
    entry_point->id = operands[1];
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_entry_point_name(
        operands, (uint16_t)(word_count - 1), &entry_point->name));
    for (iree_host_size_t i = 0; i < entry_point_count; ++i) {
      if (iree_string_view_equal(out_entry_points[i].name, entry_point->name)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "SPIR-V module has duplicate compute entry "
                                "point '%.*s'",
                                (int)entry_point->name.size,
                                entry_point->name.data);
      }
    }
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_spirv_parse_compute_workgroup_size_by_id(
            spirv_words, spirv_word_count, entry_point->id, entry_point->name,
            entry_point->workgroup_size));
    ++entry_point_count;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_spirv_parse_compute_workgroup_size(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_string_view_t entry_point, bool* out_entry_point_found,
    uint32_t out_workgroup_size[3]) {
  if (out_entry_point_found) *out_entry_point_found = false;
  memset(out_workgroup_size, 0, sizeof(uint32_t) * 3);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_spirv_verify_module(spirv_words, spirv_word_count));

  uint32_t entry_point_id = 0;
  iree_host_size_t word_offset = IREE_HAL_VULKAN_SPIRV_HEADER_WORD_COUNT;
  while (word_offset < spirv_word_count) {
    uint16_t opcode = 0;
    uint16_t word_count = 0;
    const uint32_t* operands = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_next_instruction(
        spirv_words, spirv_word_count, &word_offset, &opcode, &word_count,
        &operands));
    if (opcode != IREE_HAL_VULKAN_SPIRV_OP_ENTRY_POINT || word_count < 4) {
      continue;
    }
    if (operands[0] != IREE_HAL_VULKAN_SPIRV_EXECUTION_MODEL_GL_COMPUTE) {
      continue;
    }
    if (iree_hal_vulkan_spirv_entry_point_name_matches(
            operands, (uint16_t)(word_count - 1), entry_point)) {
      entry_point_id = operands[1];
      if (out_entry_point_found) *out_entry_point_found = true;
      break;
    }
  }
  if (entry_point_id == 0) return iree_ok_status();

  return iree_hal_vulkan_spirv_parse_compute_workgroup_size_by_id(
      spirv_words, spirv_word_count, entry_point_id, entry_point,
      out_workgroup_size);
}
