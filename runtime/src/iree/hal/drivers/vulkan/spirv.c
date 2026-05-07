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

static iree_status_t iree_hal_vulkan_spirv_entry_point_name_matches(
    const uint32_t* operands, uint16_t operand_word_count,
    iree_string_view_t entry_point, bool* out_matches) {
  *out_matches = false;
  iree_string_view_t name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_entry_point_name(
      operands, operand_word_count, &name));
  *out_matches = iree_string_view_equal(name, entry_point);
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

iree_status_t iree_hal_vulkan_spirv_analyze_module(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_hal_vulkan_spirv_module_analysis_t* out_analysis) {
  IREE_ASSERT_ARGUMENT(out_analysis);
  memset(out_analysis, 0, sizeof(*out_analysis));

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
    switch (opcode) {
      case IREE_HAL_VULKAN_SPIRV_OP_MEMORY_MODEL:
        if (word_count < 3) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "SPIR-V OpMemoryModel instruction is truncated");
        }
        out_analysis->uses_physical_storage_buffer64_glsl450 =
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
          out_analysis->has_physical_storage_buffer_addresses_capability = true;
        }
        break;
      case IREE_HAL_VULKAN_SPIRV_OP_ENTRY_POINT: {
        if (word_count < 4) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "SPIR-V OpEntryPoint instruction is truncated");
        }
        if (operands[0] != IREE_HAL_VULKAN_SPIRV_EXECUTION_MODEL_GL_COMPUTE) {
          break;
        }
        iree_string_view_t name = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_entry_point_name(
            operands, (uint16_t)(word_count - 1), &name));
        iree_host_size_t name_storage_size = 0;
        if (!iree_host_size_checked_add(name.size, /*NUL=*/1,
                                        &name_storage_size) ||
            !iree_host_size_checked_add(
                out_analysis->compute_entry_point_name_storage_size,
                name_storage_size,
                &out_analysis->compute_entry_point_name_storage_size)) {
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "SPIR-V compute entry point name storage "
                                  "overflows");
        }
        ++out_analysis->compute_entry_point_count;
        break;
      }
      case IREE_HAL_VULKAN_SPIRV_OP_DECORATE:
        if (word_count < 3) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "SPIR-V OpDecorate instruction is truncated");
        }
        if (operands[1] == IREE_HAL_VULKAN_SPIRV_DECORATION_BINDING ||
            operands[1] == IREE_HAL_VULKAN_SPIRV_DECORATION_DESCRIPTOR_SET) {
          out_analysis->has_descriptor_binding_decorations = true;
        }
        break;
      case IREE_HAL_VULKAN_SPIRV_OP_VARIABLE:
        if (word_count < 4) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "SPIR-V OpVariable instruction is truncated");
        }
        if (operands[2] == IREE_HAL_VULKAN_SPIRV_STORAGE_CLASS_PUSH_CONSTANT) {
          ++out_analysis->push_constant_variable_count;
        }
        if (iree_hal_vulkan_spirv_storage_class_is_descriptor_backed(
                operands[2])) {
          out_analysis->has_descriptor_storage_class_variables = true;
        }
        break;
      default:
        break;
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_spirv_find_single_push_constant_variable(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    uint32_t* out_type_id) {
  *out_type_id = 0;

  iree_host_size_t push_constant_variable_count = 0;
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
    if (operands[2] != IREE_HAL_VULKAN_SPIRV_STORAGE_CLASS_PUSH_CONSTANT) {
      continue;
    }
    *out_type_id = operands[0];
    ++push_constant_variable_count;
  }

  if (push_constant_variable_count != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "raw Vulkan BDA executable must declare exactly one PushConstant root "
        "variable; found %" PRIhsz,
        push_constant_variable_count);
  }
  return iree_ok_status();
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

static iree_status_t iree_hal_vulkan_spirv_find_struct_member_count(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    uint32_t struct_type_id, uint32_t* out_member_count) {
  *out_member_count = 0;

  iree_host_size_t word_offset = IREE_HAL_VULKAN_SPIRV_HEADER_WORD_COUNT;
  while (word_offset < spirv_word_count) {
    uint16_t opcode = 0;
    uint16_t word_count = 0;
    const uint32_t* operands = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_next_instruction(
        spirv_words, spirv_word_count, &word_offset, &opcode, &word_count,
        &operands));
    if (opcode != IREE_HAL_VULKAN_SPIRV_OP_TYPE_STRUCT) continue;
    if (word_count < 2) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "SPIR-V OpTypeStruct instruction is truncated");
    }
    if (operands[0] != struct_type_id) continue;
    *out_member_count = word_count - 2;
    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "raw Vulkan BDA root PushConstant struct type is not declared");
}

static iree_status_t iree_hal_vulkan_spirv_struct_has_block_decoration(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    uint32_t struct_type_id, bool* out_has_block_decoration) {
  *out_has_block_decoration = false;

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
    if (operands[0] == struct_type_id &&
        operands[1] == IREE_HAL_VULKAN_SPIRV_DECORATION_BLOCK) {
      *out_has_block_decoration = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_spirv_verify_bda_root_member_offsets(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    uint32_t struct_type_id) {
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
    if (opcode != IREE_HAL_VULKAN_SPIRV_OP_MEMBER_DECORATE) continue;
    if (word_count < 4) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "SPIR-V OpMemberDecorate instruction is truncated");
    }
    if (operands[0] != struct_type_id ||
        operands[2] != IREE_HAL_VULKAN_SPIRV_DECORATION_OFFSET) {
      continue;
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

iree_status_t iree_hal_vulkan_spirv_verify_bda_root_push_constant_layout(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count) {
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_spirv_verify_module(spirv_words, spirv_word_count));

  uint32_t root_pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_find_single_push_constant_variable(
      spirv_words, spirv_word_count, &root_pointer_type_id));

  uint32_t root_struct_type_id = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_find_push_constant_pointee_type(
      spirv_words, spirv_word_count, root_pointer_type_id,
      &root_struct_type_id));

  uint32_t member_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_find_struct_member_count(
      spirv_words, spirv_word_count, root_struct_type_id, &member_count));
  if (member_count != IREE_HAL_VULKAN_SPIRV_BDA_ROOT_MEMBER_COUNT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "raw Vulkan BDA root has %u members but expected %u", member_count,
        IREE_HAL_VULKAN_SPIRV_BDA_ROOT_MEMBER_COUNT);
  }

  bool has_block_decoration = false;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_struct_has_block_decoration(
      spirv_words, spirv_word_count, root_struct_type_id,
      &has_block_decoration));
  if (!has_block_decoration) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "raw Vulkan BDA root PushConstant struct must be Block-decorated");
  }

  return iree_hal_vulkan_spirv_verify_bda_root_member_offsets(
      spirv_words, spirv_word_count, root_struct_type_id);
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
        iree_hal_vulkan_spirv_verify_bda_root_push_constant_layout(
            spirv_words, spirv_word_count));
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

static iree_status_t iree_hal_vulkan_spirv_parse_compute_workgroup_sizes(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_host_size_t entry_point_count,
    iree_hal_vulkan_spirv_compute_entry_point_t* entry_points) {
  for (iree_host_size_t i = 0; i < entry_point_count; ++i) {
    memset(entry_points[i].workgroup_size, 0,
           sizeof(entry_points[i].workgroup_size));
  }

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
    if (operands[1] != IREE_HAL_VULKAN_SPIRV_EXECUTION_MODE_LOCAL_SIZE) {
      continue;
    }
    iree_hal_vulkan_spirv_compute_entry_point_t* entry_point =
        iree_hal_vulkan_spirv_find_compute_entry_point_by_id(
            entry_point_count, entry_points, operands[0]);
    if (!entry_point) continue;
    if (word_count < 6) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "SPIR-V LocalSize execution mode for '%.*s' is truncated",
          (int)entry_point->name.size, entry_point->name.data);
    }
    entry_point->workgroup_size[0] = operands[2];
    entry_point->workgroup_size[1] = operands[3];
    entry_point->workgroup_size[2] = operands[4];
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
    if (opcode != IREE_HAL_VULKAN_SPIRV_OP_ENTRY_POINT) continue;
    if (word_count < 4) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "SPIR-V OpEntryPoint instruction is truncated");
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
    ++entry_point_count;
  }
  return iree_hal_vulkan_spirv_parse_compute_workgroup_sizes(
      spirv_words, spirv_word_count, entry_point_count, out_entry_points);
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
    if (opcode != IREE_HAL_VULKAN_SPIRV_OP_ENTRY_POINT) continue;
    if (word_count < 4) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "SPIR-V OpEntryPoint instruction is truncated");
    }
    if (operands[0] != IREE_HAL_VULKAN_SPIRV_EXECUTION_MODEL_GL_COMPUTE) {
      continue;
    }
    bool matches = false;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_spirv_entry_point_name_matches(
        operands, (uint16_t)(word_count - 1), entry_point, &matches));
    if (matches) {
      if (entry_point_id != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "SPIR-V module has duplicate compute entry "
                                "point '%.*s'",
                                (int)entry_point.size, entry_point.data);
      }
      entry_point_id = operands[1];
      if (out_entry_point_found) *out_entry_point_found = true;
    }
  }
  if (entry_point_id == 0) return iree_ok_status();

  return iree_hal_vulkan_spirv_parse_compute_workgroup_size_by_id(
      spirv_words, spirv_word_count, entry_point_id, entry_point,
      out_workgroup_size);
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
