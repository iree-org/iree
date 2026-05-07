// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/pm4_dispatch.h"

#include <inttypes.h>
#include <string.h>

static uint32_t iree_hal_amdgpu_pm4_compute_num_thread(uint16_t value) {
  return (uint32_t)value;
}

static uint32_t iree_hal_amdgpu_pm4_kernel_descriptor_user_sgpr_count(
    const iree_hal_amdgpu_kernel_descriptor_t* descriptor) {
  return (descriptor->compute_pgm_rsrc2 &
          IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_MASK) >>
         IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_SHIFT;
}

static bool iree_hal_amdgpu_pm4_kernel_descriptor_uses_wave32(
    const iree_hal_amdgpu_kernel_descriptor_t* descriptor) {
  return iree_any_bit_set(
      descriptor->kernel_code_properties,
      IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);
}

static iree_status_t iree_hal_amdgpu_pm4_entry_address(
    uint64_t kernel_object, int64_t byte_offset, uint64_t* out_entry_address) {
  if (byte_offset >= 0) {
    const uint64_t unsigned_offset = (uint64_t)byte_offset;
    if (IREE_UNLIKELY(UINT64_MAX - kernel_object < unsigned_offset)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "kernel entry address overflows uint64_t");
    }
    *out_entry_address = kernel_object + unsigned_offset;
    return iree_ok_status();
  }
  const uint64_t unsigned_offset = (uint64_t)(-byte_offset);
  if (IREE_UNLIKELY(kernel_object < unsigned_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "kernel entry address underflows uint64_t");
  }
  *out_entry_address = kernel_object - unsigned_offset;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_dispatch_validate_descriptor_gfx10(
    const iree_hal_amdgpu_kernel_descriptor_t* descriptor) {
  if (IREE_UNLIKELY(descriptor->private_segment_fixed_size != 0)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 dispatch scratch setup is required for private segment size "
        "%" PRIu32,
        descriptor->private_segment_fixed_size);
  }
  if (IREE_UNLIKELY(iree_any_bit_set(
          descriptor->compute_pgm_rsrc2,
          IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 dispatch scratch setup is required by COMPUTE_PGM_RSRC2");
  }
  if (IREE_UNLIKELY(descriptor->kernarg_preload != 0)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 dispatch kernarg preload setup is unsupported");
  }

  const uint32_t allowed_properties =
      IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR |
      IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
  if (IREE_UNLIKELY(iree_any_bit_set(descriptor->kernel_code_properties,
                                     (uint16_t)~allowed_properties))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 dispatch supports only kernarg-segment pointer and wave32 kernel "
        "code properties, but descriptor has 0x%04" PRIx16,
        descriptor->kernel_code_properties);
  }

  const bool has_kernarg_pointer = iree_any_bit_set(
      descriptor->kernel_code_properties,
      IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR);
  const uint32_t user_data_dword_count =
      iree_hal_amdgpu_pm4_kernel_descriptor_user_sgpr_count(descriptor);
  if (IREE_UNLIKELY(user_data_dword_count >
                    IREE_HAL_AMDGPU_PM4_DISPATCH_USER_DATA_DWORD_CAPACITY)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 dispatch supports at most %u user data dwords, but descriptor "
        "requires %" PRIu32,
        IREE_HAL_AMDGPU_PM4_DISPATCH_USER_DATA_DWORD_CAPACITY,
        user_data_dword_count);
  }
  if (has_kernarg_pointer) {
    if (IREE_UNLIKELY(user_data_dword_count < 2)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "kernarg pointer dispatch expects at least two user data dwords, but "
          "descriptor requires %" PRIu32,
          user_data_dword_count);
    }
  } else if (IREE_UNLIKELY(user_data_dword_count != 0)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 dispatch non-kernarg user data dwords are unsupported");
  }
  return iree_ok_status();
}

static bool iree_hal_amdgpu_pm4_dispatch_entry_address_is_supported(
    uint64_t kernel_object, int64_t byte_offset) {
  if (byte_offset >= 0) {
    return UINT64_MAX - kernel_object >= (uint64_t)byte_offset;
  }
  return kernel_object >= (uint64_t)(-byte_offset);
}

static bool iree_hal_amdgpu_pm4_dispatch_descriptor_is_supported_gfx10(
    const iree_hal_amdgpu_kernel_descriptor_t* descriptor) {
  if (descriptor->private_segment_fixed_size != 0) return false;
  if (iree_any_bit_set(
          descriptor->compute_pgm_rsrc2,
          IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT)) {
    return false;
  }
  if (descriptor->kernarg_preload != 0) return false;

  const uint32_t allowed_properties =
      IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR |
      IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
  if (iree_any_bit_set(descriptor->kernel_code_properties,
                       (uint16_t)~allowed_properties)) {
    return false;
  }

  const uint32_t user_data_dword_count =
      iree_hal_amdgpu_pm4_kernel_descriptor_user_sgpr_count(descriptor);
  if (user_data_dword_count >
      IREE_HAL_AMDGPU_PM4_DISPATCH_USER_DATA_DWORD_CAPACITY) {
    return false;
  }
  const bool has_kernarg_pointer = iree_any_bit_set(
      descriptor->kernel_code_properties,
      IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR);
  if (has_kernarg_pointer) return user_data_dword_count >= 2;
  return user_data_dword_count == 0;
}

bool iree_hal_amdgpu_pm4_dispatch_launch_state_is_supported_gfx10(
    const iree_hal_amdgpu_kernel_descriptor_t* descriptor,
    uint64_t kernel_object, const uint16_t workgroup_size[3],
    iree_hal_amdgpu_pm4_dispatch_launch_flags_t flags) {
  if (!descriptor || !workgroup_size || kernel_object == 0) return false;
  if ((flags & ~IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE) != 0) {
    return false;
  }
  for (iree_host_size_t i = 0; i < 3; ++i) {
    if (workgroup_size[i] == 0) return false;
  }
  return iree_hal_amdgpu_pm4_dispatch_descriptor_is_supported_gfx10(
             descriptor) &&
         iree_hal_amdgpu_pm4_dispatch_entry_address_is_supported(
             kernel_object, descriptor->kernel_code_entry_byte_offset);
}

static uint32_t* iree_hal_amdgpu_pm4_dispatch_emit_set_sh_reg_sequence(
    uint32_t* dwords, uint32_t register_address, const uint32_t* values,
    uint32_t value_count) {
  dwords[0] = iree_hal_amdgpu_pm4_make_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_SET_SH_REG, 2 + value_count);
  dwords[1] = register_address - IREE_HAL_AMDGPU_PM4_PERSISTENT_SPACE_START;
  memcpy(&dwords[2], values, sizeof(values[0]) * value_count);
  return dwords + 2 + value_count;
}

iree_status_t iree_hal_amdgpu_pm4_dispatch_launch_state_initialize_gfx10(
    const iree_hal_amdgpu_kernel_descriptor_t* descriptor,
    uint64_t kernel_object, const uint16_t workgroup_size[3],
    iree_hal_amdgpu_pm4_dispatch_launch_flags_t flags,
    iree_hal_amdgpu_pm4_dispatch_launch_state_t* out_state) {
  IREE_ASSERT_ARGUMENT(descriptor);
  IREE_ASSERT_ARGUMENT(workgroup_size);
  IREE_ASSERT_ARGUMENT(out_state);
  memset(out_state, 0, sizeof(*out_state));

  if (IREE_UNLIKELY(kernel_object == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel object address is required");
  }
  if (IREE_UNLIKELY(
          (flags & ~IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE) !=
          0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported PM4 dispatch launch flags 0x%08" PRIx32, flags);
  }
  for (iree_host_size_t i = 0; i < 3; ++i) {
    if (IREE_UNLIKELY(workgroup_size[i] == 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "workgroup size dimension %" PRIhsz " must be non-zero", i);
    }
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pm4_dispatch_validate_descriptor_gfx10(descriptor));

  uint64_t entry_address = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_entry_address(
      kernel_object, descriptor->kernel_code_entry_byte_offset,
      &entry_address));

  const uint64_t program_address = entry_address >> 8;
  out_state->program[0] = (uint32_t)program_address;
  out_state->program[1] = (uint32_t)((program_address >> 32) & 0xFFu);
  out_state->program[2] = 0;
  out_state->program[3] = 0;
  out_state->program[4] = 0;
  out_state->program[5] = 0;

  out_state->resources[0] = descriptor->compute_pgm_rsrc1;
  out_state->resources[1] = descriptor->compute_pgm_rsrc2;
  out_state->resource3 = descriptor->compute_pgm_rsrc3;
  out_state->temporary_ring_size = 0;
  out_state->restart[0] = 0;
  out_state->restart[1] = 0;
  out_state->restart[2] = 0;
  out_state->resource_limits = 0;
  out_state->start_and_threads[0] = 0;
  out_state->start_and_threads[1] = 0;
  out_state->start_and_threads[2] = 0;
  out_state->start_and_threads[3] =
      iree_hal_amdgpu_pm4_compute_num_thread(workgroup_size[0]);
  out_state->start_and_threads[4] =
      iree_hal_amdgpu_pm4_compute_num_thread(workgroup_size[1]);
  out_state->start_and_threads[5] =
      iree_hal_amdgpu_pm4_compute_num_thread(workgroup_size[2]);
  out_state->start_and_threads[6] = 0;
  out_state->start_and_threads[7] = 0;
  out_state->user_data_dword_count =
      iree_hal_amdgpu_pm4_kernel_descriptor_user_sgpr_count(descriptor);

  iree_hal_amdgpu_pm4_dispatch_initiator_flags_t initiator_flags =
      IREE_HAL_AMDGPU_PM4_DISPATCH_INITIATOR_FORCE_START_AT_000;
  if (iree_any_bit_set(flags,
                       IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE)) {
    initiator_flags |= IREE_HAL_AMDGPU_PM4_DISPATCH_INITIATOR_ORDER_MODE;
  }
  if (iree_hal_amdgpu_pm4_kernel_descriptor_uses_wave32(descriptor)) {
    initiator_flags |= IREE_HAL_AMDGPU_PM4_DISPATCH_INITIATOR_CS_W32_EN;
  }
  out_state->dispatch_initiator =
      iree_hal_amdgpu_pm4_dispatch_initiator(initiator_flags);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_pm4_dispatch_emit_setup(
    const iree_hal_amdgpu_pm4_dispatch_launch_state_t* state, uint32_t capacity,
    uint32_t* target_dwords, uint32_t* out_dword_count) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(target_dwords);
  IREE_ASSERT_ARGUMENT(out_dword_count);
  *out_dword_count = 0;
  if (IREE_UNLIKELY(capacity <
                    IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "PM4 dispatch setup requires %u dwords but only %u are available",
        IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT, capacity);
  }

  uint32_t* cursor = target_dwords;
  cursor = iree_hal_amdgpu_pm4_dispatch_emit_set_sh_reg_sequence(
      cursor, IREE_HAL_AMDGPU_PM4_COMPUTE_PGM_LO_REGISTER, state->program, 6);
  cursor = iree_hal_amdgpu_pm4_dispatch_emit_set_sh_reg_sequence(
      cursor, IREE_HAL_AMDGPU_PM4_COMPUTE_PGM_RSRC1_REGISTER, state->resources,
      2);
  cursor = iree_hal_amdgpu_pm4_dispatch_emit_set_sh_reg_sequence(
      cursor, IREE_HAL_AMDGPU_PM4_COMPUTE_PGM_RSRC3_REGISTER, &state->resource3,
      1);
  cursor = iree_hal_amdgpu_pm4_dispatch_emit_set_sh_reg_sequence(
      cursor, IREE_HAL_AMDGPU_PM4_COMPUTE_TMPRING_SIZE_REGISTER,
      &state->temporary_ring_size, 1);
  cursor = iree_hal_amdgpu_pm4_dispatch_emit_set_sh_reg_sequence(
      cursor, IREE_HAL_AMDGPU_PM4_COMPUTE_RESTART_X_REGISTER, state->restart,
      3);
  cursor = iree_hal_amdgpu_pm4_dispatch_emit_set_sh_reg_sequence(
      cursor, IREE_HAL_AMDGPU_PM4_COMPUTE_RESOURCE_LIMITS_REGISTER,
      &state->resource_limits, 1);
  cursor = iree_hal_amdgpu_pm4_dispatch_emit_set_sh_reg_sequence(
      cursor, IREE_HAL_AMDGPU_PM4_COMPUTE_START_X_REGISTER,
      state->start_and_threads, 8);
  *out_dword_count = (uint32_t)(cursor - target_dwords);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_pm4_dispatch_emit_user_data(
    const iree_hal_amdgpu_pm4_dispatch_launch_state_t* state,
    uint64_t kernarg_address, uint32_t capacity, uint32_t* target_dwords,
    uint32_t* out_dword_count) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(target_dwords);
  IREE_ASSERT_ARGUMENT(out_dword_count);
  *out_dword_count = 0;
  if (state->user_data_dword_count == 0) return iree_ok_status();

  const uint32_t required_dword_count = 2 + state->user_data_dword_count;
  if (IREE_UNLIKELY(capacity < required_dword_count)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "PM4 dispatch userdata requires %u dwords but only %u are available",
        required_dword_count, capacity);
  }
  uint32_t user_data[IREE_HAL_AMDGPU_PM4_DISPATCH_USER_DATA_DWORD_CAPACITY] = {
      0};
  user_data[0] = (uint32_t)kernarg_address;
  user_data[1] = (uint32_t)(kernarg_address >> 32);
  iree_hal_amdgpu_pm4_dispatch_emit_set_sh_reg_sequence(
      target_dwords, IREE_HAL_AMDGPU_PM4_COMPUTE_USER_DATA_0_REGISTER,
      user_data, state->user_data_dword_count);
  *out_dword_count = required_dword_count;
  return iree_ok_status();
}
