// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_TARGET_ID_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_TARGET_ID_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// AMDGPU Target IDs
//===----------------------------------------------------------------------===//

// Parsed gfx IP version.
typedef struct iree_hal_amdgpu_gfxip_version_t {
  // Major gfx ISA version, such as 9, 10, 11, or 12.
  uint32_t major;
  // Minor gfx ISA version within |major|.
  uint32_t minor;
  // Stepping digit within |major|.|minor|.
  uint32_t stepping;
} iree_hal_amdgpu_gfxip_version_t;

// Target feature selector state from AMDGPU target IDs.
typedef enum iree_hal_amdgpu_target_feature_state_e {
  // Feature is not represented by the parsed target ID.
  IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY = 0,
  // Feature is known not to be supported by the target.
  IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED,
  // Feature is explicitly disabled, such as `:xnack-`.
  IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF,
  // Feature is explicitly enabled, such as `:sramecc+`.
  IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON,
} iree_hal_amdgpu_target_feature_state_t;

// Target processor name class.
typedef enum iree_hal_amdgpu_target_kind_e {
  // Exact target processor such as `gfx942` or `gfx1100`.
  IREE_HAL_AMDGPU_TARGET_KIND_EXACT = 0,
  // Generic target processor such as `gfx9-4-generic` or `gfx11-generic`.
  IREE_HAL_AMDGPU_TARGET_KIND_GENERIC,
} iree_hal_amdgpu_target_kind_t;

// Parsed AMDGPU target ID.
typedef struct iree_hal_amdgpu_target_id_t {
  // Processor name class used to interpret |version|.
  iree_hal_amdgpu_target_kind_t kind;
  // Parsed gfx IP version or generic family version.
  iree_hal_amdgpu_gfxip_version_t version;
  // Generic code-object format version from ELF e_flags, or 0 if unspecified.
  uint32_t generic_version;
  // SRAM ECC selector state.
  iree_hal_amdgpu_target_feature_state_t sramecc;
  // XNACK selector state.
  iree_hal_amdgpu_target_feature_state_t xnack;
  // Borrowed processor name without feature suffixes or HSA triple prefix.
  iree_string_view_t processor;
} iree_hal_amdgpu_target_id_t;

// Parser modes for AMDGPU target IDs.
typedef enum iree_hal_amdgpu_target_id_parse_flag_bits_e {
  // Requires a bare processor name with no feature suffixes.
  IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_NONE = 0u,
  // Accepts `amdgcn-amd-amdhsa--`-prefixed HSA ISA names.
  IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_HSA_PREFIX = 1u << 0,
  // Accepts bare processor names such as `gfx942`.
  IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_ARCH_ONLY = 1u << 1,
  // Accepts AMDGPU feature suffixes such as `:sramecc+:xnack-`.
  IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_FEATURE_SUFFIXES = 1u << 2,
} iree_hal_amdgpu_target_id_parse_flag_bits_t;
typedef uint32_t iree_hal_amdgpu_target_id_parse_flags_t;

// Compatibility reasons reported by iree_hal_amdgpu_target_id_check_compatible.
typedef enum iree_hal_amdgpu_target_compatibility_bits_e {
  // Code object target identity is compatible with the agent target identity.
  IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_COMPATIBLE = 0u,
  // Exact processor versions do not match.
  IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_PROCESSOR = 1u << 0,
  // Generic processor family does not match the agent's mapped family.
  IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_GENERIC_FAMILY = 1u << 1,
  // Generic code-object version is older than the agent's supported floor.
  IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_GENERIC_VERSION = 1u << 2,
  // Explicit SRAM ECC mode does not match the agent.
  IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_SRAMECC = 1u << 3,
  // Explicit XNACK mode does not match the agent.
  IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_XNACK = 1u << 4,
} iree_hal_amdgpu_target_compatibility_bits_t;
typedef uint32_t iree_hal_amdgpu_target_compatibility_t;

// Parses an AMDGPU target ID into |out_target_id|.
//
// String views in |out_target_id| borrow from |value| and are only valid while
// |value| storage remains live.
iree_status_t iree_hal_amdgpu_target_id_parse(
    iree_string_view_t value, iree_hal_amdgpu_target_id_parse_flags_t flags,
    iree_hal_amdgpu_target_id_t* out_target_id);

// Parses an HSA ISA name reported by HSA_ISA_INFO_NAME.
iree_status_t iree_hal_amdgpu_target_id_parse_hsa_isa_name(
    iree_string_view_t value, iree_hal_amdgpu_target_id_t* out_target_id);

// Formats |target_id| into canonical AMDGPU target-ID syntax.
//
// If |buffer_capacity| is insufficient, |out_buffer_length| still receives the
// required character length excluding the NUL terminator.
iree_status_t iree_hal_amdgpu_target_id_format(
    const iree_hal_amdgpu_target_id_t* target_id,
    iree_host_size_t buffer_capacity, char* buffer,
    iree_host_size_t* out_buffer_length);

// Maps an exact target ID to the processor used for code objects and device
// libraries. If no generic-compatible mapping is known, the exact target is
// returned unchanged.
iree_status_t iree_hal_amdgpu_target_id_lookup_code_object_target(
    const iree_hal_amdgpu_target_id_t* exact_target_id,
    iree_hal_amdgpu_target_id_t* out_code_object_target_id);

// Checks whether |code_object_target_id| can execute on |agent_target_id|.
iree_hal_amdgpu_target_compatibility_t
iree_hal_amdgpu_target_id_check_compatible(
    const iree_hal_amdgpu_target_id_t* code_object_target_id,
    const iree_hal_amdgpu_target_id_t* agent_target_id);

// Formats compatibility mismatch bits into a comma-separated diagnostic string.
//
// If |buffer_capacity| is insufficient, |out_buffer_length| still receives the
// required character length excluding the NUL terminator.
iree_status_t iree_hal_amdgpu_target_compatibility_format(
    iree_hal_amdgpu_target_compatibility_t compatibility,
    iree_host_size_t buffer_capacity, char* buffer,
    iree_host_size_t* out_buffer_length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_TARGET_ID_H_
