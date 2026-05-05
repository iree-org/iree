// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/code_object_target.h"

#include <string.h>

#include "iree/base/alignment.h"
#include "iree/hal/utils/elf_format.h"

//===----------------------------------------------------------------------===//
// AMDGPU ELF Constants
//===----------------------------------------------------------------------===//

typedef enum iree_hal_amdgpu_elf_ident_e {
  IREE_HAL_AMDGPU_ELF_EI_CLASS = 4,
  IREE_HAL_AMDGPU_ELF_EI_DATA = 5,
  IREE_HAL_AMDGPU_ELF_EI_VERSION = 6,
  IREE_HAL_AMDGPU_ELF_EI_OSABI = 7,
  IREE_HAL_AMDGPU_ELF_EI_ABIVERSION = 8,
} iree_hal_amdgpu_elf_ident_t;

typedef enum iree_hal_amdgpu_elf_class_e {
  IREE_HAL_AMDGPU_ELF_CLASS_64 = 2,
} iree_hal_amdgpu_elf_class_t;

typedef enum iree_hal_amdgpu_elf_data_e {
  IREE_HAL_AMDGPU_ELF_DATA_2LSB = 1,
} iree_hal_amdgpu_elf_data_t;

typedef enum iree_hal_amdgpu_elf_version_e {
  IREE_HAL_AMDGPU_ELF_VERSION_CURRENT = 1,
} iree_hal_amdgpu_elf_version_t;

typedef enum iree_hal_amdgpu_elf_machine_e {
  IREE_HAL_AMDGPU_ELF_MACHINE_AMDGPU = 224,
} iree_hal_amdgpu_elf_machine_t;

typedef enum iree_hal_amdgpu_elf_osabi_e {
  IREE_HAL_AMDGPU_ELF_OSABI_HSA = 64,
} iree_hal_amdgpu_elf_osabi_t;

typedef enum iree_hal_amdgpu_elf_hsa_abi_version_e {
  IREE_HAL_AMDGPU_ELF_HSA_ABI_VERSION_V3 = 1,
  IREE_HAL_AMDGPU_ELF_HSA_ABI_VERSION_V4 = 2,
  IREE_HAL_AMDGPU_ELF_HSA_ABI_VERSION_V5 = 3,
  IREE_HAL_AMDGPU_ELF_HSA_ABI_VERSION_V6 = 4,
} iree_hal_amdgpu_elf_hsa_abi_version_t;

typedef enum iree_hal_amdgpu_elf_header_offset_e {
  IREE_HAL_AMDGPU_ELF_HEADER_E_MACHINE_OFFSET = 18,
  IREE_HAL_AMDGPU_ELF_HEADER_E_VERSION_OFFSET = 20,
  IREE_HAL_AMDGPU_ELF64_HEADER_E_FLAGS_OFFSET = 48,
  IREE_HAL_AMDGPU_ELF64_HEADER_SIZE = 64,
} iree_hal_amdgpu_elf_header_offset_t;

enum {
  IREE_HAL_AMDGPU_EF_MACH = 0x0ffu,
  IREE_HAL_AMDGPU_EF_FEATURE_XNACK_V3 = 0x100u,
  IREE_HAL_AMDGPU_EF_FEATURE_SRAMECC_V3 = 0x200u,
  IREE_HAL_AMDGPU_EF_FEATURE_XNACK_V4 = 0x300u,
  IREE_HAL_AMDGPU_EF_FEATURE_XNACK_UNSUPPORTED_V4 = 0x000u,
  IREE_HAL_AMDGPU_EF_FEATURE_XNACK_ANY_V4 = 0x100u,
  IREE_HAL_AMDGPU_EF_FEATURE_XNACK_OFF_V4 = 0x200u,
  IREE_HAL_AMDGPU_EF_FEATURE_XNACK_ON_V4 = 0x300u,
  IREE_HAL_AMDGPU_EF_FEATURE_SRAMECC_V4 = 0xc00u,
  IREE_HAL_AMDGPU_EF_FEATURE_SRAMECC_UNSUPPORTED_V4 = 0x000u,
  IREE_HAL_AMDGPU_EF_FEATURE_SRAMECC_ANY_V4 = 0x400u,
  IREE_HAL_AMDGPU_EF_FEATURE_SRAMECC_OFF_V4 = 0x800u,
  IREE_HAL_AMDGPU_EF_FEATURE_SRAMECC_ON_V4 = 0xc00u,
  IREE_HAL_AMDGPU_EF_GENERIC_VERSION = 0xff000000u,
  IREE_HAL_AMDGPU_EF_GENERIC_VERSION_OFFSET = 24,
};

typedef struct iree_hal_amdgpu_elf_machine_target_t {
  // AMDGPU EF_AMDGPU_MACH_* value.
  uint32_t machine;
  // Processor string represented by |machine|.
  iree_string_view_t processor;
  // True if old V3 e_flags can explicitly encode SRAM ECC off for this target.
  bool sramecc_supported;
  // True if old V3 e_flags can explicitly encode XNACK off for this target.
  bool xnack_supported;
} iree_hal_amdgpu_elf_machine_target_t;

static const iree_hal_amdgpu_elf_machine_target_t
    iree_hal_amdgpu_elf_machine_targets[] = {
        {0x020, IREE_SVL("gfx600"), false, false},
        {0x021, IREE_SVL("gfx601"), false, false},
        {0x022, IREE_SVL("gfx700"), false, false},
        {0x023, IREE_SVL("gfx701"), false, false},
        {0x024, IREE_SVL("gfx702"), false, false},
        {0x025, IREE_SVL("gfx703"), false, false},
        {0x026, IREE_SVL("gfx704"), false, false},
        {0x028, IREE_SVL("gfx801"), false, true},
        {0x029, IREE_SVL("gfx802"), false, false},
        {0x02a, IREE_SVL("gfx803"), false, false},
        {0x02b, IREE_SVL("gfx810"), false, true},
        {0x02c, IREE_SVL("gfx900"), false, true},
        {0x02d, IREE_SVL("gfx902"), false, true},
        {0x02e, IREE_SVL("gfx904"), false, true},
        {0x02f, IREE_SVL("gfx906"), true, true},
        {0x030, IREE_SVL("gfx908"), true, true},
        {0x031, IREE_SVL("gfx909"), false, true},
        {0x032, IREE_SVL("gfx90c"), false, true},
        {0x033, IREE_SVL("gfx1010"), false, true},
        {0x034, IREE_SVL("gfx1011"), false, true},
        {0x035, IREE_SVL("gfx1012"), false, true},
        {0x036, IREE_SVL("gfx1030"), false, false},
        {0x037, IREE_SVL("gfx1031"), false, false},
        {0x038, IREE_SVL("gfx1032"), false, false},
        {0x039, IREE_SVL("gfx1033"), false, false},
        {0x03a, IREE_SVL("gfx602"), false, false},
        {0x03b, IREE_SVL("gfx705"), false, false},
        {0x03c, IREE_SVL("gfx805"), false, false},
        {0x03d, IREE_SVL("gfx1035"), false, false},
        {0x03e, IREE_SVL("gfx1034"), false, false},
        {0x03f, IREE_SVL("gfx90a"), true, true},
        {0x040, IREE_SVL("gfx940"), true, true},
        {0x041, IREE_SVL("gfx1100"), false, false},
        {0x042, IREE_SVL("gfx1013"), false, true},
        {0x043, IREE_SVL("gfx1150"), false, false},
        {0x044, IREE_SVL("gfx1103"), false, false},
        {0x045, IREE_SVL("gfx1036"), false, false},
        {0x046, IREE_SVL("gfx1101"), false, false},
        {0x047, IREE_SVL("gfx1102"), false, false},
        {0x048, IREE_SVL("gfx1200"), false, false},
        {0x049, IREE_SVL("gfx1250"), false, false},
        {0x04a, IREE_SVL("gfx1151"), false, false},
        {0x04b, IREE_SVL("gfx941"), true, true},
        {0x04c, IREE_SVL("gfx942"), true, true},
        {0x04e, IREE_SVL("gfx1201"), false, false},
        {0x04f, IREE_SVL("gfx950"), true, true},
        {0x050, IREE_SVL("gfx1310"), false, false},
        {0x051, IREE_SVL("gfx9-generic"), false, true},
        {0x052, IREE_SVL("gfx10-1-generic"), false, true},
        {0x053, IREE_SVL("gfx10-3-generic"), false, false},
        {0x054, IREE_SVL("gfx11-generic"), false, false},
        {0x055, IREE_SVL("gfx1152"), false, false},
        {0x058, IREE_SVL("gfx1153"), false, false},
        {0x059, IREE_SVL("gfx12-generic"), false, false},
        {0x05a, IREE_SVL("gfx1251"), false, false},
        {0x05b, IREE_SVL("gfx12-5-generic"), false, false},
        {0x05c, IREE_SVL("gfx1172"), false, false},
        {0x05d, IREE_SVL("gfx1170"), false, false},
        {0x05e, IREE_SVL("gfx1171"), false, false},
        {0x05f, IREE_SVL("gfx9-4-generic"), true, true},
};

//===----------------------------------------------------------------------===//
// AMDGPU Code Object Target Parsing
//===----------------------------------------------------------------------===//

static bool iree_hal_amdgpu_elf_has_available_bytes(
    iree_const_byte_span_t elf_data, iree_host_size_t byte_count) {
  return elf_data.data != NULL &&
         (elf_data.data_length == 0 || byte_count <= elf_data.data_length);
}

static const iree_hal_amdgpu_elf_machine_target_t*
iree_hal_amdgpu_lookup_elf_machine_target(uint32_t machine) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_hal_amdgpu_elf_machine_targets); ++i) {
    if (iree_hal_amdgpu_elf_machine_targets[i].machine == machine) {
      return &iree_hal_amdgpu_elf_machine_targets[i];
    }
  }
  return NULL;
}

static iree_hal_amdgpu_target_feature_state_t
iree_hal_amdgpu_code_object_decode_v4_feature(uint32_t e_flags, uint32_t mask,
                                              uint32_t any_value,
                                              uint32_t off_value,
                                              uint32_t on_value) {
  const uint32_t value = e_flags & mask;
  if (value == any_value) {
    return IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY;
  } else if (value == off_value) {
    return IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF;
  } else if (value == on_value) {
    return IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON;
  }
  return IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED;
}

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_code_object_target_id_from_elf(
    iree_const_byte_span_t elf_data,
    iree_hal_amdgpu_target_id_t* out_target_id) {
  IREE_ASSERT_ARGUMENT(out_target_id);
  memset(out_target_id, 0, sizeof(*out_target_id));

  if (!iree_hal_elf_data_starts_with_magic(elf_data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU code object does not begin with ELF magic");
  }
  if (!iree_hal_amdgpu_elf_has_available_bytes(
          elf_data, IREE_HAL_AMDGPU_ELF64_HEADER_SIZE)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU code object ELF header truncated");
  }

  const uint8_t* header = elf_data.data;
  if (header[IREE_HAL_AMDGPU_ELF_EI_CLASS] != IREE_HAL_AMDGPU_ELF_CLASS_64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU code object must be a 64-bit ELF");
  }
  if (header[IREE_HAL_AMDGPU_ELF_EI_DATA] != IREE_HAL_AMDGPU_ELF_DATA_2LSB) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU code object must be little-endian");
  }
  if (header[IREE_HAL_AMDGPU_ELF_EI_VERSION] !=
      IREE_HAL_AMDGPU_ELF_VERSION_CURRENT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported AMDGPU code object ELF version %u",
                            header[IREE_HAL_AMDGPU_ELF_EI_VERSION]);
  }
  if (header[IREE_HAL_AMDGPU_ELF_EI_OSABI] != IREE_HAL_AMDGPU_ELF_OSABI_HSA) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU code object must use HSA OSABI");
  }

  const uint16_t e_machine = iree_unaligned_load_le_u16(
      (const uint16_t*)(header + IREE_HAL_AMDGPU_ELF_HEADER_E_MACHINE_OFFSET));
  if (e_machine != IREE_HAL_AMDGPU_ELF_MACHINE_AMDGPU) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF machine %u is not AMDGPU", e_machine);
  }
  const uint32_t e_version = iree_unaligned_load_le_u32(
      (const uint32_t*)(header + IREE_HAL_AMDGPU_ELF_HEADER_E_VERSION_OFFSET));
  if (e_version != IREE_HAL_AMDGPU_ELF_VERSION_CURRENT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported AMDGPU code object e_version %u",
                            e_version);
  }

  const uint32_t e_flags = iree_unaligned_load_le_u32(
      (const uint32_t*)(header + IREE_HAL_AMDGPU_ELF64_HEADER_E_FLAGS_OFFSET));
  const iree_hal_amdgpu_elf_machine_target_t* machine_target =
      iree_hal_amdgpu_lookup_elf_machine_target(e_flags &
                                                IREE_HAL_AMDGPU_EF_MACH);
  if (machine_target == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported AMDGPU code object processor e_flags value 0x%x",
        e_flags & IREE_HAL_AMDGPU_EF_MACH);
  }

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_id_parse(
      machine_target->processor, IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_NONE,
      out_target_id));

  const uint8_t abi_version = header[IREE_HAL_AMDGPU_ELF_EI_ABIVERSION];
  if (abi_version == IREE_HAL_AMDGPU_ELF_HSA_ABI_VERSION_V3) {
    out_target_id->sramecc =
        iree_all_bits_set(e_flags, IREE_HAL_AMDGPU_EF_FEATURE_SRAMECC_V3)
            ? IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON
        : machine_target->sramecc_supported
            ? IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF
            : IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED;
    out_target_id->xnack =
        iree_all_bits_set(e_flags, IREE_HAL_AMDGPU_EF_FEATURE_XNACK_V3)
            ? IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON
        : machine_target->xnack_supported
            ? IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF
            : IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED;
  } else if (abi_version == IREE_HAL_AMDGPU_ELF_HSA_ABI_VERSION_V4 ||
             abi_version == IREE_HAL_AMDGPU_ELF_HSA_ABI_VERSION_V5 ||
             abi_version == IREE_HAL_AMDGPU_ELF_HSA_ABI_VERSION_V6) {
    out_target_id->sramecc = iree_hal_amdgpu_code_object_decode_v4_feature(
        e_flags, IREE_HAL_AMDGPU_EF_FEATURE_SRAMECC_V4,
        IREE_HAL_AMDGPU_EF_FEATURE_SRAMECC_ANY_V4,
        IREE_HAL_AMDGPU_EF_FEATURE_SRAMECC_OFF_V4,
        IREE_HAL_AMDGPU_EF_FEATURE_SRAMECC_ON_V4);
    out_target_id->xnack = iree_hal_amdgpu_code_object_decode_v4_feature(
        e_flags, IREE_HAL_AMDGPU_EF_FEATURE_XNACK_V4,
        IREE_HAL_AMDGPU_EF_FEATURE_XNACK_ANY_V4,
        IREE_HAL_AMDGPU_EF_FEATURE_XNACK_OFF_V4,
        IREE_HAL_AMDGPU_EF_FEATURE_XNACK_ON_V4);
    if (abi_version == IREE_HAL_AMDGPU_ELF_HSA_ABI_VERSION_V6) {
      out_target_id->generic_version =
          (e_flags & IREE_HAL_AMDGPU_EF_GENERIC_VERSION) >>
          IREE_HAL_AMDGPU_EF_GENERIC_VERSION_OFFSET;
    }
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported AMDGPU HSA code object ABI version %u",
                            abi_version);
  }

  if (out_target_id->kind == IREE_HAL_AMDGPU_TARGET_KIND_GENERIC &&
      abi_version != IREE_HAL_AMDGPU_ELF_HSA_ABI_VERSION_V6) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "generic AMDGPU code object target requires HSA ABI v6");
  }
  if (out_target_id->kind == IREE_HAL_AMDGPU_TARGET_KIND_GENERIC &&
      out_target_id->generic_version == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "generic AMDGPU code object target has no generic version");
  }
  if (out_target_id->kind != IREE_HAL_AMDGPU_TARGET_KIND_GENERIC &&
      out_target_id->generic_version != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-generic AMDGPU code object target has generic version %u",
        out_target_id->generic_version);
  }
  return iree_ok_status();
}
