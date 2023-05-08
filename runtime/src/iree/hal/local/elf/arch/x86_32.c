// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/target_platform.h"
#include "iree/hal/local/elf/arch.h"
#include "iree/hal/local/elf/elf_types.h"

#if defined(IREE_ARCH_X86_32)

// Documentation:
// https://uclibc.org/docs/psABI-i386.pdf

//==============================================================================
// ELF machine type/ABI
//==============================================================================

bool iree_elf_machine_is_valid(iree_elf_half_t machine) {
  return machine == 0x03;  // EM_386 / 3
}

//==============================================================================
// ELF relocations
//==============================================================================

enum {
  IREE_ELF_R_386_NONE = 0,
  IREE_ELF_R_386_32 = 1,
  IREE_ELF_R_386_PC32 = 2,
  IREE_ELF_R_386_GLOB_DAT = 6,
  IREE_ELF_R_386_JMP_SLOT = 7,
  IREE_ELF_R_386_RELATIVE = 8,
};

static iree_status_t iree_elf_arch_x86_32_apply_rel(
    iree_elf_relocation_state_t* state, iree_host_size_t rel_count,
    const iree_elf_rel_t* rel_table) {
  for (iree_host_size_t i = 0; i < rel_count; ++i) {
    const iree_elf_rel_t* rel = &rel_table[i];
    uint32_t type = IREE_ELF_R_TYPE(rel->r_info);
    if (type == IREE_ELF_R_386_NONE) continue;

    iree_elf_addr_t sym_addr = 0;
    uint32_t sym_ordinal = (uint32_t)IREE_ELF_R_SYM(rel->r_info);
    if (sym_ordinal != 0) {
      if (sym_ordinal >= state->dynsym_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "invalid symbol in relocation: %u",
                                sym_ordinal);
      }
      sym_addr = (iree_elf_addr_t)state->vaddr_bias +
                 state->dynsym[sym_ordinal].st_value;
    }

    iree_elf_addr_t instr_ptr =
        (iree_elf_addr_t)state->vaddr_bias + rel->r_offset;
    switch (type) {
        // case IREE_ELF_R_386_NONE: early-exit above
      case IREE_ELF_R_386_JMP_SLOT:
        *(uint32_t*)instr_ptr = (uint32_t)sym_addr;
        break;
      case IREE_ELF_R_386_GLOB_DAT:
        *(uint32_t*)instr_ptr = (uint32_t)sym_addr;
        break;
      case IREE_ELF_R_386_RELATIVE:
        *(uint32_t*)instr_ptr += (uint32_t)state->vaddr_bias;
        break;
      case IREE_ELF_R_386_32:
        *(uint32_t*)instr_ptr += (uint32_t)sym_addr;
        break;
      case IREE_ELF_R_386_PC32:
        *(uint32_t*)instr_ptr += (uint32_t)(sym_addr - instr_ptr);
        break;
      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unimplemented x86 relocation type %08X", type);
    }
  }
  return iree_ok_status();
}

iree_status_t iree_elf_arch_apply_relocations(
    iree_elf_relocation_state_t* state) {
  // Gather the relevant relocation tables.
  iree_host_size_t rel_count = 0;
  const iree_elf_rel_t* rel_table = NULL;
  for (iree_host_size_t i = 0; i < state->dyn_table_count; ++i) {
    const iree_elf_dyn_t* dyn = &state->dyn_table[i];
    switch (dyn->d_tag) {
      case IREE_ELF_DT_REL:
        rel_table =
            (const iree_elf_rel_t*)(state->vaddr_bias + dyn->d_un.d_ptr);
        break;
      case IREE_ELF_DT_RELSZ:
        rel_count = dyn->d_un.d_val / sizeof(iree_elf_rel_t);
        break;

      case IREE_ELF_DT_RELA:
      case IREE_ELF_DT_RELASZ:
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "unsupported DT_RELA relocations");
      default:
        // Ignored.
        break;
    }
  }
  if (!rel_table) rel_count = 0;

  if (rel_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_elf_arch_x86_32_apply_rel(state, rel_count, rel_table));
  }

  return iree_ok_status();
}

//==============================================================================
// Cross-ABI function calls
//==============================================================================

// System V i386 ABI (used in IREE):
// https://uclibc.org/docs/psABI-i386.pdf
// Arguments:
//   (reverse order on the stack; last arg furthest from stack pointer)
//
// Results:
//   EAX
//
// Non-volatile:
//   EBX, ESP, EBP, ESI, EDI
//
// MSVC shares this convention for the default cdecl calls.

void iree_elf_call_v_v(const void* symbol_ptr) {
  typedef void (*ptr_t)(void);
  ((ptr_t)symbol_ptr)();
}

void* iree_elf_call_p_i(const void* symbol_ptr, int a0) {
  typedef void* (*ptr_t)(int);
  return ((ptr_t)symbol_ptr)(a0);
}

void* iree_elf_call_p_ip(const void* symbol_ptr, int a0, void* a1) {
  typedef void* (*ptr_t)(int, void*);
  return ((ptr_t)symbol_ptr)(a0, a1);
}

int iree_elf_call_i_p(const void* symbol_ptr, void* a0) {
  typedef int (*ptr_t)(void*);
  return ((ptr_t)symbol_ptr)(a0);
}

int iree_elf_call_i_ppp(const void* symbol_ptr, void* a0, void* a1, void* a2) {
  typedef int (*ptr_t)(void*, void*, void*);
  return ((ptr_t)symbol_ptr)(a0, a1, a2);
}

void* iree_elf_call_p_ppp(const void* symbol_ptr, void* a0, void* a1,
                          void* a2) {
  typedef void* (*ptr_t)(void*, void*, void*);
  return ((ptr_t)symbol_ptr)(a0, a1, a2);
}

int iree_elf_thunk_i_ppp(const void* symbol_ptr, void* a0, void* a1, void* a2) {
  typedef int (*ptr_t)(void*, void*, void*);
  return ((ptr_t)symbol_ptr)(a0, a1, a2);
}

#endif  // IREE_ARCH_X86_32
