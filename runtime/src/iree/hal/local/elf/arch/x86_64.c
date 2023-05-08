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

#if defined(IREE_ARCH_X86_64)

// Documentation:
// https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf

//==============================================================================
// ELF machine type/ABI
//==============================================================================

bool iree_elf_machine_is_valid(iree_elf_half_t machine) {
  return machine == 0x3E;  // EM_X86_64 / 62
}

//==============================================================================
// ELF relocations
//==============================================================================

enum {
  IREE_ELF_R_X86_64_NONE = 0,       // No reloc
  IREE_ELF_R_X86_64_64 = 1,         // Direct 64 bit
  IREE_ELF_R_X86_64_PC32 = 2,       // PC relative 32 bit signed
  IREE_ELF_R_X86_64_GOT32 = 3,      // 32 bit GOT entry
  IREE_ELF_R_X86_64_PLT32 = 4,      // 32 bit PLT address
  IREE_ELF_R_X86_64_COPY = 5,       // Copy symbol at runtime
  IREE_ELF_R_X86_64_GLOB_DAT = 6,   // Create GOT entry
  IREE_ELF_R_X86_64_JUMP_SLOT = 7,  // Create PLT entry
  IREE_ELF_R_X86_64_RELATIVE = 8,   // Adjust by program base
  IREE_ELF_R_X86_64_GOTPCREL = 9,   // 32 bit signed pc relative offset to GOT
  IREE_ELF_R_X86_64_32 = 10,        // Direct 32 bit zero extended
  IREE_ELF_R_X86_64_32S = 11,       // Direct 32 bit sign extended
  IREE_ELF_R_X86_64_16 = 12,        // Direct 16 bit zero extended
  IREE_ELF_R_X86_64_PC16 = 13,      // 16 bit sign extended pc relative
  IREE_ELF_R_X86_64_8 = 14,         // Direct 8 bit sign extended
  IREE_ELF_R_X86_64_PC8 = 15,       // 8 bit sign extended pc relative
  IREE_ELF_R_X86_64_PC64 = 24,      // Place relative 64-bit signed
};

static iree_status_t iree_elf_arch_x86_64_apply_rela(
    iree_elf_relocation_state_t* state, iree_host_size_t rela_count,
    const iree_elf_rela_t* rela_table) {
  for (iree_host_size_t i = 0; i < rela_count; ++i) {
    const iree_elf_rela_t* rela = &rela_table[i];
    uint32_t type = IREE_ELF_R_TYPE(rela->r_info);
    if (type == IREE_ELF_R_X86_64_NONE) continue;

    iree_elf_addr_t sym_addr = 0;
    uint32_t sym_ordinal = (uint32_t)IREE_ELF_R_SYM(rela->r_info);
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
        (iree_elf_addr_t)state->vaddr_bias + rela->r_offset;
    switch (type) {
      // case IREE_ELF_R_X86_64_NONE: early-exit above
      case IREE_ELF_R_X86_64_RELATIVE:
        *(uint64_t*)instr_ptr = (uint64_t)(state->vaddr_bias + rela->r_addend);
        break;
      case IREE_ELF_R_X86_64_JUMP_SLOT:
        *(uint64_t*)instr_ptr = (uint64_t)sym_addr;
        break;
      case IREE_ELF_R_X86_64_GLOB_DAT:
        *(uint64_t*)instr_ptr = (uint64_t)sym_addr;
        break;
      case IREE_ELF_R_X86_64_COPY:
        *(uint64_t*)instr_ptr = (uint64_t)sym_addr;
        break;
      case IREE_ELF_R_X86_64_64:
        *(uint64_t*)instr_ptr = (uint64_t)(sym_addr + rela->r_addend);
        break;
      case IREE_ELF_R_X86_64_32:
        *(uint32_t*)instr_ptr = (uint32_t)(sym_addr + rela->r_addend);
        break;
      case IREE_ELF_R_X86_64_32S:
        *(int32_t*)instr_ptr = (int32_t)(sym_addr + rela->r_addend);
        break;
      case IREE_ELF_R_X86_64_PC32:
        *(uint32_t*)instr_ptr =
            (uint32_t)(sym_addr + rela->r_addend - instr_ptr);
        break;
      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unimplemented x86_64 relocation type %08X",
                                type);
    }
  }
  return iree_ok_status();
}

iree_status_t iree_elf_arch_apply_relocations(
    iree_elf_relocation_state_t* state) {
  // Gather the relevant relocation tables.
  iree_host_size_t rela_count = 0;
  const iree_elf_rela_t* rela_table = NULL;
  iree_host_size_t plt_rela_count = 0;
  const iree_elf_rela_t* plt_rela_table = NULL;
  for (iree_host_size_t i = 0; i < state->dyn_table_count; ++i) {
    const iree_elf_dyn_t* dyn = &state->dyn_table[i];
    switch (dyn->d_tag) {
      case IREE_ELF_DT_RELA:
        rela_table =
            (const iree_elf_rela_t*)(state->vaddr_bias + dyn->d_un.d_ptr);
        break;
      case IREE_ELF_DT_RELASZ:
        rela_count = dyn->d_un.d_val / sizeof(iree_elf_rela_t);
        break;

      case IREE_ELF_DT_PLTREL:
        // Type of reloc in PLT; we expect DT_RELA right now.
        if (dyn->d_un.d_val != IREE_ELF_DT_RELA) {
          return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "unsupported DT_PLTREL != DT_RELA");
        }
        break;
      case IREE_ELF_DT_JMPREL:
        plt_rela_table =
            (const iree_elf_rela_t*)(state->vaddr_bias + dyn->d_un.d_ptr);
        break;
      case IREE_ELF_DT_PLTRELSZ:
        plt_rela_count = dyn->d_un.d_val / sizeof(iree_elf_rela_t);
        break;

      case IREE_ELF_DT_REL:
      case IREE_ELF_DT_RELSZ:
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "unsupported DT_REL relocations");

      default:
        // Ignored.
        break;
    }
  }
  if (!rela_table) rela_count = 0;
  if (!plt_rela_table) plt_rela_count = 0;

  if (rela_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_elf_arch_x86_64_apply_rela(state, rela_count, rela_table));
  }
  if (plt_rela_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_elf_arch_x86_64_apply_rela(state, plt_rela_count, plt_rela_table));
  }

  return iree_ok_status();
}

//==============================================================================
// Cross-ABI function calls
//==============================================================================

// System V AMD64 ABI (used in IREE):
// https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
// Arguments:
//   RDI, RSI, RDX, RCX, R8, R9, [stack]...
// Results:
//   RAX, RDX
//
// Everything but Windows uses this convention (linux/bsd/mac/etc) and as such
// we can just use nice little C thunks.

#if defined(IREE_PLATFORM_WINDOWS)
// Host is using the Microsoft x64 calling convention and we need to translate
// to the System V AMD64 ABI conventions. Unfortunately MSVC does not support
// inline assembly and we have to outline the calls in x86_64_msvc.asm.
#else

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

#endif  // IREE_PLATFORM_WINDOWS

#endif  // IREE_ARCH_X86_64
