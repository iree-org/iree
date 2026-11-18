// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/local/elf/arch.h"
#include "iree/hal/local/elf/elf_types.h"

#if defined(IREE_ARCH_MIPS_64)

// Documentation:
// https://refspecs.linuxfoundation.org/elf/mipsabi.pdf (64-bit ELF Object File
//   Specification) and the MIPSpro / NUBI n64 ABI supplements.
//
// Two things make MIPS64 relocation handling different from the other
// architectures in this directory (e.g. arch/riscv.c):
//
//   1. Non-standard r_info packing. The generic ELF64 IREE_ELF_R_SYM/R_TYPE
//      macros are wrong for MIPS: n64 stores the symbol index in the low 32
//      bits and packs up to three chained relocation type bytes in the high 32
//      bits (r_type in the most-significant byte). We decode locally below.
//
//   2. The GOT is relocated implicitly. MIPS emits no explicit relocations for
//      .got entries; instead the loader must add the load bias to the local GOT
//      entries and resolve the global GOT entries from .dynsym, using the GOT
//      layout described by the DT_MIPS_* dynamic tags. This is the bulk of the
//      work and has no precedent elsewhere in this tree.

//==============================================================================
// ELF machine type/ABI
//==============================================================================

bool iree_elf_machine_is_valid(iree_elf_half_t machine) {
  return machine == 0x08;  // EM_MIPS / 8
}

//==============================================================================
// ELF relocations
//==============================================================================

enum {
  IREE_ELF_R_MIPS_NONE = 0,
  IREE_ELF_R_MIPS_REL32 = 3,
  IREE_ELF_R_MIPS_64 = 18,
};

// The MIPS ABI reserves the first two GOT entries: GOT[0] for the (unused here)
// lazy resolver and GOT[1] for the module pointer. The remaining local entries
// up to DT_MIPS_LOCAL_GOTNO just need the load bias applied.
#define IREE_ELF_MIPS_RESERVED_GOT_ENTRIES 2

// Decodes the MIPS64 n64 r_info field. Unlike the generic ELF64 packing the
// symbol index is in the low 32 bits and the (primary) relocation type is the
// high byte; a second chained type (used to widen REL32 to 64 bits) is the byte
// below it.
static inline uint32_t iree_elf_mips_r_sym(uint64_t r_info) {
  return (uint32_t)(r_info & 0xFFFFFFFFu);
}
static inline uint8_t iree_elf_mips_r_type(uint64_t r_info) {
  return (uint8_t)(r_info >> 56);
}
static inline uint8_t iree_elf_mips_r_type2(uint64_t r_info) {
  return (uint8_t)(r_info >> 48);
}

// Applies the .rel.dyn REL relocations. MIPS uses DT_REL (implicit addend read
// from the target location), not DT_RELA.
static iree_status_t iree_elf_arch_mips_apply_rel(
    iree_elf_relocation_state_t* state, iree_host_size_t rel_count,
    const iree_elf_rel_t* rel_table) {
  for (iree_host_size_t i = 0; i < rel_count; ++i) {
    const iree_elf_rel_t* rel = &rel_table[i];
    uint8_t type = iree_elf_mips_r_type(rel->r_info);
    if (type == IREE_ELF_R_MIPS_NONE) {
      continue;
    }

    uint32_t sym_ordinal = iree_elf_mips_r_sym(rel->r_info);
    // The value added to the (implicit) addend: the load bias for a local/
    // relative relocation (symbol index 0), or the resolved symbol address for
    // a reference to a defined global symbol.
    iree_elf_addr_t value = (iree_elf_addr_t)state->vaddr_bias;
    if (sym_ordinal != 0) {
      if (sym_ordinal >= state->dynsym_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "invalid symbol in relocation: %u",
                                sym_ordinal);
      }
      value = (iree_elf_addr_t)state->vaddr_bias +
              state->dynsym[sym_ordinal].st_value;
    }

    uint8_t* target = state->vaddr_bias + rel->r_offset;
    switch (type) {
      case IREE_ELF_R_MIPS_REL32:
        // lld emits REL32 for 64-bit pointers as the composite
        // REL32/R_MIPS_64/NONE; the chained R_MIPS_64 widens the field to 64
        // bits. A bare REL32 (no R_MIPS_64) stays 32-bit.
        if (iree_elf_mips_r_type2(rel->r_info) == IREE_ELF_R_MIPS_64) {
          *(uint64_t*)target += (uint64_t)value;
        } else {
          *(uint32_t*)target += (uint32_t)value;
        }
        break;
      case IREE_ELF_R_MIPS_64:
        *(uint64_t*)target += (uint64_t)value;
        break;
      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unimplemented mips64 relocation type %u",
                                type);
    }
  }
  return iree_ok_status();
}

// Relocates the GOT, which MIPS does not describe with explicit relocations.
// `pltgot` is the vaddr of .got; the DT_MIPS_* values describe its layout.
static iree_status_t iree_elf_arch_mips_relocate_got(
    iree_elf_relocation_state_t* state, iree_elf_addr_t pltgot,
    uint64_t local_gotno, uint64_t gotsym, uint64_t symtabno) {
  uint64_t* got = (uint64_t*)(state->vaddr_bias + pltgot);

  // Local GOT entries: add the load bias to each, skipping the two reserved
  // entries the loader does not own.
  for (uint64_t i = IREE_ELF_MIPS_RESERVED_GOT_ENTRIES; i < local_gotno; ++i) {
    got[i] += (uint64_t)state->vaddr_bias;
  }

  // Global GOT entries: one per .dynsym entry in [gotsym, symtabno), laid out
  // immediately after the local entries. Each resolves to its symbol address.
  if (symtabno > state->dynsym_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "DT_MIPS_SYMTABNO (%" PRIu64
                            ") exceeds .dynsym count (%" PRIhsz ")",
                            symtabno, state->dynsym_count);
  }
  uint64_t* global_got = got + local_gotno;
  for (uint64_t i = gotsym; i < symtabno; ++i) {
    const iree_elf_sym_t* sym = &state->dynsym[i];
    if (sym->st_shndx == IREE_ELF_SHN_UNDEF) {
      // An undefined global symbol is an external import (e.g. libm) that only
      // a dynamic linker could resolve. IREE's embedded modules are hermetic,
      // so reaching here means the module depends on something we cannot
      // provide.
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "embedded MIPS module references undefined "
                              "external symbol (index %" PRIu64
                              "); only hermetic modules are supported",
                              i);
    }
    global_got[i - gotsym] = (uint64_t)state->vaddr_bias + sym->st_value;
  }

  return iree_ok_status();
}

iree_status_t iree_elf_arch_apply_relocations(
    iree_elf_relocation_state_t* state) {
  // Gather the REL table and the MIPS GOT-layout tags in a single pass.
  const iree_elf_rel_t* rel_table = NULL;
  iree_host_size_t rel_count = 0;
  iree_elf_addr_t pltgot = 0;
  bool have_pltgot = false;
  uint64_t local_gotno = 0;
  uint64_t gotsym = 0;
  uint64_t symtabno = 0;
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
        // lld targeting MIPS emits REL, not RELA; RELA would need separate
        // handling (explicit addends) that we have no inputs to exercise.
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unsupported DT_RELA relocations on MIPS");
      case IREE_ELF_DT_PLTGOT:
        pltgot = dyn->d_un.d_ptr;
        have_pltgot = true;
        break;
      case IREE_ELF_DT_MIPS_LOCAL_GOTNO:
        local_gotno = dyn->d_un.d_val;
        break;
      case IREE_ELF_DT_MIPS_GOTSYM:
        gotsym = dyn->d_un.d_val;
        break;
      case IREE_ELF_DT_MIPS_SYMTABNO:
        symtabno = dyn->d_un.d_val;
        break;
      default:
        // Ignored.
        break;
    }
  }

  if (rel_table && rel_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_elf_arch_mips_apply_rel(state, rel_count, rel_table));
  }

  if (have_pltgot && local_gotno > 0) {
    IREE_RETURN_IF_ERROR(iree_elf_arch_mips_relocate_got(
        state, pltgot, local_gotno, gotsym, symtabno));
  }

  return iree_ok_status();
}

//==============================================================================
// Cross-ABI function calls
//==============================================================================
// The embedded module shares the host n64 ABI, so these are plain
// function-pointer casts (copied from the other 64-bit arch backends).

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

#endif  // IREE_ARCH_MIPS_64
