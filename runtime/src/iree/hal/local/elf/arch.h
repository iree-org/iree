// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_ELF_ARCH_H_
#define IREE_HAL_LOCAL_ELF_ARCH_H_

#include "iree/base/api.h"
#include "iree/hal/local/elf/elf_types.h"

//==============================================================================
// ELF machine type/ABI
//==============================================================================

// Returns true if the reported ELF machine specification is valid for the
// current architecture.
bool iree_elf_machine_is_valid(iree_elf_half_t machine);

//==============================================================================
// ELF relocations
//==============================================================================

// State used during relocation.
typedef struct iree_elf_relocation_state_t {
  // Bias applied to all relative addresses (from the string table, etc) in the
  // loaded module. This is an offset from the vaddr_base that may not be 0 if
  // host page granularity was larger than the ELF's defined granularity.
  uint8_t* vaddr_bias;

  // PT_DYNAMIC table.
  const iree_elf_dyn_t* dyn_table;
  iree_host_size_t dyn_table_count;

  // Dynamic symbol table (.dynsym) loaded into virtual memory.
  const iree_elf_sym_t* dynsym;   // DT_SYMTAB
  iree_host_size_t dynsym_count;  // DT_SYMENT (bytes) / sizeof(iree_elf_sym_t)
} iree_elf_relocation_state_t;

// Applies architecture-specific relocations.
iree_status_t iree_elf_arch_apply_relocations(
    iree_elf_relocation_state_t* state);

//==============================================================================
// Cross-ABI function calls
//==============================================================================

// TODO(benvanik): add thunk functions (iree_elf_thunk_*) to be used by imports
// for marshaling from linux ABI in the ELF to host ABI.

// Host -> ELF: void(*)(void)
void iree_elf_call_v_v(const void* symbol_ptr);

// Host -> ELF: void*(*)(int)
void* iree_elf_call_p_i(const void* symbol_ptr, int a0);

// Host -> ELF: void*(*)(int, void*)
void* iree_elf_call_p_ip(const void* symbol_ptr, int a0, void* a1);

// Host -> ELF: int(*)(void*)
int iree_elf_call_i_p(const void* symbol_ptr, void* a0);

// Host -> ELF: int(*)(void*, void*, void*)
int iree_elf_call_i_ppp(const void* symbol_ptr, void* a0, void* a1, void* a2);

// Host -> ELF: void*(*)(void*, void*, void*)
void* iree_elf_call_p_ppp(const void* symbol_ptr, void* a0, void* a1, void* a2);

// ELF -> Host: int(*)(void*, void*, void*)
int iree_elf_thunk_i_ppp(const void* symbol_ptr, void* a0, void* a1, void* a2);

#endif  // IREE_HAL_LOCAL_ELF_ARCH_H_
