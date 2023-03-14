// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_ELF_ELF_LINKER_H_
#define IREE_HAL_LOCAL_ELF_ELF_LINKER_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/local/elf/arch.h"       // IWYU pragma: export
#include "iree/hal/local/elf/elf_types.h"  // IWYU pragma: export

//==============================================================================
// ELF symbol import table
//==============================================================================

typedef struct iree_elf_import_t {
  const char* sym_name;
  void* thunk_ptr;
} iree_elf_import_t;

typedef struct iree_elf_import_table_t {
  iree_host_size_t import_count;
  const iree_elf_import_t* imports;
} iree_elf_import_table_t;

// TODO(benvanik): add import declaration macros that setup a unique thunk like
// IREE_ELF_DEFINE_IMPORT(foo).

//==============================================================================
// Runtime ELF module loader/linker
//==============================================================================

// An ELF module mapped directly from memory.
typedef struct iree_elf_module_t {
  // Allocator used for additional dynamic memory when needed.
  iree_allocator_t host_allocator;

  // Base host virtual address the module is loaded into.
  uint8_t* vaddr_base;
  // Total size, in bytes, of the virtual address space reservation.
  iree_host_size_t vaddr_size;

  // Bias applied to all relative addresses (from the string table, etc) in the
  // loaded module. This is an offset from the vaddr_base that may not be 0 if
  // host page granularity was larger than the ELF's defined granularity.
  uint8_t* vaddr_bias;

  // Dynamic symbol string table (.dynstr).
  const char* dynstr;            // DT_STRTAB
  iree_host_size_t dynstr_size;  // DT_STRSZ (bytes)

  // Dynamic symbol table (.dynsym).
  const iree_elf_sym_t* dynsym;   // DT_SYMTAB
  iree_host_size_t dynsym_count;  // DT_SYMENT (bytes) / sizeof(iree_elf_sym_t)
} iree_elf_module_t;

// Initializes an ELF module from the ELF |raw_data| in memory.
// |raw_data| only needs to remain valid for the initialization of the module
// and may be discarded afterward.
//
// An optional |import_table| may be specified to provide a set of symbols that
// the module may import. Strong imports will not be resolved from the host
// system and initialization will fail if any are not present in the provided
// table.
//
// Upon return |out_module| is initialized and ready for use with any present
// .init initialization functions having been executed. To release memory
// allocated by the module during loading iree_elf_module_deinitialize must be
// called to unload when it is safe (no more outstanding pointers into the
// loaded module, etc).
iree_status_t iree_elf_module_initialize_from_memory(
    iree_const_byte_span_t raw_data,
    const iree_elf_import_table_t* import_table,
    iree_allocator_t host_allocator, iree_elf_module_t* out_module);

// Deinitializes a |module|, releasing any allocated executable or data pages.
// Invalidates all symbol pointers previous retrieved from the module and any
// pointer to data that may have been in the module text or rwdata.
//
// NOTE: .fini finalizers will not be executed.
void iree_elf_module_deinitialize(iree_elf_module_t* module);

// Returns the host pointer of an exported symbol with the given |symbol_name|.
iree_status_t iree_elf_module_lookup_export(iree_elf_module_t* module,
                                            const char* symbol_name,
                                            void** out_export);

#endif  // IREE_HAL_LOCAL_ELF_ELF_LINKER_H_
