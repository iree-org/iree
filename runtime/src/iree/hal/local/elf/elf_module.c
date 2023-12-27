// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/elf/elf_module.h"

#include <inttypes.h>
#include <string.h>

#include "iree/hal/local/elf/arch.h"
#include "iree/hal/local/elf/fatelf.h"
#include "iree/hal/local/elf/platform.h"

//==============================================================================
// Verification and section/info caching
//==============================================================================

// Fields taken from the ELF headers used only during verification and loading.
typedef struct iree_elf_module_load_state_t {
  iree_memory_info_t memory_info;
  const iree_elf_ehdr_t* ehdr;
  const iree_elf_phdr_t* phdr_table;  // ehdr.e_phnum has count
  const iree_elf_shdr_t* shdr_table;  // ehdr.e_shnum has count

  const iree_elf_dyn_t* dyn_table;  // PT_DYNAMIC
  iree_host_size_t dyn_table_count;

  iree_elf_addr_t init;               // DT_INIT
  const iree_elf_addr_t* init_array;  // DT_INIT_ARRAY
  iree_host_size_t init_array_count;  // DT_INIT_ARRAYSZ
} iree_elf_module_load_state_t;

// Verifies the ELF file header and machine class.
static iree_status_t iree_elf_module_verify_ehdr(
    iree_const_byte_span_t raw_data) {
  // Size must be larger than the header we are trying to load.
  if (raw_data.data_length < sizeof(iree_elf_ehdr_t)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "ELF data provided (%" PRIhsz
                            ") is smaller than ehdr (%zu)",
                            raw_data.data_length, sizeof(iree_elf_ehdr_t));
  }

  // Check for ELF identifier.
  const iree_elf_ehdr_t* ehdr = (const iree_elf_ehdr_t*)raw_data.data;
  static const iree_elf_byte_t elf_magic[4] = {0x7F, 'E', 'L', 'F'};
  if (memcmp(ehdr->e_ident, elf_magic, sizeof(elf_magic)) != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "data provided does not contain the ELF identifier");
  }

  // Check critical identifier bytes before attempting to deal with any more of
  // the header; the class determines the size of the header fields and the
  // endianness determines how multi-byte fields are interpreted.

#if defined(IREE_PTR_SIZE_32)
  if (ehdr->e_ident[IREE_ELF_EI_CLASS] != IREE_ELF_ELFCLASS32) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "system/ELF class mismatch: expected 32-bit");
  }
#elif defined(IREE_PTR_SIZE_64)
  if (ehdr->e_ident[IREE_ELF_EI_CLASS] != IREE_ELF_ELFCLASS64) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "system/ELF class mismatch: expected 64-bit");
  }
#endif  // IREE_PTR_SIZE_*

#if defined(IREE_ENDIANNESS_LITTLE)
  if (ehdr->e_ident[IREE_ELF_EI_DATA] != IREE_ELF_ELFDATA2LSB) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "system/ELF endianness mismatch: expected little-endian");
  }
#else
  if (ehdr->e_ident[IREE_ELF_EI_DATA] != IREE_ELF_ELFDATA2MSB) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "system/ELF endianness mismatch: expected big-endian");
  }
#endif  // IREE_ENDIANNESS_*

  // ELF version == EV_CURRENT (1) is all we handle.
  // Check this before other fields as they could change meaning in other
  // versions.
  if (ehdr->e_version != 1) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "ELF version %u unsupported; expected 1");
  }

  // Ensure we have the right architecture compiled in.
  if (!iree_elf_machine_is_valid(ehdr->e_machine)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "ELF machine specification (%04X) does not match the "
        "running architecture",
        (uint32_t)ehdr->e_machine);
  }

  // We could probably support non-shared object types but no need today and it
  // allows us to make assumptions about the sections that are present (all
  // those marked as 'mandatory' in the spec.
  if (ehdr->e_type != IREE_ELF_ET_DYN) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "only shared object ELFs are supported");
  }

  // Sanity checks on entity sizes - they can be larger than what we expect,
  // but overlaying our structs onto them is not going to work if they are
  // smaller. For now we aren't doing pointer walks based on dynamic sizes so
  // we need equality, but if we ever have a reason to do so we could change all
  // array-style accesses to scale out based on the ehdr values
  if (ehdr->e_ehsize != sizeof(iree_elf_ehdr_t) ||
      ehdr->e_phentsize != sizeof(iree_elf_phdr_t) ||
      ehdr->e_shentsize != sizeof(iree_elf_shdr_t)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "ELF entity size mismatch");
  }

  // Verify the phdr table properties. This doesn't validate each phdr but just
  // ensures that the table is constructed correctly and within bounds.
  if (ehdr->e_phoff == 0 || ehdr->e_phnum == 0 ||
      (ehdr->e_phoff + ehdr->e_phnum * ehdr->e_phentsize) >
          raw_data.data_length) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "invalid mandatory phdr table");
  }

  // Verify the shdr table properties.
  if (ehdr->e_shoff == 0 || ehdr->e_shnum == 0 ||
      (ehdr->e_shoff + ehdr->e_shnum * ehdr->e_shentsize) >
          raw_data.data_length) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "invalid mandatory shdr table");
  }

  return iree_ok_status();
}

// Verifies the phdr table for supported types and in-bounds file references.
static iree_status_t iree_elf_module_verify_phdr_table(
    iree_const_byte_span_t raw_data, iree_elf_module_load_state_t* load_state) {
  for (iree_elf_half_t i = 0; i < load_state->ehdr->e_phnum; ++i) {
    const iree_elf_phdr_t* phdr = &load_state->phdr_table[i];
    if (phdr->p_type != IREE_ELF_PT_LOAD) continue;
    if (phdr->p_offset + phdr->p_filesz > raw_data.data_length) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "phdr reference outside of file extents: %" PRIu64
                              "-%" PRIu64 "of max %" PRIu64,
                              (uint64_t)phdr->p_offset,
                              (uint64_t)(phdr->p_offset + phdr->p_filesz),
                              (uint64_t)raw_data.data_length);
    }
  }
  return iree_ok_status();
}

// Parses the ELF to populate fields used during loading and runtime and verify
// that the ELF matches our very, very low expectations.
static iree_status_t iree_elf_module_parse_headers(
    iree_const_byte_span_t raw_data,
    iree_elf_module_load_state_t* out_load_state,
    iree_elf_module_t* out_module) {
  memset(out_module, 0, sizeof(*out_module));
  memset(out_load_state, 0, sizeof(*out_load_state));

  // Query the host memory information that we can use to verify we are able to
  // meet the alignment requirements of the ELF.
  out_load_state->memory_info = iree_memory_query_info();

  // Verify the ELF is an ELF and that it's for the current machine.
  // NOTE: this only verifies the ehdr is as expected and nothing else: the ELF
  // is still untrusted and may be missing mandatory sections.
  IREE_RETURN_IF_ERROR(iree_elf_module_verify_ehdr(raw_data));

  // Get the primary tables (locations verified above).
  const iree_elf_ehdr_t* ehdr = (const iree_elf_ehdr_t*)raw_data.data;
  const iree_elf_phdr_t* phdr_table =
      (const iree_elf_phdr_t*)(raw_data.data + ehdr->e_phoff);
  const iree_elf_shdr_t* shdr_table =
      (const iree_elf_shdr_t*)(raw_data.data + ehdr->e_shoff);
  out_load_state->ehdr = ehdr;
  out_load_state->phdr_table = phdr_table;
  out_load_state->shdr_table = shdr_table;

  // Verify the phdr table to ensure all bounds are in range of the file.
  IREE_RETURN_IF_ERROR(
      iree_elf_module_verify_phdr_table(raw_data, out_load_state));

  return iree_ok_status();
}

//==============================================================================
// Allocation and layout
//==============================================================================

// Calculates the in-memory layout of the ELF module as defined by its segments.
// Returns a byte range representing the minimum virtual address offset of any
// segment that can be used to offset the vaddr from the host allocation and the
// total length of the required range. The alignment will meet the requirements
// of the ELF but is yet unadjusted for host requirements. The range will have
// zero length if there are no segments to load (which would be weird).
static iree_byte_range_t iree_elf_module_calculate_vaddr_range(
    iree_elf_module_load_state_t* load_state) {
  // Min/max virtual addresses of any allocated segment.
  iree_elf_addr_t vaddr_min = IREE_ELF_ADDR_MAX;
  iree_elf_addr_t vaddr_max = IREE_ELF_ADDR_MIN;
  for (iree_elf_half_t i = 0; i < load_state->ehdr->e_phnum; ++i) {
    const iree_elf_phdr_t* phdr = &load_state->phdr_table[i];
    if (phdr->p_type != IREE_ELF_PT_LOAD) continue;
    iree_elf_addr_t p_vaddr_min =
        iree_page_align_start(phdr->p_vaddr, phdr->p_align);
    iree_elf_addr_t p_vaddr_max =
        iree_page_align_end(phdr->p_vaddr + phdr->p_memsz, phdr->p_align);
    vaddr_min = iree_min(vaddr_min, p_vaddr_min);
    vaddr_max = iree_max(vaddr_max, p_vaddr_max);
  }
  if (vaddr_min == IREE_ELF_ADDR_MAX) {
    // Did not find any segments to load.
    vaddr_min = IREE_ELF_ADDR_MIN;
    vaddr_max = IREE_ELF_ADDR_MIN;
  }
  iree_byte_range_t byte_range = {
      .offset = (iree_host_size_t)vaddr_min,
      .length = (iree_host_size_t)(vaddr_max - vaddr_min),
  };
  return byte_range;
}

// Allocates space for and loads all DT_LOAD segments into the host virtual
// address space.
static iree_status_t iree_elf_module_load_segments(
    iree_const_byte_span_t raw_data, iree_elf_module_load_state_t* load_state,
    iree_elf_module_t* module) {
  // Calculate the total internally-aligned vaddr range.
  iree_byte_range_t vaddr_range =
      iree_elf_module_calculate_vaddr_range(load_state);

  // Reserve virtual address space in the host memory space. This memory is
  // uncommitted by default as the ELF may only sparsely use the address space.
  module->vaddr_size = iree_page_align_end(
      vaddr_range.length, load_state->memory_info.normal_page_size);
  IREE_RETURN_IF_ERROR(iree_memory_view_reserve(
      IREE_MEMORY_VIEW_FLAG_MAY_EXECUTE, module->vaddr_size,
      module->host_allocator, (void**)&module->vaddr_base));
  module->vaddr_bias = module->vaddr_base - vaddr_range.offset;

  // Commit and load all of the segments.
  for (iree_elf_half_t i = 0; i < load_state->ehdr->e_phnum; ++i) {
    const iree_elf_phdr_t* phdr = &load_state->phdr_table[i];
    if (phdr->p_type != IREE_ELF_PT_LOAD) continue;

    // Commit the range of pages used by this segment, initially with write
    // access so that we can modify the pages.
    iree_byte_range_t byte_range = {
        .offset = phdr->p_vaddr,
        .length = phdr->p_memsz,
    };
    IREE_RETURN_IF_ERROR(iree_memory_view_commit_ranges(
        module->vaddr_bias, 1, &byte_range,
        IREE_MEMORY_ACCESS_READ | IREE_MEMORY_ACCESS_WRITE));

    // Copy data present in the file.
    // TODO(benvanik): infra for being able to detect if the source model is in
    // a mapped file - if it is, we can remap the page and directly reference it
    // here for read-only segments and setup copy-on-write for writeable ones.
    // We'd need a way to pass in the underlying mapping and some guarantees on
    // the lifetime of it. Today we are just always committing above and copying
    // here because it keeps this all super simple (you know, as simple as an
    // entire custom ELF loader can be :).
    if (phdr->p_filesz > 0) {
      memcpy(module->vaddr_bias + phdr->p_vaddr, raw_data.data + phdr->p_offset,
             phdr->p_filesz);
    }

    // NOTE: p_memsz may be larger than p_filesz - if so, the extra memory bytes
    // must be zeroed. We require that the initial allocation is zeroed anyway
    // so this is a no-op.

    // NOTE: the pages are still writeable; we need to apply relocations before
    // we can go back through and remove write access from read-only/executable
    // pages in iree_elf_module_protect_segments.
  }

  return iree_ok_status();
}

// Applies segment memory protection attributes.
// This will make pages read-only and must only be performed after relocation
// (which writes to pages of all types). Executable pages will be flushed from
// the instruction cache.
static iree_status_t iree_elf_module_protect_segments(
    iree_elf_module_load_state_t* load_state, iree_elf_module_t* module) {
  // PT_LOAD segments (the bulk of progbits):
  for (iree_elf_half_t i = 0; i < load_state->ehdr->e_phnum; ++i) {
    const iree_elf_phdr_t* phdr = &load_state->phdr_table[i];
    if (phdr->p_type != IREE_ELF_PT_LOAD) continue;

    // Interpret the access bits and widen to the implicit allowable
    // permissions. See Table 7-37:
    // https://docs.oracle.com/cd/E19683-01/816-1386/6m7qcoblk/index.html#chapter6-34713
    iree_memory_access_t access = 0;
    if (phdr->p_flags & IREE_ELF_PF_R) access |= IREE_MEMORY_ACCESS_READ;
    if (phdr->p_flags & IREE_ELF_PF_W) access |= IREE_MEMORY_ACCESS_WRITE;
    if (phdr->p_flags & IREE_ELF_PF_X) access |= IREE_MEMORY_ACCESS_EXECUTE;
    if (access & IREE_MEMORY_ACCESS_WRITE) access |= IREE_MEMORY_ACCESS_READ;
    if (access & IREE_MEMORY_ACCESS_EXECUTE) access |= IREE_MEMORY_ACCESS_READ;

    // We only support R+X (no W).
    if ((phdr->p_flags & IREE_ELF_PF_X) && (phdr->p_flags & IREE_ELF_PF_W)) {
      return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                              "unable to create a writable executable segment");
    }

    // Apply new access protection.
    iree_byte_range_t byte_range = {
        .offset = phdr->p_vaddr,
        .length = phdr->p_memsz,
    };
    IREE_RETURN_IF_ERROR(iree_memory_view_protect_ranges(module->vaddr_bias, 1,
                                                         &byte_range, access));

    // Flush the instruction cache if we are going to execute these pages.
    if (access & IREE_MEMORY_ACCESS_EXECUTE) {
      iree_memory_flush_icache(module->vaddr_bias + phdr->p_vaddr,
                               phdr->p_memsz);
    }
  }

  // PT_GNU_RELRO: hardening of post-relocation segments.
  // These may alias with segments above and must be processed afterward.
  for (iree_elf_half_t i = 0; i < load_state->ehdr->e_phnum; ++i) {
    const iree_elf_phdr_t* phdr = &load_state->phdr_table[i];
    if (phdr->p_type != IREE_ELF_PT_GNU_RELRO) continue;
    iree_byte_range_t byte_range = {
        .offset = phdr->p_vaddr,
        .length = phdr->p_memsz,
    };
    IREE_RETURN_IF_ERROR(iree_memory_view_protect_ranges(
        module->vaddr_bias, 1, &byte_range, IREE_MEMORY_ACCESS_READ));
  }

  return iree_ok_status();
}

// Unloads the ELF segments from memory and releases the host virtual address
// space reservation.
static void iree_elf_module_unload_segments(iree_elf_module_t* module) {
  // Decommit/unreserve the entire memory space.
  if (module->vaddr_base != NULL) {
    iree_memory_view_release(module->vaddr_base, module->vaddr_size,
                             module->host_allocator);
  }
  module->vaddr_base = NULL;
  module->vaddr_bias = NULL;
  module->vaddr_size = 0;
}

//==============================================================================
// Dynamic library handling
//==============================================================================
// NOTE: this happens *after* allocation and loading as the .dynsym and related
// segments are allocated and loaded in virtual address space.

// Parses, verifies, and populates dynamic symbol related tables for runtime
// use. These tables are all in allocated memory and use fully rebased virtual
// addresses.
static iree_status_t iree_elf_module_parse_dynamic_tables(
    iree_elf_module_load_state_t* load_state, iree_elf_module_t* module) {
  // By the spec there must only be one PT_DYNAMIC.
  // Note that we are getting the one in the loaded virtual address space.
  const iree_elf_dyn_t* dyn_table = NULL;
  iree_host_size_t dyn_table_count = 0;
  for (iree_elf_half_t i = 0; i < load_state->ehdr->e_phnum; ++i) {
    const iree_elf_phdr_t* phdr = &load_state->phdr_table[i];
    if (phdr->p_type == IREE_ELF_PT_DYNAMIC) {
      dyn_table = (const iree_elf_dyn_t*)(module->vaddr_bias + phdr->p_vaddr);
      dyn_table_count = phdr->p_filesz / sizeof(iree_elf_dyn_t);
      break;
    }
  }
  if (!dyn_table || !dyn_table_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "no PT_DYNAMIC/.dynamic segment");
  }
  load_state->dyn_table = dyn_table;
  load_state->dyn_table_count = dyn_table_count;

  for (iree_host_size_t i = 0; i < dyn_table_count; ++i) {
    const iree_elf_dyn_t* dyn = &dyn_table[i];
    switch (dyn->d_tag) {
      case IREE_ELF_DT_STRTAB:
        // .dynstr table for runtime symbol lookup.
        module->dynstr = (const char*)(module->vaddr_bias + dyn->d_un.d_ptr);
        break;
      case IREE_ELF_DT_STRSZ:
        module->dynstr_size = dyn->d_un.d_val;
        break;

      case IREE_ELF_DT_SYMTAB:
        // .dynsym table for runtime symbol lookup.
        module->dynsym =
            (const iree_elf_sym_t*)(module->vaddr_bias + dyn->d_un.d_ptr);
        break;
      case IREE_ELF_DT_SYMENT:
        if (dyn->d_un.d_val != sizeof(iree_elf_sym_t)) {
          return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "DT_SYMENT size mismatch");
        }
        break;
      case IREE_ELF_DT_HASH: {
        // NOTE: we don't care about the hash table (yet), but it is the only
        // way to get the total symbol count.
        const iree_elf_word_t* hash =
            (const iree_elf_word_t*)(module->vaddr_bias + dyn->d_un.d_ptr);
        module->dynsym_count = hash[1];  // symbol count, obviously~
        break;
      }

      case IREE_ELF_DT_INIT:
        // .init initializer function (runs before .init_array).
        load_state->init = dyn->d_un.d_ptr;
        break;
      case IREE_ELF_DT_INIT_ARRAY:
        // .init_array list of initializer functions.
        load_state->init_array =
            (const iree_elf_addr_t*)(module->vaddr_bias + dyn->d_un.d_ptr);
        break;
      case IREE_ELF_DT_INIT_ARRAYSZ:
        load_state->init_array_count = dyn->d_un.d_val;
        break;

      case IREE_ELF_DT_RELENT:
        if (dyn->d_un.d_val != sizeof(iree_elf_rel_t)) {
          return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "DT_RELENT size mismatch");
        }
        break;
      case IREE_ELF_DT_RELAENT:
        if (dyn->d_un.d_val != sizeof(iree_elf_rela_t)) {
          return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "DT_RELAENT size mismatch");
        }
        break;

      default:
        // Ignored.
        break;
    }
  }

  // Must have .dynsym/.dynstr to perform lookups.
  if (!module->dynstr || !module->dynstr_size || !module->dynsym ||
      !module->dynsym_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "missing .dynsym/.dynstr in ELF .dynamic segment");
  }

  // NOTE: we could try to verify ranges here but no one seems to do that and
  // it's somewhat annoying. You're loading untrusted code into your memory
  // space - this is the least of your concerns :)

  return iree_ok_status();
}

// Verifies that there are no dynamic imports in the module as we don't support
// them yet.
static iree_status_t iree_elf_module_verify_no_imports(
    iree_elf_module_load_state_t* load_state, iree_elf_module_t* module) {
  // NOTE: slot 0 is always the 0 placeholder.
  for (iree_host_size_t i = 1; i < module->dynsym_count; ++i) {
    const iree_elf_sym_t* sym = &module->dynsym[i];
    if (sym->st_shndx == IREE_ELF_SHN_UNDEF) {
      const char* symname IREE_ATTRIBUTE_UNUSED =
          sym->st_name ? module->dynstr + sym->st_name : NULL;
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "ELF imports one or more symbols (trying "
                              "'%s'); imports are not supported in the "
                              "platform-agnostic loader",
                              symname);
    }
  }
  return iree_ok_status();
}

//==============================================================================
// Relocation
//==============================================================================

// Applies symbol and address base relocations to the loaded sections.
static iree_status_t iree_elf_module_apply_relocations(
    iree_elf_module_load_state_t* load_state, iree_elf_module_t* module) {
  // Redirect to the architecture-specific handler.
  iree_elf_relocation_state_t reloc_state;
  memset(&reloc_state, 0, sizeof(reloc_state));
  reloc_state.vaddr_bias = module->vaddr_bias;
  reloc_state.dyn_table = load_state->dyn_table;
  reloc_state.dyn_table_count = load_state->dyn_table_count;
  reloc_state.dynsym = module->dynsym;
  reloc_state.dynsym_count = module->dynsym_count;
  return iree_elf_arch_apply_relocations(&reloc_state);
}

//==============================================================================
// Initialization/finalization
//==============================================================================

// Runs initializers defined within the module, if any.
// .init is run first and then .init_array is run in array order.
static iree_status_t iree_elf_module_run_initializers(
    iree_elf_module_load_state_t* load_state, iree_elf_module_t* module) {
  if (load_state->init != IREE_ELF_ADDR_MIN) {
    iree_elf_call_v_v((void*)(module->vaddr_bias + load_state->init));
  }

  // NOTE: entries with values of 0 or -1 must be ignored.
  for (iree_host_size_t i = 0; i < load_state->init_array_count; ++i) {
    iree_elf_addr_t symbol_ptr = load_state->init_array[i];
    if (symbol_ptr == 0 || symbol_ptr == IREE_ELF_ADDR_MAX) continue;
    iree_elf_call_v_v((void*)(module->vaddr_bias + symbol_ptr));
  }

  return iree_ok_status();
}

static void iree_elf_module_run_finalizers(iree_elf_module_t* module) {
  // NOT IMPLEMENTED
  // Android doesn't do this for its loader and nothing we do should ever need
  // them: we're not doing IO or (hopefully) anything stateful inside of our
  // HAL executables that has correctness depend on them executing.
}

//==============================================================================
// Symbol lookup
//==============================================================================

// Resolves a global symbol within the module by symbol name.
// Currently we don't support any hashing as we have a single exported symbol
// and this is a simple linear scan.
//
// If we start to get a few dozen then it may be worth it to implement the sysv
// style as it is smallest both in code size and ELF binary size. This can be
// specified using --hash-style=sysv with ld/lld. By default most linkers
// (including lld, which is what we care about) will use
// --hash-style=both and emit both `.hash` and `.gnu.hash`, but that's silly for
// us as ideally we'd have none. If we ever try to use this for larger libraries
// with many exported symbols (we shouldn't!) we can add support:
// https://docs.oracle.com/cd/E23824_01/html/819-0690/chapter6-48031.html
// https://blogs.oracle.com/solaris/gnu-hash-elf-sections-v2
static const iree_elf_sym_t* iree_elf_module_lookup_global_symbol(
    iree_elf_module_t* module, const char* symbol_name) {
  // NOTE: symtab[0] is always STN_UNDEF so we skip it.
  // NOTE: symtab has local symbols before global ones and since we are looking
  // for global symbols we iterate in reverse.
  for (int i = (int)module->dynsym_count - 1; i > 0; i--) {
    const iree_elf_sym_t* sym = &module->dynsym[i];
    iree_elf_byte_t bind = IREE_ELF_ST_BIND(sym->st_info);
    if (bind != IREE_ELF_STB_GLOBAL && bind != IREE_ELF_STB_WEAK) continue;
    if (sym->st_name == 0) continue;
    if (strcmp(module->dynstr + sym->st_name, symbol_name) == 0) {
      return sym;
    }
  }
  return NULL;
}

//==============================================================================
// API
//==============================================================================

iree_status_t iree_elf_module_initialize_from_memory(
    iree_const_byte_span_t raw_data,
    const iree_elf_import_table_t* import_table,
    iree_allocator_t host_allocator, iree_elf_module_t* out_module) {
  IREE_ASSERT_ARGUMENT(raw_data.data);
  IREE_ASSERT_ARGUMENT(out_module);
  IREE_TRACE_ZONE_BEGIN(z0);

  // If the file is a FatELF then select the ELF for this architecture.
  // Ignored of not a FatELF and otherwise errors if no compatible architecture
  // is available.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_fatelf_select(raw_data, &raw_data));

  // Parse the ELF headers and verify that it's something we can handle.
  // Temporary state required during loading such as references to subtables
  // within the ELF are tracked here on the stack while persistent fields are
  // initialized on |out_module|.
  iree_elf_module_load_state_t load_state;
  iree_status_t status =
      iree_elf_module_parse_headers(raw_data, &load_state, out_module);
  out_module->host_allocator = host_allocator;

  // Allocate and load the ELF into memory.
  iree_memory_jit_context_begin();
  if (iree_status_is_ok(status)) {
    status = iree_elf_module_load_segments(raw_data, &load_state, out_module);
  }

  // Parse required dynamic symbol tables in loaded memory. These are used for
  // runtime symbol resolution and relocation.
  if (iree_status_is_ok(status)) {
    status = iree_elf_module_parse_dynamic_tables(&load_state, out_module);
  }

  // TODO(benvanik): imports would happen here. For now we just ensure there are
  // no imports as otherwise things will fail with obscure messages later on.
  if (iree_status_is_ok(status)) {
    status = iree_elf_module_verify_no_imports(&load_state, out_module);
  }

  // Apply relocations to the loaded pages.
  if (iree_status_is_ok(status)) {
    status = iree_elf_module_apply_relocations(&load_state, out_module);
  }

  // Apply final protections to the loaded pages now that relocations have been
  // performed.
  if (iree_status_is_ok(status)) {
    status = iree_elf_module_protect_segments(&load_state, out_module);
  }
  iree_memory_jit_context_end();

  // Run initializers prior to returning to the caller.
  if (iree_status_is_ok(status)) {
    status = iree_elf_module_run_initializers(&load_state, out_module);
  }

  if (!iree_status_is_ok(status)) {
    // On failure gracefully clean up the module by releasing any allocated
    // memory during the partial initialization.
    iree_elf_module_deinitialize(out_module);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_elf_module_deinitialize(iree_elf_module_t* module) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_elf_module_run_finalizers(module);
  iree_elf_module_unload_segments(module);
  memset(module, 0, sizeof(*module));

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_elf_module_lookup_export(iree_elf_module_t* module,
                                            const char* symbol_name,
                                            void** out_export) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(out_export);
  *out_export = NULL;

  const iree_elf_sym_t* sym =
      iree_elf_module_lookup_global_symbol(module, symbol_name);
  if (IREE_UNLIKELY(!sym)) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "exported symbol with name '%s' not found in module", symbol_name);
  }

  *out_export = module->vaddr_bias + sym->st_value;
  return iree_ok_status();
}
