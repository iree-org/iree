// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_ELF_ELF_TYPES_H_
#define IREE_HAL_LOCAL_ELF_ELF_TYPES_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/target_platform.h"

// This file contains the ELF data structures we use in our runtime linker and
// the definitions to support them. The structure definitions are taken from
// the System V ABI:
//   http://www.sco.com/developers/gabi/latest/contents.html
// LLVM's BinaryFormat ELF headers:
//   third_party/llvm-project/llvm/include/llvm/BinaryFormat/ELF.h
// And the Linux specification:
//   https://linux.die.net/man/5/elf
//   https://refspecs.linuxbase.org/LSB_3.1.1/LSB-Core-generic/LSB-Core-generic.html
// (among others)
//
// We define both 32-bit and 64-bit variants of the structures as we support
// both; however we only ever use one at a time based on the target
// configuration so that we are only including the code for the
// architecture-native integer width.
//
// We purposefully avoid inserting a large number of enums that we never use:
// this implementation is just to load our own compiled HAL executables and as
// such we control both the linker configuration used to produce the inputs we
// load.
//
// Code can generally be written using only the iree_elf_* types and IREE_ELF_*
// macros; if used consistently then only one source code definition is required
// and it'll get compiled into the appropriate form with no additional
// configuration.

typedef uint8_t iree_elf32_byte_t;
typedef uint32_t iree_elf32_addr_t;
typedef uint16_t iree_elf32_half_t;
typedef uint32_t iree_elf32_off_t;
typedef int32_t iree_elf32_sword_t;
typedef uint32_t iree_elf32_word_t;

typedef uint8_t iree_elf64_byte_t;
typedef uint64_t iree_elf64_addr_t;
typedef uint16_t iree_elf64_half_t;
typedef uint64_t iree_elf64_off_t;
typedef int32_t iree_elf64_sword_t;
typedef uint32_t iree_elf64_word_t;
typedef uint64_t iree_elf64_xword_t;
typedef int64_t iree_elf64_sxword_t;

enum {
  IREE_ELF_EI_CLASS = 4,       // IREE_ELF_ELFCLASS*
  IREE_ELF_EI_DATA = 5,        // IREE_ELF_ELFDATA*
  IREE_ELF_EI_VERSION = 6,     // File version (1 expected)
  IREE_ELF_EI_OSABI = 7,       // Operating system/ABI identification
  IREE_ELF_EI_ABIVERSION = 8,  // ABI version
  IREE_ELF_EI_PAD = 9,         // Start of padding bytes
  IREE_ELF_EI_NIDENT = 16,     // Size of e_ident[]
};

enum {
  IREE_ELF_ELFCLASSNONE = 0,  // Invalid class
  IREE_ELF_ELFCLASS32 = 1,    // 32-bit objects
  IREE_ELF_ELFCLASS64 = 2,    // 64-bit objects
};

enum {
  IREE_ELF_ELFDATANONE = 0,  // Invalid data encoding
  IREE_ELF_ELFDATA2LSB = 1,  // Little-endian
  IREE_ELF_ELFDATA2MSB = 2,  // Big-endian
};

enum {
  IREE_ELF_ELFOSABI_NONE = 0,          // No extensions or unspecified
  IREE_ELF_ELFOSABI_LINUX = 3,         // Linux
  IREE_ELF_ELFOSABI_STANDALONE = 255,  // Standalone
};

enum {
  IREE_ELF_ET_NONE = 0,  // No file type
  IREE_ELF_ET_REL = 1,   // Relocatable file
  IREE_ELF_ET_EXEC = 2,  // Executable file
  IREE_ELF_ET_DYN = 3,   // Shared object file
  IREE_ELF_ET_CORE = 4,  // Core file
};

typedef struct {
  iree_elf32_byte_t e_ident[IREE_ELF_EI_NIDENT];
  iree_elf32_half_t e_type;  // IREE_ELF_ET_*
  iree_elf32_half_t e_machine;
  iree_elf32_word_t e_version;
  iree_elf32_addr_t e_entry;
  iree_elf32_off_t e_phoff;
  iree_elf32_off_t e_shoff;
  iree_elf32_word_t e_flags;
  iree_elf32_half_t e_ehsize;
  iree_elf32_half_t e_phentsize;
  iree_elf32_half_t e_phnum;
  iree_elf32_half_t e_shentsize;
  iree_elf32_half_t e_shnum;
  iree_elf32_half_t e_shstrndx;
} iree_elf32_ehdr_t;

typedef struct {
  iree_elf64_byte_t e_ident[IREE_ELF_EI_NIDENT];
  iree_elf64_half_t e_type;  // IREE_ELF_ET_*
  iree_elf64_half_t e_machine;
  iree_elf64_word_t e_version;
  iree_elf64_addr_t e_entry;
  iree_elf64_off_t e_phoff;
  iree_elf64_off_t e_shoff;
  iree_elf64_word_t e_flags;
  iree_elf64_half_t e_ehsize;
  iree_elf64_half_t e_phentsize;
  iree_elf64_half_t e_phnum;
  iree_elf64_half_t e_shentsize;
  iree_elf64_half_t e_shnum;
  iree_elf64_half_t e_shstrndx;
} iree_elf64_ehdr_t;

enum {
  IREE_ELF_PT_NULL = 0,
  IREE_ELF_PT_LOAD = 1,
  IREE_ELF_PT_DYNAMIC = 2,
  IREE_ELF_PT_INTERP = 3,
  IREE_ELF_PT_NOTE = 4,
  IREE_ELF_PT_SHLIB = 5,
  IREE_ELF_PT_PHDR = 6,
  IREE_ELF_PT_GNU_RELRO = 0x6474e552,
};

enum {
  IREE_ELF_PF_X = 0x1,  // Execute
  IREE_ELF_PF_W = 0x2,  // Write
  IREE_ELF_PF_R = 0x4,  // Read
};

typedef struct {
  iree_elf32_word_t p_type;  // IREE_ELF_PT_*
  iree_elf32_off_t p_offset;
  iree_elf32_addr_t p_vaddr;
  iree_elf32_addr_t p_paddr;
  iree_elf32_word_t p_filesz;
  iree_elf32_word_t p_memsz;
  iree_elf32_word_t p_flags;  // IREE_ELF_PF_*
  iree_elf32_word_t p_align;
} iree_elf32_phdr_t;

typedef struct {
  iree_elf64_word_t p_type;   // IREE_ELF_PT_*
  iree_elf64_word_t p_flags;  // IREE_ELF_PF_*
  iree_elf64_off_t p_offset;
  iree_elf64_addr_t p_vaddr;
  iree_elf64_addr_t p_paddr;
  iree_elf64_xword_t p_filesz;
  iree_elf64_xword_t p_memsz;
  iree_elf64_xword_t p_align;
} iree_elf64_phdr_t;

// An undefined, missing, irrelevant, or otherwise meaningless section ref.
#define IREE_ELF_SHN_UNDEF 0

enum {
  IREE_ELF_SHT_NULL = 0,
  IREE_ELF_SHT_PROGBITS = 1,
  IREE_ELF_SHT_SYMTAB = 2,
  IREE_ELF_SHT_STRTAB = 3,
  IREE_ELF_SHT_RELA = 4,
  IREE_ELF_SHT_HASH = 5,
  IREE_ELF_SHT_DYNAMIC = 6,
  IREE_ELF_SHT_NOTE = 7,
  IREE_ELF_SHT_NOBITS = 8,
  IREE_ELF_SHT_REL = 9,
  IREE_ELF_SHT_SHLIB = 10,
  IREE_ELF_SHT_DYNSYM = 11,
};

enum {
  IREE_ELF_SHF_WRITE = 0x1,
  IREE_ELF_SHF_ALLOC = 0x2,
  IREE_ELF_SHF_EXECINSTR = 0x4,
  IREE_ELF_SHF_MERGE = 0x10,
  IREE_ELF_SHF_STRINGS = 0x20,
  IREE_ELF_SHF_INFO_LINK = 0x40,
  IREE_ELF_SHF_LINK_ORDER = 0x80,
  IREE_ELF_SHF_OS_NONCONFORMING = 0x100,
  IREE_ELF_SHF_GROUP = 0x200
};

typedef struct {
  iree_elf32_word_t sh_name;
  iree_elf32_word_t sh_type;   // IREE_ELF_SHT_*
  iree_elf32_word_t sh_flags;  // IREE_ELF_SHF_*
  iree_elf32_addr_t sh_addr;
  iree_elf32_off_t sh_offset;
  iree_elf32_word_t sh_size;
  iree_elf32_word_t sh_link;
  iree_elf32_word_t sh_info;
  iree_elf32_word_t sh_addralign;
  iree_elf32_word_t sh_entsize;
} iree_elf32_shdr_t;

typedef struct {
  iree_elf64_word_t sh_name;
  iree_elf64_word_t sh_type;    // IREE_ELF_SHT_*
  iree_elf64_xword_t sh_flags;  // IREE_ELF_SHF_*
  iree_elf64_addr_t sh_addr;
  iree_elf64_off_t sh_offset;
  iree_elf64_xword_t sh_size;
  iree_elf64_word_t sh_link;
  iree_elf64_word_t sh_info;
  iree_elf64_xword_t sh_addralign;
  iree_elf64_xword_t sh_entsize;
} iree_elf64_shdr_t;

typedef struct {
  iree_elf32_word_t n_namesz;
  iree_elf32_word_t n_descsz;
  iree_elf32_word_t n_type;
} iree_elf32_nhdr_t;

typedef struct {
  iree_elf64_word_t n_namesz;
  iree_elf64_word_t n_descsz;
  iree_elf64_word_t n_type;
} iree_elf64_nhdr_t;

#define IREE_ELF_ST_INFO(bind, type) (((bind) << 4) + ((type)&0xF))

#define IREE_ELF_ST_TYPE(info) ((info)&0xF)
enum {
  IREE_ELF_STT_NOTYPE = 0,
  IREE_ELF_STT_OBJECT = 1,
  IREE_ELF_STT_FUNC = 2,
  IREE_ELF_STT_SECTION = 3,
  IREE_ELF_STT_FILE = 4,
  IREE_ELF_STT_COMMON = 5,
};

#define IREE_ELF_ST_BIND(info) ((info) >> 4)
enum {
  IREE_ELF_STB_LOCAL = 0,   // Local symbol.
  IREE_ELF_STB_GLOBAL = 1,  // Global symbol (export).
  IREE_ELF_STB_WEAK = 2,    // Weak symbol (somewhat like global).
};

#define IREE_ELF_ST_VISIBILITY(o) ((o)&0x3)
enum {
  IREE_ELF_STV_DEFAULT = 0,
  IREE_ELF_STV_INTERNAL = 1,
  IREE_ELF_STV_HIDDEN = 2,
  IREE_ELF_STV_PROTECTED = 3,
};

typedef struct {
  iree_elf32_word_t st_name;
  iree_elf32_addr_t st_value;
  iree_elf32_word_t st_size;
  iree_elf32_byte_t st_info;
  iree_elf32_byte_t st_other;
  iree_elf32_half_t st_shndx;
} iree_elf32_sym_t;

typedef struct {
  iree_elf64_word_t st_name;
  iree_elf64_byte_t st_info;
  iree_elf64_byte_t st_other;
  iree_elf64_half_t st_shndx;
  iree_elf64_addr_t st_value;
  iree_elf64_xword_t st_size;
} iree_elf64_sym_t;

enum {
  IREE_ELF_DT_NULL = 0,                   // (no data)
  IREE_ELF_DT_NEEDED = 1,                 // d_val
  IREE_ELF_DT_PLTRELSZ = 2,               // d_val
  IREE_ELF_DT_PLTGOT = 3,                 // d_ptr
  IREE_ELF_DT_HASH = 4,                   // d_ptr
  IREE_ELF_DT_STRTAB = 5,                 // d_ptr
  IREE_ELF_DT_SYMTAB = 6,                 // d_ptr
  IREE_ELF_DT_RELA = 7,                   // d_ptr
  IREE_ELF_DT_RELASZ = 8,                 // d_val
  IREE_ELF_DT_RELAENT = 9,                // d_val
  IREE_ELF_DT_STRSZ = 10,                 // d_val
  IREE_ELF_DT_SYMENT = 11,                // d_val
  IREE_ELF_DT_INIT = 12,                  // d_ptr
  IREE_ELF_DT_FINI = 13,                  // d_ptr
  IREE_ELF_DT_SONAME = 14,                // d_val
  IREE_ELF_DT_RPATH = 15,                 // d_val
  IREE_ELF_DT_SYMBOLIC = 16,              // (no data)
  IREE_ELF_DT_REL = 17,                   // d_ptr
  IREE_ELF_DT_RELSZ = 18,                 // d_val
  IREE_ELF_DT_RELENT = 19,                // d_val
  IREE_ELF_DT_PLTREL = 20,                // d_val
  IREE_ELF_DT_TEXTREL = 22,               // (no data)
  IREE_ELF_DT_JMPREL = 23,                // d_ptr
  IREE_ELF_DT_BIND_NOW = 24,              // (no data)
  IREE_ELF_DT_INIT_ARRAY = 25,            // d_ptr
  IREE_ELF_DT_FINI_ARRAY = 26,            // d_ptr
  IREE_ELF_DT_INIT_ARRAYSZ = 27,          // d_val
  IREE_ELF_DT_FINI_ARRAYSZ = 28,          // d_val
  IREE_ELF_DT_RUNPATH = 29,               // d_val
  IREE_ELF_DT_FLAGS = 30,                 // d_val
  IREE_ELF_DT_SUNW_RTLDINF = 0x6000000e,  // d_ptr
  IREE_ELF_DT_CHECKSUM = 0x6ffffdf8,      // d_val
  IREE_ELF_DT_PLTPADSZ = 0x6ffffdf9,      // d_val
  IREE_ELF_DT_MOVEENT = 0x6ffffdfa,       // d_val
  IREE_ELF_DT_MOVESZ = 0x6ffffdfb,        // d_val
  IREE_ELF_DT_FEATURE_1 = 0x6ffffdfc,     // d_val
  IREE_ELF_DT_POSFLAG_1 = 0x6ffffdfd,     // d_val
  IREE_ELF_DT_SYMINSZ = 0x6ffffdfe,       // d_val
  IREE_ELF_DT_SYMINENT = 0x6ffffdff,      // d_val
  IREE_ELF_DT_CONFIG = 0x6ffffefa,        // d_ptr
  IREE_ELF_DT_DEPAUDIT = 0x6ffffefb,      // d_ptr
  IREE_ELF_DT_AUDIT = 0x6ffffefc,         // d_ptr
  IREE_ELF_DT_PLTPAD = 0x6ffffefd,        // d_ptr
  IREE_ELF_DT_MOVETAB = 0x6ffffefe,       // d_ptr
  IREE_ELF_DT_SYMINFO = 0x6ffffeff,       // d_ptr
  IREE_ELF_DT_RELACOUNT = 0x6ffffff9,     // d_val
  IREE_ELF_DT_RELCOUNT = 0x6ffffffa,      // d_val
  IREE_ELF_DT_FLAGS_1 = 0x6ffffffb,       // d_val
  IREE_ELF_DT_VERDEF = 0x6ffffffc,        // d_ptr
  IREE_ELF_DT_VERDEFNUM = 0x6ffffffd,     // d_val
  IREE_ELF_DT_VERNEED = 0x6ffffffe,       // d_ptr
  IREE_ELF_DT_VERNEEDNUM = 0x6fffffff,    // d_val
  IREE_ELF_DT_AUXILIARY = 0x7ffffffd,     // d_val
  IREE_ELF_DT_USED = 0x7ffffffe,          // d_val
};

typedef struct {
  iree_elf32_sword_t d_tag;  // IREE_ELF_DT_*
  union {
    iree_elf32_sword_t d_val;
    iree_elf32_addr_t d_ptr;
  } d_un;
} iree_elf32_dyn_t;

typedef struct {
  iree_elf64_sxword_t d_tag;  // IREE_ELF_DT_*
  union {
    iree_elf64_xword_t d_val;
    iree_elf64_addr_t d_ptr;
  } d_un;
} iree_elf64_dyn_t;

typedef struct {
  iree_elf32_addr_t r_offset;
  iree_elf32_word_t r_info;
} iree_elf32_rel_t;

typedef struct {
  iree_elf64_addr_t r_offset;
  iree_elf64_xword_t r_info;
} iree_elf64_rel_t;

typedef struct {
  iree_elf32_addr_t r_offset;
  iree_elf32_word_t r_info;
  iree_elf32_sword_t r_addend;
} iree_elf32_rela_t;

typedef struct {
  iree_elf64_addr_t r_offset;
  iree_elf64_xword_t r_info;
  iree_elf64_sxword_t r_addend;
} iree_elf64_rela_t;

#if defined(IREE_PTR_SIZE_32)

#define IREE_ELF_ADDR_MIN 0u
#define IREE_ELF_ADDR_MAX UINT32_MAX

typedef iree_elf32_byte_t iree_elf_byte_t;
typedef iree_elf32_addr_t iree_elf_addr_t;
typedef iree_elf32_half_t iree_elf_half_t;
typedef iree_elf32_off_t iree_elf_off_t;
typedef iree_elf32_sword_t iree_elf_sword_t;
typedef iree_elf32_word_t iree_elf_word_t;

typedef iree_elf32_dyn_t iree_elf_dyn_t;
typedef iree_elf32_rel_t iree_elf_rel_t;
typedef iree_elf32_rela_t iree_elf_rela_t;
typedef iree_elf32_sym_t iree_elf_sym_t;
typedef iree_elf32_ehdr_t iree_elf_ehdr_t;
typedef iree_elf32_phdr_t iree_elf_phdr_t;
typedef iree_elf32_shdr_t iree_elf_shdr_t;
typedef iree_elf32_nhdr_t iree_elf_nhdr_t;

#define IREE_ELF_R_SYM(x) ((x) >> 8)
#define IREE_ELF_R_TYPE(x) ((x)&0xFF)

#elif defined(IREE_PTR_SIZE_64)

#define IREE_ELF_ADDR_MIN 0ull
#define IREE_ELF_ADDR_MAX UINT64_MAX

typedef iree_elf64_byte_t iree_elf_byte_t;
typedef iree_elf64_addr_t iree_elf_addr_t;
typedef iree_elf64_half_t iree_elf_half_t;
typedef iree_elf64_off_t iree_elf_off_t;
typedef iree_elf64_sword_t iree_elf_sword_t;
typedef iree_elf64_word_t iree_elf_word_t;

typedef iree_elf64_dyn_t iree_elf_dyn_t;
typedef iree_elf64_rel_t iree_elf_rel_t;
typedef iree_elf64_rela_t iree_elf_rela_t;
typedef iree_elf64_sym_t iree_elf_sym_t;
typedef iree_elf64_ehdr_t iree_elf_ehdr_t;
typedef iree_elf64_phdr_t iree_elf_phdr_t;
typedef iree_elf64_shdr_t iree_elf_shdr_t;
typedef iree_elf64_nhdr_t iree_elf_nhdr_t;

#define IREE_ELF_R_SYM(i) ((i) >> 32)
#define IREE_ELF_R_TYPE(i) ((i)&0xFFFFFFFF)

#else
#error "unsupported ELF N size (only 32/64-bits are defined)"
#endif  // IREE_PTR_SIZE_*

#endif  // IREE_HAL_LOCAL_ELF_ELF_TYPES_H_
