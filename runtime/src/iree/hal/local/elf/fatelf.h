// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_ELF_FATELF_H_
#define IREE_HAL_LOCAL_ELF_FATELF_H_

#include "iree/hal/local/elf/elf_types.h"

// This file contains the basic headers and types used in FatELF.
// https://icculus.org/fatelf/
// https://github.com/icculus/fatelf
//
// The textual specification describes the format:
// https://raw.githubusercontent.com/icculus/fatelf/main/docs/fatelf-specification.txt
//
// FatELF was going for acceptance into the Linux kernel but got rejected. It'll
// likely never be deployed anywhere and is mostly dead now. It's unfortunate as
// what it solves (fat binaries) is super interesting especially for deployment
// focused uses like ours but the prevailing opinion seemed to be that Linux
// people just build their stuff from source and don't need it ;(
//
// Though dead we use it because it's simple and exactly what it needs to be
// for our uses. We'll still have our own tools for working with them but they
// should be compatible with the original fatelf-* tools.
//
// To create a FatELF file from several ELFs:
//   iree-fatelf join elf_a.so elf_b.so elf_c.so > fatelf.sos
// To extract all ELFs from a FatELF file:
//   iree-fatelf split fatelf.sos
//
// WARNING: though there is overlap with what some of the fields represent
// (like little/big endian, etc) FatELF enum values can differ. The equivalent
// ELF fields/enums have been documented but always use the values in this file.

// Little-endian magic bytes used to identify FatELF files.
#define IREE_FATELF_MAGIC 0x1F0E70FA  // FA700E1F 'fat' 'elf' lol

// Only version 1 is defined. We may end up with our own versions if we diverge.
// FatELF doesn't have any architectural feature requirement bits which makes it
// difficult to support specialized ELFs based on things like instruction set
// support and we may want to add that.
#define IREE_FATELF_FORMAT_VERSION 1

enum {
  IREE_FATELF_WORD_SIZE_32 = 1,  // IREE_ELF_ELFCLASS32
  IREE_FATELF_WORD_SIZE_64 = 2,  // IREE_ELF_ELFCLASS64
};

enum {
  IREE_FATELF_BYTE_ORDER_MSB = 0,  // IREE_ELF_ELFDATA2MSB - big-endian
  IREE_FATELF_BYTE_ORDER_LSB = 1,  // IREE_ELF_ELFDATA2LSB - little-endian
};

// An individual record in the FatELF record table.
// This has some of the fields from the iree_elf_ehdr_t and references a header-
// relative file range of where the corresponding ELF file can be found.
typedef struct {
  iree_elf64_half_t machine;        // e_machine
  iree_elf64_byte_t osabi;          // e_ident[EI_OSABI]
  iree_elf64_byte_t osabi_version;  // e_ident[EI_ABIVERSION]
  iree_elf64_byte_t word_size;      // e_ident[EI_CLASS]
  iree_elf64_byte_t byte_order;     // e_ident[EI_DATA]
  iree_elf64_byte_t reserved0;
  iree_elf64_byte_t reserved1;
  iree_elf64_off_t offset;
  iree_elf64_xword_t size;
} iree_fatelf_record_t;
static_assert(sizeof(iree_fatelf_record_t) == 24, "must be packed");

// File header for FatELF files, starting at byte 0.
typedef struct {
  iree_elf64_word_t magic;    // IREE_FATELF_MAGIC
  iree_elf64_half_t version;  // IREE_FATELF_FORMAT_VERSION
  iree_elf64_byte_t record_count;
  iree_elf64_byte_t reserved;
  iree_fatelf_record_t records[0];  // record_count trailing records
} iree_fatelf_header_t;
static_assert(sizeof(iree_fatelf_header_t) == 8, "must be packed");

// Scans |file_data| for a FatELF header and if present selects the matching ELF
// for the current system if available.
// Upon return |out_elf_data| will either be the entire file if no FatELF header
// was found or just the bytes of the selected ELF.
iree_status_t iree_fatelf_select(iree_const_byte_span_t file_data,
                                 iree_const_byte_span_t* out_elf_data);

#endif  // IREE_HAL_LOCAL_ELF_FATELF_H_
