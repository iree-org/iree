// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/executable_format.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Writes an executable format string with optional prefix as `[prefix]suffix`
// to |buffer| with |capacity| bytes, including NUL terminator. No-op if no
// buffer is provided.
//
// Example: prefix = "foo", suffix = "bar" => "foobar\0"
static iree_status_t iree_hal_executable_write_executable_format(
    const char* prefix, const char* suffix, char* buffer,
    iree_host_size_t capacity) {
  if (!capacity || !buffer) return iree_ok_status();  // no-op
  const iree_host_size_t prefix_length = prefix ? strlen(prefix) : 0;
  const iree_host_size_t suffix_length = strlen(suffix);
  const iree_host_size_t format_length =
      prefix_length + suffix_length + /*NUL*/ 1;
  if (capacity < format_length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "insufficient capacity for format string; need %" PRIhsz
        " but have %" PRIhsz,
        format_length, capacity);
  }
  if (prefix_length > 0) {
    memcpy(buffer, prefix, prefix_length);
  }
  memcpy(buffer + prefix_length, suffix, suffix_length + /*NUL*/ 1);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// ELF
//===----------------------------------------------------------------------===//

// ELF magic bytes: 0x7F 'E' 'L' 'F'.
static const uint8_t kElfMagic[4] = {0x7F, 'E', 'L', 'F'};

typedef enum {
  IREE_ELF_EM_386 = 0x03,      // Intel 80386
  IREE_ELF_EM_ARM = 0x28,      // ARM
  IREE_ELF_EM_X86_64 = 0x3E,   // AMD x86-64 architecture
  IREE_ELF_EM_AARCH64 = 0xB7,  // ARM AARCH64
  IREE_ELF_EM_RISCV = 0xF3,    // RISC-V
} iree_elf_machine_t;

typedef struct {
  uint8_t e_ident[16];
  uint16_t e_type;
  uint16_t e_machine;
  uint32_t e_version;
} iree_elf_ehdr_common_t;

typedef struct {
  iree_elf_ehdr_common_t common;
  uint32_t e_entry;
  uint32_t e_phoff;
  uint32_t e_shoff;
  uint32_t e_flags;
  uint16_t e_ehsize;
  uint16_t e_phentsize;
  uint16_t e_phnum;
  uint16_t e_shentsize;
  uint16_t e_shnum;
  uint16_t e_shstrndx;
} iree_elf32_ehdr_t;

typedef struct {
  iree_elf_ehdr_common_t common;
  uint64_t e_entry;
  uint64_t e_phoff;
  uint64_t e_shoff;
  uint32_t e_flags;
  uint16_t e_ehsize;
  uint16_t e_phentsize;
  uint16_t e_phnum;
  uint16_t e_shentsize;
  uint16_t e_shnum;
  uint16_t e_shstrndx;
} iree_elf64_ehdr_t;

typedef struct {
  uint32_t sh_name;
  uint32_t sh_type;
  uint32_t sh_flags;
  uint32_t sh_addr;
  uint32_t sh_offset;
  uint32_t sh_size;
} iree_elf32_shdr_t;

typedef struct {
  uint32_t sh_name;
  uint32_t sh_type;
  uint64_t sh_flags;
  uint64_t sh_addr;
  uint64_t sh_offset;
  uint64_t sh_size;
} iree_elf64_shdr_t;

static bool iree_hal_executable_is_elf_file(
    iree_const_byte_span_t executable_data) {
  const bool size_unknown = executable_data.data_length == 0;
  if (!size_unknown && executable_data.data_length < sizeof(kElfMagic)) {
    return false;  // too small
  }
  return memcmp(executable_data.data, kElfMagic, sizeof(kElfMagic)) == 0;
}

// Finds the maximum extent by checking section headers (32-bit ELF).
static iree_status_t iree_hal_executable_calculate_elf_size_32(
    const iree_elf32_ehdr_t* ehdr, iree_const_byte_span_t executable_data,
    iree_host_size_t* out_size) {
  const bool size_unknown = executable_data.data_length == 0;
  uint32_t max_offset = sizeof(*ehdr);

  // Check program header table.
  if (ehdr->e_phoff != 0) {
    const uint32_t ph_end = ehdr->e_phoff + (ehdr->e_phnum * ehdr->e_phentsize);
    if (ph_end > max_offset) {
      max_offset = ph_end;  // bump extent
    }
  }

  // Check section header table and sections.
  if (ehdr->e_shoff != 0) {
    const uint32_t sh_table_end =
        ehdr->e_shoff + (ehdr->e_shnum * ehdr->e_shentsize);
    if (sh_table_end > max_offset) {
      max_offset = sh_table_end;  // bump extent
    }
    if (size_unknown || executable_data.data_length >= sh_table_end) {
      const uint8_t* sh_table = executable_data.data + ehdr->e_shoff;
      for (uint16_t i = 0; i < ehdr->e_shnum; ++i) {
        const iree_elf32_shdr_t* shdr =
            (const iree_elf32_shdr_t*)(sh_table + i * ehdr->e_shentsize);
        uint32_t section_end = shdr->sh_offset + shdr->sh_size;
        if (section_end > max_offset) {
          max_offset = section_end;  // bump extent
        }
      }
    }
  }

  *out_size = (iree_host_size_t)max_offset;
  return iree_ok_status();
}

// Finds the maximum extent by checking section headers (64-bit ELF).
static iree_status_t iree_hal_executable_calculate_elf_size_64(
    const iree_elf64_ehdr_t* ehdr, iree_const_byte_span_t executable_data,
    iree_host_size_t* out_size) {
  const bool size_unknown = executable_data.data_length == 0;
  uint64_t max_offset = sizeof(*ehdr);

  // Check program header table.
  if (ehdr->e_phoff != 0) {
    const uint64_t ph_end = ehdr->e_phoff + (ehdr->e_phnum * ehdr->e_phentsize);
    if (ph_end > max_offset) {
      max_offset = ph_end;  // bump extent
    }
  }

  // Check section header table and sections.
  if (ehdr->e_shoff != 0) {
    const uint64_t sh_table_end =
        ehdr->e_shoff + (ehdr->e_shnum * ehdr->e_shentsize);
    if (sh_table_end > max_offset) {
      max_offset = sh_table_end;  // bump extent
    }
    if (size_unknown || executable_data.data_length >= sh_table_end) {
      const uint8_t* sh_table = executable_data.data + ehdr->e_shoff;
      for (uint16_t i = 0; i < ehdr->e_shnum; ++i) {
        const iree_elf64_shdr_t* shdr =
            (const iree_elf64_shdr_t*)(sh_table + i * ehdr->e_shentsize);
        const uint64_t section_end = shdr->sh_offset + shdr->sh_size;
        if (section_end > max_offset) {
          max_offset = section_end;  // bump extent
        }
      }
    }
  }

  *out_size = (iree_host_size_t)max_offset;
  return iree_ok_status();
}

static iree_status_t iree_hal_executable_calculate_elf_size(
    const iree_elf_ehdr_common_t* ehdr, iree_const_byte_span_t executable_data,
    iree_host_size_t* out_size) {
  // Check ELF class (32-bit vs 64-bit) as they use different structures and
  // we need to interpret the contents (beyond common) differently.
  const bool is_64bit = ehdr->e_ident[4] == 2;  // EI_CLASS = ELFCLASS64
  if (is_64bit) {
    return iree_hal_executable_calculate_elf_size_64(
        (const iree_elf64_ehdr_t*)ehdr, executable_data, out_size);
  } else {
    return iree_hal_executable_calculate_elf_size_32(
        (const iree_elf32_ehdr_t*)ehdr, executable_data, out_size);
  }
}

// Returns strings like "x86_64", "arm_64", "riscv_32", "unknown".
static const char* iree_hal_executable_infer_elf_format_arch(
    const iree_elf_ehdr_common_t* ehdr) {
  const uint16_t machine = ehdr->e_machine;
  switch (machine) {
    case IREE_ELF_EM_X86_64:
      return "x86_64";
    case IREE_ELF_EM_386:
      return "x86_32";
    case IREE_ELF_EM_AARCH64:
      return "arm_64";
    case IREE_ELF_EM_ARM:
      return "arm_32";
    case IREE_ELF_EM_RISCV:
      if (ehdr->e_ident[4] == 1) {  // ELFCLASS32
        return "riscv_32";
      } else {  // ELFCLASS64
        return "riscv_64";
      }
    default:
      return "unknown";
  }
}

//===----------------------------------------------------------------------===//
// FatELF
//===----------------------------------------------------------------------===//

// FatELF magic bytes: 0xFA 0x70 0x0E 0x1F (little-endian).
static const uint8_t kFatElfMagic[4] = {0xFA, 0x70, 0x0E, 0x1F};

typedef struct {
  uint16_t machine;
  uint8_t osabi;
  uint8_t osabi_version;
  uint8_t word_size;
  uint8_t byte_order;
  uint8_t reserved0;
  uint8_t reserved1;
  uint64_t offset;
  uint64_t size;
} iree_fatelf_record_t;

typedef struct {
  uint32_t magic;
  uint16_t version;
  uint8_t record_count;
  uint8_t reserved;
} iree_fatelf_header_t;

static bool iree_hal_executable_is_fatelf_file(
    iree_const_byte_span_t executable_data) {
  const bool size_unknown = executable_data.data_length == 0;
  if (!size_unknown && executable_data.data_length < sizeof(kFatElfMagic)) {
    return false;  // too small
  }
  return memcmp(executable_data.data, kFatElfMagic, sizeof(kFatElfMagic)) == 0;
}

static iree_status_t iree_hal_executable_calculate_fatelf_size(
    iree_const_byte_span_t executable_data, iree_host_size_t* out_size) {
  const bool size_unknown = executable_data.data_length == 0;
  if (!size_unknown &&
      executable_data.data_length < sizeof(iree_fatelf_header_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "FatELF data too small for header");
  }
  const iree_fatelf_header_t* header =
      (const iree_fatelf_header_t*)executable_data.data;

  // Calculate the size needed for header and records.
  uint64_t max_offset =
      sizeof(*header) + header->record_count * sizeof(iree_fatelf_record_t);

  // Check each record to find the maximum extent.
  const iree_fatelf_record_t* records =
      (const iree_fatelf_record_t*)((const uint8_t*)header + sizeof(*header));
  for (uint8_t i = 0; i < header->record_count; ++i) {
    const uint64_t record_end = records[i].offset + records[i].size;
    if (record_end > max_offset) {
      max_offset = record_end;  // bump extent
    }
  }

  *out_size = (iree_host_size_t)max_offset;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// ELF/FatELF
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_executable_infer_elf_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  IREE_ASSERT_ARGUMENT(!executable_format_capacity || executable_format);
  IREE_ASSERT_ARGUMENT(out_inferred_size);
  *out_inferred_size = 0;

  // WARNING: when data_length is 0 we assume we can read at least 4 bytes
  // for magic number detection. This is UNSAFE and may cause access violations
  // if the data is not actually at least 4 bytes.
  const bool size_unknown = executable_data.data_length == 0;

  // If size is known verify we have enough for magic checks.
  if (!size_unknown && executable_data.data_length < 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable data too small to identify format");
  }

  // When size is unknown we create a temporary span with max size for the
  // format detection and size calculation functions.
  iree_const_byte_span_t detection_data = executable_data;
  if (size_unknown) {
    detection_data.data_length = IREE_HOST_SIZE_MAX;
  }

  if ((size_unknown || executable_data.data_length >= sizeof(kFatElfMagic)) &&
      memcmp(executable_data.data, kFatElfMagic, sizeof(kFatElfMagic)) == 0) {
    if (size_unknown) {
      IREE_RETURN_IF_ERROR(iree_hal_executable_calculate_fatelf_size(
                               detection_data, out_inferred_size),
                           "calculating FatELF size");
    } else {
      *out_inferred_size = executable_data.data_length;
    }
    return iree_hal_executable_write_executable_format(
        "embedded-", "fatelf", executable_format, executable_format_capacity);
  } else if ((size_unknown ||
              executable_data.data_length >= sizeof(kElfMagic)) &&
             memcmp(executable_data.data, kElfMagic, sizeof(kElfMagic)) == 0) {
    const iree_elf_ehdr_common_t* ehdr =
        (const iree_elf_ehdr_common_t*)executable_data.data;
    if (size_unknown) {
      IREE_RETURN_IF_ERROR(iree_hal_executable_calculate_elf_size(
                               ehdr, detection_data, out_inferred_size),
                           "calculating ELF size");
    } else {
      *out_inferred_size = executable_data.data_length;
    }
    const char* arch_suffix = iree_hal_executable_infer_elf_format_arch(ehdr);
    return iree_hal_executable_write_executable_format(
        "embedded-elf-", arch_suffix, executable_format,
        executable_format_capacity);
  }

  return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                          "executable format is not ELF or FatELF");
}

//===----------------------------------------------------------------------===//
// DLL
//===----------------------------------------------------------------------===//

// PE/DLL magic bytes: 'M' 'Z' (DOS header).
static const uint8_t kPeMagic[2] = {'M', 'Z'};

// PE machine types.
#define IREE_PE_IMAGE_FILE_MACHINE_I386 0x014C   // x86
#define IREE_PE_IMAGE_FILE_MACHINE_AMD64 0x8664  // x86_64
#define IREE_PE_IMAGE_FILE_MACHINE_ARMNT 0x01C4  // ARM
#define IREE_PE_IMAGE_FILE_MACHINE_ARM64 0xAA64  // ARM64

typedef struct {
  uint8_t e_magic[2];   // 'M' 'Z'
  uint8_t padding[58];  // skip to e_lfanew
  uint32_t e_lfanew;    // offset to PE header
} iree_dos_header_t;

typedef struct {
  uint32_t signature;  // "PE\0\0"
  uint16_t machine;
  uint16_t numberOfSections;
  uint32_t timeDateStamp;
  uint32_t pointerToSymbolTable;
  uint32_t numberOfSymbols;
  uint16_t sizeOfOptionalHeader;
  uint16_t characteristics;
} iree_pe_file_header_t;

typedef struct {
  uint8_t name[8];
  uint32_t virtualSize;
  uint32_t virtualAddress;
  uint32_t sizeOfRawData;
  uint32_t pointerToRawData;
  uint32_t pointerToRelocations;
  uint32_t pointerToLinenumbers;
  uint16_t numberOfRelocations;
  uint16_t numberOfLinenumbers;
  uint32_t characteristics;
} iree_pe_section_header_t;

static bool iree_hal_executable_is_dll_file(
    iree_const_byte_span_t executable_data) {
  const bool size_unknown = executable_data.data_length == 0;
  if (!size_unknown && executable_data.data_length < sizeof(kPeMagic)) {
    return false;  // too small
  }
  return memcmp(executable_data.data, kPeMagic, sizeof(kPeMagic)) == 0;
}

static iree_status_t iree_hal_executable_calculate_pe_size(
    const iree_pe_file_header_t* pe_header, iree_host_size_t* out_size) {
  // Skip optional header to get to section headers.
  const uint8_t* section_headers =
      (const uint8_t*)(pe_header + 1) + pe_header->sizeOfOptionalHeader;
  uint32_t max_offset = 0;
  for (uint16_t i = 0; i < pe_header->numberOfSections; ++i) {
    const iree_pe_section_header_t* section =
        (const iree_pe_section_header_t*)(section_headers +
                                          i * sizeof(iree_pe_section_header_t));
    const uint32_t section_end =
        section->pointerToRawData + section->sizeOfRawData;
    if (section_end > max_offset) {
      max_offset = section_end;  // bump end offset
    }
  }
  *out_size = max_offset;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_executable_infer_dll_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  IREE_ASSERT_ARGUMENT(!executable_format_capacity || executable_format);
  IREE_ASSERT_ARGUMENT(out_inferred_size);
  *out_inferred_size = 0;

  // WARNING: when data_length is 0 we assume we can read at least 2 bytes
  // for magic number detection. This is UNSAFE but required for compatibility.
  const bool size_unknown = executable_data.data_length == 0;

  // Check for PE/DLL magic (DOS header 'MZ').
  if (!size_unknown && executable_data.data_length < sizeof(kPeMagic)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable data too small to identify format");
  }
  if (memcmp(executable_data.data, kPeMagic, sizeof(kPeMagic)) != 0) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "executable format is not PE/DLL");
  }

  // Need DOS header to find PE header.
  if (!size_unknown &&
      executable_data.data_length < sizeof(iree_dos_header_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PE data too small for DOS header");
  }
  const iree_dos_header_t* dos_header =
      (const iree_dos_header_t*)executable_data.data;
  const uint32_t pe_offset = dos_header->e_lfanew;

  // Check PE signature and get header.
  if (!size_unknown && pe_offset + sizeof(iree_pe_file_header_t) + 4 >
                           executable_data.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PE header offset out of bounds");
  }
  const uint8_t* pe_sig = executable_data.data + pe_offset;
  if (memcmp(pe_sig, "PE\0\0", 4) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Invalid PE signature");
  }
  const iree_pe_file_header_t* pe_header =
      (const iree_pe_file_header_t*)(pe_sig + 4);

  // Calculate the file size using the parsed header if unknown.
  if (size_unknown) {
    IREE_RETURN_IF_ERROR(
        iree_hal_executable_calculate_pe_size(pe_header, out_inferred_size),
        "calculating PE size");
  } else {
    *out_inferred_size = executable_data.data_length;
  }

  // Get architecture from the parsed header.
  const char* arch_suffix = "unknown";
  switch (pe_header->machine) {
    case IREE_PE_IMAGE_FILE_MACHINE_AMD64:
      arch_suffix = "x86_64";
      break;
    case IREE_PE_IMAGE_FILE_MACHINE_I386:
      arch_suffix = "x86_32";
      break;
    case IREE_PE_IMAGE_FILE_MACHINE_ARM64:
      arch_suffix = "arm_64";
      break;
    case IREE_PE_IMAGE_FILE_MACHINE_ARMNT:
      arch_suffix = "arm_32";
      break;
    default:
      arch_suffix = "unknown";
      break;
  }
  return iree_hal_executable_write_executable_format(
      "system-dll-", arch_suffix, executable_format,
      executable_format_capacity);
}

//===----------------------------------------------------------------------===//
// MachO
//===----------------------------------------------------------------------===//

// Mach-O magic bytes (big-endian).
static const uint8_t kMachO32Magic[4] = {0xFE, 0xED, 0xFA, 0xCE};
static const uint8_t kMachO64Magic[4] = {0xFE, 0xED, 0xFA, 0xCF};
static const uint8_t kMachOFatMagic[4] = {0xCA, 0xFE, 0xBA, 0xBE};

#define IREE_MACHO_CPU_TYPE_X86 7
#define IREE_MACHO_CPU_TYPE_X86_64 (IREE_MACHO_CPU_TYPE_X86 | 0x01000000)
#define IREE_MACHO_CPU_TYPE_ARM 12
#define IREE_MACHO_CPU_TYPE_ARM64 (IREE_MACHO_CPU_TYPE_ARM | 0x01000000)

typedef struct {
  uint32_t magic;
  uint32_t cputype;
  uint32_t cpusubtype;
  uint32_t filetype;
  uint32_t ncmds;
  uint32_t sizeofcmds;
  uint32_t flags;
} iree_macho_header_32_t;

typedef struct {
  uint32_t magic;
  uint32_t cputype;
  uint32_t cpusubtype;
  uint32_t filetype;
  uint32_t ncmds;
  uint32_t sizeofcmds;
  uint32_t flags;
  uint32_t reserved;
} iree_macho_header_64_t;

typedef struct {
  uint32_t cmd;
  uint32_t cmdsize;
} iree_macho_load_command_t;

typedef struct {
  uint32_t cmd;
  uint32_t cmdsize;
  char segname[16];
  uint32_t vmaddr;
  uint32_t vmsize;
  uint32_t fileoff;
  uint32_t filesize;
} iree_macho_segment_command_32_t;

typedef struct {
  uint32_t cmd;
  uint32_t cmdsize;
  char segname[16];
  uint64_t vmaddr;
  uint64_t vmsize;
  uint64_t fileoff;
  uint64_t filesize;
} iree_macho_segment_command_64_t;

static iree_host_size_t iree_hal_executable_calculate_macho_size_32(
    const iree_macho_header_32_t* header) {
  const uint8_t* cmd_ptr = (const uint8_t*)(header + 1);
  uint32_t max_offset = sizeof(*header) + header->sizeofcmds;
  for (uint32_t i = 0; i < header->ncmds; ++i) {
    const iree_macho_load_command_t* cmd =
        (const iree_macho_load_command_t*)cmd_ptr;
    if (cmd->cmd == 0x1) {  // LC_SEGMENT
      const iree_macho_segment_command_32_t* seg =
          (const iree_macho_segment_command_32_t*)cmd;
      const uint32_t segment_end = seg->fileoff + seg->filesize;
      if (segment_end > max_offset) {
        max_offset = segment_end;  // bump extent
      }
    }
    cmd_ptr += cmd->cmdsize;
  }
  return (iree_host_size_t)max_offset;
}

static iree_host_size_t iree_hal_executable_calculate_macho_size_64(
    const iree_macho_header_64_t* header) {
  const uint8_t* cmd_ptr = (const uint8_t*)(header + 1);
  uint64_t max_offset = sizeof(*header) + header->sizeofcmds;
  for (uint32_t i = 0; i < header->ncmds; ++i) {
    const iree_macho_load_command_t* cmd =
        (const iree_macho_load_command_t*)cmd_ptr;
    if (cmd->cmd == 0x19) {  // LC_SEGMENT_64
      const iree_macho_segment_command_64_t* seg =
          (const iree_macho_segment_command_64_t*)cmd;
      const uint64_t segment_end = seg->fileoff + seg->filesize;
      if (segment_end > max_offset) {
        max_offset = segment_end;  // bump extent
      }
    }
    cmd_ptr += cmd->cmdsize;
  }
  return (iree_host_size_t)max_offset;
}

static bool iree_hal_executable_is_macho_file(
    iree_const_byte_span_t executable_data) {
  const bool size_unknown = executable_data.data_length == 0;
  if (!size_unknown && executable_data.data_length < sizeof(kMachO64Magic)) {
    return false;  // too small
  }
  return (memcmp(executable_data.data, kMachO32Magic, sizeof(kMachO32Magic)) ==
          0) ||
         (memcmp(executable_data.data, kMachO64Magic, sizeof(kMachO64Magic)) ==
          0) ||
         (memcmp(executable_data.data, kMachOFatMagic,
                 sizeof(kMachOFatMagic)) == 0);
}

IREE_API_EXPORT iree_status_t iree_hal_executable_infer_macho_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  IREE_ASSERT_ARGUMENT(!executable_format_capacity || executable_format);
  IREE_ASSERT_ARGUMENT(out_inferred_size);
  *out_inferred_size = 0;

  // WARNING: when data_length is 0 we assume we can read at least 4 bytes
  // for magic number detection. This is UNSAFE but required for compatibility.
  const bool size_unknown = executable_data.data_length == 0;
  if (!size_unknown && executable_data.data_length < sizeof(kMachO64Magic)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable data too small to identify format");
  }
  const bool is_32bit =
      memcmp(executable_data.data, kMachO32Magic, sizeof(kMachO32Magic)) == 0;
  const bool is_64bit =
      memcmp(executable_data.data, kMachO64Magic, sizeof(kMachO64Magic)) == 0;
  const bool is_fat =
      memcmp(executable_data.data, kMachOFatMagic, sizeof(kMachOFatMagic)) == 0;
  if (!is_32bit && !is_64bit && !is_fat) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "executable format is not Mach-O");
  }

  const char* arch_suffix = "unknown";
  if (is_64bit) {
    if (!size_unknown &&
        executable_data.data_length < sizeof(iree_macho_header_64_t)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Mach-O 64 data too small for header");
    }
    const iree_macho_header_64_t* header =
        (const iree_macho_header_64_t*)executable_data.data;
    if (size_unknown) {
      *out_inferred_size = iree_hal_executable_calculate_macho_size_64(header);
    } else {
      *out_inferred_size = executable_data.data_length;
    }
    if (header->cputype == IREE_MACHO_CPU_TYPE_X86_64) {
      arch_suffix = "x86_64";
    } else if (header->cputype == IREE_MACHO_CPU_TYPE_ARM64) {
      arch_suffix = "arm_64";
    }
  } else if (is_32bit) {
    if (!size_unknown &&
        executable_data.data_length < sizeof(iree_macho_header_32_t)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Mach-O 32 data too small for header");
    }
    const iree_macho_header_32_t* header =
        (const iree_macho_header_32_t*)executable_data.data;
    if (size_unknown) {
      *out_inferred_size = iree_hal_executable_calculate_macho_size_32(header);
    } else {
      *out_inferred_size = executable_data.data_length;
    }
    if (header->cputype == IREE_MACHO_CPU_TYPE_X86) {
      arch_suffix = "x86_32";
    } else if (header->cputype == IREE_MACHO_CPU_TYPE_ARM) {
      arch_suffix = "arm_32";
    }
  } else if (is_fat) {
    // Note: For fat binaries we'd need to parse the fat header to determine
    // architectures, for now we just mark as unknown.
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "fat MachO files not yet supported");
  }

  return iree_hal_executable_write_executable_format(
      "system-dylib-", arch_suffix, executable_format,
      executable_format_capacity);
}

//===----------------------------------------------------------------------===//
// WASM
//===----------------------------------------------------------------------===//

// WebAssembly magic bytes: \0 'a' 's' 'm'.
static const uint8_t kWasmMagic[4] = {0x00, 'a', 's', 'm'};

typedef struct iree_wasm_header_t {
  uint8_t magic[4];
  uint32_t version;
} iree_wasm_header_t;

// WebAssembly section IDs.
typedef enum {
  IREE_WASM_SECTION_CUSTOM = 0,
  IREE_WASM_SECTION_TYPE = 1,
  IREE_WASM_SECTION_IMPORT = 2,
  IREE_WASM_SECTION_FUNCTION = 3,
  IREE_WASM_SECTION_TABLE = 4,
  IREE_WASM_SECTION_MEMORY = 5,
  IREE_WASM_SECTION_GLOBAL = 6,
  IREE_WASM_SECTION_EXPORT = 7,
  IREE_WASM_SECTION_START = 8,
  IREE_WASM_SECTION_ELEMENT = 9,
  IREE_WASM_SECTION_CODE = 10,
  IREE_WASM_SECTION_DATA = 11,
  IREE_WASM_SECTION_DATA_COUNT = 12,  // from bulk memory proposal
} iree_wasm_section_id_t;

static bool iree_hal_executable_is_wasm_file(
    iree_const_byte_span_t executable_data) {
  const bool size_unknown = executable_data.data_length == 0;
  if (!size_unknown && executable_data.data_length < sizeof(kWasmMagic)) {
    return false;  // too small
  }
  return memcmp(executable_data.data, kWasmMagic, sizeof(kWasmMagic)) == 0;
}

// Decodes an LEB128 unsigned integer from WASM data.
// Returns the number of bytes consumed or 0 on error.
static iree_host_size_t iree_hal_wasm_decode_leb128_u32(
    const uint8_t* data, iree_host_size_t max_bytes, uint32_t* out_value) {
  if (max_bytes == 0) return 0;
  uint32_t value = 0;
  iree_host_size_t bytes_read = 0;
  uint32_t shift = 0;
  // Read up to max 5 bytes for u32.
  while (bytes_read < max_bytes && bytes_read < 5) {
    uint8_t byte = data[bytes_read++];
    value |= ((uint32_t)(byte & 0x7F)) << shift;
    if ((byte & 0x80) == 0) {
      *out_value = value;
      return bytes_read;
    }
    shift += 7;
  }
  return 0;  // error: incomplete or too large
}

// Decodes an LEB128 unsigned 64-bit integer from WASM data.
// Returns the number of bytes consumed or 0 on error.
static iree_host_size_t iree_hal_wasm_decode_leb128_u64(
    const uint8_t* data, iree_host_size_t max_bytes, uint64_t* out_value) {
  if (max_bytes == 0) return 0;
  uint64_t value = 0;
  iree_host_size_t bytes_read = 0;
  uint32_t shift = 0;
  // Read up to max 10 bytes for u64.
  while (bytes_read < max_bytes && bytes_read < 10) {
    uint8_t byte = data[bytes_read++];
    value |= ((uint64_t)(byte & 0x7F)) << shift;
    if ((byte & 0x80) == 0) {
      *out_value = value;
      return bytes_read;
    }
    shift += 7;
  }
  return 0;  // error: incomplete or too large
}

// Checks if a WASM memory section contains any 64-bit memories.
// Returns true if any memory uses 64-bit addressing (flags 0x04-0x07).
static bool iree_hal_executable_is_memory64_used(
    const uint8_t* section_data, iree_host_size_t section_size) {
  iree_host_size_t offset = 0;

  // Read number of memories.
  uint32_t num_memories = 0;
  iree_host_size_t consumed = iree_hal_wasm_decode_leb128_u32(
      section_data + offset, section_size - offset, &num_memories);
  if (consumed == 0) {
    return false;  // failed to parse
  }
  offset += consumed;

  // Check each memory's limits.
  for (uint32_t i = 0; i < num_memories && offset < section_size; ++i) {
    if (offset >= section_size) {
      return false;  // incomplete section
    }

    // Flags 0x04-0x07 indicate 64-bit memory.
    const uint8_t limits_flag = section_data[offset++];
    if (limits_flag >= 0x04 && limits_flag <= 0x07) {
      return true;  // found 64-bit memory - early exit
    }

    // Skip the limit values based on flag.
    if (limits_flag == 0x00 || limits_flag == 0x02) {
      // min:u32 only.
      uint32_t min_val = 0;
      consumed = iree_hal_wasm_decode_leb128_u32(
          section_data + offset, section_size - offset, &min_val);
      if (consumed == 0) break;
      offset += consumed;
    } else if (limits_flag == 0x01 || limits_flag == 0x03) {
      // min:u32 max:u32.
      uint32_t min_val = 0, max_val = 0;
      consumed = iree_hal_wasm_decode_leb128_u32(
          section_data + offset, section_size - offset, &min_val);
      if (consumed == 0) break;
      offset += consumed;
      consumed = iree_hal_wasm_decode_leb128_u32(
          section_data + offset, section_size - offset, &max_val);
      if (consumed == 0) break;
      offset += consumed;
    } else {
      return false;  // unknown flag, abort
    }
  }

  return false;  // no 64-bit memory found
}

IREE_API_EXPORT iree_status_t iree_hal_executable_infer_wasm_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  IREE_ASSERT_ARGUMENT(!executable_format_capacity || executable_format);
  IREE_ASSERT_ARGUMENT(out_inferred_size);
  *out_inferred_size = 0;

  // WARNING: when data_length is 0 we assume we can read at least 8 bytes
  // for magic number detection. This is UNSAFE but required for compatibility.
  const bool size_unknown = executable_data.data_length == 0;

  // If size is known verify we have enough for magic checks.
  if (!size_unknown &&
      executable_data.data_length < sizeof(iree_wasm_header_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable data too small to identify format");
  }

  iree_wasm_header_t header;
  memcpy(&header, executable_data.data, sizeof(header));

  // Check for WebAssembly magic bytes and version.
  if (memcmp(header.magic, kWasmMagic, sizeof(kWasmMagic)) != 0) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "executable format is not WebAssembly");
  }

  // Check version (should be 1).
  if (header.version != 0x00000001) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "unsupported WASM version");
  }

  const uint8_t* data = executable_data.data;
  const iree_host_size_t max_length =
      size_unknown ? IREE_HOST_SIZE_MAX : executable_data.data_length;

  // Parse sections to detect memory64 and optionally calculate size.
  iree_host_size_t offset = sizeof(header);
  iree_host_size_t max_offset = offset;
  bool has_memory64 = false;
  while (offset < max_length) {
    // Check if we can read section ID.
    if (!size_unknown && offset >= executable_data.data_length) {
      break;  // End of known data.
    }

    // Check if this is a known section ID.
    const uint8_t section_id = data[offset++];
    if (section_id > IREE_WASM_SECTION_DATA_COUNT &&
        section_id != IREE_WASM_SECTION_CUSTOM) {
      // Unknown section ID, assume end of file.
      break;
    }

    // Read section size (LEB128 encoded).
    const iree_host_size_t bytes_remaining =
        size_unknown ? 5 : (max_length > offset ? max_length - offset : 0);
    uint32_t section_size = 0;
    const iree_host_size_t leb_bytes = iree_hal_wasm_decode_leb128_u32(
        data + offset, bytes_remaining, &section_size);
    if (leb_bytes == 0) {
      // Failed to decode size, assume we've reached the end.
      break;
    }
    offset += leb_bytes;

    // Check for memory section to detect memory64.
    if (section_id == IREE_WASM_SECTION_MEMORY && !has_memory64) {
      // Check if this memory section uses 64-bit addressing.
      has_memory64 =
          iree_hal_executable_is_memory64_used(data + offset, section_size);
    }

    offset += section_size;
    if (size_unknown && offset > max_offset) {
      max_offset = offset;  // bump extent
    }
  }
  *out_inferred_size = size_unknown ? max_offset : executable_data.data_length;

  // Return format based on whether we found 64-bit memory.
  const char* format = has_memory64 ? "wasm_64" : "wasm_32";
  return iree_hal_executable_write_executable_format(
      NULL, format, executable_format, executable_format_capacity);
}

//===----------------------------------------------------------------------===//
// Automatic detection
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_executable_infer_system_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  IREE_ASSERT_ARGUMENT(!executable_format_capacity || executable_format);
  IREE_ASSERT_ARGUMENT(out_inferred_size);
  *out_inferred_size = 0;

  // WARNING: when data_length is 0 we assume we can read at least 4 bytes
  // for magic number detection. This is UNSAFE and may cause access violations
  // if the data is not actually at least 4 bytes. This is required for
  // compatibility with CUDA/HIP APIs that don't provide sizes.
  const bool size_unknown = executable_data.data_length == 0;

#if defined(IREE_PLATFORM_WINDOWS)
  if (iree_hal_executable_is_dll_file(executable_data)) {
    // Check for PE/DLL on Windows - delegate to the DLL specific function.
    return iree_hal_executable_infer_dll_format(
        executable_data, executable_format_capacity, executable_format,
        out_inferred_size);
  }
#elif defined(IREE_PLATFORM_APPLE)
  if (iree_hal_executable_is_macho_file(executable_data)) {
    return iree_hal_executable_infer_macho_format(
        executable_data, executable_format_capacity, executable_format,
        out_inferred_size);
  }
#elif defined(IREE_PLATFORM_EMSCRIPTEN)
  if (iree_hal_executable_is_wasm_file(executable_data)) {
    return iree_hal_executable_infer_wasm_format(
        executable_data, executable_format_capacity, executable_format,
        out_inferred_size);
  }
#else
  if (iree_hal_executable_is_fatelf_file(executable_data)) {
    if (size_unknown) {
      IREE_RETURN_IF_ERROR(iree_hal_executable_calculate_fatelf_size(
          executable_data, out_inferred_size));
    } else {
      *out_inferred_size = executable_data.data_length;
    }
    return iree_hal_executable_write_executable_format(
        "system-", "fatelf", executable_format, executable_format_capacity);
  } else if (iree_hal_executable_is_elf_file(executable_data)) {
    const iree_elf_ehdr_common_t* ehdr =
        (const iree_elf_ehdr_common_t*)executable_data.data;
    if (size_unknown) {
      IREE_RETURN_IF_ERROR(iree_hal_executable_calculate_elf_size(
          ehdr, executable_data, out_inferred_size));
    } else {
      *out_inferred_size = executable_data.data_length;
    }
    const char* arch_suffix = iree_hal_executable_infer_elf_format_arch(ehdr);
    return iree_hal_executable_write_executable_format(
        "system-elf-", arch_suffix, executable_format,
        executable_format_capacity);
  }
#endif  // IREE_PLATFORM_*

  return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                          "unable to detect system executable format");
}
