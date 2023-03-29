// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/internal/file_io.h"
#include "iree/base/internal/path.h"
#include "iree/hal/local/elf/fatelf.h"

// NOTE: we don't verify ELF information in here and just pass it along. Don't
// run this on untrusted ELFs.

// NOTE: errors are handled in here just enough to get error messages - we don't
// care about leaks on failure as the process is going to die right away.

// TODO(benvanik): make this based on the archs used? It needs to be the common
// page size across all targets used within the file.
#define IREE_FATELF_PAGE_SIZE 4096

#if defined(IREE_PLATFORM_WINDOWS)
#include <fcntl.h>
#include <io.h>
#define IREE_SET_BINARY_MODE(handle) _setmode(_fileno(handle), O_BINARY)
#else
#define IREE_SET_BINARY_MODE(handle) ((void)0)
#endif  // IREE_PLATFORM_WINDOWS

static int print_usage() {
  fprintf(stderr, "Syntax: iree-fatelf [join|split|select|dump] files...\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "Join multiple ELFs into a FatELF:\n");
  fprintf(stderr, "  iree-fatelf join elf_a.so elf_b.so > fatelf.sos\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "Split a FatELF into multiple ELF files (to dir):\n");
  fprintf(stderr, "  iree-fatelf split fatelf.sos\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "Select a FatELF matching the current arch:\n");
  fprintf(stderr, "  iree-fatelf select fatelf.sos > elf.so\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "Dump header records:\n");
  fprintf(stderr, "  iree-fatelf dump fatelf.sos\n");
  fprintf(stderr, "\n");
  return 1;
}

// NOTE: this is somewhat redundant with fatelf.c but that's ok - this is a
// developer tool and I'd rather have the implementation linked into every
// runtime be kept as simple as possible than not repeating 100 lines of code.
// The runtime version is also designed to gracefully accept ELF files where
// here we only want FatELF files.
static iree_status_t fatelf_parse(iree_const_byte_span_t file_data,
                                  iree_fatelf_header_t** out_header) {
  *out_header = NULL;

  if (file_data.data_length <
      sizeof(iree_fatelf_header_t) + sizeof(iree_fatelf_record_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "file does not have enough data to even hold a FatELF header");
  }

  const iree_fatelf_header_t* raw_header =
      (const iree_fatelf_header_t*)file_data.data;
  iree_fatelf_header_t host_header = {
      .magic = iree_unaligned_load_le_u32(&raw_header->magic),
      .version = iree_unaligned_load_le_u16(&raw_header->version),
      .record_count = iree_unaligned_load_le_u8(&raw_header->record_count),
      .reserved = iree_unaligned_load_le_u8(&raw_header->reserved),
  };

  if (host_header.magic != IREE_FATELF_MAGIC) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "file magic %08X does not match expected FatELF magic %08X",
        host_header.magic, IREE_FATELF_MAGIC);
  }
  if (host_header.version != IREE_FATELF_FORMAT_VERSION) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "FatELF has version %d but runtime only supports version %d",
        host_header.version, IREE_FATELF_FORMAT_VERSION);
  }

  iree_host_size_t required_bytes =
      sizeof(iree_fatelf_header_t) +
      host_header.record_count * sizeof(iree_fatelf_record_t);
  if (file_data.data_length < required_bytes) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "FatELF file truncated, requires at least %" PRIhsz
                            "B for headers but only have %" PRIhsz
                            "B available",
                            required_bytes, file_data.data_length);
  }

  // Allocate storage for the parsed header and records.
  iree_fatelf_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      iree_allocator_system(),
      sizeof(iree_fatelf_header_t) +
          host_header.record_count * sizeof(iree_fatelf_record_t),
      (void**)&header));
  memcpy(header, &host_header, sizeof(*header));
  for (iree_elf64_byte_t i = 0; i < host_header.record_count; ++i) {
    const iree_fatelf_record_t* raw_record = &raw_header->records[i];
    const iree_fatelf_record_t host_record = {
        .machine = iree_unaligned_load_le_u16(&raw_record->machine),
        .osabi = iree_unaligned_load_le_u8(&raw_record->osabi),
        .osabi_version = iree_unaligned_load_le_u8(&raw_record->osabi_version),
        .word_size = iree_unaligned_load_le_u8(&raw_record->word_size),
        .byte_order = iree_unaligned_load_le_u8(&raw_record->byte_order),
        .reserved0 = iree_unaligned_load_le_u8(&raw_record->reserved0),
        .reserved1 = iree_unaligned_load_le_u8(&raw_record->reserved1),
        .offset = iree_unaligned_load_le_u64(&raw_record->offset),
        .size = iree_unaligned_load_le_u64(&raw_record->size),
    };
    memcpy(&header->records[i], &host_record, sizeof(host_record));
  }

  *out_header = header;
  return iree_ok_status();
}

// Tries to parse basic ELF metadata from |elf_data|.
// Very little verification done. Note that this must support both 32 and 64-bit
// ELF files regardless of the tool host configuration.
// The returned fields match the ELF spec and may differ from FatELF.
static iree_status_t fatelf_parse_elf_metadata(
    iree_const_byte_span_t elf_data, iree_elf64_half_t* out_machine,
    iree_elf64_byte_t* out_osabi, iree_elf64_byte_t* out_osabi_version,
    iree_elf64_byte_t* out_elf_class, iree_elf64_byte_t* out_elf_data) {
  *out_machine = 0;
  *out_osabi = 0;
  *out_osabi_version = 0;
  *out_elf_data = 0;
  *out_elf_class = 0;

  if (elf_data.data_length < sizeof(iree_elf32_ehdr_t)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "ELF data provided (%zu) is smaller than ehdr (%zu)",
        elf_data.data_length, sizeof(iree_elf32_ehdr_t));
  }

  // The fields we're checking are the same in both 32 and 64 classes so we just
  // use 32 for consistency.
  const iree_elf32_ehdr_t* ehdr = (const iree_elf32_ehdr_t*)elf_data.data;
  static const iree_elf_byte_t elf_magic[4] = {0x7F, 'E', 'L', 'F'};
  if (memcmp(ehdr->e_ident, elf_magic, sizeof(elf_magic)) != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "data provided does not contain the ELF identifier");
  }

  *out_osabi = ehdr->e_ident[IREE_ELF_EI_OSABI];
  *out_osabi_version = ehdr->e_ident[IREE_ELF_EI_ABIVERSION];
  *out_elf_class = ehdr->e_ident[IREE_ELF_EI_CLASS];
  *out_elf_data = ehdr->e_ident[IREE_ELF_EI_DATA];

  // Note machine is multibyte and respects the declared endianness.
  if (ehdr->e_ident[IREE_ELF_EI_DATA] == IREE_ELF_ELFDATA2LSB) {
    *out_machine = iree_unaligned_load_le_u16(&ehdr->e_machine);
  } else {
#if IREE_ENDIANNESS_BIG
// TODO(benvanik): helpers for big<->little endian
// *out_machine = iree_unaligned_load_be_u16(&ehdr->e_machine);
#error "ELF parsing support only available on little-endian systems today"
#endif  // IREE_ENDIANNESS_BIG
  }

  return iree_ok_status();
}

typedef struct {
  uint64_t offset;
  iree_file_contents_t* contents;
  iree_const_byte_span_t elf_data;
} fatelf_entry_t;

// Joins one or more ELF files together and writes the output to stdout.
static iree_status_t fatelf_join(int argc, char** argv) {
  IREE_SET_BINARY_MODE(stdout);  // ensure binary output mode

#if IREE_ENDIANNESS_BIG
#error "FatELF writing support only available on little-endian systems today"
#endif  // IREE_ENDIANNESS_BIG

  // Load all source files.
  iree_elf64_byte_t entry_count = argc;
  fatelf_entry_t* entries =
      (fatelf_entry_t*)iree_alloca(entry_count * sizeof(fatelf_entry_t));
  memset(entries, 0, entry_count * sizeof(*entries));
  for (iree_elf64_byte_t i = 0; i < entry_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_file_read_contents(
        argv[i], iree_allocator_system(), &entries[i].contents));
    entries[i].elf_data = entries[i].contents->const_buffer;
  }

  // Compute offsets of all files based on their size and padding.
  uint64_t file_offset = iree_host_align(
      sizeof(iree_fatelf_header_t) + entry_count * sizeof(iree_fatelf_record_t),
      IREE_FATELF_PAGE_SIZE);
  for (iree_elf64_byte_t i = 0; i < entry_count; ++i) {
    entries[i].offset = file_offset;
    file_offset += iree_host_align(
        entries[i].contents->const_buffer.data_length, IREE_FATELF_PAGE_SIZE);
  }

  // Write header without records.
  iree_fatelf_header_t host_header = {
      .magic = IREE_FATELF_MAGIC,
      .version = IREE_FATELF_FORMAT_VERSION,
      .record_count = entry_count,
      .reserved = 0,
  };
  fwrite(&host_header, 1, sizeof(host_header), stdout);

  // Write all records.
  for (iree_elf64_byte_t i = 0; i < entry_count; ++i) {
    iree_elf64_half_t machine = 0;
    iree_elf64_byte_t osabi = 0;
    iree_elf64_byte_t osabi_version = 0;
    iree_elf64_byte_t elf_class = 0;
    iree_elf64_byte_t elf_data = 0;
    IREE_RETURN_IF_ERROR(
        fatelf_parse_elf_metadata(entries[i].elf_data, &machine, &osabi,
                                  &osabi_version, &elf_class, &elf_data));
    iree_fatelf_record_t host_record = {
        .machine = machine,
        .osabi = osabi,
        .osabi_version = osabi_version,
        .word_size = elf_class == IREE_ELF_ELFCLASS32
                         ? IREE_FATELF_WORD_SIZE_32
                         : IREE_FATELF_WORD_SIZE_64,
        .byte_order = elf_data == IREE_ELF_ELFDATA2LSB
                          ? IREE_FATELF_BYTE_ORDER_LSB
                          : IREE_FATELF_BYTE_ORDER_MSB,
        .reserved0 = 0,
        .reserved1 = 0,
        .offset = (iree_elf64_off_t)entries[i].offset,
        .size = (iree_elf64_xword_t)entries[i].elf_data.data_length,
    };
    fwrite(&host_record, 1, sizeof(host_record), stdout);
  }

  // Write all files, padding with zeros in-between as needed.
  uint64_t write_offset =
      sizeof(iree_fatelf_header_t) + entry_count * sizeof(iree_fatelf_record_t);
  for (iree_elf64_byte_t i = 0; i < entry_count; ++i) {
    uint64_t padding = entries[i].offset - write_offset;
    for (uint64_t i = 0; i < padding; ++i) fputc(0, stdout);
    write_offset += padding;
    if (write_offset != entries[i].offset) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "actual offset does not match expected");
    }
    fwrite(entries[i].elf_data.data, 1, entries[i].elf_data.data_length,
           stdout);
    write_offset += entries[i].elf_data.data_length;
  }
  fflush(stdout);

  for (iree_elf64_byte_t i = 0; i < entry_count; ++i) {
    iree_file_contents_free(entries[i].contents);
  }
  return iree_ok_status();
}

static const char* fatelf_machine_id_str(iree_elf64_half_t value) {
  // TODO(benvanik): include a full table from the spec?
  // http://formats.kaitai.io/elf/ has a good source of canonical short names.
  // For now we just support what we have in our ELF loader.
  switch (value) {
    case 0x03:  // EM_386 / 3
      return "x86";
    case 0x28:  // EM_ARM / 40
      return "arm";
    case 0xB7:  // EM_AARCH64 / 183
      return "aarch64";
    case 0xF3:  // EM_RISCV / 243
      return "risvc";
    case 0x3E:  // EM_X86_64 / 62
      return "x86_64";
    default:
      return "unknown";
  }
}

static const char* fatelf_osabi_id_str(iree_elf64_byte_t value) {
  switch (value) {
    case IREE_ELF_ELFOSABI_NONE:
      return "none";
    case IREE_ELF_ELFOSABI_LINUX:
      return "linux";
    case IREE_ELF_ELFOSABI_STANDALONE:
      return "standalone";
    default:
      return "unknown";
  }
}

static const char* fatelf_word_size_id_str(iree_elf64_byte_t value) {
  switch (value) {
    case IREE_FATELF_WORD_SIZE_32:
      return "lp32";
    case IREE_FATELF_WORD_SIZE_64:
      return "lp64";
    default:
      return "lpUNK";
  }
}

static const char* fatelf_byte_order_id_str(iree_elf64_byte_t value) {
  switch (value) {
    case IREE_FATELF_BYTE_ORDER_MSB:
      return "be";
    case IREE_FATELF_BYTE_ORDER_LSB:
      return "le";
    default:
      return "xx";
  }
}

// Splits a FatELF into multiple files, writing each beside the input file.
static iree_status_t fatelf_split(int argc, char** argv) {
  iree_file_contents_t* fatelf_contents = NULL;
  IREE_RETURN_IF_ERROR(iree_file_read_contents(argv[0], iree_allocator_system(),
                                               &fatelf_contents));
  iree_fatelf_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(fatelf_parse(fatelf_contents->const_buffer, &header));

  iree_string_view_t dirname, basename;
  iree_file_path_split(iree_make_cstring_view(argv[0]), &dirname, &basename);
  iree_string_view_t stem, extension;
  iree_file_path_split_basename(basename, &stem, &extension);

  for (iree_elf64_byte_t i = 0; i < header->record_count; ++i) {
    const iree_fatelf_record_t* record = &header->records[i];

    const char* machine_str = fatelf_machine_id_str(record->machine);
    const char* osabi_str = fatelf_osabi_id_str(record->osabi);
    const char* word_size_str = fatelf_word_size_id_str(record->word_size);
    const char* byte_order_str = fatelf_byte_order_id_str(record->byte_order);

    char record_path[2048];
    iree_host_size_t record_path_length =
        snprintf(record_path, IREE_ARRAYSIZE(record_path),
                 "%.*s%s%.*s.%s_%s_%s%s.so", (int)dirname.size, dirname.data,
                 dirname.size ? "/" : "", (int)stem.size, stem.data,
                 machine_str, osabi_str, word_size_str, byte_order_str);
    record_path_length =
        iree_file_path_canonicalize(record_path, record_path_length);

    fprintf(stdout, "Writing record[%d] to '%.*s'...\n", i,
            (int)record_path_length, record_path);

    iree_const_byte_span_t record_data = iree_make_const_byte_span(
        fatelf_contents->const_buffer.data + record->offset, record->size);
    IREE_RETURN_IF_ERROR(iree_file_write_contents(record_path, record_data));
  }

  fprintf(stdout, "Wrote %d records to %.*s!\n", header->record_count,
          (int)dirname.size, dirname.data);
  iree_allocator_free(iree_allocator_system(), header);
  iree_file_contents_free(fatelf_contents);
  return iree_ok_status();
}

// Selects the ELF matching the current host config from a FatELF and writes
// it to stdout.
static iree_status_t fatelf_select(int argc, char** argv) {
  IREE_SET_BINARY_MODE(stdout);  // ensure binary output mode
  iree_file_contents_t* fatelf_contents = NULL;
  IREE_RETURN_IF_ERROR(iree_file_read_contents(argv[0], iree_allocator_system(),
                                               &fatelf_contents));
  iree_const_byte_span_t elf_data = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(
      iree_fatelf_select(fatelf_contents->const_buffer, &elf_data));
  fwrite(elf_data.data, 1, elf_data.data_length, stdout);
  iree_file_contents_free(fatelf_contents);
  return iree_ok_status();
}

static const char* fatelf_word_size_enum_str(iree_elf64_byte_t value) {
  switch (value) {
    case IREE_FATELF_WORD_SIZE_32:
      return "ELFCLASS32";
    case IREE_FATELF_WORD_SIZE_64:
      return "ELFCLASS64";
    default:
      return "<unknown>";
  }
}

static const char* fatelf_byte_order_enum_str(iree_elf64_byte_t value) {
  switch (value) {
    case IREE_FATELF_BYTE_ORDER_MSB:
      return "ELFDATA2MSB (big-endian)";
    case IREE_FATELF_BYTE_ORDER_LSB:
      return "ELFDATA2LSB (little-endian)";
    default:
      return "<unknown>";
  }
}

// Dumps the FatELF file records.
static iree_status_t fatelf_dump(int argc, char** argv) {
  iree_file_contents_t* fatelf_contents = NULL;
  IREE_RETURN_IF_ERROR(iree_file_read_contents(argv[0], iree_allocator_system(),
                                               &fatelf_contents));
  iree_fatelf_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(fatelf_parse(fatelf_contents->const_buffer, &header));

  fprintf(stdout, "iree_fatelf_header_t:\n");
  fprintf(stdout, "    magic: %" PRIX32 "\n", header->magic);
  fprintf(stdout, "  version: %d\n", header->version);
  fprintf(stdout, "  records: %d\n", header->record_count);
  fprintf(stdout, " reserved: %" PRIX8 "\n", header->reserved);
  fprintf(stdout, "\n");

  for (iree_elf64_byte_t i = 0; i < header->record_count; ++i) {
    const iree_fatelf_record_t* record = &header->records[i];
    fprintf(stdout, "iree_fatelf_record_t[%d]:\n", i);
    fprintf(stdout, "    machine: %d / %04X = %s\n", record->machine,
            record->machine, fatelf_machine_id_str(record->machine));
    fprintf(stdout, "      osabi: %d / %02X = %s\n", record->osabi,
            record->osabi, fatelf_osabi_id_str(record->osabi));
    fprintf(stdout, "    version: %d / %02X\n", record->osabi_version,
            record->osabi_version);
    fprintf(stdout, "  word_size: %d / %02X = %s\n", record->word_size,
            record->word_size, fatelf_word_size_enum_str(record->word_size));
    fprintf(stdout, " byte_order: %d / %02X = %s\n", record->byte_order,
            record->byte_order, fatelf_byte_order_enum_str(record->byte_order));
    fprintf(stdout, "  reserved0: %d / %02X\n", record->reserved0,
            record->reserved0);
    fprintf(stdout, "  reserved1: %d / %02X\n", record->reserved1,
            record->reserved1);
    fprintf(stdout, "     offset: %" PRIu64 " / %016" PRIX64 "\n",
            record->offset, record->offset);
    fprintf(stdout, "       size: %" PRIu64 " / %016" PRIX64 "\n", record->size,
            record->size);
    fprintf(stdout, "\n");
  }

  iree_allocator_free(iree_allocator_system(), header);
  iree_file_contents_free(fatelf_contents);
  return iree_ok_status();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    return print_usage();
  }

  char* command = argv[1];
  int command_argc = argc - 2;
  char** command_argv = argv + 2;

  iree_status_t status = iree_ok_status();
  if (strcmp(command, "join") == 0) {
    if (command_argc < 1) return print_usage();
    status = fatelf_join(command_argc, command_argv);
  } else if (strcmp(command, "split") == 0) {
    if (command_argc != 1) return print_usage();
    status = fatelf_split(command_argc, command_argv);
  } else if (strcmp(command, "select") == 0) {
    if (command_argc != 1) return print_usage();
    status = fatelf_select(command_argc, command_argv);
  } else if (strcmp(command, "dump") == 0) {
    if (command_argc != 1) return print_usage();
    status = fatelf_dump(command_argc, command_argv);
  } else {
    return print_usage();
  }

  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "iree-fatelf encountered error:\n");
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }
  return 0;
}
