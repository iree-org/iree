// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_NATIVE_EXECUTABLE_ELF_H_
#define IREE_HAL_DRIVERS_HIP_NATIVE_EXECUTABLE_ELF_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Fat binary wrapper magic and version (matching CUDA/HIP fat binary format).
#define IREE_HAL_HIP_FAT_BINARY_MAGIC 0x48495046u  // Magic for fat binary
#define IREE_HAL_HIP_FAT_BINARY_VERSION 1

// Clang offload bundle magic strings (matching HIP runtime).
// Uncompressed bundle format.
#define IREE_HAL_HIP_OFFLOAD_BUNDLE_MAGIC "__CLANG_OFFLOAD_BUNDLE__"
#define IREE_HAL_HIP_OFFLOAD_BUNDLE_MAGIC_SIZE 24

// Uncompressed bundle format magic as uint32_t (first 4 bytes: "__CL").
// Note: "__CLANG_OFFLOAD_BUNDLE__" starts with "__CL" = 0x5f5f434c (big-endian)
// or 0x4c435f5f (little-endian when read as uint32_t).
#define IREE_HAL_HIP_OFFLOAD_BUNDLE_MAGIC_INT 0x4c435f5f

// Compressed bundle format.
#define IREE_HAL_HIP_OFFLOAD_BUNDLE_COMPRESSED_MAGIC "CCOB"
#define IREE_HAL_HIP_OFFLOAD_BUNDLE_COMPRESSED_MAGIC_SIZE 4
#define IREE_HAL_HIP_OFFLOAD_BUNDLE_COMPRESSED_MAGIC_INT 0x424f4343

// ELF identification constants.
#define IREE_HAL_HIP_ELF_MAGIC0 0x7f
#define IREE_HAL_HIP_ELF_MAGIC1 'E'
#define IREE_HAL_HIP_ELF_MAGIC2 'L'
#define IREE_HAL_HIP_ELF_MAGIC3 'F'
#define IREE_HAL_HIP_ELF_MAGIC 0x464c457f

// ELF class (32/64 bit).
#define IREE_HAL_HIP_ELFCLASS64 2

// ELF data encoding.
#define IREE_HAL_HIP_ELFDATA2LSB 1  // Little-endian

// ELF machine types.
#define IREE_HAL_HIP_EM_AMDGPU 224  // AMD GPU architecture

// Offload bundle entry structure.
// This describes a single device-specific binary within an offload bundle.
typedef struct iree_hal_hip_bundle_entry_t {
  uint64_t offset;       // Offset of the bundle entry in the file
  uint64_t size;         // Size of the bundle entry
  uint64_t triple_size;  // Size of the triple string
} iree_hal_hip_bundle_entry_t;

// Fat binary wrapper structure.
// This wraps the actual executable data (which can be a compressed bundle,
// uncompressed bundle, or raw ELF). Similar to __CudaFatBinaryWrapper but
// cleaner for IREE's usage.
typedef struct iree_hal_hip_fat_binary_header_t {
  uint32_t magic;    // Magic number (IREE_HAL_HIP_FAT_BINARY_MAGIC)
  uint32_t version;  // Version of the fat binary format
  void* binary;      // Offset to the binary data from start of this structure
  void* reserved;    // Reserved for future use
} iree_hal_hip_fat_binary_header_t;
static_assert(sizeof(iree_hal_hip_fat_binary_header_t) == 24,
              "Fat binary header must be 24 bytes");

// ELF64 header structure.
// Based on the ELF64 specification for determining file size.
typedef struct iree_hal_hip_elf64_header_t {
  uint8_t magic[4];      // ELF magic: 0x7f, 'E', 'L', 'F'
  uint8_t class;         // 1=32-bit, 2=64-bit
  uint8_t data;          // 1=little-endian, 2=big-endian
  uint8_t version;       // ELF version (should be 1)
  uint8_t osabi;         // OS/ABI identification
  uint8_t abiversion;    // ABI version
  uint8_t padding[7];    // Padding to 16 bytes
  uint16_t type;         // Object file type
  uint16_t machine;      // Machine architecture
  uint32_t elf_version;  // Object file version
  uint64_t entry;        // Entry point address
  uint64_t phoff;        // Program header offset
  uint64_t shoff;        // Section header offset
  uint32_t flags;        // Processor-specific flags
  uint16_t ehsize;       // ELF header size
  uint16_t phentsize;    // Program header entry size
  uint16_t phnum;        // Number of program header entries
  uint16_t shentsize;    // Section header entry size
  uint16_t shnum;        // Number of section header entries
  uint16_t shstrndx;     // Section header string table index
} iree_hal_hip_elf64_header_t;

// ELF64 section header structure.
typedef struct iree_hal_hip_elf64_section_header_t {
  uint32_t sh_name;       // Section name (index into string table)
  uint32_t sh_type;       // Section type
  uint64_t sh_flags;      // Section flags
  uint64_t sh_addr;       // Virtual address in memory
  uint64_t sh_offset;     // Offset in file
  uint64_t sh_size;       // Size of section
  uint32_t sh_link;       // Link to other section
  uint32_t sh_info;       // Miscellaneous information
  uint64_t sh_addralign;  // Address alignment boundary
  uint64_t sh_entsize;    // Size of entries, if section has table
} iree_hal_hip_elf64_section_header_t;

// ELF64 symbol table entry structure.
typedef struct iree_hal_hip_elf64_symbol_t {
  uint32_t st_name;   // Symbol name (index into string table)
  uint8_t st_info;    // Symbol type and binding
  uint8_t st_other;   // Symbol visibility
  uint16_t st_shndx;  // Section index
  uint64_t st_value;  // Symbol value
  uint64_t st_size;   // Symbol size
} iree_hal_hip_elf64_symbol_t;

// ELF section types
#define IREE_HAL_HIP_SHT_NULL 0
#define IREE_HAL_HIP_SHT_PROGBITS 1
#define IREE_HAL_HIP_SHT_SYMTAB 2
#define IREE_HAL_HIP_SHT_STRTAB 3
#define IREE_HAL_HIP_SHT_NOTE 7
#define IREE_HAL_HIP_SHT_DYNSYM 11

// Symbol type extraction macro
#define IREE_HAL_HIP_ELF64_ST_TYPE(info) ((info) & 0xf)
#define IREE_HAL_HIP_STT_FUNC 2

// AMD Kernel Descriptor structure (AMDHSA ABI).
// This structure describes kernel launch parameters and resource usage.
// Reference: https://llvm.org/docs/AMDGPUUsage.html#kernel-descriptor
typedef struct iree_hal_hip_amd_kernel_descriptor_t {
  uint32_t group_segment_fixed_size;
  uint32_t private_segment_fixed_size;
  uint32_t kernarg_size;  // Total size of kernel arguments
  uint8_t reserved0[4];
  int64_t kernel_code_entry_byte_offset;
  uint8_t reserved1[20];
  uint32_t compute_pgm_rsrc3;
  uint32_t compute_pgm_rsrc1;
  uint32_t compute_pgm_rsrc2;
  uint16_t kernel_code_properties;
  uint8_t reserved2[6];
} iree_hal_hip_amd_kernel_descriptor_t;

// Kernel parameter information.
// Each kernel parameter has a type, size, and offset within the kernarg buffer.
typedef struct iree_hal_hip_kernel_param_t {
  uint32_t offset;  // Offset within kernarg buffer
  uint32_t size;    // Size in bytes
  uint8_t type;     // Parameter type (0=value, 1=pointer, etc.)
} iree_hal_hip_kernel_param_t;

// Kernel information extracted from an ELF file.
typedef struct iree_hal_hip_kernel_info_t {
  iree_string_view_t name;  // Kernel function name

  // Pointer to allocated name storage (NULL if name is not owned/copied).
  // When non-NULL, this memory must be freed when the kernel info is destroyed.
  char* allocated_name;

  // Block dimensions (workgroup size hints from metadata).
  uint32_t block_dims[3];

  // Number of bindings (typically pointer parameters).
  uint32_t binding_count;

  // Number of push constants (typically value parameters).
  uint32_t constant_count;

  // Actual number of parameters in the 'parameters' array.
  // This is the total count of all parsed arguments from metadata.
  uint32_t parameter_count;

  // Array of parameter information (allocated separately).
  // Total count is 'parameter_count'.
  iree_hal_hip_kernel_param_t* parameters;
} iree_hal_hip_kernel_info_t;

// Parsed fat binary information.
typedef struct iree_hal_hip_fat_binary_info_t {
  // Pointer to the bundle data (within the original executable data).
  const uint8_t* bundle_data;
  // Size of the entire bundle.
  iree_host_size_t bundle_size;
  // Number of kernels found across all ELF files.
  iree_host_size_t kernel_count;
  // Array of kernel information (allocated separately).
  iree_hal_hip_kernel_info_t* kernels;
} iree_hal_hip_fat_binary_info_t;

// Reads and validates native HIP executable data wrapped in a fat binary.
// This function parses the fat binary wrapper, then detects the inner format
// (compressed bundle, uncompressed bundle, or raw ELF) and returns the
// appropriate ELF data range in |out_elf_data|.
//
// Expected structure:
//   [Fat Binary Header (24 bytes)][Binary Data (variable size)]
//
// The binary data can be in one of these formats:
//   1. Compressed offload bundle (starts with "CCOB") - PARTIALLY IMPLEMENTED
//      - CCOB header parsing is supported
//      - Decompression requires linking with zlib/zstd (NOT YET IMPLEMENTED)
//   2. Uncompressed offload bundle (starts with "__CLANG_OFFLOAD_BUNDLE__")
//   3. Raw ELF binary (starts with ELF magic bytes)
//
// For uncompressed offload bundles, this function parses the bundle structure
// and extracts the first ELF binary found. For compressed bundles, an error
// is returned. For raw ELF, it validates and returns the data directly.
//
// When |unsafe_infer_size| is true, the total size is inferred from the fat
// binary header offset and the ELF header's section table information.
iree_status_t iree_hal_hip_read_native_header(
    iree_const_byte_span_t executable_data, bool unsafe_infer_size,
    iree_const_byte_span_t* out_elf_data);

// Parses a fat binary and extracts kernel names from the ELF file matching
// the specified triple. The caller must free the kernel array using
// iree_hal_hip_free_kernel_info.
//
// |target_triple| specifies which ELF to extract kernels from (e.g.,
// "hip-amdgcn-amd-amdhsa--gfx942"). Only the ELF matching this triple will be
// parsed.
//
// |out_info| will contain:
//   - bundle_data: Pointer to the bundle data within executable_data
//   - bundle_size: Total size of the bundle
//   - kernel_count: Number of kernels found in the matching ELF
//   - kernels: Array of kernel info (must be freed with iree_hal_hip_free_kernel_info)
iree_status_t iree_hal_hip_parse_fat_binary_kernels(
    iree_const_byte_span_t executable_data, iree_string_view_t target_triple,
    iree_allocator_t allocator, iree_hal_hip_fat_binary_info_t* out_info);

// Frees kernel info array and any allocated kernel names.
// Walks through the kernel array and frees any individual allocated names
// (where allocated_name is non-NULL), then frees the array itself.
void iree_hal_hip_free_kernel_info(iree_allocator_t allocator,
                                    iree_host_size_t kernel_count,
                                    iree_hal_hip_kernel_info_t* kernels);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_HIP_NATIVE_EXECUTABLE_ELF_H_
