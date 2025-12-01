// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HSA_NATIVE_EXECUTABLE_HSAF_H_
#define IREE_HAL_DRIVERS_HSA_NATIVE_EXECUTABLE_HSAF_H_

// HSA fat binary format - reuses HIP's fat binary parsing since both
// target AMD GPUs with the same executable format (HSACO).

#include "iree/hal/drivers/hip/native_executable_hipf.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Alias HIP's fat binary structures for HSA use.
// Both HIP and HSA target AMD GPUs and use compatible binary formats.
typedef iree_hal_hip_fat_binary_header_t iree_hal_hsa_fat_binary_header_t;
typedef iree_hal_hip_bundle_entry_t iree_hal_hsa_bundle_entry_t;
typedef iree_hal_hip_elf64_header_t iree_hal_hsa_elf64_header_t;
typedef iree_hal_hip_elf64_section_header_t iree_hal_hsa_elf64_section_header_t;
typedef iree_hal_hip_elf64_symbol_t iree_hal_hsa_elf64_symbol_t;
typedef iree_hal_hip_amd_kernel_descriptor_t iree_hal_hsa_amd_kernel_descriptor_t;
typedef iree_hal_hip_kernel_param_t iree_hal_hsa_kernel_param_t;
typedef iree_hal_hip_kernel_info_t iree_hal_hsa_kernel_info_t;
typedef iree_hal_hip_fat_binary_info_t iree_hal_hsa_fat_binary_info_t;

// Magic numbers (same as HIP since both use AMD GPU binaries)
#define IREE_HAL_HSA_FAT_BINARY_MAGIC IREE_HAL_HIP_FAT_BINARY_MAGIC
#define IREE_HAL_HSA_FAT_BINARY_VERSION IREE_HAL_HIP_FAT_BINARY_VERSION
#define IREE_HAL_HSA_OFFLOAD_BUNDLE_MAGIC IREE_HAL_HIP_OFFLOAD_BUNDLE_MAGIC
#define IREE_HAL_HSA_OFFLOAD_BUNDLE_MAGIC_SIZE IREE_HAL_HIP_OFFLOAD_BUNDLE_MAGIC_SIZE
#define IREE_HAL_HSA_OFFLOAD_BUNDLE_COMPRESSED_MAGIC IREE_HAL_HIP_OFFLOAD_BUNDLE_COMPRESSED_MAGIC
#define IREE_HAL_HSA_OFFLOAD_BUNDLE_COMPRESSED_MAGIC_SIZE IREE_HAL_HIP_OFFLOAD_BUNDLE_COMPRESSED_MAGIC_SIZE
#define IREE_HAL_HSA_OFFLOAD_BUNDLE_COMPRESSED_MAGIC_INT IREE_HAL_HIP_OFFLOAD_BUNDLE_COMPRESSED_MAGIC_INT
#define IREE_HAL_HSA_ELF_MAGIC IREE_HAL_HIP_ELF_MAGIC
#define IREE_HAL_HSA_ELF_MAGIC0 IREE_HAL_HIP_ELF_MAGIC0
#define IREE_HAL_HSA_ELF_MAGIC1 IREE_HAL_HIP_ELF_MAGIC1
#define IREE_HAL_HSA_ELF_MAGIC2 IREE_HAL_HIP_ELF_MAGIC2
#define IREE_HAL_HSA_ELF_MAGIC3 IREE_HAL_HIP_ELF_MAGIC3
#define IREE_HAL_HSA_ELFCLASS64 IREE_HAL_HIP_ELFCLASS64
#define IREE_HAL_HSA_ELFDATA2LSB IREE_HAL_HIP_ELFDATA2LSB
#define IREE_HAL_HSA_EM_AMDGPU IREE_HAL_HIP_EM_AMDGPU
#define IREE_HAL_HSA_SHT_NULL IREE_HAL_HIP_SHT_NULL
#define IREE_HAL_HSA_SHT_PROGBITS IREE_HAL_HIP_SHT_PROGBITS
#define IREE_HAL_HSA_SHT_SYMTAB IREE_HAL_HIP_SHT_SYMTAB
#define IREE_HAL_HSA_SHT_STRTAB IREE_HAL_HIP_SHT_STRTAB
#define IREE_HAL_HSA_SHT_NOTE IREE_HAL_HIP_SHT_NOTE
#define IREE_HAL_HSA_SHT_DYNSYM IREE_HAL_HIP_SHT_DYNSYM
#define IREE_HAL_HSA_ELF64_ST_TYPE IREE_HAL_HIP_ELF64_ST_TYPE
#define IREE_HAL_HSA_STT_FUNC IREE_HAL_HIP_STT_FUNC

// Reads and validates native HSA executable data wrapped in a fat binary.
// This directly delegates to the HIP implementation since both target AMD GPUs.
static inline iree_status_t iree_hal_hsa_read_native_header(
    iree_const_byte_span_t executable_data, bool unsafe_infer_size,
    iree_const_byte_span_t* out_elf_data) {
  return iree_hal_hip_read_native_header(executable_data, unsafe_infer_size,
                                          out_elf_data);
}

// Parses a fat binary and extracts kernel names from the ELF file matching
// the specified triple. This directly delegates to the HIP implementation.
static inline iree_status_t iree_hal_hsa_parse_fat_binary_kernels(
    iree_const_byte_span_t executable_data, iree_string_view_t target_triple,
    iree_allocator_t allocator, iree_hal_hsa_fat_binary_info_t* out_info) {
  return iree_hal_hip_parse_fat_binary_kernels(
      executable_data, target_triple, allocator,
      (iree_hal_hip_fat_binary_info_t*)out_info);
}

// Frees kernel info array and any allocated kernel names.
// This directly delegates to the HIP implementation.
static inline void iree_hal_hsa_free_kernel_info(
    iree_allocator_t allocator, iree_host_size_t kernel_count,
    iree_hal_hsa_kernel_info_t* kernels) {
  iree_hal_hip_free_kernel_info(allocator, kernel_count,
                                 (iree_hal_hip_kernel_info_t*)kernels);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_HSA_NATIVE_EXECUTABLE_HSAF_H_

