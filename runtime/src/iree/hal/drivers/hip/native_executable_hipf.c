// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/native_executable_hipf.h"

#include <string.h>

#include "iree/base/api.h"

#ifdef IREE_HAL_HIP_ENABLE_ZSTD
  #include "zstd.h"
#endif


// CCOB (Compressed Clang Offload Bundle) header structure
typedef struct iree_hal_hip_ccob_header_v1_t {
  uint8_t magic[4];           // "CCOB"
  uint16_t version;           // Version number
  uint16_t method;            // Compression method (0=zlib, 1=zstd)
  uint32_t uncompressed_size; // Size of uncompressed data
  uint64_t hash;              // Hash of uncompressed data
} iree_hal_hip_ccob_header_v1_t;

typedef struct iree_hal_hip_ccob_header_v2_t {
  uint8_t magic[4];           // "CCOB"
  uint16_t version;           // Version number (2+)
  uint16_t method;            // Compression method (0=zlib, 1=zstd)
  uint32_t file_size;         // Total file size
  uint32_t uncompressed_size; // Size of uncompressed data
  uint64_t hash;              // Hash of uncompressed data
} iree_hal_hip_ccob_header_v2_t;

typedef struct iree_hal_hip_ccob_header_v3_t {
  uint8_t magic[4];           // "CCOB"
  uint16_t version;           // Version number (2+)
  uint16_t method;            // Compression method (0=zlib, 1=zstd)
  uint64_t file_size;         // Total file size
  uint64_t uncompressed_size; // Size of uncompressed data
  uint64_t hash;              // Hash of uncompressed data
} iree_hal_hip_ccob_header_v3_t;

// Compression method constants
#define IREE_HAL_HIP_CCOB_METHOD_ZLIB 0
#define IREE_HAL_HIP_CCOB_METHOD_ZSTD 1

// Checks if the data starts with compressed offload bundle magic ("CCOB").
// When unsafe_infer_size is true, assumes the buffer is at least large enough.
static bool iree_hal_hip_is_code_object_compressed(iree_const_byte_span_t data,
                                                   bool unsafe_infer_size) {
  if (!unsafe_infer_size &&
      data.data_length < IREE_HAL_HIP_OFFLOAD_BUNDLE_COMPRESSED_MAGIC_SIZE) {
    return false;
  }
  return memcmp(data.data, IREE_HAL_HIP_OFFLOAD_BUNDLE_COMPRESSED_MAGIC,
                IREE_HAL_HIP_OFFLOAD_BUNDLE_COMPRESSED_MAGIC_SIZE) == 0;
}

// Checks if the data starts with uncompressed offload bundle magic
// ("__CLANG_OFFLOAD_BUNDLE__").
// When unsafe_infer_size is true, assumes the buffer is at least large enough.
static bool iree_hal_hip_is_code_object_uncompressed(
    iree_const_byte_span_t data, bool unsafe_infer_size) {
  if (!unsafe_infer_size &&
      data.data_length < IREE_HAL_HIP_OFFLOAD_BUNDLE_MAGIC_SIZE) {
    return false;
  }
  return memcmp(data.data, IREE_HAL_HIP_OFFLOAD_BUNDLE_MAGIC,
                IREE_HAL_HIP_OFFLOAD_BUNDLE_MAGIC_SIZE) == 0;
}

// Checks if the data starts with ELF magic bytes.
// When unsafe_infer_size is true, assumes the buffer is at least large enough.
static bool iree_hal_hip_is_code_object_elf(iree_const_byte_span_t data,
                                            bool unsafe_infer_size) {
  if (!unsafe_infer_size && data.data_length < 4) {
    return false;
  }
  const uint8_t* bytes = (const uint8_t*)data.data;
  return bytes[0] == IREE_HAL_HIP_ELF_MAGIC0 &&
         bytes[1] == IREE_HAL_HIP_ELF_MAGIC1 &&
         bytes[2] == IREE_HAL_HIP_ELF_MAGIC2 &&
         bytes[3] == IREE_HAL_HIP_ELF_MAGIC3;
}

// Validates the ELF header for AMD GPU compatibility and calculates size.
// Returns the inferred ELF size in |out_elf_size|.
static iree_status_t iree_hal_hip_validate_elf_header(
    iree_const_byte_span_t elf_data, bool unsafe_infer_size,
    iree_host_size_t* out_elf_size) {
  if (!unsafe_infer_size &&
      elf_data.data_length < sizeof(iree_hal_hip_elf64_header_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "ELF data too small (expected at least %zu bytes, got %" PRIhsz
        " bytes)",
        sizeof(iree_hal_hip_elf64_header_t), elf_data.data_length);
  }

  const iree_hal_hip_elf64_header_t* elf_header =
      (const iree_hal_hip_elf64_header_t*)elf_data.data;

  // Validate ELF magic bytes (already checked by is_code_object_elf, but
  // double-check).
  if (elf_header->magic[0] != IREE_HAL_HIP_ELF_MAGIC0 ||
      elf_header->magic[1] != IREE_HAL_HIP_ELF_MAGIC1 ||
      elf_header->magic[2] != IREE_HAL_HIP_ELF_MAGIC2 ||
      elf_header->magic[3] != IREE_HAL_HIP_ELF_MAGIC3) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid ELF magic bytes");
  }

  // Validate ELF class (must be 64-bit).
  if (elf_header->class != IREE_HAL_HIP_ELFCLASS64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF class must be 64-bit (ELFCLASS64), got %d",
                            elf_header->class);
  }

  // Validate data encoding (must be little-endian).
  if (elf_header->data != IREE_HAL_HIP_ELFDATA2LSB) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF data encoding must be little-endian, got %d",
                            elf_header->data);
  }

  // Validate machine type (must be AMD GPU).
  if (elf_header->machine != IREE_HAL_HIP_EM_AMDGPU) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "ELF machine type must be AMD GPU (EM_AMDGPU=%d), got %d",
        IREE_HAL_HIP_EM_AMDGPU, elf_header->machine);
  }

  // Calculate the ELF size based on section header table.
  // ELF size = section header offset + (section header size * number of
  // sections)
  iree_host_size_t elf_size =
      (iree_host_size_t)(elf_header->shoff +
                         (elf_header->shentsize * elf_header->shnum));

  // Validate that the calculated size is reasonable.
  if (!unsafe_infer_size && elf_size > elf_data.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF file claims size of %" PRIhsz
                            " bytes but only %" PRIhsz " bytes available",
                            elf_size, elf_data.data_length);
  }

  if (out_elf_size) {
    *out_elf_size = elf_size;
  }

  return iree_ok_status();
}

// Validates CCOB (Compressed Clang Offload Bundle) header and calculates total
// file size. Returns the total size of the CCOB file in |out_ccob_size|.
static iree_status_t iree_hal_hip_validate_ccob_header(
    iree_const_byte_span_t ccob_data, bool unsafe_infer_size,
    iree_host_size_t* out_ccob_size) {
  // Minimum size for v1 header
  const iree_host_size_t v1_header_size =
      sizeof(iree_hal_hip_ccob_header_v1_t);
  const iree_host_size_t v2_header_size =
      sizeof(iree_hal_hip_ccob_header_v2_t);
  const iree_host_size_t v3_header_size =
      sizeof(iree_hal_hip_ccob_header_v3_t);

  if (!unsafe_infer_size && ccob_data.data_length < v1_header_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CCOB data too small for header (expected at least "
                            "%" PRIhsz " bytes, got %" PRIhsz " bytes)",
                            v1_header_size, ccob_data.data_length);
  }

  // Validate magic bytes
  if (memcmp(ccob_data.data, IREE_HAL_HIP_OFFLOAD_BUNDLE_COMPRESSED_MAGIC,
             IREE_HAL_HIP_OFFLOAD_BUNDLE_COMPRESSED_MAGIC_SIZE) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid CCOB magic bytes");
  }

  // Read version to determine header structure
  uint16_t version;
  memcpy(&version, ccob_data.data + 4, sizeof(uint16_t));

  if (version == 1) {
    // V1: header size + compressed data size is inferred from file
    // Since v1 doesn't have file_size field, we need to use data_length
    if (!unsafe_infer_size) {
      *out_ccob_size = ccob_data.data_length;
    } else {
      // When unsafe_infer_size is true, we cannot determine size for v1
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "CCOB v1 does not store total file size; cannot infer size safely");
    }
  } else if (version >= 2) {
    if (!unsafe_infer_size && ccob_data.data_length < v2_header_size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "CCOB v2+ data too small for header");
    }

    // V2+: file_size field contains the total size
    iree_hal_hip_ccob_header_v2_t header;
    memcpy(&header, ccob_data.data, sizeof(header));

    // Validate that claimed file size is reasonable
    if (!unsafe_infer_size && header.file_size > ccob_data.data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "CCOB v2 claims file size of %" PRIu32
                              " bytes but only %" PRIhsz " bytes available",
                              header.file_size, ccob_data.data_length);
    }

    *out_ccob_size = (iree_host_size_t)header.file_size;
  } else if (version >= 3) {
    if (!unsafe_infer_size && ccob_data.data_length < v3_header_size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "CCOB v2+ data too small for header");
    }

    // V2+: file_size field contains the total size
    iree_hal_hip_ccob_header_v3_t header;
    memcpy(&header, ccob_data.data, sizeof(header));

    // Validate that claimed file size is reasonable
    if (!unsafe_infer_size && header.file_size > ccob_data.data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "CCOB v2 claims file size of %" PRIu64
                              " bytes but only %" PRIhsz " bytes available",
                              header.file_size, ccob_data.data_length);
    }

    *out_ccob_size = (iree_host_size_t)header.file_size;
  } else {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "unsupported CCOB version %u", version);
  }

  return iree_ok_status();
}

// Parses a CCOB (Compressed Clang Offload Bundle) header and validates it.
// Returns the decompressed size and compression method.
static iree_status_t iree_hal_hip_parse_ccob_header(
    iree_const_byte_span_t ccob_data, bool unsafe_infer_size,
    uint16_t* out_version, uint16_t* out_method,
    uint32_t* out_uncompressed_size, uint32_t* out_file_size,
    iree_host_size_t* out_compressed_data_offset) {
  // Minimum size for v1 header
  const iree_host_size_t v1_header_size = sizeof(iree_hal_hip_ccob_header_v1_t);
  const iree_host_size_t v2_header_size = sizeof(iree_hal_hip_ccob_header_v2_t);
  const iree_host_size_t v3_header_size = sizeof(iree_hal_hip_ccob_header_v3_t);

  if (!unsafe_infer_size && ccob_data.data_length < v1_header_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CCOB data too small for header (expected at least "
                            "%" PRIhsz " bytes, got %" PRIhsz " bytes)",
                            v1_header_size, ccob_data.data_length);
  }

  // Read version first
  uint16_t version;
  memcpy(&version, ccob_data.data + 4, sizeof(uint16_t));
  *out_version = version;

  if (version == 1) {
    iree_hal_hip_ccob_header_v1_t header;
    memcpy(&header, ccob_data.data, sizeof(header));

    *out_method = header.method;
    *out_uncompressed_size = header.uncompressed_size;
    *out_file_size = 0;  // Not available in v1
    *out_compressed_data_offset = v1_header_size;
  } else if (version >= 2) {
    if (!unsafe_infer_size && ccob_data.data_length < v2_header_size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "CCOB v2+ data too small for header");
    }

    iree_hal_hip_ccob_header_v2_t header;
    memcpy(&header, ccob_data.data, sizeof(header));

    *out_method = header.method;
    *out_uncompressed_size = header.uncompressed_size;
    *out_file_size = header.file_size;
    *out_compressed_data_offset = v2_header_size;
  } else if (version >= 3) {
    if (!unsafe_infer_size && ccob_data.data_length < v3_header_size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "CCOB v2+ data too small for header");
    }

    iree_hal_hip_ccob_header_v3_t header;
    memcpy(&header, ccob_data.data, sizeof(header));

    *out_method = header.method;
    *out_uncompressed_size = header.uncompressed_size;
    *out_file_size = header.file_size;
    *out_compressed_data_offset = v3_header_size;
  } else {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "unsupported CCOB version %u", version);
  }

  // Validate compression method
  if (*out_method != IREE_HAL_HIP_CCOB_METHOD_ZLIB &&
      *out_method != IREE_HAL_HIP_CCOB_METHOD_ZSTD) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown CCOB compression method %u", *out_method);
  }

  return iree_ok_status();
}

// Decompresses a CCOB (Compressed Clang Offload Bundle).
// Supports zstd compression. zlib support would require additional dependencies.
static iree_status_t iree_hal_hip_decompress_ccob(
    iree_const_byte_span_t compressed_data, uint16_t method,
    uint32_t uncompressed_size, iree_allocator_t allocator,
    uint8_t** out_decompressed_data,
    iree_host_size_t* out_decompressed_size) {
  if (method == IREE_HAL_HIP_CCOB_METHOD_ZLIB) {
    // zlib support not yet implemented
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "CCOB decompression with zlib is not yet implemented. "
        "The bundle contains zlib-compressed data (%" PRIu32
        " bytes uncompressed). "
        "Please use zstd compression or an uncompressed bundle.",
        uncompressed_size);
  }

  if (method != IREE_HAL_HIP_CCOB_METHOD_ZSTD) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown CCOB compression method %u", method);
  }

  // Allocate buffer for decompressed data
  uint8_t* decompressed_data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, (iree_host_size_t)uncompressed_size,
      (void**)&decompressed_data));

  // Decompress using zstd
  size_t decompressed_size = ZSTD_decompress(
      decompressed_data, (size_t)uncompressed_size, compressed_data.data,
      compressed_data.data_length);

  // Check for errors
  if (ZSTD_isError(decompressed_size)) {
    iree_allocator_free(allocator, decompressed_data);
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "zstd decompression failed: %s",
                            ZSTD_getErrorName(decompressed_size));
  }

  // Verify decompressed size matches expected size
  if (decompressed_size != (size_t)uncompressed_size) {
    iree_allocator_free(allocator, decompressed_data);
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "zstd decompressed size mismatch: expected %" PRIu32
        " bytes, got %zu bytes",
        uncompressed_size, decompressed_size);
  }

  *out_decompressed_data = decompressed_data;
  *out_decompressed_size = (iree_host_size_t)decompressed_size;
  return iree_ok_status();
}

// Forward declaration
static iree_status_t iree_hal_hip_parse_elf_kernels(
    iree_const_byte_span_t elf_data, iree_allocator_t allocator,
    bool copy_kernel_names, iree_host_size_t* kernel_count,
    iree_hal_hip_kernel_info_t** out_kernels);

// Parses an uncompressed offload bundle and extracts kernels from the ELF
// matching the target triple.
//
// Bundle format:
//   [Magic (24 bytes)][Number of bundles (8 bytes)]
//   [Entry 0: offset, size, triple_size (24 bytes)][Triple string (variable)]
//   [Entry 1: ...]
//   [Binary data for entry 0]
//   [Binary data for entry 1]
//
// If copy_kernel_names is true, kernel names will be copied (needed when
// bundle_data is temporary, e.g., from decompression).
//
// Returns:
//   - matched_kernel_count: Number of kernels found in the matching ELF
//   - matched_kernels: Array of kernel info (caller must free)
//   - bundle_size: Total size of the bundle
static iree_status_t iree_hal_hip_parse_offload_bundle_kernels(
    iree_const_byte_span_t bundle_data, iree_string_view_t target_triple,
    iree_allocator_t allocator, bool copy_kernel_names,
    iree_host_size_t* out_matched_kernel_count,
    iree_hal_hip_kernel_info_t** out_matched_kernels,
    iree_host_size_t* out_bundle_size) {
  // Skip magic (already validated by caller).
  const uint8_t* bundle_ptr =
      bundle_data.data + IREE_HAL_HIP_OFFLOAD_BUNDLE_MAGIC_SIZE;
  iree_host_size_t remaining =
      bundle_data.data_length - IREE_HAL_HIP_OFFLOAD_BUNDLE_MAGIC_SIZE;

  // Read number of bundles.
  if (remaining < sizeof(uint64_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "offload bundle too small for bundle count");
  }

  uint64_t num_bundles;
  memcpy(&num_bundles, bundle_ptr, sizeof(num_bundles));
  bundle_ptr += sizeof(num_bundles);
  remaining -= sizeof(num_bundles);

  if (num_bundles == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "offload bundle contains no entries");
  }

  // Track the maximum extent of all ELF files to calculate total bundle size.
  uint64_t max_elf_end = 0;
  iree_hal_hip_kernel_info_t* matched_kernels = NULL;
  iree_host_size_t matched_kernel_count = 0;
  bool found_match = false;

  // Process all bundle entries.
  for (uint64_t i = 0; i < num_bundles; ++i) {
    // Read bundle entry header.
    if (remaining < sizeof(iree_hal_hip_bundle_entry_t)) {
      iree_hal_hip_free_kernel_info(allocator, matched_kernel_count,
                                     matched_kernels);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "offload bundle too small for entry header [%" PRIu64 "]", i);
    }

    iree_hal_hip_bundle_entry_t entry;
    memcpy(&entry, bundle_ptr, sizeof(entry));
    bundle_ptr += sizeof(entry);
    remaining -= sizeof(entry);

    // Skip the triple string.
    if (remaining < entry.triple_size) {
      iree_hal_hip_free_kernel_info(allocator, matched_kernel_count,
                                     matched_kernels);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "offload bundle too small for triple string [%" PRIu64 "]", i);
    }

    // Read the triple for comparison
    iree_string_view_t entry_triple =
        iree_make_string_view((const char*)bundle_ptr, entry.triple_size);
    bundle_ptr += entry.triple_size;
    remaining -= entry.triple_size;

    // The entry offset is from the start of the bundle data.
    // Validate the offset and extract the ELF binary.
    const uint8_t* entry_data = bundle_data.data + (iree_host_size_t)entry.offset;
    iree_host_size_t entry_size = (iree_host_size_t)entry.size;

    // Validate entry is within bundle_data bounds.
    if (entry.offset >= bundle_data.data_length ||
        entry.offset + entry_size > bundle_data.data_length) {
      iree_hal_hip_free_kernel_info(allocator, matched_kernel_count,
                                     matched_kernels);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "offload bundle entry [%" PRIu64
                              "] offset %" PRIu64 " size %" PRIu64
                              " is out of bounds",
                              i, entry.offset, entry.size);
    }

    // Verify it's an ELF file.
    iree_const_byte_span_t entry_span =
        iree_make_const_byte_span(entry_data, entry_size);
    if (!iree_hal_hip_is_code_object_elf(entry_span, true)) {
      iree_hal_hip_free_kernel_info(allocator, matched_kernel_count,
                                     matched_kernels);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "offload bundle entry [%" PRIu64
                              "] does not contain an ELF binary",
                              i);
    }

    // Validate and get the ELF size.
    iree_host_size_t elf_size = 0;
    iree_status_t elf_status =
        iree_hal_hip_validate_elf_header(entry_span, true, &elf_size);
    if (!iree_status_is_ok(elf_status)) {
      iree_hal_hip_free_kernel_info(allocator, matched_kernel_count,
                                     matched_kernels);
      return elf_status;
    }

    // Update the maximum extent.
    uint64_t elf_end = entry.offset + elf_size;
    if (elf_end > max_elf_end) {
      max_elf_end = elf_end;
    }

    // Check if this triple matches the target
    if (iree_string_view_equal(entry_triple, target_triple)) {
      found_match = true;

      // Parse kernels from this ELF
      entry_span.data_length = elf_size;
      iree_status_t status = iree_hal_hip_parse_elf_kernels(
          entry_span, allocator, copy_kernel_names, &matched_kernel_count,
          &matched_kernels);
      if (!iree_status_is_ok(status)) {
        return status;
      }
    }
  }

  if (!found_match) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no ELF found for target triple '%.*s' in offload "
                            "bundle (searched %" PRIu64 " entries)",
                            (int)target_triple.size, target_triple.data,
                            num_bundles);
  }

  // Return results
  *out_matched_kernel_count = matched_kernel_count;
  *out_matched_kernels = matched_kernels;
  if (out_bundle_size) {
    *out_bundle_size = (iree_host_size_t)max_elf_end;
  }

  return iree_ok_status();
}

// Parses an uncompressed offload bundle and validates all ELF binaries.
// Returns the total bundle size in |out_bundle_size| (from start of bundle
// to the end of the last ELF file).
//
// Bundle format:
//   [Magic (24 bytes)][Number of bundles (8 bytes)]
//   [Entry 0: offset, size, triple_size (24 bytes)][Triple string (variable)]
//   [Entry 1: ...]
//   [Binary data for entry 0]
//   [Binary data for entry 1]
static iree_status_t iree_hal_hip_parse_offload_bundle(
    iree_const_byte_span_t bundle_data, bool unsafe_infer_size,
    iree_host_size_t* out_bundle_size) {
  // Skip magic (already validated by caller).
  const uint8_t* bundle_ptr =
      bundle_data.data + IREE_HAL_HIP_OFFLOAD_BUNDLE_MAGIC_SIZE;
  iree_host_size_t remaining =
      unsafe_infer_size
          ? 0
          : (bundle_data.data_length - IREE_HAL_HIP_OFFLOAD_BUNDLE_MAGIC_SIZE);

  // Read number of bundles.
  if (!unsafe_infer_size && remaining < sizeof(uint64_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "offload bundle too small for bundle count");
  }

  uint64_t num_bundles;
  memcpy(&num_bundles, bundle_ptr, sizeof(num_bundles));
  bundle_ptr += sizeof(num_bundles);
  remaining -= unsafe_infer_size ? 0 : sizeof(num_bundles);

  if (num_bundles == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "offload bundle contains no entries");
  }

  // Track the maximum extent of all ELF files to calculate total bundle size.
  uint64_t max_elf_end = 0;

  // Process all bundle entries.
  for (uint64_t i = 0; i < num_bundles; ++i) {
    // Read bundle entry header.
    if (!unsafe_infer_size && remaining < sizeof(iree_hal_hip_bundle_entry_t)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "offload bundle too small for entry header [%" PRIu64 "]", i);
    }

    iree_hal_hip_bundle_entry_t entry;
    memcpy(&entry, bundle_ptr, sizeof(entry));
    bundle_ptr += sizeof(entry);
    remaining -= unsafe_infer_size ? 0 : sizeof(entry);

    // Skip the triple string.
    if (!unsafe_infer_size && remaining < entry.triple_size) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "offload bundle too small for triple string [%" PRIu64 "]", i);
    }
    bundle_ptr += entry.triple_size;
    remaining -= unsafe_infer_size ? 0 : entry.triple_size;

    // The entry offset is from the start of the bundle data.
    // Validate the offset and extract the ELF binary.
    const uint8_t* entry_data =
        bundle_data.data + (iree_host_size_t)entry.offset;
    iree_host_size_t entry_size = (iree_host_size_t)entry.size;

    if (!unsafe_infer_size) {
      // Validate entry is within bundle_data bounds.
      if (entry.offset >= bundle_data.data_length ||
          entry.offset + entry_size > bundle_data.data_length) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "offload bundle entry [%" PRIu64
                                "] offset %" PRIu64 " size %" PRIu64
                                " is out of bounds",
                                i, entry.offset, entry.size);
      }
    }

    // Verify it's an ELF file.
    iree_const_byte_span_t entry_span =
        iree_make_const_byte_span(entry_data, entry_size);
    if (!iree_hal_hip_is_code_object_elf(entry_span, unsafe_infer_size)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "offload bundle entry [%" PRIu64
                              "] does not contain an ELF binary",
                              i);
    }

    // Validate and get the ELF size.
    iree_host_size_t elf_size = 0;
    IREE_RETURN_IF_ERROR(iree_hal_hip_validate_elf_header(
        entry_span, unsafe_infer_size, &elf_size));

    // Update the maximum extent.
    uint64_t elf_end = entry.offset + elf_size;
    if (elf_end > max_elf_end) {
      max_elf_end = elf_end;
    }
  }

  // Calculate the total bundle size (from start of bundle to end of last ELF).
  if (out_bundle_size) {
    *out_bundle_size = (iree_host_size_t)max_elf_end;
  }

  return iree_ok_status();
}

// MessagePack type constants
#define MSGPACK_FIXMAP_MASK 0x80
#define MSGPACK_FIXARRAY_MASK 0x90
#define MSGPACK_FIXSTR_MASK 0xa0
#define MSGPACK_NIL 0xc0
#define MSGPACK_FALSE 0xc2
#define MSGPACK_TRUE 0xc3
#define MSGPACK_UINT8 0xcc
#define MSGPACK_UINT16 0xcd
#define MSGPACK_UINT32 0xce
#define MSGPACK_UINT64 0xcf
#define MSGPACK_STR8 0xd9
#define MSGPACK_STR16 0xda
#define MSGPACK_STR32 0xdb
#define MSGPACK_ARRAY16 0xdc
#define MSGPACK_ARRAY32 0xdd
#define MSGPACK_MAP16 0xde
#define MSGPACK_MAP32 0xdf

// Kernel argument value kinds
typedef enum {
  IREE_HAL_HIP_ARG_KIND_UNKNOWN = 0,
  IREE_HAL_HIP_ARG_KIND_BY_VALUE = 1,
  IREE_HAL_HIP_ARG_KIND_GLOBAL_BUFFER = 2,
  IREE_HAL_HIP_ARG_KIND_HIDDEN = 3,
} iree_hal_hip_arg_value_kind_t;

// Parse a msgpack string at the given offset
static iree_status_t iree_hal_hip_msgpack_read_string(
    const uint8_t* data, iree_host_size_t size, iree_host_size_t offset,
    iree_string_view_t* out_string, iree_host_size_t* out_consumed) {
  if (offset >= size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "msgpack offset OOB");
  }

  uint8_t type = data[offset];
  uint32_t len = 0;
  iree_host_size_t consumed = 1;

  // Fixstr (0xa0 - 0xbf)
  if ((type & 0xe0) == 0xa0) {
    len = type & 0x1f;
  } else if (type == MSGPACK_STR8) {
    if (offset + 1 >= size) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "truncated");
    }
    len = data[offset + 1];
    consumed += 1;
  } else if (type == MSGPACK_STR16) {
    if (offset + 2 >= size) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "truncated");
    }
    len = (data[offset + 1] << 8) + data[offset + 2];
    consumed += 2;
  } else if (type == MSGPACK_STR32) {
    if (offset + 4 >= size) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "truncated");
    }
    len = (data[offset + 1] << 24) + (data[offset + 2] << 16) +
          (data[offset + 3] << 8) + data[offset + 4];
    consumed += 4;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected msgpack string");
  }

  if (offset + consumed + len > size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "string OOB");
  }

  *out_string =
      iree_make_string_view((const char*)data + offset + consumed, len);
  *out_consumed = consumed + len;
  return iree_ok_status();
}

// Parse a msgpack unsigned integer
static iree_status_t iree_hal_hip_msgpack_read_uint(
    const uint8_t* data, iree_host_size_t size, iree_host_size_t offset,
    uint64_t* out_value, iree_host_size_t* out_consumed) {
  if (offset >= size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "offset OOB");
  }

  uint8_t type = data[offset];

  // Positive fixint (0x00 - 0x7f)
  if (type <= 0x7f) {
    *out_value = type;
    *out_consumed = 1;
    return iree_ok_status();
  }

  switch (type) {
    case MSGPACK_UINT8:
      if (offset + 1 >= size) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "truncated");
      }
      *out_value = data[offset + 1];
      *out_consumed = 2;
      return iree_ok_status();

    case MSGPACK_UINT16:
      if (offset + 2 >= size) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "truncated");
      }
      *out_value = (data[offset + 1] << 8) + data[offset + 2];
      *out_consumed = 3;
      return iree_ok_status();

    case MSGPACK_UINT32:
      if (offset + 4 >= size) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "truncated");
      }
      *out_value = ((uint64_t)data[offset + 1] << 24) +
                   (data[offset + 2] << 16) + (data[offset + 3] << 8) +
                   data[offset + 4];
      *out_consumed = 5;
      return iree_ok_status();

    case MSGPACK_UINT64:
      if (offset + 8 >= size) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "truncated");
      }
      *out_value = ((uint64_t)data[offset + 1] << 56) +
                   ((uint64_t)data[offset + 2] << 48) +
                   ((uint64_t)data[offset + 3] << 40) +
                   ((uint64_t)data[offset + 4] << 32) +
                   ((uint64_t)data[offset + 5] << 24) +
                   ((uint64_t)data[offset + 6] << 16) +
                   ((uint64_t)data[offset + 7] << 8) + data[offset + 8];
      *out_consumed = 9;
      return iree_ok_status();

    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected msgpack uint");
  }
}

// Get the element count from a msgpack map
static iree_status_t iree_hal_hip_msgpack_read_map_size(
    const uint8_t* data, iree_host_size_t size, iree_host_size_t offset,
    uint32_t* out_count, iree_host_size_t* out_consumed) {
  if (offset >= size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "offset OOB");
  }

  uint8_t type = data[offset];

  // Fixmap (0x80 - 0x8f)
  if ((type & 0xf0) == 0x80) {
    *out_count = type & 0x0f;
    *out_consumed = 1;
    return iree_ok_status();
  }

  if (type == MSGPACK_MAP16) {
    if (offset + 2 >= size) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "truncated");
    }
    *out_count = (data[offset + 1] << 8) + data[offset + 2];
    *out_consumed = 3;
    return iree_ok_status();
  }

  if (type == MSGPACK_MAP32) {
    if (offset + 4 >= size) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "truncated");
    }
    *out_count = (data[offset + 1] << 24) + (data[offset + 2] << 16) +
                 (data[offset + 3] << 8) + data[offset + 4];
    *out_consumed = 5;
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "expected msgpack map");
}

// Get the element count from a msgpack array
static iree_status_t iree_hal_hip_msgpack_read_array_size(
    const uint8_t* data, iree_host_size_t size, iree_host_size_t offset,
    uint32_t* out_count, iree_host_size_t* out_consumed) {
  if (offset >= size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "offset OOB");
  }

  uint8_t type = data[offset];

  // Fixarray (0x90 - 0x9f)
  if ((type & 0xf0) == 0x90) {
    *out_count = type & 0x0f;
    *out_consumed = 1;
    return iree_ok_status();
  }

  if (type == MSGPACK_ARRAY16) {
    if (offset + 2 >= size) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "truncated");
    }
    *out_count = (data[offset + 1] << 8) + data[offset + 2];
    *out_consumed = 3;
    return iree_ok_status();
  }

  if (type == MSGPACK_ARRAY32) {
    if (offset + 4 >= size) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "truncated");
    }
    *out_count = (data[offset + 1] << 24) + (data[offset + 2] << 16) +
                 (data[offset + 3] << 8) + data[offset + 4];
    *out_consumed = 5;
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "expected msgpack array");
}

// Skip a msgpack value (for skipping unwanted fields)
static iree_host_size_t iree_hal_hip_msgpack_skip_value(
    const uint8_t* data, iree_host_size_t size, iree_host_size_t offset) {
  if (offset >= size) return 0;

  uint8_t type = data[offset];
  iree_host_size_t consumed = 1;

  // Positive fixint (0x00 - 0x7f)
  if (type <= 0x7f) return consumed;

  // Negative fixint (0xe0 - 0xff)
  if (type >= 0xe0) return consumed;

  // Fixmap (0x80 - 0x8f)
  if ((type & 0xf0) == 0x80) {
    uint32_t count = type & 0x0f;
    for (uint32_t i = 0; i < count * 2; ++i) {
      iree_host_size_t skip =
          iree_hal_hip_msgpack_skip_value(data, size, offset + consumed);
      if (skip == 0) return 0;
      consumed += skip;
    }
    return consumed;
  }

  // Fixarray (0x90 - 0x9f)
  if ((type & 0xf0) == 0x90) {
    uint32_t count = type & 0x0f;
    for (uint32_t i = 0; i < count; ++i) {
      iree_host_size_t skip =
          iree_hal_hip_msgpack_skip_value(data, size, offset + consumed);
      if (skip == 0) return 0;
      consumed += skip;
    }
    return consumed;
  }

  // Fixstr (0xa0 - 0xbf)
  if ((type & 0xe0) == 0xa0) {
    uint32_t len = type & 0x1f;
    return consumed + len;
  }

  // Other types
  switch (type) {
    case MSGPACK_NIL:
    case MSGPACK_FALSE:
    case MSGPACK_TRUE:
      return consumed;

    case MSGPACK_UINT8:
      return consumed + 1;
    case MSGPACK_UINT16:
      return consumed + 2;
    case MSGPACK_UINT32:
      return consumed + 4;
    case MSGPACK_UINT64:
      return consumed + 8;

    case MSGPACK_STR8:
      if (offset + 1 >= size) return 0;
      return consumed + 1 + data[offset + 1];

    case MSGPACK_STR16:
      if (offset + 2 >= size) return 0;
      return consumed + 2 + ((uint32_t)data[offset + 1] << 8) +
             data[offset + 2];

    case MSGPACK_STR32:
      if (offset + 4 >= size) return 0;
      return consumed + 4 + ((uint32_t)data[offset + 1] << 24) +
             ((uint32_t)data[offset + 2] << 16) +
             ((uint32_t)data[offset + 3] << 8) + data[offset + 4];

    case MSGPACK_ARRAY16: {
      if (offset + 2 >= size) return 0;
      uint32_t count = (data[offset + 1] << 8) + data[offset + 2];
      consumed += 2;
      for (uint32_t i = 0; i < count; ++i) {
        iree_host_size_t skip =
            iree_hal_hip_msgpack_skip_value(data, size, offset + consumed);
        if (skip == 0) return 0;
        consumed += skip;
      }
      return consumed;
    }

    case MSGPACK_ARRAY32: {
      if (offset + 4 >= size) return 0;
      uint32_t count = ((uint32_t)data[offset + 1] << 24) +
                       ((uint32_t)data[offset + 2] << 16) +
                       ((uint32_t)data[offset + 3] << 8) + data[offset + 4];
      consumed += 4;
      for (uint32_t i = 0; i < count; ++i) {
        iree_host_size_t skip =
            iree_hal_hip_msgpack_skip_value(data, size, offset + consumed);
        if (skip == 0) return 0;
        consumed += skip;
      }
      return consumed;
    }

    case MSGPACK_MAP16: {
      if (offset + 2 >= size) return 0;
      uint32_t count = (data[offset + 1] << 8) + data[offset + 2];
      consumed += 2;
      for (uint32_t i = 0; i < count * 2; ++i) {
        iree_host_size_t skip =
            iree_hal_hip_msgpack_skip_value(data, size, offset + consumed);
        if (skip == 0) return 0;
        consumed += skip;
      }
      return consumed;
    }

    case MSGPACK_MAP32: {
      if (offset + 4 >= size) return 0;
      uint32_t count = ((uint32_t)data[offset + 1] << 24) +
                       ((uint32_t)data[offset + 2] << 16) +
                       ((uint32_t)data[offset + 3] << 8) + data[offset + 4];
      consumed += 4;
      for (uint32_t i = 0; i < count * 2; ++i) {
        iree_host_size_t skip =
            iree_hal_hip_msgpack_skip_value(data, size, offset + consumed);
        if (skip == 0) return 0;
        consumed += skip;
      }
      return consumed;
    }

    default:
      return 0;  // Unknown type
  }
}

// Parsed kernel argument from AMD metadata
typedef struct {
  uint32_t offset;
  uint32_t size;
  iree_hal_hip_arg_value_kind_t value_kind;
} iree_hal_hip_parsed_arg_t;

// Parse AMD kernel metadata for a specific kernel from the .note section
// Extracts .args array and .reqd_workgroup_size
//
// Note format: The .note section starts with an ELF note header, followed by
// msgpack data:
//   [namesz (4 bytes)][descsz (4 bytes)][type (4 bytes)]
//   [name (namesz bytes, padded to 4-byte alignment)]
//   [desc (msgpack data)]
static iree_status_t iree_hal_hip_parse_amd_kernel_metadata(
    const uint8_t* note_data, iree_host_size_t note_size,
    iree_string_view_t kernel_name, iree_allocator_t allocator,
    uint32_t* out_workgroup_size_x, uint32_t* out_workgroup_size_y,
    uint32_t* out_workgroup_size_z, iree_host_size_t* out_arg_count,
    iree_hal_hip_parsed_arg_t** out_args) {
  // Default values
  *out_workgroup_size_x = 1;
  *out_workgroup_size_y = 1;
  *out_workgroup_size_z = 1;
  *out_arg_count = 0;
  *out_args = NULL;

  // Parse ELF note header to find where msgpack data starts
  // Format: namesz (4), descsz (4), type (4), name (namesz padded to 4)
  if (note_size < 12) {
    return iree_ok_status();  // Too small, skip silently
  }

  uint32_t namesz;
  memcpy(&namesz, note_data, sizeof(namesz));

  // Name is padded to 4-byte alignment
  iree_host_size_t name_padded = (namesz + 3) & ~3;
  iree_host_size_t msgpack_offset = 12 + name_padded;

  if (msgpack_offset >= note_size) {
    return iree_ok_status();  // Invalid offset, skip silently
  }

  // AMD metadata is msgpack encoded starting at msgpack_offset
  // Top level is typically a map with keys like:
  // "amdhsa.kernels" -> array of kernel maps
  // Each kernel map has:
  //   ".name" -> kernel name string
  //   ".args" -> array of arg maps
  //   ".reqd_workgroup_size" -> array [x, y, z]
  //
  // We need to:
  // 1. Find the top-level map
  // 2. Find "amdhsa.kernels" key
  // 3. Iterate through kernels array to find matching name
  // 4. Extract .args and .reqd_workgroup_size from that kernel

  iree_host_size_t offset = msgpack_offset;

  // Read top-level map
  uint32_t top_map_size = 0;
  iree_host_size_t consumed = 0;
  iree_status_t status = iree_hal_hip_msgpack_read_map_size(
      note_data, note_size, offset, &top_map_size, &consumed);
  if (!iree_status_is_ok(status)) {
    return status;  // Not fatal - metadata might not be present
  }
  offset += consumed;

  // Look for "amdhsa.kernels" key in top-level map
  for (uint32_t i = 0; i < top_map_size; ++i) {
    // Read key
    iree_string_view_t key;
    status = iree_hal_hip_msgpack_read_string(note_data, note_size, offset,
                                              &key, &consumed);
    if (!iree_status_is_ok(status)) {
      // Skip this key-value pair
      iree_host_size_t skip =
          iree_hal_hip_msgpack_skip_value(note_data, note_size, offset);
      if (skip == 0) break;
      offset += skip;
      skip = iree_hal_hip_msgpack_skip_value(note_data, note_size, offset);
      if (skip == 0) break;
      offset += skip;
      continue;
    }
    offset += consumed;

    // Check if this is "amdhsa.kernels"
    if (iree_string_view_equal(key, iree_make_cstring_view("amdhsa.kernels"))) {
      // Read the kernels array
      uint32_t kernels_count = 0;
      status = iree_hal_hip_msgpack_read_array_size(
          note_data, note_size, offset, &kernels_count, &consumed);
      if (!iree_status_is_ok(status)) break;
      offset += consumed;

      // Iterate through kernels to find matching name
      for (uint32_t k = 0; k < kernels_count; ++k) {
        // Each kernel is a map
        uint32_t kernel_map_size = 0;
        status = iree_hal_hip_msgpack_read_map_size(
            note_data, note_size, offset, &kernel_map_size, &consumed);
        if (!iree_status_is_ok(status)) {
          // Skip this kernel
          iree_host_size_t skip =
              iree_hal_hip_msgpack_skip_value(note_data, note_size, offset);
          if (skip == 0) return iree_ok_status();
          offset += skip;
          continue;
        }
        offset += consumed;

        // Look for ".name" key in kernel map
        bool name_matches = false;
        iree_host_size_t args_offset = 0;
        iree_host_size_t wgs_offset = 0;

        for (uint32_t j = 0; j < kernel_map_size; ++j) {
          // Read key
          iree_string_view_t kernel_key;
          status = iree_hal_hip_msgpack_read_string(
              note_data, note_size, offset, &kernel_key, &consumed);
          if (!iree_status_is_ok(status)) {
            // Skip key and value
            iree_host_size_t skip =
                iree_hal_hip_msgpack_skip_value(note_data, note_size, offset);
            if (skip == 0) break;
            offset += skip;
            skip =
                iree_hal_hip_msgpack_skip_value(note_data, note_size, offset);
            if (skip == 0) break;
            offset += skip;
            continue;
          }
          offset += consumed;

          // Check key type
          if (iree_string_view_equal(kernel_key,
                                     iree_make_cstring_view(".name"))) {
            // Read name value
            iree_string_view_t name_value;
            status = iree_hal_hip_msgpack_read_string(
                note_data, note_size, offset, &name_value, &consumed);
            if (iree_status_is_ok(status) &&
                iree_string_view_equal(name_value, kernel_name)) {
              name_matches = true;
            }
            offset += consumed;
          } else if (iree_string_view_equal(kernel_key,
                                            iree_make_cstring_view(".args"))) {
            // Remember args offset for later parsing
            args_offset = offset;
            // Skip for now
            iree_host_size_t skip =
                iree_hal_hip_msgpack_skip_value(note_data, note_size, offset);
            if (skip == 0) break;
            offset += skip;
          } else if (iree_string_view_equal(
                         kernel_key,
                         iree_make_cstring_view(".reqd_workgroup_size"))) {
            // Remember workgroup size offset
            wgs_offset = offset;
            // Skip for now
            iree_host_size_t skip =
                iree_hal_hip_msgpack_skip_value(note_data, note_size, offset);
            if (skip == 0) break;
            offset += skip;
          } else {
            // Skip unknown key's value
            iree_host_size_t skip =
                iree_hal_hip_msgpack_skip_value(note_data, note_size, offset);
            if (skip == 0) break;
            offset += skip;
          }
        }

        // If name matches, parse .args and .reqd_workgroup_size
        if (name_matches) {
          // Parse .reqd_workgroup_size if present
          if (wgs_offset > 0) {
            uint32_t wgs_array_size = 0;
            status = iree_hal_hip_msgpack_read_array_size(
                note_data, note_size, wgs_offset, &wgs_array_size, &consumed);
            if (iree_status_is_ok(status) && wgs_array_size >= 3) {
              iree_host_size_t wgs_off = wgs_offset + consumed;
              uint64_t val;
              // X
              status = iree_hal_hip_msgpack_read_uint(note_data, note_size,
                                                      wgs_off, &val, &consumed);
              if (iree_status_is_ok(status)) {
                *out_workgroup_size_x = (uint32_t)val;
                wgs_off += consumed;
              }
              // Y
              status = iree_hal_hip_msgpack_read_uint(note_data, note_size,
                                                      wgs_off, &val, &consumed);
              if (iree_status_is_ok(status)) {
                *out_workgroup_size_y = (uint32_t)val;
                wgs_off += consumed;
              }
              // Z
              status = iree_hal_hip_msgpack_read_uint(note_data, note_size,
                                                      wgs_off, &val, &consumed);
              if (iree_status_is_ok(status)) {
                *out_workgroup_size_z = (uint32_t)val;
              }
            }
          }

          // Parse .args array
          if (args_offset > 0) {
            uint32_t args_array_size = 0;
            status = iree_hal_hip_msgpack_read_array_size(
                note_data, note_size, args_offset, &args_array_size, &consumed);
            if (!iree_status_is_ok(status)) {
              return iree_ok_status();  // Found kernel but no args
            }

            // Allocate args array
            iree_hal_hip_parsed_arg_t* args = NULL;
            IREE_RETURN_IF_ERROR(iree_allocator_malloc(
                allocator, args_array_size * sizeof(iree_hal_hip_parsed_arg_t),
                (void**)&args));
            memset(args, 0,
                   args_array_size * sizeof(iree_hal_hip_parsed_arg_t));

            iree_host_size_t arg_offset = args_offset + consumed;
            iree_host_size_t explicit_count = 0;

            // Parse each argument
            for (uint32_t a = 0; a < args_array_size; ++a) {
              // Each arg is a map
              uint32_t arg_map_size = 0;
              status = iree_hal_hip_msgpack_read_map_size(
                  note_data, note_size, arg_offset, &arg_map_size, &consumed);
              if (!iree_status_is_ok(status)) {
                iree_host_size_t skip = iree_hal_hip_msgpack_skip_value(
                    note_data, note_size, arg_offset);
                if (skip == 0) break;
                arg_offset += skip;
                continue;
              }
              arg_offset += consumed;

              bool is_hidden = false;
              uint32_t arg_offset_val = 0;
              uint32_t arg_size_val = 0;
              iree_hal_hip_arg_value_kind_t value_kind =
                  IREE_HAL_HIP_ARG_KIND_UNKNOWN;

              // Parse arg map fields
              for (uint32_t f = 0; f < arg_map_size; ++f) {
                iree_string_view_t field_key;
                status = iree_hal_hip_msgpack_read_string(
                    note_data, note_size, arg_offset, &field_key, &consumed);
                if (!iree_status_is_ok(status)) {
                  // Skip
                  iree_host_size_t skip = iree_hal_hip_msgpack_skip_value(
                      note_data, note_size, arg_offset);
                  if (skip == 0) break;
                  arg_offset += skip;
                  skip = iree_hal_hip_msgpack_skip_value(note_data, note_size,
                                                         arg_offset);
                  if (skip == 0) break;
                  arg_offset += skip;
                  continue;
                }
                arg_offset += consumed;

                if (iree_string_view_equal(field_key,
                                           iree_make_cstring_view(".offset"))) {
                  uint64_t val;
                  status = iree_hal_hip_msgpack_read_uint(
                      note_data, note_size, arg_offset, &val, &consumed);
                  if (iree_status_is_ok(status)) {
                    arg_offset_val = (uint32_t)val;
                  }
                  arg_offset += consumed;
                } else if (iree_string_view_equal(
                               field_key, iree_make_cstring_view(".size"))) {
                  uint64_t val;
                  status = iree_hal_hip_msgpack_read_uint(
                      note_data, note_size, arg_offset, &val, &consumed);
                  if (iree_status_is_ok(status)) {
                    arg_size_val = (uint32_t)val;
                  }
                  arg_offset += consumed;
                } else if (iree_string_view_equal(
                               field_key,
                               iree_make_cstring_view(".value_kind"))) {
                  iree_string_view_t vk_str;
                  status = iree_hal_hip_msgpack_read_string(
                      note_data, note_size, arg_offset, &vk_str, &consumed);
                  if (iree_status_is_ok(status)) {
                    if (iree_string_view_equal(
                            vk_str, iree_make_cstring_view("global_buffer"))) {
                      value_kind = IREE_HAL_HIP_ARG_KIND_GLOBAL_BUFFER;
                    } else if (iree_string_view_equal(
                                   vk_str,
                                   iree_make_cstring_view("by_value"))) {
                      value_kind = IREE_HAL_HIP_ARG_KIND_BY_VALUE;
                    } else if (iree_string_view_starts_with(
                                   vk_str, iree_make_cstring_view("hidden_"))) {
                      is_hidden = true;
                      value_kind = IREE_HAL_HIP_ARG_KIND_HIDDEN;
                    }
                  }
                  arg_offset += consumed;
                } else if (iree_string_view_equal(
                               field_key, iree_make_cstring_view(".name"))) {
                  // Check if name starts with "hidden_"
                  iree_string_view_t name_str;
                  status = iree_hal_hip_msgpack_read_string(
                      note_data, note_size, arg_offset, &name_str, &consumed);
                  if (iree_status_is_ok(status) &&
                      iree_string_view_starts_with(
                          name_str, iree_make_cstring_view("hidden_"))) {
                    is_hidden = true;
                  }
                  arg_offset += consumed;
                } else {
                  // Skip unknown field
                  iree_host_size_t skip = iree_hal_hip_msgpack_skip_value(
                      note_data, note_size, arg_offset);
                  if (skip == 0) break;
                  arg_offset += skip;
                }
              }

              // Only store explicit (non-hidden) arguments
              if (!is_hidden && value_kind != IREE_HAL_HIP_ARG_KIND_UNKNOWN) {
                args[explicit_count].offset = arg_offset_val;
                args[explicit_count].size = arg_size_val;
                args[explicit_count].value_kind = value_kind;
                explicit_count++;
              }
            }

            *out_arg_count = explicit_count;
            *out_args = args;
          }

          return iree_ok_status();  // Found and parsed our kernel
        }

        // Name didn't match, skip to next kernel
        // (offset already advanced through kernel map)
      }

      break;  // Processed amdhsa.kernels, no need to continue
    } else {
      // Skip this value
      iree_host_size_t skip =
          iree_hal_hip_msgpack_skip_value(note_data, note_size, offset);
      if (skip == 0) break;
      offset += skip;
    }
  }

  return iree_ok_status();  // Not found, but not an error
}

// Helper function to read a null-terminated string from a string table.
static iree_status_t iree_hal_hip_read_string_from_table(
    const uint8_t* string_table, iree_host_size_t string_table_size,
    uint32_t offset, iree_string_view_t* out_string) {
  if (offset >= string_table_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "string offset %" PRIu32
                            " out of bounds (table size %" PRIhsz ")",
                            offset, string_table_size);
  }

  const char* str = (const char*)(string_table + offset);
  iree_host_size_t max_len = string_table_size - offset;

  // Find the null terminator
  iree_host_size_t len = 0;
  while (len < max_len && str[len] != '\0') {
    ++len;
  }

  if (len == max_len) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "string at offset %" PRIu32 " not null-terminated",
                            offset);
  }

  *out_string = iree_make_string_view(str, len);
  return iree_ok_status();
}

// Parses an ELF file and extracts kernel function information from the symbol
// table, including parameters from kernel descriptors.
//
// If copy_kernel_names is true, kernel names are copied into the allocation
// (appended after the kernel_info array). This is necessary when the ELF data
// is temporary (e.g., from decompression) and will be freed.
static iree_status_t iree_hal_hip_parse_elf_kernels(
    iree_const_byte_span_t elf_data, iree_allocator_t allocator,
    bool copy_kernel_names, iree_host_size_t* kernel_count,
    iree_hal_hip_kernel_info_t** out_kernels) {
  const uint8_t* elf_base = elf_data.data;

  // Read ELF header
  if (elf_data.data_length < sizeof(iree_hal_hip_elf64_header_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF data too small for header");
  }

  const iree_hal_hip_elf64_header_t* elf_header =
      (const iree_hal_hip_elf64_header_t*)elf_base;

  // Verify basic ELF properties (already validated, but double-check)
  if (elf_header->magic[0] != IREE_HAL_HIP_ELF_MAGIC0 ||
      elf_header->magic[1] != IREE_HAL_HIP_ELF_MAGIC1 ||
      elf_header->magic[2] != IREE_HAL_HIP_ELF_MAGIC2 ||
      elf_header->magic[3] != IREE_HAL_HIP_ELF_MAGIC3) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid ELF magic");
  }

  // Find the symbol table and string table sections
  const iree_hal_hip_elf64_section_header_t* symtab_section = NULL;
  const iree_hal_hip_elf64_section_header_t* strtab_section = NULL;

  for (uint16_t i = 0; i < elf_header->shnum; ++i) {
    iree_host_size_t section_offset =
        elf_header->shoff + (i * sizeof(iree_hal_hip_elf64_section_header_t));
    if (section_offset + sizeof(iree_hal_hip_elf64_section_header_t) >
        elf_data.data_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "section header [%u] out of bounds", i);
    }

    const iree_hal_hip_elf64_section_header_t* section =
        (const iree_hal_hip_elf64_section_header_t*)(elf_base + section_offset);

    if (section->sh_type == IREE_HAL_HIP_SHT_DYNSYM) {
      symtab_section = section;
      // The sh_link field points to the associated string table
      iree_host_size_t strtab_section_offset =
          elf_header->shoff +
          (section->sh_link * sizeof(iree_hal_hip_elf64_section_header_t));
      if (strtab_section_offset + sizeof(iree_hal_hip_elf64_section_header_t) <=
          elf_data.data_length) {
        strtab_section =
            (const iree_hal_hip_elf64_section_header_t*)(elf_base +
                                                         strtab_section_offset);
      }
      break;
    }
  }

  if (!symtab_section || !strtab_section) {
    // No symbol table found - no kernels to extract
    *kernel_count = 0;
    *out_kernels = NULL;
    return iree_ok_status();
  }

  // Validate string table bounds
  if (strtab_section->sh_offset + strtab_section->sh_size >
      elf_data.data_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "string table out of bounds");
  }

  const uint8_t* string_table = elf_base + strtab_section->sh_offset;
  iree_host_size_t string_table_size = strtab_section->sh_size;

  // First pass: count function symbols
  iree_host_size_t num_symbols =
      symtab_section->sh_size / sizeof(iree_hal_hip_elf64_symbol_t);
  iree_host_size_t func_count = 0;

  for (iree_host_size_t i = 0; i < num_symbols; ++i) {
    iree_host_size_t symbol_offset =
        symtab_section->sh_offset + (i * sizeof(iree_hal_hip_elf64_symbol_t));
    if (symbol_offset + sizeof(iree_hal_hip_elf64_symbol_t) >
        elf_data.data_length) {
      continue;
    }

    const iree_hal_hip_elf64_symbol_t* symbol =
        (const iree_hal_hip_elf64_symbol_t*)(elf_base + symbol_offset);

    // Check if this is a function symbol
    if (IREE_HAL_HIP_ELF64_ST_TYPE(symbol->st_info) == IREE_HAL_HIP_STT_FUNC) {
      ++func_count;
    }
  }

  if (func_count == 0) {
    *kernel_count = 0;
    *out_kernels = NULL;
    return iree_ok_status();
  }

  // Allocate array for kernel information (no longer appending string storage)
  iree_host_size_t alloc_size = func_count * sizeof(iree_hal_hip_kernel_info_t);
  iree_hal_hip_kernel_info_t* kernels = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, alloc_size, (void**)&kernels));
  memset(kernels, 0, func_count * sizeof(iree_hal_hip_kernel_info_t));

  // Second pass: extract function names and kernel descriptors
  iree_host_size_t kernel_index = 0;
  for (iree_host_size_t i = 0; i < num_symbols && kernel_index < func_count;
       ++i) {
    iree_host_size_t symbol_offset =
        symtab_section->sh_offset + (i * sizeof(iree_hal_hip_elf64_symbol_t));
    if (symbol_offset + sizeof(iree_hal_hip_elf64_symbol_t) >
        elf_data.data_length) {
      continue;
    }

    const iree_hal_hip_elf64_symbol_t* symbol =
        (const iree_hal_hip_elf64_symbol_t*)(elf_base + symbol_offset);

    // Check if this is a function symbol
    if (IREE_HAL_HIP_ELF64_ST_TYPE(symbol->st_info) == IREE_HAL_HIP_STT_FUNC) {
      iree_string_view_t name;
      iree_status_t status = iree_hal_hip_read_string_from_table(
          string_table, string_table_size, symbol->st_name, &name);
      if (!iree_status_is_ok(status) || name.size == 0) {
        continue;
      }

      // Copy or reference the kernel name
      if (copy_kernel_names) {
        // Allocate separate memory for this kernel's name
        char* allocated_name = NULL;
        iree_status_t alloc_status =
            iree_allocator_malloc(allocator, name.size, (void**)&allocated_name);
        if (!iree_status_is_ok(alloc_status)) {
          // Clean up any previously allocated names
          for (iree_host_size_t k = 0; k < kernel_index; ++k) {
            if (kernels[k].allocated_name) {
              iree_allocator_free(allocator, kernels[k].allocated_name);
            }
          }
          iree_allocator_free(allocator, kernels);
          return alloc_status;
        }
        memcpy(allocated_name, name.data, name.size);
        kernels[kernel_index].name = iree_make_string_view(allocated_name, name.size);
        kernels[kernel_index].allocated_name = allocated_name;
      } else {
        // Reference the name directly in the ELF string table
        kernels[kernel_index].name = name;
        kernels[kernel_index].allocated_name = NULL;
      }

      // Look for the corresponding kernel descriptor (.kd suffix)
      // Search for a symbol with the same name + ".kd"
      for (iree_host_size_t j = 0; j < num_symbols; ++j) {
        iree_host_size_t kd_symbol_offset =
            symtab_section->sh_offset +
            (j * sizeof(iree_hal_hip_elf64_symbol_t));
        if (kd_symbol_offset + sizeof(iree_hal_hip_elf64_symbol_t) >
            elf_data.data_length) {
          continue;
        }

        const iree_hal_hip_elf64_symbol_t* kd_symbol =
            (const iree_hal_hip_elf64_symbol_t*)(elf_base + kd_symbol_offset);

        iree_string_view_t kd_name;
        status = iree_hal_hip_read_string_from_table(
            string_table, string_table_size, kd_symbol->st_name, &kd_name);
        if (!iree_status_is_ok(status)) {
          continue;
        }

        // Check if this symbol name ends with ".kd" and matches our function
        if (kd_name.size == name.size + 3 &&
            memcmp(kd_name.data, name.data, name.size) == 0 &&
            memcmp(kd_name.data + name.size, ".kd", 3) == 0) {
          // Found the kernel descriptor - parse it
          if (kd_symbol->st_value +
                  sizeof(iree_hal_hip_amd_kernel_descriptor_t) <=
              elf_data.data_length) {
            const iree_hal_hip_amd_kernel_descriptor_t* kd =
                (const iree_hal_hip_amd_kernel_descriptor_t*)(elf_base +
                                                              kd_symbol
                                                                  ->st_value);

            // Store kernel descriptor for metadata parsing
            // We'll parse metadata from .note section below
            (void)kd;  // Mark as used
          }
          break;
        }
      }

      ++kernel_index;
    }
  }
  
  // Find the .note section containing metadata
  const iree_hal_hip_elf64_section_header_t* note_section = NULL;
  const iree_hal_hip_elf64_section_header_t* shstrtab_section = NULL;

  // First find the section name string table
  if (elf_header->shstrndx < elf_header->shnum) {
    iree_host_size_t shstrtab_offset =
        elf_header->shoff +
        (elf_header->shstrndx * sizeof(iree_hal_hip_elf64_section_header_t));
    if (shstrtab_offset + sizeof(iree_hal_hip_elf64_section_header_t) <=
        elf_data.data_length) {
      shstrtab_section =
          (const iree_hal_hip_elf64_section_header_t*)(elf_base +
                                                       shstrtab_offset);
    }
  }

  // Find .note section by name
  if (shstrtab_section &&
      shstrtab_section->sh_offset + shstrtab_section->sh_size <=
          elf_data.data_length) {
    const uint8_t* shstrtab = elf_base + shstrtab_section->sh_offset;

    for (uint16_t i = 0; i < elf_header->shnum; ++i) {
      iree_host_size_t section_offset =
          elf_header->shoff + (i * sizeof(iree_hal_hip_elf64_section_header_t));
      if (section_offset + sizeof(iree_hal_hip_elf64_section_header_t) >
          elf_data.data_length) {
        continue;
      }

      const iree_hal_hip_elf64_section_header_t* section =
          (const iree_hal_hip_elf64_section_header_t*)(elf_base +
                                                       section_offset);

      // Check if this is SHT_NOTE type
      if (section->sh_type == IREE_HAL_HIP_SHT_NOTE) {
        // Check if the section name is ".note"
        if (section->sh_name < shstrtab_section->sh_size) {
          const char* section_name = (const char*)(shstrtab + section->sh_name);
          iree_host_size_t max_name_len =
              shstrtab_section->sh_size - section->sh_name;
          if (max_name_len > 5 && strncmp(section_name, ".note", 5) == 0) {
            note_section = section;
            break;
          }
        }
      }
    }
  }

  // Parse metadata from .note section for each kernel
  if (note_section &&
      note_section->sh_offset + note_section->sh_size <= elf_data.data_length) {
    const uint8_t* note_data = elf_base + note_section->sh_offset;
    iree_host_size_t note_size = note_section->sh_size;

    // For each kernel, parse comprehensive metadata from msgpack
    for (iree_host_size_t k = 0; k < kernel_index; ++k) {
      uint32_t wg_size_x, wg_size_y, wg_size_z;
      iree_host_size_t arg_count = 0;
      iree_hal_hip_parsed_arg_t* parsed_args = NULL;

      // Parse metadata for this specific kernel
      iree_status_t status = iree_hal_hip_parse_amd_kernel_metadata(
          note_data, note_size, kernels[k].name, allocator, &wg_size_x,
          &wg_size_y, &wg_size_z, &arg_count, &parsed_args);

      if (iree_status_is_ok(status) && arg_count > 0) {
        // Store workgroup size
        kernels[k].block_dims[0] = wg_size_x;
        kernels[k].block_dims[1] = wg_size_y;
        kernels[k].block_dims[2] = wg_size_z;

        // Count bindings vs constants
        uint32_t binding_count = 0;
        uint32_t constant_count = 0;
        for (iree_host_size_t a = 0; a < arg_count; ++a) {
          if (parsed_args[a].value_kind ==
              IREE_HAL_HIP_ARG_KIND_GLOBAL_BUFFER) {
            binding_count++;
          } else if (parsed_args[a].value_kind ==
                     IREE_HAL_HIP_ARG_KIND_BY_VALUE) {
            constant_count++;
          }
        }

        kernels[k].binding_count = binding_count;
        kernels[k].constant_count = constant_count;

        // Allocate and fill parameter array with actual metadata
        iree_hal_hip_kernel_param_t* params = NULL;
        iree_status_t alloc_status = iree_allocator_malloc(
            allocator, arg_count * sizeof(iree_hal_hip_kernel_param_t),
            (void**)&params);
        if (iree_status_is_ok(alloc_status)) {
          for (iree_host_size_t a = 0; a < arg_count; ++a) {
            params[a].offset = parsed_args[a].offset;
            params[a].size = parsed_args[a].size;
            params[a].type = (parsed_args[a].value_kind ==
                              IREE_HAL_HIP_ARG_KIND_GLOBAL_BUFFER)
                                 ? 1
                                 : 0;  // 1=pointer, 0=value
          }
          kernels[k].parameters = params;
        }

        // Free temporary parsed args
        iree_allocator_free(allocator, parsed_args);
      } else {
        // Fallback: no metadata found, use defaults
        kernels[k].block_dims[0] = 1;
        kernels[k].block_dims[1] = 1;
        kernels[k].block_dims[2] = 1;
        kernels[k].binding_count = 0;
        kernels[k].constant_count = 0;
        kernels[k].parameters = NULL;
      }
    }
  }

  *kernel_count = kernel_index;
  *out_kernels = kernels;

  return iree_ok_status();
}

void iree_hal_hip_free_kernel_info(iree_allocator_t allocator,
                                    iree_host_size_t kernel_count,
                                    iree_hal_hip_kernel_info_t* kernels) {
  if (!kernels) {
    return;
  }

  // Free any individually allocated kernel names
  for (iree_host_size_t i = 0; i < kernel_count; ++i) {
    if (kernels[i].allocated_name) {
      iree_allocator_free(allocator, kernels[i].allocated_name);
    }
  }

  // Free the kernel array itself
  iree_allocator_free(allocator, kernels);
}

iree_status_t iree_hal_hip_parse_fat_binary_kernels(
    iree_const_byte_span_t executable_data, iree_string_view_t target_triple,
    iree_allocator_t allocator, iree_hal_hip_fat_binary_info_t* out_info) {
  if (!out_info) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_info must not be NULL");
  }

  memset(out_info, 0, sizeof(*out_info));

  // Parse the fat binary header
  if (executable_data.data_length < sizeof(iree_hal_hip_fat_binary_header_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable data too small for fat binary header");
  }

  // Check if this is a raw ELF file (no fat binary wrapper)
  if (iree_hal_hip_is_code_object_elf(executable_data, false)) {
    // It's a raw ELF binary - parse kernels directly
    // No need to copy names - executable_data persists
    iree_hal_hip_kernel_info_t* matched_kernels = NULL;
    iree_host_size_t matched_kernel_count = 0;
    iree_status_t status = iree_hal_hip_parse_elf_kernels(
        executable_data, allocator, /*copy_kernel_names=*/false,
        &matched_kernel_count, &matched_kernels);
    if (!iree_status_is_ok(status)) {
      return status;
    }
    // Fill output structure
    out_info->bundle_data = executable_data.data;
    out_info->bundle_size = executable_data.data_length;
    out_info->kernel_count = matched_kernel_count;
    out_info->kernels = matched_kernels;
    return iree_ok_status();
  }

  // Check if this is a CCOB file (compressed bundle, no fat binary wrapper)
  if (iree_hal_hip_is_code_object_compressed(executable_data, false)) {
    // Parse CCOB header
    uint16_t version, method;
    uint32_t uncompressed_size, file_size;
    iree_host_size_t compressed_data_offset;

    IREE_RETURN_IF_ERROR(iree_hal_hip_parse_ccob_header(
        executable_data, false, &version, &method, &uncompressed_size,
        &file_size, &compressed_data_offset));

    // Get the compressed data (after the header)
    iree_const_byte_span_t compressed_data = iree_make_const_byte_span(
        executable_data.data + compressed_data_offset,
        executable_data.data_length - compressed_data_offset);

    // Decompress the CCOB data
    uint8_t* decompressed_data = NULL;
    iree_host_size_t decompressed_size = 0;
    iree_status_t decompress_status = iree_hal_hip_decompress_ccob(
        compressed_data, method, uncompressed_size, allocator,
        &decompressed_data, &decompressed_size);

    if (!iree_status_is_ok(decompress_status)) {
      return decompress_status;
    }

    // Parse the decompressed data as an offload bundle or ELF
    iree_const_byte_span_t decompressed_span =
        iree_make_const_byte_span(decompressed_data, decompressed_size);

    // Check if decompressed data is an ELF or offload bundle
    iree_hal_hip_kernel_info_t* matched_kernels = NULL;
    iree_host_size_t matched_kernel_count = 0;
    iree_host_size_t bundle_size = 0;
    iree_status_t status;

    if (iree_hal_hip_is_code_object_elf(decompressed_span, false)) {
      // Parse as ELF directly
      // MUST copy names - decompressed_data will be freed after this
      status = iree_hal_hip_parse_elf_kernels(
          decompressed_span, allocator, /*copy_kernel_names=*/true,
          &matched_kernel_count, &matched_kernels);
      bundle_size = decompressed_size;
    } else if (iree_hal_hip_is_code_object_uncompressed(decompressed_span,
                                                         false)) {
      // Parse as offload bundle - extract the matching ELF
      // MUST copy names - decompressed_data will be freed after this
      status = iree_hal_hip_parse_offload_bundle_kernels(
          decompressed_span, target_triple, allocator,
          /*copy_kernel_names=*/true, &matched_kernel_count, &matched_kernels,
          &bundle_size);
    } else {
      iree_allocator_free(allocator, decompressed_data);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "CCOB decompressed data is not a recognized format (not ELF or "
          "offload bundle)");
    }

    // Free the decompressed data (it's been parsed into kernels)
    iree_allocator_free(allocator, decompressed_data);

    if (!iree_status_is_ok(status)) {
      return status;
    }

    // Fill output structure
    out_info->bundle_data = executable_data.data;
    out_info->bundle_size = executable_data.data_length;
    out_info->kernel_count = matched_kernel_count;
    out_info->kernels = matched_kernels;
    return iree_ok_status();
  }

  // Read the fat binary header
  iree_hal_hip_fat_binary_header_t fat_header;
  memcpy(&fat_header, executable_data.data, sizeof(fat_header));

  if (fat_header.magic != IREE_HAL_HIP_FAT_BINARY_MAGIC) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid fat binary magic");
  }

  if (fat_header.version != IREE_HAL_HIP_FAT_BINARY_VERSION) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "unsupported fat binary version");
  }

  // Get the binary data span
  iree_const_byte_span_t binary_data = iree_make_const_byte_span(
      fat_header.binary,
      executable_data.data_length -
          ((uint8_t*)fat_header.binary - (uint8_t*)executable_data.data));
  
  bool is_compressed =
      iree_hal_hip_is_code_object_compressed(binary_data, false);
  // Check if this is an uncompressed offload bundle
  bool is_uncompressed =
      iree_hal_hip_is_code_object_uncompressed(binary_data, false);

  if (is_compressed) {
    // Parse CCOB header to provide detailed error information
    uint16_t version, method;
    uint32_t uncompressed_size, file_size;
    iree_host_size_t compressed_data_offset;

    iree_status_t status = iree_hal_hip_parse_ccob_header(
        binary_data, false, &version, &method,
        &uncompressed_size, &file_size, &compressed_data_offset);

    if (iree_status_is_ok(status)) {
      const char* method_name = (method == IREE_HAL_HIP_CCOB_METHOD_ZLIB) ? "zlib" : "zstd";
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "compressed offload bundle (CCOB v%u) detected for kernel parsing. "
          "Uses %s compression with uncompressed size of %" PRIu32 " bytes. "
          "CCOB kernel parsing requires decompression support which is not yet implemented.",
          version, method_name, uncompressed_size);
    } else {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "compressed offload bundle detected but failed to parse CCOB header");
    }
  }

  if (!is_uncompressed) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "only uncompressed offload bundles are supported for kernel parsing");
  }

  // Parse the offload bundle and extract kernels for the target triple
  // No need to copy names - binary_data is from fat binary and persists
  iree_hal_hip_kernel_info_t* matched_kernels = NULL;
  iree_host_size_t matched_kernel_count = 0;
  iree_host_size_t bundle_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_hip_parse_offload_bundle_kernels(
      binary_data, target_triple, allocator, /*copy_kernel_names=*/false,
      &matched_kernel_count, &matched_kernels, &bundle_size));

  // Fill output structure
  out_info->bundle_data = binary_data.data;
  out_info->bundle_size = bundle_size;
  out_info->kernel_count = matched_kernel_count;
  out_info->kernels = matched_kernels;

  return iree_ok_status();
}

iree_status_t iree_hal_hip_read_native_header(
    iree_const_byte_span_t executable_data, bool unsafe_infer_size,
    iree_const_byte_span_t* out_elf_data) {
  if (!out_elf_data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_elf_data must not be NULL");
  }

  // Check for minimum size for fat binary wrapper.
  if (!unsafe_infer_size &&
      executable_data.data_length < sizeof(iree_hal_hip_fat_binary_header_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "executable data too small (expected at least %zu bytes for fat binary "
        "header, got %" PRIhsz " bytes)",
        sizeof(iree_hal_hip_fat_binary_header_t), executable_data.data_length);
  }

  // Check for fat binary wrapper magic.
  uint32_t magic;
  memcpy(&magic, executable_data.data, sizeof(magic));

  if (magic == IREE_HAL_HIP_ELF_MAGIC) {
    // It's a raw ELF binary. Validate the header and get the size.
    iree_host_size_t elf_size = 0;
    IREE_RETURN_IF_ERROR(iree_hal_hip_validate_elf_header(
        executable_data, unsafe_infer_size, &elf_size));
    *out_elf_data = iree_make_const_byte_span(executable_data.data, elf_size);
    return iree_ok_status();
  }

  if (magic == IREE_HAL_HIP_OFFLOAD_BUNDLE_COMPRESSED_MAGIC_INT) {
    // It's a raw ELF binary. Validate the header and get the size.
    iree_host_size_t elf_size = 0;
    IREE_RETURN_IF_ERROR(iree_hal_hip_validate_ccob_header(
        executable_data, unsafe_infer_size, &elf_size));
    *out_elf_data = iree_make_const_byte_span(executable_data.data, elf_size);
    return iree_ok_status();
  }

  if (magic != IREE_HAL_HIP_FAT_BINARY_MAGIC) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "executable data does not have fat binary magic (expected 0x%08x, got "
        "0x%08x)",
        IREE_HAL_HIP_FAT_BINARY_MAGIC, magic);
  }

  // Read the fat binary header.
  iree_hal_hip_fat_binary_header_t fat_header;
  memcpy(&fat_header, executable_data.data, sizeof(fat_header));

  // Validate version.
  if (fat_header.version != IREE_HAL_HIP_FAT_BINARY_VERSION) {
    return iree_make_status(
        IREE_STATUS_INCOMPATIBLE,
        "fat binary version %u not compatible (expected version %u)",
        fat_header.version, IREE_HAL_HIP_FAT_BINARY_VERSION);
  }

  // Validate offset is within bounds.
  if (!unsafe_infer_size &&
      ((uint8_t*)fat_header.binary >=
       executable_data.data + executable_data.data_length)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fat binary  %p"
        " is beyond data end %p",
        (void*)fat_header.binary,
        (void*)(executable_data.data + executable_data.data_length));
  }

  // Get the binary data span (skipping the fat binary header).
  iree_const_byte_span_t binary_data = iree_make_const_byte_span(
      fat_header.binary,
      unsafe_infer_size
          ? 0
          : (executable_data.data_length -
             ((uint8_t*)fat_header.binary - (uint8_t*)executable_data.data)));

  // Check for minimum size to determine format.
  if (!unsafe_infer_size && binary_data.data_length < 4) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "binary data too small (expected at least 4 bytes for magic "
        "detection, got %" PRIhsz " bytes)",
        binary_data.data_length);
  }

  // Check format in the same order as HIP runtime:
  // 1. Compressed bundle
  // 2. Uncompressed bundle
  // 3. Raw ELF

  bool is_compressed =
      iree_hal_hip_is_code_object_compressed(binary_data, unsafe_infer_size);
  bool is_uncompressed =
      iree_hal_hip_is_code_object_uncompressed(binary_data, unsafe_infer_size);

  if (is_compressed) {
    // Compressed offload bundle format (CCOB).
    // Parse CCOB header to get compression details
    uint16_t version, method;
    uint32_t uncompressed_size, file_size;
    iree_host_size_t compressed_data_offset;

    IREE_RETURN_IF_ERROR(iree_hal_hip_parse_ccob_header(
        binary_data, unsafe_infer_size, &version, &method,
        &uncompressed_size, &file_size, &compressed_data_offset));

    // For now, we don't support actual decompression, so return an error
    // with details about the compressed bundle
    const char* method_name = (method == IREE_HAL_HIP_CCOB_METHOD_ZLIB) ? "zlib" : "zstd";
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "compressed offload bundle format (CCOB v%u) detected using %s compression. "
        "Uncompressed size: %" PRIu32 " bytes. "
        "CCOB decompression is not yet fully implemented. "
        "Please use an uncompressed bundle format instead.",
        version, method_name, uncompressed_size);
  }


  if (is_uncompressed) {
    // Uncompressed offload bundle format (__CLANG_OFFLOAD_BUNDLE__).
    iree_host_size_t bundle_size = 0;
    IREE_RETURN_IF_ERROR(iree_hal_hip_parse_offload_bundle(
        binary_data, unsafe_infer_size, &bundle_size));

    iree_host_size_t total_size =
        offsetof(iree_hal_hip_fat_binary_header_t, binary) + bundle_size;
    // Return the bundle data with the calculated size.
    *out_elf_data = iree_make_const_byte_span(executable_data.data, total_size);
    return iree_ok_status();
  }

  // It better be ELF if it's neither compressed nor uncompressed.
  if (!iree_hal_hip_is_code_object_elf(binary_data, unsafe_infer_size)) {
    // Not a recognized format.
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "binary data is not a recognized format (not compressed bundle, "
        "uncompressed bundle, or ELF); first 4 bytes: 0x%02x 0x%02x 0x%02x "
        "0x%02x",
        ((const uint8_t*)binary_data.data)[0],
        ((const uint8_t*)binary_data.data)[1],
        ((const uint8_t*)binary_data.data)[2],
        ((const uint8_t*)binary_data.data)[3]);
  }

  // It's a raw ELF binary. Validate the header and get the size.
  iree_host_size_t elf_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_hip_validate_elf_header(
      binary_data, unsafe_infer_size, &elf_size));

  // Return the ELF data with the inferred size.
  // When unsafe_infer_size is true, use the calculated size from the ELF
  // header. When false, use the provided binary_data length (which should
  // already be validated). The total inferred size is the difference between
  // the start of the original data and the end of the ELF section.
  if (unsafe_infer_size) {
    // Total size = fat binary header offset + ELF size
    iree_host_size_t total_size =
        offsetof(iree_hal_hip_fat_binary_header_t, binary) + elf_size;
    *out_elf_data = iree_make_const_byte_span(executable_data.data, total_size);
  } else {
    *out_elf_data = binary_data;
  }

  return iree_ok_status();
}
