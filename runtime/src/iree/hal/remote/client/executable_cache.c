// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/executable_cache.h"

#include "iree/hal/remote/client/device.h"
#include "iree/hal/remote/client/executable.h"
#include "iree/hal/remote/protocol/common.h"
#include "iree/hal/remote/protocol/control.h"

static const iree_hal_executable_cache_vtable_t
    iree_hal_remote_client_executable_cache_vtable;

typedef struct iree_hal_remote_client_executable_cache_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_remote_client_device_t* device;
} iree_hal_remote_client_executable_cache_t;

static void iree_hal_remote_client_executable_cache_destroy(
    iree_hal_executable_cache_t* base_cache) {
  iree_hal_remote_client_executable_cache_t* cache =
      (iree_hal_remote_client_executable_cache_t*)base_cache;
  iree_allocator_free(cache->host_allocator, cache);
}

//===----------------------------------------------------------------------===//
// Local HIP binary format inference
//===----------------------------------------------------------------------===//
// The remote client must be able to infer the format and total size of HIP
// executable binaries locally (without an RPC to the server) so that the
// streaming layer can determine how much data to upload. These binaries arrive
// via __hipRegisterFatBinary and may be wrapped in several layers:
//   1. Fat binary wrapper (magic 0x48495046) -> pointer to inner binary
//   2. Raw ELF (magic 0x7fELF)
//   3. CCOB compressed bundle (magic "CCOB")
//   4. Uncompressed offload bundle (magic "__CLANG_OFFLOAD_BUNDLE__")

// Fat binary wrapper.
#define IREE_REMOTE_FAT_BINARY_MAGIC 0x48495046u
#define IREE_REMOTE_FAT_BINARY_VERSION 1
typedef struct iree_remote_fat_binary_header_t {
  uint32_t magic;
  uint32_t version;
  void* binary;
  void* reserved;
} iree_remote_fat_binary_header_t;

// ELF64 header (subset for size inference).
#define IREE_REMOTE_ELF_MAGIC 0x464c457fu

// CCOB header versions.
#define IREE_REMOTE_CCOB_MAGIC 0x424f4343u
typedef struct iree_remote_ccob_v2_t {
  uint8_t magic[4];
  uint16_t version;
  uint16_t method;
  uint32_t file_size;
  uint32_t uncompressed_size;
  uint64_t hash;
} iree_remote_ccob_v2_t;
typedef struct iree_remote_ccob_v3_t {
  uint8_t magic[4];
  uint16_t version;
  uint16_t method;
  uint64_t file_size;
  uint64_t uncompressed_size;
  uint64_t hash;
} iree_remote_ccob_v3_t;

// Offload bundle entry.
#define IREE_REMOTE_OFFLOAD_BUNDLE_MAGIC "__CLANG_OFFLOAD_BUNDLE__"
#define IREE_REMOTE_OFFLOAD_BUNDLE_MAGIC_SIZE 24

// Attempts to infer the format and total byte size of a HIP binary starting at
// |data|. When |data_length| is 0, size is inferred purely from header fields
// (unsafe mode, used when the caller doesn't know the length).
//
// On success, writes the format string into |out_format| (capacity
// |format_capacity|) and the total binary size into |*out_size|.
static iree_status_t iree_remote_infer_hip_binary(
    const uint8_t* data, iree_host_size_t data_length,
    iree_host_size_t format_capacity, char* out_format,
    iree_host_size_t* out_size) {
  if (!data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "NULL data pointer");
  }

  const bool unsafe = (data_length == 0);

  // We need at least 4 bytes to read a magic number.
  if (!unsafe && data_length < 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "data too small for magic detection");
  }

  uint32_t magic;
  memcpy(&magic, data, sizeof(magic));

  // ---- Fat binary wrapper ----
  if (magic == IREE_REMOTE_FAT_BINARY_MAGIC) {
    if (!unsafe &&
        data_length < sizeof(iree_remote_fat_binary_header_t)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "data too small for fat binary header");
    }
    iree_remote_fat_binary_header_t fat;
    memcpy(&fat, data, sizeof(fat));
    if (fat.version != IREE_REMOTE_FAT_BINARY_VERSION) {
      return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                              "unsupported fat binary version %u", fat.version);
    }
    // The `binary` field is a pointer into the process address space.
    // Recurse to detect the inner format and compute its size.
    const uint8_t* inner = (const uint8_t*)fat.binary;
    iree_host_size_t inner_size = 0;
    IREE_RETURN_IF_ERROR(
        iree_remote_infer_hip_binary(inner, /*data_length=*/0,
                                     format_capacity, out_format, &inner_size));
    *out_size = inner_size;
    return iree_ok_status();
  }

  // ---- Raw ELF ----
  if (magic == IREE_REMOTE_ELF_MAGIC) {
    // ELF64: total size = e_shoff + e_shnum * e_shentsize.
    // e_shoff   is at offset 0x28 (8 bytes, uint64_t)
    // e_shentsize is at offset 0x3a (2 bytes, uint16_t)
    // e_shnum     is at offset 0x3c (2 bytes, uint16_t)
    if (!unsafe && data_length < 64) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "ELF data too small for header");
    }
    uint64_t e_shoff;
    uint16_t e_shentsize, e_shnum;
    memcpy(&e_shoff, data + 0x28, sizeof(e_shoff));
    memcpy(&e_shentsize, data + 0x3a, sizeof(e_shentsize));
    memcpy(&e_shnum, data + 0x3c, sizeof(e_shnum));
    *out_size = (iree_host_size_t)(e_shoff +
                                   (iree_host_size_t)e_shentsize * e_shnum);
    snprintf(out_format, format_capacity, "FPIH");
    return iree_ok_status();
  }

  // ---- CCOB (compressed clang offload bundle) ----
  if (magic == IREE_REMOTE_CCOB_MAGIC) {
    uint16_t version;
    memcpy(&version, data + 4, sizeof(version));
    if (version == 1) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "CCOB v1 does not store total file size");
    } else if (version == 2) {
      if (!unsafe && data_length < sizeof(iree_remote_ccob_v2_t)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "data too small for CCOB v2 header");
      }
      uint32_t file_size;
      memcpy(&file_size, data + 8, sizeof(file_size));
      *out_size = (iree_host_size_t)file_size;
      snprintf(out_format, format_capacity, "FPIH");
      return iree_ok_status();
    } else if (version >= 3) {
      if (!unsafe && data_length < sizeof(iree_remote_ccob_v3_t)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "data too small for CCOB v3 header");
      }
      uint64_t file_size;
      memcpy(&file_size, data + 8, sizeof(file_size));
      *out_size = (iree_host_size_t)file_size;
      snprintf(out_format, format_capacity, "FPIH");
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "unsupported CCOB version %u", version);
  }

  // ---- Uncompressed clang offload bundle ----
  if ((!unsafe && data_length >= IREE_REMOTE_OFFLOAD_BUNDLE_MAGIC_SIZE) ||
      unsafe) {
    if (memcmp(data, IREE_REMOTE_OFFLOAD_BUNDLE_MAGIC,
               IREE_REMOTE_OFFLOAD_BUNDLE_MAGIC_SIZE) == 0) {
      // Header: [magic 24B][num_entries 8B]
      // Per entry: [offset 8B][size 8B][id_size 8B][id id_size B]
      uint64_t num_entries;
      memcpy(&num_entries, data + IREE_REMOTE_OFFLOAD_BUNDLE_MAGIC_SIZE,
             sizeof(num_entries));
      const uint8_t* cursor =
          data + IREE_REMOTE_OFFLOAD_BUNDLE_MAGIC_SIZE + sizeof(num_entries);
      iree_host_size_t max_end =
          IREE_REMOTE_OFFLOAD_BUNDLE_MAGIC_SIZE + sizeof(num_entries);
      for (uint64_t i = 0; i < num_entries; ++i) {
        uint64_t entry_offset, entry_size, entry_id_size;
        memcpy(&entry_offset, cursor, sizeof(entry_offset));
        memcpy(&entry_size, cursor + 8, sizeof(entry_size));
        memcpy(&entry_id_size, cursor + 16, sizeof(entry_id_size));
        iree_host_size_t end =
            (iree_host_size_t)(entry_offset + entry_size);
        if (end > max_end) max_end = end;
        cursor += 24 + entry_id_size;
      }
      *out_size = max_end;
      snprintf(out_format, format_capacity, "FPIH");
      return iree_ok_status();
    }
  }

  return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                          "unrecognized HIP binary format (magic 0x%08x)",
                          magic);
}

static iree_status_t iree_hal_remote_client_executable_cache_infer_format(
    iree_hal_executable_cache_t* base_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  IREE_ASSERT_ARGUMENT(executable_format);
  IREE_ASSERT_ARGUMENT(out_inferred_size);
  *out_inferred_size = 0;
  if (executable_format_capacity > 0) {
    executable_format[0] = '\0';
  }

  return iree_remote_infer_hip_binary(
      executable_data.data, executable_data.data_length,
      executable_format_capacity, executable_format, out_inferred_size);
}

static bool iree_hal_remote_client_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* base_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  // The remote client can attempt to prepare any format — the server will
  // reject incompatible formats during EXECUTABLE_UPLOAD.
  return true;
}

static iree_status_t iree_hal_remote_client_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* base_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  iree_hal_remote_client_executable_cache_t* cache =
      (iree_hal_remote_client_executable_cache_t*)base_cache;
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Build the EXECUTABLE_UPLOAD request with overflow-checked sizes.
  iree_host_size_t format_length = executable_params->executable_format.size;
  iree_host_size_t format_padded = iree_host_align(format_length, 8);
  iree_host_size_t constants_size = 0;
  if (!iree_host_size_checked_mul(executable_params->constant_count,
                                  sizeof(uint32_t), &constants_size)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable upload constants size overflow");
  }
  iree_host_size_t constants_padded = iree_host_align(constants_size, 8);
  iree_host_size_t data_length = executable_params->executable_data.data_length;

  iree_host_size_t header_size =
      sizeof(iree_hal_remote_control_envelope_t) +
      sizeof(iree_hal_remote_executable_upload_request_t);
  iree_host_size_t message_length = 0;
  if (!iree_host_size_checked_add(header_size, format_padded,
                                  &message_length) ||
      !iree_host_size_checked_add(message_length, constants_padded,
                                  &message_length) ||
      !iree_host_size_checked_add(message_length, data_length,
                                  &message_length)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable upload message size overflow");
  }

  // Heap-allocate for the variable-length message (executable binaries can
  // be large).
  uint8_t* message_buffer = NULL;
  iree_status_t status = iree_allocator_malloc(
      cache->host_allocator, message_length, (void**)&message_buffer);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  memset(message_buffer, 0, message_length);

  // Envelope.
  iree_hal_remote_control_envelope_t* envelope =
      (iree_hal_remote_control_envelope_t*)message_buffer;
  envelope->message_type = IREE_HAL_REMOTE_CONTROL_EXECUTABLE_UPLOAD;

  // Request header.
  iree_hal_remote_executable_upload_request_t* request =
      (iree_hal_remote_executable_upload_request_t*)(envelope + 1);
  request->provisional_id = IREE_HAL_REMOTE_RESOURCE_ID_PROVISIONAL(
      IREE_HAL_REMOTE_RESOURCE_TYPE_EXECUTABLE, 0);
  request->format_length = (uint16_t)format_length;
  request->constant_count = (uint16_t)executable_params->constant_count;
  request->upload_flags = IREE_HAL_REMOTE_UPLOAD_FLAG_INLINE_DATA;
  request->data_length = data_length;

  // Variable-length payload: format string (padded to 8-byte alignment).
  uint8_t* cursor = (uint8_t*)(request + 1);
  if (format_length > 0) {
    memcpy(cursor, executable_params->executable_format.data, format_length);
  }
  cursor += format_padded;

  // Constants (padded to 8-byte alignment).
  if (constants_size > 0) {
    memcpy(cursor, executable_params->constants, constants_size);
  }
  cursor += constants_padded;

  // Inline executable data.
  if (data_length > 0) {
    memcpy(cursor, executable_params->executable_data.data, data_length);
  }

  // Send RPC.
  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  status = iree_hal_remote_client_device_control_rpc(
      cache->device, iree_make_const_byte_span(message_buffer, message_length),
      &response_payload, &response_lease);
  if (!iree_status_is_ok(status)) {
    // Error will propagate via iree_status_t return.
  }

  iree_allocator_free(cache->host_allocator, message_buffer);

  // Parse response.
  if (iree_status_is_ok(status)) {
    if (response_payload.data_length <
        sizeof(iree_hal_remote_executable_upload_response_t)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "EXECUTABLE_UPLOAD response too short: "
          "%" PRIhsz " < %" PRIhsz,
          response_payload.data_length,
          sizeof(iree_hal_remote_executable_upload_response_t));
    }
  }

  if (iree_status_is_ok(status)) {
    const iree_hal_remote_executable_upload_response_t* response =
        (const iree_hal_remote_executable_upload_response_t*)
            response_payload.data;
    status = iree_hal_remote_client_executable_create(
        cache->device, response->resolved_id,
        (iree_host_size_t)response->export_count, cache->host_allocator,
        out_executable);
    // Error will propagate via iree_status_t return.
  }

  iree_async_buffer_lease_release(&response_lease);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_executable_cache_create(
    iree_hal_remote_client_device_t* device, iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  *out_executable_cache = NULL;

  iree_hal_remote_client_executable_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*cache), (void**)&cache));
  iree_hal_resource_initialize(&iree_hal_remote_client_executable_cache_vtable,
                               &cache->resource);
  cache->host_allocator = host_allocator;
  cache->device = device;

  *out_executable_cache = (iree_hal_executable_cache_t*)cache;
  return iree_ok_status();
}

static const iree_hal_executable_cache_vtable_t
    iree_hal_remote_client_executable_cache_vtable = {
        .destroy = iree_hal_remote_client_executable_cache_destroy,
        .infer_format = iree_hal_remote_client_executable_cache_infer_format,
        .can_prepare_format =
            iree_hal_remote_client_executable_cache_can_prepare_format,
        .prepare_executable =
            iree_hal_remote_client_executable_cache_prepare_executable,
};
