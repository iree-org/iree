// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_LIBAQLPROFILE_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_LIBAQLPROFILE_H_

#include "iree/hal/drivers/amdgpu/abi/queue.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_dynamic_library_t iree_dynamic_library_t;

//===----------------------------------------------------------------------===//
// aqlprofile SDK ABI subset
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_aqlprofile_handle_t {
  // Opaque handle owned by the aqlprofile runtime.
  uint64_t handle;
} iree_hal_amdgpu_aqlprofile_handle_t;

typedef struct iree_hal_amdgpu_aqlprofile_version_t {
  // Major aqlprofile runtime version.
  uint32_t major;
  // Minor aqlprofile runtime version.
  uint32_t minor;
  // Patch aqlprofile runtime version.
  uint32_t patch;
} iree_hal_amdgpu_aqlprofile_version_t;

typedef enum iree_hal_amdgpu_aqlprofile_memory_hint_e {
  IREE_HAL_AMDGPU_AQLPROFILE_MEMORY_HINT_NONE = 0,
  IREE_HAL_AMDGPU_AQLPROFILE_MEMORY_HINT_HOST = 1,
  IREE_HAL_AMDGPU_AQLPROFILE_MEMORY_HINT_DEVICE_UNCACHED = 2,
  IREE_HAL_AMDGPU_AQLPROFILE_MEMORY_HINT_DEVICE_COHERENT = 3,
  IREE_HAL_AMDGPU_AQLPROFILE_MEMORY_HINT_DEVICE_NONCOHERENT = 4,
} iree_hal_amdgpu_aqlprofile_memory_hint_t;

typedef enum iree_hal_amdgpu_aqlprofile_agent_version_e {
  IREE_HAL_AMDGPU_AQLPROFILE_AGENT_VERSION_NONE = 0,
  IREE_HAL_AMDGPU_AQLPROFILE_AGENT_VERSION_V0 = 1,
  IREE_HAL_AMDGPU_AQLPROFILE_AGENT_VERSION_V1 = 2,
} iree_hal_amdgpu_aqlprofile_agent_version_t;

typedef uint32_t iree_hal_amdgpu_aqlprofile_block_name_t;
enum iree_hal_amdgpu_aqlprofile_block_name_e {
  IREE_HAL_AMDGPU_AQLPROFILE_BLOCK_NAME_SQ = 6,
};

typedef union iree_hal_amdgpu_aqlprofile_buffer_desc_flags_t {
  // Raw aqlprofile buffer descriptor flags.
  uint32_t raw;
  // Decoded aqlprofile buffer descriptor fields.
  struct {
    // True when the requested allocation must be visible to the profiled GPU.
    uint32_t device_access : 1;
    // True when the requested allocation must be visible to the host.
    uint32_t host_access : 1;
    // Requested memory placement hint.
    uint32_t memory_hint : 6;
    // Reserved bits reported by aqlprofile.
    uint32_t reserved : 24;
  };
} iree_hal_amdgpu_aqlprofile_buffer_desc_flags_t;

typedef hsa_status_t(
    IREE_API_PTR* iree_hal_amdgpu_aqlprofile_memory_alloc_callback_t)(
    void** ptr, uint64_t size,
    iree_hal_amdgpu_aqlprofile_buffer_desc_flags_t flags, void* user_data);

typedef void(
    IREE_API_PTR* iree_hal_amdgpu_aqlprofile_memory_dealloc_callback_t)(
    void* ptr, void* user_data);

typedef hsa_status_t(
    IREE_API_PTR* iree_hal_amdgpu_aqlprofile_memory_copy_callback_t)(
    void* target, const void* source, size_t size, void* user_data);

typedef union iree_hal_amdgpu_aqlprofile_pmc_event_flags_t {
  // Raw aqlprofile PMC event flags.
  uint32_t raw;
  // Decoded SQ event flag fields.
  struct {
    // SQ accumulation mode requested by the event.
    uint32_t accum : 3;
    // Reserved bits reported by aqlprofile.
    uint32_t reserved : 25;
    // SPM decode depth requested by the event.
    uint32_t depth : 4;
  } sq_flags;
  // Decoded SPM event flag fields.
  struct {
    // Reserved bits reported by aqlprofile.
    uint32_t reserved : 28;
    // SPM decode depth requested by the event.
    uint32_t depth : 4;
  } spm_flags;
} iree_hal_amdgpu_aqlprofile_pmc_event_flags_t;

typedef struct iree_hal_amdgpu_aqlprofile_pmc_event_t {
  // Hardware block instance index.
  uint32_t block_index;
  // Hardware event selector id within |block_name|.
  uint32_t event_id;
  // Event-specific flags such as accumulation mode.
  iree_hal_amdgpu_aqlprofile_pmc_event_flags_t flags;
  // Hardware block family containing |event_id|.
  iree_hal_amdgpu_aqlprofile_block_name_t block_name;
} iree_hal_amdgpu_aqlprofile_pmc_event_t;

typedef struct iree_hal_amdgpu_aqlprofile_agent_info_v1_t {
  // NUL-terminated GPU ISA name such as "gfx1100".
  const char* agent_gfxip;
  // Number of XCC partitions reported for the GPU.
  uint32_t xcc_num;
  // Number of shader engines reported for the GPU.
  uint32_t se_num;
  // Number of compute units reported for the GPU.
  uint32_t cu_num;
  // Number of shader arrays per shader engine.
  uint32_t shader_arrays_per_se;
  // PCI domain reported for the GPU.
  uint32_t domain;
  // PCI BDF location id reported for the GPU.
  uint32_t location_id;
} iree_hal_amdgpu_aqlprofile_agent_info_v1_t;

typedef struct iree_hal_amdgpu_aqlprofile_agent_handle_t {
  // Opaque registered-agent handle owned by the aqlprofile runtime.
  uint64_t handle;
} iree_hal_amdgpu_aqlprofile_agent_handle_t;

typedef struct iree_hal_amdgpu_aqlprofile_pmc_profile_t {
  // Registered aqlprofile GPU agent handle.
  iree_hal_amdgpu_aqlprofile_agent_handle_t agent;
  // Borrowed array of hardware PMC event requests.
  const iree_hal_amdgpu_aqlprofile_pmc_event_t* events;
  // Number of entries in |events|.
  uint32_t event_count;
} iree_hal_amdgpu_aqlprofile_pmc_profile_t;

typedef hsa_status_t(
    IREE_API_PTR* iree_hal_amdgpu_aqlprofile_pmc_data_callback_t)(
    iree_hal_amdgpu_aqlprofile_pmc_event_t event, uint64_t counter_id,
    uint64_t counter_value, void* user_data);

typedef struct iree_hal_amdgpu_aqlprofile_pmc_aql_packets_t {
  // AQL PM4-IB packet that resets and starts the selected counters.
  iree_hsa_amd_aql_pm4_ib_packet_t start_packet;
  // AQL PM4-IB packet that stops the selected counters.
  iree_hsa_amd_aql_pm4_ib_packet_t stop_packet;
  // AQL PM4-IB packet that reads the selected counters to output storage.
  iree_hsa_amd_aql_pm4_ib_packet_t read_packet;
} iree_hal_amdgpu_aqlprofile_pmc_aql_packets_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_libaqlprofile_t
//===----------------------------------------------------------------------===//

// Dynamically loaded libhsa-amd-aqlprofile64.so.
//
// Thread-safe; immutable after initialization.
typedef struct iree_hal_amdgpu_libaqlprofile_t {
  // Loaded aqlprofile dynamic library.
  iree_dynamic_library_t* library;

  // Returns the aqlprofile runtime version.
  hsa_status_t(HSA_API* aqlprofile_get_version)(
      iree_hal_amdgpu_aqlprofile_version_t* version);

  // Registers one HSA agent with the aqlprofile runtime.
  hsa_status_t(HSA_API* aqlprofile_register_agent_info)(
      iree_hal_amdgpu_aqlprofile_agent_handle_t* agent_id,
      const void* agent_info,
      iree_hal_amdgpu_aqlprofile_agent_version_t version);

  // Validates that a PMC event can be collected on an agent.
  hsa_status_t(HSA_API* aqlprofile_validate_pmc_event)(
      iree_hal_amdgpu_aqlprofile_agent_handle_t agent,
      const iree_hal_amdgpu_aqlprofile_pmc_event_t* event, bool* result);

  // Creates persistent PM4 programs and AQL PM4-IB packet templates for one
  // PMC profile handle.
  hsa_status_t(HSA_API* aqlprofile_pmc_create_packets)(
      iree_hal_amdgpu_aqlprofile_handle_t* handle,
      iree_hal_amdgpu_aqlprofile_pmc_aql_packets_t* packets,
      iree_hal_amdgpu_aqlprofile_pmc_profile_t profile,
      iree_hal_amdgpu_aqlprofile_memory_alloc_callback_t alloc_cb,
      iree_hal_amdgpu_aqlprofile_memory_dealloc_callback_t dealloc_cb,
      iree_hal_amdgpu_aqlprofile_memory_copy_callback_t memcpy_cb,
      void* user_data);

  // Deletes PM4 programs and output buffers associated with a PMC handle.
  void(HSA_API* aqlprofile_pmc_delete_packets)(
      iree_hal_amdgpu_aqlprofile_handle_t handle);

  // Iterates decoded PMC values from a completed profile handle.
  hsa_status_t(HSA_API* aqlprofile_pmc_iterate_data)(
      iree_hal_amdgpu_aqlprofile_handle_t handle,
      iree_hal_amdgpu_aqlprofile_pmc_data_callback_t callback, void* user_data);

  // Optionally returns a textual error for the most recent aqlprofile error.
  hsa_status_t(HSA_API* hsa_ven_amd_aqlprofile_error_string)(const char** str);
} iree_hal_amdgpu_libaqlprofile_t;

// Initializes |out_libaqlprofile| by dynamically loading the aqlprofile SDK.
//
// |search_paths| overrides the default library search paths and looks for the
// canonical library file under each path before falling back to the HSA library
// directory, IREE_HAL_AMDGPU_LIBAQLPROFILE_PATH, and system search paths.
iree_status_t iree_hal_amdgpu_libaqlprofile_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_string_view_list_t search_paths, iree_allocator_t host_allocator,
    iree_hal_amdgpu_libaqlprofile_t* out_libaqlprofile);

// Deinitializes |libaqlprofile| by unloading the backing library.
void iree_hal_amdgpu_libaqlprofile_deinitialize(
    iree_hal_amdgpu_libaqlprofile_t* libaqlprofile);

// Returns an IREE status with the aqlprofile error string when available.
iree_status_t iree_status_from_aqlprofile_status(
    const iree_hal_amdgpu_libaqlprofile_t* libaqlprofile, const char* file,
    uint32_t line, hsa_status_t hsa_status, const char* symbol,
    const char* message);

// Wraps an iree_hal_amdgpu_libaqlprofile_t* for error helper calls.
#define IREE_LIBAQLPROFILE(libaqlprofile) (libaqlprofile), __FILE__, __LINE__

#define IREE_RETURN_IF_AQLPROFILE_ERROR(libaqlprofile, expr, message)        \
  do {                                                                       \
    hsa_status_t hsa_status_ = (expr);                                       \
    if (IREE_UNLIKELY(hsa_status_ != HSA_STATUS_SUCCESS)) {                  \
      return iree_status_from_aqlprofile_status(                             \
          (libaqlprofile), __FILE__, __LINE__, hsa_status_, #expr, message); \
    }                                                                        \
  } while (0)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_LIBAQLPROFILE_H_
