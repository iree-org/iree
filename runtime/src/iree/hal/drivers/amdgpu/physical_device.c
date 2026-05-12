// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/physical_device.h"

#include <stdio.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/hal/drivers/amdgpu/abi/signal.h"
#include "iree/hal/drivers/amdgpu/queue_affinity.h"
#include "iree/hal/drivers/amdgpu/slab_provider.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/epoch_signal_table.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/memory/tlsf_pool.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_physical_device_options_t
//===----------------------------------------------------------------------===//

#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_MIN_SMALL_DEVICE_BLOCK_SIZE (1 * 1024)
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_MIN_LARGE_DEVICE_BLOCK_SIZE (64 * 1024)

// Not currently configurable but could be:
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_COARSE_BLOCK_POOL_SMALL_PAGE_SIZE 128
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_COARSE_BLOCK_POOL_LARGE_PAGE_SIZE 4096
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_FINE_BLOCK_POOL_SMALL_PAGE_SIZE 128
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_FINE_BLOCK_POOL_LARGE_PAGE_SIZE 4096

// Catch-all priority for direct allocations in the default pool set.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_POOL_PRIORITY_OVERSIZED 0

// Preferred priority for pooled allocations in the default pool set.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_POOL_PRIORITY_TLSF 10

typedef struct iree_hal_amdgpu_agent_first_isa_t {
  // Number of ISAs seen during iteration.
  uint32_t count;
  // First ISA returned by the HSA agent iterator.
  hsa_isa_t value;
} iree_hal_amdgpu_agent_first_isa_t;

static hsa_status_t iree_hal_amdgpu_record_first_isa(hsa_isa_t isa,
                                                     void* user_data) {
  iree_hal_amdgpu_agent_first_isa_t* first_isa =
      (iree_hal_amdgpu_agent_first_isa_t*)user_data;
  if (first_isa->count++ == 0) {
    first_isa->value = isa;
  }
  return HSA_STATUS_SUCCESS;
}

static bool iree_hal_amdgpu_parse_hex_digit(char c, uint32_t* out_value) {
  if (c >= '0' && c <= '9') {
    *out_value = (uint32_t)(c - '0');
    return true;
  } else if (c >= 'a' && c <= 'f') {
    *out_value = (uint32_t)(c - 'a' + 10);
    return true;
  } else if (c >= 'A' && c <= 'F') {
    *out_value = (uint32_t)(c - 'A' + 10);
    return true;
  }
  return false;
}

static iree_status_t iree_hal_amdgpu_query_agent_target_id(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    iree_host_size_t target_id_processor_capacity,
    char* target_id_processor_storage,
    iree_hal_amdgpu_target_id_t* out_target_id) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(target_id_processor_storage);
  IREE_ASSERT_ARGUMENT(out_target_id);

  iree_hal_amdgpu_agent_first_isa_t first_isa;
  memset(&first_isa, 0, sizeof(first_isa));
  IREE_RETURN_IF_ERROR(iree_hsa_agent_iterate_isas(
      IREE_LIBHSA(libhsa), agent, iree_hal_amdgpu_record_first_isa,
      &first_isa));
  if (first_isa.count == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "GPU agent has no reported HSA ISA");
  }

  char isa_name_buffer[128] = {0};
  uint32_t isa_name_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_hsa_isa_get_info_alt(IREE_LIBHSA(libhsa), first_isa.value,
                                HSA_ISA_INFO_NAME_LENGTH, &isa_name_length));
  if (isa_name_length == 0 ||
      isa_name_length > IREE_ARRAYSIZE(isa_name_buffer)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ISA name length invalid: %u", isa_name_length);
  }
  IREE_RETURN_IF_ERROR(
      iree_hsa_isa_get_info_alt(IREE_LIBHSA(libhsa), first_isa.value,
                                HSA_ISA_INFO_NAME, isa_name_buffer));
  iree_hal_amdgpu_target_id_t target_id;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_id_parse_hsa_isa_name(
      iree_make_string_view(isa_name_buffer, isa_name_length - /*NUL*/ 1),
      &target_id));
  if (target_id.processor.size >= target_id_processor_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "target ID processor storage too small");
  }
  memcpy(target_id_processor_storage, target_id.processor.data,
         target_id.processor.size);
  target_id_processor_storage[target_id.processor.size] = 0;
  target_id.processor = iree_make_string_view(target_id_processor_storage,
                                              target_id.processor.size);
  *out_target_id = target_id;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_query_agent_uuid(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    uint8_t out_uuid[16], bool* out_has_uuid) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_uuid);
  IREE_ASSERT_ARGUMENT(out_has_uuid);

  memset(out_uuid, 0, 16);
  *out_has_uuid = false;

  // HSA returns a prefixed ASCII string such as "GPU-4939e1d93d24ff77".
  // Unsupported devices may return fallback strings such as "GPU-XX"; those
  // are valid HSA responses but not stable identifiers for profile records.
  char uuid_string[64] = {0};
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_UUID,
      uuid_string));

  iree_string_view_t uuid_hex = iree_make_cstring_view(uuid_string);
  if (!iree_string_view_consume_prefix(&uuid_hex, IREE_SV("GPU-")) &&
      !iree_string_view_consume_prefix(&uuid_hex, IREE_SV("CPU-")) &&
      !iree_string_view_consume_prefix(&uuid_hex, IREE_SV("DSP-")) &&
      !iree_string_view_consume_prefix(&uuid_hex, IREE_SV("AIE-"))) {
    return iree_ok_status();
  }

  iree_host_size_t parsed_length = 0;
  if (uuid_hex.size == 16) {
    parsed_length = 8;
  } else if (uuid_hex.size == 32) {
    parsed_length = 16;
  } else {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < uuid_hex.size; ++i) {
    uint32_t value = 0;
    if (!iree_hal_amdgpu_parse_hex_digit(uuid_hex.data[i], &value)) {
      return iree_ok_status();
    }
  }

  if (!iree_string_view_parse_hex_bytes(uuid_hex, parsed_length, out_uuid)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "HSA device UUID was prevalidated but failed to parse: %.*s",
        (int)uuid_hex.size, uuid_hex.data);
  }
  *out_has_uuid = true;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_query_agent_pci_identity(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    iree_hal_amdgpu_physical_device_t* out_physical_device) {
  uint32_t pci_domain = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN,
      &pci_domain));
  uint32_t bdfid = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID,
      &bdfid));

  out_physical_device->pci_domain = pci_domain;
  out_physical_device->pci_bus = (bdfid >> 8) & 0xFFu;
  out_physical_device->pci_device = (bdfid >> 3) & 0x1Fu;
  out_physical_device->pci_function = bdfid & 0x7u;
  out_physical_device->has_pci_identity = 1u;
  return iree_ok_status();
}

static bool iree_hal_amdgpu_physical_device_query_pool_epoch(
    void* user_data, iree_async_axis_t axis, uint64_t epoch) {
  iree_hal_amdgpu_epoch_signal_table_t* epoch_signal_table =
      (iree_hal_amdgpu_epoch_signal_table_t*)user_data;
  hsa_signal_t epoch_signal = {0};
  if (!iree_hal_amdgpu_epoch_signal_table_lookup(epoch_signal_table, axis,
                                                 &epoch_signal)) {
    return false;
  }
  iree_amd_signal_t* signal =
      (iree_amd_signal_t*)(uintptr_t)epoch_signal.handle;
  const iree_hsa_signal_value_t current_value = iree_atomic_load(
      (iree_atomic_int64_t*)&signal->value, iree_memory_order_acquire);
  if (IREE_UNLIKELY(current_value < 0 ||
                    current_value > IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE)) {
    return false;
  }
  const uint64_t current_epoch =
      (uint64_t)IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE - (uint64_t)current_value;
  return current_epoch >= epoch;
}

static iree_string_view_t iree_hal_amdgpu_format_pool_trace_name(
    char* buffer, iree_host_size_t buffer_capacity, const char* pool_name,
    iree_host_size_t device_ordinal) {
  if (IREE_UNLIKELY(buffer_capacity == 0)) return iree_string_view_empty();
  const int name_length =
      snprintf(buffer, buffer_capacity, "iree-hal-amdgpu-l0p%" PRIhsz "-%s",
               device_ordinal, pool_name);
  if (IREE_UNLIKELY(name_length < 0)) return iree_string_view_empty();
  iree_host_size_t safe_length = (iree_host_size_t)name_length;
  if (safe_length >= buffer_capacity) safe_length = buffer_capacity - 1;
  return iree_make_string_view(buffer, safe_length);
}

static iree_hal_buffer_usage_t
iree_hal_amdgpu_physical_device_mappable_pool_supported_usage(void) {
  const iree_hal_buffer_usage_t sharing_usage =
      IREE_HAL_BUFFER_USAGE_SHARING_REPLICATE |
      IREE_HAL_BUFFER_USAGE_SHARING_CONCURRENT |
      IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE;
  return IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH |
         sharing_usage | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
         IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
         IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL |
         IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM |
         IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE;
}

void iree_hal_amdgpu_physical_device_options_initialize(
    iree_hal_amdgpu_physical_device_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));

  out_options->device_block_pools.small.block_size =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_SMALL_DEVICE_BLOCK_SIZE_DEFAULT;
  out_options->device_block_pools.small.min_blocks_per_allocation =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_SMALL_DEVICE_BLOCKS_PER_ALLOCATION_DEFAULT;
  out_options->device_block_pools.small.initial_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_SMALL_DEVICE_BLOCK_INITIAL_CAPACITY_DEFAULT;

  out_options->device_block_pools.large.block_size =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCK_SIZE_DEFAULT;
  out_options->device_block_pools.large.min_blocks_per_allocation =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCKS_PER_ALLOCATION_DEFAULT;
  out_options->device_block_pools.large.initial_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCK_INITIAL_CAPACITY_DEFAULT;

  out_options->host_block_pool_size =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_HOST_BLOCK_SIZE_DEFAULT;

  out_options->host_queue_count =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_QUEUE_COUNT;
  out_options->host_queue_aql_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_HOST_QUEUE_AQL_CAPACITY;
  out_options->host_queue_notification_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_HOST_QUEUE_NOTIFICATION_CAPACITY;
  out_options->host_queue_kernarg_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_HOST_QUEUE_KERNARG_CAPACITY;
  out_options->host_queue_upload_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_HOST_QUEUE_UPLOAD_CAPACITY;

  out_options->default_pool.range_length =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_POOL_RANGE_LENGTH_DEFAULT;
  out_options->default_pool.alignment =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_POOL_ALIGNMENT_DEFAULT;
  out_options->default_pool.frontier_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_POOL_FRONTIER_CAPACITY_DEFAULT;

  iree_hal_amdgpu_staging_pool_options_initialize(&out_options->file_staging);
}

iree_status_t iree_hal_amdgpu_physical_device_options_verify(
    const iree_hal_amdgpu_physical_device_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t cpu_agent,
    hsa_agent_t gpu_agent) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(libhsa);

  // Verify pool sizes.
  if (options->device_block_pools.small.block_size <
          IREE_HAL_AMDGPU_PHYSICAL_DEVICE_MIN_SMALL_DEVICE_BLOCK_SIZE ||
      !iree_host_size_is_power_of_two(
          options->device_block_pools.small.block_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "small device block pool size invalid, expected a "
        "power-of-two greater than %d and got %" PRIhsz,
        IREE_HAL_AMDGPU_PHYSICAL_DEVICE_MIN_SMALL_DEVICE_BLOCK_SIZE,
        options->device_block_pools.small.block_size);
  }
  if (options->device_block_pools.large.block_size <
          IREE_HAL_AMDGPU_PHYSICAL_DEVICE_MIN_LARGE_DEVICE_BLOCK_SIZE ||
      !iree_host_size_is_power_of_two(
          options->device_block_pools.large.block_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "large device block pool size invalid, expected a "
        "power-of-two greater than %d and got %" PRIhsz,
        IREE_HAL_AMDGPU_PHYSICAL_DEVICE_MIN_LARGE_DEVICE_BLOCK_SIZE,
        options->device_block_pools.large.block_size);
  }

  if (options->host_queue_count == 0 || options->host_queue_count > UINT8_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "host queue count must be in [1, %u] to fit the queue-axis encoding "
        "(got %" PRIhsz ")",
        UINT8_MAX, options->host_queue_count);
  }
  if (!iree_host_size_is_power_of_two(options->host_queue_aql_capacity) ||
      !iree_host_size_is_power_of_two(
          options->host_queue_notification_capacity) ||
      !iree_host_size_is_power_of_two(options->host_queue_kernarg_capacity) ||
      (options->host_queue_upload_capacity != 0 &&
       !iree_host_size_is_power_of_two(options->host_queue_upload_capacity))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "host queue AQL, notification, kernarg, and upload capacities must all "
        "be powers of two, with zero allowed for disabled upload capacity (got "
        "aql=%u, notification=%u, kernarg_blocks=%u, upload_bytes=%u)",
        options->host_queue_aql_capacity,
        options->host_queue_notification_capacity,
        options->host_queue_kernarg_capacity,
        options->host_queue_upload_capacity);
  }
  if (options->host_queue_kernarg_capacity / 2u <
      options->host_queue_aql_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "host queue kernarg capacity must be at least 2x the AQL queue "
        "capacity to cover one tail-padding gap at wrap (got "
        "kernarg_blocks=%u, "
        "aql_packets=%u)",
        options->host_queue_kernarg_capacity, options->host_queue_aql_capacity);
  }

  if (options->default_pool.range_length == 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "default pool range_length must be non-zero");
  }
  if (options->default_pool.alignment < IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT ||
      !iree_device_size_is_power_of_two(options->default_pool.alignment)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "default pool alignment must be a power of two >= %" PRIu64
        " (got %" PRIu64 ")",
        (uint64_t)IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT,
        (uint64_t)options->default_pool.alignment);
  }
  if (options->default_pool.range_length < options->default_pool.alignment) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "default pool range_length (%" PRIu64
                            ") must be >= alignment (%" PRIu64 ")",
                            (uint64_t)options->default_pool.range_length,
                            (uint64_t)options->default_pool.alignment);
  }
  if (options->default_pool.frontier_capacity == 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "default pool frontier_capacity must be non-zero");
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_staging_pool_options_verify(&options->file_staging));

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_physical_device_t
//===----------------------------------------------------------------------===//

iree_host_size_t iree_hal_amdgpu_physical_device_calculate_size(
    const iree_hal_amdgpu_physical_device_options_t* options) {
  IREE_ASSERT_ARGUMENT(options);
  return iree_host_align(
      sizeof(iree_hal_amdgpu_physical_device_t) +
          sizeof(iree_hal_amdgpu_host_queue_t) * options->host_queue_count,
      iree_max_align_t);
}

static iree_status_t iree_hal_amdgpu_physical_device_initialize_identity(
    iree_hal_amdgpu_system_t* system,
    const iree_hal_amdgpu_physical_device_options_t* options,
    iree_host_size_t host_ordinal,
    const iree_hal_amdgpu_host_memory_pools_t* host_memory_pools,
    iree_host_size_t device_ordinal,
    iree_hal_amdgpu_physical_device_t* out_physical_device) {
  iree_hal_amdgpu_libhsa_t* libhsa = &system->libhsa;
  hsa_agent_t device_agent = system->topology.gpu_agents[device_ordinal];
  hsa_agent_t host_agent = system->topology.cpu_agents[host_ordinal];

  // Zeroing allows deinitialization to run after any partial initialization
  // failure below.
  memset(out_physical_device, 0, sizeof(*out_physical_device));
  out_physical_device->device_agent = device_agent;
  out_physical_device->device_ordinal = device_ordinal;
  out_physical_device->host_memory_pools = *host_memory_pools;
  out_physical_device->host_queue_capacity = options->host_queue_count;
  out_physical_device->host_queue_aql_capacity =
      options->host_queue_aql_capacity;
  out_physical_device->host_queue_notification_capacity =
      options->host_queue_notification_capacity;
  out_physical_device->host_queue_kernarg_capacity =
      options->host_queue_kernarg_capacity;
  out_physical_device->host_queue_upload_capacity =
      options->host_queue_upload_capacity;

  IREE_RETURN_IF_ERROR(
      iree_hsa_agent_get_info(IREE_LIBHSA(libhsa), device_agent,
                              (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DRIVER_UID,
                              &out_physical_device->driver_uid));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_query_agent_pci_identity(
      libhsa, device_agent, out_physical_device));
  bool has_physical_device_uuid = false;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_query_agent_uuid(
      libhsa, device_agent, out_physical_device->physical_device_uuid,
      &has_physical_device_uuid));
  out_physical_device->has_physical_device_uuid =
      has_physical_device_uuid ? 1u : 0u;
  uint32_t host_numa_node = UINT32_MAX;
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), host_agent, HSA_AGENT_INFO_NODE, &host_numa_node));
  out_physical_device->host_numa_node = host_numa_node;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_physical_device_initialize_host_pools(
    const iree_hal_amdgpu_physical_device_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_physical_device_t* out_physical_device) {
  // This should be pinned to the host NUMA node associated with the devices but
  // today we rely on the OS to migrate pages as needed.
  iree_arena_block_pool_initialize(options->host_block_pool_size,
                                   host_allocator,
                                   &out_physical_device->fine_host_block_pool);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_transient_buffer_pool_initialize(
      &out_physical_device->fine_host_block_pool,
      &out_physical_device->transient_buffer_pool));
  return iree_hal_amdgpu_buffer_pool_initialize(
      &out_physical_device->fine_host_block_pool,
      &out_physical_device->materialized_buffer_pool);
}

static iree_status_t iree_hal_amdgpu_physical_device_query_global_memory_pools(
    iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t device_agent,
    hsa_amd_memory_pool_t* out_coarse_block_memory_pool,
    hsa_amd_memory_pool_t* out_fine_block_memory_pool) {
  iree_status_t status = iree_hal_amdgpu_find_coarse_global_memory_pool(
      libhsa, device_agent, out_coarse_block_memory_pool);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_find_fine_global_memory_pool(
        libhsa, device_agent, out_fine_block_memory_pool);
  }
  if (!iree_status_is_ok(status)) {
    status = iree_status_annotate(
        status, IREE_SV("AMDGPU physical device requires coarse and fine "
                        "device-local global memory pools"));
  }
  return status;
}

typedef struct iree_hal_amdgpu_physical_device_kernarg_ring_memory_t {
  // Descriptor consumed by host queue initialization.
  iree_hal_amdgpu_kernarg_ring_memory_t descriptor;
  // Host fallback access-agent list referenced by |descriptor|.
  hsa_agent_t host_access_agents[1];
} iree_hal_amdgpu_physical_device_kernarg_ring_memory_t;

static iree_status_t iree_hal_amdgpu_physical_device_query_memory_pool_access(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    hsa_amd_memory_pool_t memory_pool,
    hsa_amd_memory_pool_access_t* out_access) {
  *out_access = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  return iree_hsa_amd_agent_memory_pool_get_info(
      IREE_LIBHSA(libhsa), agent, memory_pool,
      HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, out_access);
}

static void iree_hal_amdgpu_physical_device_use_host_kernarg_memory(
    const iree_hal_amdgpu_host_memory_pools_t* host_memory_pools,
    hsa_agent_t device_agent,
    iree_hal_amdgpu_physical_device_kernarg_ring_memory_t* out_memory) {
  memset(out_memory, 0, sizeof(*out_memory));
  out_memory->host_access_agents[0] = device_agent;
  out_memory->descriptor = (iree_hal_amdgpu_kernarg_ring_memory_t){
      .memory_pool = host_memory_pools->kernarg_pool,
      .access_agents = out_memory->host_access_agents,
      .access_agent_count = 1,
  };
}

static void iree_hal_amdgpu_physical_device_use_cpu_visible_kernarg_memory(
    const iree_hal_amdgpu_cpu_visible_device_coarse_memory_t* capability,
    iree_hal_amdgpu_physical_device_kernarg_ring_memory_t* out_memory) {
  memset(out_memory, 0, sizeof(*out_memory));
  out_memory->descriptor = (iree_hal_amdgpu_kernarg_ring_memory_t){
      .memory_pool = capability->memory_pool,
      .access_agents = capability->access_agents,
      .access_agent_count = capability->access_agent_count,
      .publication = capability->host_write_publication,
  };
}

static hsa_amd_hdp_flush_t
iree_hal_amdgpu_physical_device_query_hdp_flush_registers(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t device_agent) {
  hsa_amd_hdp_flush_t hdp_flush = {0};
  const hsa_status_t hsa_status = iree_hsa_agent_get_info_raw(
      libhsa, device_agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_HDP_FLUSH,
      &hdp_flush);
  if (hsa_status != HSA_STATUS_SUCCESS) {
    memset(&hdp_flush, 0, sizeof(hdp_flush));
  }
  return hdp_flush;
}

static iree_status_t
iree_hal_amdgpu_physical_device_query_svm_direct_host_access(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t device_agent,
    bool* out_direct_host_access) {
  *out_direct_host_access = false;
  return iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), device_agent,
      (hsa_agent_info_t)HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS,
      out_direct_host_access);
}

static iree_status_t
iree_hal_amdgpu_physical_device_initialize_cpu_visible_device_coarse_memory(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t device_agent,
    hsa_amd_memory_pool_t device_coarse_memory_pool,
    iree_hal_amdgpu_gfxip_version_t gfxip_version,
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_amdgpu_cpu_visible_device_coarse_memory_t* out_memory) {
  memset(out_memory, 0, sizeof(*out_memory));
  if (!device_coarse_memory_pool.handle || topology->cpu_agent_count == 0) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(topology->cpu_agent_count >
                    IREE_HAL_AMDGPU_MAX_CPU_AGENT)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU topology has %" PRIhsz
        " CPU agents but CPU-visible coarse memory tracks at most %d",
        topology->cpu_agent_count, IREE_HAL_AMDGPU_MAX_CPU_AGENT);
  }
  if (!iree_hal_amdgpu_kernarg_ring_supports_host_write_publication()) {
    return iree_ok_status();
  }
  if (!iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(gfxip_version)) {
    return iree_ok_status();
  }

  const hsa_amd_hdp_flush_t hdp_flush =
      iree_hal_amdgpu_physical_device_query_hdp_flush_registers(libhsa,
                                                                device_agent);
  if (!hdp_flush.HDP_MEM_FLUSH_CNTL || !hdp_flush.HDP_REG_FLUSH_CNTL) {
    return iree_ok_status();
  }

  hsa_amd_memory_pool_access_t cpu_access[IREE_HAL_AMDGPU_MAX_CPU_AGENT] = {0};
  for (iree_host_size_t i = 0; i < topology->cpu_agent_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_physical_device_query_memory_pool_access(
            libhsa, topology->cpu_agents[i], device_coarse_memory_pool,
            &cpu_access[i]));
  }

  const iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection = {
      .device_agent = device_agent,
      .memory_pool = device_coarse_memory_pool,
      .gfxip_version = gfxip_version,
      .cpu =
          {
              .agents = topology->cpu_agents,
              .access = cpu_access,
              .count = topology->cpu_agent_count,
          },
      .hdp =
          {
              .registers = hdp_flush,
          },
      .flags =
          IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_SELECTION_FLAG_HOST_WRITE_PUBLICATION_SUPPORTED,
  };
  return iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(&selection,
                                                                 out_memory);
}

static iree_status_t
iree_hal_amdgpu_physical_device_initialize_memory_system_capabilities(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_system_info_t* system_info, hsa_agent_t device_agent,
    hsa_amd_memory_pool_t fine_block_memory_pool,
    const iree_hal_amdgpu_cpu_visible_device_coarse_memory_t*
        cpu_visible_device_coarse_memory,
    iree_hal_amdgpu_memory_system_capabilities_t* out_capabilities) {
  bool svm_direct_host_access = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_physical_device_query_svm_direct_host_access(
          libhsa, device_agent, &svm_direct_host_access));

  const iree_hal_amdgpu_memory_system_capabilities_selection_t selection = {
      .svm =
          {
              .supported = system_info->svm.supported,
              .accessible_by_default = system_info->svm.accessible_by_default,
              .xnack_enabled = system_info->svm.xnack_enabled,
              .direct_host_access = svm_direct_host_access ? 1u : 0u,
          },
      .device_local =
          {
              .fine_memory_pool = fine_block_memory_pool,
              .coarse_cpu_visible_memory = cpu_visible_device_coarse_memory,
          },
  };
  iree_hal_amdgpu_select_memory_system_capabilities(&selection,
                                                    out_capabilities);
  return iree_ok_status();
}

static void iree_hal_amdgpu_physical_device_select_kernarg_ring_memory(
    const iree_hal_amdgpu_physical_device_t* physical_device,
    const iree_hal_amdgpu_host_memory_pools_t* host_memory_pools,
    iree_hal_amdgpu_physical_device_kernarg_ring_memory_t* out_memory) {
  iree_hal_amdgpu_physical_device_use_host_kernarg_memory(
      host_memory_pools, physical_device->device_agent, out_memory);
  if (!iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
          &physical_device->cpu_visible_device_coarse_memory)) {
    return;
  }
  iree_hal_amdgpu_physical_device_use_cpu_visible_kernarg_memory(
      &physical_device->cpu_visible_device_coarse_memory, out_memory);
}

static iree_status_t iree_hal_amdgpu_physical_device_initialize_block_pool(
    iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_block_pool_options_t pool_options, hsa_agent_t device_agent,
    hsa_amd_memory_pool_t memory_pool, const char* trace_name_prefix,
    iree_host_size_t device_ordinal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_block_pool_t* out_block_pool) {
  char trace_name[64] = {0};
  pool_options.trace_name = iree_hal_amdgpu_format_pool_trace_name(
      trace_name, IREE_ARRAYSIZE(trace_name), trace_name_prefix,
      device_ordinal);
  return iree_hal_amdgpu_block_pool_initialize(libhsa, pool_options,
                                               device_agent, memory_pool,
                                               host_allocator, out_block_pool);
}

static iree_status_t
iree_hal_amdgpu_physical_device_initialize_device_block_pools_and_allocators(
    iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_physical_device_options_t* options,
    hsa_agent_t device_agent, iree_host_size_t device_ordinal,
    hsa_amd_memory_pool_t coarse_block_memory_pool,
    hsa_amd_memory_pool_t fine_block_memory_pool,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_physical_device_t* out_physical_device) {
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_physical_device_initialize_block_pool(
      libhsa, options->device_block_pools.small, device_agent,
      coarse_block_memory_pool, "coarse-small-block", device_ordinal,
      host_allocator, &out_physical_device->coarse_block_pools.small));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_physical_device_initialize_block_pool(
      libhsa, options->device_block_pools.large, device_agent,
      coarse_block_memory_pool, "coarse-large-block", device_ordinal,
      host_allocator, &out_physical_device->coarse_block_pools.large));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_physical_device_initialize_block_pool(
      libhsa, options->device_block_pools.small, device_agent,
      fine_block_memory_pool, "fine-small-block", device_ordinal,
      host_allocator, &out_physical_device->fine_block_pools.small));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_physical_device_initialize_block_pool(
      libhsa, options->device_block_pools.large, device_agent,
      fine_block_memory_pool, "fine-large-block", device_ordinal,
      host_allocator, &out_physical_device->fine_block_pools.large));

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_block_allocator_initialize(
      &out_physical_device->coarse_block_pools.small,
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_COARSE_BLOCK_POOL_SMALL_PAGE_SIZE,
      &out_physical_device->coarse_block_allocators.small));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_block_allocator_initialize(
      &out_physical_device->coarse_block_pools.large,
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_COARSE_BLOCK_POOL_LARGE_PAGE_SIZE,
      &out_physical_device->coarse_block_allocators.large));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_block_allocator_initialize(
      &out_physical_device->fine_block_pools.small,
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_FINE_BLOCK_POOL_SMALL_PAGE_SIZE,
      &out_physical_device->fine_block_allocators.small));
  return iree_hal_amdgpu_block_allocator_initialize(
      &out_physical_device->fine_block_pools.large,
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_FINE_BLOCK_POOL_LARGE_PAGE_SIZE,
      &out_physical_device->fine_block_allocators.large);
}

static iree_status_t iree_hal_amdgpu_physical_device_preallocate_host_pool(
    const iree_hal_amdgpu_physical_device_options_t* options,
    iree_hal_amdgpu_physical_device_t* out_physical_device) {
  if (!options->host_block_pool_initial_capacity) return iree_ok_status();
  return iree_arena_block_pool_preallocate(
      &out_physical_device->fine_host_block_pool,
      options->host_block_pool_initial_capacity);
}

static iree_status_t
iree_hal_amdgpu_physical_device_initialize_default_pool_resources(
    iree_hal_device_t* logical_device, iree_hal_amdgpu_system_t* system,
    const iree_hal_amdgpu_physical_device_options_t* options,
    iree_async_proactor_t* proactor, iree_host_size_t device_ordinal,
    hsa_amd_memory_pool_t coarse_block_memory_pool,
    iree_hal_queue_affinity_t queue_affinity_mask,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_physical_device_t* out_physical_device) {
  iree_hal_amdgpu_libhsa_t* libhsa = &system->libhsa;

  IREE_RETURN_IF_ERROR(iree_async_notification_create(
      proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE,
      &out_physical_device->default_pool_notification));

  char trace_name[64] = {0};
  iree_string_view_t slab_trace_name = iree_hal_amdgpu_format_pool_trace_name(
      trace_name, IREE_ARRAYSIZE(trace_name), "default-slab", device_ordinal);
  iree_hal_amdgpu_slab_provider_memory_pool_properties_t properties;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_slab_provider_query_memory_pool_properties(
          libhsa, coarse_block_memory_pool, &properties));
  const iree_hal_amdgpu_slab_provider_options_t default_slab_options = {
      .memory_pool = coarse_block_memory_pool,
      .memory_type = properties.memory_type,
      .supported_usage = properties.supported_usage,
  };
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_slab_provider_create(
      logical_device, libhsa, &system->topology, default_slab_options,
      device_ordinal, queue_affinity_mask,
      &out_physical_device->materialized_buffer_pool, slab_trace_name,
      host_allocator, &out_physical_device->default_slab_provider));

  if (properties.allocation_alignment < options->default_pool.alignment) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "default pool alignment %" PRIu64
        " exceeds HSA memory pool allocation alignment %" PRIu64,
        (uint64_t)options->default_pool.alignment,
        (uint64_t)properties.allocation_alignment);
  }
  iree_device_size_t range_length = options->default_pool.range_length;
  if (!iree_device_size_checked_align(
          range_length, properties.allocation_granule, &range_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "default pool range_length %" PRIu64
        " overflows while aligning to HSA allocation granule %" PRIu64,
        (uint64_t)options->default_pool.range_length,
        (uint64_t)properties.allocation_granule);
  }
  out_physical_device->default_pool_options = (iree_hal_tlsf_pool_options_t){
      .tlsf_options =
          {
              .range_length = range_length,
              .alignment = options->default_pool.alignment,
              .frontier_capacity = options->default_pool.frontier_capacity,
          },
      .budget_limit = 0,
  };

  iree_hal_amdgpu_slab_provider_memory_pool_properties_t host_properties;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_slab_provider_query_memory_pool_properties(
          libhsa, out_physical_device->host_memory_pools.fine_pool,
          &host_properties));
  if (host_properties.allocation_alignment < options->default_pool.alignment) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "host queue allocation pool alignment %" PRIu64
        " exceeds HSA host memory pool allocation alignment %" PRIu64,
        (uint64_t)options->default_pool.alignment,
        (uint64_t)host_properties.allocation_alignment);
  }
  char host_trace_name[64] = {0};
  iree_string_view_t host_slab_trace_name =
      iree_hal_amdgpu_format_pool_trace_name(host_trace_name,
                                             IREE_ARRAYSIZE(host_trace_name),
                                             "host-slab", device_ordinal);
  const iree_hal_amdgpu_slab_provider_options_t host_slab_options = {
      .memory_pool = out_physical_device->host_memory_pools.fine_pool,
      .memory_type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .supported_usage =
          iree_hal_amdgpu_physical_device_mappable_pool_supported_usage(),
  };
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_slab_provider_create(
      logical_device, libhsa, &system->topology, host_slab_options,
      device_ordinal, queue_affinity_mask,
      &out_physical_device->materialized_buffer_pool, host_slab_trace_name,
      host_allocator, &out_physical_device->default_host_slab_provider));
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_physical_device_initialize_staging(
    iree_hal_device_t* logical_device, iree_hal_amdgpu_system_t* system,
    const iree_hal_amdgpu_physical_device_options_t* options,
    const iree_hal_amdgpu_host_memory_pools_t* host_memory_pools,
    iree_hal_queue_affinity_t queue_affinity_mask,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_physical_device_t* out_physical_device) {
  return iree_hal_amdgpu_staging_pool_initialize(
      logical_device, &system->libhsa, &system->topology, host_memory_pools,
      queue_affinity_mask, &options->file_staging, host_allocator,
      &out_physical_device->file_staging_pool);
}

static iree_status_t iree_hal_amdgpu_physical_device_initialize_signal_pool(
    iree_hal_amdgpu_libhsa_t* libhsa, iree_allocator_t host_allocator,
    iree_hal_amdgpu_physical_device_t* out_physical_device) {
  return iree_hal_amdgpu_host_signal_pool_initialize(
      libhsa,
      /*initial_capacity=*/IREE_HAL_AMDGPU_HOST_SIGNAL_POOL_BATCH_SIZE_DEFAULT,
      /*batch_size=*/0, host_allocator, &out_physical_device->host_signal_pool);
}

static iree_status_t
iree_hal_amdgpu_physical_device_initialize_device_library_and_blit_context(
    iree_hal_amdgpu_system_t* system, hsa_agent_t device_agent,
    iree_host_size_t device_ordinal,
    iree_hal_amdgpu_physical_device_t* out_physical_device) {
  iree_hal_amdgpu_libhsa_t* libhsa = &system->libhsa;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_device_library_populate_agent_kernels(
      &system->device_library, device_agent,
      &out_physical_device->device_kernels));

  uint32_t compute_unit_count = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), device_agent,
      (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
      &compute_unit_count));
  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_hsa_agent_get_info(IREE_LIBHSA(libhsa), device_agent,
                              HSA_AGENT_INFO_WAVEFRONT_SIZE, &wavefront_size));

  // Validate launch metadata before passing it to the blit context. A broken
  // HSA bring-up that returns garbage here must fail loud with a clear message
  // rather than letting the blit path silently dispatch with wrong geometry.
  if (compute_unit_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HSA reported 0 compute units for device agent "
                            "ordinal %" PRIhsz,
                            device_ordinal);
  }
  if (wavefront_size != 32 && wavefront_size != 64) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HSA reported unsupported wavefront size %u for device agent ordinal "
        "%" PRIhsz " (expected 32 or 64)",
        wavefront_size, device_ordinal);
  }
  iree_hal_amdgpu_device_buffer_transfer_context_initialize(
      &out_physical_device->device_kernels, compute_unit_count, wavefront_size,
      &out_physical_device->buffer_transfer_context);
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_physical_device_initialize_vendor_packet_strategy(
    iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_physical_device_options_t* options,
    hsa_agent_t device_agent,
    iree_hal_amdgpu_physical_device_t* out_physical_device) {
  iree_hal_amdgpu_target_id_t target_id;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_query_agent_target_id(
      libhsa, device_agent,
      sizeof(out_physical_device->isa.target_id_processor),
      out_physical_device->isa.target_id_processor, &target_id));
  iree_hal_amdgpu_gfxip_version_t gfxip_version = target_id.version;

  iree_hal_amdgpu_vendor_packet_capability_flags_t vendor_packet_capabilities =
      iree_hal_amdgpu_select_vendor_packet_capabilities(gfxip_version);
  iree_hal_amdgpu_pm4_timestamp_strategy_t pm4_timestamp_strategy =
      iree_hal_amdgpu_select_pm4_timestamp_strategy(gfxip_version);
  iree_hal_amdgpu_wait_barrier_strategy_t wait_barrier_strategy =
      IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_DEFER;
  if (!options->force_wait_barrier_defer) {
    wait_barrier_strategy = iree_hal_amdgpu_select_wait_barrier_strategy(
        vendor_packet_capabilities);
  }
  out_physical_device->isa.target_id = target_id;
  out_physical_device->vendor_packet_capabilities = vendor_packet_capabilities;
  out_physical_device->wait_barrier_strategy = wait_barrier_strategy;
  out_physical_device->pm4_timestamp_strategy = pm4_timestamp_strategy;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_physical_device_initialize(
    iree_hal_device_t* logical_device, iree_hal_amdgpu_system_t* system,
    const iree_hal_amdgpu_physical_device_options_t* options,
    iree_async_proactor_t* proactor, iree_host_size_t host_ordinal,
    const iree_hal_amdgpu_host_memory_pools_t* host_memory_pools,
    iree_host_size_t device_ordinal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_physical_device_t* out_physical_device) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(system);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(host_memory_pools);
  IREE_ASSERT_ARGUMENT(out_physical_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_libhsa_t* libhsa = &system->libhsa;
  hsa_agent_t device_agent = system->topology.gpu_agents[device_ordinal];

  iree_status_t status = iree_hal_amdgpu_physical_device_initialize_identity(
      system, options, host_ordinal, host_memory_pools, device_ordinal,
      out_physical_device);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_physical_device_initialize_host_pools(
        options, host_allocator, out_physical_device);
  }

  // Find the device memory pools and create block pools/allocators.
  hsa_amd_memory_pool_t coarse_block_memory_pool = {0};
  hsa_amd_memory_pool_t fine_block_memory_pool = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_physical_device_query_global_memory_pools(
        libhsa, device_agent, &coarse_block_memory_pool,
        &fine_block_memory_pool);
  }
  if (iree_status_is_ok(status)) {
    out_physical_device->prepublished_kernarg_storage =
        iree_hal_amdgpu_select_prepublished_kernarg_storage(
            fine_block_memory_pool);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_physical_device_preallocate_host_pool(
        options, out_physical_device);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amdgpu_physical_device_initialize_device_block_pools_and_allocators(
            libhsa, options, device_agent, device_ordinal,
            coarse_block_memory_pool, fine_block_memory_pool, host_allocator,
            out_physical_device);
  }

  // Create the default queue-allocation slab provider over the device
  // coarse-grained HSA pool and derive the TLSF policy used once topology
  // assignment provides an epoch query.
  iree_hal_queue_affinity_t queue_affinity_mask = 0;
  const iree_hal_amdgpu_queue_affinity_domain_t queue_affinity_domain = {
      .supported_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .physical_device_count = system->topology.gpu_agent_count,
      .queue_count_per_physical_device = options->host_queue_count,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_queue_affinity_for_physical_device(
        queue_affinity_domain, device_ordinal, &queue_affinity_mask);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_physical_device_initialize_default_pool_resources(
        logical_device, system, options, proactor, device_ordinal,
        coarse_block_memory_pool, queue_affinity_mask, host_allocator,
        out_physical_device);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_physical_device_initialize_staging(
        logical_device, system, options, host_memory_pools, queue_affinity_mask,
        host_allocator, out_physical_device);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_physical_device_initialize_signal_pool(
        libhsa, host_allocator, out_physical_device);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amdgpu_physical_device_initialize_device_library_and_blit_context(
            system, device_agent, device_ordinal, out_physical_device);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_physical_device_initialize_vendor_packet_strategy(
        libhsa, options, device_agent, out_physical_device);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amdgpu_physical_device_initialize_cpu_visible_device_coarse_memory(
            libhsa, device_agent, coarse_block_memory_pool,
            out_physical_device->isa.target_id.version, &system->topology,
            &out_physical_device->cpu_visible_device_coarse_memory);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amdgpu_physical_device_initialize_memory_system_capabilities(
            libhsa, &system->info, device_agent, fine_block_memory_pool,
            &out_physical_device->cpu_visible_device_coarse_memory,
            &out_physical_device->memory_system);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_physical_device_deinitialize(out_physical_device);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_physical_device_create_pool_pair(
    iree_hal_amdgpu_physical_device_t* physical_device,
    iree_hal_amdgpu_epoch_signal_table_t* epoch_signal_table,
    iree_hal_slab_provider_t* slab_provider,
    iree_hal_tlsf_pool_options_t pool_options, const char* pool_name,
    const char* oversized_pool_name, iree_allocator_t host_allocator,
    iree_hal_pool_t** out_pool, iree_hal_pool_t** out_oversized_pool) {
  char pool_trace_name[64] = {0};
  pool_options.trace_name = iree_hal_amdgpu_format_pool_trace_name(
      pool_trace_name, IREE_ARRAYSIZE(pool_trace_name), pool_name,
      physical_device->device_ordinal);
  iree_status_t status = iree_hal_tlsf_pool_create(
      pool_options, slab_provider, physical_device->default_pool_notification,
      (iree_hal_pool_epoch_query_t){
          .fn = iree_hal_amdgpu_physical_device_query_pool_epoch,
          .user_data = epoch_signal_table,
      },
      host_allocator, out_pool);

  char oversized_pool_trace_name[64] = {0};
  if (iree_status_is_ok(status)) {
    iree_hal_passthrough_pool_options_t oversized_pool_options = {
        .trace_name = iree_hal_amdgpu_format_pool_trace_name(
            oversized_pool_trace_name,
            IREE_ARRAYSIZE(oversized_pool_trace_name), oversized_pool_name,
            physical_device->device_ordinal),
    };
    status = iree_hal_passthrough_pool_create(
        oversized_pool_options, slab_provider,
        physical_device->default_pool_notification, host_allocator,
        out_oversized_pool);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_physical_device_create_default_pools(
    iree_hal_amdgpu_physical_device_t* physical_device,
    iree_hal_amdgpu_epoch_signal_table_t* epoch_signal_table,
    iree_allocator_t host_allocator) {
  IREE_RETURN_IF_ERROR(iree_hal_pool_set_initialize(
      /*initial_capacity=*/4, host_allocator,
      &physical_device->default_pool_set));

  iree_status_t status = iree_hal_amdgpu_physical_device_create_pool_pair(
      physical_device, epoch_signal_table,
      physical_device->default_slab_provider,
      physical_device->default_pool_options, "tlsf", "oversized",
      host_allocator, &physical_device->default_pool,
      &physical_device->default_oversized_pool);
  iree_hal_tlsf_pool_options_t host_pool_options =
      physical_device->default_pool_options;
  host_pool_options.tlsf_options.range_length =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_HOST_POOL_RANGE_LENGTH_DEFAULT;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_physical_device_create_pool_pair(
        physical_device, epoch_signal_table,
        physical_device->default_host_slab_provider, host_pool_options,
        "host-tlsf", "host-oversized", host_allocator,
        &physical_device->default_host_pool,
        &physical_device->default_host_oversized_pool);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_set_register(
        &physical_device->default_pool_set,
        IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_POOL_PRIORITY_OVERSIZED,
        physical_device->default_oversized_pool);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_set_register(
        &physical_device->default_pool_set,
        IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_POOL_PRIORITY_OVERSIZED,
        physical_device->default_host_oversized_pool);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_set_register(
        &physical_device->default_pool_set,
        IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_POOL_PRIORITY_TLSF,
        physical_device->default_pool);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_set_register(
        &physical_device->default_pool_set,
        IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_POOL_PRIORITY_TLSF,
        physical_device->default_host_pool);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_physical_device_assign_frontier(
    iree_hal_device_t* logical_device, iree_hal_amdgpu_system_t* system,
    iree_async_proactor_t* proactor,
    iree_async_frontier_tracker_t* frontier_tracker,
    iree_async_axis_t base_axis,
    iree_hal_amdgpu_epoch_signal_table_t* epoch_signal_table,
    const iree_hal_amdgpu_host_memory_pools_t* host_memory_pools,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_physical_device_t* physical_device) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_libhsa_t* libhsa = &system->libhsa;
  const uint8_t session_epoch = iree_async_axis_session(base_axis);
  const uint8_t machine_index = iree_async_axis_machine(base_axis);
  iree_status_t status = iree_hal_amdgpu_physical_device_create_default_pools(
      physical_device, epoch_signal_table, host_allocator);
  const iree_hal_amdgpu_queue_affinity_domain_t queue_affinity_domain = {
      .supported_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .physical_device_count = system->topology.gpu_agent_count,
      .queue_count_per_physical_device = physical_device->host_queue_capacity,
  };
  iree_hal_amdgpu_physical_device_kernarg_ring_memory_t kernarg_ring_memory;
  iree_hal_amdgpu_physical_device_select_kernarg_ring_memory(
      physical_device, host_memory_pools, &kernarg_ring_memory);
  for (iree_host_size_t queue_ordinal = 0;
       queue_ordinal < physical_device->host_queue_capacity &&
       iree_status_is_ok(status);
       ++queue_ordinal) {
    const iree_host_size_t logical_queue_ordinal =
        physical_device->device_ordinal * physical_device->host_queue_capacity +
        queue_ordinal;
    iree_hal_amdgpu_queue_affinity_resolved_t resolved;
    status = iree_hal_amdgpu_queue_affinity_resolve_ordinal(
        queue_affinity_domain, logical_queue_ordinal, &resolved);
    if (!iree_status_is_ok(status)) break;
    iree_async_axis_t queue_axis = iree_async_axis_make_queue(
        session_epoch, machine_index, (uint8_t)physical_device->device_ordinal,
        (uint8_t)queue_ordinal);
    iree_thread_affinity_t completion_thread_affinity;
    iree_thread_affinity_set_group_any(physical_device->host_numa_node,
                                       &completion_thread_affinity);
    status = iree_hal_amdgpu_host_queue_initialize(
        libhsa, logical_device, proactor, physical_device->device_agent,
        &kernarg_ring_memory.descriptor, host_memory_pools->fine_pool,
        frontier_tracker, queue_axis, resolved.queue_affinity,
        completion_thread_affinity, physical_device->wait_barrier_strategy,
        physical_device->vendor_packet_capabilities,
        physical_device->pm4_timestamp_strategy, epoch_signal_table,
        &physical_device->fine_host_block_pool,
        &physical_device->fine_block_pools.small,
        &physical_device->buffer_transfer_context,
        &physical_device->default_pool_set, physical_device->default_pool,
        &physical_device->transient_buffer_pool,
        &physical_device->file_staging_pool, physical_device->device_ordinal,
        physical_device->host_queue_aql_capacity,
        physical_device->host_queue_notification_capacity,
        physical_device->host_queue_kernarg_capacity,
        physical_device->host_queue_upload_capacity, host_allocator,
        &physical_device->host_queues[queue_ordinal]);
    if (iree_status_is_ok(status)) {
      physical_device->host_queue_count = queue_ordinal + 1;
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_physical_device_deassign_frontier(physical_device);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_physical_device_deassign_frontier(
    iree_hal_amdgpu_physical_device_t* physical_device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  for (iree_host_size_t i = 0; i < physical_device->host_queue_count; ++i) {
    iree_hal_amdgpu_host_queue_deinitialize(&physical_device->host_queues[i]);
  }
  physical_device->host_queue_count = 0;
  if (physical_device->default_pool_set.entries) {
    iree_hal_pool_set_deinitialize(&physical_device->default_pool_set);
  }
  iree_hal_pool_release(physical_device->default_host_oversized_pool);
  physical_device->default_host_oversized_pool = NULL;
  iree_hal_pool_release(physical_device->default_host_pool);
  physical_device->default_host_pool = NULL;
  iree_hal_pool_release(physical_device->default_oversized_pool);
  physical_device->default_oversized_pool = NULL;
  iree_hal_pool_release(physical_device->default_pool);
  physical_device->default_pool = NULL;
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_physical_device_set_hsa_profiling_enabled(
    iree_hal_amdgpu_physical_device_t* physical_device, bool enabled) {
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, enabled ? 1 : 0);

  iree_status_t status = iree_ok_status();
  iree_host_size_t changed_count = 0;
  for (iree_host_size_t i = 0;
       i < physical_device->host_queue_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_amdgpu_host_queue_set_hsa_profiling_enabled(
        &physical_device->host_queues[i], enabled);
    if (iree_status_is_ok(status)) {
      ++changed_count;
    }
  }

  if (!iree_status_is_ok(status) && enabled) {
    for (iree_host_size_t i = 0; i < changed_count; ++i) {
      status = iree_status_join(
          status, iree_hal_amdgpu_host_queue_set_hsa_profiling_enabled(
                      &physical_device->host_queues[i], false));
    }
  } else if (!enabled) {
    for (iree_host_size_t i = changed_count;
         i < physical_device->host_queue_count; ++i) {
      status = iree_status_join(
          status, iree_hal_amdgpu_host_queue_set_hsa_profiling_enabled(
                      &physical_device->host_queues[i], false));
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_physical_device_deinitialize(
    iree_hal_amdgpu_physical_device_t* physical_device) {
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_physical_device_deassign_frontier(physical_device);

  iree_hal_amdgpu_host_signal_pool_deinitialize(
      &physical_device->host_signal_pool);

  iree_hal_slab_provider_release(physical_device->default_slab_provider);
  iree_hal_slab_provider_release(physical_device->default_host_slab_provider);
  iree_async_notification_release(physical_device->default_pool_notification);

  iree_hal_amdgpu_staging_pool_deinitialize(
      &physical_device->file_staging_pool);

  iree_hal_amdgpu_transient_buffer_pool_deinitialize(
      &physical_device->transient_buffer_pool);
  iree_hal_amdgpu_buffer_pool_deinitialize(
      &physical_device->materialized_buffer_pool);

  iree_arena_block_pool_deinitialize(&physical_device->fine_host_block_pool);

  iree_hal_amdgpu_block_allocator_deinitialize(
      &physical_device->coarse_block_allocators.small);
  iree_hal_amdgpu_block_allocator_deinitialize(
      &physical_device->coarse_block_allocators.large);
  iree_hal_amdgpu_block_allocator_deinitialize(
      &physical_device->fine_block_allocators.small);
  iree_hal_amdgpu_block_allocator_deinitialize(
      &physical_device->fine_block_allocators.large);
  iree_hal_amdgpu_block_pool_deinitialize(
      &physical_device->coarse_block_pools.small);
  iree_hal_amdgpu_block_pool_deinitialize(
      &physical_device->coarse_block_pools.large);
  iree_hal_amdgpu_block_pool_deinitialize(
      &physical_device->fine_block_pools.small);
  iree_hal_amdgpu_block_pool_deinitialize(
      &physical_device->fine_block_pools.large);

  memset(physical_device, 0, sizeof(*physical_device));

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_physical_device_trim(
    iree_hal_amdgpu_physical_device_t* physical_device) {
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < physical_device->host_queue_count; ++i) {
    physical_device->host_queues[i].base.vtable->trim(
        &physical_device->host_queues[i].base);
  }

  iree_hal_amdgpu_block_pool_trim(&physical_device->coarse_block_pools.small);
  iree_hal_amdgpu_block_pool_trim(&physical_device->coarse_block_pools.large);
  iree_hal_amdgpu_block_pool_trim(&physical_device->fine_block_pools.small);
  iree_hal_amdgpu_block_pool_trim(&physical_device->fine_block_pools.large);

  iree_arena_block_pool_trim(&physical_device->fine_host_block_pool);

  iree_status_t status = iree_ok_status();
  if (physical_device->default_pool) {
    status = iree_hal_pool_trim(physical_device->default_pool);
  }
  if (iree_status_is_ok(status) && physical_device->default_oversized_pool) {
    status = iree_hal_pool_trim(physical_device->default_oversized_pool);
  }
  if (iree_status_is_ok(status) && physical_device->default_host_pool) {
    status = iree_hal_pool_trim(physical_device->default_host_pool);
  }
  if (iree_status_is_ok(status) &&
      physical_device->default_host_oversized_pool) {
    status = iree_hal_pool_trim(physical_device->default_host_oversized_pool);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
