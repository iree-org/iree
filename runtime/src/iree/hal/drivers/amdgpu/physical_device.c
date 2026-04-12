// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/physical_device.h"

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/hal/drivers/amdgpu/abi/signal.h"
#include "iree/hal/drivers/amdgpu/slab_provider.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/epoch_signal_table.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"
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

typedef struct iree_hal_amdgpu_gfxip_version_t {
  uint32_t major;
  uint32_t minor;
  uint32_t stepping;
} iree_hal_amdgpu_gfxip_version_t;

typedef struct iree_hal_amdgpu_agent_first_isa_t {
  uint32_t count;
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

static bool iree_hal_amdgpu_parse_decimal_digit(char c, uint32_t* out_value) {
  if (c < '0' || c > '9') return false;
  *out_value = (uint32_t)(c - '0');
  return true;
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

static iree_status_t iree_hal_amdgpu_parse_gfxip_version(
    iree_string_view_t isa_name, iree_hal_amdgpu_gfxip_version_t* out_version) {
  IREE_ASSERT_ARGUMENT(out_version);
  memset(out_version, 0, sizeof(*out_version));

  iree_host_size_t gfx_ordinal = isa_name.size;
  for (iree_host_size_t i = 0; i + 3 <= isa_name.size; ++i) {
    if (isa_name.data[i] == 'g' && isa_name.data[i + 1] == 'f' &&
        isa_name.data[i + 2] == 'x') {
      gfx_ordinal = i + 3;
      break;
    }
  }
  if (gfx_ordinal >= isa_name.size) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "ISA name does not contain gfx version: %.*s",
                            (int)isa_name.size, isa_name.data);
  }

  const char* version = isa_name.data + gfx_ordinal;
  const iree_host_size_t version_length = isa_name.size - gfx_ordinal;
  uint32_t major0 = 0;
  uint32_t major1 = 0;
  uint32_t minor = 0;
  uint32_t stepping = 0;
  if (version_length >= 4 &&
      iree_hal_amdgpu_parse_decimal_digit(version[0], &major0) && major0 == 1 &&
      iree_hal_amdgpu_parse_decimal_digit(version[1], &major1) &&
      iree_hal_amdgpu_parse_decimal_digit(version[2], &minor) &&
      iree_hal_amdgpu_parse_hex_digit(version[3], &stepping)) {
    out_version->major = 10 + major1;
    out_version->minor = minor;
    out_version->stepping = stepping;
    return iree_ok_status();
  }
  if (version_length >= 3 &&
      iree_hal_amdgpu_parse_decimal_digit(version[0], &major0) &&
      iree_hal_amdgpu_parse_decimal_digit(version[1], &minor) &&
      iree_hal_amdgpu_parse_hex_digit(version[2], &stepping)) {
    out_version->major = major0;
    out_version->minor = minor;
    out_version->stepping = stepping;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported ISA gfx version syntax: %.*s",
                          (int)isa_name.size, isa_name.data);
}

static iree_status_t iree_hal_amdgpu_query_agent_gfxip_version(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    iree_hal_amdgpu_gfxip_version_t* out_version) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_version);

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
  return iree_hal_amdgpu_parse_gfxip_version(
      iree_make_string_view(isa_name_buffer, isa_name_length - /*NUL*/ 1),
      out_version);
}

static iree_hal_amdgpu_wait_barrier_strategy_t
iree_hal_amdgpu_select_wait_barrier_strategy(
    iree_hal_amdgpu_gfxip_version_t version) {
  // Matches CLR's barrier_value_packet_ gate:
  //   gfx9.0.10 or gfx9.[minor >= 4].[stepping 0..2]
  if (version.major == 9 && ((version.minor == 0 && version.stepping == 10) ||
                             (version.minor >= 4 && version.stepping <= 2))) {
    return IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_AQL_BARRIER_VALUE;
  }
  // WAIT_REG_MEM64 is present in the gfx10+ PM4 packet tables. Gfx9 has only
  // the 32-bit WAIT_REG_MEM variant, and non-CDNA gfx9 therefore defers.
  if (version.major >= 10) {
    return IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_PM4_WAIT_REG_MEM64;
  }
  return IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_DEFER;
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

  out_options->default_pool.range_length =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_POOL_RANGE_LENGTH_DEFAULT;
  out_options->default_pool.alignment =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_POOL_ALIGNMENT_DEFAULT;
  out_options->default_pool.frontier_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_POOL_FRONTIER_CAPACITY_DEFAULT;
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
        IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCK_SIZE_DEFAULT,
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
      !iree_host_size_is_power_of_two(options->host_queue_kernarg_capacity)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "host queue AQL, notification, and kernarg capacities must all be "
        "powers of two (got aql=%u, notification=%u, kernarg_blocks=%u)",
        options->host_queue_aql_capacity,
        options->host_queue_notification_capacity,
        options->host_queue_kernarg_capacity);
  }
  if (options->host_queue_kernarg_capacity / 2u <
      options->host_queue_aql_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "host queue kernarg capacity must be at least 2x the AQL queue "
        "capacity because each staged kernarg block consumes one reserved "
        "AQL slot and wrap padding may skip one tail fragment (got "
        "kernarg_blocks=%u, aql_packets=%u)",
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
  hsa_agent_t host_agent = system->topology.cpu_agents[host_ordinal];

  // Zeroing allows for deinitialization to happen midway through initialization
  // if something fails.
  memset(out_physical_device, 0, sizeof(*out_physical_device));

  out_physical_device->device_agent = device_agent;
  out_physical_device->device_ordinal = device_ordinal;
  uint32_t host_numa_node = UINT32_MAX;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_agent_get_info(IREE_LIBHSA(libhsa), host_agent,
                                  HSA_AGENT_INFO_NODE, &host_numa_node));
  out_physical_device->host_numa_node = host_numa_node;
  out_physical_device->host_queue_capacity = options->host_queue_count;
  out_physical_device->host_queue_aql_capacity =
      options->host_queue_aql_capacity;
  out_physical_device->host_queue_notification_capacity =
      options->host_queue_notification_capacity;
  out_physical_device->host_queue_kernarg_capacity =
      options->host_queue_kernarg_capacity;

  // Initialize the per-device host block pool.
  // This should be pinned to the host NUMA node associated with the devices but
  // today we rely on the OS to migrate pages as needed.
  iree_arena_block_pool_initialize(options->host_block_pool_size,
                                   host_allocator,
                                   &out_physical_device->fine_host_block_pool);

  // Find the device memory pools and create block pools/allocators.
  hsa_amd_memory_pool_t coarse_block_memory_pool = {0};
  hsa_amd_memory_pool_t fine_block_memory_pool = {0};
  iree_status_t status = iree_hal_amdgpu_find_coarse_global_memory_pool(
      libhsa, device_agent, &coarse_block_memory_pool);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_find_fine_global_memory_pool(
        libhsa, device_agent, &fine_block_memory_pool);
  }
  if (iree_status_is_ok(status) && options->host_block_pool_initial_capacity) {
    status = iree_arena_block_pool_preallocate(
        &out_physical_device->fine_host_block_pool,
        options->host_block_pool_initial_capacity);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_block_pool_initialize(
        libhsa, options->device_block_pools.small, device_agent,
        coarse_block_memory_pool, host_allocator,
        &out_physical_device->coarse_block_pools.small);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_block_pool_initialize(
        libhsa, options->device_block_pools.large, device_agent,
        coarse_block_memory_pool, host_allocator,
        &out_physical_device->coarse_block_pools.large);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_block_pool_initialize(
        libhsa, options->device_block_pools.small, device_agent,
        fine_block_memory_pool, host_allocator,
        &out_physical_device->fine_block_pools.small);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_block_pool_initialize(
        libhsa, options->device_block_pools.large, device_agent,
        fine_block_memory_pool, host_allocator,
        &out_physical_device->fine_block_pools.large);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_block_allocator_initialize(
        &out_physical_device->coarse_block_pools.small,
        IREE_HAL_AMDGPU_PHYSICAL_DEVICE_COARSE_BLOCK_POOL_SMALL_PAGE_SIZE,
        &out_physical_device->coarse_block_allocators.small);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_block_allocator_initialize(
        &out_physical_device->coarse_block_pools.large,
        IREE_HAL_AMDGPU_PHYSICAL_DEVICE_COARSE_BLOCK_POOL_LARGE_PAGE_SIZE,
        &out_physical_device->coarse_block_allocators.large);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_block_allocator_initialize(
        &out_physical_device->fine_block_pools.small,
        IREE_HAL_AMDGPU_PHYSICAL_DEVICE_FINE_BLOCK_POOL_SMALL_PAGE_SIZE,
        &out_physical_device->fine_block_allocators.small);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_block_allocator_initialize(
        &out_physical_device->fine_block_pools.large,
        IREE_HAL_AMDGPU_PHYSICAL_DEVICE_FINE_BLOCK_POOL_LARGE_PAGE_SIZE,
        &out_physical_device->fine_block_allocators.large);
  }

  // Create the default queue-allocation slab provider over the device
  // fine-grained HSA pool and derive the TLSF policy used once topology
  // assignment provides an epoch query.
  iree_hal_queue_affinity_t queue_affinity_mask = 0;
  for (iree_host_size_t queue_ordinal = 0;
       queue_ordinal < options->host_queue_count; ++queue_ordinal) {
    const iree_host_size_t logical_queue_ordinal =
        device_ordinal * options->host_queue_count + queue_ordinal;
    iree_hal_queue_affinity_or_into(queue_affinity_mask,
                                    1ull << logical_queue_ordinal);
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_notification_create(
        proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE,
        &out_physical_device->default_pool_notification);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_slab_provider_create(
        logical_device, libhsa, &system->topology, fine_block_memory_pool,
        queue_affinity_mask, host_allocator,
        &out_physical_device->default_slab_provider);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_slab_provider_memory_pool_properties_t properties;
    status = iree_hal_amdgpu_slab_provider_query_memory_pool_properties(
        libhsa, fine_block_memory_pool, &properties);
    if (iree_status_is_ok(status) &&
        properties.allocation_alignment < options->default_pool.alignment) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "default pool alignment %" PRIu64
          " exceeds HSA memory pool allocation alignment %" PRIu64,
          (uint64_t)options->default_pool.alignment,
          (uint64_t)properties.allocation_alignment);
    }
    iree_device_size_t range_length = options->default_pool.range_length;
    if (iree_status_is_ok(status) &&
        !iree_device_size_checked_align(
            range_length, properties.allocation_granule, &range_length)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "default pool range_length %" PRIu64
          " overflows while aligning to HSA allocation granule %" PRIu64,
          (uint64_t)options->default_pool.range_length,
          (uint64_t)properties.allocation_granule);
    }
    if (iree_status_is_ok(status)) {
      out_physical_device->default_pool_options =
          (iree_hal_tlsf_pool_options_t){
              .tlsf_options =
                  {
                      .range_length = range_length,
                      .alignment = options->default_pool.alignment,
                      .frontier_capacity =
                          options->default_pool.frontier_capacity,
                  },
              .budget_limit = 0,
          };
    }
  }

  // Initialize the host signal pool.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_signal_pool_initialize(
        libhsa,
        /*initial_capacity=*/
        IREE_HAL_AMDGPU_HOST_SIGNAL_POOL_BATCH_SIZE_DEFAULT,
        /*batch_size=*/0, host_allocator,
        &out_physical_device->host_signal_pool);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_device_library_populate_agent_kernels(
        &system->device_library, device_agent,
        &out_physical_device->device_kernels);
  }
  uint32_t compute_unit_count = 0;
  uint32_t wavefront_size = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hsa_agent_get_info(
        IREE_LIBHSA(libhsa), device_agent,
        (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
        &compute_unit_count);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hsa_agent_get_info(IREE_LIBHSA(libhsa), device_agent,
                                HSA_AGENT_INFO_WAVEFRONT_SIZE, &wavefront_size);
  }
  // Validate launch metadata before passing it to the blit context. A broken
  // HSA bring-up that returns garbage here must fail loud with a clear message
  // rather than letting the blit path silently dispatch with wrong geometry.
  if (iree_status_is_ok(status) && compute_unit_count == 0) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "HSA reported 0 compute units for device agent "
                              "ordinal %" PRIhsz,
                              device_ordinal);
  }
  if (iree_status_is_ok(status) && wavefront_size != 32 &&
      wavefront_size != 64) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HSA reported unsupported wavefront size %u for device agent ordinal "
        "%" PRIhsz " (expected 32 or 64)",
        wavefront_size, device_ordinal);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_device_buffer_transfer_context_initialize(
        &out_physical_device->device_kernels, compute_unit_count,
        wavefront_size, &out_physical_device->buffer_transfer_context);
  }
  iree_hal_amdgpu_wait_barrier_strategy_t wait_barrier_strategy =
      IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_DEFER;
  iree_hal_amdgpu_gfxip_version_t gfxip_version = {0};
  if (iree_status_is_ok(status) && !options->force_wait_barrier_defer) {
    status = iree_hal_amdgpu_query_agent_gfxip_version(libhsa, device_agent,
                                                       &gfxip_version);
  }
  if (iree_status_is_ok(status) && !options->force_wait_barrier_defer) {
    wait_barrier_strategy =
        iree_hal_amdgpu_select_wait_barrier_strategy(gfxip_version);
  }
  out_physical_device->wait_barrier_strategy = wait_barrier_strategy;

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_physical_device_deinitialize(out_physical_device);
  }

  IREE_TRACE_ZONE_END(z0);
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
  iree_status_t status = iree_hal_tlsf_pool_create(
      physical_device->default_pool_options,
      physical_device->default_slab_provider,
      physical_device->default_pool_notification,
      (iree_hal_pool_epoch_query_t){
          .fn = iree_hal_amdgpu_physical_device_query_pool_epoch,
          .user_data = epoch_signal_table,
      },
      host_allocator, &physical_device->default_pool);
  for (iree_host_size_t queue_ordinal = 0;
       queue_ordinal < physical_device->host_queue_capacity &&
       iree_status_is_ok(status);
       ++queue_ordinal) {
    const iree_host_size_t logical_queue_ordinal =
        physical_device->device_ordinal * physical_device->host_queue_capacity +
        queue_ordinal;
    const iree_hal_queue_affinity_t queue_affinity =
        ((iree_hal_queue_affinity_t)1) << logical_queue_ordinal;
    iree_async_axis_t queue_axis = iree_async_axis_make_queue(
        session_epoch, machine_index, (uint8_t)physical_device->device_ordinal,
        (uint8_t)queue_ordinal);
    iree_thread_affinity_t completion_thread_affinity;
    iree_thread_affinity_set_group_any(physical_device->host_numa_node,
                                       &completion_thread_affinity);
    status = iree_hal_amdgpu_host_queue_initialize(
        libhsa, logical_device, proactor, physical_device->device_agent,
        host_memory_pools->coarse_pool, host_memory_pools->fine_pool,
        frontier_tracker, queue_axis, queue_affinity,
        completion_thread_affinity, physical_device->wait_barrier_strategy,
        epoch_signal_table, &physical_device->fine_host_block_pool,
        &physical_device->buffer_transfer_context,
        physical_device->default_pool, physical_device->device_ordinal,
        physical_device->host_queue_aql_capacity,
        physical_device->host_queue_notification_capacity,
        physical_device->host_queue_kernarg_capacity, host_allocator,
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
  for (iree_host_size_t i = 0; i < physical_device->host_queue_count; ++i) {
    iree_hal_amdgpu_host_queue_deinitialize(&physical_device->host_queues[i]);
  }
  physical_device->host_queue_count = 0;
  iree_hal_pool_release(physical_device->default_pool);
  physical_device->default_pool = NULL;
}

void iree_hal_amdgpu_physical_device_deinitialize(
    iree_hal_amdgpu_physical_device_t* physical_device) {
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_physical_device_deassign_frontier(physical_device);

  iree_hal_amdgpu_host_signal_pool_deinitialize(
      &physical_device->host_signal_pool);

  iree_hal_slab_provider_release(physical_device->default_slab_provider);
  iree_async_notification_release(physical_device->default_pool_notification);

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

  IREE_TRACE_ZONE_END(z0);
  return status;
}
