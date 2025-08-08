// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "experimental/streaming/internal.h"

//===----------------------------------------------------------------------===//
// Device management
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_device_count(iree_host_size_t* out_count) {
  IREE_ASSERT_ARGUMENT(out_count);
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }

  *out_count = device_registry->device_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_device_by_ordinal(
    iree_hal_streaming_device_ordinal_t ordinal,
    iree_hal_streaming_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }

  if (ordinal >= device_registry->device_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device ordinal %zu out of range [0, %zu)", ordinal,
                            device_registry->device_count);
  }

  iree_hal_streaming_device_t* device = &device_registry->devices[ordinal];

  // Device is always created during initialization.
  // Primary context is created lazily on first access.
  IREE_ASSERT(device->hal_device);

  *out_device = device;

  return iree_ok_status();
}

iree_status_t iree_hal_streaming_device_name(
    iree_hal_streaming_device_ordinal_t ordinal, char* name,
    iree_host_size_t name_size) {
  IREE_ASSERT_ARGUMENT(name);
  if (name_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "name_size must be > 0");
  }

  iree_hal_streaming_device_t* device = NULL;
  iree_status_t status = iree_hal_streaming_device_by_ordinal(ordinal, &device);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  // Calculate safe copy length: min(source_length, dest_size - 1)
  const iree_host_size_t source_len = device->info.name.size;
  const iree_host_size_t copy_len =
      source_len < (name_size - 1) ? source_len : (name_size - 1);

  // Copy the name data safely
  if (copy_len > 0) {
    memcpy(name, device->info.name.data, copy_len);
  }

  // Always null-terminate
  name[copy_len] = '\0';

  return iree_ok_status();
}

iree_hal_streaming_p2p_link_t* iree_hal_streaming_device_lookup_p2p_link(
    iree_hal_streaming_device_ordinal_t src_device,
    iree_hal_streaming_device_ordinal_t dst_device) {
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry || !device_registry->p2p_topology) {
    return NULL;
  }

  const iree_host_size_t device_count = device_registry->device_count;
  if (src_device >= device_count || dst_device >= device_count) {
    return NULL;
  }

  // Links are stored in row-major order: [src][dst].
  const iree_host_size_t link_index = src_device * device_count + dst_device;
  return &device_registry->p2p_topology[link_index];
}

iree_status_t iree_hal_streaming_device_memory_info(
    iree_hal_streaming_device_ordinal_t ordinal,
    iree_device_size_t* out_free_memory, iree_device_size_t* out_total_memory) {
  IREE_ASSERT_ARGUMENT(out_free_memory);
  IREE_ASSERT_ARGUMENT(out_total_memory);
  *out_free_memory = 0;
  *out_total_memory = 0;

  iree_hal_streaming_device_t* device = NULL;
  iree_status_t status = iree_hal_streaming_device_by_ordinal(ordinal, &device);
  if (iree_status_is_ok(status)) {
    *out_free_memory = device->free_memory;
    *out_total_memory = device->total_memory;
  }
  return status;
}

iree_status_t iree_hal_streaming_device_can_access_peer(
    iree_hal_streaming_device_ordinal_t device_ordinal,
    iree_hal_streaming_device_ordinal_t peer_device_ordinal, bool* can_access) {
  IREE_ASSERT_ARGUMENT(can_access);
  IREE_TRACE_ZONE_BEGIN(z0);
  *can_access = false;

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                             "HAL stream layer not initialized"));
  }

  const iree_host_size_t device_count = device_registry->device_count;
  if (device_ordinal >= device_count || peer_device_ordinal >= device_count) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "device ordinals out of range"));
  }

  // Look up P2P link in topology.
  iree_hal_streaming_p2p_link_t* link =
      iree_hal_streaming_device_lookup_p2p_link(device_ordinal,
                                                peer_device_ordinal);
  if (!link) {
    *can_access = true;
  } else {
    *can_access = link->access_supported ? true : false;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_device_set_primary_context_flags(
    iree_hal_streaming_device_ordinal_t device_ordinal,
    const iree_hal_streaming_context_flags_t* flags) {
  IREE_ASSERT_ARGUMENT(flags);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_device_t* device =
      iree_hal_streaming_device_entry(device_ordinal);
  if (!device) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "invalid device ordinal %" PRIhsz, device_ordinal));
  }

  // Update the primary context flags.
  device->primary_context_flags = *flags;

  // If the primary context exists, we might need to update it.
  // For now, we just store the flags for new contexts.

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_device_primary_context_state(
    iree_hal_streaming_device_ordinal_t device_ordinal,
    iree_hal_streaming_context_flags_t* out_flags, bool* out_active) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_device_t* device =
      iree_hal_streaming_device_entry(device_ordinal);
  if (!device) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "invalid device ordinal %" PRIhsz, device_ordinal));
  }

  if (out_flags) {
    *out_flags = device->primary_context_flags;
  }

  // Context is active if reference count > 0.
  if (out_active) {
    iree_slim_mutex_lock(&device->primary_context_mutex);
    *out_active = (device->primary_context_ref_count > 0);
    iree_slim_mutex_unlock(&device->primary_context_mutex);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_device_get_or_create_primary_context(
    iree_hal_streaming_device_t* device,
    iree_hal_streaming_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_context);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Fast path: context already exists.
  if (device->primary_context) {
    *out_context = device->primary_context;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Slow path: need to create the context (with mutex protection).
  iree_slim_mutex_lock(&device->primary_context_mutex);

  // Double-check inside the lock - another thread may have created it.
  if (device->primary_context) {
    *out_context = device->primary_context;
    iree_slim_mutex_unlock(&device->primary_context_mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Create the primary context.
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    iree_slim_mutex_unlock(&device->primary_context_mutex);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                             "device registry not initialized"));
  }

  iree_status_t status = iree_hal_streaming_context_create(
      device, device->primary_context_flags, device_registry->host_allocator,
      &device->primary_context);

  // Create default memory pool for this device if context was created
  // successfully.
  if (iree_status_is_ok(status)) {
    // Get device ordinal from registry.
    iree_host_size_t device_ordinal = device - device_registry->devices;

    iree_hal_streaming_mem_pool_props_t props = {
        .alloc_handle_type = IREE_HAL_STREAMING_MEM_HANDLE_TYPE_NONE,
        .location_type = IREE_HAL_STREAMING_MEM_LOCATION_TYPE_DEVICE,
        .location_id = device_ordinal,
    };

    status = iree_hal_streaming_mem_pool_create(device->primary_context, &props,
                                                device_registry->host_allocator,
                                                &device->default_mem_pool);
  }

  if (iree_status_is_ok(status)) {
    *out_context = device->primary_context;
  }

  iree_slim_mutex_unlock(&device->primary_context_mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_device_retain_primary_context(
    iree_hal_streaming_device_t* device,
    iree_hal_streaming_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_context);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&device->primary_context_mutex);

  // Increment reference count.
  device->primary_context_ref_count++;

  // If this is the first retain (count went from 0 to 1), create the context.
  if (device->primary_context_ref_count == 1 && !device->primary_context) {
    iree_hal_streaming_device_registry_t* device_registry =
        iree_hal_streaming_device_registry();
    if (!device_registry) {
      device->primary_context_ref_count--;
      iree_slim_mutex_unlock(&device->primary_context_mutex);
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                               "device registry not initialized"));
    }

    iree_status_t status = iree_hal_streaming_context_create(
        device, device->primary_context_flags, device_registry->host_allocator,
        &device->primary_context);

    // Create default memory pool if context was created successfully.
    if (iree_status_is_ok(status) && !device->default_mem_pool) {
      // Get device ordinal from registry.
      iree_host_size_t device_ordinal = device - device_registry->devices;

      iree_hal_streaming_mem_pool_props_t props = {
          .alloc_handle_type = IREE_HAL_STREAMING_MEM_HANDLE_TYPE_NONE,
          .location_type = IREE_HAL_STREAMING_MEM_LOCATION_TYPE_DEVICE,
          .location_id = device_ordinal,
      };

      status = iree_hal_streaming_mem_pool_create(
          device->primary_context, &props, device_registry->host_allocator,
          &device->default_mem_pool);

      if (iree_status_is_ok(status)) {
        // Set current pool to default pool.
        device->current_mem_pool = device->default_mem_pool;
        iree_hal_streaming_mem_pool_retain(device->current_mem_pool);
      }
    }

    if (!iree_status_is_ok(status)) {
      // Creation failed - decrement ref count back to 0.
      device->primary_context_ref_count--;
      if (device->primary_context) {
        iree_hal_streaming_context_release(device->primary_context);
        device->primary_context = NULL;
      }
      iree_slim_mutex_unlock(&device->primary_context_mutex);
      IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
    }
  }

  // Retain the context for the caller.
  iree_hal_streaming_context_retain(device->primary_context);
  *out_context = device->primary_context;

  iree_slim_mutex_unlock(&device->primary_context_mutex);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_device_release_primary_context(
    iree_hal_streaming_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&device->primary_context_mutex);

  // Check if context is retained.
  if (device->primary_context_ref_count == 0) {
    iree_slim_mutex_unlock(&device->primary_context_mutex);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "primary context not retained"));
  }

  // Decrement reference count.
  device->primary_context_ref_count--;

  // If count reached 0, destroy the context.
  if (device->primary_context_ref_count == 0 && device->primary_context) {
    // Wait for all operations to complete.
    iree_status_t status = iree_hal_streaming_context_wait_idle(
        device->primary_context, iree_infinite_timeout());
    if (!iree_status_is_ok(status)) {
      iree_status_free(status);
    }

    // Release the context.
    iree_hal_streaming_context_release(device->primary_context);
    device->primary_context = NULL;

    // Also clear memory pools.
    if (device->default_mem_pool) {
      iree_hal_streaming_mem_pool_release(device->default_mem_pool);
      device->default_mem_pool = NULL;
    }
    if (device->current_mem_pool) {
      iree_hal_streaming_mem_pool_release(device->current_mem_pool);
      device->current_mem_pool = NULL;
    }

    // Clear current context if it was the primary context.
    iree_hal_streaming_context_t* current_context =
        iree_hal_streaming_context_current();
    if (current_context && current_context == device->primary_context) {
      iree_hal_streaming_context_set_current(NULL);
    }
  }

  iree_slim_mutex_unlock(&device->primary_context_mutex);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Occupancy calculation helpers
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_calculate_max_active_blocks_per_multiprocessor(
    iree_hal_streaming_device_t* device, iree_hal_streaming_symbol_t* symbol,
    uint32_t block_size, uint32_t dynamic_shared_mem_size,
    uint32_t* out_max_blocks) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(out_max_blocks);

  // Verify the symbol is a function.
  if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol is not a function (type=%d)", symbol->type);
  }

  if (block_size <= 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "block size must be positive");
  }

  // Calculate constraints.
  // 1. Thread constraint: blocks limited by max threads per SM.
  const int blocks_by_threads =
      device->max_threads_per_multiprocessor / block_size;

  // 2. Block constraint: hardware limit on blocks per SM.
  const int blocks_by_limit = device->max_blocks_per_multiprocessor;

  // 3. Register constraint: blocks limited by register usage.
  uint32_t blocks_by_regs = 1000000;  // Large number as default.
  if (symbol->num_regs > 0) {
    // Round up register allocation to warp granularity.
    const int warps_per_block =
        (block_size + device->warp_size - 1) / device->warp_size;
    const int regs_per_block =
        symbol->num_regs * warps_per_block * device->warp_size;
    if (regs_per_block > 0) {
      blocks_by_regs =
          device->max_registers_per_multiprocessor / regs_per_block;
    }
  }

  // 4. Shared memory constraint.
  uint32_t blocks_by_smem = 1000000;  // Large number as default.
  const uint32_t total_smem =
      symbol->shared_size_bytes + dynamic_shared_mem_size;
  if (total_smem > 0) {
    blocks_by_smem = device->max_shared_memory_per_multiprocessor / total_smem;
  }

  // Take the minimum of all constraints.
  uint32_t max_blocks = blocks_by_threads;
  if (blocks_by_limit < max_blocks) max_blocks = blocks_by_limit;
  if (blocks_by_regs < max_blocks) max_blocks = blocks_by_regs;
  if (blocks_by_smem < max_blocks) max_blocks = blocks_by_smem;

  // Ensure at least 0 blocks.
  if (max_blocks < 0) max_blocks = 0;

  *out_max_blocks = max_blocks;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_calculate_optimal_block_size(
    iree_hal_streaming_device_t* device, iree_hal_streaming_symbol_t* symbol,
    uint32_t dynamic_shared_mem_size,
    iree_hal_streaming_block_to_dynamic_smem_fn_t dynamic_shared_mem_callback,
    uint32_t block_size_limit, uint32_t* out_block_size,
    uint32_t* out_min_grid_size) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(out_block_size);
  IREE_ASSERT_ARGUMENT(out_min_grid_size);

  // Verify the symbol is a function.
  if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol is not a function (type=%d)", symbol->type);
  }

  // Determine the maximum block size.
  uint32_t max_block_size = symbol->max_threads_per_block;
  if (block_size_limit > 0 && block_size_limit < max_block_size) {
    max_block_size = block_size_limit;
  }
  if (max_block_size > 1024) max_block_size = 1024;  // Hardware limit.

  // Try different block sizes and find the one with best occupancy.
  uint32_t best_block_size = 32;
  uint32_t best_occupancy = 0;

  // Test common block sizes: 32, 64, 128, 256, 512, 768, 1024.
  const uint32_t block_sizes[] = {32, 64, 128, 256, 512, 768, 1024};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(block_sizes); ++i) {
    const uint32_t test_size = block_sizes[i];
    if (test_size > max_block_size) break;

    // Calculate dynamic shared memory size for this block size.
    const uint32_t dynamic_smem = dynamic_shared_mem_callback
                                      ? dynamic_shared_mem_callback(test_size)
                                      : dynamic_shared_mem_size;

    // Get max active blocks for this configuration.
    int active_blocks = 0;
    iree_status_t status =
        iree_hal_streaming_calculate_max_active_blocks_per_multiprocessor(
            device, symbol, test_size, dynamic_smem, &active_blocks);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      continue;
    }

    // Calculate occupancy (active warps).
    const uint32_t occupancy = active_blocks * test_size;

    // Update best if this is better.
    if (occupancy > best_occupancy) {
      best_occupancy = occupancy;
      best_block_size = test_size;
    }
  }

  // Calculate grid size with the best block size.
  const uint32_t mp_count =
      device->multiprocessor_count > 0 ? device->multiprocessor_count : 1;

  // Get dynamic shared memory for the best block size.
  const uint32_t best_dynamic_smem =
      dynamic_shared_mem_callback ? dynamic_shared_mem_callback(best_block_size)
                                  : 0;

  uint32_t blocks_per_mp = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_calculate_max_active_blocks_per_multiprocessor(
          device, symbol, best_block_size, best_dynamic_smem, &blocks_per_mp));

  *out_block_size = best_block_size;
  *out_min_grid_size = blocks_per_mp * mp_count;

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Cooperative launch calculation helpers
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_calculate_max_cooperative_blocks(
    iree_hal_streaming_device_t* device, iree_hal_streaming_symbol_t* symbol,
    uint32_t block_size, uint32_t dynamic_shared_mem_size,
    uint32_t* out_max_blocks) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(out_max_blocks);

  // Check if device supports cooperative launch.
  // If not, return success with max blocks set to 0.
  if (!device->supports_cooperative_launch) {
    *out_max_blocks = 0;
    return iree_ok_status();
  }

  // For cooperative kernels, all blocks must be resident on the device at once.
  // Calculate the maximum number of active blocks per multiprocessor.
  uint32_t max_blocks_per_sm = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_calculate_max_active_blocks_per_multiprocessor(
          device, symbol, block_size, dynamic_shared_mem_size,
          &max_blocks_per_sm));

  // Total max blocks is limited by the number of SMs on the device.
  *out_max_blocks = max_blocks_per_sm * device->multiprocessor_count;

  return iree_ok_status();
}
