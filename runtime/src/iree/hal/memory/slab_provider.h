// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_SLAB_PROVIDER_H_
#define IREE_HAL_MEMORY_SLAB_PROVIDER_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_slab_provider_t iree_hal_slab_provider_t;
typedef struct iree_hal_slab_provider_vtable_t iree_hal_slab_provider_vtable_t;

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// Flags controlling trim behavior.
typedef uint32_t iree_hal_slab_provider_trim_flags_t;
enum iree_hal_slab_provider_trim_flag_bits_e {
  IREE_HAL_SLAB_PROVIDER_TRIM_FLAG_NONE = 0u,
  // Release all cached/unused resources regardless of retention policy.
  IREE_HAL_SLAB_PROVIDER_TRIM_FLAG_ALL = 1u << 0,
  // Release only resources above the target retention level.
  IREE_HAL_SLAB_PROVIDER_TRIM_FLAG_EXCESS = 1u << 1,
};

// Running statistics for a slab provider (and any inner providers in its
// chain). Stats accumulate from the innermost provider outward, with each
// layer adding its own contributions.
typedef struct iree_hal_slab_provider_stats_t {
  // Cumulative slabs acquired from the platform/driver.
  uint64_t total_acquired;
  // Cumulative slabs released to the platform/driver.
  uint64_t total_released;

  // Cache-layer statistics. Zero for non-caching providers.
  struct {
    // Number of slabs currently in the cache freelist.
    uint32_t count;
    // Acquires satisfied from the freelist (fast path).
    uint64_t hit_count;
    // Acquires that required waiting or fell through to the inner provider.
    uint64_t miss_count;
    // Current EMA of reuse interval in nanoseconds. Tracks how quickly
    // cached slabs are recycled. Short intervals indicate bursty reuse.
    uint64_t ema_reuse_interval_nanoseconds;
    // Cumulative time spent in prefault callbacks (nanoseconds).
    uint64_t prefault_time_nanoseconds;
  } cache;
} iree_hal_slab_provider_stats_t;

// Maximum number of providers tracked in a visited set. If a provider chain
// is deeper than this, stats queries bail at the top level.
#define IREE_HAL_SLAB_PROVIDER_MAX_VISITED 32

// Set of already-visited providers for deduplicating stats queries across
// shared provider chains. Multiple pools may share the same provider (or
// provider chain); the visited set prevents double-counting.
//
// Stack-allocate before a stats query:
//   iree_hal_slab_provider_visited_set_t visited = {0};
typedef struct iree_hal_slab_provider_visited_set_t {
  iree_host_size_t count;
  const iree_hal_slab_provider_t* providers[IREE_HAL_SLAB_PROVIDER_MAX_VISITED];
} iree_hal_slab_provider_visited_set_t;

//===----------------------------------------------------------------------===//
// iree_hal_slab_t
//===----------------------------------------------------------------------===//

// A contiguous region of physical memory acquired from a slab provider.
// Slabs are the backing store for pool offset allocators: the pool manages
// offsets within [0, length), and the slab provider materializes buffers that
// reference physical memory at those offsets.
//
// |base_ptr| and |provider_handle| are provider-owned payloads, not generic
// pool data. Host providers may store a host pointer in |base_ptr|. Device
// providers may store a device VA token, leave |base_ptr| NULL, and keep the
// real driver handle in |provider_handle|. Generic pools must not dereference
// |base_ptr| or inspect |provider_handle| directly.
//
// To expose a reservation as a HAL buffer, pools pass the slab plus an offset
// range to iree_hal_slab_provider_wrap_buffer(), which creates a
// provider-specific buffer referencing the physical memory.
typedef struct iree_hal_slab_t {
  // Base address of the slab's memory region.
  uint8_t* base_ptr;

  // Length of the slab in bytes.
  iree_device_size_t length;

  // Provider-specific opaque handle used to release the slab. The provider
  // interprets this however it needs — a pointer to internal state, an
  // allocation handle, a VMM mapping descriptor, etc.
  uint64_t provider_handle;
} iree_hal_slab_t;

//===----------------------------------------------------------------------===//
// iree_hal_slab_provider_t
//===----------------------------------------------------------------------===//

// Acquires and releases large chunks of physical memory from the platform or
// GPU driver. This is the only component that makes driver API calls for
// memory allocation — everything above it (offset allocators, pools) is pure
// host-side bookkeeping.
//
// Slab providers are ref-counted. Pools retain their slab provider at creation
// time and release it on destruction, ensuring the provider outlives all pools
// that use it. Slab providers are typically created by the device at init time
// and shared across pools on the same device. Different memory classes
// (DEVICE_LOCAL, HOST_VISIBLE, HOST_CACHED) require separate slab providers
// because they draw from different physical memory.
//
// Concrete implementations embed this struct at offset 0.
struct iree_hal_slab_provider_t {
  iree_atomic_ref_count_t ref_count;
  const iree_hal_slab_provider_vtable_t* vtable;
};

// Initializes the base slab provider fields. Called by concrete
// implementations during their create function.
void iree_hal_slab_provider_initialize(
    const iree_hal_slab_provider_vtable_t* vtable,
    iree_hal_slab_provider_t* provider);

// Retains a reference to the slab provider.
void iree_hal_slab_provider_retain(iree_hal_slab_provider_t* provider);

// Releases a reference. Destroys the provider when the count reaches zero.
void iree_hal_slab_provider_release(iree_hal_slab_provider_t* provider);

// Acquires a slab of at least |min_length| bytes from the provider.
iree_status_t iree_hal_slab_provider_acquire_slab(
    iree_hal_slab_provider_t* provider, iree_device_size_t min_length,
    iree_hal_slab_t* out_slab);

// Releases a previously acquired slab back to the provider.
void iree_hal_slab_provider_release_slab(iree_hal_slab_provider_t* provider,
                                         const iree_hal_slab_t* slab);

// Wraps a byte range within |slab| as a HAL buffer using the provider-specific
// buffer implementation for that slab's memory.
//
// |slab_offset| and |allocation_size| define the byte range relative to the
// slab's base. The range must lie fully inside [0, slab->length). |params| is
// canonicalized before dispatch.
iree_status_t iree_hal_slab_provider_wrap_buffer(
    iree_hal_slab_provider_t* provider, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_buffer_params_t params,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer);

// Prepares a slab for use (page faulting, NUMA pinning, etc.).
void iree_hal_slab_provider_prefault(iree_hal_slab_provider_t* provider,
                                     iree_hal_slab_t* slab);

// Releases unused cached resources. Passes |flags| through to the provider
// and any inner providers in the chain.
void iree_hal_slab_provider_trim(iree_hal_slab_provider_t* provider,
                                 iree_hal_slab_provider_trim_flags_t flags);

// Accumulates statistics from the provider (and any inner providers).
// |visited| prevents double-counting across shared provider chains.
void iree_hal_slab_provider_query_stats(
    const iree_hal_slab_provider_t* provider,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats);

// Queries the memory properties of slabs from this provider.
void iree_hal_slab_provider_query_properties(
    const iree_hal_slab_provider_t* provider,
    iree_hal_memory_type_t* out_memory_type,
    iree_hal_buffer_usage_t* out_supported_usage);

// Returns true if |provider| has already been visited (should be skipped).
// If not visited, adds it to the set and returns false. Returns true without
// adding if the set is full (conservative: skip rather than double-count).
bool iree_hal_slab_provider_visited(
    iree_hal_slab_provider_visited_set_t* visited,
    const iree_hal_slab_provider_t* provider);

//===----------------------------------------------------------------------===//
// iree_hal_slab_provider_t vtable
//===----------------------------------------------------------------------===//

struct iree_hal_slab_provider_vtable_t {
  void (*destroy)(iree_hal_slab_provider_t* provider);

  // Acquires a slab of at least |min_length| bytes. The returned slab may
  // be larger than requested. The caller must release the slab via
  // release_slab when done.
  iree_status_t (*acquire_slab)(iree_hal_slab_provider_t* provider,
                                iree_device_size_t min_length,
                                iree_hal_slab_t* out_slab);

  // Releases a previously acquired slab back to the platform.
  void (*release_slab)(iree_hal_slab_provider_t* provider,
                       const iree_hal_slab_t* slab);

  // Wraps [slab_offset, slab_offset + allocation_size) in |slab| as a HAL
  // buffer whose backing and concrete buffer type are owned by this provider.
  //
  // Generic pools must use this hook instead of interpreting slab payload
  // fields directly. For CPU providers this typically wraps a host pointer in
  // iree_hal_heap_buffer_t; GPU or remote providers can create device-specific
  // buffer objects that retain opaque driver state from |slab|.
  //
  // The returned buffer invokes |release_callback| when destroyed. The pool
  // uses that callback to release its reservation bookkeeping; the provider
  // remains responsible for releasing whole slabs via release_slab().
  // If |release_callback.fn| is NULL, the returned buffer is still a borrowed
  // slab view and must not release the slab or underlying physical storage.
  iree_status_t (*wrap_buffer)(
      iree_hal_slab_provider_t* provider, const iree_hal_slab_t* slab,
      iree_device_size_t slab_offset, iree_device_size_t allocation_size,
      iree_hal_buffer_params_t params,
      iree_hal_buffer_release_callback_t release_callback,
      iree_hal_buffer_t** out_buffer);

  // Prepares a slab for use after acquisition. Called by the slab cache's
  // background thread after acquire_slab() succeeds and before the slab is
  // placed on the ready freelist.
  //
  // Provider-specific preparation:
  //   CPU (Linux): madvise(MADV_POPULATE_WRITE) to force page allocation
  //     and zeroing, eliminating lazy zero-fill page faults on first write.
  //   CPU (Windows): PrefetchVirtualMemory or page-strided writes.
  //   CPU (any): NUMA pinning via mbind + first-touch policy.
  //   CUDA: cudaMemAdvise for managed memory migration to device.
  //
  // Providers where acquire_slab already commits all pages implement this
  // as an empty function.
  void (*prefault)(iree_hal_slab_provider_t* provider, iree_hal_slab_t* slab);

  // Releases unused cached resources. Caching providers release freelisted
  // slabs back to their inner provider. Non-caching providers do nothing.
  // |flags| controls which resources are released (ALL vs EXCESS).
  void (*trim)(iree_hal_slab_provider_t* provider,
               iree_hal_slab_provider_trim_flags_t flags);

  // Accumulates this provider's statistics into |out_stats|. Wrapping
  // providers call into their inner provider first, then add their own
  // contributions. |visited| prevents double-counting when multiple pools
  // share a provider chain — providers already in the set return immediately.
  void (*query_stats)(const iree_hal_slab_provider_t* provider,
                      iree_hal_slab_provider_visited_set_t* visited,
                      iree_hal_slab_provider_stats_t* out_stats);

  // Queries the memory properties of slabs from this provider.
  void (*query_properties)(const iree_hal_slab_provider_t* provider,
                           iree_hal_memory_type_t* out_memory_type,
                           iree_hal_buffer_usage_t* out_supported_usage);
};

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_MEMORY_SLAB_PROVIDER_H_
