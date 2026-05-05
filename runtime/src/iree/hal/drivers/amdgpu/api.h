// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_API_H_
#define IREE_HAL_DRIVERS_AMDGPU_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

// Exported for API interop:
#include "iree/hal/drivers/amdgpu/util/libhsa.h"    // IWYU pragma: export
#include "iree/hal/drivers/amdgpu/util/topology.h"  // IWYU pragma: export

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_logical_device_t
//===----------------------------------------------------------------------===//

// Controls where the queue operates.
typedef enum iree_hal_amdgpu_queue_placement_e {
  // Automatically select the best supported placement. Today this selects the
  // host queue path because device-side queue scheduling is not implemented.
  IREE_HAL_AMDGPU_QUEUE_PLACEMENT_ANY = 0,
  // Queue executes entirely on the host via iree_hal_amdgpu_host_queue_t.
  // This introduces additional latency on all queue operations but can operate
  // on systems without host/device atomics (PCIe atomics, xGMI, etc). It is
  // also useful for debugging.
  IREE_HAL_AMDGPU_QUEUE_PLACEMENT_HOST,
  // Queue executes entirely on the device. Not implemented; requests for this
  // explicit placement fail during device option verification.
  IREE_HAL_AMDGPU_QUEUE_PLACEMENT_DEVICE,
} iree_hal_amdgpu_queue_placement_t;

// Parameters configuring an iree_hal_amdgpu_logical_device_t.
// Must be initialized with iree_hal_amdgpu_logical_device_options_initialize
// prior to use.
typedef struct iree_hal_amdgpu_logical_device_options_t {
  // Size of a block in each host block pool.
  struct {
    // Small host block pool options.
    struct {
      // Size in bytes of a small host block. Must be a power of two.
      iree_host_size_t block_size;
    } small;
    // Large host block pool options.
    struct {
      // Size in bytes of a large host block. Must be a power of two.
      iree_host_size_t block_size;
    } large;
    // Command-buffer host block pool options.
    struct {
      // Usable byte capacity of a command-buffer recording block. Must be a
      // power of two.
      iree_host_size_t usable_block_size;
    } command_buffer;
  } host_block_pools;

  // Size of a block in each device block pool.
  struct {
    struct {
      // Size in bytes of a small device block. Must be a power of two.
      iree_device_size_t block_size;
      // Initial small block pool block allocation count.
      iree_host_size_t initial_capacity;
    } small;
    struct {
      // Size in bytes of a large device block. Must be a power of two.
      iree_device_size_t block_size;
      // Initial large block pool block allocation count.
      iree_host_size_t initial_capacity;
    } large;
  } device_block_pools;

  // Default queue-allocation pool policy.
  struct {
    // Logical byte length of the default TLSF pool range per physical device.
    iree_device_size_t range_length;

    // Minimum byte alignment for every default-pool reservation.
    iree_device_size_t alignment;

    // Maximum death-frontier entry count stored per free TLSF block.
    uint8_t frontier_capacity;
  } default_pool;

  // Controls where queues are placed. ANY and HOST currently select host
  // queues. DEVICE is reserved for future device-side scheduling and fails
  // loudly until that path is implemented.
  iree_hal_amdgpu_queue_placement_t queue_placement;

  // Per-physical-device host queue policy.
  struct {
    // HSA AQL ring capacity in packets for each host queue. Must be a power of
    // two. Larger rings allow more in-flight packet work before submitters see
    // AQL backpressure.
    uint32_t aql_capacity;
    // Completion/reclaim ring capacity for each host queue. Must be a power of
    // two. This bounds in-flight host-visible completion epochs before replay
    // must park and resume after drain.
    uint32_t notification_capacity;
    // Kernarg ring capacity in 64-byte blocks for each host queue. Must be a
    // power of two and at least 2x |aql_capacity| to cover one tail-padding
    // gap at wrap. Submission admission checks kernarg and AQL capacity
    // together before publishing packets.
    uint32_t kernarg_capacity;
    // Device-visible control upload ring capacity in bytes for each host queue.
    // Zero disables the optional upload ring; non-zero values must be powers of
    // two. This carries queue-ordered metadata such as device-side
    // command-buffer fixup inputs without using the file staging pool.
    uint32_t upload_capacity;
  } host_queues;

  // Preallocates a reasonable number of resources in pools to reduce initial
  // execution latency.
  uint64_t preallocate_pools : 1;

  // Reserved for a future exclusive queue scheduling mode. Unsupported today;
  // enabling it fails option verification.
  uint64_t exclusive_execution : 1;

  // Forces cross-queue wait barriers to use software deferral instead of the
  // device-side strategy selected from the GPU ISA. Useful for testing the
  // conservative host-only fallback path.
  uint64_t force_wait_barrier_defer : 1;

  // Reserved for future HSA active-wait tuning. Must be zero today because no
  // wait path consumes it yet.
  iree_duration_t wait_active_for_ns;
} iree_hal_amdgpu_logical_device_options_t;

// Initializes |out_options| to default values.
IREE_API_EXPORT void iree_hal_amdgpu_logical_device_options_initialize(
    iree_hal_amdgpu_logical_device_options_t* out_options);

// Parses |params| and updates |options|. No AMDGPU logical-device string
// parameters are currently supported; nonempty lists fail loudly instead of
// being ignored.
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_logical_device_options_parse(
    iree_hal_amdgpu_logical_device_options_t* options,
    iree_string_pair_list_t params);

// Creates a AMDGPU HAL device with the given |options| and |topology|.
//
// The provided |identifier| will be used by programs to distinguish the device
// type from other HAL implementations. If compiling programs with the IREE
// compiler this must match the value used by `IREE::HAL::TargetDevice`.
//
// |options|, |libhsa|, and |topology| will be cloned into the device and need
// not live beyond the call.
//
// |out_device| must be released by the caller (see iree_hal_device_release).
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_logical_device_create(
    iree_string_view_t identifier,
    const iree_hal_amdgpu_logical_device_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_driver_t
//===----------------------------------------------------------------------===//

// Parameters for configuring an iree_hal_amdgpu_driver_t.
// Must be initialized with iree_hal_amdgpu_driver_options_initialize prior to
// use.
typedef struct iree_hal_amdgpu_driver_options_t {
  // Search paths (directories or files) for finding the HSA runtime shared
  // library. Driver creation clones these strings; callers only need to keep
  // them live until iree_hal_amdgpu_driver_create returns.
  iree_string_view_list_t libhsa_search_paths;

  // Default device options when none are provided during device creation.
  iree_hal_amdgpu_logical_device_options_t default_device_options;
} iree_hal_amdgpu_driver_options_t;

// Initializes the given |out_options| with default driver creation options.
IREE_API_EXPORT void iree_hal_amdgpu_driver_options_initialize(
    iree_hal_amdgpu_driver_options_t* out_options);

// Parses |params| and updates |options|. No AMDGPU driver string parameters are
// currently supported; nonempty lists fail loudly instead of being ignored.
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_driver_options_parse(
    iree_hal_amdgpu_driver_options_t* options, iree_string_pair_list_t params);

// Creates a AMDGPU HAL driver with the given |options|, from which AMDGPU
// devices can be enumerated and created with specific parameters.
//
// The provided |identifier| will be used by programs to distinguish the device
// type from other HAL implementations. If compiling programs with the IREE
// compiler this must match the value used by IREE::HAL::TargetDevice.
//
// |out_driver| must be released by the caller (see iree_hal_driver_release).
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_driver_create(
    iree_string_view_t identifier,
    const iree_hal_amdgpu_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_API_H_
