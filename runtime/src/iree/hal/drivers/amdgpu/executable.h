// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/abi/kernel_args.h"
#include "iree/hal/drivers/amdgpu/device/dispatch.h"
#include "iree/hal/drivers/amdgpu/profile_metadata.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// ISA Support
//===----------------------------------------------------------------------===//

// Returns success if all GPU agents in the topology support the same ISA.
// Several places in the code assume homogeneous devices and will need to change
// in heterogeneous cases where some executables can only be used on a subset of
// GPUs. This wouldn't be terrible to support but devices with different
// attributes are expected to be their own unique HAL devices.
iree_status_t iree_hal_amdgpu_verify_device_isa_commonality(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology);

// Returns whether the IREE HAL executable |format| is supported by all GPU
// devices in |topology|. Some devices may support multiple ISAs.
//
// Supports AMDGPU target IDs in both compiler spelling (`gfx1100`,
// `gfx942:xnack-`) and the canonical ISA names reported by HSA
// (`amdgcn-amd-amdhsa--gfx1100`). Matching uses structured target-ID
// compatibility so generic code-object targets and explicit feature modes can
// be checked without relying on string equality.
//
// Optionally |out_isa| can be used to get the agent ISA for the given format.
// Note that this will be from the first device but should match all other
// devices in the topology.
iree_status_t iree_hal_amdgpu_executable_format_supported(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t device_agent,
    iree_string_view_t format, bool* out_supported, hsa_isa_t* out_isa);

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_executable_t
//===----------------------------------------------------------------------===//

// The maximum number of per-dispatch bindings allowed.
// This is limited by the field size in iree_hal_amdgpu_device_kernel_args_t.
#define IREE_HAL_AMDGPU_MAX_DISPATCH_BINDING_COUNT UINT16_MAX

// The maximum number of per-dispatch constants allowed.
// This is limited by the field size in iree_hal_amdgpu_device_kernel_args_t.
#define IREE_HAL_AMDGPU_MAX_DISPATCH_CONSTANT_COUNT UINT16_MAX

// Host-resident dispatch metadata precomputed for one executable export on one
// physical device.
//
// Descriptors are immutable after executable creation and remain valid for the
// lifetime of the executable. They intentionally duplicate the device-visible
// kernel argument table in ordinary host memory so queue submission does not
// read per-dispatch metadata from memory allocated for GPU visibility.
typedef struct iree_hal_amdgpu_executable_dispatch_descriptor_t {
  // Device-specific kernel arguments with a valid kernel_object for dispatch.
  iree_hal_amdgpu_device_kernel_args_t kernel_args;
  // HAL ABI kernarg layout derived from |kernel_args|.
  iree_hal_amdgpu_device_dispatch_kernarg_layout_t hal_kernarg_layout;
  // Custom direct-argument kernarg layout derived from |kernel_args|.
  iree_hal_amdgpu_device_dispatch_kernarg_layout_t custom_kernarg_layout;
  // Queue kernarg-ring block count for HAL ABI dispatches.
  uint32_t hal_kernarg_block_count;
  // Queue kernarg-ring block count for custom direct-argument dispatches.
  uint32_t custom_kernarg_block_count;
  // Maximum static workgroup count accepted for each dimension.
  uint32_t max_workgroup_count[3];
  // Maximum dynamic group-memory byte count accepted for this export.
  uint32_t max_dynamic_workgroup_local_memory;
} iree_hal_amdgpu_executable_dispatch_descriptor_t;

// Infers the format of the executable and calculates its total size.
// If executable_data.data_length is 0 attempts to infer size from the data.
// Returns the canonical target-ID format string and total size of the
// executable data.
//
// Wrapped AMDGPU flatbuffers infer the target ID from the embedded ELF image
// instead of trusting the flatbuffer metadata target label.
iree_status_t iree_hal_amdgpu_executable_infer_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_allocator_t host_allocator, iree_host_size_t* out_inferred_size);

// Creates a AMDGPU executable from a binary in memory. Each executable may
// contain multiple entry points and be composed of several modules presented to
// the HAL as a single instance. See iree_hal_executable_params_t for more
// information about the lifetime of the resources referenced within.
//
// |libhsa| and |topology| are captured by-reference and must remain valid for
// the lifetime of the cache.
//
// Exact code-object image bytes and loader load ranges are retained in profile
// metadata for offline trace/disassembly workflows. Executable trace profiling
// may begin after executable preparation, so this cold-path metadata is always
// durable instead of being gated on an active profiling session.
iree_status_t iree_hal_amdgpu_executable_create(
    iree_hal_device_t* device, const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_amdgpu_profile_metadata_registry_t* profile_metadata,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Returns the producer-local profile executable id assigned at creation.
uint64_t iree_hal_amdgpu_executable_profile_id(
    iree_hal_executable_t* executable);

// Returns metadata about an exported kernel function in host memory.
// The returned pointers will remain valid for the lifetime of the executable.
// The returned kernel_object field is undefined in the returned args as there
// is no host representation and objects are per agent. To get an agent-specific
// kernel_object use iree_hal_amdgpu_executable_lookup_kernel_args_for_device.
iree_status_t iree_hal_amdgpu_executable_lookup_kernel_args_for_host(
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_amdgpu_device_kernel_args_t** out_kernel_args);

// Returns metadata about an exported kernel function in device memory.
// Kernel arguments are specific to the physical device specified by
// |device_ordinal| in the topology and cannot be used on any other device. The
// lookup fails if the executable queue affinity did not include
// |device_ordinal| at load time. The returned pointers will remain valid for
// the lifetime of the executable.
iree_status_t iree_hal_amdgpu_executable_lookup_kernel_args_for_device(
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t device_ordinal,
    const iree_hal_amdgpu_device_kernel_args_t** out_kernel_args);

// Returns host-resident dispatch metadata for an exported kernel function on a
// physical device.
//
// The returned descriptor is specific to |device_ordinal| because the kernel
// object embedded in the dispatch packet is per device. The lookup fails if the
// executable queue affinity did not include |device_ordinal| at load time. The
// pointer remains valid for the lifetime of the executable.
iree_status_t iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_device(
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t device_ordinal,
    const iree_hal_amdgpu_executable_dispatch_descriptor_t** out_descriptor);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_H_
