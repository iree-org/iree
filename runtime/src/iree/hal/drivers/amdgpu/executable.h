// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/device/support/kernel_args.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;

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

// Returns whether the canonical IREE HAL executable |format| is supported by
// all GPU devices in |topology|. Some devices may support multiple ISAs.
//
// To avoid creating yet another naming scheme we directly use the ISA names
// reported by HSA, e.g. `amdgcn-amd-amdhsa--gfx1100`. It's not pretty, but it
// is precise for this particular HAL and lets us avoid any potential runtime
// changes if LLVM<->HSA naming changes with new code object versions.
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

// Creates a AMDGPU executable from a binary in memory. Each executable may
// contain multiple entry points and be composed of several modules presented to
// the HAL as a single instance. See iree_hal_executable_params_t for more
// information about the lifetime of the resources referenced within.
//
// |libhsa| and |topology| are captured by-reference and must remain valid for
// the lifetime of the cache.
iree_status_t iree_hal_amdgpu_executable_create(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Returns metadata about an exported kernel function in host memory.
// The returned pointers will remain valid for the lifetime of the executable.
// The returned kernel_object field is undefined in the returned args as there
// is no host representation and objects are per agent. To get an agent-specific
// kernel_object use iree_hal_amdgpu_executable_lookup_kernel_args_for_device.
iree_status_t iree_hal_amdgpu_executable_lookup_kernel_args_for_host(
    iree_hal_executable_t* executable, iree_host_size_t entry_point,
    const iree_hal_amdgpu_device_kernel_args_t** out_kernel_args);

// Returns metadata about an exported kernel function in device memory.
// Kernel arguments are specific to the physical device specified by
// |device_ordinal| in the topology and cannot be used on any other device. The
// returned pointers will remain valid for the lifetime of the executable.
iree_status_t iree_hal_amdgpu_executable_lookup_kernel_args_for_device(
    iree_hal_executable_t* executable, iree_host_size_t entry_point,
    iree_host_size_t device_ordinal,
    const iree_hal_amdgpu_device_kernel_args_t** out_kernel_args);

#endif  // IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_H_
