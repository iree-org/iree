// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/executable_library_demo.h"

#include <stddef.h>

// An executable entry point, called one or more times based on the 3D XYZ
// workgroup count specified during the dispatch. Each invocation gets access to
// the dispatch state via |dispatch_state| such as workgroup parameters, push
// constants providing small arguments, and buffer bindings.
//
// See the iree_hal_executable_dispatch_state_v0_t struct for more
// information on the fields here and how they can be used.
//
// WARNING: these functions must not access mutable global state: read-only data
// may be used but as each invocation may be running concurrently with any
// number of other invocations (from any number of user sessions!) all
// communication between invocations must use the buffer bindings for I/O.
//
// This is a simple scalar addition:
//    binding[1] = binding[0] + constant[0]
static int dispatch_tile_a(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  const dispatch_tile_a_constants_t* constants =
      (const dispatch_tile_a_constants_t*)dispatch_state->constants;
  const float* src = ((const float*)dispatch_state->binding_ptrs[0]);
  float* dst = ((float*)dispatch_state->binding_ptrs[1]);
  const uint32_t x = workgroup_state->workgroup_id_x;
  dst[x] = src[x] + constants->f0;
  return 0;
}

// Just another entry point.
static int dispatch_tile_b(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  return 0;
}

// Version/metadata header.
static const iree_hal_executable_library_header_t header = {
    // Declares what library version is present: newer runtimes may support
    // loading older executables but newer executables cannot load on older
    // runtimes.
    .version = IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST,
    // Name used for logging/diagnostics and rendezvous.
    .name = "demo_library",
    .features = IREE_HAL_EXECUTABLE_LIBRARY_FEATURE_NONE,
    .sanitizer = IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE,
};

// Table of export function entry points.
static const iree_hal_executable_dispatch_v0_t entry_points[2] = {
    dispatch_tile_a,
    dispatch_tile_b,
};

// Dispatch parameters for dispatch_tile_a.
// This describes the function's ABI: 1 constant and 2 bindings.
static const iree_hal_executable_dispatch_parameter_v0_t
    dispatch_tile_a_params[3] = {
        {
            // The float constant 'f0' used for addition.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
            .size = sizeof(float),
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 0,    // "scalar_addend"
            .offset = 0,  // starts at constants[0]
        },
        {
            // Input buffer binding.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .size = 0,  // unused for bindings
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 1,    // "src_buffer"
            .offset = 0,  // binding[0]
        },
        {
            // Output buffer binding.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .size = 0,  // unused for bindings
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 2,    // "dst_buffer"
            .offset = 1,  // binding[1]
        },
};

// Dispatch parameters for dispatch_tile_b.
// This function only uses 2 bindings and no constants.
static const iree_hal_executable_dispatch_parameter_v0_t
    dispatch_tile_b_params[2] = {
        {
            // Input buffer binding.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .size = 0,  // unused for bindings
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 1,    // "src_buffer"
            .offset = 0,  // binding[0]
        },
        {
            // Output buffer binding.
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .size = 0,  // unused for bindings
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 2,    // "dst_buffer"
            .offset = 1,  // binding[1]
        },
};

// Parameter table - one array per dispatch entry point.
// NULL entries indicate the dispatch follows the standard HAL ABI.
static const iree_hal_executable_dispatch_parameter_v0_t* entry_params[2] = {
    dispatch_tile_a_params,
    dispatch_tile_b_params,
};

// String table for parameter names.
// These are referenced by the 'name' field in the param structs above.
static const char* parameter_names[3] = {
    "scalar_addend",
    "src_buffer",
    "dst_buffer",
};

// Attributes for each dispatch function used by the runtime.
// Required to specify constant and binding counts for dispatch validation.
static const iree_hal_executable_dispatch_attrs_v0_t entry_attrs[2] = {
    {
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        .local_memory_pages = 0,
        .constant_count = 1,
        .binding_count = 2,
        .workgroup_size_x = 0,  // runtime specified
        .workgroup_size_y = 0,  // runtime specified
        .workgroup_size_z = 0,  // runtime specified
        .parameter_count =
            sizeof(dispatch_tile_a_params) / sizeof(dispatch_tile_a_params[0]),
    },
    {
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        .local_memory_pages = 0,
        .constant_count = 0,
        .binding_count = 2,
        .workgroup_size_x = 0,  // runtime specified
        .workgroup_size_y = 0,  // runtime specified
        .workgroup_size_z = 0,  // runtime specified
        .parameter_count =
            sizeof(dispatch_tile_b_params) / sizeof(dispatch_tile_b_params[0]),
    },
};

// Names for each entry point.
static const char* entry_point_names[2] = {
    "dispatch_tile_a",
    "dispatch_tile_b",
};

// User tags for debugging/logging; not used for anything but presentation.
static const char* entry_point_tags[2] = {
    "matmul+div",
    "conv2d[512x512]",
};

static const iree_hal_executable_library_v0_t library = {
    .header = &header,
    .imports =
        {
            .count = 0,
            .symbols = NULL,
        },
    .exports =
        {
            .count = 2,
            .ptrs = entry_points,
            .attrs = entry_attrs,
            .params = entry_params,
            .occupancy = NULL,  // no occupancy info provided
            .names = entry_point_names,
            .tags = entry_point_tags,
            .parameter_names = parameter_names,
            .source_locations = NULL,
            .stage_locations = NULL,
        },
    .constants =
        {
            .count = 0,
        },
};

// The primary access point to the executable: in a static library this is
// just like any other C symbol that can be called from other code (like
// executable_library_test.c does), and in dynamic libraries this is the symbol
// that you would be dlsym'ing.
//
// This is just code: if the executable wants to return different headers based
// on the currently executing architecture or the requested version it can. For
// example, an executable may want to swap out a few entry points to an
// architecture-specific version.
const iree_hal_executable_library_header_t** demo_executable_library_query(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment) {
  return max_version <= IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST
             ? (const iree_hal_executable_library_header_t**)&library
             : NULL;
}
