// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/metal/builtin_executables.h"

#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/metal/builtin/metal_buffer_kernels.h"

typedef struct iree_hal_metal_builtin_executable_data_t {
  const char* entry_point;
  uint32_t file_index;
} iree_hal_metal_builtin_executable_data_t;

// The list of builtin executable entry points and their source file index in builtin exectuable
// embedded data. This MUST be consistent with kernel function names in MSL source code and the file
// order in embedded data.
static iree_hal_metal_builtin_executable_data_t iree_hal_metal_builtin_executable_entry_points[] = {
    {"fill_buffer_16byte", 1},  // Buffer fills; 16-byte aligned offset/length
    {"fill_buffer_4byte", 1},   // Buffer fills; 4-byte aligned offset/length
    {"fill_buffer_1byte", 1},   // Buffer fills; 1-byte aligned offset/length
    {"copy_buffer_1byte", 0},   // Buffer copies; 1-byte aligned offset/length
};

// The buffer fill specificiation. This MUST be consistent with the same struct in MSL source code.
typedef struct iree_hal_metal_buffer_fill_spec_t {
  uint64_t buffer_offset;  // Buffer offset to fill (in bytes)
  uint64_t buffer_length;  // Buffer length to fill (in bytes)
  uint32_t pattern;        // 32-bit fill pattern
} iree_hal_metal_buffer_fill_spec_t;

typedef struct iree_hal_metal_buffer_copy_spec_t {
  uint64_t src_buffer_offset;  // Source buffer offset (in bytes)
  uint64_t dst_buffer_offset;  // Destination buffer offset (in bytes)
  uint64_t length;             // Buffer length to fill (in bytes)
} iree_hal_metal_buffer_copy_spec_t;

iree_status_t iree_hal_metal_builtin_executable_create(
    id<MTLDevice> device, iree_allocator_t host_allocator,
    iree_hal_metal_builtin_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_metal_builtin_executable_t* executable = NULL;
  iree_host_size_t entry_point_count =
      IREE_ARRAYSIZE(iree_hal_metal_builtin_executable_entry_points);
  iree_host_size_t total_size =
      sizeof(*executable) + entry_point_count * sizeof(executable->entry_points[0]);
  iree_status_t status = iree_allocator_malloc(host_allocator, total_size, (void**)&executable);

  if (iree_status_is_ok(status)) {
    executable->host_allocator = host_allocator;
    executable->entry_point_count = entry_point_count;

    // Compile each MSL source string into a MTLLibrary and get the MTLFunction for the entry point
    // to build the pipeline state object.
    // TODO(antiagainst): We are performing synchronous compilation at runtime here. This is good
    // for debugging purposes but bad for performance. Enable offline compilation and make that as
    // the default.

    MTLCompileOptions* compile_options = [MTLCompileOptions new];  // +1
    compile_options.languageVersion = MTLLanguageVersion3_0;

    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(iree_hal_metal_builtin_executable_entry_points);
         ++i) {
      const char* entry_point = iree_hal_metal_builtin_executable_entry_points[i].entry_point;
      uint32_t file_index = iree_hal_metal_builtin_executable_entry_points[i].file_index;
      iree_file_toc_t source_code = metal_buffer_kernels_create()[file_index];

      id<MTLLibrary> library = nil;
      id<MTLFunction> function = nil;
      id<MTLComputePipelineState> pso = nil;
      status = iree_hal_metal_compile_msl_and_create_pipeline_object(
          iree_make_string_view(source_code.data, source_code.size), IREE_SV(entry_point), device,
          compile_options, &library, &function, &pso);
      if (!iree_status_is_ok(status)) break;

      // Package required parameters for kernel launches for each entry point.
      // Thread group size for builtin executables are determined at runtime dispatch time.
      // We don't need the layout information for builtins either.
      iree_hal_metal_kernel_params_t* params = &executable->entry_points[i];
      memset(params, 0, sizeof(*params));
      params->library = library;
      params->function = function;
      params->pso = pso;

      // Stash the entry point name in the string table for use when tracing.
      IREE_TRACE({ params->function_name = IREE_SV(entry_point); });
    }

    [compile_options release];  // -1
  }

  if (iree_status_is_ok(status)) {
    *out_executable = executable;
  } else {
    iree_hal_metal_builtin_executable_destroy(executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_metal_builtin_executable_destroy(iree_hal_metal_builtin_executable_t* executable) {
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->entry_point_count; ++i) {
    iree_hal_metal_kernel_params_t* entry_point = &executable->entry_points[i];
    [entry_point->pso release];
    [entry_point->function release];
    [entry_point->library release];
    IREE_ASSERT_EQ(entry_point->layout, NULL);
  }
  iree_allocator_free(executable->host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static inline iree_device_size_t iree_hal_metal_ceil_div(iree_device_size_t a,
                                                         iree_device_size_t b) {
  return (a + b - 1) / b;
}

iree_status_t iree_hal_metal_builtin_executable_fill_buffer(
    const iree_hal_metal_builtin_executable_t* executable, id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> target_buffer, iree_device_size_t target_offset, iree_device_size_t length,
    uint32_t pattern) {
  id<MTLComputePipelineState> pso = nil;
  MTLResourceUsage usage = MTLResourceUsageWrite;
  const iree_device_size_t workgroup_size = 32;
  iree_device_size_t workgroup_count = 0;

  if (target_offset % 16 == 0 && length % 16 == 0) {  // 16-byte aligned case
    pso = executable->entry_points[0].pso;
    workgroup_count = iree_hal_metal_ceil_div(length, workgroup_size * 16);
  } else if (target_offset % 4 == 0 && length % 4 == 0) {  // 4-byte aligned case
    pso = executable->entry_points[1].pso;
    workgroup_count = iree_hal_metal_ceil_div(length, workgroup_size * 4);
  } else {  // 1-byte aligned case
    pso = executable->entry_points[2].pso;
    // We may potentially need to read some 32-bit scalars at unaligned addresses.
    usage |= MTLResourceUsageRead;
    // Calculate unaligned partial prefix/suffix byte count, and then get the middle aligned byte
    // count for distributing threads. This logic MUST be consistent with the MSL source code.
    iree_device_size_t left_byte_count = target_offset % 4;
    iree_device_size_t right_byte_count = (target_offset + length) % 4;
    int64_t middle_byte_count = length - left_byte_count - right_byte_count;
    // Note that in the extreme case, we don't have aligned bytes in the middle (0), or actually
    // prefix and suffix partial bytes are the same (< 0). We'd need one thread to handle the
    // partial bytes at least.
    if (middle_byte_count <= 0) middle_byte_count = 1;
    workgroup_count = iree_hal_metal_ceil_div(middle_byte_count, workgroup_size * 4);
  }

  iree_hal_metal_buffer_fill_spec_t spec = {
      .buffer_offset = target_offset,
      .buffer_length = length,
      .pattern = pattern,
  };

  [encoder setComputePipelineState:pso];

  // The following MUST exactly match the pipeline layout from MSL source code.
  // buffer(0) is the target buffer to fill. Note that we MUST set 0 as offset here--the offset
  // is to be handled directly in the kernels!
  [encoder setBuffer:target_buffer offset:0 atIndex:0];
  [encoder useResource:target_buffer usage:usage];
  // buffer(1) is the buffer fill spec.
  [encoder setBytes:&spec length:sizeof(spec) atIndex:1];

  // Encode the dispatch.
  [encoder dispatchThreadgroups:MTLSizeMake(workgroup_count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(workgroup_size, 1, 1)];
  return iree_ok_status();
}

iree_status_t iree_hal_metal_builtin_executable_copy_buffer(
    const iree_hal_metal_builtin_executable_t* executable, id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> source_buffer, iree_device_size_t source_offset, id<MTLBuffer> target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  id<MTLComputePipelineState> pso = executable->entry_points[3].pso;
  const iree_device_size_t workgroup_size = 32;
  iree_device_size_t workgroup_count = iree_hal_metal_ceil_div(length, workgroup_size * 4);

  iree_hal_metal_buffer_copy_spec_t spec = {
      .src_buffer_offset = source_offset,
      .dst_buffer_offset = target_offset,
      .length = length,
  };

  [encoder setComputePipelineState:pso];

  // The following MUST exactly match the pipeline layout from MSL source code.
  // buffer(0) is the source buffer. Note that we MUST set 0 as offset here--the offset is to be
  // handled directly in the kernels!
  [encoder setBuffer:source_buffer offset:0 atIndex:0];
  [encoder useResource:source_buffer usage:MTLResourceUsageRead];
  // buffer(0) is the target buffer. Note that we MUST set 0 as offset here--the offset is to be
  // handled directly in the kernels!
  [encoder setBuffer:target_buffer offset:0 atIndex:1];
  [encoder useResource:target_buffer usage:MTLResourceUsageWrite];
  // buffer(1) is the buffer copy spec.
  [encoder setBytes:&spec length:sizeof(spec) atIndex:2];

  // Encode the dispatch.
  [encoder dispatchThreadgroups:MTLSizeMake(workgroup_count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(workgroup_size, 1, 1)];
  return iree_ok_status();
}
