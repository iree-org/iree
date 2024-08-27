// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/metal/builtin_executables.h"

#include <Foundation/Foundation.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/metal/builtin/metal_buffer_kernels.h"

typedef struct iree_hal_metal_builtin_pipeline_info_t {
  iree_string_view_t entry_point;
  uint32_t file_index;
} iree_hal_metal_builtin_pipeline_info_t;

// The list of builtin executable entry points and their source file index in builtin exectuable
// embedded data.
//
// NOTE: must be consistent with the same struct in MSL source code.
// NOTE: the exact order here is assumed below and must not change (that should be fixed...).
static iree_hal_metal_builtin_pipeline_info_t iree_hal_metal_builtin_pipeline_info[] = {
    {IREE_SVL("fill_buffer_16byte"), 1},  // Buffer fills; 16-byte aligned offset/length
    {IREE_SVL("fill_buffer_4byte"), 1},   // Buffer fills; 4-byte aligned offset/length
    {IREE_SVL("fill_buffer_1byte"), 1},   // Buffer fills; 1-byte aligned offset/length
    {IREE_SVL("copy_buffer_1byte"), 0},   // Buffer copies; 1-byte aligned offset/length
};

// NOTE: must be consistent with the same struct in MSL source code.
typedef struct iree_hal_metal_buffer_fill_spec_t {
  uint64_t buffer_offset;  // Buffer offset to fill (in bytes)
  uint64_t buffer_length;  // Buffer length to fill (in bytes)
  uint32_t pattern;        // 32-bit fill pattern
} iree_hal_metal_buffer_fill_spec_t;

// NOTE: must be consistent with the same struct in MSL source code.
typedef struct iree_hal_metal_buffer_copy_spec_t {
  uint64_t src_buffer_offset;  // Source buffer offset (in bytes)
  uint64_t dst_buffer_offset;  // Destination buffer offset (in bytes)
  uint64_t length;             // Buffer length to fill (in bytes)
} iree_hal_metal_buffer_copy_spec_t;

// Compiles |source_file| as MSL source into a MTLLibrary for the given |device|.
//
// TODO: we should be precompiling this and shipping a binary metallib instead: compiling from
// source at runtime is _extremely_ inefficient.
static iree_status_t iree_hal_metal_compile_embedded_msl(id<MTLDevice> device,
                                                         iree_file_toc_t source_file,
                                                         id<MTLLibrary>* out_library) {
  *out_library = nil;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, source_file.name);

  iree_status_t status = iree_ok_status();
  id<MTLLibrary> library = nil;
  @autoreleasepool {
    MTLCompileOptions* compile_options = [[MTLCompileOptions new] autorelease];
    compile_options.languageVersion = MTLLanguageVersion3_0;

    NSString* shader_source =
        [[[NSString alloc] initWithBytes:source_file.data
                                  length:source_file.size
                                encoding:[NSString defaultCStringEncoding]] autorelease];

    NSError* error = nil;
    library = [device newLibraryWithSource:shader_source
                                   options:compile_options
                                     error:&error];  // +1
    if (IREE_UNLIKELY(library == nil)) {
      const char* ns_c_error = [error.localizedDescription
          cStringUsingEncoding:[NSString defaultCStringEncoding]];  // autoreleased
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "failed to create MTLLibrary from shader source in %s: %s",
                                source_file.name, ns_c_error);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_library = library;
  } else {
    [library release];
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Loads all MTLLibrary instances required by the builtin pipelines.
static iree_status_t iree_hal_metal_load_builtin_libraries(
    id<MTLDevice> device, NSArray<id<MTLLibrary>>** out_libraries) {
  *out_libraries = nil;
  IREE_TRACE_ZONE_BEGIN(z0);

  NSMutableArray<id<MTLLibrary>>* libraries = [[NSMutableArray alloc] init];  // +1

  // TODO: don't compile sources and instead embed the libraries in binary form.
  // Embedding source files is an anti-pattern.
  iree_status_t status = iree_ok_status();
  const iree_file_toc_t* embedded_files = metal_buffer_kernels_create();
  for (iree_host_size_t i = 0; i < metal_buffer_kernels_size(); ++i) {
    iree_file_toc_t source_file = embedded_files[i];
    id<MTLLibrary> library = nil;
    status = iree_hal_metal_compile_embedded_msl(device, source_file, &library);
    if (!iree_status_is_ok(status)) break;
    [libraries addObject:library];
  }

  if (iree_status_is_ok(status)) {
    *out_libraries = libraries;
  } else {
    [libraries release];  // -1
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Creates MTL compute pipeline objects for the given |entry_point| in |library| and writes to
// |out_function| and |out_pipeline_state|. The caller should release |out_function| and
// |out_pipeline_state| after done.
static iree_status_t iree_hal_metal_create_builtin_pipeline(
    id<MTLDevice> device, id<MTLLibrary> library, iree_string_view_t entry_point,
    iree_hal_metal_builtin_pipeline_t* out_pipeline) {
  @autoreleasepool {
    NSString* function_name =
        [[[NSString alloc] initWithBytes:entry_point.data
                                  length:entry_point.size
                                encoding:[NSString defaultCStringEncoding]] autorelease];
    id<MTLFunction> function = [[library newFunctionWithName:function_name] autorelease];
    if (IREE_UNLIKELY(function == nil)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "function %.*s not found in the provided MTLLibrary",
                              (int)entry_point.size, entry_point.data);
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline_state =
        [device newComputePipelineStateWithFunction:function error:&error];  // +1
    if (IREE_UNLIKELY(pipeline_state == nil)) {
      const char* ns_c_error = [error.localizedDescription
          cStringUsingEncoding:[NSString defaultCStringEncoding]];  // autoreleased
      return iree_make_status(IREE_STATUS_INTERNAL, "invalid shader source for builtin %.*s: %s",
                              (int)entry_point.size, entry_point.data, ns_c_error);
    }

    out_pipeline->pipeline_state = pipeline_state;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_metal_builtin_executable_create(
    id<MTLDevice> device, iree_allocator_t host_allocator,
    iree_hal_metal_builtin_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_metal_builtin_executable_t* executable = NULL;
  iree_host_size_t pipeline_count = IREE_ARRAYSIZE(iree_hal_metal_builtin_pipeline_info);
  iree_host_size_t total_size =
      sizeof(*executable) + pipeline_count * sizeof(executable->pipelines[0]);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  executable->host_allocator = host_allocator;
  executable->pipeline_count = pipeline_count;

  // Load all MTLLibrary instances used by the pipelines.
  iree_status_t status = iree_hal_metal_load_builtin_libraries(device, &executable->libraries);
  if (iree_status_is_ok(status)) {
    // Create pipelines using the loaded libraries.
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(iree_hal_metal_builtin_pipeline_info); ++i) {
      const iree_hal_metal_builtin_pipeline_info_t* pipeline_info =
          &iree_hal_metal_builtin_pipeline_info[i];
      IREE_TRACE_ZONE_BEGIN(z_pipeline);
      IREE_TRACE_ZONE_APPEND_TEXT(z_pipeline, pipeline_info->entry_point.data,
                                  pipeline_info->entry_point.size);

      iree_hal_metal_builtin_pipeline_t* pipeline = &executable->pipelines[i];
      IREE_TRACE({
        const iree_file_toc_t* embedded_files = metal_buffer_kernels_create();
        iree_file_toc_t source_file = embedded_files[pipeline_info->file_index];
        pipeline->source_location.func_name = pipeline_info->entry_point;
        pipeline->source_location.file_name = IREE_SV(source_file.name);
        pipeline->source_location.line = 0;
      });

      id<MTLLibrary> library =
          [executable->libraries objectAtIndex:pipeline_info->file_index];  // unretained
      status = iree_hal_metal_create_builtin_pipeline(device, library, pipeline_info->entry_point,
                                                      pipeline);

      IREE_TRACE_ZONE_END(z_pipeline);
      if (!iree_status_is_ok(status)) break;
    }
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

  for (iree_host_size_t i = 0; i < executable->pipeline_count; ++i) {
    iree_hal_metal_builtin_pipeline_t* pipeline = &executable->pipelines[i];
    [pipeline->pipeline_state release];
  }

  [executable->libraries release];

  iree_allocator_free(executable->host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_metal_builtin_executable_fill_buffer(
    const iree_hal_metal_builtin_executable_t* executable, id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> target_buffer, iree_device_size_t target_offset, iree_device_size_t length,
    uint32_t pattern) {
  id<MTLComputePipelineState> pipeline_state = nil;
  MTLResourceUsage usage = MTLResourceUsageWrite;
  const iree_device_size_t workgroup_size = 32;
  iree_device_size_t workgroup_count = 0;
  if (target_offset % 16 == 0 && length % 16 == 0) {  // 16-byte aligned case
    pipeline_state = executable->pipelines[0].pipeline_state;
    workgroup_count = iree_device_size_ceil_div(length, workgroup_size * 16);
  } else if (target_offset % 4 == 0 && length % 4 == 0) {  // 4-byte aligned case
    pipeline_state = executable->pipelines[1].pipeline_state;
    workgroup_count = iree_device_size_ceil_div(length, workgroup_size * 4);
  } else {  // 1-byte aligned case
    pipeline_state = executable->pipelines[2].pipeline_state;
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
    workgroup_count = iree_device_size_ceil_div(middle_byte_count, workgroup_size * 4);
  }
  [encoder setComputePipelineState:pipeline_state];

  // The following MUST exactly match the pipeline layout from MSL source code.
  // buffer(0) is the target buffer to fill. Note that we MUST set 0 as offset here--the offset
  // is to be handled directly in the kernels!
  [encoder setBuffer:target_buffer offset:0 atIndex:0];
  [encoder useResource:target_buffer usage:usage];

  // buffer(1) is the buffer fill spec.
  iree_hal_metal_buffer_fill_spec_t spec = {
      .buffer_offset = target_offset,
      .buffer_length = length,
      .pattern = pattern,
  };
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
  id<MTLComputePipelineState> pipeline_state = executable->pipelines[3].pipeline_state;
  [encoder setComputePipelineState:pipeline_state];

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
  iree_hal_metal_buffer_copy_spec_t spec = {
      .src_buffer_offset = source_offset,
      .dst_buffer_offset = target_offset,
      .length = length,
  };
  [encoder setBytes:&spec length:sizeof(spec) atIndex:2];

  // Encode the dispatch.
  const iree_device_size_t workgroup_size = 32;
  iree_device_size_t workgroup_count = iree_device_size_ceil_div(length, workgroup_size * 4);
  [encoder dispatchThreadgroups:MTLSizeMake(workgroup_count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(workgroup_size, 1, 1)];

  return iree_ok_status();
}
