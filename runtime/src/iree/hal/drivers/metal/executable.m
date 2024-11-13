// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/metal/executable.h"

#include <Metal/Metal.h>
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/utils/executable_debug_info.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/executable_debug_info_reader.h"
#include "iree/schemas/executable_debug_info_verifier.h"
#include "iree/schemas/metal_executable_def_reader.h"
#include "iree/schemas/metal_executable_def_verifier.h"

typedef struct iree_hal_metal_executable_t {
  // Abstract resource used for injecting reference counting and vtable; must be at offset 0.
  iree_hal_resource_t resource;

  iree_allocator_t host_allocator;

  NSArray<id<MTLLibrary>>* libraries;

  iree_host_size_t pipeline_count;
  iree_hal_metal_pipeline_t pipelines[];
} iree_hal_metal_executable_t;

static const iree_hal_executable_vtable_t iree_hal_metal_executable_vtable;

static iree_hal_metal_executable_t* iree_hal_metal_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_executable_vtable);
  return (iree_hal_metal_executable_t*)base_value;
}

static const iree_hal_metal_executable_t* iree_hal_metal_executable_const_cast(
    const iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_executable_vtable);
  return (const iree_hal_metal_executable_t*)base_value;
}

// Verifies the structure of the flatbuffer so that we can avoid doing so during runtime.
//
// There are still some conditions we must be aware of (such as omitted names on functions with
// internal linkage), however we shouldn't need to bounds check anything within the flatbuffer
// after this succeeds.
static iree_status_t iree_hal_metal_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer data is not present or less than 16 bytes (%zu total)",
                            flatbuffer_data.data_length);
  }

  // Run flatcc generated verification. This ensures all pointers are in-bounds and that we can
  // safely walk the file, but not that the actual contents of the flatbuffer meet our expectations.
  int verify_ret = iree_hal_metal_ExecutableDef_verify_as_root(flatbuffer_data.data,
                                                               flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_hal_metal_ExecutableDef_table_t executable_def =
      iree_hal_metal_ExecutableDef_as_root(flatbuffer_data.data);

  iree_hal_metal_LibraryDef_vec_t libraries_vec =
      iree_hal_metal_ExecutableDef_libraries_get(executable_def);
  iree_host_size_t library_count = iree_hal_metal_LibraryDef_vec_len(libraries_vec);
  for (iree_host_size_t i = 0; i < library_count; ++i) {
    iree_hal_metal_LibraryDef_table_t library_def =
        iree_hal_metal_LibraryDef_vec_at(libraries_vec, i);
    if (!library_def) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "libraries[%" PRIhsz "] is NULL", i);
    }

    // NOTE: the source is optional but if present must be valid.
    iree_hal_metal_MSLSourceDef_table_t source_def =
        iree_hal_metal_LibraryDef_source_get(library_def);
    if (source_def) {
      // NOTE: the version check just ensures that we don't pass garbage to Metal; the current
      // platform may not support the version even if the enum is valid and we won't know until we
      // try compiling it.
      uint32_t version = iree_hal_metal_MSLSourceDef_version_get(source_def);
      if (version > MTLLanguageVersion3_0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "libraries[%" PRIhsz
                                "] MSL language version %u is unsupported by the compiled runtime",
                                i, version);
      }
      flatbuffers_string_t code = iree_hal_metal_MSLSourceDef_code_get(source_def);
      if (flatbuffers_string_len(code) == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "libraries[%" PRIhsz "] MSL source is empty", i);
      }
    }

    // Require that source is provided if no metallib is.
    flatbuffers_string_t metallib = iree_hal_metal_LibraryDef_metallib_get(library_def);
    if (flatbuffers_string_len(metallib) == 0 && !source_def) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "libraries[%" PRIhsz "] has neither source or binary data", i);
    }
  }

  iree_hal_metal_PipelineDef_vec_t pipelines_vec =
      iree_hal_metal_ExecutableDef_pipelines_get(executable_def);
  for (iree_host_size_t i = 0; i < iree_hal_metal_PipelineDef_vec_len(pipelines_vec); ++i) {
    iree_hal_metal_PipelineDef_table_t pipeline_def =
        iree_hal_metal_PipelineDef_vec_at(pipelines_vec, i);
    if (!pipeline_def) continue;

    uint32_t library_ordinal = iree_hal_metal_PipelineDef_library_ordinal_get(pipeline_def);
    if (library_ordinal >= library_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pipelines[%" PRIhsz "] library_ordinal %u is out of bounds %" PRIhsz,
                              i, library_ordinal, library_count);
    }

    if (flatbuffers_string_len(iree_hal_metal_PipelineDef_entry_point_get(pipeline_def)) == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pipelines[%" PRIhsz "] entry point name is empty", i);
    }

    if (!iree_hal_metal_PipelineDef_threadgroup_size_is_present(pipeline_def)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pipelines[%" PRIhsz "] threadgroup size is missing", i);
    }

    uint32_t constant_count = iree_hal_metal_PipelineDef_constant_count_get(pipeline_def);
    if (constant_count > IREE_HAL_METAL_MAX_PUSH_CONSTANT_COUNT) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pipelines[%" PRIhsz "] constant_count %u exceeds maximum of %u", i,
                              constant_count, IREE_HAL_METAL_MAX_PUSH_CONSTANT_COUNT);
    }

    iree_hal_metal_BindingBits_vec_t binding_flags_vec =
        iree_hal_metal_PipelineDef_binding_flags_get(pipeline_def);
    if (iree_hal_metal_BindingBits_vec_len(binding_flags_vec) >
        IREE_HAL_METAL_MAX_DESCRIPTOR_SET_BINDING_COUNT) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pipelines[%" PRIhsz
                              "] binding_flags count %zu exceeds maximum of %u",
                              i, iree_hal_metal_BindingBits_vec_len(binding_flags_vec),
                              IREE_HAL_METAL_MAX_DESCRIPTOR_SET_BINDING_COUNT);
    }

    IREE_RETURN_IF_ERROR(
        iree_hal_debug_verify_export_def(iree_hal_metal_PipelineDef_debug_info_get(pipeline_def)));
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_metal_compile_source(id<MTLDevice> device,
                                                   iree_hal_metal_MSLSourceDef_table_t source_def,
                                                   id<MTLLibrary>* out_library) {
  *out_library = nil;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  id<MTLLibrary> library = nil;
  @autoreleasepool {
    MTLCompileOptions* compile_options = [[MTLCompileOptions new] autorelease];
    compile_options.languageVersion = MTLLanguageVersion3_0;
    if (iree_hal_metal_MSLSourceDef_version_is_present(source_def)) {
      compile_options.languageVersion =
          (MTLLanguageVersion)iree_hal_metal_MSLSourceDef_version_get(source_def);
    }

    flatbuffers_string_t code = iree_hal_metal_MSLSourceDef_code_get(source_def);
    NSString* code_str =
        [[[NSString alloc] initWithBytes:code
                                  length:flatbuffers_string_len(code)
                                encoding:[NSString defaultCStringEncoding]] autorelease];

    NSError* error = nil;
    library = [device newLibraryWithSource:code_str options:compile_options error:&error];  // +1
    if (IREE_UNLIKELY(library == nil)) {
      const char* ns_c_error = [error.localizedDescription
          cStringUsingEncoding:[NSString defaultCStringEncoding]];  // autoreleased
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "failed to create MTLLibrary: %s",
                                ns_c_error);
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

static iree_status_t iree_hal_metal_load_library(id<MTLDevice> device,
                                                 flatbuffers_string_t metallib,
                                                 flatbuffers_string_t metallibsym,
                                                 id<MTLLibrary>* out_library) {
  *out_library = nil;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  id<MTLLibrary> library = nil;
  @autoreleasepool {
    dispatch_data_t data = dispatch_data_create(metallib, flatbuffers_string_len(metallib),
                                                /*queue=*/NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError* error = nil;
    library = [device newLibraryWithData:data error:&error];  // +1
    if (IREE_UNLIKELY(library == nil)) {
      const char* ns_c_error = [error.localizedDescription
          cStringUsingEncoding:[NSString defaultCStringEncoding]];  // autoreleased
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "failed to create MTLLibrary: %s",
                                ns_c_error);
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

// Loads all MTLLibrary instances in the executable and returns an array with matching order.
static iree_status_t iree_hal_metal_load_libraries(id<MTLDevice> device,
                                                   iree_hal_metal_LibraryDef_vec_t libraries_vec,
                                                   NSArray<id<MTLLibrary>>** out_libraries) {
  *out_libraries = nil;
  IREE_TRACE_ZONE_BEGIN(z0);

  NSMutableArray<id<MTLLibrary>>* libraries = [[NSMutableArray alloc] init];  // +1

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < iree_hal_metal_LibraryDef_vec_len(libraries_vec); ++i) {
    iree_hal_metal_LibraryDef_table_t library_def =
        iree_hal_metal_LibraryDef_vec_at(libraries_vec, i);
    id<MTLLibrary> library = nil;
    if (iree_hal_metal_LibraryDef_metallib_is_present(library_def)) {
      // Load binary MTLLibrary (with optional symbols).
      flatbuffers_string_t metallib = iree_hal_metal_LibraryDef_metallib_get(library_def);
      flatbuffers_string_t metallibsym = iree_hal_metal_LibraryDef_metallibsym_get(library_def);
      status = iree_hal_metal_load_library(device, metallib, metallibsym, &library);
    } else {
      // Compile MSL source code into a MTLLibrary.
      iree_hal_metal_MSLSourceDef_table_t source_def =
          iree_hal_metal_LibraryDef_source_get(library_def);
      status = iree_hal_metal_compile_source(device, source_def, &library);
    }
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

#if 0
// Creates MTL compute pipeline objects for the given |entry_point| in |library| and writes to
// |out_function| and |out_pipeline_state|. The caller should release |out_function| and
// |out_pipeline_state| after done.
static iree_status_t iree_hal_metal_create_pipeline_state(
    id<MTLLibrary> library, iree_string_view_t entry_point, const char* source_code,
    id<MTLDevice> device, id<MTLFunction>* out_function,
    id<MTLComputePipelineState>* out_pipeline_state) {
  @autoreleasepool {
    NSError* error = nil;

    // TODO(#14047): Enable async pipeline creation at runtime.
    *out_pipeline_state = [device newComputePipelineStateWithFunction:*out_function
                                                                error:&error];  // +1
    if (IREE_UNLIKELY(*out_pipeline_state == nil)) {
      [*out_function release];
      return iree_hal_metal_get_invalid_kernel_status(
          "invalid shader source", "when creating MTLComputePipelineState with NSError: %s", error,
          entry_point, source_code);
    }
  }
  return iree_ok_status();
}
#endif  // 0

static iree_status_t iree_hal_metal_create_pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                                    iree_hal_metal_PipelineDef_table_t pipeline_def,
                                                    iree_hal_metal_pipeline_t* out_pipeline) {
  IREE_TRACE_ZONE_BEGIN(z0);
  flatbuffers_string_t entry_point = iree_hal_metal_PipelineDef_entry_point_get(pipeline_def);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, entry_point);

  iree_status_t status = iree_ok_status();
  @autoreleasepool {
    NSString* function_name =
        [[[NSString alloc] initWithBytes:entry_point
                                  length:flatbuffers_string_len(entry_point)
                                encoding:[NSString defaultCStringEncoding]] autorelease];
    out_pipeline->function = [library newFunctionWithName:function_name];  // +1
    if (IREE_UNLIKELY(out_pipeline->function == nil)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "function `%.*s` not found in MTLLibrary",
                           (int)flatbuffers_string_len(entry_point), entry_point);
    }

    if (iree_status_is_ok(status)) {
      MTLComputePipelineDescriptor* descriptor =
          [[[MTLComputePipelineDescriptor alloc] init] autorelease];
      [descriptor setComputeFunction:out_pipeline->function];
      [descriptor setLabel:function_name];
      if (iree_hal_metal_PipelineDef_max_threads_per_threadgroup_is_present(pipeline_def)) {
        [descriptor setMaxTotalThreadsPerThreadgroup:
                        iree_hal_metal_PipelineDef_max_threads_per_threadgroup_get(pipeline_def)];
      }
      if (iree_hal_metal_PipelineDef_threadgroup_size_aligned_is_present(pipeline_def)) {
        [descriptor setThreadGroupSizeIsMultipleOfThreadExecutionWidth:
                        iree_hal_metal_PipelineDef_threadgroup_size_aligned_get(pipeline_def)];
      }
      [[[descriptor buffers] objectAtIndexedSubscript:0] setMutability:MTLMutabilityImmutable];
      [[[descriptor buffers] objectAtIndexedSubscript:IREE_HAL_METAL_PUSH_CONSTANT_BUFFER_INDEX]
          setMutability:MTLMutabilityImmutable];

      NSError* error = nil;
      out_pipeline->pipeline_state =
          [device newComputePipelineStateWithDescriptor:descriptor
                                                options:MTLPipelineOptionNone
                                             reflection:nil
                                                  error:&error];
      if (IREE_UNLIKELY(out_pipeline->pipeline_state == nil)) {
        const char* ns_c_error = [error.localizedDescription
            cStringUsingEncoding:[NSString defaultCStringEncoding]];  // autoreleased
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT, "failed to create pipeline with function `%.*s`: %s",
            (int)flatbuffers_string_len(entry_point), entry_point, ns_c_error);
      }
    }
  }

  if (iree_status_is_ok(status)) {
    const iree_hal_metal_ThreadgroupSize_t* threadgroup_size =
        iree_hal_metal_PipelineDef_threadgroup_size_get(pipeline_def);
    out_pipeline->threadgroup_size =
        MTLSizeMake(threadgroup_size->x, threadgroup_size->y, threadgroup_size->z);

    out_pipeline->constant_count = iree_hal_metal_PipelineDef_constant_count_get(pipeline_def);
    iree_hal_metal_BindingBits_vec_t binding_flags_vec =
        iree_hal_metal_PipelineDef_binding_flags_get(pipeline_def);
    out_pipeline->binding_count = iree_hal_metal_BindingBits_vec_len(binding_flags_vec);

    out_pipeline->binding_read_only_bits = 0;
    for (iree_host_size_t i = 0; i < out_pipeline->binding_count; ++i) {
      iree_hal_metal_BindingBits_enum_t binding_flags =
          iree_hal_metal_BindingBits_vec_at(binding_flags_vec, i);
      if (iree_all_bits_set(binding_flags, iree_hal_metal_BindingBits_IMMUTABLE)) {
        out_pipeline->binding_read_only_bits |= 1ull << i;
      }
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_metal_executable_create(
    id<MTLDevice> device, const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_IF_ERROR(
      iree_hal_metal_executable_flatbuffer_verify(executable_params->executable_data));

  iree_hal_metal_ExecutableDef_table_t executable_def =
      iree_hal_metal_ExecutableDef_as_root(executable_params->executable_data.data);

  iree_hal_metal_PipelineDef_vec_t pipelines_vec =
      iree_hal_metal_ExecutableDef_pipelines_get(executable_def);
  iree_host_size_t pipeline_count = flatbuffers_string_vec_len(pipelines_vec);

  // Calculate the total number of characters across all entry point names. This
  // is only required when tracing so that we can store copies of the names as
  // the flatbuffer storing the strings may be released while the executable is
  // still live.
  iree_host_size_t total_debug_info_length = 0;
  IREE_TRACE({
    for (iree_host_size_t i = 0; i < pipeline_count; ++i) {
      iree_hal_metal_PipelineDef_table_t pipeline_def =
          iree_hal_metal_PipelineDef_vec_at(pipelines_vec, i);
      total_debug_info_length += iree_hal_debug_calculate_export_info_size(
          iree_hal_metal_PipelineDef_debug_info_get(pipeline_def));
    }
  });

  // Allocate storage for the executable and its associated data structures.
  iree_hal_metal_executable_t* executable = NULL;
  iree_host_size_t total_size = sizeof(*executable) +
                                pipeline_count * sizeof(executable->pipelines[0]) +
                                total_debug_info_length;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  iree_hal_resource_initialize(&iree_hal_metal_executable_vtable, &executable->resource);
  executable->host_allocator = host_allocator;
  executable->pipeline_count = pipeline_count;
  IREE_TRACE(uint8_t* export_info_ptr = ((uint8_t*)executable->pipelines +
                                         pipeline_count * sizeof(executable->pipelines[0])));

  // Publish any embedded source files to the tracing infrastructure.
  iree_hal_debug_publish_source_files(
      iree_hal_metal_ExecutableDef_source_files_get(executable_def));

  // Load all libraries that may be referenced by the pipelines.
  iree_hal_metal_LibraryDef_vec_t libraries_vec =
      iree_hal_metal_ExecutableDef_libraries_get(executable_def);
  iree_status_t status =
      iree_hal_metal_load_libraries(device, libraries_vec, &executable->libraries);

  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < pipeline_count; ++i) {
      iree_hal_metal_PipelineDef_table_t pipeline_def =
          iree_hal_metal_PipelineDef_vec_at(pipelines_vec, i);

      uint32_t library_ordinal = iree_hal_metal_PipelineDef_library_ordinal_get(pipeline_def);
      id<MTLLibrary> library = [executable->libraries objectAtIndex:library_ordinal];  // unretained

      iree_hal_metal_pipeline_t* pipeline = &executable->pipelines[i];
      status = iree_hal_metal_create_pipeline(device, library, pipeline_def, pipeline);
      if (!iree_status_is_ok(status)) break;

      IREE_TRACE({
        iree_hal_debug_export_info_t* export_info = (iree_hal_debug_export_info_t*)export_info_ptr;
        export_info_ptr += iree_hal_debug_copy_export_info(
            iree_hal_metal_PipelineDef_debug_info_get(pipeline_def), export_info);
        pipeline->source_location.func_name = export_info->function_name;
        pipeline->source_location.file_name = export_info->source_filename;
        pipeline->source_location.line = export_info->source_line;
      });
    }
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_metal_executable_destroy(iree_hal_executable_t* base_executable) {
  iree_hal_metal_executable_t* executable = iree_hal_metal_executable_cast(base_executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->pipeline_count; ++i) {
    iree_hal_metal_pipeline_t* entry_point = &executable->pipelines[i];
    [entry_point->pipeline_state release];  // -1
    [entry_point->function release];        // -1
  }

  [executable->libraries release];  // -1

  iree_allocator_free(executable->host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_metal_executable_lookup_pipeline(
    const iree_hal_executable_t* base_executable, uint32_t entry_point,
    const iree_hal_metal_pipeline_t** out_pipeline) {
  const iree_hal_metal_executable_t* executable =
      iree_hal_metal_executable_const_cast(base_executable);
  if (entry_point >= executable->pipeline_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "invalid entry point ordinal %u",
                            entry_point);
  }
  *out_pipeline = &executable->pipelines[entry_point];
  return iree_ok_status();
}

static const iree_hal_executable_vtable_t iree_hal_metal_executable_vtable = {
    .destroy = iree_hal_metal_executable_destroy,
};
