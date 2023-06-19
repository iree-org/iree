// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/metal/metal_kernel_library.h"

#include <stddef.h>

#include "iree/base/api.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/metal_executable_def_reader.h"
#include "iree/schemas/metal_executable_def_verifier.h"

typedef struct iree_hal_metal_kernel_library_t {
  // Abstract resource used for injecting reference counting and vtable; must be at offset 0.
  iree_hal_resource_t resource;

  iree_allocator_t host_allocator;

  iree_host_size_t entry_point_count;
  iree_hal_metal_kernel_params_t entry_points[];
} iree_hal_metal_kernel_library_t;

static const iree_hal_executable_vtable_t iree_hal_metal_kernel_library_vtable;

static iree_hal_metal_kernel_library_t* iree_hal_metal_kernel_library_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_kernel_library_vtable);
  return (iree_hal_metal_kernel_library_t*)base_value;
}

static const iree_hal_metal_kernel_library_t* iree_hal_metal_kernel_library_const_cast(
    const iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_kernel_library_vtable);
  return (const iree_hal_metal_kernel_library_t*)base_value;
}

// Verifies the structure of the flatbuffer so that we can avoid doing so during runtime.
//
// There are still some conditions we must be aware of (such as omitted names on functions with
// internal linkage), however we shouldn't need to bounds check anything within the flatbuffer
// after this succeeds.
static iree_status_t iree_hal_metal_kernel_library_flatbuffer_verify(
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

  flatbuffers_string_vec_t entry_points_vec =
      iree_hal_metal_ExecutableDef_entry_points_get(executable_def);
  size_t entry_point_count = flatbuffers_string_vec_len(entry_points_vec);
  for (size_t i = 0; i < entry_point_count; ++i) {
    if (!flatbuffers_string_len(flatbuffers_string_vec_at(entry_points_vec, i))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable entry point %zu has no name", i);
    }
  }

  iree_hal_metal_ThreadgroupSize_vec_t threadgroup_sizes_vec =
      iree_hal_metal_ExecutableDef_threadgroup_sizes(executable_def);
  size_t threadgroup_size_count = iree_hal_metal_ThreadgroupSize_vec_len(threadgroup_sizes_vec);
  if (!threadgroup_size_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "no threadgroup sizes present");
  }

  if (entry_point_count != threadgroup_size_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry points (%zu) and thread group sizes (%zu) count mismatch",
                            entry_point_count, threadgroup_size_count);
  }

  flatbuffers_string_vec_t shader_libraries_vec =
      iree_hal_metal_ExecutableDef_shader_libraries_get(executable_def);
  size_t shader_library_count = flatbuffers_string_vec_len(shader_libraries_vec);
  for (size_t i = 0; i < shader_library_count; ++i) {
    if (!flatbuffers_string_len(flatbuffers_string_vec_at(shader_libraries_vec, i))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable shader library %zu is empty", i);
    }
  }
  if (shader_library_count != 0 && entry_point_count != shader_library_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry points (%zu) and source libraries (%zu) count mismatch",
                            entry_point_count, shader_library_count);
  }

  flatbuffers_string_vec_t shader_sources_vec =
      iree_hal_metal_ExecutableDef_shader_sources_get(executable_def);
  size_t shader_source_count = flatbuffers_string_vec_len(shader_sources_vec);
  for (size_t i = 0; i < shader_source_count; ++i) {
    if (!flatbuffers_string_len(flatbuffers_string_vec_at(shader_sources_vec, i))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "executable shader source %zu is empty",
                              i);
    }
  }

  if (shader_source_count != 0 && entry_point_count != shader_source_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry points (%zu) and source strings (%zu) count mismatch",
                            entry_point_count, shader_source_count);
  }

  if (!shader_library_count && !shader_source_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "missing shader library or source strings");
  }

  return iree_ok_status();
}

// Returns an invalid argument status with proper Metal NSError annotations during compute pipeline
// creation.
static iree_status_t iree_hal_metal_get_invalid_kernel_status(const char* iree_error_template,
                                                              const char* metal_error_template,
                                                              NSError* ns_error,
                                                              iree_string_view_t entry_point,
                                                              const char* shader_source) {
  iree_status_t status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, iree_error_template);
  const char* ns_c_error = [ns_error.localizedDescription
      cStringUsingEncoding:[NSString defaultCStringEncoding]];  // autoreleased
  status = iree_status_annotate_f(status, metal_error_template, ns_c_error);
  if (shader_source) {
    return iree_status_annotate_f(status, "for entry point '%.*s' in MSL source:\n%s\n",
                                  (int)entry_point.size, entry_point.data, shader_source);
  }
  return iree_status_annotate_f(status, "for entry point '%.*s' in MTLLibrary\n",
                                (int)entry_point.size, entry_point.data);
}

// Compiles the given |entry_point| in the MSL |source_code| into MTLLibrary and writes to
// |out_library|. The caller should release |out_library| after done.
iree_status_t iree_hal_metal_compile_msl(iree_string_view_t source_code,
                                         iree_string_view_t entry_point, id<MTLDevice> device,
                                         MTLCompileOptions* compile_options,
                                         id<MTLLibrary>* out_library) {
  @autoreleasepool {
    NSError* error = nil;
    NSString* shader_source =
        [[[NSString alloc] initWithBytes:source_code.data
                                  length:source_code.size
                                encoding:[NSString defaultCStringEncoding]] autorelease];
    *out_library = [device newLibraryWithSource:shader_source
                                        options:compile_options
                                          error:&error];  // +1
    if (IREE_UNLIKELY(*out_library == nil)) {
      return iree_hal_metal_get_invalid_kernel_status(
          "failed to create MTLLibrary from shader source",
          "when creating MTLLibrary with NSError: %.*s", error, entry_point, source_code.data);
    }
  }

  return iree_ok_status();
}

// Compiles the given |entry_point| in the MSL library |source_data| into MTLLibrary and writes to
// |out_library|. The caller should release |out_library| after done.
static iree_status_t iree_hal_metal_load_mtllib(iree_const_byte_span_t source_data,
                                                iree_string_view_t entry_point,
                                                id<MTLDevice> device, id<MTLLibrary>* out_library) {
  @autoreleasepool {
    NSError* error = nil;
    dispatch_data_t data = dispatch_data_create(source_data.data, source_data.data_length,
                                                /*queue=*/NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    *out_library = [device newLibraryWithData:data error:&error];  // +1
    if (IREE_UNLIKELY(*out_library == nil)) {
      return iree_hal_metal_get_invalid_kernel_status(
          "failed to create MTLLibrary from shader source",
          "when creating MTLLibrary with NSError: %s", error, entry_point, NULL);
    }
  }

  return iree_ok_status();
}

// Creates MTL compute pipeline objects for the given |entry_point| in |library| and writes to
// |out_function| and |out_pso|. The caller should release |out_function| and |out_pso| after done.
static iree_status_t iree_hal_metal_create_pipline_object(
    id<MTLLibrary> library, iree_string_view_t entry_point, const char* source_code,
    id<MTLDevice> device, id<MTLFunction>* out_function, id<MTLComputePipelineState>* out_pso) {
  @autoreleasepool {
    NSError* error = nil;
    NSString* function_name =
        [[[NSString alloc] initWithBytes:entry_point.data
                                  length:entry_point.size
                                encoding:[NSString defaultCStringEncoding]] autorelease];
    *out_function = [library newFunctionWithName:function_name];  // +1
    if (IREE_UNLIKELY(*out_function == nil)) {
      return iree_hal_metal_get_invalid_kernel_status("cannot find entry point in shader source",
                                                      "when creating MTLFunction with NSError: %s",
                                                      error, entry_point, source_code);
    }

    // TODO(#14047): Enable async pipeline creation at runtime.
    *out_pso = [device newComputePipelineStateWithFunction:*out_function error:&error];  // +1
    if (IREE_UNLIKELY(*out_pso == nil)) {
      [*out_function release];
      return iree_hal_metal_get_invalid_kernel_status(
          "invalid shader source", "when creating MTLComputePipelineState with NSError: %s", error,
          entry_point, source_code);
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_metal_compile_msl_and_create_pipeline_object(
    iree_string_view_t source_code, iree_string_view_t entry_point, id<MTLDevice> device,
    MTLCompileOptions* compile_options, id<MTLLibrary>* out_library, id<MTLFunction>* out_function,
    id<MTLComputePipelineState>* out_pso) {
  IREE_RETURN_IF_ERROR(
      iree_hal_metal_compile_msl(source_code, entry_point, device, compile_options, out_library));
  return iree_hal_metal_create_pipline_object(*out_library, entry_point, source_code.data, device,
                                              out_function, out_pso);
}

iree_status_t iree_hal_metal_kernel_library_create(
    id<MTLDevice> device, const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_metal_kernel_library_t* executable = NULL;

  IREE_RETURN_IF_ERROR(
      iree_hal_metal_kernel_library_flatbuffer_verify(executable_params->executable_data));

  iree_hal_metal_ExecutableDef_table_t executable_def =
      iree_hal_metal_ExecutableDef_as_root(executable_params->executable_data.data);

  flatbuffers_string_vec_t entry_points_vec =
      iree_hal_metal_ExecutableDef_entry_points_get(executable_def);
  iree_hal_metal_ThreadgroupSize_vec_t threadgroup_sizes_vec =
      iree_hal_metal_ExecutableDef_threadgroup_sizes(executable_def);
  flatbuffers_string_vec_t shader_libraries_vec =
      iree_hal_metal_ExecutableDef_shader_libraries_get(executable_def);
  flatbuffers_string_vec_t shader_sources_vec =
      iree_hal_metal_ExecutableDef_shader_sources_get(executable_def);
  iree_host_size_t entry_point_count = flatbuffers_string_vec_len(entry_points_vec);

  // Calculate the total number of characters across all entry point names. This is only required
  // when tracing so that we can store copies of the names as the flatbuffer storing the strings
  // may be released while the executable is still live.
  iree_host_size_t total_entry_point_name_chars = 0;
  IREE_TRACE({
    for (iree_host_size_t i = 0; i < entry_point_count; i++) {
      const char* entry_name = flatbuffers_string_vec_at(entry_points_vec, i);
      total_entry_point_name_chars += flatbuffers_string_len(entry_name);
    }
  });

  // Create the kernel library.
  iree_host_size_t total_size = sizeof(*executable) +
                                entry_point_count * sizeof(executable->entry_points[0]) +
                                total_entry_point_name_chars;
  iree_status_t status = iree_allocator_malloc(host_allocator, total_size, (void**)&executable);
  IREE_TRACE(char* string_table_buffer =
                 (char*)((char*)executable + sizeof(*executable) +
                         entry_point_count * sizeof(executable->entry_points[0])));
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_metal_kernel_library_vtable, &executable->resource);
    executable->host_allocator = host_allocator;
    executable->entry_point_count = entry_point_count;

    size_t shader_library_count = flatbuffers_string_vec_len(shader_libraries_vec);
    size_t shader_source_count = flatbuffers_string_vec_len(shader_sources_vec);

    // Try to load as Metal library first. Otherwise, compile each MSL source string into a
    // MTLLibrary and get the MTLFunction for the entry point to build the pipeline state object.
    // TODO(#14047): Enable async MSL compilation at runtime.

    MTLCompileOptions* compile_options = [MTLCompileOptions new];  // +1
    compile_options.languageVersion = MTLLanguageVersion3_0;

    for (size_t i = 0, e = iree_max(shader_library_count, shader_source_count); i < e; ++i) {
      id<MTLLibrary> library = nil;
      id<MTLFunction> function = nil;
      id<MTLComputePipelineState> pso = nil;

      flatbuffers_string_t source_code = NULL;
      flatbuffers_string_t entry_point = flatbuffers_string_vec_at(entry_points_vec, i);
      iree_string_view_t entry_point_view =
          iree_make_string_view(entry_point, flatbuffers_string_len(entry_point));

      if (shader_library_count != 0) {
        flatbuffers_string_t source_library = flatbuffers_string_vec_at(shader_libraries_vec, i);
        status = iree_hal_metal_load_mtllib(
            iree_make_const_byte_span(source_library, flatbuffers_string_len(source_library)),
            entry_point_view, device, &library);
      } else {
        source_code = flatbuffers_string_vec_at(shader_sources_vec, i);
        status = iree_hal_metal_compile_msl(
            iree_make_string_view(source_code, flatbuffers_string_len(source_code)),
            entry_point_view, device, compile_options, &library);
      }
      if (!iree_status_is_ok(status)) break;

      status = iree_hal_metal_create_pipline_object(library, entry_point_view, source_code, device,
                                                    &function, &pso);
      if (!iree_status_is_ok(status)) break;

      // Package required parameters for kernel launches for each entry point.
      iree_hal_metal_kernel_params_t* params = &executable->entry_points[i];
      params->library = library;
      params->function = function;
      params->pso = pso;
      params->threadgroup_size[0] = threadgroup_sizes_vec[i].x;
      params->threadgroup_size[1] = threadgroup_sizes_vec[i].y;
      params->threadgroup_size[2] = threadgroup_sizes_vec[i].z;
      params->layout = executable_params->pipeline_layouts[i];
      iree_hal_pipeline_layout_retain(params->layout);

      // Stash the entry point name in the string table for use when tracing.
      IREE_TRACE({
        iree_host_size_t entry_name_length = flatbuffers_string_len(entry_point);
        memcpy(string_table_buffer, entry_point, entry_name_length);
        params->function_name = iree_make_string_view(string_table_buffer, entry_name_length);
        string_table_buffer += entry_name_length;
      });
    }

    [compile_options release];  // -1
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_metal_kernel_library_destroy(iree_hal_executable_t* base_executable) {
  iree_hal_metal_kernel_library_t* executable = iree_hal_metal_kernel_library_cast(base_executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->entry_point_count; ++i) {
    iree_hal_metal_kernel_params_t* entry_point = &executable->entry_points[i];
    [entry_point->pso release];       // -1
    [entry_point->function release];  // -1
    [entry_point->library release];   // -1
    iree_hal_pipeline_layout_release(entry_point->layout);
  }
  iree_allocator_free(executable->host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_metal_kernel_library_entry_point_kernel_params(
    const iree_hal_executable_t* base_executable, int32_t entry_point,
    iree_hal_metal_kernel_params_t* out_params) {
  const iree_hal_metal_kernel_library_t* executable =
      iree_hal_metal_kernel_library_const_cast(base_executable);
  if (entry_point >= executable->entry_point_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "invalid entry point ordinal %d",
                            entry_point);
  }
  memcpy(out_params, &executable->entry_points[entry_point], sizeof(*out_params));
  return iree_ok_status();
}

static const iree_hal_executable_vtable_t iree_hal_metal_kernel_library_vtable = {
    .destroy = iree_hal_metal_kernel_library_destroy,
};
