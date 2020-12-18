// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/metal/metal_kernel_library.h"

#include "iree/base/memory.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"

// NOTE: starting to port this to ObjC.

// Verifies the structure of the flatbuffer so that we can avoid doing so during
// runtime. There are still some conditions we must be aware of (such as omitted
// names on functions with internal linkage), however we shouldn't need to
// bounds check anything within the flatbuffer after this succeeds.
static iree_status_t iree_hal_metal_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer data is not present or less than 16 bytes (%zu total)",
                            flatbuffer_data.data_length);
  }

  // Run flatcc generated verification. This ensures all pointers are in-bounds
  // and that we can safely walk the file, but not that the actual contents of
  // the flatbuffer meet our expectations.
  int verify_ret =
      iree_MetalExecutableDef_verify_as_root(flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_MetalExecutableDef_table_t executable_def =
      iree_MetalExecutableDef_as_root(flatbuffer_data.data);

  flatbuffers_string_vec_t entry_points_vec =
      iree_MetalExecutableDef_entry_points_get(executable_def);
  size_t entry_point_count = flatbuffers_string_vec_len(entry_points_vec);
  for (size_t i = 0; i < entry_point_count; ++i) {
    if (!flatbuffers_string_len(flatbuffers_string_vec_at(entry_points_vec, i))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable entry point %zu has no name", i);
    }
  }

  iree_MetalThreadgroupSize_vec_t threadgroup_sizes_vec =
      iree_MetalExecutableDef_threadgroup_sizes(executable_def);
  size_t threadgroup_size_count = iree_MetalThreadgroupSize_vec_len(threadgroup_sizes_vec);
  if (!threadgroup_size_count) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "No threadgroup sizes present";
  }

  flatbuffers_string_vec_t shader_sources_vec =
      iree_MetalExecutableDef_shader_sources_get(executable_def);
  size_t shader_source_count = flatbuffers_string_vec_len(shader_sources_vec);
  for (size_t i = 0; i < shader_source_count; ++i) {
    if (!flatbuffers_string_len(flatbuffers_string_vec_at(shader_sources_vec, i))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "executable shader source %zu is empty",
                              i);
    }
  }

  if (entry_point_count != threadgroup_size_count || entry_point_count != shader_source_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "mismatch among the numbers of entry points (%zu), thread group sizes "
                            "(%zu), and source strings (%zu)",
                            entry_point_count, threadgroup_size_count, shader_source_count);
  }

  return iree_ok_status();
}

namespace iree {
namespace hal {
namespace metal {

// static
StatusOr<ref_ptr<MetalKernelLibrary>> MetalKernelLibrary::Create(id<MTLDevice> device,
                                                                 iree_hal_executable_caching_mode_t mode,
                                                                 iree_const_byte_span_t executable_data) {
  IREE_TRACE_SCOPE0("MetalKernelLibrary::Create");

  // Verify and fetch the executable flatbuffer wrapper.
  iree_const_byte_span_t executable_data =
      iree_make_const_byte_span(spec.executable_data.data(), spec.executable_data.size());
  IREE_RETURN_IF_ERROR(iree_hal_metal_executable_flatbuffer_verify(executable_data));
  iree_MetalExecutableDef_table_t executable_def =
      iree_MetalExecutableDef_as_root(executable_data.data);

  flatbuffers_string_vec_t entry_points_vec =
      iree_MetalExecutableDef_entry_points_get(executable_def);
  iree_MetalThreadgroupSize_vec_t threadgroup_sizes_vec =
      iree_MetalExecutableDef_threadgroup_sizes(executable_def);
  flatbuffers_string_vec_t shader_sources_vec =
      iree_MetalExecutableDef_shader_sources_get(executable_def);

  // Compile each MSL source string into a MTLLibrary and get the MTLFunction for the entry point to
  // build the pipeline state object.

  absl::InlinedVector<id<MTLLibrary>, 4> libraries;
  absl::InlinedVector<KernelObjects, 4> kernel_objects;

  MTLCompileOptions* msl_compile_options = [MTLCompileOptions new];
  msl_compile_options.languageVersion = MTLLanguageVersion2_0;

  auto cleanup = MakeCleanup([&]() {
    for (const auto& kernel : kernel_objects) {
      [kernel.pipeline_state release];
      [kernel.function release];
    }
    for (id<MTLLibrary> library : libraries) [library release];
    [msl_compile_options release];
  });

  // TODO(antiagainst): We are performing synchronous compilation at runtime here. This is good for
  // debugging purposes but bad for performance. Enable offline compilation and make that as the
  // default.

  for (size_t entry_ordinal = 0; entry_ordinal < flatbuffers_string_vec_len(shader_sources_vec);
       ++entry_ordinal) {
    flatbuffers_string_t entry_point = flatbuffers_string_vec_at(entry_points_vec, entry_ordinal);
    @autoreleasepool {
      NSError* error = nil;

      NSString* shader_source =
          [NSString stringWithCString:flatbuffers_string_vec_at(shader_sources_vec, entry_ordinal)
                             encoding:[NSString defaultCStringEncoding]];
      id<MTLLibrary> library = [device newLibraryWithSource:shader_source
                                                    options:msl_compile_options
                                                      error:&error];
      if (!library) {
        NSLog(@"Failed to create MTLLibrary: %@", error);
#ifndef NDEBUG
        NSLog(@"Original MSL source: %@", shader_source);
#endif
        return InvalidArgumentErrorBuilder(IREE_LOC) << "Invalid MSL source";
      }
      libraries.push_back(library);

      id<MTLFunction> function = [library
          newFunctionWithName:[NSString stringWithCString:entry_point
                                                 encoding:[NSString defaultCStringEncoding]]];
      if (!function) {
        NSLog(@"Failed to create MTLFunction");
#ifndef NDEBUG
        NSLog(@"Original MSL source: %@", shader_source);
#endif
        return InvalidArgumentErrorBuilder(IREE_LOC)
               << "Cannot find entry point '" << entry_point << "' in shader source index "
               << entry_ordinal;
      }

      id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:function
                                                                              error:&error];
      if (!pso) {
        NSLog(@"Failed to create MTLComputePipelineState: %@", error);
#ifndef NDEBUG
        NSLog(@"Original MSL source: %@", shader_source);
#endif
        return InvalidArgumentErrorBuilder(IREE_LOC) << "Invalid MSL source";
      }

      kernel_objects.push_back(
          KernelObjects{function, {static_cast<uint32_t>(iree_MetalThreadgroupSize__size())}, pso});
    }
  }

  return assign_ref(
      new MetalKernelLibrary([device retain], std::move(libraries), std::move(kernel_objects)));
}

MetalKernelLibrary::MetalKernelLibrary(id<MTLDevice> device,
                                       absl::InlinedVector<id<MTLLibrary>, 4> libraries,
                                       absl::InlinedVector<KernelObjects, 4> kernel_objects)
    : device_(device),
      libraries_(std::move(libraries)),
      kernel_objects_(std::move(kernel_objects)) {}

MetalKernelLibrary::~MetalKernelLibrary() {
  IREE_TRACE_SCOPE0("MetalKernelLibrary::dtor");
  for (const auto& kernel : kernel_objects_) {
    [kernel.pipeline_state release];
    [kernel.function release];
  }
  for (id<MTLLibrary> library : libraries_) [library release];
}

StatusOr<id<MTLFunction>> MetalKernelLibrary::GetKernelForEntryPoint(int ordinal) const {
  if (ordinal < 0 || ordinal >= kernel_objects_.size()) {
    return OutOfRangeErrorBuilder(IREE_LOC) << "Invalid entry point ordinal: " << ordinal;
  }
  return kernel_objects_[ordinal].function;
}

StatusOr<iree_MetalThreadgroupSize_t> MetalKernelLibrary::GetThreadgroupSizeForEntryPoint(
    int ordinal) const {
  if (ordinal < 0 || ordinal >= kernel_objects_.size()) {
    return OutOfRangeErrorBuilder(IREE_LOC) << "Invalid entry point ordinal: " << ordinal;
  }
  return kernel_objects_[ordinal].threadgroup_size;
}

StatusOr<id<MTLComputePipelineState>> MetalKernelLibrary::GetPipelineStateForEntryPoint(
    int ordinal) const {
  if (ordinal < 0 || ordinal >= kernel_objects_.size()) {
    return OutOfRangeErrorBuilder(IREE_LOC) << "Invalid entry point ordinal: " << ordinal;
  }
  return kernel_objects_[ordinal].pipeline_state;
}

}  // namespace metal
}  // namespace hal
}  // namespace iree
