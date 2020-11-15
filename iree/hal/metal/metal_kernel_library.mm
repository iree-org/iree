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

namespace iree {
namespace hal {
namespace metal {

// static
StatusOr<ref_ptr<MetalKernelLibrary>> MetalKernelLibrary::Create(
    id<MTLDevice> device, ExecutableCachingModeBitfield mode,
    const MetalExecutableDef& metal_executable_def) {
  IREE_TRACE_SCOPE0("MetalKernelLibrary::Create");
  if (!metal_executable_def.entry_points() || metal_executable_def.entry_points()->size() == 0) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "No entry points defined";
  }
  if (!metal_executable_def.threadgroup_sizes() ||
      metal_executable_def.threadgroup_sizes()->size() == 0) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "No threadgroup sizes present";
  }
  if (!metal_executable_def.shader_sources() ||
      metal_executable_def.shader_sources()->size() == 0) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "No MSL source string present";
  }

  const auto& entry_points = *metal_executable_def.entry_points();
  const auto& threadgroup_sizes = *metal_executable_def.threadgroup_sizes();
  const auto& msl_sources = *metal_executable_def.shader_sources();

  if (entry_points.size() != threadgroup_sizes.size() ||
      entry_points.size() != msl_sources.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Mismatch among the numbers of entry points, thread group sizes, and source strings";
  }

  // Compile each MSL source string into a MTLLibrary and get the MTLFunction for the entry point to
  // build the pipeline state object.

  absl::InlinedVector<id<MTLLibrary>, 1> libraries;
  absl::InlinedVector<KernelObjects, 1> kernel_objects;

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

  for (int i = 0; i < msl_sources.size(); ++i) {
    @autoreleasepool {
      NSError* error = nil;

      NSString* shader_source = [NSString stringWithCString:msl_sources[i]->c_str()
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

      NSString* entry_point = [NSString stringWithCString:entry_points[i]->c_str()
                                                 encoding:[NSString defaultCStringEncoding]];
      id<MTLFunction> function = [library newFunctionWithName:entry_point];
      if (!function) {
        NSLog(@"Failed to create MTLFunction");
#ifndef NDEBUG
        NSLog(@"Original MSL source: %@", shader_source);
#endif
        return InvalidArgumentErrorBuilder(IREE_LOC)
               << "Cannot find entry point '" << entry_points[i] << "' in shader source index "
               << i;
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

      kernel_objects.push_back(KernelObjects{function, *threadgroup_sizes[i], pso});
    }
  }

  return assign_ref(new MetalKernelLibrary([device retain], std::move(libraries),
                                           std::move(kernel_objects)));
}

MetalKernelLibrary::MetalKernelLibrary(id<MTLDevice> device,
                                       absl::InlinedVector<id<MTLLibrary>, 1> libraries,
                                       absl::InlinedVector<KernelObjects, 1> kernel_objects)
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

StatusOr<MetalThreadgroupSize> MetalKernelLibrary::GetThreadgroupSizeForEntryPoint(
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
