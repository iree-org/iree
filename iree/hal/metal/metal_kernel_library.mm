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
  if (!metal_executable_def.shader_sources() ||
      metal_executable_def.shader_sources()->size() == 0) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "No MSL source string present";
  }

  const auto& entry_points = *metal_executable_def.entry_points();
  const auto& msl_sources = *metal_executable_def.shader_sources();

  if (entry_points.size() != msl_sources.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Number of entry points and source strings mismatch";
  }

  // Compile each MSL source string into a MTLLibrary and get the MTLFunction for the entry point to
  // build the pipeline state object.

  absl::InlinedVector<id<MTLLibrary>, 1> libraries;
  absl::InlinedVector<id<MTLFunction>, 1> functions;
  absl::InlinedVector<id<MTLComputePipelineState>, 1> states;

  MTLCompileOptions* msl_compile_options = [MTLCompileOptions new];
  msl_compile_options.languageVersion = MTLLanguageVersion2_0;

  auto cleanup = MakeCleanup([&]() {
    for (id<MTLComputePipelineState> state : states) [state release];
    for (id<MTLFunction> function : functions) [function release];
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
      functions.push_back(function);

      id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:function
                                                                              error:&error];
      if (!pso) {
        NSLog(@"Failed to create MTLComputePipelineState: %@", error);
#ifndef NDEBUG
        NSLog(@"Original MSL source: %@", shader_source);
#endif
        return InvalidArgumentErrorBuilder(IREE_LOC) << "Invalid MSL source";
      }
      states.push_back(pso);
    }
  }

  std::string tag = metal_executable_def.tag() ? metal_executable_def.tag()->str() : "";
  return assign_ref(new MetalKernelLibrary([device retain], std::move(libraries),
                                           std::move(functions), std::move(states),
                                           std::move(tag)));
}

MetalKernelLibrary::MetalKernelLibrary(
    id<MTLDevice> device, absl::InlinedVector<id<MTLLibrary>, 1> libraries,
    absl::InlinedVector<id<MTLFunction>, 1> functions,
    absl::InlinedVector<id<MTLComputePipelineState>, 1> pipelines, std::string tag)
    : tag_(std::move(tag)),
      device_(device),
      libraries_(libraries),
      functions_(functions),
      pipelines_(pipelines) {}

MetalKernelLibrary::~MetalKernelLibrary() {
  IREE_TRACE_SCOPE0("MetalKernelLibrary::dtor");
  for (id<MTLComputePipelineState> pso : pipelines_) [pso release];
  for (id<MTLFunction> function : functions_) [function release];
  for (id<MTLLibrary> library : libraries_) [library release];
}

StatusOr<id<MTLComputePipelineState>> MetalKernelLibrary::GetPipelineStateForEntryPoint(
    int ordinal) const {
  if (ordinal < 0 || ordinal >= pipelines_.size()) {
    return OutOfRangeErrorBuilder(IREE_LOC) << "Invalid entry point ordinal: " << ordinal;
  }
  return pipelines_[ordinal];
}

}  // namespace metal
}  // namespace hal
}  // namespace iree
