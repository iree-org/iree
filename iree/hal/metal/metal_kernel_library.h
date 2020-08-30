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

#ifndef IREE_HAL_METAL_METAL_KERNEL_LIBRARY_H_
#define IREE_HAL_METAL_METAL_KERNEL_LIBRARY_H_

#import <Metal/Metal.h>

#include <string>

#include "absl/container/inlined_vector.h"
#include "iree/base/status.h"
#include "iree/hal/executable.h"
#include "iree/hal/executable_cache.h"
#include "iree/hal/executable_spec.h"
#include "iree/schemas/metal_executable_def_generated.h"

namespace iree {
namespace hal {
namespace metal {

// An executable implementation for Metal that wraps MTLLibrary and MTLFunction.
//
// Metal represents compute kernels as MTLFunctions. MTLLibrary is just an
// allocation of MTLFunctions. One creates a MTLComputePipelineState from a
// MTLFunction and uses the pipeline state for creating compute pipelines.
// This class bundles all the necesary Metal objects for getting pipeline state
// objects for a compute kernel.
class MetalKernelLibrary final : public Executable {
 public:
  static StatusOr<ref_ptr<MetalKernelLibrary>> Create(
      id<MTLDevice> device, ExecutableCachingModeBitfield mode,
      const MetalExecutableDef& metal_executable_def);
  ~MetalKernelLibrary() override;

  bool supports_debugging() const override { return false; }

  // Returns the MTLFunction for the entry point with the given |ordinal|.
  StatusOr<id<MTLFunction>> GetKernelForEntryPoint(int ordinal) const;

  // Returns the threadgroup size for the entry point with the given |ordinal|.
  StatusOr<MetalThreadgroupSize> GetThreadgroupSizeForEntryPoint(
      int ordinal) const;

  // Returns the pipeline state object for the entry point with the given
  // |ordinal|.
  StatusOr<id<MTLComputePipelineState>> GetPipelineStateForEntryPoint(
      int ordinal) const;

 private:
  struct KernelObjects {
    id<MTLFunction> function;
    MetalThreadgroupSize threadgroup_size;
    // Baked pipeline state object.
    id<MTLComputePipelineState> pipeline_state;
  };

  // Creates a MetalKernelLibrary assuming all Metal objects are already
  // retained before passing in.
  MetalKernelLibrary(id<MTLDevice> device,
                     absl::InlinedVector<id<MTLLibrary>, 1> libraries,
                     absl::InlinedVector<KernelObjects, 1> kernel_objects,
                     std::string tag);

  // Tag coming from Metal executable FlatBuffer.
  std::string tag_;

  id<MTLDevice> device_;

  absl::InlinedVector<id<MTLLibrary>, 1> libraries_;
  absl::InlinedVector<KernelObjects, 1> kernel_objects_;
};

}  // namespace metal
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_METAL_METAL_KERNEL_LIBRARY_H_
