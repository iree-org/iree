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
#include "iree/hal/cc/executable.h"
#include "iree/hal/cc/executable_cache.h"

// flatcc schemas:
#include "iree/base/flatcc.h"
#include "iree/schemas/metal_executable_def_builder.h"
#include "iree/schemas/metal_executable_def_reader.h"
#include "iree/schemas/metal_executable_def_verifier.h"

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
      id<MTLDevice> device, iree_hal_executable_caching_mode_t mode,
      iree_const_byte_span_t executable_data);
  ~MetalKernelLibrary() override;

  // Returns the MTLFunction for the entry point with the given |ordinal|.
  StatusOr<id<MTLFunction>> GetKernelForEntryPoint(int ordinal) const;

  // Returns the threadgroup size for the entry point with the given |ordinal|.
  StatusOr<iree_MetalThreadgroupSize_t> GetThreadgroupSizeForEntryPoint(
      int ordinal) const;

  // Returns the pipeline state object for the entry point with the given
  // |ordinal|.
  StatusOr<id<MTLComputePipelineState>> GetPipelineStateForEntryPoint(
      int ordinal) const;

 private:
  struct KernelObjects {
    id<MTLFunction> function;
    iree_MetalThreadgroupSize_t threadgroup_size;
    // Baked pipeline state object.
    id<MTLComputePipelineState> pipeline_state;
  };

  // Creates a MetalKernelLibrary assuming all Metal objects are already
  // retained before passing in.
  MetalKernelLibrary(id<MTLDevice> device,
                     absl::InlinedVector<id<MTLLibrary>, 4> libraries,
                     absl::InlinedVector<KernelObjects, 4> kernel_objects);

  id<MTLDevice> device_;

  absl::InlinedVector<id<MTLLibrary>, 4> libraries_;
  absl::InlinedVector<KernelObjects, 4> kernel_objects_;
};

}  // namespace metal
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_METAL_METAL_KERNEL_LIBRARY_H_
