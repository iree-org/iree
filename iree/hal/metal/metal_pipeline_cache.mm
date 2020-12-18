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

#include "iree/hal/metal/metal_pipeline_cache.h"

#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/executable_format.h"
#include "iree/hal/metal/metal_kernel_library.h"

namespace iree {
namespace hal {
namespace metal {

MetalPipelineCache::MetalPipelineCache(id<MTLDevice> device) : metal_device_([device retain]) {}

MetalPipelineCache::~MetalPipelineCache() { [metal_device_ release]; }

bool MetalPipelineCache::CanPrepareFormat(ExecutableFormat format) const {
  return format == kExecutableFormatMetal;
}

StatusOr<ref_ptr<Executable>> MetalPipelineCache::PrepareExecutable(
    ExecutableLayout* executable_layout, ExecutableCachingModeBitfield mode,
    iree_const_byte_span_t executable_data) {
  IREE_TRACE_SCOPE0("MetalPipelineCache::PrepareExecutable");

  // Create the Metal library (which may itself own many pipeline states).
  IREE_ASSIGN_OR_RETURN(auto executable, MetalKernelLibrary::Create(metal_device_, mode, executable_data));

  return executable;
}

}  // namespace metal
}  // namespace hal
}  // namespace iree
