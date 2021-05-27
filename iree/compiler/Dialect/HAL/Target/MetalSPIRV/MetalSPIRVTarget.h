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

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_METALSPIRV_METALSPIRVTARGET_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_METALSPIRV_METALSPIRVTARGET_H_

#include <functional>

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Options controlling SPIR-V compilation for Metal.
struct MetalSPIRVTargetOptions {
  // TODO(antiagainst): Metal GPU family
};

// Returns a MetalSPIRVTargetOptions struct initialized with Metal/SPIR-V
// related command-line flags.
MetalSPIRVTargetOptions getMetalSPIRVTargetOptionsFromFlags();

// Registers the Metal/SPIR-V backends.
void registerMetalSPIRVTargetBackends(
    std::function<MetalSPIRVTargetOptions()> queryOptions);

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_METALSPIRV_METALSPIRVTARGET_H_
