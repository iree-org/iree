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

#ifndef IREE_COMPILER_TRANSLATION_SPIRV_LINALGTOSPIRV_LOWERTOSPIRV_H
#define IREE_COMPILER_TRANSLATION_SPIRV_LINALGTOSPIRV_LOWERTOSPIRV_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

/// Populates passes needed to lower a XLA HLO op to SPIR-V dialect through
/// Linalg dialect. The pass manager `pm` in here operate on the module within
/// the IREE::HAL::ExecutableOp. The `workGroupSize` can be used to control the
/// work group size used in the code-generation and is intended for testing
/// purposes only. The pass pipeline will set an appropriate workgroup size.
void addHLOToLinalgToSPIRVPasses(OpPassManager &pm,
                                 ArrayRef<int64_t> workGroupSize = {});

/// Populates passes needed to lower a linalg op (on buffers) to SPIR-V dialect.
/// The pass manager `pm` in here operate on the module within the
/// IREE::HAL::ExecutableOp. The `workGroupSize` can be used to control the work
/// group size used in the code-generation and is intended for testing purposes
/// only. The pass pipeline will set an appropriate workgroup size.
void addLinalgToSPIRVPasses(OpPassManager &pm,
                            ArrayRef<int64_t> workGroupSize = {});

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_SPIRV_XLATOLINALGSPIRV_LOWERTOSPIRV_H
