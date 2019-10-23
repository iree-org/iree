// Copyright 2019 Google LLC
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

#ifndef IREE_COMPILER_TRANSLATION_SPIRV_EMBEDDEDKERNELS_H_
#define IREE_COMPILER_TRANSLATION_SPIRV_EMBEDDEDKERNELS_H_

#include "compiler/IR/StructureOps.h"
#include "flatbuffers/flatbuffers.h"
#include "mlir/Support/LogicalResult.h"
#include "schemas/spirv_executable_def_generated.h"

namespace mlir {
namespace iree_compiler {

// Tries to match the |executableOp| against an embedded kernel and if matched
// will populate |out_def| with the kernel.
// Returns true if the kernel matched and was populated.
bool tryEmbeddedKernelRewrite(IREE::ExecutableOp executableOp,
                              iree::SpirVExecutableDefT* out_def);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_SPIRV_EMBEDDEDKERNELS_H_
