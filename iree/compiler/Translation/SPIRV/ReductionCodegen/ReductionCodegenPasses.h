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

//===- ReductionCodegenPasses.h --------------------------------*- C++//-*-===//
//
// Passes used for reduction kernel generation.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_TRANSLATION_SPIRV_REDUCETIONCODEGEN_REDUCTIONCODEGEN_H
#define IREE_COMPILER_TRANSLATION_SPIRV_REDUCETIONCODEGEN_REDUCTIONCODEGEN_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class FuncOp;
namespace iree_compiler {

/// Converts entry functions for reductions into a form convenient for SPIR-V
/// lowering.
// TODO(ravishankarm) : This is a placeholder pass. Eventually this
// pass should not be needed.
std::unique_ptr<Pass> createPrepareReductionDispatchPass();

/// Legalizes the apply function within a reduction dispatch. This is not a pass
/// since lowering of the entry function and the reduction apply function has to
/// happen simultaneously to avoid dangling symbols.
// TODO(ravishankarm) : Can probably fix that by adding an empty function and
// then materializing it. There is no need to do that right now, but maybe in
// the future.
LogicalResult lowerReductionApplyFunction(MLIRContext *context,
                                          ArrayRef<Operation *> fns);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_SPIRV_REDUCETIONCODEGEN_REDUCTIONCODEGEN_H
