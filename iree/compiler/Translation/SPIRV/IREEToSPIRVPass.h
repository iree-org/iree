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

//===- IREEToSPIRVPass.h ---------------------------------------*- C++//-*-===//
//
// Pass to translate iree executables for vulkan-spirv.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_TRANSLATION_SPIRV_IREETOSPIRVPASS_H
#define IREE_COMPILER_TRANSLATION_SPIRV_IREETOSPIRVPASS_H

#include "mlir/Pass/Pass.h"

// TODO(ravishankarm): Rename this file to something more meaningful.

namespace mlir {
class FuncOp;
namespace iree_compiler {

/// Generates a spirv::ModuleOp from the module within an IREE Executable with
/// target-config vulkan-spirv.
std::unique_ptr<OpPassBase<ModuleOp>> createIREEToSPIRVPass();

/// Performs analysis to compute affine maps that represent the index of the
/// elements of tensor values needed within a workitem.
std::unique_ptr<OpPassBase<FuncOp>> createIndexComputationPass();

// Legalizes integer width from i1, i8 and i64 types to i32 type.
std::unique_ptr<Pass> createAdjustIntegerWidthPass();

/// A pass that convertes entry functions for reductions into a form
/// that is convenient to lower to SPIR-V.
// TODO(ravishankarm) : This is a placeholder pass. Eventually this pass should
// not be needed.
std::unique_ptr<Pass> createPrepareReductionDispatchPass();

/// Method to legalize the apply function within a reduction dispatch. This is
/// not a pass since lowering of the entry function and the reduction apply
/// function has to happen simultaneously to avoid dangling symbols.
// TODO(ravishankarm) : Can probably fix that by adding an empty function and
// then materializing it. There is no need to do that right now, but maybe in
// the future.
LogicalResult lowerReductionApplyFunction(MLIRContext *context,
                                          ArrayRef<Operation *> fns);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_SPIRV_IREETOSPIRVPASS_H
