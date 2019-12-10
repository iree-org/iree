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

namespace mlir {
namespace iree_compiler {

// Generates a spirv::ModuleOp from the module within an IREE Executable with
// target-config vulkan-spirv.
std::unique_ptr<OpPassBase<ModuleOp>> createIREEToSPIRVPass();

// Performs analysis to compute affine maps that represent the index of the
// elements of tensor values needed within a workitem.
std::unique_ptr<OpPassBase<FuncOp>> createIndexComputationPass();

// Legalizes integer width from i1 and i64 types to i8 and i32 types
// respectively.
std::unique_ptr<Pass> createAdjustIntegerWidthPass();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_SPIRV_IREETOSPIRVPASS_H
