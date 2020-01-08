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

//===- IREECodegenUtils.h --------------------------------------*- C++//-*-===//
//
// Utility functions used during code-generation (for SPIR-V)
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_UTILS_IREECODEGENUTILS_H
#define IREE_COMPILER_UTILS_IREECODEGENUTILS_H

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/IR/Function.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {

/// Gets the launch size associated with the dispatch function.
LogicalResult getLaunchSize(FuncOp funcOp,
                            SmallVectorImpl<int64_t> &launchSize);

/// Gets the workgroup size.
LogicalResult getWorkGroupSize(FuncOp funcOp,
                               SmallVectorImpl<int32_t> &workGroupSize);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_IREECODEGENUTILS_H
