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

//===- KernelDispatchUtils.h - Utilities for generating dispatch info -----===//
//
// This file declares utility functions that can be used to create information
// the dispatch on the host side needs to execute an entry point function, like
// the number of workgroups to use for launch, etc.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_CONVERSION_LINALGTOSPIRV_KERNELDISPATCHUTILS_H_
#define IREE_COMPILER_CONVERSION_LINALGTOSPIRV_KERNELDISPATCHUTILS_H_

#include "mlir/Support/LLVM.h"

namespace mlir {
class FuncOp;
class LogicalResult;
class PatternRewriter;
class ShapedType;
class Value;

namespace linalg {
class LinalgOp;
}

namespace iree_compiler {

/// Generates a function that computes the number of workgroups as
///  [ceil(`parallelLoopRange`[2] / `tileSizes`[2]),
///   ceil(`parallelLoopRange`[1] / `tileSizes`[1]),
///   ceil(`parallelLoopRange`[0] / `tileSizes`[0])]
/// where `parallelLoopRange` is the ranges of the parallel loops of `linalgOp`
/// distributed across workgroups.
LogicalResult createNumWorkgroupsFromResultShape(PatternRewriter &rewriter,
                                                 linalg::LinalgOp linalgOp,
                                                 FuncOp entryPointFn,
                                                 ArrayRef<int64_t> tileSizes);

/// Generates a function that computes the number of workgroups as
///  ceil(`parallelLoopRange`[0] * `parallelLoopRange`[1] * ... *
///       `parallelLoopRange`[n-1]  /  `workgroupSizeX`)
/// where `parallelLoopRange` is the ranges of the parallel loops of `linalgOp`
/// distributed across workgroups.
LogicalResult createNumWorkgroupsFromLinearizedResultShape(
    PatternRewriter &rewriter, linalg::LinalgOp linalgOp, FuncOp entryPointFn,
    int64_t workgroupSizeX);

/// For a given `entryPointFn` return the function that computes the number of
/// workgroups to use at launch time.
FuncOp getNumWorkgroupsFn(FuncOp entryPointFn);

}  // namespace iree_compiler
}  // namespace mlir
#endif  // IREE_COMPILER_CONVERSION_LINALGTOSPIRV_DISPATCHUTILS_H_
