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

#ifndef IREE_COMPILER_CONVERSION_CODEGENUTILS_GETNUMWORKGROUPS_H_
#define IREE_COMPILER_CONVERSION_CODEGENUTILS_GETNUMWORKGROUPS_H_

#include <array>
#include <cstdint>

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {

/// Generates a function that computes the number of workgroups as
///  [ceil(`loopUpperBounds`[2] / `tileSizes`[2]),
///   ceil(`loopUpperBounds`[1] / `tileSizes`[1]),
///   ceil(`loopUpperBounds`[0] / `tileSizes`[0])]
/// where `loopUpperBounds` is the ranges of the parallel loops of `linalgOp`
///  distributed across workgroups. `distributedLoops` are the loop dimensions
///  that are distributed.
LogicalResult createNumWorkgroupsFromResultShape(
    OpBuilder &builder, linalg::LinalgOp linalgOp, FuncOp entryPointFn,
    llvm::StringRef numWorkgroupsFnAttr, llvm::ArrayRef<int64_t> tileSizes,
    llvm::ArrayRef<unsigned> distributedLoops);

/// Generates a function that computes the number of workgroups as
///  [ceil(`loopUpperBounds`[2] / `tileSizes`[2]),
///   ceil(`loopUpperBounds`[1] / `tileSizes`[1]),
///   ceil(`loopUpperBounds`[0] / `tileSizes`[0])]
/// where `loopUpperBounds` is the ranges of the parallel loops of `linalgOp`
/// distributed across workgroups. Assumes that upto 3 outer parallel loops of
/// the `linalgOp` are distributed.
LogicalResult createNumWorkgroupsFromResultShape(
    PatternRewriter &rewriter, linalg::LinalgOp linalgOp, FuncOp entryPointFn,
    llvm::StringRef numWorkgroupsFnAttr, llvm::ArrayRef<int64_t> tileSizes);

/// Generates a function that computes the number of workgroups as
///  ceil(`loopUpperBounds`[0] * `loopUpperBounds`[1] * ... *
///       `loopUpperBounds`[n-1]  /  `workgroupSizeX`)
/// where `loopUpperBounds` is the ranges of the parallel loops of `linalgOp`
/// distributed across workgroups.
LogicalResult createNumWorkgroupsFromLinearizedResultShape(
    ConversionPatternRewriter &rewriter, linalg::LinalgOp linalgOp,
    FuncOp entryPointFn, llvm::StringRef numWorkgroupsFnAttr,
    int64_t workgroupSizeX);

/// For a given `entryPointFn` return the function that computes the number of
/// workgroups to use at launch time.
FuncOp getNumWorkgroupsFn(FuncOp entryPointFn,
                          llvm::StringRef numWorkgroupsFnAttr);

/// The codegeneration emits a function `numWorkgroupsFn` for each entry point
/// function. This function has arguments the !shapex.ranked_shape for all the
/// input and output shaped types. Using this the function returns the number of
/// workgroups to use. To use this function on the host side, generate the
/// !shapex.ranked_shape values that describe the shape of the inputs and
/// outputs of the dispatch region and "inline" the function body.
std::array<Value, 3> calculateWorkgroupCountFromNumWorkgroupsFn(
    Location loc, FuncOp numWorkgroupsFn,
    mlir::iree_compiler::IREE::HAL::InterfaceOp interface,
    llvm::ArrayRef<
        llvm::Optional<mlir::iree_compiler::IREE::HAL::TensorRewriteAdaptor>>
        operands,
    llvm::ArrayRef<
        llvm::Optional<mlir::iree_compiler::IREE::HAL::TensorRewriteAdaptor>>
        results,
    ConversionPatternRewriter &rewriter);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_CODEGENUTILS_GETNUMWORKGROUPS_H_
