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

#ifndef MLIR_EDGE_BENCHMARKS_STRATEGIES_WORKGROUPCALULCATION_H_
#define MLIR_EDGE_BENCHMARKS_STRATEGIES_WORKGROUPCALULCATION_H_

#include <cstdint>

namespace llvm {
class StringRef;
template <typename T>
class ArrayRef;
template <typename T>
class Optional;
}  // namespace llvm

namespace mlir {
class Location;
class FuncOp;
class LogicalResult;
class PatternRewriter;
class ConversionPatternRewriter;
class Value;
namespace linalg {
class LinalgOp;
}  // namespace linalg

namespace iree_compiler {
namespace IREE {
namespace HAL {
class InterfaceOp;
class TensorRewriteAdaptor;
}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
namespace iree_compiler {
namespace utils {
/// Generates a function that computes the number of workgroups as
///  [ceil(`parallelLoopRange`[2] / `tileSizes`[2]),
///   ceil(`parallelLoopRange`[1] / `tileSizes`[1]),
///   ceil(`parallelLoopRange`[0] / `tileSizes`[0])]
/// where `parallelLoopRange` is the ranges of the parallel loops of `linalgOp`
/// distributed across workgroups.
LogicalResult createNumWorkgroupsFromResultShape(
    PatternRewriter &rewriter, linalg::LinalgOp linalgOp, FuncOp entryPointFn,
    llvm::StringRef numWorkgroupsFnAttr, llvm::ArrayRef<int64_t> tileSizes);

/// Generates a function that computes the number of workgroups as
///  ceil(`parallelLoopRange`[0] * `parallelLoopRange`[1] * ... *
///       `parallelLoopRange`[n-1]  /  `workgroupSizeX`)
/// where `parallelLoopRange` is the ranges of the parallel loops of `linalgOp`
/// distributed across workgroups.
LogicalResult createNumWorkgroupsFromLinearizedResultShape(
    PatternRewriter &rewriter, linalg::LinalgOp linalgOp, FuncOp entryPointFn,
    llvm::StringRef numWorkgroupsFnAttr, int64_t workgroupSizeX);

/// For a given `entryPointFn` return the function that computes the number of
/// workgroups to use at launch time.
FuncOp getNumWorkgroupsFn(FuncOp entryPointFn,
                          llvm::StringRef numWorkgroupsFnAttr);

LogicalResult createNumWorkgroupsFromLinearizedResultShape(
    PatternRewriter &rewriter, linalg::LinalgOp linalgOp, FuncOp entryPointFn,
    llvm::StringRef numWorkgroupsFnAttr, int64_t workgroupSizeX);

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

}  // namespace utils
}  // namespace iree_compiler
}  // namespace mlir

#endif  // MLIR_EDGE_BENCHMARKS_STRATEGIES_WORKGROUPCALULCATION_H_
