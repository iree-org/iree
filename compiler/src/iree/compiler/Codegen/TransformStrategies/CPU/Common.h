// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_CPU_COMMON_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_CPU_COMMON_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::iree_compiler::cpu {

//===----------------------------------------------------------------------===//
// Mid-level problem-specific strategy builder APIs, follow MLIR-style builders.
//===----------------------------------------------------------------------===//
/// Take care of the last common steps in a CPU strategy (i.e. vectorize,
/// bufferize, maps to blocks/workgroups and lower vectors).
/// Return the handles to the updated variant and the func::FuncOp ops under
/// the variant op.
// TODO: pass control to LowerVectorsOp once the builder allows it.
std::pair<Value, Value> buildCommonTrailingStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const vector::LowerVectorsOptions &lowerVectorsOpts);

//===----------------------------------------------------------------------===//
// Higher-level problem-specific strategy creation APIs, these should favor
// user-friendliness.
//===----------------------------------------------------------------------===//
/// Placeholder for some hardware model proxy that contains relevant information
/// to configure the reduction strategy. In the future, this will need to be
/// driven by some contract with the runtime.
struct CPUModel {
  static constexpr StringLiteral kDefaultCPU = "DefaultCPU";
  StringRef model = kDefaultCPU;
};

/// Map an N-D parallel, 1-D reduction operation with optional leading and
/// optional trailing elementwise operations.
/// The 1-D reduction dimension must be in the most minor dimension.
/// The innermost dimensions of the leading and trailing operations must be most
/// minor along all accesses.
/// Return failure if matching fails.
/// On a successful match, configure a reduction strategy based on a proxy model
/// of the hardware and construct transform dialect IR that implements the
/// reduction strategy. The transform dialect IR is added in a top-level
/// ModuleOp after the `entryPoint` func::FuncOp.
LogicalResult matchAndSetReductionStrategy(func::FuncOp entryPoint,
                                           linalg::LinalgOp op,
                                           const CPUModel &cpuModel);

} // namespace mlir::iree_compiler::cpu

#endif // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_CPU_COMMON_H_
