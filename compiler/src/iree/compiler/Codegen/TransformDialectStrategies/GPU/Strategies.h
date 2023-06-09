// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_STRATEGIES_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_STRATEGIES_H_

#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/MatmulTensorCoreStrategy.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/PadStrategy.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/SmallReductionStrategy.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/StagedReductionStrategy.h"

namespace mlir {
class ImplicitLocOpBuilder;
class Value;
namespace iree_compiler {
namespace gpu {

/// Placeholder for some hardware model proxy that contains relevant information
/// to configure the reduction strategy. In the future, this will need to be
/// driven by some contract with the runtime.
struct GPUModel {
  static constexpr StringLiteral kDefaultGPU = "DefaultGPU";
  StringRef model = kDefaultGPU;
  bool hasWarpShuffle = false;
  bool hasTF32TensorCore = false;
};

//===--------------------------------------------------------------------===//
// Matmul strategies.
//===--------------------------------------------------------------------===//
/// Entry point to build the transform IR corresponding to a tensorcore-based
/// strategy for linalg.fill + linalg.matmul on f32.
/// Does not support leading or trailing operations atm.
void buildMatmulTensorCoreStrategy(ImplicitLocOpBuilder &b, Value variantH,
                                   const MatmulStrategy &strategy);

//===--------------------------------------------------------------------===//
// Pad strategies.
//===--------------------------------------------------------------------===//
/// Entry point to build the transform IR corresponding to a simple pad
/// strategy.
/// Does not support leading or trailing operations atm.
void buildPadStrategy(ImplicitLocOpBuilder &b, Value variantH,
                      const PadStrategy &strategy);

//===--------------------------------------------------------------------===//
// Reduction strategies.
//===--------------------------------------------------------------------===//
/// Structure to hold a summary of HW-derived properties to configure the
/// reduction strategy.
/// The objective of this struct is to act as a minimal summary of key
/// properties derived from the hardware (e.g. by an oracle) and that are
/// sufficient to steer the strategy to produce a good version.
/// These can be thought of as latent variables or embeddings that directly
/// control the strategy and can be derived from the hardware by some procedure.
enum class ReductionStrategy { Small, Staged };
struct ReductionConfig {
  int64_t maxNumThreads;
  int64_t vectorSize;
  ReductionStrategy strategy;
};

/// Entry point to build the transform IR corresponding to a staged reduction
/// strategy.
/// This is used for mapping a N-D parallel, 1-D reduction operation with a
/// small reduction on which the default staged reduction strategy is otherwise
/// inefficient.
/// The 1-D reduction dimensions must be in the most minor dimension.
/// Supports an optional leading and an optional trailing elementwise operation.
void buildSmallReductionStrategy(ImplicitLocOpBuilder &b, Value variantH,
                                 const SmallReductionStrategy &strategy);

/// Entry point to build the transform IR corresponding to a staged reduction
/// strategy.
/// This is used for mapping a N-D parallel, 1-D reduction operation.
/// The 1-D reduction dimensions must be in the most minor dimension.
/// Supports an optional leading and an optional trailing elementwise operation.
void buildStagedReductionStrategy(ImplicitLocOpBuilder &b, Value variantH,
                                  const StagedReductionStrategy &strategy);

}  // namespace gpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_STRATEGIES_H_
