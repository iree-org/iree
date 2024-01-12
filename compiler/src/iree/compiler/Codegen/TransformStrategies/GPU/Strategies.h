// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_STRATEGIES_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_STRATEGIES_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
class ImplicitLocOpBuilder;
class Value;
} // namespace mlir

namespace mlir::iree_compiler::gpu {

/// Forward declarations of all supported strategies.
class BatchMatmulStrategy;
class MatmulStrategy;
class PadStrategy;
class SmallReductionStrategy;
class StagedReductionStrategy;

static constexpr int64_t kCudaWarpSize = 32;
static constexpr int64_t kCudaMaxNumThreads = 1024;

/// Struct for representing supported WMMA/Cooperative Matrix configurations.
/// This is a reflection of SPIRV_CooperativeMatrixPropertiesNVAttr.
struct MMAConfig {
  int64_t m;
  int64_t n;
  int64_t k;
  Type aType;
  Type bType;
  Type cType;
};

/// Placeholder for some hardware model proxy that contains relevant information
/// to configure the strategies. In the future, this will need to be
/// driven by some contract with the runtime.
struct GPUModel {
  static constexpr llvm::StringLiteral kDefaultGPU = "DefaultGPU";
  llvm::StringRef model = kDefaultGPU;
  /// TODO: Support a range of subgroup sizes.
  int64_t subgroupSize = kCudaWarpSize;
  std::optional<int> minSubgroupSize = std::nullopt;
  std::optional<int> maxSubgroupSize = std::nullopt;
  int64_t maxWorkGroupInvocations = kCudaMaxNumThreads;
  int64_t maxWorkGroupSize[3] = {1024, 1024, 64};
  bool hasWarpShuffle = false;
  bool hasTF32TensorCore = false;
  bool hasMmaSync = false;
  SmallVector<MMAConfig> supportedWMMAConfigs = {};
};

//===--------------------------------------------------------------------===//
// GPU strategy base.
//===--------------------------------------------------------------------===//
/// Basic structure to hold target specific information needed for all gpu
/// strategies. Certain quantities that can be dynamically selected, such as
/// subgroup size, will need to be configured with some contract with the
/// runtime.
struct GPUStrategy {
  /// TODO: Configure subgroup size with the strategy and return the selected
  /// size to the target (i.e. LLVMGPU or SPIR-V).
  GPUStrategy(const GPUModel &gpuModel) : subgroupSize(gpuModel.subgroupSize) {}
  /// TODO: Add other quantities relevant to strategy builders.
  int64_t subgroupSize;
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
// Batch matmul strategies.
//===--------------------------------------------------------------------===//
/// Entry point to build the transform IR corresponding to an FMA-based strategy
/// for linalg.fill + linalg.batch_matmul.
void buildBatchMatmulStrategy(ImplicitLocOpBuilder &b, Value variantH,
                              const BatchMatmulStrategy &strategy);

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

//===----------------------------------------------------------------------===//
// Higher-level strategy creation APIs, these should favor
// user-friendliness.
//===----------------------------------------------------------------------===//

/// Try to find an exisiting transform dialect strategy for a given entry point.
LogicalResult matchAndSetTransformStrategy(func::FuncOp entryPoint,
                                           Operation *op,
                                           const GPUModel &gpuModel);

} // namespace mlir::iree_compiler::gpu

#endif // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_STRATEGIES_H_
