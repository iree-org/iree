// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TransformDialectStrategiesGPU.h"

#include <numeric>
#include <type_traits>

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/Common/TransformDialectStrategies.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

// TODO: significantly better namespacing.
using iree_compiler::IREE::transform_dialect::ApplyPatternsOp;
using iree_compiler::IREE::transform_dialect::ConfigExtractPart;
using iree_compiler::IREE::transform_dialect::ForeachThreadToWorkgroupOp;
using iree_compiler::IREE::transform_dialect::IREEBufferizeOp;
using iree_compiler::IREE::transform_dialect::
    IREEEraseHALDescriptorTypeFromMemRefOp;
using iree_compiler::IREE::transform_dialect::
    MapNestedForeachThreadToGpuThreadsOp;
using iree_compiler::IREE::transform_dialect::
    TileToForeachThreadAndWorkgroupCountRegionOp;
using iree_compiler::IREE::transform_dialect::VectorToWarpExecuteOnLane0Op;
using iree_compiler::IREE::transform_dialect::VectorWarpDistributionOp;
using transform::FuseIntoContainingOp;
using transform::MatchOp;
using transform::MergeHandlesOp;
using transform::PrintOp;
using transform::SequenceOp;
using transform::SplitHandlesOp;
using transform::SplitReductionOp;
using transform::TileToForeachThreadOp;
using transform::VectorizeOp;
using transform_ext::AllDims;
using transform_ext::IsPermutation;
using transform_ext::m_StructuredOp;
using transform_ext::NumEqualsTo;
using transform_ext::ShapeKind;
using transform_ext::StructuredOpMatcher;

/// Matches `args` within `targetH` and unpacks a number of handles `N`.
/// Assumes there are exactly `N` matched ops (but could be relaxed).
/// Returns the tuple of handles.
template <int N, typename... MatchingArgs>
auto matchAndUnpack(ImplicitLocOpBuilder &b, Value targetH,
                    MatchingArgs... args) {
  Value matchedH = b.create<MatchOp>(targetH, args...);
  auto matchOp = b.create<SplitHandlesOp>(matchedH,
                                          /*numHandles=*/N);
  assert(matchOp->getNumResults() == N && "Unexpected number of results");
  std::array<Value, N> a;
  for (int64_t i = 0; i < N; ++i) a[i] = matchOp->getResult(i);
  return std::tuple_cat(a);
}

//===----------------------------------------------------------------------===//
// Higher-level problem-specific strategy creation APIs, these should favor
// user-friendliness.
//===----------------------------------------------------------------------===//

// TODO: consider passing a problem-specific struct to control information.
static Value createReductionStrategyThreadDistributionPart(
    ImplicitLocOpBuilder &b, Value variantH, ArrayRef<int64_t> tileSizes1Fill,
    ArrayRef<int64_t> tileSizes1Generic, bool hasLeadingEltwise,
    bool hasTrailingEltwise) {
  // TODO: Relying on ordering is brittle, harden this.
  Value matchedH = b.create<MatchOp>(
      variantH, ArrayRef<StringRef>{linalg::GenericOp::getOperationName(),
                                    linalg::FillOp::getOperationName()});
  auto split = b.create<SplitHandlesOp>(
      matchedH,
      /*numResultHandles=*/4 + hasLeadingEltwise + hasTrailingEltwise);
  Value firstFusionRootH = split.getResults()[1 + hasLeadingEltwise];
  SmallVector<Value> firstFusionGroupHs =
      split.getResults().take_front(1 + hasLeadingEltwise);
  Value secondFusionRootH = split.getResults().back();
  SmallVector<Value> secondFusionGroupHs =
      split.getResults().drop_front(2 + hasLeadingEltwise).drop_back();

  auto z = mlir::gpu::GPUThreadMappingAttr::get(b.getContext(),
                                                ::mlir::gpu::Threads::DimZ);
  auto y = mlir::gpu::GPUThreadMappingAttr::get(b.getContext(),
                                                ::mlir::gpu::Threads::DimY);

  // clang-format off
  iree_compiler::buildTileFuseDistToForeachThreadWithTileSizes(b,
                   /*rootH=*/secondFusionRootH,
                   /*opsHToFuse=*/secondFusionGroupHs,
                   /*tileSizes=*/getAsOpFoldResult(b.getI64ArrayAttr(tileSizes1Fill)),
                   /*threadDimMapping=*/b.getArrayAttr({z}));
  iree_compiler::buildTileFuseDistToForeachThreadWithTileSizes(b,
                   /*rootH=*/firstFusionRootH,
                   /*opsHToFuse=*/firstFusionGroupHs,
                   /*tileSizes=*/getAsOpFoldResult(b.getI64ArrayAttr(tileSizes1Generic)),
                   /*threadDimMapping=*/b.getArrayAttr({z,y}));
  // clang-format on
  return variantH;
}

/// Structure to hold the parameters related to GPU reduction strategy.
struct GPUReductionStrategyInfos {
  std::array<int64_t, 3> workgroupSize;
  SmallVector<int64_t> workgroupTileSizes;
  SmallVector<int64_t> fillSecondTileSizes;
  SmallVector<int64_t> genericSecondTileSizes;
  bool hasLeadingEltwise;
  bool hasTrailingEltwise;
};

/// Returns a triple of handles: the leading elementwise operation, the
/// reduction operation and the fusion root. The leading elementwise and the
/// fusion root may be null. If the fusion root is null, the reduction operation
/// should be used as fusion root instead.
// TODO: consider passing a problem-specific struct to control information.
static std::tuple<Value, Value, Value>
createMatchReductionBlockDistributionHandles(ImplicitLocOpBuilder &b,
                                             Value variantH,
                                             bool hasLeadingEltwise,
                                             bool hasTrailingEltwise) {
  Value originalGenericH =
      b.create<MatchOp>(variantH, linalg::GenericOp::getOperationName());
  auto op = b.create<SplitHandlesOp>(
      originalGenericH,
      /*numResultHandles=*/1 + hasLeadingEltwise + hasTrailingEltwise);
  return std::make_tuple(hasLeadingEltwise ? op.getResults().front() : Value(),
                         op.getResults().drop_front(hasLeadingEltwise).front(),
                         hasTrailingEltwise ? op.getResults().back() : Value());
}

// TODO: generalize and automate over and over.
// TODO: significantly shrink this down.
// TODO: consider passing a problem-specific struct to control information.
static void createReductionCudaStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const GPUReductionStrategyInfos &infos) {
  // Step 0. Match the ops.
  Value originalFillH =
      b.create<MatchOp>(variantH, linalg::FillOp::getOperationName());
  auto [leadingH, reductionH, fusionRootH] =
      createMatchReductionBlockDistributionHandles(
          b, variantH, infos.hasLeadingEltwise, infos.hasTrailingEltwise);

  // Step 1: Distribute to blocks using the current IREE lowering config.
  variantH = iree_compiler::createReductionStrategyBlockDistributionPart(
      b, variantH, originalFillH, reductionH, fusionRootH,
      getAsOpFoldResult(b.getI64ArrayAttr(infos.workgroupTileSizes)),
      infos.hasLeadingEltwise, infos.hasTrailingEltwise);

  // Step 2. Second level of tiling + fusion parallelizes to threads.
  variantH = createReductionStrategyThreadDistributionPart(
      b, variantH, infos.fillSecondTileSizes, infos.genericSecondTileSizes,
      infos.hasLeadingEltwise, infos.hasTrailingEltwise);

  // Step 3. Rank-reduce and vectorize.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = iree_compiler::buildVectorize(b, funcH);

  // Step 4. Bufferize and drop HAL descriptor from memref ops.
  variantH = iree_compiler::buildBufferize(b, variantH, /*targetGpu=*/true);

  // Step 5. Post-bufferization mapping to blocks and threads.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH =
      iree_compiler::buildMapToBlockAndThreads(b, funcH, infos.workgroupSize);

  // Step 6. Post-bufferization vector distribution with rank-reduction.
  iree_compiler::buildDistributeVectors(b, variantH, funcH);
}

// TODO: consider passing a problem-specific struct to control information.
static bool matchGPUReduction(linalg::LinalgOp op,
                              GPUReductionStrategyInfos &info) {
  // TODO: match the sequence the strategy supports.
  StructuredOpMatcher pattern, fill, leadingEltwise, trailingEltwise;
  makeReductionMatcher(pattern, fill, leadingEltwise, trailingEltwise);
  if (!matchPattern(op, pattern)) return false;

  info.hasLeadingEltwise = leadingEltwise.getCaptured() != nullptr;
  info.hasTrailingEltwise = trailingEltwise.getCaptured() != nullptr;

  // Hardcoded workagroup size, this could be deduced from the reduction dim.
  info.workgroupSize = {32, 2, 1};
  SmallVector<unsigned> partitionedLoops =
      cast<iree_compiler::PartitionableLoopsInterface>(op.getOperation())
          .getPartitionableLoops(iree_compiler::kNumMaxParallelDims);
  size_t numLoops = partitionedLoops.empty() ? 0 : partitionedLoops.back() + 1;
  // Tile all the parallel dimension to 1.
  info.workgroupTileSizes.append(numLoops, 1);
  info.fillSecondTileSizes = {1, 0, 0};
  info.genericSecondTileSizes = {1, 1, 0};
  return true;
}

LogicalResult iree_compiler::matchAndSetGPUReductionTransformStrategy(
    func::FuncOp entryPoint, linalg::LinalgOp op) {
  // 1. Match
  GPUReductionStrategyInfos infos;
  if (!matchGPUReduction(op, infos)) return failure();
  auto strategyBuilder = [&](ImplicitLocOpBuilder &b, Value variant) {
    return createReductionCudaStrategy(b, variant, infos);
  };
  // 2. Add the strategy.
  createTransformRegion(entryPoint, strategyBuilder);
  return success();
}
