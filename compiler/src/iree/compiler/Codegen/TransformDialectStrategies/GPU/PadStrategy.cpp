// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/PadStrategy.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/Common.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/Strategies.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

// TODO: significantly better namespacing.
using iree_compiler::blockX;
using iree_compiler::blockY;
using iree_compiler::blockZ;
using iree_compiler::buildPad;
using iree_compiler::buildTileFuseDistToForallWithNumThreads;
using iree_compiler::buildTileFuseDistToForallWithTileSizes;
using iree_compiler::TileToForallAndFuseAndDistributeResult;
using iree_compiler::gpu::buildBufferize;
using iree_compiler::gpu::buildConvertToAsyncCopies;
using iree_compiler::gpu::buildDistributeOnePadOrCopyWithNumThreads;
using iree_compiler::gpu::buildDistributeOnePadOrCopyWithTileSizes;
using iree_compiler::gpu::kCudaWarpSize;
using iree_compiler::gpu::PadStrategy;
using iree_compiler::IREE::transform_dialect::ApplyPatternsOp;
using iree_compiler::IREE::transform_dialect::ApplyPatternsOpPatterns;
using iree_compiler::IREE::transform_dialect::
    IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp;
using transform::MatchOp;
using transform_ext::RegisterMatchCallbacksOp;

static llvm::cl::list<int64_t> clBlockTileSizes(
    "td-pad-strategy-blk-sizes",
    llvm::cl::desc("block tile sizes for dims (x,y,z) for the transform "
                   "dialect pad strategy"),
    llvm::cl::list_init(ArrayRef<int64_t>{64, 64, 1}),
    llvm::cl::CommaSeparated);
static llvm::cl::list<int64_t> clNumThreads(
    "td-pad-strategy-num-threads",
    llvm::cl::desc("number of threads for dims (x,y,z) for the transform "
                   "dialect pad strategy"),
    llvm::cl::list_init(ArrayRef<int64_t>{16, 16, 1}),
    llvm::cl::CommaSeparated);
static llvm::cl::list<int64_t> clVectorSize(
    "td-pad-strategy-vector-size",
    llvm::cl::desc("vector size for the transform dialect pad strategy"),
    llvm::cl::list_init(ArrayRef<int64_t>{4, 4}), llvm::cl::CommaSeparated);
static llvm::cl::opt<bool> clUseAsyncCopies(
    "td-pad-strategy-use-async-copies",
    llvm::cl::desc(
        "use async copies through shared memory for the pad strategy"),
    llvm::cl::init(false));

void iree_compiler::gpu::PadStrategy::initDefaultValues() {
  blockTileSizes =
      SmallVector<int64_t>{clBlockTileSizes.begin(), clBlockTileSizes.end()};
  numThreads = SmallVector<int64_t>{clNumThreads.begin(), clNumThreads.end()};
  vectorSize = SmallVector<int64_t>{clVectorSize.begin(), clVectorSize.end()};
  useAsyncCopies = clUseAsyncCopies;
}

void iree_compiler::gpu::PadStrategy::configure(GPUModel gpuModel) {}

static std::tuple<Value, Value> buildPadStrategyBlockDistribution(
    ImplicitLocOpBuilder &b, Value variantH, const PadStrategy &strategy) {
  // Step 1. Call the matcher. Note that this is the same matcher as used to
  // trigger this compilation path, so it must always apply.
  b.create<RegisterMatchCallbacksOp>();
  auto [padH] = unpackRegisteredMatchCallback<1>(
      b, "pad", transform::FailurePropagationMode::Propagate, variantH);

  // Step 2. Create the block/mapping tiling level.
  MLIRContext *ctx = b.getContext();
  auto [tiledPadH, forallH] = buildDistributeOnePadOrCopyWithTileSizes(
      b, variantH, padH,
      /*tileSizes=*/{strategy.blockTileSizeY(), strategy.blockTileSizeX()},
      /*threadDimMapping=*/{blockY(ctx), blockX(ctx)}, /*foldIfBranch=*/true);

  // Step 3.Handle the workgroup count region.
  b.create<IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp>(forallH);
  return std::make_tuple(tiledPadH, forallH);
}

void iree_compiler::gpu::buildPadStrategy(ImplicitLocOpBuilder &b,
                                          Value variantH,
                                          const PadStrategy &strategy) {
  MLIRContext *ctx = b.getContext();
  // Step 1. Apply block-level part of the strategy.
  auto [padBlockH, forallBlockH] =
      buildPadStrategyBlockDistribution(b, variantH, strategy);

  // Step 2. Apply thread-level part of the strategy.
  auto padThreadH = buildDistributeOnePadOrCopyWithNumThreads(
      b, variantH, padBlockH,
      /*numThreads=*/{strategy.numThreadsY(), strategy.numThreadsX()},
      /*threadDimMapping=*/{threadY(ctx), threadX(ctx)}, /*foldIfBranch=*/true);

  // Step 3. Masked vectorization.
  b.create<transform::MaskedVectorizeOp>(padThreadH, ValueRange(), false,
                                         strategy.vectorSize);

  // Step 4. Lower all masked vector transfers at this point, as they make
  // canonicalization generate incorrect IR.
  // TODO: don't rematch, apply on the variant op directly.
  Value funcH =
      b.create<transform::MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = buildLowerMaskedTransfersAndCleanup(b, funcH);

  // Step 5. Vectorize the rest of func normally.
  funcH = buildVectorize(b, funcH, /*applyCleanups=*/true);

  // Step 6. Bufferize and drop HAL descriptor from memref ops.
  variantH = buildBufferize(b, variantH);

  // Step 7. Post-bufferization mapping to blocks and threads.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, needs hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = buildMapToBlockAndThreads(
      b, funcH,
      /*blockSize=*/
      {strategy.numThreadsX(), strategy.numThreadsY(), strategy.numThreadsZ()});

  // TODO: Multi-buffering and async copies in cases where HW supports it.
  assert(!strategy.useAsyncCopies && "not implemented yet");

  // Step 8. Lower masks before returning to the default lowering pipeline.
  buildLowerVectorMasksAndCleanup(b, funcH);
}
