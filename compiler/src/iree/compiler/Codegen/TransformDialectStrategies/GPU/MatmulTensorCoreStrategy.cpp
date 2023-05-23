// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/MatmulTensorCoreStrategy.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/Common.h"
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
using iree_compiler::buildPad;
using iree_compiler::buildTileFuseDistToForallWithNumThreads;
using iree_compiler::buildTileFuseDistToForallWithTileSizes;
using iree_compiler::TileToForallAndFuseAndDistributeResult;
using iree_compiler::gpu::buildBufferize;
using iree_compiler::gpu::buildConvertToAsyncCopies;
using iree_compiler::gpu::buildConvertToTensorCoreOp;
using iree_compiler::gpu::buildDistributeCopies;
using iree_compiler::gpu::buildHoistOutputPaddingOp;
using iree_compiler::gpu::buildMatmulVectorization;
using iree_compiler::gpu::buildMultiBuffering;
using iree_compiler::gpu::buildPipelineSharedMemoryCopies;
using iree_compiler::gpu::kCudaWarpSize;
using iree_compiler::gpu::MatmulStrategy;
using iree_compiler::gpu::scaleUpByBitWidth;
using iree_compiler::IREE::transform_dialect::ApplyPatternsOpPatterns;
using iree_compiler::IREE::transform_dialect::
    IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp;
using transform::MatchOp;
using transform_ext::RegisterMatchCallbacksOp;

/// Options to set the default values of the matmul strategy.

/// Block tile size X, Y, Z.
static llvm::cl::opt<int64_t> clBlockTileSizeX(
    "td-matmul-strategy-blk-size-x",
    llvm::cl::desc("block tile size for dim X (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::init(128));
static llvm::cl::opt<int64_t> clBlockTileSizeY(
    "td-matmul-strategy-blk-size-y",
    llvm::cl::desc("block tile size for dim Y (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::init(128));
static llvm::cl::opt<int64_t> clBlockTileSizeZ(
    "td-matmul-strategy-blk-size-z",
    llvm::cl::desc("block tile size for dim z (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::init(1));

static llvm::cl::opt<int64_t> clReductionTileSize(
    "td-matmul-strategy-reduc-size",
    llvm::cl::desc(
        "reduction tile sized for the transform dialect matmul strategy"),
    llvm::cl::init(16));

/// Number of threads X, Y, Z.
static llvm::cl::opt<int64_t> clNumThreadsX(
    "td-matmul-strategy-num-threads-x",
    llvm::cl::desc("number of threads for dim X (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::init(64));
static llvm::cl::opt<int64_t> clNumThreadsY(
    "td-matmul-strategy-num-threads-y",
    llvm::cl::desc("number of threads for dim Y (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::init(2));
static llvm::cl::opt<int64_t> clNumThreadsZ(
    "td-matmul-strategy-num-threads-z",
    llvm::cl::desc("number of threads for dim z (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::init(1));

/// Number of warps X, Y, Z.
static llvm::cl::opt<int64_t> clNumWarpsX(
    "td-matmul-strategy-num-warps-x",
    llvm::cl::desc("number of warps for dim X (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::init(2));
static llvm::cl::opt<int64_t> clNumWarpsY(
    "td-matmul-strategy-num-warps-y",
    llvm::cl::desc("number of warps for dim Y (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::init(2));
static llvm::cl::opt<int64_t> clNumWarpsZ(
    "td-matmul-strategy-num-warps-z",
    llvm::cl::desc("number of warps for dim z (x,y,z) for the transform "
                   "dialect matmul strategy"),
    llvm::cl::init(1));

static llvm::cl::opt<bool> clUseAsyncCopies(
    "td-matmul-strategy-use-async-copies",
    llvm::cl::desc("use mma sync for the transform dialect matmul strategy"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clUseMmaSync(
    "td-matmul-strategy-use-mma-sync",
    llvm::cl::desc("use mma sync for the transform dialect matmul strategy"),
    llvm::cl::init(false));

static llvm::cl::opt<int64_t> clPipelineDepth(
    "td-matmul-strategy-pipeline-depth",
    llvm::cl::desc("pipeline depth for the transform dialect matmul strategy"),
    llvm::cl::init(3));

void MatmulStrategy::initDefaultValues() {
  blockTileSizes = {clBlockTileSizeX, clBlockTileSizeY, clBlockTileSizeZ};
  reductionTileSize = clReductionTileSize;
  numThreads = {clNumThreadsX, clNumThreadsY, clNumThreadsZ};
  numWarps = {clNumWarpsX, clNumThreadsY, clNumThreadsZ};
  useAsyncCopies = clUseAsyncCopies;
  useMmaSync = clUseMmaSync;
  pipelineDepth = clPipelineDepth;

  // TODO: Capture input/output element types properly for configuring the
  // padding values.
  paddingValues = {0.0f, 0.0f, 0.0f};
  paddingDimensions = {0, 1, 2};
  packingDimensions = {1, 1, 1};
}

LLVM_DUMP_METHOD void MatmulStrategy::dump() const { print(llvm::errs()); }

void MatmulStrategy::print(llvm::raw_ostream &os) const {
  os << "\n--- Matmul strategy ---\n";
  os << "- block tile sizes: {";
  bool isFirst = true;
  for (int64_t blockTileSize : blockTileSizes) {
    if (!isFirst) os << ", ";
    os << blockTileSize;
    isFirst = false;
  }
  os << "}\n";
  os << "- reduction tile size: " << reductionTileSize << '\n';

  os << "- number of threads: {";
  isFirst = true;
  for (int64_t numThreadsForDim : numThreads) {
    if (!isFirst) os << ", ";
    os << numThreadsForDim;
    isFirst = false;
  }
  os << "}\n";

  os << "- number of warps: {";
  isFirst = true;
  for (int64_t numWarpsForDim : numWarps) {
    if (!isFirst) os << ", ";
    os << numWarpsForDim;
    isFirst = false;
  }
  os << "}\n";

  os << "- use async copies: " << useAsyncCopies << '\n';
  os << "- use mma sync: " << useMmaSync << '\n';
  os << "- pipeline depth: " << pipelineDepth << '\n';
}

static std::tuple<Value, Value, Value, Value>
buildMatmulStrategyBlockDistribution(ImplicitLocOpBuilder &b, Value variantH,
                                     const MatmulStrategy &strategy) {
  // Step 1. Call the matcher. Note that this is the same matcher as used to
  // trigger this compilation path, so it must always apply.
  b.create<RegisterMatchCallbacksOp>();
  auto [fillH, matmulH, maybeTrailingH] = unpackRegisteredMatchCallback<3>(
      b, "matmul", transform::FailurePropagationMode::Propagate, variantH);

  // Step 2. Create the block/mapping tiling level and fusee.
  // auto [fusionTargetH, fusionGroupH] =
  //     buildSelectFirstNonEmpty(b, maybeTrailingH, matmulH);
  MatmulStrategy::MappingInfo blockMapping = strategy.getBlockMapping();
  TileToForallAndFuseAndDistributeResult tileResult =
      buildTileFuseDistToForallWithTileSizes(
          /*builder=*/b,
          /*isolatedParentOpH=*/variantH,
          /*rootH=*/matmulH,
          /*opsToFuseH=*/fillH,
          /*tileSizes=*/
          getAsOpFoldResult(b.getI64ArrayAttr(blockMapping.tileSizes)),
          /*threadDimMapping=*/
          b.getArrayAttr(blockMapping.threadMapping));

  // Handle the workgroup count region.
  b.create<IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp>(
      tileResult.forallH);

  // TODO: handle trailing op.
  return std::make_tuple(tileResult.resultingFusedOpsHandles.front(),
                         tileResult.tiledOpH, Value(), tileResult.forallH);
}

void iree_compiler::gpu::buildMatmulTensorCoreStrategy(
    ImplicitLocOpBuilder &b, Value variantH, const MatmulStrategy &strategy) {
  assert(strategy.totalNumThreads() ==
             strategy.totalNumWarps() * kCudaWarpSize &&
         "Number of threads specified by warps must match total number of "
         "threads");
  // Step 1. Apply block-level part of the strategy, keeps everything fused.
  auto [fillH, matmulH, maybeTiledTrailingHBlock, forall] =
      buildMatmulStrategyBlockDistribution(b, variantH, strategy);
  // Tile reduction loop.
  SmallVector<int64_t> tileSizes{0, 0, strategy.reductionTileSize};
  auto tileReductionResult =
      buildTileFuseToScfFor(b, variantH, matmulH, {},
                            getAsOpFoldResult(b.getI64ArrayAttr(tileSizes)));

  // Step 2. Pad the matmul op.
  // TODO: use captured type information to configure the padding values.
  auto paddedMatmulOpH =
      buildPad(b, tileReductionResult.tiledOpH,
               b.getF32ArrayAttr(strategy.paddingValues).getValue(),
               strategy.paddingDimensions, strategy.packingDimensions);

  // Step 3. Hoist the padding of the output operand above the reduction loop.
  // The resulting fillOp will be mapped with the contraction using an SIMD
  // programming model.
  Value fillOpH = buildHoistOutputPaddingOp(b, variantH, paddedMatmulOpH);

  // Step 4. Distribute pad and copies: SIMT programming model.
  auto [lhsCopyOpH, rhsCopyOpH, copyBackOpH] =
      buildDistributeCopies(b, variantH, paddedMatmulOpH, strategy);

  // Step 5. Distribute to warps: SIMD programming model.
  // TODO: get the number of warps from strategy.
  MatmulStrategy::MappingInfo computeMapping = strategy.computeMapping();
  buildTileFuseDistToForallWithNumThreads(
      b, variantH, paddedMatmulOpH, ValueRange(),
      getAsOpFoldResult(b.getI64ArrayAttr(computeMapping.numThreads)),
      b.getArrayAttr(computeMapping.threadMapping));
  buildTileFuseDistToForallWithNumThreads(
      b, variantH, fillOpH, ValueRange(),
      getAsOpFoldResult(b.getI64ArrayAttr(computeMapping.numThreads)),
      b.getArrayAttr(computeMapping.threadMapping));

  // Step 6. Rank-reduce and vectorize.
  buildMatmulVectorization(b, variantH, lhsCopyOpH, rhsCopyOpH, copyBackOpH,
                           strategy);

  // Step 7. Bufferize and drop HAL descriptor from memref ops.
  variantH = buildBufferize(b, variantH);

  // Step 8. Post-bufferization mapping to blocks and threads.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, needs hardening.
  // TODO: extract info from strategy.
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = buildMapToBlockAndThreads(b, funcH, strategy.numThreads,
                                    strategy.numWarps);

  // Step 9. Convert to tensor core ops.
  // TODO: avoid consuming handles and returning here.
  funcH = buildConvertToTensorCoreOp(b, funcH, strategy);

  if (strategy.useAsyncCopies) {
    // Step 10. Multi-buffering.
    buildMultiBuffering(b, funcH, strategy);

    // Step 11. Convert to async copies.
    // TODO: avoid consuming handles and returning here.
    funcH = buildConvertToAsyncCopies(b, funcH, strategy);

    // Step 12. Pipeline shared memory copies.
    buildPipelineSharedMemoryCopies(b, funcH, strategy);
  }

  // Step 13. Late lowerings and cleanups.
  // TODO: not a functional style op to avoid invalidating artificially.
  funcH = b.create<transform::LowerMasksOp>(
      pdl::OperationType::get(b.getContext()), funcH);
  // TODO: not a functional style op to avoid invalidating artificially.
  funcH = b.create<transform::MaterializeMasksOp>(
      pdl::OperationType::get(b.getContext()), funcH);
  {
    ApplyPatternsOpPatterns config;
    config.foldMemrefAliases = true;
    iree_compiler::buildCanonicalizationAndEnablingTransforms(b, config, funcH);
  }
}
