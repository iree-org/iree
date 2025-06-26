// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"

#include <cstdint>
#include <numeric>
#include <optional>

#include "compiler/src/iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUSelectUKernels.h"
#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/Interfaces/UKernelOpInterface.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#define DEBUG_TYPE "iree-llvmgpu-kernel-config"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
namespace mlir::iree_compiler {

llvm::cl::opt<bool> clGPUEarlyTileAndFuseMatmul(
    "iree-codegen-llvmgpu-early-tile-and-fuse-matmul",
    llvm::cl::desc("test the the tile and fuse pipeline for matmul"),
    llvm::cl::init(false));

llvm::cl::opt<bool> clGPUTestTileAndFuseVectorize(
    "iree-codegen-llvmgpu-test-tile-and-fuse-vectorize",
    llvm::cl::desc(
        "test the tile and fuse pipeline for all supported operations"),
    llvm::cl::init(false));

llvm::cl::opt<bool> clLLVMGPUVectorizePipeline(
    "iree-codegen-llvmgpu-vectorize-pipeline",
    llvm::cl::desc("forces use of the legacy LLVMGPU vectorize pipeline"),
    llvm::cl::init(false));

llvm::cl::opt<bool> clGPUEnableVectorDistribution(
    "iree-codegen-llvmgpu-use-vector-distribution",
    llvm::cl::desc("enable the usage of the vector distribution pipeline"),
    llvm::cl::init(true));

// TODO (nirvedhmeshram): Drop this whole path after we have support with
// TileAndFuse pipeline from completion of
// https://github.com/iree-org/iree/issues/18858
llvm::cl::opt<bool> clGPUUnalignedGEMMVectorDistribution(
    "iree-codegen-llvmgpu-use-unaligned-gemm-vector-distribution",
    llvm::cl::desc("enable the usage of the vector distribution pipeline for "
                   "unaligned GEMMs when supported"),
    llvm::cl::init(false));

llvm::cl::opt<bool> clGPUUseTileAndFuseConvolution(
    "iree-codegen-llvmgpu-use-tile-and-fuse-convolution",
    llvm::cl::desc(
        "enable the tile and fuse pipeline for supported convolutions"),
    llvm::cl::init(true));

/// Flag to force using WMMA tensorcore operations.
llvm::cl::opt<bool>
    clGPUUseWMMA("iree-codegen-llvmgpu-use-wmma",
                 llvm::cl::desc("force use of wmma operations for tensorcore"),
                 llvm::cl::init(false));

/// Flag used to toggle using mma.sync vs wmma when targeting tensorcore.
llvm::cl::opt<bool>
    clGPUUseMMASync("iree-codegen-llvmgpu-use-mma-sync",
                    llvm::cl::desc("force use mma sync instead of wmma ops"),
                    llvm::cl::init(false));

llvm::cl::opt<int> clGPUMatmulCThreshold(
    "iree-codegen-llvmgpu-matmul-c-matrix-threshold",
    llvm::cl::desc("matmul c matrix element count threshold to be considered "
                   "as small vs. large when deciding MMA schedule"),
    // TODO: We should get this value from the target's parallelism.
    llvm::cl::init(512 * 512));

static llvm::cl::opt<bool> clLLVMGPUEnablePrefetch(
    "iree-llvmgpu-enable-prefetch",
    llvm::cl::desc("Enable prefetch in the vector distribute pipeline"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    clLLVMGPUUseIgemm("iree-codegen-llvmgpu-use-igemm",
                      llvm::cl::desc("Enable implicit gemm for convolutions."),
                      llvm::cl::init(true));

static llvm::cl::opt<bool>
    clUseDirectLoad("iree-llvmgpu-use-direct-load",
                    llvm::cl::desc("Use global load DMA for direct load ops."),
                    llvm::cl::Hidden, llvm::cl::init(false));

namespace {

using CodeGenPipeline = IREE::Codegen::DispatchLoweringPassPipeline;

// Threshold used to determine whether a matmul dimension is 'very skinny'.
constexpr int64_t kVerySkinnyDimThreshold = 4;

struct TileWorkgroupSizePair {
  // How many scalar elements each workgroup should handle along each dimension.
  std::array<int64_t, 3> tileSize;
  std::array<int64_t, 3> workgroupSize;
  int64_t pipelineDepth;
};

// Simt codegen does not do software pipelining.
constexpr unsigned softwarePipelineDepthSimt = 0;

} // namespace

bool isROCmBackend(IREE::GPU::TargetAttr target) {
  return target.getArch().starts_with("gfx");
}

static bool needsLoweringConfigPropagation(
    IREE::Codegen::DispatchLoweringPassPipeline pipeline) {
  using Pipeline = IREE::Codegen::DispatchLoweringPassPipeline;
  // Pipelines that do not need propagation of lowering config.
  Pipeline supportedPipelines[] = {Pipeline::LLVMGPUTileAndFuse,
                                   Pipeline::LLVMGPUVectorDistribute};
  return !llvm::is_contained(supportedPipelines, pipeline);
}

//====---------------------------------------------------------------------===//
// Matmul Configuration Helpers
//====---------------------------------------------------------------------===//

/// Return the best combination of tile size and wg size. It will then used to
/// pick the best size aligned with the shape dimension.
static SmallVector<TileWorkgroupSizePair>
getMatmulConfig(IREE::GPU::TargetAttr target) {
  SmallVector<TileWorkgroupSizePair> tileSizes;
  // Pick tile size so that M*K and K*N divisible by wgSize * \*vecSize=*\4.
  // This way workgroup memory copy don't need to be masked. Once we support
  // masked load we can get performance out of more configuration.

  // Make use of the full subgroup when possible.
  if (target.getPreferredSubgroupSize() == 64) {
    tileSizes.push_back(TileWorkgroupSizePair({{64, 128, 64}, {64, 16, 1}, 1}));
  }

  llvm::append_values(tileSizes,
                      TileWorkgroupSizePair({{32, 128, 32}, {32, 8, 1}, 1}),
                      TileWorkgroupSizePair({{128, 64, 8}, {16, 8, 1}, 1}),
                      TileWorkgroupSizePair({{16, 256, 32}, {64, 2, 1}, 1}),
                      TileWorkgroupSizePair({{8, 32, 32}, {8, 8, 1}, 1}),

                      TileWorkgroupSizePair({{32, 128, 4}, {32, 8, 1}, 1}),
                      TileWorkgroupSizePair({{8, 128, 4}, {32, 1, 1}, 1}),
                      TileWorkgroupSizePair({{16, 64, 4}, {16, 2, 1}, 1}),
                      TileWorkgroupSizePair({{1, 128, 8}, {32, 1, 1}, 1}));
  return tileSizes;
}

/// Return the best combination of tile size and wg size when using tensorcore
/// operations.
static void
getTensorCoreConfig(SmallVectorImpl<TileWorkgroupSizePair> &tileSizes,
                    Type elementType, int64_t M, int64_t N, int64_t K) {
  // Based on early analysis we found that 128x256x32_3 gives acceptable
  // performance across many of the large matrix sizes for f16 and fp32. This
  // needs to be refined into a better strategy based on empirical data but this
  // gives us a quick solution to achieve performance in the right order of
  // magnitude for large square like cases.
  int64_t parallelDim = M * N;
  static constexpr int64_t kLargDimThreashold = 1536;
  if (elementType.isF16()) {
    if (parallelDim >= kLargDimThreashold * kLargDimThreashold) {
      tileSizes.push_back(
          TileWorkgroupSizePair({{128, 256, 32}, {128, 2, 1}, 3}));
    }
    tileSizes.push_back(TileWorkgroupSizePair({{32, 32, 32}, {64, 2, 1}, 4}));
  } else {
    if (parallelDim >= kLargDimThreashold * kLargDimThreashold) {
      tileSizes.push_back(
          TileWorkgroupSizePair({{128, 256, 16}, {128, 2, 1}, 4}));
    }
    llvm::append_values(tileSizes,
                        TileWorkgroupSizePair({{32, 32, 16}, {64, 2, 1}, 4}),
                        TileWorkgroupSizePair({{16, 32, 16}, {64, 1, 1}, 4}),
                        TileWorkgroupSizePair({{32, 16, 16}, {32, 2, 1}, 4}),
                        TileWorkgroupSizePair({{16, 16, 16}, {32, 1, 1}, 4}));
  }
}

static bool supportsTensorCore(IREE::GPU::TargetAttr target,
                               linalg::LinalgOp op) {
  // Limit tensor core pipeline to matmul as not all combinations of transpose
  // are supported upstream.
  if (!target.supportsSyncMMAOps())
    return false;
  if (!(isa<linalg::MatmulOp>(op) || isa<linalg::BatchMatmulOp>(op))) {
    assert(linalg::isaContractionOpInterface(op));
    // If this is not a named op matmul check some properties to make sure that
    // we can map it to tensorcore ops. We should have only mulAdd in the region
    // and the output map should have no permutation and the last dimension
    // should be a reduce.
    Region &body = op->getRegion(0);
    Region::OpIterator it = body.op_begin();
    if (it == body.op_end() || !isa<arith::MulFOp>(*(it++)))
      return false;
    if (it == body.op_end() || !isa<arith::AddFOp>(*(it++)))
      return false;
    if (it == body.op_end() || !isa<linalg::YieldOp>(*(it++)))
      return false;
    AffineMap outputMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
    if (outputMap.getNumResults() != outputMap.getNumDims() - 1)
      return false;
    OpBuilder b(op);
    for (unsigned i = 0, e = outputMap.getNumResults(); i < e - 1; i++) {
      if (outputMap.getResult(i) != b.getAffineDimExpr(i))
        return false;
    }
  }
  return true;
}

/// Decides which tensorcore operations to use.
static CodeGenPipeline getTensorCorePipeline(Type elementType) {
  // Currently mma.sync is on by default for fp16 only.
  CodeGenPipeline codegenPipeline = CodeGenPipeline::LLVMGPUMatmulTensorCore;

  // For F16 and F32 use mmasync by default.
  if (elementType.isF16() || elementType.isF32()) {
    codegenPipeline = CodeGenPipeline::LLVMGPUMatmulTensorCoreMmaSync;
  }

  // Override the decision based on cl flags.
  assert(!(clGPUUseWMMA && clGPUUseMMASync) && "incompatible options.");
  if (clGPUUseMMASync) {
    codegenPipeline = CodeGenPipeline::LLVMGPUMatmulTensorCoreMmaSync;
  }
  if (clGPUUseWMMA) {
    codegenPipeline = CodeGenPipeline::LLVMGPUMatmulTensorCore;
  };
  return codegenPipeline;
}

//====---------------------------------------------------------------------===//
// Vector Distribution Reduction Pipeline Configuration
//====---------------------------------------------------------------------===//
//

static bool isMatmulLike(linalg::LinalgOp &linalgOp) {
  return linalg::isaContractionOpInterface(linalgOp) &&
         linalgOp.getNumParallelLoops() >= 1;
};

/// Check if `op` is a linalg.reduce or a linalg.generic that has at least one
/// reduction iterator.
static bool hasReductionIterator(linalg::LinalgOp &op) {
  return isa<linalg::ReduceOp, linalg::GenericOp>(op) &&
         llvm::any_of(op.getIteratorTypesArray(), linalg::isReductionIterator);
}

/// The bitwidth of the init and src operands is used to determine the
/// bitwidth of the operation. The bitwidth is minimum of the init and src
/// operands.
static FailureOr<int64_t> getBitWidth(linalg::LinalgOp op) {
  Value init = op.getDpsInitOperand(0)->get();
  Type initElemType = getElementTypeOrSelf(init);
  Value src = op.getDpsInputOperand(0)->get();
  Type srcElemType = getElementTypeOrSelf(src);

  if (auto initOp = init.getDefiningOp<linalg::GenericOp>()) {
    if (IREE::LinalgExt::isBitExtendOp(initOp)) {
      initElemType = getElementTypeOrSelf(initOp.getDpsInputs()[0]);
    }
  }

  if (auto srcOp = src.getDefiningOp<linalg::GenericOp>()) {
    if (IREE::LinalgExt::isBitExtendOp(srcOp)) {
      srcElemType = getElementTypeOrSelf(srcOp.getDpsInputs()[0]);
    }
  }

  if (!initElemType.isIntOrFloat() || !srcElemType.isIntOrFloat()) {
    return failure();
  }

  int64_t bitWidth = std::max(initElemType.getIntOrFloatBitWidth(),
                              srcElemType.getIntOrFloatBitWidth());
  return bitWidth;
}

/// Check if the reduction op has a single combiner operation.
static LogicalResult checkSingleCombiner(linalg::LinalgOp op) {
  bool foundSingleReductionOutput = false;
  for (auto [index, initOpOperand] : llvm::enumerate(op.getDpsInitsMutable())) {
    // Only single combiner operations are supported for now.
    SmallVector<Operation *> combinerOps;
    if (matchReduction(op.getRegionOutputArgs(), index, combinerOps) &&
        combinerOps.size() == 1) {
      if (foundSingleReductionOutput)
        return failure();
      foundSingleReductionOutput = true;
      continue;
    }
    if (!op.getMatchingIndexingMap(&initOpOperand).isIdentity())
      return failure();
  }
  if (!foundSingleReductionOutput) {
    return failure();
  }

  return success();
}

static llvm::FailureOr<IREE::GPU::LoweringConfigAttr>
getVectorDistributeReductionConfig(
    linalg::LinalgOp op, IREE::GPU::TargetAttr target,
    llvm::SmallDenseMap<unsigned, unsigned> &sharedWgpTiles,
    int64_t workgroupSize, int64_t subgroupSize, int64_t threadLoads) {
  MLIRContext *context = op.getContext();
  Builder b(context);

  SmallVector<unsigned> parallelDims;
  SmallVector<unsigned> reductionDims;
  op.getParallelDims(parallelDims);
  op.getReductionDims(reductionDims);

  SmallVector<int64_t> bounds = op.getStaticLoopRanges();

  SmallVector<int64_t> workgroupTileSizes(op.getNumLoops(), 0);
  SmallVector<int64_t> threadTileSizes(op.getNumLoops(), 0);
  SmallVector<int64_t> threadCounts(op.getNumLoops(), 1);
  SmallVector<int64_t> subGroupCounts(op.getNumLoops(), 1);
  SmallVector<int64_t> mapping(op.getNumLoops());
  std::iota(mapping.begin(), mapping.end(), 0);

  // Set the configuration for the operation with no reduction dims.
  // The workgroup tile sizes are set by the reduction operation.
  if (reductionDims.empty()) {
    SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 1);

    // For the shared wgp dimension, set the reduction tile sizes to be zero.
    // Copy the workgroup tiles sizes from the sharedWgpDims.
    for (const auto &[dim, tile_size] : sharedWgpTiles) {
      reductionTileSizes[dim] = 0;
      workgroupTileSizes[dim] = tile_size;
    }

    int64_t parallelSize = bounds[parallelDims.back()];
    if (ShapedType::isDynamic(parallelSize) ||
        parallelSize % threadLoads != 0) {
      return failure();
    }
    int64_t lastDimReductionTileSize = workgroupSize * threadLoads;

    // Setting subgroupBasis to minimum i.e., 1 and threadBasis
    // to maximum i.e., subgroupSize.
    int64_t subgroupBasis = 1;
    int64_t threadBasis = subgroupSize;

    lastDimReductionTileSize =
        llvm::APIntOps::GreatestCommonDivisor(
            {64, static_cast<uint64_t>(parallelSize)},
            {64, static_cast<uint64_t>(lastDimReductionTileSize)})
            .getZExtValue();

    if (!(sharedWgpTiles.size() == op.getNumLoops())) {
      int subgroupStride = threadBasis * threadLoads;
      while (lastDimReductionTileSize % subgroupStride != 0) {
        threadBasis >>= 1;
        subgroupStride = threadBasis * threadLoads;
      }
      int subgroup = lastDimReductionTileSize / subgroupStride;
      subgroupBasis = (subgroup == 0) ? 1 : subgroup;
    }

    // Since all the dimensions are contained within the shared parallel
    // dimension, set the tile sizes to 1.
    if (sharedWgpTiles.size() == op.getNumLoops()) {
      lastDimReductionTileSize = 1;
      threadLoads = 1;
      threadBasis = 1;
      subgroupBasis = 1;
    }

    reductionTileSizes[parallelDims.back()] = lastDimReductionTileSize;
    threadTileSizes[parallelDims.back()] = threadLoads;
    threadCounts[parallelDims.back()] = threadBasis;
    subGroupCounts[parallelDims.back()] = subgroupBasis;

    ArrayAttr subgroupBasisAttr = b.getArrayAttr(
        {b.getI64ArrayAttr(subGroupCounts), b.getI64ArrayAttr(mapping)});

    ArrayAttr threadBasisAttr = b.getArrayAttr(
        {b.getI64ArrayAttr(threadCounts), b.getI64ArrayAttr(mapping)});

    SmallVector<NamedAttribute, 4> configAttrs = {
        NamedAttribute("workgroup", b.getI64ArrayAttr(workgroupTileSizes)),
        NamedAttribute("reduction", b.getI64ArrayAttr(reductionTileSizes)),
        NamedAttribute("thread", b.getI64ArrayAttr(threadTileSizes)),
        NamedAttribute("thread_basis", threadBasisAttr),
        NamedAttribute("subgroup_basis", subgroupBasisAttr)};

    auto configDict = b.getDictionaryAttr(configAttrs);
    auto loweringConfig =
        IREE::GPU::LoweringConfigAttr::get(context, configDict);
    return loweringConfig;
  }

  // Setting the config for operation with atleast one reduction dimension.
  SmallVector<int64_t> partialReductionTileSizes(op.getNumLoops(), 0);
  int64_t lastReductionDim = reductionDims.back();

  // TODO: This is enabled for matvec on ROCm for now. We should
  // validate this strategy and extend to more linalg generics and to CUDA.
  if (isROCmBackend(target) && !ShapedType::isDynamicShape(bounds) &&
      isMatmulLike(op)) {
    int64_t parallelIdx = *llvm::find_if(
        parallelDims, [&](int64_t currIdx) { return bounds[currIdx] != 1; });
    int64_t parallelBound = bounds[parallelIdx];
    int64_t numParallelReductions = 1;
    const int64_t maxParallelFactor = workgroupSize / 4;
    for (int64_t parallelFactor = 2; (parallelFactor < maxParallelFactor) &&
                                     (parallelBound % parallelFactor == 0) &&
                                     (parallelBound > parallelFactor);
         parallelFactor *= 2) {
      numParallelReductions = parallelFactor;
    }
    sharedWgpTiles[parallelIdx] = numParallelReductions;
  }

  // Set the workgroup tile sizes according to the sharedWgpDims.
  for (const auto &[dim, tile_size] : sharedWgpTiles) {
    workgroupTileSizes[dim] = tile_size;
  }

  for (int64_t dim : reductionDims) {
    threadTileSizes[dim] = 1;
    partialReductionTileSizes[dim] = 1;
  }

  int64_t lastReductionDimSize = bounds[reductionDims.back()];
  if (ShapedType::isDynamic(lastReductionDimSize)) {
    return failure();
  }
  if (lastReductionDimSize % threadLoads != 0) {
    return failure();
  }

  int64_t partialReductionSize = workgroupSize * threadLoads;
  partialReductionSize = llvm::APIntOps::GreatestCommonDivisor(
                             {64, static_cast<uint64_t>(partialReductionSize)},
                             {64, static_cast<uint64_t>(lastReductionDimSize)})
                             .getZExtValue();

  int64_t threadBasis = subgroupSize;
  int subgroupStride = threadBasis * threadLoads;
  while (partialReductionSize % subgroupStride != 0) {
    threadBasis >>= 1;
    subgroupStride = threadBasis * threadLoads;
  }
  int subgroup = partialReductionSize / subgroupStride;
  int64_t subgroupBasis = (subgroup == 0) ? 1 : subgroup;

  partialReductionTileSizes[lastReductionDim] = partialReductionSize;
  threadTileSizes[lastReductionDim] = threadLoads;
  threadCounts[lastReductionDim] = threadBasis;
  subGroupCounts[lastReductionDim] = subgroupBasis;

  ArrayAttr subgroupBasisAttr = b.getArrayAttr(
      {b.getI64ArrayAttr(subGroupCounts), b.getI64ArrayAttr(mapping)});

  ArrayAttr threadBasisAttr = b.getArrayAttr(
      {b.getI64ArrayAttr(threadCounts), b.getI64ArrayAttr(mapping)});

  SmallVector<NamedAttribute, 5> configAttrs = {
      NamedAttribute("workgroup", b.getI64ArrayAttr(workgroupTileSizes)),
      NamedAttribute("partial_reduction",
                     b.getI64ArrayAttr(partialReductionTileSizes)),
      NamedAttribute("thread", b.getI64ArrayAttr(threadTileSizes)),
      NamedAttribute("thread_basis", threadBasisAttr),
      NamedAttribute("subgroup_basis", subgroupBasisAttr)};

  auto configDict = b.getDictionaryAttr(configAttrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);
  return loweringConfig;
}

static LogicalResult
populateConfigInfo(const llvm::SetVector<linalg::LinalgOp> &computeOps,
                   IREE::GPU::TargetAttr target, int64_t workgroupSize,
                   int64_t subgroupSize, int64_t threadLoads) {
  if (computeOps.empty()) {
    return failure();
  }

  SmallVector<unsigned> sharedParallelDims;
  linalg::LinalgOp op = computeOps.front();
  op.getParallelDims(sharedParallelDims);
  llvm::SmallDenseSet<unsigned> sharedParallelSet(sharedParallelDims.begin(),
                                                  sharedParallelDims.end());
  for (linalg::LinalgOp linalgOp : computeOps) {
    SmallVector<unsigned> currParallelDims;
    linalgOp.getParallelDims(currParallelDims);
    llvm::SmallDenseSet<unsigned> currParallelSet(currParallelDims.begin(),
                                                  currParallelDims.end());
    llvm::set_intersect(sharedParallelSet, currParallelSet);
  }
  llvm::SmallDenseMap<unsigned, unsigned> sharedWgpTiles;

  // Initialize the tile sizes of the shared workgroup dims to be 1.
  for (auto i : sharedParallelSet) {
    sharedWgpTiles[i] = 1;
  }

  // Determines if a lowering configuration should be attached to the given
  // LinalgOp with only parallel dims. This is needed if the op cannot be fused
  // with a reduction or introduces new loop dimensions.
  auto shouldAttachLoweringConfig = [&](linalg::LinalgOp linalgOp) -> bool {
    // If some of the users are in computeOps and some are outside of
    // computeOps; attach lowering config, since the op can't be fused.
    if (llvm::any_of(linalgOp->getUsers(),
                     [&](Operation *user) {
                       auto linalgUser = dyn_cast<linalg::LinalgOp>(user);
                       return linalgUser && computeOps.contains(linalgUser);
                     }) &&
        llvm::any_of(linalgOp->getUsers(), [&](Operation *user) {
          auto linalgUser = dyn_cast<linalg::LinalgOp>(user);
          return !linalgUser;
        })) {
      return true;
    }

    // If the indexing map introduces new dimensions (more inputs than results),
    // attach a lowering config.
    for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
      int64_t operandIdx = linalgOp.getIndexingMapIndex(operand);
      AffineMap indexingMap = linalgOp.getIndexingMapsArray()[operandIdx];
      if (indexingMap.getNumResults() > 0 &&
          indexingMap.getNumInputs() > indexingMap.getNumResults()) {
        return true;
      }
    }

    return false;
  };

  for (linalg::LinalgOp linalgOp : computeOps) {
    if (hasReductionIterator(linalgOp) ||
        shouldAttachLoweringConfig(linalgOp)) {
      auto loweringConfig = getVectorDistributeReductionConfig(
          linalgOp, target, sharedWgpTiles, workgroupSize, subgroupSize,
          threadLoads);
      if (failed(loweringConfig)) {
        return failure();
      }
      setLoweringConfig(linalgOp, *loweringConfig);
    }
  }
  return success();
}

/// Check if the dispatch has a single store operation.
/// If the dispatch meets the criterion, it returns the set of
/// compute ops.
static FailureOr<SetVector<linalg::LinalgOp>>
checkDispatchForVectorDistribution(mlir::FunctionOpInterface entryPoint) {
  SmallVector<Operation *> storeOps;

  entryPoint.walk([&](IREE::TensorExt::DispatchTensorStoreOp op) {
    storeOps.push_back(op.getOperation());
  });

  if (storeOps.empty()) {
    return failure();
  }

  BackwardSliceOptions sliceOptions;
  sliceOptions.inclusive = false;
  sliceOptions.omitBlockArguments = true;
  sliceOptions.omitUsesFromAbove = false;
  SetVector<Operation *> slice;

  for (Operation *op : storeOps) {
    [[maybe_unused]] LogicalResult result =
        getBackwardSlice(op, &slice, sliceOptions);
    assert(result.succeeded());
  }

  SetVector<linalg::LinalgOp> computeOps;
  bool containsValidReductionOp = true;
  for (Operation *op : llvm::reverse(slice)) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      if (isa<linalg::FillOp>(op)) {
        continue;
      }
      if (hasReductionIterator(linalgOp) &&
          failed(checkSingleCombiner(linalgOp))) {
        containsValidReductionOp = false;
        break;
      }
      computeOps.insert(linalgOp);
    }
  }

  // Return failure if the dispatch contains no reduction op.
  if (!containsValidReductionOp) {
    return failure();
  }

  // Get the reduction dimensions.
  auto getReductionDims = [](linalg::LinalgOp &linalgOp) -> SetVector<int64_t> {
    SetVector<int64_t> reductionDims;
    for (auto [idx, iterator] :
         llvm::enumerate(linalgOp.getIteratorTypesArray())) {
      if (linalg::isReductionIterator(iterator)) {
        reductionDims.insert(idx);
      }
    }
    return reductionDims;
  };

  for (linalg::LinalgOp linalgOp : computeOps) {
    for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
      int64_t operandIdx = linalgOp.getIndexingMapIndex(operand);
      AffineMap indexingMap = linalgOp.getIndexingMapsArray()[operandIdx];

      // Check whether the producer exists.
      Operation *producer = dyn_cast<OpResult>(operand->get()).getOwner();
      auto producerOp = dyn_cast<linalg::LinalgOp>(producer);
      if (!producerOp || !computeOps.contains(producerOp)) {
        continue;
      }

      // Check whether the op operand is not reduced and producer of that
      // operand is not a reduction op.
      auto reductionDims = getReductionDims(linalgOp);
      bool isOperandReduced = llvm::any_of(
          llvm::seq<int>(indexingMap.getNumResults()), [&](int val) {
            return reductionDims.contains(indexingMap.getDimPosition(val));
          });
      if (isOperandReduced && hasReductionIterator(producerOp)) {
        return failure();
      }
    }
  }
  return computeOps;
}

/// The `setReductionVectorDistributionConfig` attaches `lowering_config` to
/// multiple operations within a dispatch containing at least a single reduction
/// operation. It's divided into two parts:
/// 1. `checkDispatchForVectorDistribution` checks that the dispatch is
/// compatible with the vector distribution pipeline. If it's compatible, then
/// it returns a set of linalg operations to which `lowering_config` might be
/// attached.
/// 2. `populateConfigInfo` determines to which linalg operations it might
/// attach `lowering_config`. Currently, it attaches `lowering_config` to
/// reduction operations and parallel operations that have new dimensions.
///   a. `getVectorDistributeReductionConfig` determines the `lowering_config`
///   for the reduction as well as parallel operations with new dimension.

/// The workgroup, subgroup, and threadTileSizes are determined by the
/// `setReductionVectorDistributionConfig` operation, which are global
/// information that is used by `populateConfigInfo` while determining the
/// `lowering_config`.

/// TODO (pashu123):
/// The threadTileSizes should be determined per operation rather than passed as
/// global information. This is due to the current limitation of the vector
/// distribution pipeline, which demands that the `vector.transfer_read` with
/// multiple users have the same layout.
/// TODO (pashu123):
/// The workgroup and subgroup sizes are determined by the single operation
/// within the dispatch. Extend it to analyze the dispatch and determine the
/// workgroup and subgroup sizes.
static LogicalResult
setReductionVectorDistributionConfig(IREE::GPU::TargetAttr target,
                                     mlir::FunctionOpInterface entryPoint,
                                     linalg::LinalgOp op) {
  MLIRContext *context = op.getContext();
  OpBuilder b(context);

  if (!hasReductionIterator(op)) {
    return failure();
  }

  FailureOr<SetVector<linalg::LinalgOp>> computeOps =
      checkDispatchForVectorDistribution(entryPoint);

  if (failed(computeOps)) {
    return failure();
  }

  SmallVector<unsigned> parallelDims;
  SmallVector<unsigned> reductionDims;
  op.getParallelDims(parallelDims);
  op.getReductionDims(reductionDims);

  SmallVector<int64_t> bounds = op.getStaticLoopRanges();
  IREE::GPU::TargetWgpAttr wgp = target.getWgp();
  int64_t reductionSize = bounds[reductionDims.back()];

  int64_t numDynamicReductionDims = 0;
  for (unsigned dim : reductionDims) {
    if (ShapedType::isDynamic(bounds[dim])) {
      ++numDynamicReductionDims;
    }
  }

  int64_t subgroupSize = 0;
  for (int s : wgp.getSubgroupSizeChoices().asArrayRef()) {
    if (reductionSize % s == 0 || numDynamicReductionDims > 0) {
      subgroupSize = s;
      break;
    }
  }

  if (subgroupSize == 0)
    return failure();

  auto bitWidth = getBitWidth(op);
  if (failed(bitWidth)) {
    return failure();
  }

  // Reduction distribution only supports 8/16/32 bit types now.
  if (!llvm::is_contained({8, 16, 32}, *bitWidth)) {
    return failure();
  }

  const std::optional<int64_t> maxLoadBits = wgp.getMaxLoadInstructionBits();
  const unsigned largestLoadSizeInBits =
      maxLoadBits.has_value() ? *maxLoadBits : 128;

  unsigned threadLoads = largestLoadSizeInBits / *bitWidth;
  if (numDynamicReductionDims == 0) {
    while ((reductionSize / threadLoads) % subgroupSize != 0) {
      threadLoads /= 2;
    }
  }
  // Deduce the workgroup size we should use for reduction. Currently a
  // workgroup processes all elements in reduction dimensions. Need to make sure
  // the workgroup size we use can divide the total reduction size, and it's
  // also within hardware limitations.
  const int64_t maxWorkgroupSize = 1024;
  int64_t workgroupSize = reductionSize / threadLoads;
  if (workgroupSize > maxWorkgroupSize) {
    workgroupSize = llvm::APIntOps::GreatestCommonDivisor(
                        {64, static_cast<uint64_t>(workgroupSize)},
                        {64, static_cast<uint64_t>(maxWorkgroupSize)})
                        .getZExtValue();
  }

  std::optional<int64_t> parallelSize = 1;
  for (int64_t dim : parallelDims) {
    if (ShapedType::isDynamic(bounds[dim])) {
      parallelSize = std::nullopt;
      break;
    }
    *parallelSize *= bounds[dim];
  }

  // Total parallel size that can fill the GPU with enough workgorups.
  // TODO: query from the target device; roughly 2x hardware compute unit.
  const int parallelThreshold = 256;
  // How many 128-bit vectors each thread should at least read.
  const int targetVectorCount = 8;
  while (parallelSize && *parallelSize > parallelThreshold &&
         (workgroupSize / 2) % subgroupSize == 0 &&
         reductionSize / (workgroupSize * threadLoads) < targetVectorCount) {
    // Use less subgroups per workgroup..
    workgroupSize /= 2;
    // in order to host more workgroups per hardware compute unit.
    *parallelSize /= 2;
  }

  // TODO(pashu123): Currently, the threadLoads is done on the basis of
  // the root operation and ignores other operation within a dispatch.
  // Extend it to use per operation within a dispatch.
  if (failed(populateConfigInfo(*computeOps, target, workgroupSize,
                                subgroupSize, threadLoads))) {
    return failure();
  }

  auto pipelineOptions = IREE::GPU::GPUPipelineOptionsAttr::get(context);
  SmallVector<NamedAttribute, 1> pipelineAttrs = {NamedAttribute(
      IREE::GPU::GPUPipelineOptionsAttr::getDictKeyName(), pipelineOptions)};

  auto pipelineConfig = b.getDictionaryAttr(pipelineAttrs);

  auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      context, CodeGenPipeline::LLVMGPUVectorDistribute, SymbolRefAttr(),
      {workgroupSize, 1, 1}, subgroupSize, pipelineConfig);

  return setTranslationInfo(entryPoint, translationInfo);
}

//====---------------------------------------------------------------------===//
// Vector Distribution Contraction/Convolution Pipeline Configuration
//====---------------------------------------------------------------------===//

static LogicalResult
setConvolutionVectorDistributionConfig(IREE::GPU::TargetAttr target,
                                       mlir::FunctionOpInterface entryPoint,
                                       linalg::LinalgOp op) {
  if (target.getWgp().getMma().empty())
    return failure();

  const int64_t targetSubgroupSize = target.getPreferredSubgroupSize();

  SmallVector<int64_t> bounds = op.getStaticLoopRanges();
  FailureOr<mlir::linalg::ConvolutionDimensions> convolutionDims =
      mlir::linalg::inferConvolutionDims(op);
  if (failed(convolutionDims)) {
    return failure();
  }

  // This strategy turns non-strided/dilated convolution problems into matmul
  // problems by tiling certain dimensions to 1:
  //  - Batch dimensions (parallel shared by the image and output)
  //  - Filter dimensions (reduction on the filter, and convolved on the image)
  //  - All output image dimensions except the outermost one
  //
  // After this, the remaining non-unit dimensions are:
  //  - One output image dimension corresponding to the M dimension of a matmul.
  //  - The output channel dimension, corresponding to the N dimension.
  //  - The input channel dimension, corresponding to the K dimension.

  // TODO: Relax this condition to strictly alignment requirements.
  if (convolutionDims->outputChannel.size() < 1 ||
      convolutionDims->inputChannel.size() < 1 ||
      convolutionDims->filterLoop.size() < 1 ||
      convolutionDims->outputImage.size() < 1 ||
      convolutionDims->depth.size() != 0) {
    return failure();
  }

  auto isAllOnesList = [](ArrayRef<int64_t> list) {
    return llvm::all_of(list, [](int64_t i) { return i == 1; });
  };

  // TODO: Support non-unit strides/dilations.
  if (!isAllOnesList(convolutionDims->strides) ||
      !isAllOnesList(convolutionDims->dilations)) {
    return failure();
  }

  int64_t mDim = convolutionDims->outputImage.back();
  int64_t nDim = convolutionDims->outputChannel.back();
  // TODO: Support NCHW convolutions. This is just a matmul_transpose_a, however
  // the distribution patterns currently do not support that variant.
  if (mDim > nDim) {
    return failure();
  }
  int64_t kDim = convolutionDims->inputChannel.back();

  Value lhs = op.getDpsInputOperand(0)->get();
  Value rhs = op.getDpsInputOperand(1)->get();
  Value init = op.getDpsInitOperand(0)->get();

  Type lhsElemType = getElementTypeOrSelf(lhs);
  Type rhsElemType = getElementTypeOrSelf(rhs);
  Type initElemType = getElementTypeOrSelf(init);

  // TODO(Max191): Support multiple M/N/K dimension problems for MMASchedules
  // once the pipeline is able to support it. After adding multiple dimensions,
  // all instances of schedule->m/nSubgroupCounts[0] and
  // schedule->m/n/kTileSizes[0] need to use the full list of sizes instead of
  // just the first element.
  GPUMatmulShapeType problem{bounds[mDim], bounds[nDim], bounds[kDim],
                             lhsElemType,  rhsElemType,  initElemType};

  // Helper fn to store mma information.
  auto storeMmaInfo = [](IREE::GPU::MmaInterfaceAttr mma,
                         SmallVector<GPUIntrinsicType> &intrinsics) {
    auto [mSize, nSize, kSize] = mma.getMNKShape();
    auto [aType, bType, cType] = mma.getABCElementTypes();
    intrinsics.emplace_back(mSize, nSize, kSize, aType, bType, cType, mma);
  };

  SmallVector<GPUIntrinsicType> intrinsics;
  intrinsics.reserve(target.getWgp().getMma().size());
  MLIRContext *context = op.getContext();
  for (IREE::GPU::MMAAttr mma : target.getWgp().getMma()) {
    if (mma.getSubgroupSize() != targetSubgroupSize)
      continue;
    storeMmaInfo(mma, intrinsics);
    // Store info on virtual intrinsics based on current mma if any
    for (IREE::GPU::VirtualMMAIntrinsic virtualIntrinsic :
         mma.getVirtualIntrinsics()) {
      auto virtualMma =
          IREE::GPU::VirtualMMAAttr::get(context, virtualIntrinsic);
      storeMmaInfo(virtualMma, intrinsics);
    }
  }

  if (intrinsics.empty())
    return failure();

  // Note that the following heuristic seeds are just placeholder values.
  // We need to clean it up and make it adjusting to different targets.
  // See https://github.com/iree-org/iree/issues/16341 for details.
  GPUMMAHeuristicSeeds seeds{/*bestSubgroupCountPerWorkgroup=*/4,
                             /*bestMNTileCountPerSubgroup=*/8,
                             /*bestKTileCountPerSubgroup=*/2};

  int64_t maxSharedMemoryBytes = target.getWgp().getMaxWorkgroupMemoryBytes();

  // First try to find a schedule with an exactly matching intrinsic.
  FailureOr<GPUMMASchedule> schedule = deduceMMASchedule(
      problem, intrinsics, seeds, maxSharedMemoryBytes, targetSubgroupSize);
  if (failed(schedule)) {
    // Then try again by allowing upcasting accumulator.
    schedule = deduceMMASchedule(
        problem, intrinsics, seeds, maxSharedMemoryBytes, targetSubgroupSize,
        /*transposedLhs*/ false, /*transposedRhs*/ false,
        /*canUpcastAcc=*/true);
  }
  if (failed(schedule)) {
    return failure();
  }

  int64_t flatWorkgroupSize =
      targetSubgroupSize *
      ShapedType::getNumElements(schedule->nSubgroupCounts) *
      ShapedType::getNumElements(schedule->mSubgroupCounts);
  std::array<int64_t, 3> workgroupSize{flatWorkgroupSize, 1, 1};

  SmallVector<int64_t> workgroupTileSizes(op.getNumLoops(), 0);
  SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 0);
  // Tile all batch dimensions with unit size.
  for (int64_t batch : convolutionDims->batch) {
    workgroupTileSizes[batch] = 1;
  }
  // Tile all output image dimensions with unit size except the last one.
  for (int64_t oi : llvm::drop_end(convolutionDims->outputImage)) {
    workgroupTileSizes[oi] = 1;
  }
  for (int64_t oc : llvm::drop_end(convolutionDims->outputChannel)) {
    workgroupTileSizes[oc] = 1;
  }
  for (int64_t ic : llvm::drop_end(convolutionDims->inputChannel)) {
    reductionTileSizes[ic] = 1;
  }
  // Compute the M/N dimension tile size by multiply subgroup information.
  workgroupTileSizes[mDim] =
      schedule->mSubgroupCounts[0] * schedule->mTileSizes[0] * schedule->mSize;
  workgroupTileSizes[nDim] =
      schedule->nSubgroupCounts[0] * schedule->nTileSizes[0] * schedule->nSize;

  reductionTileSizes[kDim] = schedule->kTileSizes[0] * schedule->kSize;

  // Tile all filter loop dimensions to 1.
  for (int64_t filterDim : convolutionDims->filterLoop) {
    reductionTileSizes[filterDim] = 1;
  }

  Builder b(context);
  SmallVector<NamedAttribute, 2> attrs = {
      NamedAttribute("workgroup", b.getI64ArrayAttr(workgroupTileSizes)),
      NamedAttribute("reduction", b.getI64ArrayAttr(reductionTileSizes))};
  IREE::GPU::appendPromotedOperandsList(context, attrs, {0, 1});
  IREE::GPU::setMmaKind(context, attrs, schedule->mmaKind);
  IREE::GPU::setSubgroupMCount(context, attrs, schedule->mSubgroupCounts[0]);
  IREE::GPU::setSubgroupNCount(context, attrs, schedule->nSubgroupCounts[0]);

  auto configDict = DictionaryAttr::get(context, attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  SmallVector<NamedAttribute, 1> pipelineAttrs;

  // Prefetch shared memory if requested.
  if (clLLVMGPUEnablePrefetch) {
    auto pipelineOptions = IREE::GPU::GPUPipelineOptionsAttr::get(
        context, /*prefetchSharedMemory=*/true,
        /*no_reduce_shared_memory_bank_conflicts=*/false,
        /*use_igemm_convolution=*/false,
        /*reorder_workgroups_strategy=*/std::nullopt);
    pipelineAttrs.emplace_back(
        IREE::GPU::GPUPipelineOptionsAttr::getDictKeyName(), pipelineOptions);
  }

  auto pipelineConfig = DictionaryAttr::get(context, pipelineAttrs);

  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, loweringConfig, CodeGenPipeline::LLVMGPUVectorDistribute,
      workgroupSize, targetSubgroupSize, pipelineConfig);
}

[[maybe_unused]] static void
debugPrintContractionInfo(StringRef label, unsigned numLoops,
                          linalg::ContractionDimensions contractionDims,
                          ArrayRef<int64_t> sizes) {
  ArrayRef<unsigned> dimVals[] = {contractionDims.batch, contractionDims.m,
                                  contractionDims.n, contractionDims.k};
  std::string dimSymbols(numLoops, '*');
  for (auto [idx, val] : llvm::enumerate(dimSymbols)) {
    for (auto [letter, dim] : llvm::zip_equal(StringRef("bmnk"), dimVals))
      if (llvm::is_contained(dim, idx))
        val = letter;
  }
  DBGS() << "Contraction dims: " << llvm::interleaved_array(dimSymbols) << "\n";
  DBGS() << label << ": " << llvm::interleaved_array(sizes) << "\n";
}

static LogicalResult
setMatmulVectorDistributionConfig(IREE::GPU::TargetAttr target,
                                  mlir::FunctionOpInterface entryPoint,
                                  linalg::LinalgOp op) {
  if (target.getWgp().getMma().empty())
    return failure();

  const int64_t targetSubgroupSize = target.getPreferredSubgroupSize();

  SmallVector<int64_t> bounds = op.getStaticLoopRanges();
  FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
      mlir::linalg::inferContractionDims(op);
  if (failed(contractionDims)) {
    assert(IREE::LinalgExt::isaHorizontallyFusedContraction(op) &&
           "expected horizontally fused contraction op");
    SmallVector<AffineMap> indexingMaps;
    indexingMaps.push_back(op.getMatchingIndexingMap(op.getDpsInputOperand(0)));
    indexingMaps.push_back(op.getMatchingIndexingMap(op.getDpsInputOperand(1)));
    indexingMaps.push_back(op.getMatchingIndexingMap(op.getDpsInitOperand(0)));
    contractionDims = mlir::linalg::inferContractionDims(indexingMaps);
  }
  assert(succeeded(contractionDims) && "Could not infer contraction dims");

  if (contractionDims->k.size() < 1 || contractionDims->m.size() < 1 ||
      contractionDims->n.size() < 1) {
    return failure();
  }

  LLVM_DEBUG(debugPrintContractionInfo("Problem size", op.getNumLoops(),
                                       *contractionDims, bounds));

  // For now we are not being smart and trying to reshape dimensions to allow
  // for better usage of intrinsics, and instead are tiling all dimensions
  // except the inner most m, n, and k dimensions to 1.
  int64_t mDim = contractionDims->m.back();
  int64_t nDim = contractionDims->n.back();
  int64_t kDim = contractionDims->k.back();

  // Dynamic dims are expected to be taken care of earlier in the pipeline.
  if (ShapedType::isDynamic(bounds[mDim]) ||
      ShapedType::isDynamic(bounds[nDim]) ||
      ShapedType::isDynamic(bounds[kDim])) {
    return failure();
  }

  // Bail out on matvec-like cases.
  if (bounds[mDim] == 1 || bounds[nDim] == 1) {
    return failure();
  }

  Value lhs = op.getDpsInputOperand(0)->get();
  Value rhs = op.getDpsInputOperand(1)->get();
  Value init = op.getDpsInitOperand(0)->get();

  Type lhsElemType = getElementTypeOrSelf(lhs);
  Type rhsElemType = getElementTypeOrSelf(rhs);
  Type initElemType = getElementTypeOrSelf(init);

  if (auto lhsOp = lhs.getDefiningOp<linalg::GenericOp>()) {
    if (IREE::LinalgExt::isBitExtendOp(lhsOp))
      lhsElemType = getElementTypeOrSelf(lhsOp.getDpsInputs()[0]);
  }
  if (auto rhsOp = rhs.getDefiningOp<linalg::GenericOp>()) {
    if (IREE::LinalgExt::isBitExtendOp(rhsOp))
      rhsElemType = getElementTypeOrSelf(rhsOp.getDpsInputs()[0]);
  }

  SmallVector<int64_t> batchDims;
  for (int64_t batchDim : contractionDims->batch) {
    if (!ShapedType::isDynamic(bounds[batchDim])) {
      batchDims.push_back(batchDim);
    }
  }
  auto getDimBounds = [&](SmallVector<int64_t> dims) -> SmallVector<int64_t> {
    return llvm::map_to_vector(dims, [&](int64_t dim) { return bounds[dim]; });
  };

  // TODO(Max191): Support multiple M/N/K dimension problems for MMASchedules
  // once the pipeline is able to support it. After adding multiple dimensions,
  // all instances of schedule->m/nSubgroupCounts[0] and
  // schedule->m/n/kTileSizes[0] need to use the full list of sizes instead of
  // just the first element.
  GPUMatmulShapeType problem{
      {bounds[mDim]}, {bounds[nDim]}, {bounds[kDim]}, getDimBounds(batchDims),
      lhsElemType,    rhsElemType,    initElemType};

  // Helper fn to store mma information.
  auto storeMmaInfo = [](IREE::GPU::MmaInterfaceAttr mma,
                         SmallVector<GPUIntrinsicType> &intrinsics) {
    auto [mSize, nSize, kSize] = mma.getMNKShape();
    auto [aType, bType, cType] = mma.getABCElementTypes();
    intrinsics.emplace_back(mSize, nSize, kSize, aType, bType, cType, mma);
  };

  SmallVector<GPUIntrinsicType> intrinsics;
  intrinsics.reserve(target.getWgp().getMma().size());
  MLIRContext *context = op.getContext();
  for (IREE::GPU::MMAAttr mma : target.getWgp().getMma()) {
    if (mma.getSubgroupSize() != targetSubgroupSize)
      continue;
    storeMmaInfo(mma, intrinsics);
    // Store info on virtual intrinsics based on current mma if any
    for (IREE::GPU::VirtualMMAIntrinsic virtualIntrinsic :
         mma.getVirtualIntrinsics()) {
      auto virtualMma =
          IREE::GPU::VirtualMMAAttr::get(context, virtualIntrinsic);
      storeMmaInfo(virtualMma, intrinsics);
    }
  }

  if (intrinsics.empty())
    return failure();

  GPUMMAHeuristicSeeds seeds;

  // Note that the following heuristic seeds are just placeholder values.
  // We need to clean it up and make it adjusting to different targets.
  // See https://github.com/iree-org/iree/issues/16341 for details.
  if (problem.mSizes[0] * problem.nSizes[0] <= clGPUMatmulCThreshold) {
    // For matmuls with small M*N size, we want to distribute M*N onto more
    // workgroups to fill the GPU. Use a smaller bestMNTileCountPerSubgroup
    // and a larger bestKTileCountPerSubgroup.
    seeds = {/*bestSubgroupCountPerWorkgroup=*/4,
             /*bestMNTileCountPerSubgroup=*/4,
             /*bestKTileCountPerSubgroup=*/8};
  } else {
    seeds = {/*bestSubgroupCountPerWorkgroup=*/4,
             /*bestMNTileCountPerSubgroup=*/8,
             /*bestKTileCountPerSubgroup=*/4};
  }
  // Scale the seed by number of contractions of horizontally fused case.
  seeds.bestMNTileCountPerSubgroup /= op.getNumDpsInputs() - 1;

  int64_t maxSharedMemoryBytes = target.getWgp().getMaxWorkgroupMemoryBytes();

  LDBG("Matmul Vector Distribution Config");

  auto pipeline = CodeGenPipeline::LLVMGPUVectorDistribute;

  // Infer if lhs or rhs is transposed to help generate better schedule.
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  bool transposedLhs =
      kDim !=
      llvm::cast<AffineDimExpr>(maps[0].getResults().back()).getPosition();
  bool transposedRhs =
      nDim !=
      llvm::cast<AffineDimExpr>(maps[1].getResults().back()).getPosition();

  // First try to find a schedule with an exactly matching intrinsic.
  std::optional<GPUMMASchedule> schedule = deduceMMASchedule(
      problem, intrinsics, seeds, maxSharedMemoryBytes, targetSubgroupSize);
  if (!schedule) {
    // Then try again by allowing upcasting accumulator.
    schedule =
        deduceMMASchedule(problem, intrinsics, seeds, maxSharedMemoryBytes,
                          targetSubgroupSize, transposedLhs, transposedRhs,
                          /*canUpcastAcc=*/true);
  }

  if (!schedule) {
    LDBG("Failed to deduce MMA schedule");
    return failure();
  }

  LDBG("Target Subgroup size: " << targetSubgroupSize);
  LDBG("Schedule: " << schedule);

  int64_t flatWorkgroupSize =
      targetSubgroupSize *
      ShapedType::getNumElements(schedule->nSubgroupCounts) *
      ShapedType::getNumElements(schedule->mSubgroupCounts);
  std::array<int64_t, 3> workgroupSize{flatWorkgroupSize, 1, 1};

  SmallVector<int64_t> workgroupTileSizes(op.getNumLoops(), 0);
  SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 0);
  // Tile all batch dimensions with unit size.
  for (int64_t batch : contractionDims->batch) {
    workgroupTileSizes[batch] = 1;
  }

  // Tile all m, n, and k dimensions to 1 except the innermost. Unit dims
  // from this tiling are folded before vectorization.
  for (int64_t m : llvm::drop_end(contractionDims->m)) {
    workgroupTileSizes[m] = 1;
  }
  for (int64_t n : llvm::drop_end(contractionDims->n)) {
    workgroupTileSizes[n] = 1;
  }
  for (int64_t k : llvm::drop_end(contractionDims->k)) {
    reductionTileSizes[k] = 1;
  }

  // Compute the M/N dimension tile size by multiply subgroup information.
  workgroupTileSizes[mDim] =
      schedule->mSubgroupCounts[0] * schedule->mTileSizes[0] * schedule->mSize;
  workgroupTileSizes[nDim] =
      schedule->nSubgroupCounts[0] * schedule->nTileSizes[0] * schedule->nSize;

  reductionTileSizes[kDim] = schedule->kTileSizes[0] * schedule->kSize;

  LLVM_DEBUG(debugPrintContractionInfo("Workgroup tile sizes", op.getNumLoops(),
                                       *contractionDims, workgroupTileSizes));
  LLVM_DEBUG(debugPrintContractionInfo("Reduction tile sizes", op.getNumLoops(),
                                       *contractionDims, reductionTileSizes));

  Builder b(context);
  SmallVector<NamedAttribute, 2> attrs = {
      NamedAttribute("workgroup", b.getI64ArrayAttr(workgroupTileSizes)),
      NamedAttribute("reduction", b.getI64ArrayAttr(reductionTileSizes))};
  auto promotedOperands =
      llvm::to_vector(llvm::seq<int64_t>(op.getNumDpsInputs()));
  IREE::GPU::appendPromotedOperandsList(context, attrs, promotedOperands);
  IREE::GPU::setMmaKind(context, attrs, schedule->mmaKind);
  IREE::GPU::setSubgroupMCount(context, attrs, schedule->mSubgroupCounts[0]);
  IREE::GPU::setSubgroupNCount(context, attrs, schedule->nSubgroupCounts[0]);

  auto configDict = DictionaryAttr::get(context, attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  // Attach the MMA schedule as an attribute to the entry point export function
  // for later access in the pipeline.
  SmallVector<NamedAttribute, 1> pipelineAttrs;

  // Prefetch shared memory if requested.
  if (clLLVMGPUEnablePrefetch) {
    auto pipelineOptions = IREE::GPU::GPUPipelineOptionsAttr::get(
        context, /*prefetchSharedMemory=*/true,
        /*no_reduce_shared_memory_bank_conflicts=*/false,
        /*use_igemm_convolution=*/false,
        /*reorder_workgroups_strategy=*/std::nullopt);
    pipelineAttrs.emplace_back(
        StringAttr::get(context,
                        IREE::GPU::GPUPipelineOptionsAttr::getDictKeyName()),
        pipelineOptions);
  }

  auto pipelineConfig = DictionaryAttr::get(context, pipelineAttrs);

  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, loweringConfig, pipeline, workgroupSize,
      targetSubgroupSize, pipelineConfig);
}

static LogicalResult setAttentionIntrinsicBasedVectorDistributionConfig(
    IREE::GPU::TargetAttr target, mlir::FunctionOpInterface entryPoint,
    IREE::LinalgExt::AttentionOp op) {
  if (target.getWgp().getMma().empty())
    return failure();

  const int64_t targetSubgroupSize = target.getPreferredSubgroupSize();

  // Get iteration domain bounds.
  OpBuilder b(op);
  FailureOr<SmallVector<int64_t>> maybeBounds = op.getStaticLoopRanges();
  if (failed(maybeBounds)) {
    return failure();
  }

  ArrayRef<int64_t> bounds = maybeBounds.value();

  auto opInfo =
      IREE::LinalgExt::AttentionOpDetail::get(
          op.getQueryMap(), op.getKeyMap(), op.getValueMap(), op.getOutputMap())
          .value();

  int64_t mDim = opInfo.getMDims().back();
  int64_t k1Dim = opInfo.getK1Dims().back();
  int64_t k2Dim = opInfo.getK2Dims().back();
  int64_t nDim = opInfo.getNDims().back();

  // Dynamic dims are expected to be taken care of earlier in the pipeline.
  if (ShapedType::isDynamic(bounds[mDim]) ||
      ShapedType::isDynamic(bounds[k1Dim]) ||
      ShapedType::isDynamic(bounds[k2Dim]) ||
      ShapedType::isDynamic(bounds[nDim])) {
    return failure();
  }

  // Bail out on skinny attention.
  if (bounds[mDim] <= kVerySkinnyDimThreshold) {
    return failure();
  }

  Value qMatrix = op.getQuery();
  Value kMatrix = op.getKey();
  Value vMatrix = op.getValue();

  // Helper fn to store mma information.
  auto storeMmaInfo = [](IREE::GPU::MmaInterfaceAttr mma,
                         SmallVector<GPUIntrinsicType> &intrinsics) {
    auto [mSize, nSize, kSize] = mma.getMNKShape();
    auto [aType, bType, cType] = mma.getABCElementTypes();
    intrinsics.emplace_back(mSize, nSize, kSize, aType, bType, cType, mma);
  };

  SmallVector<GPUIntrinsicType> intrinsics;
  intrinsics.reserve(target.getWgp().getMma().size());
  MLIRContext *context = op.getContext();
  for (IREE::GPU::MMAAttr mma : target.getWgp().getMma()) {
    if (mma.getSubgroupSize() != targetSubgroupSize)
      continue;
    storeMmaInfo(mma, intrinsics);
    // Store info on virtual intrinsics based on current mma if any
    for (IREE::GPU::VirtualMMAIntrinsic virtualIntrinsic :
         mma.getVirtualIntrinsics()) {
      auto virtualMma =
          IREE::GPU::VirtualMMAAttr::get(context, virtualIntrinsic);
      storeMmaInfo(virtualMma, intrinsics);
    }
  }

  if (intrinsics.empty())
    return failure();

  // We assume that P uses the element type of V for input
  // and both matmuls have f32 as output. It is possible to use other element
  // types also.
  Type qElementType = getElementTypeOrSelf(qMatrix);
  Type kElementType = getElementTypeOrSelf(kMatrix);
  Type vElementType = getElementTypeOrSelf(vMatrix);
  Type f32Type = b.getF32Type();
  GPUMatmulShapeType qkMatmul{
      /*m=*/bounds[mDim],
      /*n=*/bounds[k2Dim],
      /*k=*/bounds[k1Dim],
      /*lhsType=*/qElementType,
      /*rhsType=*/kElementType,
      /*accType=*/f32Type,
  };
  GPUMatmulShapeType pvMatmul{/*m=*/bounds[mDim],
                              /*n=*/bounds[nDim],
                              /*k=*/bounds[k2Dim],
                              /*lhsType=*/vElementType,
                              /*rhsType=*/vElementType,
                              /*accType=*/f32Type};

  // TODO: Currently, we are forcing number of subgroups to be 1. This can be
  // fixed by teaching vector distribution chained matmul.
  GPUMMAHeuristicSeeds pvMatmulSeeds = {/*bestSubgroupCountPerWorkgroup=*/4,
                                        /*bestMNTileCountPerSubgroup=*/4,
                                        /*bestKTileCountPerSubgroup=*/4};

  LDBG("Attention Vector Distribution Config");

  // Infer if Q, K and V are transposed to help generate better schedule.
  bool transposedQ =
      k1Dim != llvm::cast<AffineDimExpr>(op.getQueryMap().getResults().back())
                   .getPosition();
  bool transposedK =
      k1Dim != llvm::cast<AffineDimExpr>(op.getKeyMap().getResults().back())
                   .getPosition();
  bool transposedV =
      k2Dim != llvm::cast<AffineDimExpr>(op.getValueMap().getResults().back())
                   .getPosition();

  int64_t maxSharedMemoryBytes = target.getWgp().getMaxWorkgroupMemoryBytes();

  // First try to find a schedule with an exactly matching intrinsic.
  std::optional<GPUMMASchedule> schedule = deduceAttentionSchedule(
      qkMatmul, pvMatmul, intrinsics, pvMatmulSeeds, maxSharedMemoryBytes,
      targetSubgroupSize, transposedQ, transposedK, transposedV);
  if (!schedule) {
    // Then try again by allowing upcasting accumulator.
    schedule = deduceAttentionSchedule(
        qkMatmul, pvMatmul, intrinsics, pvMatmulSeeds, maxSharedMemoryBytes,
        targetSubgroupSize, transposedQ, transposedK, transposedV,
        /*canUpcastAcc=*/true);
  }

  if (!schedule) {
    LDBG("Failed to deduce Attention schedule");
    return failure();
  }

  // TODO: Due to a bug in layout configuration, we cannot set warp count on
  // the N dimension. This is however ok, because we generally do not want to
  // distribute subgroups on N dimension anyway.
  if (schedule->nSubgroupCounts[0] != 1) {
    schedule->nTileSizes[0] *= schedule->nSubgroupCounts[0];
    schedule->nSubgroupCounts[0] = 1;
  }

  LDBG("Target Subgroup size: " << targetSubgroupSize);
  LDBG("Schedule: " << schedule);

  int64_t flatWorkgroupSize =
      targetSubgroupSize *
      ShapedType::getNumElements(schedule->nSubgroupCounts) *
      ShapedType::getNumElements(schedule->mSubgroupCounts);
  std::array<int64_t, 3> workgroupSize{flatWorkgroupSize, 1, 1};

  SmallVector<int64_t> workgroupTileSizes(opInfo.getDomainRank(), 0);
  SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 0);
  // Tile all batch dimensions with unit size.
  for (int64_t batch : opInfo.getBatchDims()) {
    workgroupTileSizes[batch] = 1;
  }

  // Tile all m, n, and k2 dimensions to 1 except the innermost. Unit dims
  // from this tiling are folded before vectorization. k1 dimension cannot be
  // tiled, so we leave it.
  for (int64_t m : llvm::drop_end(opInfo.getMDims())) {
    workgroupTileSizes[m] = 1;
  }
  for (int64_t n : llvm::drop_end(opInfo.getNDims())) {
    workgroupTileSizes[n] = 1;
  }
  for (int64_t k2 : llvm::drop_end(opInfo.getK2Dims())) {
    reductionTileSizes[k2] = 1;
  }

  // Compute the M/N dimension tile size by multiply subgroup information.
  workgroupTileSizes[mDim] =
      schedule->mSubgroupCounts[0] * schedule->mTileSizes[0] * schedule->mSize;
  workgroupTileSizes[nDim] =
      schedule->nSubgroupCounts[0] * schedule->nTileSizes[0] * schedule->nSize;

  reductionTileSizes[k2Dim] = schedule->kTileSizes[0] * schedule->kSize;

  SmallVector<NamedAttribute, 2> attrs = {
      NamedAttribute("workgroup", b.getI64ArrayAttr(workgroupTileSizes)),
      NamedAttribute("reduction", b.getI64ArrayAttr(reductionTileSizes))};
  IREE::GPU::appendPromotedOperandsList(context, attrs, {0, 1, 2});

  SmallVector<NamedAttribute, 2> qkConfig;
  SmallVector<NamedAttribute, 2> pvConfig;

  // On attention subgroup distribution:
  // The subgroup distribution in attention is controlled by the second matmul
  // (Parallel dimension distribution is usually (almost always) controlled by
  // the last reduction operation in a dispatch). Since VectorDistribution
  // doesn't have logic to set subgroup and thread layouts separately, we
  // explicitly set the subgroup count for the first matmul as well,
  // corresponding to what the second matmul dictates.

  // Configuring for qk matmul.
  // subgroup_n count for qk matmul is always 1, since we do not tile K1.
  IREE::GPU::appendPromotedOperandsList(context, qkConfig, {0, 1});
  IREE::GPU::setMmaKind(context, qkConfig, schedule->mmaKind);
  IREE::GPU::setSubgroupMCount(context, qkConfig, schedule->mSubgroupCounts[0]);
  IREE::GPU::setSubgroupNCount(context, qkConfig, 1);

  // Configuring for pv matmul.
  IREE::GPU::appendPromotedOperandsList(context, pvConfig, {1});
  IREE::GPU::setMmaKind(context, pvConfig, schedule->mmaKind);
  IREE::GPU::setSubgroupMCount(context, pvConfig, schedule->mSubgroupCounts[0]);
  IREE::GPU::setSubgroupNCount(context, pvConfig, schedule->nSubgroupCounts[0]);

  SmallVector<NamedAttribute, 2> qkAttrs;
  SmallVector<NamedAttribute, 2> pvAttrs;
  qkAttrs.emplace_back("attention_qk_matmul", b.getUnitAttr());
  pvAttrs.emplace_back("attention_pv_matmul", b.getUnitAttr());

  auto qkConfigDict = b.getDictionaryAttr(qkConfig);
  auto pvConfigDict = b.getDictionaryAttr(pvConfig);

  auto qkLoweringConfig =
      IREE::GPU::LoweringConfigAttr::get(context, qkConfigDict);
  auto pvLoweringConfig =
      IREE::GPU::LoweringConfigAttr::get(context, pvConfigDict);

  qkAttrs.emplace_back("lowering_config", qkLoweringConfig);
  pvAttrs.emplace_back("lowering_config", pvLoweringConfig);

  auto qkAttrDict = b.getDictionaryAttr(qkAttrs);
  auto pvAttrDict = b.getDictionaryAttr(pvAttrs);

  SmallVector<NamedAttribute, 2> decompositionConfig;
  decompositionConfig.emplace_back(
      b.getNamedAttr(IREE::LinalgExt::AttentionOp::getQKAttrStr(), qkAttrDict));
  decompositionConfig.emplace_back(
      b.getNamedAttr(IREE::LinalgExt::AttentionOp::getPVAttrStr(), pvAttrDict));

  DictionaryAttr decompositionConfigDict =
      b.getDictionaryAttr(decompositionConfig);

  auto configDict = b.getDictionaryAttr(attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  SmallVector<NamedAttribute, 1> pipelineAttrs;

  // TODO: We do not turn prefetching on even when requested by the prefetching
  // flag because there is a shared memory allocation the two matmuls, which
  // the prefetching pass cannot understand.

  auto pipelineConfig = DictionaryAttr::get(context, pipelineAttrs);

  // Set attention decomposition control config.
  op.setDecompositionConfigAttr(decompositionConfigDict);

  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, loweringConfig, CodeGenPipeline::LLVMGPUVectorDistribute,
      workgroupSize, targetSubgroupSize, pipelineConfig);
}

static IREE::GPU::Basis projectBasis(const IREE::GPU::Basis &basis,
                                     ArrayRef<int64_t> projectedDims) {
  // Projection simply involves projecting the mapping and keeping the counts.
  IREE::GPU::Basis projectedBasis;
  projectedBasis.counts = basis.counts;
  SetVector<int64_t> projected(projectedDims.begin(), projectedDims.end());
  for (auto [dim, map] : llvm::enumerate(basis.mapping)) {
    if (projected.contains(dim)) {
      continue;
    }
    projectedBasis.mapping.push_back(map);
  }
  return projectedBasis;
}

static LogicalResult
setAttentionVectorDistributionConfig(IREE::GPU::TargetAttr target,
                                     mlir::FunctionOpInterface entryPoint,
                                     IREE::LinalgExt::AttentionOp op) {
  // This configuration is not really smart right now. It just makes sure that
  // attention always compiles and tries to distribute workload on threads,
  // subgroups and workgroups as much as it can.
  // TODO: Update this configuration with target information, like the
  // WarpReduction pipeline does.
  const int64_t targetSubgroupSize = target.getPreferredSubgroupSize();

  // Get iteration domain bounds.
  OpBuilder b(op);
  FailureOr<SmallVector<int64_t>> maybeBounds = op.getStaticLoopRanges();
  if (failed(maybeBounds)) {
    return failure();
  }

  SmallVector<int64_t> bounds = maybeBounds.value();

  auto opInfo =
      IREE::LinalgExt::AttentionOpDetail::get(
          op.getQueryMap(), op.getKeyMap(), op.getValueMap(), op.getOutputMap())
          .value();

  // Distribute the 'available' resource to the basis on the given dimensions.
  // `currDim` tracks number of dims on which resources have already been
  // distributed (to keep track of order of dimension distribution).
  auto distributeDimensionsToBasisGreedily =
      [&bounds](int64_t available, ArrayRef<int64_t> dims,
                IREE::GPU::Basis &basis, int64_t &currDim) {
        // Iterate over dimensions and try to distribute resources over them.
        for (int64_t dim : dims) {
          // We iterate over the basis in a reverse dimension to get smaller
          // strides for inner dimensions.
          int64_t rCurrDim = basis.counts.size() - currDim - 1;
          ++currDim;
          // Keep track of the order the dimensions are distributed in.
          basis.mapping[dim] = rCurrDim;
          // Try to distribute the resources over the dimensions greedily.
          int64_t dimSize = bounds[dim];
          if (ShapedType::isDynamic(dimSize)) {
            // We do not distribute over dynamic dimensions yet. It's possible
            // to do it since we have masking, it's just not clear what
            // heuristic to use.
            basis.counts[rCurrDim] = 1;
            continue;
          }
          int64_t used = std::gcd(available, dimSize);
          available /= used;
          bounds[dim] /= used;
          basis.counts[rCurrDim] = used;
        }
        return available;
      };

  SmallVector<int64_t> workgroupTileSizes(opInfo.getDomainRank(), 0);
  SmallVector<int64_t> threadTileSizes(opInfo.getDomainRank(), 0);
  // Distribute all batch and M dimensions to workgroups. We are memory bound,
  // and we have enough unrolling from K1 and N dimensions to not need more.
  for (int64_t dim : opInfo.getBatchDims()) {
    workgroupTileSizes[dim] = 1;
    bounds[dim] = 1;
  }
  for (int64_t dim : opInfo.getMDims()) {
    workgroupTileSizes[dim] = 1;
    bounds[dim] = 1;
  }

  // For memory bound attention, per workgroup, we have input shapes:
  //
  // Q: 1x1 xK1
  // K: 1xK2xK1
  // V: 1xK2xN
  // O: 1x1 xN
  //
  // We only care about our read/write bandwidth, Q and O are too small for us
  // to care, so we focus most of our attention (pun not intended) on K and V.
  // We want to get good global reads on K and V.
  //
  // Due to different transpose layouts, we can have different optimal
  // distributions for K and V. Ideally, we would use something like data-tiling
  // to ensure a good read layout, which would look something like:
  //
  // K: batch_k2 X batch_k1 X
  //    subgroup_tile_K2 X
  //    thread_tile_K1 X thread_tile_K2 X
  //    vector_size_K1
  // V: batch_k2 X batch_n X
  //    subgroup_tile_K2 X
  //    thread_tile_N X thread_tile_K2 X
  //    vector_size_N
  //
  // but if we don't have that, for now, we assume a default layout (that will
  // work well), that has it's inner dimensions as:
  //
  // K : ... X K2_inner x K1
  // V : ... X K2_inner K N

  // Make thread tile sizes for K1 and N read 128bits.
  int64_t keyBitwidth =
      IREE::Util::getTypeBitWidth(getElementTypeOrSelf(op.getKey().getType()));
  int64_t valueBitwidth = IREE::Util::getTypeBitWidth(
      getElementTypeOrSelf(op.getValue().getType()));

  // TODO: Support more exotic bitwidths.
  assert(128 % keyBitwidth == 0);
  assert(128 % valueBitwidth == 0);

  int64_t keyVectorSize = 128 / keyBitwidth;
  int64_t valueVectorSize = 128 / valueBitwidth;
  threadTileSizes[opInfo.getK1Dims().back()] = keyVectorSize;
  bounds[opInfo.getK1Dims().back()] /= keyVectorSize;
  threadTileSizes[opInfo.getNDims().back()] = valueVectorSize;
  bounds[opInfo.getNDims().back()] /= valueVectorSize;

  IREE::GPU::Basis qkThreadBasis = {
      SmallVector<int64_t>(opInfo.getDomainRank(), 1),
      SmallVector<int64_t>(opInfo.getDomainRank())};
  IREE::GPU::Basis pvThreadBasis = {
      SmallVector<int64_t>(opInfo.getDomainRank(), 1),
      SmallVector<int64_t>(opInfo.getDomainRank())};

  int64_t qkRemainingThreads = targetSubgroupSize;

  // Distribute both basis on K2 equally.
  int64_t qkCurrDim = 0;
  qkRemainingThreads = distributeDimensionsToBasisGreedily(
      qkRemainingThreads, opInfo.getK2Dims(), qkThreadBasis, qkCurrDim);

  pvThreadBasis = qkThreadBasis;
  int64_t pvRemainingThreads = qkRemainingThreads;
  int64_t pvCurrDim = qkCurrDim;

  // If the target doesn't support subgroup shuffle, we should still be
  // distributing on threads. It's the backends problem to not use shuffles, and
  // instead use shared memory for reduction.

  // Distribute K1 on QK basis and N on nothing.
  qkRemainingThreads = distributeDimensionsToBasisGreedily(
      qkRemainingThreads, opInfo.getK1Dims(), qkThreadBasis, qkCurrDim);
  distributeDimensionsToBasisGreedily(1, opInfo.getNDims(), qkThreadBasis,
                                      qkCurrDim);
  // Distribute N on PV basis and K1 on nothing.
  pvRemainingThreads = distributeDimensionsToBasisGreedily(
      pvRemainingThreads, opInfo.getNDims(), pvThreadBasis, pvCurrDim);
  distributeDimensionsToBasisGreedily(1, opInfo.getK1Dims(), pvThreadBasis,
                                      pvCurrDim);

  // We already tiled B/M on workgroups, so it doesn't really matter how we
  // distribute them here.
  qkRemainingThreads = distributeDimensionsToBasisGreedily(
      qkRemainingThreads, opInfo.getBatchDims(), qkThreadBasis, qkCurrDim);
  qkRemainingThreads = distributeDimensionsToBasisGreedily(
      qkRemainingThreads, opInfo.getMDims(), qkThreadBasis, qkCurrDim);

  pvRemainingThreads = distributeDimensionsToBasisGreedily(
      pvRemainingThreads, opInfo.getBatchDims(), pvThreadBasis, pvCurrDim);
  pvRemainingThreads = distributeDimensionsToBasisGreedily(
      pvRemainingThreads, opInfo.getMDims(), pvThreadBasis, pvCurrDim);

  // Do not distribute on subgroups for now. We want to distribute the reduction
  // dimension on subgroups, but until the masked reduction work lands, we do
  // nothing.
  IREE::GPU::Basis subgroupBasis = {
      SmallVector<int64_t>(opInfo.getDomainRank(), 1),
      llvm::to_vector(llvm::seq<int64_t>(opInfo.getDomainRank()))};

  LDBG("QK Basis");
  LDBG("Thread Basis");
  LDBG(llvm::interleaved(qkThreadBasis.counts));
  LDBG(llvm::interleaved(qkThreadBasis.mapping));
  LDBG("Subgroup Basis");
  LDBG(llvm::interleaved(subgroupBasis.counts));
  LDBG(llvm::interleaved(subgroupBasis.mapping));
  LDBG("PV Basis");
  LDBG("Thread Basis");
  LDBG(llvm::interleaved(pvThreadBasis.counts));
  LDBG(llvm::interleaved(pvThreadBasis.mapping));
  LDBG("Subgroup Basis");
  LDBG(llvm::interleaved(subgroupBasis.counts));
  LDBG(llvm::interleaved(subgroupBasis.mapping));

  // Tile N parallel dimensions if they are to big to workgroups.
  for (int64_t dim : opInfo.getNDims()) {
    if (ShapedType::isDynamic(dim)) {
      workgroupTileSizes[dim] = 1;
    }
    if (bounds[dim] >= 128) {
      workgroupTileSizes[dim] = 128;
    }
  }

  // Tile remaining reduction dimensions to serial loops.
  SmallVector<int64_t> reductionTileSizes(opInfo.getDomainRank(), 0);
  for (int64_t dim : opInfo.getK2Dims()) {
    if (ShapedType::isDynamic(dim)) {
      reductionTileSizes[dim] = 1;
    }
    if (bounds[dim] != 1) {
      int64_t threadCount = qkThreadBasis.counts[qkThreadBasis.mapping[dim]];
      int64_t subgroupCount = subgroupBasis.counts[subgroupBasis.mapping[dim]];
      reductionTileSizes[dim] = threadCount * subgroupCount;
    }
  }

  int64_t flatWorkgroupSize =
      targetSubgroupSize * ShapedType::getNumElements(subgroupBasis.counts);
  std::array<int64_t, 3> workgroupSize{flatWorkgroupSize, 1, 1};

  MLIRContext *context = op.getContext();

  SmallVector<NamedAttribute, 2> attrs = {
      NamedAttribute("workgroup", b.getI64ArrayAttr(workgroupTileSizes)),
      NamedAttribute("partial_reduction",
                     b.getI64ArrayAttr(reductionTileSizes))};

  // Create projected QK thread tile sizes by removing N dimensions.
  SmallVector<int64_t> qkThreadTileSizes;
  for (auto [i, tile] : llvm::enumerate(threadTileSizes)) {
    if (llvm::find(opInfo.getNDims(), i) != opInfo.getNDims().end()) {
      continue;
    }
    qkThreadTileSizes.push_back(tile);
  }
  SmallVector<NamedAttribute> qkConfig = {
      NamedAttribute("thread", b.getI64ArrayAttr(qkThreadTileSizes))};
  IREE::GPU::setBasis(context, qkConfig, IREE::GPU::TilingLevel::Subgroup,
                      projectBasis(subgroupBasis, opInfo.getNDims()));
  IREE::GPU::setBasis(context, qkConfig, IREE::GPU::TilingLevel::Thread,
                      projectBasis(qkThreadBasis, opInfo.getNDims()));

  // Create projected QK thread tile sizes by removing N dimensions.
  SmallVector<int64_t> pvThreadTileSizes;
  for (auto [i, tile] : llvm::enumerate(threadTileSizes)) {
    if (llvm::find(opInfo.getK1Dims(), i) != opInfo.getK1Dims().end()) {
      continue;
    }
    pvThreadTileSizes.push_back(tile);
  }
  SmallVector<NamedAttribute> pvConfig = {
      NamedAttribute("thread", b.getI64ArrayAttr(pvThreadTileSizes))};
  IREE::GPU::setBasis(context, pvConfig, IREE::GPU::TilingLevel::Subgroup,
                      projectBasis(subgroupBasis, opInfo.getK1Dims()));
  IREE::GPU::setBasis(context, pvConfig, IREE::GPU::TilingLevel::Thread,
                      projectBasis(pvThreadBasis, opInfo.getK1Dims()));

  SmallVector<NamedAttribute, 2> qkAttrs;
  SmallVector<NamedAttribute, 2> pvAttrs;

  auto qkConfigDict = b.getDictionaryAttr(qkConfig);
  auto pvConfigDict = b.getDictionaryAttr(pvConfig);

  auto qkLoweringConfig =
      IREE::GPU::LoweringConfigAttr::get(context, qkConfigDict);
  auto pvLoweringConfig =
      IREE::GPU::LoweringConfigAttr::get(context, pvConfigDict);

  qkAttrs.emplace_back("lowering_config", qkLoweringConfig);
  pvAttrs.emplace_back("lowering_config", pvLoweringConfig);

  auto qkAttrDict = b.getDictionaryAttr(qkAttrs);
  auto pvAttrDict = b.getDictionaryAttr(pvAttrs);

  SmallVector<NamedAttribute, 2> decompositionConfig;
  decompositionConfig.emplace_back(IREE::LinalgExt::AttentionOp::getQKAttrStr(),
                                   qkAttrDict);
  decompositionConfig.emplace_back(IREE::LinalgExt::AttentionOp::getPVAttrStr(),
                                   pvAttrDict);

  // Set attention decomposition control config.
  op.setDecompositionConfigAttr(b.getDictionaryAttr(decompositionConfig));

  auto configDict = b.getDictionaryAttr(attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, loweringConfig, CodeGenPipeline::LLVMGPUVectorDistribute,
      workgroupSize, targetSubgroupSize);

  return success();
}

static LogicalResult
setVectorDistributionConfig(IREE::GPU::TargetAttr target,
                            mlir::FunctionOpInterface entryPoint,
                            Operation *computeOp) {
  // We haven't properly plumbed through MMA op layouts and conversions for CUDA
  // to target NVIDIA GPUs. So disable the vector distribution pass for it.
  if (!isROCmBackend(target))
    return failure();

  if (!clGPUEnableVectorDistribution) {
    LDBG("Vector Distribution not enabled, skipping...");
    return failure();
  }

  LDBG("VectorDistribution: finding a suitable config...");

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(computeOp)) {
    if (linalg::isaContractionOpInterface(linalgOp) ||
        IREE::LinalgExt::isaHorizontallyFusedContraction(linalgOp)) {
      LDBG("VectorDistribution: trying to find a suitable contraction config");
      return setMatmulVectorDistributionConfig(target, entryPoint, linalgOp);
    }
    if (linalg::isaConvolutionOpInterface(linalgOp)) {
      LDBG("VectorDistribution: trying to find a suitable convolution config");
      return setConvolutionVectorDistributionConfig(target, entryPoint,
                                                    linalgOp);
    }
  }

  if (auto attnOp = dyn_cast<IREE::LinalgExt::AttentionOp>(computeOp)) {
    LDBG("VectorDistribution: trying to find a suitable attention config");
    if (succeeded(setAttentionIntrinsicBasedVectorDistributionConfig(
            target, entryPoint, attnOp))) {
      return success();
    }
    return setAttentionVectorDistributionConfig(target, entryPoint, attnOp);
  }

  LDBG("VectorDistribution: failed to find a suitable config");
  return failure();
}

//====---------------------------------------------------------------------===//
// Contraction Pipeline Configuration
//====---------------------------------------------------------------------===//

static LogicalResult setContractConfig(IREE::GPU::TargetAttr target,
                                       mlir::FunctionOpInterface entryPoint,
                                       linalg::LinalgOp op) {
  if (!linalg::isaContractionOpInterface(op) || op.getNumParallelLoops() < 2) {
    return failure();
  }

  // Also exclude the case of matvec, which has only one non-unit parallel dim.
  // They should go down different pipelines.
  // Currently dynamic dimensions are tiled with size=1 in codegen.
  int staticNonUnitParallelDimCount = 0;
  SmallVector<int64_t> bounds = op.getStaticLoopRanges();
  FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
      mlir::linalg::inferContractionDims(op);
  assert(succeeded(contractionDims) && "Could not infer contraction dims");
  for (auto mDim : contractionDims->m) {
    staticNonUnitParallelDimCount +=
        bounds[mDim] != 1 && !ShapedType::isDynamic(bounds[mDim]);
  }
  for (auto nDim : contractionDims->n) {
    staticNonUnitParallelDimCount +=
        bounds[nDim] != 1 && !ShapedType::isDynamic(bounds[nDim]);
  }
  if (staticNonUnitParallelDimCount <= 1)
    return failure();

  // Don't consider operations that don't have a broadcast, those should go
  // through reductions.
  if (llvm::any_of(op.getIndexingMapsArray(),
                   [](AffineMap m) { return m.isPermutation(); })) {
    return failure();
  }

  // Send very skinny, {2-4}xNxK and Mx{2-4}xK, matmuls to the vector reduction
  // pipeline, similar to matvec. Note: Because of reassociation in the vector
  // reduction pipeline, this may lead to precision loss. If this ever becomes
  // an issue, we can hide this behind a flag.
  if (llvm::all_equal({contractionDims->m.size(), contractionDims->n.size(),
                       contractionDims->k.size(), size_t{1}}) &&
      contractionDims->batch.empty()) {
    int64_t mSize = bounds[contractionDims->m.front()];
    int64_t nSize = bounds[contractionDims->n.front()];
    int64_t preferredSubgroupSize = target.getPreferredSubgroupSize();
    if ((mSize <= kVerySkinnyDimThreshold &&
         (nSize > preferredSubgroupSize || ShapedType::isDynamic(nSize))) ||
        (nSize <= kVerySkinnyDimThreshold &&
         (mSize > preferredSubgroupSize || ShapedType::isDynamic(mSize)))) {
      return failure();
    }
  }

  // TODO: Properly rematerialize leading elementwise with shared memory
  // promotion.
  if (hasFusedLeadingOp(op)) {
    return failure();
  }

  auto setMatmulConfig = [&entryPoint, &op](int64_t tileX, int64_t tileY,
                                            int64_t tileK,
                                            ArrayRef<int64_t> workgroupSize,
                                            ArrayRef<int32_t> subgroupSizes,
                                            unsigned softwarePipelineDepth,
                                            CodeGenPipeline pipeline) {
    TileSizesListType tileSizes;
    unsigned numParallelLoops = op.getNumParallelLoops();
    unsigned numReductionLoops = op.getNumReductionLoops();
    SmallVector<int64_t> workgroupTileSizes(
        numParallelLoops + numReductionLoops, 1);
    workgroupTileSizes[numParallelLoops - 2] = tileX;
    workgroupTileSizes[numParallelLoops - 1] = tileY;

    SmallVector<unsigned> partitionedLoops =
        cast<PartitionableLoopsInterface>(op.getOperation())
            .getPartitionableLoops(/*maxNumPartitionedLoops=*/std::nullopt);
    llvm::SmallDenseSet<unsigned, 4> partitionedLoopsSet;
    partitionedLoopsSet.insert(partitionedLoops.begin(),
                               partitionedLoops.end());
    for (auto loopID : llvm::seq<unsigned>(0, numParallelLoops)) {
      if (!partitionedLoopsSet.count(loopID)) {
        workgroupTileSizes[loopID] = 0;
      }
    }

    std::optional<int64_t> subgroupSize = std::nullopt;
    if (!subgroupSizes.empty())
      subgroupSize = subgroupSizes.front();

    // For the LLVMGPUTileAndFuse pipeline, we need to split tile sizes
    // for workgroup, thread, and reduction.
    if (pipeline == CodeGenPipeline::LLVMGPUTileAndFuse) {

      auto context = op.getContext();
      Builder b(context);

      SmallVector<int64_t> threadTileSizes(numParallelLoops + numReductionLoops,
                                           0);
      std::fill(threadTileSizes.begin(),
                threadTileSizes.begin() + numParallelLoops, 1);

      threadTileSizes[numParallelLoops - 2] =
          (tileX / workgroupSize[0]) < 1 ? 1 : (tileX / workgroupSize[0]);
      threadTileSizes[numParallelLoops - 1] =
          (tileY / workgroupSize[1]) < 1 ? 1 : (tileY / workgroupSize[1]);

      SmallVector<int64_t> reductionTileSizes(
          numParallelLoops + numReductionLoops, 0);
      reductionTileSizes[numParallelLoops + numReductionLoops - 1] = tileK;

      SmallVector<NamedAttribute, 3> attrs = {
          NamedAttribute("workgroup", b.getI64ArrayAttr(workgroupTileSizes)),
          NamedAttribute("thread", b.getI64ArrayAttr(threadTileSizes)),
          NamedAttribute("reduction", b.getI64ArrayAttr(reductionTileSizes))};

      auto configDict = b.getDictionaryAttr(attrs);
      auto loweringConfig =
          IREE::GPU::LoweringConfigAttr::get(context, configDict);
      SmallVector<NamedAttribute, 1> pipelineAttrs;
      auto pipelineOptions = IREE::GPU::GPUPipelineOptionsAttr::get(
          context, /*prefetchSharedMemory=*/false,
          /*no_reduce_shared_memory_bank_conflicts=*/true,
          /*use_igemm_convolution=*/false,
          /*reorder_workgroups_strategy=*/std::nullopt);
      pipelineAttrs.emplace_back(
          b.getStringAttr(IREE::GPU::GPUPipelineOptionsAttr::getDictKeyName()),
          pipelineOptions);
      auto pipelineConfig = b.getDictionaryAttr(pipelineAttrs);

      return setOpConfigAndEntryPointFnTranslation(
          entryPoint, op, loweringConfig, pipeline, workgroupSize, subgroupSize,
          pipelineConfig);
    }

    // Other pipeline (MatmulTensorCore) expect the reduction tile size to be in
    // the same list.
    workgroupTileSizes[numParallelLoops + numReductionLoops - 1] = tileK;
    tileSizes.emplace_back(std::move(workgroupTileSizes));

    return setOpConfigAndEntryPointFnTranslation(
        entryPoint, op, tileSizes, pipeline, workgroupSize, subgroupSize,
        getSoftwarePipeliningAttrDict(op->getContext(), softwarePipelineDepth,
                                      /*softwarePipelineStoreStage=*/1));
  };
  // Infer the MxN size of the matmul based on operands and indexing maps.
  auto lhsShape =
      llvm::cast<ShapedType>(op.getDpsInputOperand(0)->get().getType())
          .getShape();
  auto rhsShape =
      llvm::cast<ShapedType>(op.getDpsInputOperand(1)->get().getType())
          .getShape();
  int64_t sizeM = ShapedType::kDynamic;
  int64_t sizeN = ShapedType::kDynamic;
  int64_t sizeK = ShapedType::kDynamic;
  auto outputMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
  for (unsigned i = 0; i < lhsShape.size(); i++) {
    if (op.getMatchingIndexingMap(op.getDpsInputOperand(0)).getDimPosition(i) ==
        outputMap.getDimPosition(outputMap.getNumResults() - 2)) {
      sizeM = lhsShape[i];
      break;
    }
  }
  for (unsigned i = 0; i < rhsShape.size(); i++) {
    if (op.getMatchingIndexingMap(op.getDpsInputOperand(1)).getDimPosition(i) ==
        outputMap.getDimPosition(outputMap.getNumResults() - 1)) {
      sizeN = rhsShape[i];
      break;
    }
  }
  SmallVector<unsigned> exprs;
  op.getReductionDims(exprs);
  if (exprs.size() == 1) {
    for (unsigned i = 0; i < lhsShape.size(); i++) {
      if (op.getMatchingIndexingMap(op.getDpsInputOperand(0))
              .getDimPosition(i) == exprs[0]) {
        sizeK = lhsShape[i];
        break;
      }
    }
  }
  bool isStaticSize = !ShapedType::isDynamic(sizeM) &&
                      !ShapedType::isDynamic(sizeN) &&
                      !ShapedType::isDynamic(sizeK);
  if (isStaticSize) {
    /// Try tensorcore config first.
    if (supportsTensorCore(target, op)) {
      SmallVector<TileWorkgroupSizePair> TCtileSizeConfig;
      Type elementType =
          cast<ShapedType>(op.getDpsInputOperand(0)->get().getType())
              .getElementType();

      getTensorCoreConfig(TCtileSizeConfig, elementType, sizeM, sizeN, sizeK);
      // Pick the best configuration where the original shape is aligned on the
      // tile size.
      for (TileWorkgroupSizePair &config : TCtileSizeConfig) {
        if (sizeK % config.tileSize[2] == 0 &&
            sizeN % config.tileSize[1] == 0 &&
            sizeM % config.tileSize[0] == 0) {
          CodeGenPipeline codegenPipeline = getTensorCorePipeline(elementType);
          return setMatmulConfig(
              config.tileSize[0], config.tileSize[1], config.tileSize[2],
              config.workgroupSize,
              target.getWgp().getSubgroupSizeChoices().asArrayRef(),
              sizeK == config.tileSize[2] ? 1 : config.pipelineDepth,
              codegenPipeline);
        }
      }
    }
    // Special case for very small matrices.
    if (sizeM * sizeN <= target.getPreferredSubgroupSize()) {
      return setMatmulConfig(
          sizeN, sizeM, 4, {sizeM, sizeN, 1},
          target.getWgp().getSubgroupSizeChoices().asArrayRef(),
          softwarePipelineDepthSimt, CodeGenPipeline::LLVMGPUTileAndFuse);
    }

    // SIMT matmul case. Query the best configuration.
    SmallVector<TileWorkgroupSizePair> tileSizeConfig = getMatmulConfig(target);
    // Pick the best configuration where the original shape is aligned on the
    // tile size.
    for (TileWorkgroupSizePair &config : tileSizeConfig) {
      if (sizeN % config.tileSize[1] == 0 && sizeM % config.tileSize[0] == 0 &&
          sizeK % config.tileSize[2] == 0) {
        return setMatmulConfig(
            config.tileSize[0], config.tileSize[1], config.tileSize[2],
            config.workgroupSize,
            target.getWgp().getSubgroupSizeChoices().asArrayRef(),
            softwarePipelineDepthSimt, CodeGenPipeline::LLVMGPUTileAndFuse);
      }
    }
  }
  // If we haven't found any config, use the best tile size hoping that
  // the workgroup specialization handles the main tile path efficiently.
  SmallVector<TileWorkgroupSizePair> tileSizeConfig = getMatmulConfig(target);
  constexpr size_t configIndex = 0;
  const TileWorkgroupSizePair &config = tileSizeConfig[configIndex];
  const int64_t tileX = config.tileSize[0];
  const int64_t tileY = config.tileSize[1];
  int64_t tileK = config.tileSize[2];
  // Since specialization doesn't work for K loop and peeling is not enabled yet
  // we pick a tileK size that is aligned on the K size.
  if (ShapedType::isDynamic(sizeK))
    tileK = 1;
  while (sizeK % tileK != 0) {
    tileK >>= 1;
  }
  const std::array<int64_t, 3> workgroupSize{config.workgroupSize[0],
                                             config.workgroupSize[1],
                                             config.workgroupSize[2]};
  return setMatmulConfig(tileX, tileY, tileK, workgroupSize,
                         target.getWgp().getSubgroupSizeChoices().asArrayRef(),
                         softwarePipelineDepthSimt,
                         CodeGenPipeline::LLVMGPUTileAndFuse);
}

//====---------------------------------------------------------------------===//
// FFT Pipeline Configuration
//====---------------------------------------------------------------------===//

static LogicalResult setFftConfig(IREE::GPU::TargetAttr target,
                                  mlir::FunctionOpInterface entryPoint,
                                  IREE::LinalgExt::FftOp op) {
  auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
  unsigned loopDepth = partitionedLoops.back() + 1;
  SmallVector<int64_t> workgroupTileSize(loopDepth, 0);
  SmallVector<int64_t, 3> workgroupSize = {target.getPreferredSubgroupSize(), 1,
                                           1};

  // Tiling along partitioned loops with size 1.
  for (int64_t loopIndex : partitionedLoops) {
    workgroupTileSize[loopIndex] = 1;
  }
  auto rank = op.getOperandRank();
  if (workgroupTileSize.size() >= rank && workgroupTileSize[rank - 1] != 0) {
    APInt value;
    if (matchPattern(op.getStage(), m_ConstantInt(&value))) {
      workgroupTileSize[rank - 1] = 1ll << value.getSExtValue();
    } else {
      op.emitError("non-constant stage might not work for fft op");
      return failure();
    }
  }
  TileSizesListType tileSizes = {workgroupTileSize};
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, tileSizes, CodeGenPipeline::LLVMGPUDistribute,
      workgroupSize);
}

//===----------------------------------------------------------------------===//
// Winograd Pipeline Configuration
//===----------------------------------------------------------------------===//
template <typename WinogradOp>
static LogicalResult setWinogradOpConfig(IREE::GPU::TargetAttr target,
                                         mlir::FunctionOpInterface entryPoint,
                                         WinogradOp op) {
  static_assert(
      llvm::is_one_of<WinogradOp, IREE::LinalgExt::WinogradInputTransformOp,
                      IREE::LinalgExt::WinogradFilterTransformOp,
                      IREE::LinalgExt::WinogradOutputTransformOp>::value,
      "expected winograd transform op");
  auto pipeline = CodeGenPipeline::LLVMGPUWinogradVectorize;
  TileSizesListType tileSizes;
  std::array<int64_t, 3> workgroupSize = {32, 4, 4};
  int64_t iterationRank = op.getIterationDomainRank();
  SmallVector<int64_t> workgroupTileSizes(iterationRank, 4);
  // Set batch workgroup size
  workgroupTileSizes.front() = 1;
  // Set input channel workgroup size
  workgroupTileSizes.back() = 32;
  if (isa<IREE::LinalgExt::WinogradFilterTransformOp>(op)) {
    // Set input channel workgroup size
    workgroupTileSizes.front() = 32;
    // Set output channel workgroup size
    workgroupTileSizes.back() = 16;
    workgroupSize = {16, 32, 1};
  }
  tileSizes.push_back(workgroupTileSizes);
  SmallVector<int64_t> threadTileSizes(iterationRank, 1);
  tileSizes.push_back(threadTileSizes);
  return setOpConfigAndEntryPointFnTranslation(entryPoint, op, tileSizes,
                                               pipeline, workgroupSize);
}

//====---------------------------------------------------------------------===//
// Sort Pipeline Configuration
//====---------------------------------------------------------------------===//

static LogicalResult setSortConfig(IREE::GPU::TargetAttr target,
                                   mlir::FunctionOpInterface entryPoint,
                                   Operation *op) {
  TileSizesListType tileSizes;
  auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
  if (partitionedLoops.empty()) {
    tileSizes.push_back({});
    return setOpConfigAndEntryPointFnTranslation(
        entryPoint, op, tileSizes, CodeGenPipeline::LLVMGPUDistribute,
        {1, 1, 1});
  }
  size_t numLoops = partitionedLoops.back() + 1;
  // To get peak occupancy we need a workgroup size of at least two warps
  std::array<int64_t, 3> workgroupSize = {2 * target.getPreferredSubgroupSize(),
                                          1, 1};
  SmallVector<int64_t> workgroupTileSizes(numLoops, 1);
  // Set all non-parallel loops to zero tile size.
  llvm::DenseSet<unsigned> partitionedLoopsSet(partitionedLoops.begin(),
                                               partitionedLoops.end());
  for (auto depth : llvm::seq<int64_t>(0, numLoops)) {
    if (!partitionedLoopsSet.count(depth)) {
      workgroupTileSizes[depth] = 0;
    }
  }

  // Tile to have one element per thread.
  for (int64_t depth = numLoops; depth > 0; depth--) {
    if (partitionedLoopsSet.count(depth - 1)) {
      workgroupTileSizes[depth - 1] = workgroupSize[0];
      break;
    }
  }
  tileSizes.emplace_back(std::move(workgroupTileSizes)); // Workgroup level
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, tileSizes, CodeGenPipeline::LLVMGPUDistribute,
      workgroupSize);
}

//====---------------------------------------------------------------------===//
// Default Pipeline Configuration
//====---------------------------------------------------------------------===//

// Basic default properties for linalg ops that haven't been tuned.
static LogicalResult setRootDefaultConfig(IREE::GPU::TargetAttr target,
                                          mlir::FunctionOpInterface entryPoint,
                                          Operation *op) {
  CodeGenPipeline passPipeline = CodeGenPipeline::LLVMGPUDistribute;
  TileSizesListType tileSizes;
  auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
  auto partitionedLoops = interfaceOp.getPartitionableLoops(std::nullopt);
  if (partitionedLoops.empty()) {
    tileSizes.push_back({});
    return setOpConfigAndEntryPointFnTranslation(entryPoint, op, tileSizes,
                                                 passPipeline, {1, 1, 1});
  }

  const int preferredSubgroupSize = target.getPreferredSubgroupSize();
  size_t numLoops = partitionedLoops.back() + 1;
  // To get peak occupancy we need a workgroup size of at least two warps.
  std::array<int64_t, 3> workgroupSize = {2 * preferredSubgroupSize, 1, 1};
  unsigned vectorSize = 4;
  SmallVector<int64_t> workgroupTileSizes(numLoops, 1);
  // Set all non-parallel loops to zero tile size.
  llvm::DenseSet<unsigned> partitionedLoopsSet(partitionedLoops.begin(),
                                               partitionedLoops.end());
  for (auto depth : llvm::seq<int64_t>(0, numLoops)) {
    if (!partitionedLoopsSet.count(depth)) {
      workgroupTileSizes[depth] = 0;
    }
  }
  int64_t skipInnerTiling = 0;
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    for (auto [index, outputOperand] :
         llvm::enumerate(genericOp.getDpsInitsMutable())) {
      if (!genericOp.getMatchingIndexingMap(&outputOperand)
               .isProjectedPermutation()) {
        vectorSize = 1;
        break;
      }
      ArrayRef<int64_t> shape =
          llvm::cast<ShapedType>(outputOperand.get().getType()).getShape();
      if (ShapedType::isDynamicShape(shape)) {
        vectorSize = 1;
        break;
      }
      // Since we vectorize along the most inner dimension, make sure if can be
      // divided by number of threads * vectorSize.
      while (vectorSize > 1 &&
             shape.back() % (workgroupSize[0] * vectorSize) != 0) {
        vectorSize /= 2;
      }
      if (vectorSize == 1) // assume there is fastpath + slowpath
        vectorSize = 4;
      int64_t problemSize = std::accumulate(
          shape.begin(), shape.end(), 1,
          [](const int64_t &a, const int64_t &b) { return a * b; });
      if ((problemSize / (preferredSubgroupSize * vectorSize)) < 64) {
        vectorSize = 1;
        break;
      }
      // If the inner dimension is too small to have one element per thread
      // reduce the workgroup size try to distribute amongst more dimensions.
      if (shape.back() < vectorSize * workgroupSize[0]) {
        int64_t flatWG = workgroupSize[0];
        vectorSize = 1;
        int64_t id = 0;
        for (int64_t dim : llvm::reverse(shape)) {
          // Unit loops are already skipped.
          if (dim == 1)
            continue;
          if (dim < flatWG) {
            skipInnerTiling++;
            workgroupSize[id] = dim;
          } else {
            workgroupSize[id] = flatWG;
            break;
          }
          flatWG = flatWG / dim;
          id++;
          if (flatWG <= 1 || id >= workgroupSize.size())
            break;
        }
        break;
      }
    }
  }

  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  // Pick a vectorSize of 1 for op that we know won't get vectorized.
  // Also skip vectorization for linalg on memref (no result) as the pipeline
  // relies on tensor level tiling.
  // TODO(thomasraoux): This could be improved by checking if the linalg op
  // would fail vectorization.
  if (!linalgOp || op->getNumResults() != 1 ||
      llvm::any_of(linalgOp.getIndexingMapsArray(),
                   [](AffineMap m) { return !m.isProjectedPermutation(); })) {
    vectorSize = 1;
  } else {
    passPipeline = CodeGenPipeline::LLVMGPUVectorize;
  }

  int64_t id = 0;
  // Set the inner most parallel loop to `lowerTs`.
  for (int64_t depth = numLoops; depth > 0; depth--) {
    if (partitionedLoopsSet.count(depth - 1)) {
      if (skipInnerTiling > 0) {
        // For dimensions that don't need to be distributed across blocks skip
        // tiling by setting tile size to 0.
        workgroupTileSizes[depth - 1] = 0;
        skipInnerTiling--;
        id++;
        if (id >= workgroupSize.size())
          break;
        continue;
      }
      workgroupTileSizes[depth - 1] = workgroupSize[id] * vectorSize;
      break;
    }
  }

  if (linalgOp) {
    // Tile reduction dimension to 4 to allow doing load4 if the reduction size
    // is the most inner dimension.
    workgroupTileSizes.append(linalgOp.getNumReductionLoops(), 4);
  }
  tileSizes.emplace_back(std::move(workgroupTileSizes)); // Workgroup level
  return setOpConfigAndEntryPointFnTranslation(entryPoint, op, tileSizes,
                                               passPipeline, workgroupSize,
                                               preferredSubgroupSize);
}

/// Returns true if it's MatVec like i.e., either the bound of M or N dim = 1,
/// or one of M, N dim isn't present.
static bool isMatvecLike(linalg::LinalgOp linalgOp) {

  SmallVector<int64_t> bounds = linalgOp.getStaticLoopRanges();
  SmallVector<unsigned> parallelDims;
  linalgOp.getParallelDims(parallelDims);

  // Validate that there's exactly one parallel dimension with size != 1.
  unsigned nonUnitParallelDimsCount = llvm::count_if(
      parallelDims, [&bounds](unsigned idx) { return bounds[idx] != 1; });

  // No. of parallel dims size shouldn't exceed 2.
  // There should be exactly one reduction loop.
  if (parallelDims.size() > 2 || nonUnitParallelDimsCount != 1 ||
      linalgOp.getNumReductionLoops() != 1) {
    return false;
  }

  // TODO: Allow for matvec with fused dequantization.
  FailureOr<linalg::ContractionDimensions> dims =
      linalg::inferContractionDims(linalgOp);
  if (failed(dims))
    return false;

  // TODO: Support batch matvec.
  if (!dims->batch.empty())
    return false;

  if (dims->m.size() >= 2 || dims->n.size() >= 2 ||
      !llvm::hasSingleElement(dims->k)) {
    return false;
  }

  return true;
}

//====---------------------------------------------------------------------===//
// Warp Reduction Pipeline Configuration
//====---------------------------------------------------------------------===//

/// Set the configuration for reductions that can be mapped to warp reductions.
static LogicalResult
setWarpReductionConfig(IREE::GPU::TargetAttr target,
                       mlir::FunctionOpInterface entryPoint,
                       linalg::LinalgOp op) {
  if (!target.supportsSubgroupShuffle())
    return failure();

  SmallVector<unsigned> parallelDims;
  SmallVector<unsigned> reductionDims;
  op.getParallelDims(parallelDims);
  op.getReductionDims(reductionDims);

  SmallVector<int64_t> bounds = op.getStaticLoopRanges();
  int64_t numParallelDims = op.getNumParallelLoops();

  if (reductionDims.empty())
    return failure();

  // Make sure reduction dimensions are static and innermost ones.
  int64_t numDynamicReductionDims = 0;
  for (unsigned dim : reductionDims) {
    if (ShapedType::isDynamic(bounds[dim])) {
      numDynamicReductionDims++;
    }
    if (dim < numParallelDims) {
      return failure();
    }
  }
  int numDynamicDims = llvm::count_if(bounds, ShapedType::isDynamic);

  // Distribution of multi-dim masked writes currently aren't fully supported.
  if (numDynamicReductionDims > 1) {
    return failure();
  }

  if (op.getRegionOutputArgs().size() != 1)
    return failure();

  // Only support projected permutation, this could be extended to projected
  // permutated with broadcast.
  if (llvm::any_of(op.getDpsInputOperands(), [&](OpOperand *input) {
        return !op.getMatchingIndexingMap(input).isProjectedPermutation();
      }))
    return failure();

  bool foundSingleReductionOutput = false;
  for (auto [index, initOpOperand] : llvm::enumerate(op.getDpsInitsMutable())) {
    // Only single combiner operations are supported for now.
    SmallVector<Operation *> combinerOps;
    if (matchReduction(op.getRegionOutputArgs(), index, combinerOps) &&
        combinerOps.size() == 1) {
      if (foundSingleReductionOutput)
        return failure();
      foundSingleReductionOutput = true;
      continue;
    }
    if (!op.getMatchingIndexingMap(&initOpOperand).isIdentity())
      return failure();
  }
  if (!foundSingleReductionOutput)
    return failure();

  SmallVector<int64_t> workgroupTileSizes(op.getNumParallelLoops(), 1);

  int64_t reductionSize = 1;
  for (int64_t dim : reductionDims)
    reductionSize *= bounds[dim];

  int64_t subgroupSize = 0;
  for (int s : target.getWgp().getSubgroupSizeChoices().asArrayRef()) {
    if (reductionSize % s == 0) {
      subgroupSize = s;
      break;
    }
  }
  if (subgroupSize == 0)
    return failure();

  // Without any bounds on dynamic dims, we need specialization to
  // get peak performance. For now, just use the warp size.
  if (numDynamicDims > 0) {
    SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 0);
    int64_t preferredSubgroupSize = target.getPreferredSubgroupSize();
    // We should set the subgroup size on:
    // Priority 1: The innermost reduction dimension with static shapes.
    // Priority 2: If there's no reduction dimension with static shapes
    // then the innermost reduction dim.
    unsigned lastNonDynamicReductionDim = reductionDims.back();
    if (reductionDims.size() > 1) {
      for (unsigned dim : reductionDims) {
        if (ShapedType::isDynamic(bounds[dim])) {
          reductionTileSizes[dim] = 1;
        } else {
          lastNonDynamicReductionDim = dim;
        }
      }
    }
    reductionTileSizes[lastNonDynamicReductionDim] = preferredSubgroupSize;
    TileSizesListType tileSizes;
    tileSizes.emplace_back(std::move(workgroupTileSizes)); // Workgroup level
    tileSizes.emplace_back(std::move(reductionTileSizes)); // Reduction level
    std::array<int64_t, 3> workgroupSize = {preferredSubgroupSize, 1, 1};
    if (failed(setOpConfigAndEntryPointFnTranslation(
            entryPoint, op, tileSizes, CodeGenPipeline::LLVMGPUWarpReduction,
            workgroupSize, preferredSubgroupSize))) {
      return failure();
    }
    return success();
  }

  const Type elementType =
      llvm::cast<ShapedType>(op.getDpsInitOperand(0)->get().getType())
          .getElementType();
  if (!elementType.isIntOrFloat())
    return failure();
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  // Reduction distribution only supports 8/16/32 bit types now.
  if (bitWidth != 32 && bitWidth != 16 && bitWidth != 8)
    return failure();

  const unsigned largestLoadSizeInBits = 128;
  unsigned vectorSize = largestLoadSizeInBits / bitWidth;
  while ((reductionSize / vectorSize) % subgroupSize != 0)
    vectorSize /= 2;

  // Deduce the workgroup size we should use for reduction. Currently a
  // workgroup processes all elements in reduction dimensions. Need to make sure
  // the workgroup size we use can divide the total reduction size, and it's
  // also within hardware limitations.
  const int64_t maxWorkgroupSize = 1024;
  int64_t groupSize = reductionSize / vectorSize;
  if (groupSize > maxWorkgroupSize) {
    groupSize = llvm::APIntOps::GreatestCommonDivisor(
                    {64, uint64_t(groupSize)}, {64, uint64_t(maxWorkgroupSize)})
                    .getZExtValue();
  }

  // Then we need to strike a balance--
  // 1) parallel dimensions are distributed to workgroups. If there are many
  //    workgroups dispatched, we'd want to have each GPU core hosting multiple
  //    of them for occupancy.
  // 2) we want each thread to read quite a few 128-bit vectors for better
  //    memory cache behavior.
  // Both means we cannot use a too large workgroup size.

  std::optional<int64_t> parallelSize = 1;
  for (int64_t dim : parallelDims) {
    if (ShapedType::isDynamic(bounds[dim])) {
      parallelSize = std::nullopt;
      break;
    }
    *parallelSize *= bounds[dim];
  }
  // Total parallel size that can fill the GPU with enough workgorups.
  // TODO: query from the target device; roughly 2x hardware compute unit.
  const int parallelThreshold = 256;
  // How many 128-bit vectors each thread should at least read.
  const int targetVectorCount = 8;
  while (parallelSize && *parallelSize > parallelThreshold &&
         (groupSize / 2) % subgroupSize == 0 &&
         reductionSize / (groupSize * vectorSize) < targetVectorCount) {
    // Use less subgroups per workgroup..
    groupSize /= 2;
    // in order to host more workgroups per hardware compute unit.
    *parallelSize /= 2;
  }

  // Current warp reduction pattern is a two step butterfly warp reduce.
  // First, do warp reductions along multiple subgroups.
  // Second, reduce results from multiple subgroups using single warp reduce.
  // The final warp reduce requires subgroup count <= subgroup size to work.
  if ((groupSize / subgroupSize) > subgroupSize)
    return failure();

  // With just one subgroup per workgroup, make each subgroup do more work and
  // process a few reductions (rows) along the last parallel dimension.
  //
  // TODO: This is enabled for matvec on ROCm for now. We should
  // validate this strategy and extend to more linalg generics and to CUDA.
  if (isROCmBackend(target) && !ShapedType::isDynamicShape(bounds) &&
      isMatvecLike(op)) {
    int64_t parallelIdx = *llvm::find_if(
        parallelDims, [&](int64_t currIdx) { return bounds[currIdx] != 1; });
    int64_t parallelBound = bounds[parallelIdx];
    int64_t numParallelReductions = 1;
    const int64_t maxParallelFactor = groupSize / 4;
    for (int64_t parallelFactor = 2; (parallelFactor < maxParallelFactor) &&
                                     (parallelBound % parallelFactor == 0) &&
                                     (parallelBound > parallelFactor);
         parallelFactor *= 2) {
      numParallelReductions = parallelFactor;
    }
    workgroupTileSizes[parallelIdx] = numParallelReductions;
  }

  std::array<int64_t, 3> workgroupSize = {groupSize, 1, 1};
  SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 0);
  int64_t remainingGroupSize = groupSize;
  for (int i = reductionDims.size() - 1; i >= 0; --i) {
    int64_t dim = reductionDims[i];
    int64_t bound = bounds[dim];
    if (i == reductionDims.size() - 1)
      bound /= vectorSize;
    APInt size = llvm::APIntOps::GreatestCommonDivisor(
        {64, uint64_t(remainingGroupSize)}, {64, uint64_t(bound)});
    reductionTileSizes[dim] = size.getSExtValue();
    if (i == reductionDims.size() - 1)
      reductionTileSizes[dim] *= vectorSize;
    remainingGroupSize /= size.getSExtValue();
  }
  TileSizesListType tileSizes;
  tileSizes.emplace_back(std::move(workgroupTileSizes)); // Workgroup level
  tileSizes.emplace_back(std::move(reductionTileSizes)); // Reduction level
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, tileSizes, CodeGenPipeline::LLVMGPUWarpReduction,
      workgroupSize, subgroupSize);
  return success();
}

static bool hasTwoOrThreeLoopsInfo(linalg::LinalgOp linalgOp) {
  return linalgOp.getNumParallelLoops() >= 2 &&
         linalgOp.getNumParallelLoops() <= 3;
}

//====---------------------------------------------------------------------===//
// Transpose Pipeline Configuration
//====---------------------------------------------------------------------===//

static LogicalResult setTransposeConfig(mlir::FunctionOpInterface entryPoint,
                                        linalg::LinalgOp linalgOp) {
  LinalgOpInfo opInfo(linalgOp, sharedMemTransposeFilter);

  // Checks preconditions for shared mem transpose.
  if (!opInfo.isTranspose() || opInfo.isDynamic() || opInfo.isReduction() ||
      !isa<linalg::GenericOp>(linalgOp) || !hasTwoOrThreeLoopsInfo(linalgOp)) {
    return failure();
  }

  ArrayRef<OpOperand *> transposedOperands = opInfo.getTransposeOperands();

  // Determine the fastest moving dimensions for the source/destination indices
  // of each transpose. These inform the tile sizes.
  int64_t outputFastestDim = linalgOp.getNumLoops() - 1;
  int64_t inputFastestDim =
      linalgOp.getMatchingIndexingMap(transposedOperands[0])
          .getDimPosition(outputFastestDim);
  // Ensure the other transposed operands match
  for (int i = 1; i < transposedOperands.size(); ++i) {
    if (inputFastestDim !=
        linalgOp.getMatchingIndexingMap(transposedOperands[i])
            .getDimPosition(outputFastestDim)) {
      return failure();
    }
  }

  int32_t tileM = 32;
  int32_t tileN = 32;
  TileSizesListType tileSizes;
  // Set all tile sizes to 1 except for fastest moving dimensions.
  SmallVector<int64_t> tileSizesTemp(linalgOp.getNumLoops(), 1);
  tileSizesTemp[outputFastestDim] = 32;
  tileSizesTemp[inputFastestDim] = 32;
  tileSizes.push_back(tileSizesTemp);

  // Check alignment with tile size for each transpose. Only the fastest moving
  // dims need to match the transpose tile.
  auto loopRanges = linalgOp.getStaticLoopRanges();
  if (loopRanges[outputFastestDim] % tileM != 0 ||
      loopRanges[inputFastestDim] % tileN != 0) {
    return failure();
  }

  // Workgroup size contains 8 warps. Configured with 8 threads on fastest
  // moving dimension so each thread can execute a vectorized copy of 4
  // contiguous elements at a time from the 32 block.
  std::array<int64_t, 3> workgroupSize = {8, 32, 1};

  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, linalgOp, tileSizes,
      CodeGenPipeline::LLVMGPUTransposeSharedMem, workgroupSize);
}

//====---------------------------------------------------------------------===//
// UKernel Pipeline Configuration
//====---------------------------------------------------------------------===//

/// Set the configuration for argmax when ukernels are enabled.
/// Distribute all parallel dim across different workgroups, and only use single
/// subgroup per workgroup.
static LogicalResult setArgmaxUkernelConfig(
    IREE::GPU::TargetAttr target, mlir::FunctionOpInterface entryPoint,
    linalg::GenericOp op, IREE::GPU::UKernelConfigAttr ukernelConfig) {
  SmallVector<unsigned> parallelDims;
  SmallVector<unsigned> reductionDims;
  op.getParallelDims(parallelDims);
  op.getReductionDims(reductionDims);

  // Currently Argmax UKernel only support 1 reduction dim.
  if (reductionDims.size() != 1)
    return failure();

  // Make sure reduction dimensions are static and innermost ones.
  SmallVector<int64_t> bounds = op.getStaticLoopRanges();
  int64_t numParallelDims = op.getNumParallelLoops();
  int64_t numDynamicReductionDims = 0;
  for (unsigned dim : reductionDims) {
    if (ShapedType::isDynamic(bounds[dim])) {
      numDynamicReductionDims++;
    }
    if (dim < numParallelDims) {
      return failure();
    }
  }

  // Distribution of multi-dim masked writes currently aren't fully supported.
  if (numDynamicReductionDims > 1) {
    return failure();
  }

  // Tile all the parallel dimension to 1. This is a requirement of the ukernel.
  SmallVector<unsigned> partitionedLoops =
      cast<PartitionableLoopsInterface>(op.getOperation())
          .getPartitionableLoops(kNumMaxParallelDims);
  size_t numLoops = partitionedLoops.empty() ? 0 : partitionedLoops.back() + 1;
  SmallVector<int64_t> workgroupTileSizes(numLoops, 1);

  // Currently Argmax Ukernel lets every thread reduce reductionDim/WarpSize
  // number of elements, and then it does a single step butterfly warp reduce.
  // Hence it expects workgroupSize to be warpSize(subgroupSize), and
  // reductionTileSize to be size of the reduction dim.
  SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 0);
  int64_t preferredSubgroupSize = target.getPreferredSubgroupSize();
  reductionTileSizes[reductionDims[0]] = preferredSubgroupSize;
  std::array<int64_t, 3> workgroupSize = {preferredSubgroupSize, 1, 1};

  MLIRContext *context = op->getContext();
  Builder b(context);
  SmallVector<NamedAttribute, 3> attrs = {
      NamedAttribute("workgroup", b.getI64ArrayAttr(workgroupTileSizes)),
      NamedAttribute("reduction", b.getI64ArrayAttr(reductionTileSizes)),
      NamedAttribute("ukernel", ukernelConfig)};
  IREE::GPU::appendPromotedOperandsList(context, attrs, {0, 1});
  auto configDict = DictionaryAttr::get(context, attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);
  if (failed(setOpConfigAndEntryPointFnTranslation(
          entryPoint, op, loweringConfig, CodeGenPipeline::LLVMGPUDefault,
          workgroupSize))) {
    return failure();
  }
  return success();
}

/// Decides the tiling and distribution parameters for one convolution
/// dimension. Returns true if we can successfully deduce.
///
/// - `inputDim` is the size of the dimension to be distributed.
/// - `residualThreads` is the remaining threads we can distribute.
/// - `residualTilingFactor` indicates the remaining tiling scale factor.
/// - `wgDimSize` will be updated with the decided workgroup dimension size.
/// - `wgTileSize` will be updated with the decided workgroup tile size.
/// - `invoTileSize` will be updated with the decided invocation tile size.
static bool distributeToOneDim(const int64_t inputDim,
                               const bool isInnerMostDim,
                               int64_t &residualThreads,
                               int64_t &residualTilingFactor,
                               int64_t &wgDimSize, int64_t &wgTileSize) {
  const int64_t lb = isInnerMostDim ? 2 : 1;
  for (int64_t dim = residualThreads; dim >= lb; dim >>= 1) {
    int64_t chosenTileSize = 0;
    if (isInnerMostDim) {
      // Handle 4 elements per thread for the innermost dimension. We need
      // this for vectorized load.
      chosenTileSize = 4;
      if (inputDim % (dim * chosenTileSize) != 0)
        continue;
    } else {
      for (int64_t t = residualTilingFactor; t >= 1; t >>= 1)
        if (inputDim % (dim * t) == 0) {
          chosenTileSize = t;
          break;
        }
    }
    if (chosenTileSize) {
      wgDimSize = dim;
      wgTileSize = dim * chosenTileSize;
      residualThreads /= dim;
      residualTilingFactor /= chosenTileSize;
      return true;
    }
  }
  return false;
};

/// Decides the tiling and distribution parameters for two convolution window
/// dimensions to two workgroup dimensions as a square. Returns true if we can
/// successfully deduce.
static bool distributeToSquare(const int64_t oh, const int64_t ow,
                               int64_t &residualThreads,
                               int64_t &residualTilingFactor,
                               MutableArrayRef<int64_t> wgDimSizes,
                               MutableArrayRef<int64_t> wgTileSizes) {
  assert(wgDimSizes.size() == 2 && wgTileSizes.size() == 2);

  const unsigned log2Threads = llvm::Log2_64(residualThreads);
  if (oh == ow && residualThreads != 1 && log2Threads % 2 == 0) {
    const int64_t yz = 1ll << (log2Threads / 2);

    int64_t chosenTileSize = 1ll << (llvm::Log2_64(residualTilingFactor) / 2);
    while (chosenTileSize >= 1 && ow % (yz * chosenTileSize) != 0) {
      chosenTileSize >>= 1;
    }

    if (chosenTileSize != 0) {
      wgDimSizes.front() = wgDimSizes.back() = yz;
      wgTileSizes.front() = wgTileSizes.back() = yz * chosenTileSize;
      return true;
    }
  }
  return false;
}

//====---------------------------------------------------------------------===//
// Convolution Pipeline Configuration
//====---------------------------------------------------------------------===//

static LogicalResult setConvolutionConfig(
    IREE::GPU::TargetAttr target, mlir::FunctionOpInterface entryPointFn,
    linalg::LinalgOp linalgOp, const int64_t bestTilingFactor) {
  if (!isa<linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNchwFchwOp>(linalgOp)) {
    return failure();
  }
  if (clGPUUseTileAndFuseConvolution) {
    if (succeeded(IREE::GPU::setTileAndFuseLoweringConfig(target, entryPointFn,
                                                          linalgOp))) {
      LDBG("Tile and fuse convolution config");
      return success();
    }
  }
  const bool isNCHW = isa<linalg::Conv2DNchwFchwOp>(*linalgOp);
  const bool isNHWC = isa<linalg::Conv2DNhwcHwcfOp>(*linalgOp);

  const int ohIndex = isNHWC ? 1 : 2;
  const int owIndex = isNHWC ? 2 : 3;
  const int ocIndex = isNHWC ? 3 : 1;

  Type inputType = linalgOp.getDpsInputOperand(0)->get().getType();
  ArrayRef<int64_t> inputShape = llvm::cast<ShapedType>(inputType).getShape();
  Type outputType = linalgOp.getDpsInitOperand(0)->get().getType();
  ArrayRef<int64_t> outputShape = llvm::cast<ShapedType>(outputType).getShape();
  if (ShapedType::isDynamic(inputShape[3]) ||
      ShapedType::isDynamicShape(outputShape.drop_front())) {
    return failure();
  }
  int64_t oh = outputShape[ohIndex], ow = outputShape[owIndex],
          oc = outputShape[ocIndex];

  // The core idea is to distribute the convolution dimensions to the workgroup
  // Z/Y/X dimensions, with each thread in a workgroup handling multiple vector
  // elements. We try to 1) utilize all threads in a subgroup, and 2) handle an
  // optimal tile size along each dimension.
  int64_t residualThreads = target.getPreferredSubgroupSize();
  int64_t residualTilingFactor = bestTilingFactor;

  SmallVector<int64_t, 3> workgroupSize(3, 1); // (X, Y, Z)
  SmallVector<int64_t> workgroupTileSizes(4, 1);

  if (isNCHW) {
    // OW -> x, OH -> y, OC -> z
    if (!distributeToOneDim(ow, /*isInnerMostDim=*/true, residualThreads,
                            residualTilingFactor, workgroupSize[0],
                            workgroupTileSizes[3]) ||
        !distributeToOneDim(oh, /*isInnerMostDim=*/false, residualThreads,
                            residualTilingFactor, workgroupSize[1],
                            workgroupTileSizes[2]) ||
        !distributeToOneDim(oc, /*isInnerMostDim=*/false, residualThreads,
                            residualTilingFactor, workgroupSize[2],
                            workgroupTileSizes[1])) {
      return failure();
    }
  } else {
    // OC -> x
    if (!distributeToOneDim(oc, /*isInnerMostDim=*/true, residualThreads,
                            residualTilingFactor, workgroupSize[0],
                            workgroupTileSizes[3]))
      return failure();

    // Deduce the configruation for the OW and OH dimension. Try to make them
    // even if possible given we typically have images with the same height
    // and width.
    const bool tileToSquare = distributeToSquare(
        oh, ow, residualThreads, residualTilingFactor,
        llvm::MutableArrayRef(workgroupSize).drop_front(),
        llvm::MutableArrayRef(workgroupTileSizes).drop_front().drop_back());

    // Otherwise treat OW and OH separately to allow them to have different
    // number of threads and tiling size.
    if (!tileToSquare) {
      if (!distributeToOneDim(ow, /*isInnerMostDim=*/false, residualThreads,
                              residualTilingFactor, workgroupSize[1],
                              workgroupTileSizes[2]) ||
          !distributeToOneDim(oh, /*isInnerMostDim=*/false, residualThreads,
                              residualTilingFactor, workgroupSize[2],
                              workgroupTileSizes[1])) {
        return failure();
      }
    }
  }
  auto pipeline = CodeGenPipeline::LLVMGPUVectorize;
  TileSizesListType tileSizes;
  // Add reduction tile sizes.
  if (isNCHW)
    workgroupTileSizes.append({4, 1, 1});
  else if (isNHWC)
    workgroupTileSizes.append({1, 1, 4});
  tileSizes.push_back(workgroupTileSizes);

  // Tile along OH by size 1 to enable downsizing 2-D convolution to 1-D.
  SmallVector<int64_t> windowTileSizes(4, 0);
  windowTileSizes[ohIndex] = 1;
  tileSizes.push_back(windowTileSizes);
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, linalgOp, tileSizes, pipeline, workgroupSize);
}

//====---------------------------------------------------------------------===//
// Pipeline Configuration
//====---------------------------------------------------------------------===//

static LogicalResult setRootConfig(IREE::GPU::TargetAttr target,
                                   mlir::FunctionOpInterface entryPointFn,
                                   Operation *computeOp) {
  IREE::GPU::UKernelConfigAttr ukernelConfig = selectUKernel(computeOp);
  LLVM_DEBUG({
    DBGS() << "Selecting root config for: ";
    computeOp->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
    llvm::dbgs() << "\n";
  });
  if (succeeded(setDataTiledMultiMmaLoweringConfig(target, entryPointFn,
                                                   computeOp, ukernelConfig))) {
    LDBG("Tile and fuse data tiled MMA inner_tiled config");
    return success();
  }
  if (clGPUEarlyTileAndFuseMatmul) {
    if (succeeded(IREE::GPU::setMatmulLoweringConfig(
            target, entryPointFn, computeOp, clUseDirectLoad))) {
      LDBG("Tile and fuse matmul config");
      return success();
    }
  }
  if (clLLVMGPUUseIgemm) {
    if (succeeded(IREE::GPU::setIGEMMConvolutionLoweringConfig(
            target, entryPointFn, computeOp, clUseDirectLoad))) {
      LDBG("Tile and fuse IGEMM config");
      return success();
    }
  }
  if (clGPUTestTileAndFuseVectorize) {
    if (succeeded(IREE::GPU::setTileAndFuseLoweringConfig(target, entryPointFn,
                                                          computeOp))) {
      LDBG("Tile and fuse default config");
      return success();
    }
  }
  if (succeeded(setVectorDistributionConfig(target, entryPointFn, computeOp))) {
    return success();
  }
  // TODO (nirvedhmeshram, qedawkins) : remove this when tile and fuse backend
  // config becomes the default for matmul.
  if (succeeded(IREE::GPU::setMatmulLoweringConfig(target, entryPointFn,
                                                   computeOp))) {
    LDBG("Tile and fuse matmul config after no vector distribute config");
    return success();
  }

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(computeOp)) {
    if (succeeded(setContractConfig(target, entryPointFn, linalgOp))) {
      LDBG("Contract Config");
      return success();
    }
    if (succeeded(setReductionVectorDistributionConfig(target, entryPointFn,
                                                       linalgOp))) {
      LDBG("Vector Distribution Subgroup Reduction Config");
      return success();
    }
    if (succeeded(setWarpReductionConfig(target, entryPointFn, linalgOp))) {
      LDBG("Warp Reduction Config");
      return success();
    }
    if (succeeded(setConvolutionConfig(target, entryPointFn, linalgOp, 16))) {
      LDBG("Convolution Config");
      return success();
    }
    auto genericOp = dyn_cast<linalg::GenericOp>(computeOp);
    if (genericOp) {
      if (succeeded(setTransposeConfig(entryPointFn, genericOp))) {
        LDBG("Transpose Config");
        return success();
      } else if (ukernelConfig &&
                 succeeded(setArgmaxUkernelConfig(target, entryPointFn,
                                                  genericOp, ukernelConfig))) {
        LDBG("Argmax Ukernel Config");
        return success();
      } else if (succeeded(IREE::GPU::setTileAndFuseLoweringConfig(
                     target, entryPointFn, linalgOp))) {
        LDBG("Tile and Fuse Config");
        return success();
      }
    }
  }
  return TypeSwitch<Operation *, LogicalResult>(computeOp)
      .Case<IREE::LinalgExt::FftOp>([&](auto fftOp) {
        LDBG("FFT Config");
        return setFftConfig(target, entryPointFn, fftOp);
      })
      .Case<IREE::LinalgExt::SortOp>([&](auto sortOp) {
        LDBG("Sort Config");
        return IREE::GPU::setSortConfig(target, entryPointFn, sortOp);
      })
      .Case<IREE::LinalgExt::WinogradInputTransformOp,
            IREE::LinalgExt::WinogradOutputTransformOp,
            IREE::LinalgExt::WinogradFilterTransformOp>([&](auto winogradOp) {
        LDBG("Winograd Config");
        return setWinogradOpConfig(target, entryPointFn, winogradOp);
      })
      .Case<IREE::LinalgExt::CustomOp>([&](auto customOp) {
        LDBG("CustomOp Config");
        return setDefaultCustomOpLoweringConfig(entryPointFn, customOp,
                                                initGPULaunchConfig);
      })
      .Case<IREE::LinalgExt::ScatterOp>([&](auto scatterOp) {
        LDBG("ScatterOp Config");
        if (failed(IREE::GPU::setScatterLoweringConfig(target, entryPointFn,
                                                       scatterOp))) {
          return setRootDefaultConfig(target, entryPointFn, computeOp);
        }
        return success();
      })
      .Default([&](auto op) {
        LDBG("Default Config");
        if (clLLVMGPUVectorizePipeline) {
          return setRootDefaultConfig(target, entryPointFn, computeOp);
        }
        if (succeeded(IREE::GPU::setTileAndFuseLoweringConfig(
                target, entryPointFn, computeOp))) {
          LDBG("Tile and fuse default config");
          return success();
        }
        return setRootDefaultConfig(target, entryPointFn, computeOp);
      });
}

// Propogate the configuration to the other ops.
// TODO(ravishankarm, thomasraoux): This is a very specific use (and
// fragile). In general, this should not be needed. Things are already tiled
// and distributed. The rest of the compilation must be structured to either
// use `TileAndFuse` or they are independent configurations that are
// determined based on the op.
static void propagateLoweringConfig(Operation *rootOperation,
                                    SmallVector<Operation *> computeOps) {
  if (IREE::Codegen::LoweringConfigAttrInterface config =
          getLoweringConfig(rootOperation)) {
    for (auto op : computeOps) {
      if (op == rootOperation)
        continue;
      setLoweringConfig(op, config);
    }
  }
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//
LogicalResult initGPULaunchConfig(FunctionOpInterface funcOp) {
  IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
  if (!target)
    return funcOp.emitError("missing GPU target in #hal.executable.target");

  auto exportOp = getEntryPoint(funcOp);
  if (!getTranslationInfo(funcOp) && exportOp) {
    // If no translation info set, first check whether we already have
    // workgroup count set--it's a "contract" to indicate that we should
    // bypass all tiling and distribution to go down just the most basic
    // lowering flow.
    if (Block *body = exportOp->getWorkgroupCountBody()) {
      auto retOp = cast<IREE::HAL::ReturnOp>(body->getTerminator());
      // For scalar dispatch cases--using just one thread of one workgroup.
      auto isOne = [](Value value) { return matchPattern(value, m_One()); };
      if (llvm::all_of(retOp.getOperands(), isOne)) {
        SmallVector<int64_t, 3> workgroupSize = {1, 1, 1};
        auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
            funcOp.getContext(), CodeGenPipeline::LLVMGPUBaseLowering,
            workgroupSize);
        if (failed(setTranslationInfo(funcOp, translationInfo))) {
          return failure();
        }
        return success();
      }
    }
  }

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  if (IREE::Codegen::TranslationInfoAttr translationInfo =
          getTranslationInfo(funcOp)) {
    // Currently some ROCDL requires propagation of user lowering configs.
    if (needsLoweringConfigPropagation(
            translationInfo.getDispatchLoweringPassPipeline())) {
      for (auto op : computeOps) {
        if (getLoweringConfig(op)) {
          propagateLoweringConfig(op, computeOps);
          break;
        }
      }
    }
    // Translation info (lowering pipeline) is already set.
    return success();
  }

  Operation *rootOperation = nullptr;

  // Find the root operation. linalg.generic, linalg.fill, linalg.pack,
  // linalg.unpack, and scatter are not root operations if there are other
  // compute operations present. Also, construct a set of generic ops that
  // are to be skipped. These generic ops that are used to compute scatter
  // indices are not root operations.
  llvm::SmallDenseSet<Operation *, 4> genericToSkip;
  for (Operation *op : llvm::reverse(computeOps)) {
    if (!isa<linalg::GenericOp, linalg::FillOp, IREE::LinalgExt::ScatterOp,
             IREE::LinalgExt::MapScatterOp, linalg::PackOp, linalg::UnPackOp>(
            op)) {
      rootOperation = op;
      break;
    }
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      // linalg.generic with `reduction` iterator types are roots as well.
      if (genericOp.getNumLoops() != genericOp.getNumParallelLoops()) {
        rootOperation = op;
        break;
      }
    }

    if (auto scatterOp = dyn_cast<IREE::LinalgExt::ScatterOp>(op)) {
      Value indices = scatterOp.getIndices();
      if (!indices.getDefiningOp()) {
        continue;
      }

      // Mark scatter's backward slices(inclusive) as to skip.
      BackwardSliceOptions options;
      options.inclusive = true;
      SetVector<Operation *> slices;
      [[maybe_unused]] LogicalResult result =
          getBackwardSlice(indices, &slices, options);
      assert(result.succeeded());
      genericToSkip.insert(slices.begin(), slices.end());
    }
  }

  // Generic ops take priority over pack, unpack, scatter, and fill ops as the
  // root op.
  if (!rootOperation) {
    for (Operation *op : llvm::reverse(computeOps)) {
      if (isa<linalg::GenericOp>(op) && !genericToSkip.contains(op)) {
        rootOperation = op;
        break;
      }
    }
  }

  // Pack and unpack ops take priority over scatter and fill ops as the root op.
  if (!rootOperation) {
    for (Operation *op : llvm::reverse(computeOps)) {
      if (isa<linalg::PackOp, linalg::UnPackOp>(op)) {
        rootOperation = op;
        break;
      }
    }
  }

  if (!rootOperation) {
    for (Operation *op : llvm::reverse(computeOps)) {
      if (isa<IREE::LinalgExt::ScatterOp, IREE::LinalgExt::MapScatterOp,
              linalg::FillOp>(op)) {
        rootOperation = op;
        break;
      }
    }
  }

  if (!rootOperation) {
    // No root operation found, set it to none.
    auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
        funcOp.getContext(), CodeGenPipeline::None);
    if (failed(setTranslationInfo(funcOp, translationInfo))) {
      return failure();
    }
    return success();
  }

  if (failed(setRootConfig(target, funcOp, rootOperation)))
    return funcOp.emitOpError("failed to set root config");

  if (IREE::Codegen::TranslationInfoAttr translationInfo =
          getTranslationInfo(funcOp)) {
    // Currently some ROCDL requires propagation of user lowering configs.
    if (!needsLoweringConfigPropagation(
            translationInfo.getDispatchLoweringPassPipeline())) {
      return success();
    }
  }

  propagateLoweringConfig(rootOperation, computeOps);
  return success();
}

} // namespace mlir::iree_compiler
