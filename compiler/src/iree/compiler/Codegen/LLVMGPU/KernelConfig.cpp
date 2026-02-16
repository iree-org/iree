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
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/MatchUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#define DEBUG_TYPE "iree-llvmgpu-kernel-config"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
namespace mlir::iree_compiler {

static llvm::cl::opt<bool> clGPUUseTileAndFuseMatmul(
    "iree-codegen-llvmgpu-use-tile-and-fuse-matmul",
    llvm::cl::desc("test the the tile and fuse pipeline for matmul"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clGPUTestTileAndFuseVectorize(
    "iree-codegen-llvmgpu-test-tile-and-fuse-vectorize",
    llvm::cl::desc(
        "test the tile and fuse pipeline for all supported operations"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clLLVMGPUVectorizePipeline(
    "iree-codegen-llvmgpu-vectorize-pipeline",
    llvm::cl::desc("forces use of the legacy LLVMGPU vectorize pipeline"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clGPUEnableVectorDistribution(
    "iree-codegen-llvmgpu-use-vector-distribution",
    llvm::cl::desc("enable the usage of the vector distribution pipeline"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clGPUEnableReductionVectorDistribution(
    "iree-codegen-llvmgpu-use-reduction-vector-distribution",
    llvm::cl::desc(
        "enable the usage of the reduction vector distribution pipeline"),
    llvm::cl::init(true));

// TODO (nirvedhmeshram): Drop this whole path after we have support with
// TileAndFuse pipeline from completion of
// https://github.com/iree-org/iree/issues/18858
static llvm::cl::opt<bool> clGPUUnalignedGEMMVectorDistribution(
    "iree-codegen-llvmgpu-use-unaligned-gemm-vector-distribution",
    llvm::cl::desc("enable the usage of the vector distribution pipeline for "
                   "unaligned GEMMs when supported"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clGPUUseTileAndFuseConvolution(
    "iree-codegen-llvmgpu-use-tile-and-fuse-convolution",
    llvm::cl::desc(
        "enable the tile and fuse pipeline for supported convolutions"),
    llvm::cl::init(true));

static llvm::cl::opt<int> clGPUMatmulCThreshold(
    "iree-codegen-llvmgpu-matmul-c-matrix-threshold",
    llvm::cl::desc("matmul c matrix element count threshold to be considered "
                   "as small vs. large when deciding MMA schedule"),
    // TODO: We should get this value from the target's parallelism.
    llvm::cl::init(512 * 512));

static llvm::cl::opt<bool>
    clLLVMGPUUseIgemm("iree-codegen-llvmgpu-use-igemm",
                      llvm::cl::desc("Enable implicit gemm for convolutions."),
                      llvm::cl::init(true));

static llvm::cl::opt<bool> clGPUPadConvolution(
    "iree-codegen-llvmgpu-igemm-pad-convolution",
    llvm::cl::desc("enable pre-padding for convolutions in igemm path"),
    llvm::cl::init(true));

static llvm::cl::opt<bool>
    clUseDirectLoad("iree-llvmgpu-use-direct-load",
                    llvm::cl::desc("Use global load DMA for direct load ops."),
                    llvm::cl::Hidden, llvm::cl::init(false));

static llvm::cl::opt<bool> clDirectConvolution(
    "iree-codegen-llvmgpu-use-direct-convolution",
    llvm::cl::desc("Use direct convolution in tile and fuse pipeline"),
    llvm::cl::init(false));

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

static bool isROCmBackend(IREE::GPU::TargetAttr target) {
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

//====---------------------------------------------------------------------===//
// Vector Distribution Contraction/Convolution Pipeline Configuration
//====---------------------------------------------------------------------===//

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

static LogicalResult setConvolutionVectorDistributionConfig(
    IREE::GPU::TargetAttr target, mlir::FunctionOpInterface entryPoint,
    linalg::LinalgOp op, const GPUCodegenOptions &gpuOpts) {
  if (target.getWgp().getMma().empty()) {
    return failure();
  }

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
  if (convolutionDims->outputChannel.empty() ||
      convolutionDims->inputChannel.empty() ||
      convolutionDims->filterLoop.empty() ||
      convolutionDims->outputImage.empty() || !convolutionDims->depth.empty()) {
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
  // all instances of schedule->m/nSubgroupCounts[0],
  // schedule->m/n/kTileSizes[0] and schedule->m/n/kSizes[0] need to use the
  // full list of sizes instead of just the first element.
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
    if (mma.getSubgroupSize() != targetSubgroupSize) {
      continue;
    }
    // Intrinsics without distribution mapping cannot be distributed.
    if (!mma.getDistributionMappingKind()) {
      continue;
    }
    storeMmaInfo(mma, intrinsics);
    // Skip adding any virtual intrinsics since they are not tested for
    // convolutions.
  }

  if (intrinsics.empty()) {
    return failure();
  }

  // TODO: Replace the below with algorithm described in
  // https://github.com/iree-org/iree/discussions/21506.
  // This is already implemented in KernelConfig.cpp in tileAndFuse pipeline
  // and should be ported to here once its perf results are verified.
  GPUMMAHeuristicSeeds seeds{/*bestSubgroupCountPerWorkgroup=*/4,
                             /*bestMNTileCountPerSubgroup=*/8,
                             /*bestKTileCountPerSubgroup=*/2};

  int64_t maxSharedMemoryBytes = target.getWgp().getMaxWorkgroupMemoryBytes();

  std::optional<int64_t> wgpCount = std::nullopt;
  if (IREE::GPU::TargetChipAttr chip = target.getChip()) {
    wgpCount = chip.getWgpCount();
  }
  // First try to find a schedule with an exactly matching intrinsic.
  FailureOr<GPUMMASchedule> schedule =
      deduceMMASchedule(problem, intrinsics, seeds, maxSharedMemoryBytes,
                        targetSubgroupSize, wgpCount, op.getLoc());
  if (failed(schedule)) {
    // Then try again by allowing upcasting accumulator.
    schedule =
        deduceMMASchedule(problem, intrinsics, seeds, maxSharedMemoryBytes,
                          targetSubgroupSize, wgpCount, op.getLoc(),
                          /*transposedLhs*/ false, /*transposedRhs*/ false,
                          /*canUpcastAcc=*/true);
  }
  if (failed(schedule)) {
    return failure();
  }

  LDBG() << "Schedule: " << schedule;

  assert(schedule->hasSingleDimensions() && "expected single M/N/K dimension");

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
  workgroupTileSizes[mDim] = schedule->mSubgroupCounts[0] *
                             schedule->mTileSizes[0] * schedule->mSizes[0];
  workgroupTileSizes[nDim] = schedule->nSubgroupCounts[0] *
                             schedule->nTileSizes[0] * schedule->nSizes[0];

  reductionTileSizes[kDim] = schedule->kTileSizes[0] * schedule->kSizes[0];

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
  IREE::GPU::Basis subgroupBasis = {
      SmallVector<int64_t>(op.getNumLoops(), 1),
      // Distribute subgroups from outer to inner. Mostly an arbitrary choice.
      // We can change this if it matters.
      llvm::to_vector(llvm::seq<int64_t>(op.getNumLoops()))};
  subgroupBasis.counts[mDim] = schedule->mSubgroupCounts[0];
  subgroupBasis.counts[nDim] = schedule->nSubgroupCounts[0];
  IREE::GPU::setBasis(context, attrs, IREE::GPU::TilingLevel::Subgroup,
                      subgroupBasis);

  auto configDict = DictionaryAttr::get(context, attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  SmallVector<NamedAttribute, 1> pipelineAttrs;

  // Prefetch shared memory if requested.
  if (gpuOpts.enablePrefetch) {
    auto pipelineOptions = IREE::GPU::GPUPipelineOptionsAttr::get(
        context, /*prefetch_num_stages=*/2,
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
                          const linalg::ContractionDimensions &contractionDims,
                          ArrayRef<int64_t> sizes) {
  ArrayRef<unsigned> dimVals[] = {contractionDims.batch, contractionDims.m,
                                  contractionDims.n, contractionDims.k};
  std::string dimSymbols(numLoops, '*');
  for (auto [idx, val] : llvm::enumerate(dimSymbols)) {
    for (auto [letter, dim] : llvm::zip_equal(StringRef("bmnk"), dimVals)) {
      if (llvm::is_contained(dim, idx)) {
        val = letter;
      }
    }
  }
  DBGS() << "Contraction dims: " << llvm::interleaved_array(dimSymbols) << "\n";
  DBGS() << label << ": " << llvm::interleaved_array(sizes) << "\n";
}

static LogicalResult setMatmulVectorDistributionConfig(
    IREE::GPU::TargetAttr target, mlir::FunctionOpInterface entryPoint,
    linalg::LinalgOp op, const GPUCodegenOptions &gpuOpts) {
  if (target.getWgp().getMma().empty()) {
    return failure();
  }

  const int64_t targetSubgroupSize = target.getPreferredSubgroupSize();

  SmallVector<int64_t> bounds = op.getStaticLoopRanges();
  FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
      mlir::linalg::inferContractionDims(op);

  // Used for calculating correct shared memory usage when the op is a
  // horizontally fused contraction.
  int64_t numHorizontallyFusedOps = 1;
  if (failed(contractionDims)) {
    assert(IREE::LinalgExt::isaHorizontallyFusedContraction(op) &&
           "expected horizontally fused contraction op");
    SmallVector<AffineMap> indexingMaps;
    indexingMaps.push_back(op.getMatchingIndexingMap(op.getDpsInputOperand(0)));
    indexingMaps.push_back(op.getMatchingIndexingMap(op.getDpsInputOperand(1)));
    indexingMaps.push_back(op.getMatchingIndexingMap(op.getDpsInitOperand(0)));
    contractionDims = mlir::linalg::inferContractionDims(indexingMaps);
    numHorizontallyFusedOps = op.getNumDpsInputs() - 1;
  }
  assert(succeeded(contractionDims) && "Could not infer contraction dims");

  if (contractionDims->k.empty() || contractionDims->m.empty() ||
      contractionDims->n.empty()) {
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

  SmallVector<int64_t> batchDims;
  for (int64_t batchDim : contractionDims->batch) {
    if (ShapedType::isStatic(bounds[batchDim])) {
      batchDims.push_back(batchDim);
    }
  }
  auto getDimBounds = [&](SmallVector<int64_t> dims) -> SmallVector<int64_t> {
    return llvm::map_to_vector(dims, [&](int64_t dim) { return bounds[dim]; });
  };

  // TODO(Max191): Support multiple M/N/K dimension problems for MMASchedules
  // once the pipeline is able to support it. After adding multiple dimensions,
  // all instances of schedule->m/nSubgroupCounts[0],
  // schedule->m/n/kTileSizes[0] and schedule->m/n/kSizes[0] need to use the
  // full list of sizes instead of just the first element.
  GPUMatmulShapeType problem{{bounds[mDim]},     {bounds[nDim]},
                             {bounds[kDim]},     getDimBounds(batchDims),
                             lhsElemType,        rhsElemType,
                             initElemType,
                             /*aScale=*/nullptr,
                             /*bScale=*/nullptr, numHorizontallyFusedOps};

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
    if (mma.getSubgroupSize() != targetSubgroupSize) {
      continue;
    }
    // Intrinsics without distribution mapping cannot be distributed.
    if (!mma.getDistributionMappingKind()) {
      continue;
    }
    storeMmaInfo(mma, intrinsics);
    // Skip adding any virtual intrinsics since they are not tested for matmuls.
  }

  if (intrinsics.empty()) {
    return failure();
  }

  GPUMMAHeuristicSeeds seeds;

  // TODO: Replace the below with algorithm described in
  // https://github.com/iree-org/iree/discussions/21506.
  // This is already implemented in KernelConfig.cpp in tileAndFuse pipeline
  // and should be ported to here once its perf results are verified.
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

  LDBG() << "Matmul Vector Distribution Config";

  auto pipeline = CodeGenPipeline::LLVMGPUVectorDistribute;

  // Infer if lhs or rhs is transposed to help generate better schedule.
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  bool transposedLhs =
      kDim != cast<AffineDimExpr>(maps[0].getResults().back()).getPosition();
  bool transposedRhs =
      nDim != cast<AffineDimExpr>(maps[1].getResults().back()).getPosition();

  std::optional<int64_t> wgpCount = std::nullopt;
  if (IREE::GPU::TargetChipAttr chip = target.getChip()) {
    wgpCount = chip.getWgpCount();
  }

  // First try to find a schedule with an exactly matching intrinsic.
  std::optional<GPUMMASchedule> schedule =
      deduceMMASchedule(problem, intrinsics, seeds, maxSharedMemoryBytes,
                        targetSubgroupSize, wgpCount, op.getLoc());
  if (!schedule) {
    // Then try again by allowing upcasting accumulator.
    schedule =
        deduceMMASchedule(problem, intrinsics, seeds, maxSharedMemoryBytes,
                          targetSubgroupSize, wgpCount, op.getLoc(),
                          transposedLhs, transposedRhs, /*canUpcastAcc=*/true);
  }

  if (!schedule) {
    LDBG() << "Failed to deduce MMA schedule";
    return failure();
  }

  LDBG() << "Target Subgroup size: " << targetSubgroupSize;
  LDBG() << "Schedule: " << schedule;

  assert(schedule->hasSingleDimensions() && "expected single M/N/K dimension");

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
  workgroupTileSizes[mDim] = schedule->mSubgroupCounts[0] *
                             schedule->mTileSizes[0] * schedule->mSizes[0];
  workgroupTileSizes[nDim] = schedule->nSubgroupCounts[0] *
                             schedule->nTileSizes[0] * schedule->nSizes[0];

  reductionTileSizes[kDim] = schedule->kTileSizes[0] * schedule->kSizes[0];

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
  IREE::GPU::Basis subgroupBasis = {
      SmallVector<int64_t>(op.getNumLoops(), 1),
      // Distribute subgroups from outer to inner. Mostly an arbitrary choice.
      // We can change this if it matters.
      llvm::to_vector(llvm::seq<int64_t>(op.getNumLoops()))};
  subgroupBasis.counts[mDim] = schedule->mSubgroupCounts[0];
  subgroupBasis.counts[nDim] = schedule->nSubgroupCounts[0];
  IREE::GPU::setBasis(context, attrs, IREE::GPU::TilingLevel::Subgroup,
                      subgroupBasis);

  auto configDict = DictionaryAttr::get(context, attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  // Attach the MMA schedule as an attribute to the entry point export function
  // for later access in the pipeline.
  SmallVector<NamedAttribute, 1> pipelineAttrs;

  // Prefetch shared memory if requested.
  if (gpuOpts.enablePrefetch) {
    auto pipelineOptions = IREE::GPU::GPUPipelineOptionsAttr::get(
        context, /*prefetch_num_stages=*/2,
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

/// Sets attention specific pipeline attributes.
static void
setAttentionPipelineAttributes(IREE::GPU::TargetAttr target,
                               SmallVectorImpl<NamedAttribute> &pipelineAttrs) {
  pipelineAttrs.emplace_back(
      IREE::Codegen::DenormalFpMathAttr::getFP32DictKeyName(),
      IREE::Codegen::DenormalFpMathAttr::get(
          target.getContext(), IREE::Codegen::DenormalFpMath::PreserveSign));
}

static LogicalResult setAttentionIntrinsicBasedVectorDistributionConfig(
    IREE::GPU::TargetAttr target, mlir::FunctionOpInterface entryPoint,
    IREE::LinalgExt::AttentionOp op) {
  if (target.getWgp().getMma().empty()) {
    return failure();
  }

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

  auto getDimBounds = [&](ArrayRef<int64_t> dims) -> SmallVector<int64_t> {
    return llvm::map_to_vector(dims, [&](int64_t dim) { return bounds[dim]; });
  };

  SmallVector<int64_t> mBounds = getDimBounds(opInfo.getMDims());
  int64_t mSize = llvm::accumulate(
      mBounds, int64_t(1), [](int64_t a, int64_t b) -> int64_t {
        if (ShapedType::isDynamic(a) || ShapedType::isDynamic(b)) {
          return ShapedType::kDynamic;
        }
        return a * b;
      });
  // Bail out on skinny M dimension. This will be handled by the warp reduction
  // pipeline.
  if (!ShapedType::isDynamic(mSize) && mSize <= kVerySkinnyDimThreshold) {
    LDBG() << "Bailing out due to skinny M dimension: " << mSize;
    return failure();
  }

  // TODO: Add dynamic shape support for inner most dims, this is mostly
  // possible in VectorDistribute through masking, just not tested enough to
  // enable it.
  if (ShapedType::isDynamic(bounds[opInfo.getK1Dims().back()]) ||
      ShapedType::isDynamic(bounds[opInfo.getK2Dims().back()]) ||
      ShapedType::isDynamic(bounds[opInfo.getNDims().back()]) ||
      ShapedType::isDynamic(bounds[opInfo.getMDims().back()])) {
    return failure();
  }

  // Gather all static M, N, and K dimensions to deduce the MMASchedule. Dynamic
  // dimensions will be tiled to 1 in workgroup tiling, so they are ignored when
  // computing an MMA schedule.
  auto getStaticDims = [&](ArrayRef<int64_t> dims) {
    SmallVector<int64_t> staticDims;
    for (int64_t dim : dims) {
      if (!ShapedType::isDynamic(bounds[dim])) {
        staticDims.push_back(dim);
      }
    }
    return staticDims;
  };
  SmallVector<int64_t> batchDims, mDims, k2Dims, k1Dims, nDims;
  batchDims = getStaticDims(opInfo.getBatchDims());
  mDims = getStaticDims(opInfo.getMDims());
  k2Dims = getStaticDims(opInfo.getK2Dims());
  k1Dims = getStaticDims(opInfo.getK1Dims());
  nDims = getStaticDims(opInfo.getNDims());

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
    if (mma.getSubgroupSize() != targetSubgroupSize) {
      continue;
    }
    // Intrinsics without distribution mapping cannot be distributed.
    if (!mma.getDistributionMappingKind()) {
      continue;
    }
    storeMmaInfo(mma, intrinsics);
    // Store info on virtual intrinsics based on current mma if any
    for (IREE::GPU::VirtualMMAIntrinsic virtualIntrinsic :
         mma.getVirtualIntrinsics()) {
      auto virtualMma =
          IREE::GPU::VirtualMMAAttr::get(context, virtualIntrinsic);
      storeMmaInfo(virtualMma, intrinsics);
    }
  }

  if (intrinsics.empty()) {
    return failure();
  }

  // We assume that P uses the element type of V for input
  // and both matmuls have f32 as output. It is possible to use other element
  // types also.
  Type qElementType = getElementTypeOrSelf(qMatrix);
  Type kElementType = getElementTypeOrSelf(kMatrix);
  Type vElementType = getElementTypeOrSelf(vMatrix);
  Type f32Type = b.getF32Type();
  GPUMatmulShapeType qkMatmul{
      /*m=*/getDimBounds(mDims),
      /*n=*/getDimBounds(nDims),
      /*k=*/getDimBounds(k1Dims),
      /*batch=*/getDimBounds(batchDims),
      /*a=*/qElementType,
      /*b=*/kElementType,
      /*c=*/f32Type,
  };
  GPUMatmulShapeType pvMatmul{/*m=*/getDimBounds(mDims),
                              /*n=*/getDimBounds(nDims),
                              /*k=*/getDimBounds(k2Dims),
                              /*batch=*/getDimBounds(batchDims),
                              /*a=*/vElementType,
                              /*b=*/vElementType,
                              /*c=*/f32Type};

  GPUMMAHeuristicSeeds pvMatmulSeeds = {/*bestSubgroupCountPerWorkgroup=*/4,
                                        /*bestMNTileCountPerSubgroup=*/4,
                                        /*bestKTileCountPerSubgroup=*/4};

  LDBG() << "Attention Vector Distribution Config";

  // Infer if Q, K and V are transposed to help generate better schedule.
  bool transposedQ =
      k1Dims.back() !=
      cast<AffineDimExpr>(op.getQueryMap().getResults().back()).getPosition();
  bool transposedK =
      k1Dims.back() !=
      cast<AffineDimExpr>(op.getKeyMap().getResults().back()).getPosition();
  bool transposedV =
      k2Dims.back() !=
      cast<AffineDimExpr>(op.getValueMap().getResults().back()).getPosition();

  int64_t maxSharedMemoryBytes = target.getWgp().getMaxWorkgroupMemoryBytes();
  // First try to find a schedule with an exactly matching intrinsic.
  std::optional<std::pair<GPUMMASchedule, GPUMMASchedule>> attSchedule =
      deduceAttentionSchedule(qkMatmul, pvMatmul, intrinsics, pvMatmulSeeds,
                              maxSharedMemoryBytes, targetSubgroupSize,
                              transposedQ, transposedK, transposedV);
  if (!attSchedule) {
    // Then try again by allowing upcasting accumulator.
    attSchedule = deduceAttentionSchedule(
        qkMatmul, pvMatmul, intrinsics, pvMatmulSeeds, maxSharedMemoryBytes,
        targetSubgroupSize, transposedQ, transposedK, transposedV,
        /*canUpcastAcc=*/true);
  }

  if (!attSchedule) {
    LDBG() << "Failed to deduce Attention schedule";
    return failure();
  }

  auto [qkSchedule, pvSchedule] = attSchedule.value();

  LDBG() << "Target Subgroup size: " << targetSubgroupSize;
  LDBG() << "QK Schedule: " << qkSchedule;
  LDBG() << "PV Schedule: " << pvSchedule;

  int64_t flatWorkgroupSize =
      targetSubgroupSize *
      ShapedType::getNumElements(pvSchedule.nSubgroupCounts) *
      ShapedType::getNumElements(pvSchedule.mSubgroupCounts);
  std::array<int64_t, 3> workgroupSize{flatWorkgroupSize, 1, 1};

  SmallVector<int64_t> workgroupTileSizes(opInfo.getDomainRank(), 0);
  SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 0);
  IREE::GPU::Basis subgroupBasis = {
      SmallVector<int64_t>(opInfo.getDomainRank(), 1),
      // Distribute subgroups from outer to inner. Mostly an arbitrary choice.
      // We can change this if it matters.
      llvm::to_vector(llvm::seq<int64_t>(opInfo.getDomainRank()))};

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
  for (auto [i, mDim] : llvm::enumerate(mDims)) {
    workgroupTileSizes[mDim] =
        pvSchedule.mSubgroupCounts[i] * pvSchedule.mTileSizes[i];
    // Multiply by the intrinsic shape for the inner most dim.
    if (i == mDims.size() - 1) {
      workgroupTileSizes[mDim] *= llvm::product_of(pvSchedule.mSizes);
    }
    subgroupBasis.counts[mDim] = pvSchedule.mSubgroupCounts[i];
  }
  for (auto [i, nDim] : llvm::enumerate(nDims)) {
    workgroupTileSizes[nDim] =
        pvSchedule.nSubgroupCounts[i] * pvSchedule.nTileSizes[i];
    // Multiply by the intrinsic shape for the inner most dim.
    if (i == nDims.size() - 1) {
      workgroupTileSizes[nDim] *= llvm::product_of(pvSchedule.nSizes);
    }
    subgroupBasis.counts[nDim] = pvSchedule.nSubgroupCounts[i];
  }
  for (auto [i, k2Dim] : llvm::enumerate(k2Dims)) {
    reductionTileSizes[k2Dim] = pvSchedule.kTileSizes[i];
    // Multiply by the intrinsic shape for the inner most dim.
    if (i == k2Dims.size() - 1) {
      reductionTileSizes[k2Dim] *= llvm::product_of(pvSchedule.kSizes);
    }
  }

  SmallVector<NamedAttribute, 2> attrs = {
      NamedAttribute("workgroup", b.getI64ArrayAttr(workgroupTileSizes)),
      NamedAttribute("reduction", b.getI64ArrayAttr(reductionTileSizes))};
  IREE::GPU::appendPromotedOperandsList(context, attrs, {0, 1, 2});

  SmallVector<NamedAttribute, 2> qkConfig;
  SmallVector<NamedAttribute, 2> pvConfig;

  // Configuring for qk matmul.
  IREE::GPU::appendPromotedOperandsList(context, qkConfig, {0, 1});
  IREE::GPU::setMmaKind(context, qkConfig, qkSchedule.mmaKind);
  IREE::GPU::setBasis(context, qkConfig, IREE::GPU::TilingLevel::Subgroup,
                      projectBasis(subgroupBasis, opInfo.getNDims()));

  // Configuring for pv matmul.
  IREE::GPU::appendPromotedOperandsList(context, pvConfig, {1});
  IREE::GPU::setMmaKind(context, pvConfig, pvSchedule.mmaKind);
  IREE::GPU::setBasis(context, pvConfig, IREE::GPU::TilingLevel::Subgroup,
                      projectBasis(subgroupBasis, opInfo.getK1Dims()));

  SmallVector<NamedAttribute, 2> qkAttrs = {
      {"attention_qk_matmul", b.getUnitAttr()}};
  SmallVector<NamedAttribute, 2> pvAttrs = {
      {"attention_pv_matmul", b.getUnitAttr()}};

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

  DictionaryAttr decompositionConfigDict =
      b.getDictionaryAttr(decompositionConfig);

  auto configDict = b.getDictionaryAttr(attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  SmallVector<NamedAttribute, 1> pipelineAttrs;

  setAttentionPipelineAttributes(target, pipelineAttrs);

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

struct AttentionReductionHeuristicSeeds {
  int64_t numKeyVectors;
  int64_t numValueVectors;
  int64_t numSubgroups;
  int64_t keyVectorSize;
  int64_t valueVectorSize;
};

static LogicalResult setAttentionReductionConfig(
    AttentionReductionHeuristicSeeds &seeds, IREE::GPU::TargetAttr target,
    FunctionOpInterface entryPoint, IREE::LinalgExt::AttentionOp op) {

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
  // Dynamic dimensions are treated as inf (distribute everything).
  auto distributeDimensionsToBasisGreedily =
      [&bounds](int64_t available, ArrayRef<int64_t> dims,
                IREE::GPU::Basis &basis, int64_t &currDim) {
        // Iterate over dimensions and try to distribute resources over them.
        for (int64_t dim : llvm::reverse(dims)) {
          // We iterate over the basis in a reverse dimension to get smaller
          // strides for inner dimensions.
          int64_t rCurrDim = basis.counts.size() - currDim - 1;
          ++currDim;
          // Keep track of the order the dimensions are distributed in.
          basis.mapping[dim] = rCurrDim;
          // Try to distribute the resources over the dimensions greedily.
          int64_t dimSize = bounds[dim];
          if (ShapedType::isDynamic(dimSize)) {
            // Distribute remaining resources on the dynamic dim.
            basis.counts[rCurrDim] = available;
            available = 1;
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
  threadTileSizes[opInfo.getK1Dims().back()] = seeds.keyVectorSize;
  bounds[opInfo.getK1Dims().back()] =
      std::ceil(float(bounds[opInfo.getK1Dims().back()]) / seeds.keyVectorSize);
  threadTileSizes[opInfo.getNDims().back()] = seeds.valueVectorSize;
  bounds[opInfo.getNDims().back()] = std::ceil(
      float(bounds[opInfo.getNDims().back()]) / seeds.valueVectorSize);

  // Select the thread split between K2 and K1/N dimensions. We select this
  // based on the number of key vectors, and use the same split for value
  // vectors.
  auto getNumVectors = [&bounds](ArrayRef<int64_t> dims) {
    int64_t numVectors = 1;
    for (int64_t dim : dims) {
      numVectors *= bounds[dim];
    }
    return numVectors;
  };
  int64_t numK1Vectors = getNumVectors(opInfo.getK1Dims());
  int64_t numK1Tiles = std::ceil(float(numK1Vectors) / seeds.numKeyVectors);

  int64_t k1ThreadSplit =
      std::min(int64_t(llvm::PowerOf2Ceil(numK1Tiles)), targetSubgroupSize);
  int64_t k2ThreadSplit = targetSubgroupSize / k1ThreadSplit;

  IREE::GPU::Basis qkThreadBasis = {
      SmallVector<int64_t>(opInfo.getDomainRank(), 1),
      SmallVector<int64_t>(opInfo.getDomainRank())};
  IREE::GPU::Basis pvThreadBasis = {
      SmallVector<int64_t>(opInfo.getDomainRank(), 1),
      SmallVector<int64_t>(opInfo.getDomainRank())};

  {
    int64_t k2RemainingThreads = k2ThreadSplit;
    int64_t k1RemainingThreads = k1ThreadSplit;
    int64_t nRemainingThreads = k1ThreadSplit;

    // Distribute both basis on K2 equally.
    int64_t qkCurrDim = 0;
    k2RemainingThreads = distributeDimensionsToBasisGreedily(
        k2RemainingThreads, opInfo.getK2Dims(), qkThreadBasis, qkCurrDim);

    pvThreadBasis = qkThreadBasis;
    int64_t pvCurrDim = qkCurrDim;

    // If the target doesn't support subgroup shuffle, we should still be
    // distributing on threads. It's the backends problem to not use shuffles,
    // and instead use shared memory for reduction.

    // Distribute K1 on QK basis and N on nothing.
    k1RemainingThreads = distributeDimensionsToBasisGreedily(
        k1RemainingThreads, opInfo.getK1Dims(), qkThreadBasis, qkCurrDim);
    distributeDimensionsToBasisGreedily(1, opInfo.getNDims(), qkThreadBasis,
                                        qkCurrDim);
    // Distribute N on PV basis and K1 on nothing.
    nRemainingThreads = distributeDimensionsToBasisGreedily(
        nRemainingThreads, opInfo.getNDims(), pvThreadBasis, pvCurrDim);
    distributeDimensionsToBasisGreedily(1, opInfo.getK1Dims(), pvThreadBasis,
                                        pvCurrDim);

    // We already tiled B/M on workgroups, so it doesn't really matter how we
    // distribute them here.
    distributeDimensionsToBasisGreedily(1, opInfo.getBatchDims(), qkThreadBasis,
                                        qkCurrDim);
    distributeDimensionsToBasisGreedily(1, opInfo.getMDims(), qkThreadBasis,
                                        qkCurrDim);

    distributeDimensionsToBasisGreedily(1, opInfo.getBatchDims(), pvThreadBasis,
                                        pvCurrDim);
    distributeDimensionsToBasisGreedily(1, opInfo.getMDims(), pvThreadBasis,
                                        pvCurrDim);
  }

  // Distribute subgroups on K2 dimension only.
  IREE::GPU::Basis subgroupBasis = {
      SmallVector<int64_t>(opInfo.getDomainRank(), 1),
      SmallVector<int64_t>(opInfo.getDomainRank())};

  {
    int64_t numRemainingSubgroups = seeds.numSubgroups;
    // Distribute both basis on K2 equally.
    int64_t currDim = 0;
    numRemainingSubgroups = distributeDimensionsToBasisGreedily(
        numRemainingSubgroups, opInfo.getK2Dims(), subgroupBasis, currDim);

    // Distribute N, K1, M, B on nothing.
    distributeDimensionsToBasisGreedily(1, opInfo.getNDims(), subgroupBasis,
                                        currDim);
    distributeDimensionsToBasisGreedily(1, opInfo.getK1Dims(), subgroupBasis,
                                        currDim);
    distributeDimensionsToBasisGreedily(1, opInfo.getMDims(), subgroupBasis,
                                        currDim);
    distributeDimensionsToBasisGreedily(1, opInfo.getBatchDims(), subgroupBasis,
                                        currDim);
  }

  LDBG() << "QK Basis";
  LDBG() << "Thread Basis";
  LDBG() << llvm::interleaved(qkThreadBasis.counts);
  LDBG() << llvm::interleaved(qkThreadBasis.mapping);
  LDBG() << "Subgroup Basis";
  LDBG() << llvm::interleaved(subgroupBasis.counts);
  LDBG() << llvm::interleaved(subgroupBasis.mapping);
  LDBG() << "PV Basis";
  LDBG() << "Thread Basis";
  LDBG() << llvm::interleaved(pvThreadBasis.counts);
  LDBG() << llvm::interleaved(pvThreadBasis.mapping);
  LDBG() << "Subgroup Basis";
  LDBG() << llvm::interleaved(subgroupBasis.counts);
  LDBG() << llvm::interleaved(subgroupBasis.mapping);

  // Tile N parallel dimensions to value tile size fetched in a single
  // iteration.
  for (int64_t dim : opInfo.getNDims()) {
    int64_t threadCount = pvThreadBasis.counts[pvThreadBasis.mapping[dim]];
    int64_t dimSize = threadCount;
    if (dim == opInfo.getNDims().back()) {
      dimSize *= seeds.numValueVectors * seeds.valueVectorSize;
    }
    workgroupTileSizes[dim] = dimSize;
  }

  // Tile remaining reduction dimensions to serial loops.
  SmallVector<int64_t> reductionTileSizes(opInfo.getDomainRank(), 0);
  for (int64_t dim : opInfo.getK2Dims()) {
    int64_t threadCount = qkThreadBasis.counts[qkThreadBasis.mapping[dim]];
    int64_t subgroupCount = subgroupBasis.counts[subgroupBasis.mapping[dim]];
    reductionTileSizes[dim] = threadCount * subgroupCount;
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

  SmallVector<NamedAttribute, 1> pipelineAttrs;
  setAttentionPipelineAttributes(target, pipelineAttrs);

  // Set attention decomposition control config.
  op.setDecompositionConfigAttr(b.getDictionaryAttr(decompositionConfig));

  auto configDict = b.getDictionaryAttr(attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);
  auto pipelineConfig = DictionaryAttr::get(context, pipelineAttrs);

  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, loweringConfig, CodeGenPipeline::LLVMGPUVectorDistribute,
      workgroupSize, targetSubgroupSize, pipelineConfig);

  return success();
}

static LogicalResult
setAttentionVectorDistributionConfig(IREE::GPU::TargetAttr target,
                                     FunctionOpInterface entryPoint,
                                     IREE::LinalgExt::AttentionOp op) {

  // This configuration is not really smart right now. It just makes sure that
  // attention always compiles and tries to distribute workload on threads,
  // subgroups and workgroups as much as it can.
  // TODO: Update this configuration with target information, like the
  // WarpReduction pipeline does.

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

  AttentionReductionHeuristicSeeds seeds{/*numKeyVectors=*/8,
                                         /*numValueVectors=*/2,
                                         /*numSubgroups=*/8,
                                         /*keyVectorSize=*/keyVectorSize,
                                         /*valueVectorSize=*/valueVectorSize};

  return setAttentionReductionConfig(seeds, target, entryPoint, op);
}

static LogicalResult setVectorDistributionConfig(
    IREE::GPU::TargetAttr target, mlir::FunctionOpInterface entryPoint,
    Operation *computeOp, const GPUCodegenOptions &gpuOpts) {
  if (!clGPUEnableVectorDistribution) {
    LDBG() << "Vector Distribution not enabled, skipping...";
    return failure();
  }

  LDBG() << "VectorDistribution: finding a suitable config...";

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(computeOp)) {
    if (linalg::isaContractionOpInterface(linalgOp) ||
        IREE::LinalgExt::isaHorizontallyFusedContraction(linalgOp)) {
      LDBG()
          << "VectorDistribution: trying to find a suitable contraction config";
      return setMatmulVectorDistributionConfig(target, entryPoint, linalgOp,
                                               gpuOpts);
    }
    if (linalg::isaConvolutionOpInterface(linalgOp)) {
      LDBG()
          << "VectorDistribution: trying to find a suitable convolution config";
      return setConvolutionVectorDistributionConfig(target, entryPoint,
                                                    linalgOp, gpuOpts);
    }
  }

  if (auto attnOp = dyn_cast<IREE::LinalgExt::AttentionOp>(computeOp)) {
    LDBG() << "VectorDistribution: trying to find a suitable attention config";
    if (succeeded(setAttentionIntrinsicBasedVectorDistributionConfig(
            target, entryPoint, attnOp))) {
      return success();
    }
    return setAttentionVectorDistributionConfig(target, entryPoint, attnOp);
  }

  LDBG() << "VectorDistribution: failed to find a suitable config";
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
        bounds[mDim] != 1 && ShapedType::isStatic(bounds[mDim]);
  }
  for (auto nDim : contractionDims->n) {
    staticNonUnitParallelDimCount +=
        bounds[nDim] != 1 && ShapedType::isStatic(bounds[nDim]);
  }
  if (staticNonUnitParallelDimCount <= 1) {
    return failure();
  }

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
      if (!partitionedLoopsSet.contains(loopID)) {
        workgroupTileSizes[loopID] = 0;
      }
    }

    std::optional<int64_t> subgroupSize = std::nullopt;
    if (!subgroupSizes.empty()) {
      subgroupSize = subgroupSizes.front();
    }

    // For the LLVMGPUTileAndFuse pipeline, we need to split tile sizes
    // for workgroup, thread, and reduction.
    if (pipeline == CodeGenPipeline::LLVMGPUTileAndFuse) {

      MLIRContext *context = op.getContext();
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
          context, /*prefetch_num_stages=*/0,
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
      cast<ShapedType>(op.getDpsInputOperand(0)->get().getType()).getShape();
  auto rhsShape =
      cast<ShapedType>(op.getDpsInputOperand(1)->get().getType()).getShape();
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
  for (unsigned i = 0, e = rhsShape.size(); i < e; ++i) {
    if (op.getMatchingIndexingMap(op.getDpsInputOperand(1)).getDimPosition(i) ==
        outputMap.getDimPosition(outputMap.getNumResults() - 1)) {
      sizeN = rhsShape[i];
      break;
    }
  }
  SmallVector<unsigned> exprs;
  op.getReductionDims(exprs);
  if (exprs.size() == 1) {
    for (unsigned i = 0, e = lhsShape.size(); i < e; ++i) {
      if (op.getMatchingIndexingMap(op.getDpsInputOperand(0))
              .getDimPosition(i) == exprs[0]) {
        sizeK = lhsShape[i];
        break;
      }
    }
  }
  bool isStaticSize = ShapedType::isStatic(sizeM) &&
                      ShapedType::isStatic(sizeN) &&
                      ShapedType::isStatic(sizeK);
  if (isStaticSize) {
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
  if (ShapedType::isDynamic(sizeK)) {
    tileK = 1;
  }
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
    if (!partitionedLoopsSet.contains(depth)) {
      workgroupTileSizes[depth] = 0;
    }
  }

  // Tile to have one element per thread.
  for (int64_t depth = numLoops; depth > 0; depth--) {
    if (partitionedLoopsSet.contains(depth - 1)) {
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
    if (!partitionedLoopsSet.contains(depth)) {
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
          cast<ShapedType>(outputOperand.get().getType()).getShape();
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
      if (vectorSize == 1) { // assume there is fastpath + slowpath
        vectorSize = 4;
      }
      int64_t problemSize = llvm::product_of(shape);
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
          if (dim == 1) {
            continue;
          }
          if (dim < flatWG) {
            skipInnerTiling++;
            workgroupSize[id] = dim;
          } else {
            workgroupSize[id] = flatWG;
            break;
          }
          flatWG = flatWG / dim;
          id++;
          if (flatWG <= 1 || id >= workgroupSize.size()) {
            break;
          }
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
  for (int64_t depth = numLoops; depth > 0; --depth) {
    if (partitionedLoopsSet.contains(depth - 1)) {
      if (skipInnerTiling > 0) {
        // For dimensions that don't need to be distributed across blocks skip
        // tiling by setting tile size to 0.
        workgroupTileSizes[depth - 1] = 0;
        --skipInnerTiling;
        ++id;
        if (id >= workgroupSize.size()) {
          break;
        }
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
  if (failed(dims)) {
    return false;
  }

  // TODO: Support batch matvec.
  if (!dims->batch.empty()) {
    return false;
  }

  if (dims->m.size() >= 2 || dims->n.size() >= 2 ||
      !llvm::hasSingleElement(dims->k)) {
    return false;
  }

  return true;
}

static bool hasTwoOrThreeLoopsInfo(linalg::LinalgOp linalgOp) {
  return linalgOp.getNumParallelLoops() >= 2 &&
         linalgOp.getNumParallelLoops() <= 3;
}

//====---------------------------------------------------------------------===//
// Transpose Pipeline Configuration
//====---------------------------------------------------------------------===//

static LogicalResult setTransposeConfig(IREE::GPU::TargetAttr target,
                                        mlir::FunctionOpInterface entryPoint,
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
  // Set all tile sizes to 1 except for fastest moving dimensions.
  SmallVector<int64_t> workgroupTileSizes(linalgOp.getNumLoops(), 1);
  workgroupTileSizes[outputFastestDim] = 32;
  workgroupTileSizes[inputFastestDim] = 32;

  // Set the thread tile sizes to 1 for all dims except the fastest varying
  // output dim which we set to 4. Because we promote the transposed input
  // operands, this gives both vectorized global reads and writes.
  SmallVector<int64_t> threadTileSizes(linalgOp.getNumLoops(), 1);
  threadTileSizes[outputFastestDim] = 4;

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

  MLIRContext *context = linalgOp.getContext();
  Builder b(context);
  SmallVector<NamedAttribute> attrs = {
      {"workgroup", b.getI64ArrayAttr(workgroupTileSizes)},
      {"thread", b.getI64ArrayAttr(threadTileSizes)}};
  SmallVector<int64_t> promotedOperands;
  for (OpOperand *operand : transposedOperands) {
    promotedOperands.push_back(operand->getOperandNumber());
  }
  IREE::GPU::appendPromotedOperandsList(context, attrs, promotedOperands);
  DictionaryAttr configDict = DictionaryAttr::get(context, attrs);
  IREE::GPU::LoweringConfigAttr loweringConfig =
      IREE::GPU::LoweringConfigAttr::get(context, configDict);

  IREE::GPU::GPUPipelineOptionsAttr pipelineOptions =
      IREE::GPU::GPUPipelineOptionsAttr::get(
          context, /*prefetch_num_stages=*/0,
          /*no_reduce_shared_memory_bank_conflicts=*/false,
          /*use_igemm_convolution=*/false,
          /*reorder_workgroups_strategy=*/std::nullopt);
  DictionaryAttr pipelineConfig = DictionaryAttr::get(
      context,
      {NamedAttribute(IREE::GPU::GPUPipelineOptionsAttr::getDictKeyName(),
                      pipelineOptions)});
  const int64_t targetSubgroupSize = target.getPreferredSubgroupSize();

  // TODO(qedawkins): Use a shared pipeline identifier here.
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, linalgOp, loweringConfig,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTileAndFuse,
      workgroupSize, targetSubgroupSize, pipelineConfig);
}

//====---------------------------------------------------------------------===//
// UKernel Pipeline Configuration
//====---------------------------------------------------------------------===//

/// Set the configuration for argmax when ukernels are enabled.
/// Distribute all parallel dim across different workgroups, and only use single
/// subgroup per workgroup.
static LogicalResult setArgmaxUkernelConfig(
    IREE::GPU::TargetAttr target, mlir::FunctionOpInterface entryPoint,
    linalg::GenericOp op, IREE::Codegen::UKernelDescriptorAttr ukernelConfig) {
  SmallVector<unsigned> parallelDims;
  SmallVector<unsigned> reductionDims;
  op.getParallelDims(parallelDims);
  op.getReductionDims(reductionDims);

  // Currently Argmax UKernel only support 1 reduction dim.
  if (reductionDims.size() != 1) {
    return failure();
  }

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
      {"workgroup", b.getI64ArrayAttr(workgroupTileSizes)},
      {"reduction", b.getI64ArrayAttr(reductionTileSizes)}};
  op->setAttr(kUkernelAttrName, ukernelConfig);
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
      if (inputDim % (dim * chosenTileSize) != 0) {
        continue;
      }
    } else {
      for (int64_t t = residualTilingFactor; t >= 1; t >>= 1) {
        if (inputDim % (dim * t) == 0) {
          chosenTileSize = t;
          break;
        }
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
      LDBG() << "Tile and fuse convolution config";
      return success();
    }
  }
  const bool isNCHW = isa<linalg::Conv2DNchwFchwOp>(*linalgOp);
  const bool isNHWC = isa<linalg::Conv2DNhwcHwcfOp>(*linalgOp);

  const int ohIndex = isNHWC ? 1 : 2;
  const int owIndex = isNHWC ? 2 : 3;
  const int ocIndex = isNHWC ? 3 : 1;

  Type inputType = linalgOp.getDpsInputOperand(0)->get().getType();
  ArrayRef<int64_t> inputShape = cast<ShapedType>(inputType).getShape();
  Type outputType = linalgOp.getDpsInitOperand(0)->get().getType();
  ArrayRef<int64_t> outputShape = cast<ShapedType>(outputType).getShape();
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
                            workgroupTileSizes[3])) {
      return failure();
    }

    // Deduce the configuration for the OW and OH dimension. Try to make them
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
  if (isNCHW) {
    workgroupTileSizes.append({4, 1, 1});
  } else if (isNHWC) {
    workgroupTileSizes.append({1, 1, 4});
  }
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
                                   Operation *computeOp,
                                   const GPUCodegenOptions &gpuOpts) {
  IREE::Codegen::UKernelDescriptorAttr ukernelConfig = selectUKernel(computeOp);
  LLVM_DEBUG({
    DBGS() << "Selecting root config for: ";
    computeOp->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
    llvm::dbgs() << "\n";
  });
  if (succeeded(setDataTiledMmaInnerTiledLoweringConfig(
          target, entryPointFn, computeOp, ukernelConfig))) {
    LDBG() << "Tile and fuse data tiled MMA inner_tiled config";
    return success();
  }
  if (clGPUUseTileAndFuseMatmul) {
    if (succeeded(IREE::GPU::setMatmulLoweringConfig(
            target, entryPointFn, computeOp, clUseDirectLoad))) {
      LDBG() << "Tile and fuse matmul config";
      return success();
    }
  }
  if (clDirectConvolution) {
    if (succeeded(IREE::GPU::setDirectConvolutionLoweringConfig(
            target, entryPointFn, computeOp))) {
      LDBG() << "Tile and fuse direct convolution config";
      return success();
    }
  }
  if (clLLVMGPUUseIgemm) {
    if (succeeded(IREE::GPU::setIGEMMConvolutionLoweringConfig(
            target, entryPointFn, computeOp, clUseDirectLoad,
            clGPUPadConvolution))) {
      LDBG() << "Tile and fuse IGEMM config";
      return success();
    }
  }
  if (clGPUTestTileAndFuseVectorize) {
    if (succeeded(IREE::GPU::setTileAndFuseLoweringConfig(target, entryPointFn,
                                                          computeOp))) {
      LDBG() << "Tile and fuse default config";
      return success();
    }
  }
  if (succeeded(setVectorDistributionConfig(target, entryPointFn, computeOp,
                                            gpuOpts))) {
    return success();
  }

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(computeOp)) {
    if (succeeded(setContractConfig(target, entryPointFn, linalgOp))) {
      LDBG() << "Contract Config";
      return success();
    }
    if (clGPUEnableReductionVectorDistribution) {
      LDBG() << "ReductionVectorDistribution: finding a suitable config...";
      if (succeeded(
              IREE::GPU::setReductionConfig(target, entryPointFn, linalgOp))) {
        LDBG() << "Vector Distribution Subgroup Reduction Config";
        return success();
      }
      LDBG() << "ReductionVectorDistribution: failed to find a suitable config";
    }
    if (succeeded(setConvolutionConfig(target, entryPointFn, linalgOp, 16))) {
      LDBG() << "Convolution Config";
      return success();
    }
    auto genericOp = dyn_cast<linalg::GenericOp>(computeOp);
    if (genericOp) {
      if (genericOp &&
          succeeded(setTransposeConfig(target, entryPointFn, genericOp))) {
        LDBG() << "Transpose Config";
        return success();
      }
      if (ukernelConfig &&
          succeeded(setArgmaxUkernelConfig(target, entryPointFn, genericOp,
                                           ukernelConfig))) {
        LDBG() << "Argmax Ukernel Config";
        return success();
      }
      if (succeeded(IREE::GPU::setTileAndFuseLoweringConfig(
              target, entryPointFn, linalgOp))) {
        LDBG() << "Tile and Fuse Config";
        return success();
      }
    }
  }
  return TypeSwitch<Operation *, LogicalResult>(computeOp)
      .Case([&](IREE::LinalgExt::FftOp fftOp) {
        LDBG() << "FFT Config";
        return setFftConfig(target, entryPointFn, fftOp);
      })
      .Case([&](IREE::LinalgExt::SortOp sortOp) {
        LDBG() << "Sort Config";
        return IREE::GPU::setSortConfig(target, entryPointFn, sortOp);
      })
      .Case<IREE::LinalgExt::WinogradInputTransformOp,
            IREE::LinalgExt::WinogradOutputTransformOp,
            IREE::LinalgExt::WinogradFilterTransformOp>([&](auto winogradOp) {
        LDBG() << "Winograd Config";
        return setWinogradOpConfig(target, entryPointFn, winogradOp);
      })
      .Case([&](IREE::LinalgExt::CustomOp customOp) {
        LDBG() << "CustomOp Config";
        return setDefaultCustomOpLoweringConfig(
            entryPointFn, customOp, [&](FunctionOpInterface funcOp) {
              return initGPULaunchConfig(funcOp, gpuOpts);
            });
      })
      .Case([&](IREE::LinalgExt::ScatterOp scatterOp) {
        LDBG() << "ScatterOp Config";
        if (failed(IREE::GPU::setScatterLoweringConfig(target, entryPointFn,
                                                       scatterOp))) {
          return setRootDefaultConfig(target, entryPointFn, computeOp);
        }
        return success();
      })
      .Default([&](auto op) {
        LDBG() << "Default Config";
        if (clLLVMGPUVectorizePipeline) {
          return setRootDefaultConfig(target, entryPointFn, computeOp);
        }
        if (succeeded(IREE::GPU::setTileAndFuseLoweringConfig(
                target, entryPointFn, computeOp))) {
          LDBG() << "Tile and fuse default config";
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
static void
propagateLoweringConfig(Operation *rootOperation,
                        const SmallVector<Operation *> &computeOps) {
  if (IREE::Codegen::LoweringConfigAttrInterface config =
          getLoweringConfig(rootOperation)) {
    for (Operation *op : computeOps) {
      if (op == rootOperation) {
        continue;
      }
      setLoweringConfig(op, config);
    }
  }
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//
LogicalResult initGPULaunchConfig(FunctionOpInterface funcOp,
                                  const GPUCodegenOptions &gpuOpts) {
  IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
  if (!target) {
    return funcOp.emitError("missing GPU target in #hal.executable.target");
  }

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
      for (Operation *op : computeOps) {
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
    if (!isa<linalg::CopyOp, linalg::GenericOp, linalg::FillOp,
             IREE::LinalgExt::ScatterOp, IREE::LinalgExt::MapStoreOp,
             linalg::PackOp, linalg::UnPackOp>(op)) {
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
      if (isa<IREE::LinalgExt::ScatterOp, IREE::LinalgExt::MapStoreOp,
              linalg::CopyOp, linalg::FillOp>(op)) {
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

  if (failed(setRootConfig(target, funcOp, rootOperation, gpuOpts))) {
    return funcOp.emitOpError("failed to set root config");
  }

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
