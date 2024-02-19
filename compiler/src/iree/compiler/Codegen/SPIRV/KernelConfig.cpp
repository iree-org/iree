// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Strategies.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "iree-spirv-kernel-config"

using llvm::divideCeil;
using llvm::APIntOps::GreatestCommonDivisor;

// The default number of tiles along K dimension to use per workgroup.
constexpr unsigned numTilesPerSubgroupDimK = 2;

constexpr int kMaxVectorNumBits = 128;

namespace mlir::iree_compiler {

llvm::cl::opt<bool> clSPIRVEnableTransformDialectJit(
    "iree-spirv-enable-transform-dialect-jit",
    llvm::cl::desc("Enable the usage of the transform dialect JIT"),
    llvm::cl::init(false));

using CodeGenPipeline = IREE::Codegen::DispatchLoweringPassPipeline;

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

bool isMatmulOrBatchMatmul(linalg::LinalgOp linalgOp) {
  // (Batch) matmul should be a reduction op with 2/3 parallel dimensions.
  if (!linalg::isaContractionOpInterface(linalgOp) ||
      !llvm::is_contained({2u, 3u}, linalgOp.getNumParallelLoops()))
    return false;

  // Also exclude the case of matvec, which has only one non-unit parallel dim.
  // They should go down different pipelines.
  int nonUnitParallelDimCount = 0;
  SmallVector<int64_t, 4> bounds = linalgOp.getStaticLoopRanges();
  FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
      mlir::linalg::inferContractionDims(linalgOp);
  assert(succeeded(contractionDims) && "Could not infer contraction dims");
  for (auto mDim : contractionDims->m) {
    nonUnitParallelDimCount += bounds[mDim] != 1;
  }
  for (auto nDim : contractionDims->n) {
    nonUnitParallelDimCount += bounds[nDim] != 1;
  }
  return nonUnitParallelDimCount > 1;
}

// Check if the given linalg op is fused with another op that may result
// in too much shared memory usage.
static bool fusedOpMayUseExtraSharedMemory(linalg::LinalgOp matmul) {
  if (matmul->getNumResults() != 1)
    return true;

  auto entryPoint = matmul->getParentOfType<mlir::FunctionOpInterface>();

  auto getResultBits = [](linalg::LinalgOp linalgOp) {
    auto shapedType = llvm::cast<ShapedType>(linalgOp->getResult(0).getType());
    return IREE::Util::getTypeBitWidth(shapedType.getElementType());
  };
  auto matmulResultBits = getResultBits(matmul);

  bool fusedWithOp = false;
  entryPoint.walk([&](linalg::LinalgOp linalgOp) {
    if (linalgOp == matmul || isMatmulOrBatchMatmul(linalgOp) ||
        isa<linalg::FillOp>(linalgOp)) {
      return WalkResult::advance();
    }

    if (linalgOp->getNumResults() != 1 ||
        getResultBits(linalgOp) != matmulResultBits) {
      fusedWithOp = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return fusedWithOp;
}

//===----------------------------------------------------------------------===//
// Convolution Default Configuration
//===----------------------------------------------------------------------===//

/// Decides the tiling and distribution parameters for one convolution
/// dimension. Returns true if we can succesfully deduce.
///
/// - `inputDim` is the size of the dimension to be distributed.
/// - `residualThreads` is the remaining threads we can distribute.
/// - `residualTilingFactor` indicates the remaining tiling scale factor.
/// - `wgDimSize` will be updated with the decided workgroup dimension size.
/// - `wgTileSize` will be updated with the decided workgroup tile size.
static bool tileConvOneDim(const int64_t inputDim, const bool isInnerMostDim,
                           int vectorSize, int64_t &residualThreads,
                           int64_t &residualTilingFactor, int64_t &wgDimSize,
                           int64_t &wgTileSize) {
  const int64_t lb = isInnerMostDim ? 2 : 1;
  for (int64_t dim = residualThreads; dim >= lb; dim >>= 1) {
    int64_t chosenTileSize = 0;
    if (isInnerMostDim) {
      // Handle `vectorSize` elements per thread for the innermost dimension.
      // We need this for the best utilization of memory.
      chosenTileSize = vectorSize;
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
/// succesfully deduce.
static bool tileConvSquare(const int64_t oh, const int64_t ow,
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

namespace detail {

LogicalResult setConvOpConfig(linalg::LinalgOp linalgOp,
                              const int64_t subgroupSize,
                              const int64_t bestTilingFactor) {
  assert(isa<linalg::ConvolutionOpInterface>(*linalgOp));
  LLVM_DEBUG(llvm::dbgs() << "trying to deduce config as convolution...\n");

  Type inputType = linalgOp.getDpsInputOperand(0)->get().getType();
  ArrayRef<int64_t> inputShape = llvm::cast<ShapedType>(inputType).getShape();
  Type outputType = linalgOp.getDpsInitOperand(0)->get().getType();
  ArrayRef<int64_t> outputShape = llvm::cast<ShapedType>(outputType).getShape();
  // Restrict to pure 4-D input/output shapes for now. This excludes convolution
  // ops with 1- or 3-D window sizes. It also excludes 2-D-window convolution
  // ops like `linalg.depthwise_conv_2d_nhwc_hwcm`.
  if (inputShape.size() != 4 || outputShape.size() != 4)
    return failure();

  auto convDimsOrFailure = linalg::inferConvolutionDims(linalgOp);
  if (failed(convDimsOrFailure))
    return failure();
  const mlir::linalg::ConvolutionDimensions &convDims = *convDimsOrFailure;
  LLVM_DEBUG({
    llvm::dbgs() << "conv: " << linalgOp;
    llvm::dbgs() << "\nconv batch dim: ";
    llvm::interleaveComma(convDims.batch, llvm::dbgs());
    llvm::dbgs() << "\nconv output window dims: ";
    llvm::interleaveComma(convDims.outputImage, llvm::dbgs());
    llvm::dbgs() << "\nconv output channel dim: ";
    llvm::interleaveComma(convDims.outputChannel, llvm::dbgs());
    llvm::dbgs() << "\nconv filter window dims: ";
    llvm::interleaveComma(convDims.filterLoop, llvm::dbgs());
    llvm::dbgs() << "\nconv input channel dims: ";
    llvm::interleaveComma(convDims.inputChannel, llvm::dbgs());
    llvm::dbgs() << "\nconv depth multiplier: ";
    llvm::interleaveComma(convDims.depth, llvm::dbgs());
    llvm::dbgs() << "\n";
  });
  assert(convDims.outputImage.size() == 2);
  assert(convDims.filterLoop.size() == 2);

  SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();

  const int ohIndex = convDims.outputImage.front();
  const int64_t oh = loopRanges[ohIndex];
  const int64_t ow = loopRanges[convDims.outputImage.back()];
  int ocIndex;
  if (!convDims.outputChannel.empty()) {
    assert(convDims.outputChannel.size() == 1);
    ocIndex = convDims.outputChannel.front();
  } else if (!convDims.depth.empty()) {
    // For depthwise convolution ops with multipler 1, we have the same
    // input/filter/output channel size, which is being categorized as the
    // multipler.
    assert(convDims.depth.size() == 1);
    ocIndex = convDims.depth.front();
  } else {
    // For pooling ops, the input/output channel size will be categorized
    // as the additional batch dimension.
    assert(convDims.batch.size() == 2);
    ocIndex = convDims.batch.back();
  }
  const int64_t oc = loopRanges[ocIndex];
  // We may not have an input channel dimension in the case of depthwise
  // convolution ops.
  std::optional<int64_t> ic = std::nullopt;
  if (!convDims.inputChannel.empty()) {
    assert(convDims.inputChannel.size() == 1);
    ic = loopRanges[convDims.inputChannel.front()];
  }

  if ((ic && ShapedType::isDynamic(*ic)) ||
      llvm::any_of(outputShape.drop_front(), ShapedType::isDynamic)) {
    return failure();
  }

  const int bitwidth =
      llvm::cast<ShapedType>(outputType).getElementTypeBitWidth();
  const int vectorSize = kMaxVectorNumBits / bitwidth;

  // We use `vectorSize` as the tile size along IC dimension. If smaller than
  // 4, it will be unrolled into size 1.
  if (ic && !(*ic % vectorSize == 0 || *ic < 4))
    return failure();

  // The core idea is to distribute the convolution dimensions to the workgroup
  // Z/Y/X dimensions, with each thread in a workgroup handling multiple vector
  // elements. We try to 1) utilize all threads in a subgroup, and 2) handle an
  // optimal tile size along each dimension.

  int64_t residualThreads = subgroupSize;
  int64_t residualTilingFactor = bestTilingFactor;

  SmallVector<int64_t, 3> workgroupSize(3, 1); // (X, Y, Z)
  SmallVector<int64_t> workgroupTileSizes(4, 0);

  const bool isNCHW = ocIndex < ohIndex;
  if (isNCHW) {
    // OW -> x, OH -> y, OC -> z
    if (!tileConvOneDim(ow, /*isInnerMostDim=*/true, vectorSize,
                        residualThreads, residualTilingFactor, workgroupSize[0],
                        workgroupTileSizes[3]) ||
        !tileConvOneDim(oh, /*isInnerMostDim=*/false, vectorSize,
                        residualThreads, residualTilingFactor, workgroupSize[1],
                        workgroupTileSizes[2]) ||
        !tileConvOneDim(oc, /*isInnerMostDim=*/false, vectorSize,
                        residualThreads, residualTilingFactor, workgroupSize[2],
                        workgroupTileSizes[1])) {
      return failure();
    }
  } else {
    // OC -> x
    if (!tileConvOneDim(oc, /*isInnerMostDim=*/true, vectorSize,
                        residualThreads, residualTilingFactor, workgroupSize[0],
                        workgroupTileSizes[3]))
      return failure();

    // Deduce the configruation for the OW and OH dimension. Try to make them
    // even if possible given we typically have images with the same height
    // and width.
    const bool tileToSquare = tileConvSquare(
        oh, ow, residualThreads, residualTilingFactor,
        llvm::MutableArrayRef(workgroupSize).drop_front(),
        llvm::MutableArrayRef(workgroupTileSizes).drop_front().drop_back());

    // Otherwise treat OW and OH separately to allow them to have different
    // number of threads and tiling size.
    if (!tileToSquare) {
      if (!tileConvOneDim(ow, /*isInnerMostDim=*/false, vectorSize,
                          residualThreads, residualTilingFactor,
                          workgroupSize[1], workgroupTileSizes[2]) ||
          !tileConvOneDim(oh, /*isInnerMostDim=*/false, vectorSize,
                          residualThreads, residualTilingFactor,
                          workgroupSize[2], workgroupTileSizes[1])) {
        return failure();
      }
    }
  }

  SmallVector<int64_t> threadTileSizes(4, 0);
  threadTileSizes[0] = 1; // Tile along the N dimension with size 1
  for (int i = 1; i <= 3; ++i) {
    threadTileSizes[i] = workgroupTileSizes[i] / workgroupSize[3 - i];
  }

  auto pipeline = CodeGenPipeline::SPIRVBaseVectorize;
  TileSizesListType tileSizes;
  tileSizes.push_back(workgroupTileSizes);
  tileSizes.push_back(threadTileSizes);

  // Tiling along reduction dimensions.
  SmallVector<int64_t> reductionTileSizes(loopRanges.size(), 0);
  // For filter window dimensions, use tile size 1 to reduce the contraction
  // problem to be similar to a normal matmul.
  reductionTileSizes[convDims.filterLoop.front()] = 1;
  reductionTileSizes[convDims.filterLoop.back()] = 1;
  if (ic) {
    // Tile input channel dimension with size 4 to avoid code bloat in
    // vectorization later.
    reductionTileSizes[convDims.inputChannel.front()] = vectorSize;
  }
  tileSizes.push_back(reductionTileSizes);

  // Tile along OH by size 1 to enable downsizing 2-D convolution to 1-D.
  SmallVector<int64_t> windowTileSizes(4, 0);
  windowTileSizes[ohIndex] = 1;
  tileSizes.push_back(windowTileSizes);

  auto funcOp = linalgOp->getParentOfType<mlir::FunctionOpInterface>();
  return setOpConfigAndEntryPointFnTranslation(funcOp, linalgOp, tileSizes,
                                               pipeline, workgroupSize);
}

} // namespace detail

//===----------------------------------------------------------------------===//
// Matmul Default Configuration
//===----------------------------------------------------------------------===//

/// Given the linalg `op` with `lhsShape` and `rhsShape`, tries to treat as a
/// (batch) matmul like op and deduce the index of the loop corresponding to
/// B/M/N/K dimension respectively. Returns -1 as the index if unable to deduce.
std::tuple<int, int, int, int> getMatmulBMNKIndex(linalg::LinalgOp op,
                                                  int *lastParallelDim) {
  OpOperand *lhs = op.getDpsInputOperand(0);
  OpOperand *rhs = op.getDpsInputOperand(1);
  auto lhsShape = llvm::cast<ShapedType>(lhs->get().getType()).getShape();
  auto rhsShape = llvm::cast<ShapedType>(rhs->get().getType()).getShape();

  auto lhsLoopIndices =
      llvm::map_to_vector(llvm::seq<int>(0, lhsShape.size()), [&](int i) {
        return op.getMatchingIndexingMap(lhs).getDimPosition(i);
      });
  auto rhsLoopIndices =
      llvm::map_to_vector(llvm::seq<int>(0, rhsShape.size()), [&](int i) {
        return op.getMatchingIndexingMap(rhs).getDimPosition(i);
      });

  // Figure out what dimension each loop corresponds to.
  int bIndex = -1, mIndex = -1, nIndex = -1, kIndex = -1;
  for (unsigned i = 0; i < op.getNumLoops(); ++i) {
    if (linalg::isReductionIterator(op.getIteratorTypesArray()[i])) {
      kIndex = i;
      continue;
    }

    const bool inLHS = llvm::is_contained(lhsLoopIndices, i);
    const bool inRHS = llvm::is_contained(rhsLoopIndices, i);
    if (inLHS && inRHS) {
      bIndex = i;
    } else if (inLHS) {
      // For cases where we have two parallel dimensions only accessed by
      // the LHS, treat the outer one of them as the batch dimension.
      if (mIndex >= 0 && bIndex < 0)
        bIndex = mIndex;
      mIndex = i;
    } else if (inRHS) {
      // For cases where we have two parallel dimensions only accessed by
      // the RHS, treat the outer one of them as the batch dimension.
      if (nIndex >= 0 && bIndex < 0)
        bIndex = nIndex;
      nIndex = i;
    }
    if (lastParallelDim)
      *lastParallelDim = i;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "(B, M, N, K) indices = (" << bIndex << ", " << mIndex
                 << ", " << nIndex << ", " << kIndex << ")\n";
  });
  return {bIndex, mIndex, nIndex, kIndex};
}

/// Decides the tiling and distribution parameters for matmul's N dimension to
/// workgroup X dimension.
static bool tileMatmulNToWorkgroupX(const int64_t dimN,
                                    const int64_t bestThreadN,
                                    int64_t &residualThreads,
                                    const int64_t bestX,
                                    int64_t &residualTilingFactor,
                                    int64_t &wgDimSize, int64_t &wgTileSize) {
  // Deduce the configuration for the N dimension. Start with the best workgroup
  // X size, and reduce by a factor of two each time.
  for (int64_t x = bestX; x >= 2; x >>= 1) {
    // Handle 4 elements per thread for the innermost dimension. We need this
    // for vectorized load.
    int64_t chosenTileSize = bestThreadN;
    if (dimN % (x * chosenTileSize) == 0) {
      wgDimSize = x;
      wgTileSize = x * chosenTileSize;
      residualThreads /= x;
      assert(residualTilingFactor % chosenTileSize == 0);
      residualTilingFactor /= chosenTileSize;
      return true;
    }
  }
  return false;
}

/// Decides the tiling and distribution parameters for matmul's M dimension to
/// workgroup Y dimension.
static bool tileMatmulMToWorkgroupY(const int64_t dimM,
                                    const int64_t bestThreadM,
                                    int64_t &residualThreads,
                                    const int64_t bestY,
                                    int64_t &residualTilingFactor,
                                    int64_t &wgDimSize, int64_t &wgTileSize) {
  // Deduce the configuration for the M dimension. Start with the best workgroup
  // Y size, and reduce by a factor of two each time.
  for (int64_t y = residualThreads; y >= 1; y >>= 1) {
    int64_t chosenTileSize = 0;
    // Reduce the thread tiling size by one each time. We read one row each
    // time; so it's fine to not be some power of two here.
    for (int64_t t = bestThreadM; t >= 1; --t) {
      if (dimM % (y * t) == 0) {
        chosenTileSize = t;
        break;
      }
    }
    if (chosenTileSize) {
      wgDimSize = y;
      wgTileSize = y * chosenTileSize;
      assert(residualTilingFactor > chosenTileSize);
      residualTilingFactor -= chosenTileSize;
      return true;
    }
  }
  return false;
}

/// Decides the tiling parameters for matmul's K dimension.
static bool tileMatmulK(const int64_t dimK, const int64_t residualTilingFactor,
                        int64_t &tileSize) {
  // Deduce the configuration for the K dimension. We need some power of two
  // here so that we can do vector load.
  for (int64_t t = llvm::bit_floor<uint64_t>(residualTilingFactor); t >= 2;
       t >>= 1) {
    if (dimK % t == 0) {
      tileSize = t;
      return true;
    }
  }
  return false;
}

int64_t getTileBytes(int64_t mTileSize, int64_t nTileSize, int64_t kTileSize,
                     int64_t elementBits, bool promoteC) {
  int64_t paddingBits = detail::bankConflictReductionPaddingBits / elementBits;
  int64_t count = (mTileSize + nTileSize) * (kTileSize + paddingBits);
  if (promoteC)
    count += mTileSize * (nTileSize + paddingBits);
  return (elementBits / 8) * count;
}

int64_t getMultiBufferMemoryUsage(int64_t singleBufferBytes, unsigned depth,
                                  unsigned storeStage) {
  if (depth == 0)
    return singleBufferBytes;
  return singleBufferBytes * (storeStage == 1 ? depth : depth + 1);
};

/// Tries to adjust workgroup and tile sizes to enable vector load for both
/// matmul LHS and RHS. Returns false only when it's not beneficial to promote.
static bool adjustToVectorLoad(ArrayRef<int64_t> dimMNKSize, int64_t &mTileSize,
                               int64_t &nTileSize, int64_t &kTileSize,
                               SmallVectorImpl<int64_t> &wgSize,
                               const int64_t subgroupSize, int64_t vectorSize) {
  const int64_t totalThreads = wgSize[0] * wgSize[1] * wgSize[2];
  LLVM_DEBUG(llvm::dbgs() << "initial total thread = " << totalThreads << "\n");
  if (totalThreads <= subgroupSize)
    return false;

  const bool canVectorLoadLHS = canPerformVectorAccessUsingAllThreads(
      {mTileSize, kTileSize}, totalThreads, vectorSize);
  const bool canVectorLoadRHS = canPerformVectorAccessUsingAllThreads(
      {kTileSize, nTileSize}, totalThreads, vectorSize);
  LLVM_DEBUG(llvm::dbgs() << "LHS vector load: " << canVectorLoadLHS << "\n");
  LLVM_DEBUG(llvm::dbgs() << "RHS vector load: " << canVectorLoadRHS << "\n");

  // If we can perform vector load of neither, just don't use shared memory.
  if (!canVectorLoadLHS && !canVectorLoadRHS)
    return false;

  // If we can only perform vector load of one operands, adjust the tiling
  // scheme to see if we can make both work. Increase K to load more data for
  // the smaller tile; decrease M or N, for the larger tile.
  if (canVectorLoadLHS && !canVectorLoadRHS) {
    for (const int scale : {2, 4}) {
      const int64_t newKTileSize = kTileSize * scale;
      if (dimMNKSize[2] % newKTileSize != 0)
        continue;
      const int64_t newMTileSize = mTileSize / scale;
      const int64_t newWgMDim = wgSize[1] / scale;
      if (newMTileSize == 0 || newWgMDim == 0)
        continue;
      const int64_t newCount = wgSize[0] * newWgMDim * wgSize[2];
      if (newCount <= subgroupSize)
        continue;
      if (!canPerformVectorAccessUsingAllThreads({newMTileSize, newKTileSize},
                                                 newCount, vectorSize) ||
          !canPerformVectorAccessUsingAllThreads({newKTileSize, nTileSize},
                                                 newCount, vectorSize)) {
        continue;
      }
      LLVM_DEBUG({
        llvm::dbgs() << "initial [M, N, K] tile size = [" << mTileSize << ", "
                     << nTileSize << ", " << kTileSize << "]\n";
        llvm::dbgs() << "revised [M, N, K] tile size = [" << newMTileSize
                     << ", " << nTileSize << ", " << newKTileSize << "]\n";
      });
      mTileSize = newMTileSize;
      kTileSize = newKTileSize;
      wgSize[1] = newWgMDim;
      break;
    }
  }
  // TODO: improve (!canVectorLoadLHS && canVectorLoadRHS)

  return true;
}

/// Tries to adjust workgorup and tile sizes to promote matmul LHS and RHS and
/// returns true if it's beneficial to promote.
static bool adjustToPromote(ArrayRef<int64_t> dimMNKSize, int64_t &mTileSize,
                            int64_t &nTileSize, int64_t &kTileSize,
                            SmallVectorImpl<int64_t> &wgSize,
                            unsigned &pipelineDepth, unsigned &storeStage,
                            const int subgroupSize, const int maxBytes,
                            const int elementBits) {
  LLVM_DEBUG(llvm::dbgs() << "subgroup size = " << subgroupSize << "\n");
  const int vectorSize = kMaxVectorNumBits / elementBits;
  if (!adjustToVectorLoad(dimMNKSize, mTileSize, nTileSize, kTileSize, wgSize,
                          subgroupSize, vectorSize))
    return false;

  // Don't do multibuffering if the inner reduction loop is folded out.
  if (dimMNKSize[2] == kTileSize) {
    pipelineDepth = 1;
    storeStage = 1;
  }

  auto usedBytes =
      getTileBytes(mTileSize, nTileSize, kTileSize, elementBits, false);

  LLVM_DEBUG(llvm::dbgs() << "initial multibuffering bytes = "
                          << getMultiBufferMemoryUsage(usedBytes, pipelineDepth,
                                                       storeStage)
                          << "\n");

  // First try to fit the given tile sizes with the largest pipelining depth
  // possible.
  do {
    if (getMultiBufferMemoryUsage(usedBytes, pipelineDepth, storeStage) <=
        maxBytes)
      return true;
  } while (pipelineDepth-- > 1);

  // If we can't fit in workgroup memory, don't multibuffer.
  pipelineDepth = 1;

  if (storeStage == 0) {
    storeStage = 1;
    if (getMultiBufferMemoryUsage(usedBytes, pipelineDepth, storeStage) <=
        maxBytes)
      return true;
  }

  // Using too much workgroup memory. Try to reduce the tile size for X/Y once
  // by a factor of two.
  int64_t &wgDimSize = wgSize[0] > wgSize[1] ? wgSize[0] : wgSize[1];
  int64_t &tileSize = wgSize[0] > wgSize[1] ? nTileSize : mTileSize;
  assert(wgDimSize % 2 == 0);
  wgDimSize /= 2;
  tileSize /= 2;

  int64_t totalThreads = wgSize[0] * wgSize[1] * wgSize[2];
  LLVM_DEBUG(llvm::dbgs() << "revised total thread = " << totalThreads << "\n");
  usedBytes = getTileBytes(mTileSize, nTileSize, kTileSize, elementBits, false);
  LLVM_DEBUG(llvm::dbgs() << "revised tile bytes = " << usedBytes << "\n");
  return totalThreads > subgroupSize && usedBytes <= maxBytes;
}

namespace detail {

LogicalResult setMatmulOpConfig(spirv::ResourceLimitsAttr limits,
                                linalg::LinalgOp op,
                                std::array<int64_t, 2> bestWorkgroupSizeXY,
                                std::array<int64_t, 3> bestThreadTileSizeMNK,
                                bool enablePromotion,
                                unsigned softwarePipelineDepth,
                                unsigned softwarePipelineStoreStage) {
  LLVM_DEBUG(llvm::dbgs() << "trying to deduce config as matmul...\n");
  OpOperand *lhs = op.getDpsInputOperand(0);
  OpOperand *rhs = op.getDpsInputOperand(1);

  auto lhsType = llvm::cast<ShapedType>(lhs->get().getType());
  auto rhsType = llvm::cast<ShapedType>(rhs->get().getType());
  auto elementBits =
      static_cast<int>(IREE::Util::getTypeBitWidth(lhsType.getElementType()));
  if (!llvm::is_contained({8, 16, 32}, elementBits))
    return failure();

  ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ArrayRef<int64_t> rhsShape = rhsType.getShape();
  if (llvm::any_of(lhsShape, ShapedType::isDynamic))
    return failure();
  if (llvm::any_of(rhsShape, ShapedType::isDynamic))
    return failure();

  assert(llvm::is_contained({2u, 3u}, op.getNumParallelLoops()));

  int lastParallelDim = -1;
  const auto [bIndex, mIndex, nIndex, kIndex] =
      getMatmulBMNKIndex(op, &lastParallelDim);
  if (mIndex < 0 || nIndex < 0 || kIndex < 0)
    return failure();
  const bool isBM = bIndex >= 0;

  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  const unsigned numLoops = loopRanges.size();

  const int64_t dimM = loopRanges[mIndex];
  const int64_t dimK = loopRanges[kIndex];
  const int64_t dimN = loopRanges[nIndex];

  // The core idea is to distribute the matmul M/N dimension to the workgroup
  // Y/X dimension, with each thread in a workgroup handling multiple vector
  // elements. We start from the best (X, Y) and the tiling sizes for (M, N, K)
  // and try different configurations by scaling them down until we find a
  // configuration that can perfectly tile the input matmul.

  const int64_t bestThreadM = bestThreadTileSizeMNK[0],
                bestThreadN = bestThreadTileSizeMNK[1],
                bestThreadK = bestThreadTileSizeMNK[2];

  int64_t bestX = bestWorkgroupSizeXY[0], bestY = bestWorkgroupSizeXY[1];
  // We will deduce a configuration first for x and then y. But look at y here
  // to see if the problem size is too small; for such cases, "shift" the
  // parallelism to x.
  if (dimM < bestThreadM) {
    int64_t factor = llvm::PowerOf2Ceil(divideCeil(bestThreadM, dimM));
    bestX *= factor;
    bestY = divideCeil(bestY, factor);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "best thread tile size (M, N, K) = (" << bestThreadM << ", "
                 << bestThreadN << ", " << bestThreadK << ")\n";
    llvm::dbgs() << "best workgroup size (X, Y) = (" << bestX << ", " << bestY
                 << ")\n";
  });

  int64_t residualThreads = bestX * bestY;
  int64_t residualTilingFactor = (bestThreadM + bestThreadK) * bestThreadN;

  SmallVector<int64_t, 3> workgroupSize(3, 1); // (X, Y, Z)
  SmallVector<int64_t> workgroupTileSizes(numLoops, 0);
  SmallVector<int64_t> reductionTileSizes(numLoops, 0);

  if (isBM)
    workgroupTileSizes[bIndex] = 1;

  if (!tileMatmulNToWorkgroupX(dimN, bestThreadN, residualThreads, bestX,
                               residualTilingFactor, workgroupSize[0],
                               workgroupTileSizes[nIndex]) ||
      !tileMatmulMToWorkgroupY(dimM, bestThreadM, residualThreads, bestY,
                               residualTilingFactor, workgroupSize[1],
                               workgroupTileSizes[mIndex]) ||
      !tileMatmulK(dimK, residualTilingFactor, reductionTileSizes[kIndex])) {
    return failure();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "workgroup tile size before promotion = (";
    llvm::interleaveComma(workgroupTileSizes, llvm::dbgs());
    llvm::dbgs() << ")\n";
    llvm::dbgs() << "reduction tile size before promotion = (";
    llvm::interleaveComma(reductionTileSizes, llvm::dbgs());
    llvm::dbgs() << ")\n";
    llvm::dbgs() << "workgroup size before promotion = (";
    llvm::interleaveComma(workgroupSize, llvm::dbgs());
    llvm::dbgs() << ")\n";
  });

  const int subgroupSize = limits.getSubgroupSize();
  const int maxBytes = limits.getMaxComputeSharedMemorySize();

  // We want a 2-stage pipeline without multi-buffering if the depth is 0 to
  // keep the default for compilation configs that don't specify a pipeline
  // depth.
  auto pipelineDepth = softwarePipelineDepth ? softwarePipelineDepth : 1;
  auto storeStage = softwarePipelineStoreStage;

  // TODO: Remove this check once either bufferization doesn't produce an extra
  // buffer when fused with something like elementwise extf, or the shared
  // memory calculation incorporates the fused op properly.
  if ((pipelineDepth != 1 || storeStage != 1) &&
      fusedOpMayUseExtraSharedMemory(op)) {
    pipelineDepth = 1;
    storeStage = 1;
  }

  // TODO: Enable multibuffering with leading elementwise.
  if (hasFusedLeadingOp(op)) {
    pipelineDepth = 0;
    storeStage = 1;
  }

  // Try to adjust tiling sizes to fit in shared memory.
  auto usePromotionPipeline =
      enablePromotion &&
      adjustToPromote({dimM, dimN, dimK}, workgroupTileSizes[mIndex],
                      workgroupTileSizes[nIndex], reductionTileSizes[kIndex],
                      workgroupSize, pipelineDepth, storeStage, subgroupSize,
                      maxBytes, elementBits);

  // Tile all additional reduction dimensions with size 1 to materialize loops.
  for (auto [i, it] : llvm::enumerate(op.getIteratorTypesArray())) {
    if (linalg::isReductionIterator(it) && reductionTileSizes[i] == 0)
      reductionTileSizes[i] = 1;
  }

  TileSizesListType tileSizes;

  // Only the promotion pipeline has multibuffering + pipelining.
  if (usePromotionPipeline) {
    // Merge reductionTileSizes into workgroupTileSizes--this is needed by the
    // pipeline passes shared between SPIR-V and LLVMGPU.
    for (auto [i, it] : llvm::enumerate(op.getIteratorTypesArray())) {
      if (linalg::isReductionIterator(it))
        workgroupTileSizes[i] = reductionTileSizes[i];
    }
    tileSizes.push_back(workgroupTileSizes);

    return setOpConfigAndEntryPointFnTranslation(
        op->getParentOfType<mlir::FunctionOpInterface>(), op, tileSizes,
        CodeGenPipeline::SPIRVMatmulPromoteVectorize, workgroupSize,
        /*subgroupSize=*/std::nullopt,
        getSoftwarePipeliningAttrDict(op->getContext(), pipelineDepth,
                                      storeStage));
  }

  SmallVector<int64_t> threadTileSizes(numLoops, 0);
  if (isBM) {
    threadTileSizes[bIndex] = workgroupTileSizes[bIndex] / workgroupSize[2];
  }
  threadTileSizes[mIndex] = workgroupTileSizes[mIndex] / workgroupSize[1];
  threadTileSizes[nIndex] = workgroupTileSizes[nIndex] / workgroupSize[0];

  workgroupTileSizes.resize(lastParallelDim + 1);
  threadTileSizes.resize(lastParallelDim + 1);
  tileSizes.push_back(workgroupTileSizes);
  tileSizes.push_back(threadTileSizes);
  tileSizes.push_back(reductionTileSizes);
  return setOpConfigAndEntryPointFnTranslation(
      op->getParentOfType<mlir::FunctionOpInterface>(), op, tileSizes,
      CodeGenPipeline::SPIRVBaseVectorize, workgroupSize);
}

} // namespace detail

//===----------------------------------------------------------------------===//
// Cooperative Matrix Default Configuration
//===----------------------------------------------------------------------===//

bool isCooperativeMatrixFusable(linalg::GenericOp genericOp) {
  if (genericOp.getNumLoops() != genericOp.getNumParallelLoops())
    return false;

  // Look at fused elementwise ops to make sure they are allowed by the
  // cooperative matrix spec.
  for (Operation &op : genericOp.getBlock()->without_terminator()) {
    if (!isa<
            // These ops are directly allowed to use cooperative matrix types.
            arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp,
            arith::DivFOp, arith::DivSIOp, arith::DivUIOp, arith::NegFOp,
            arith::TruncFOp, arith::TruncIOp, arith::ExtFOp, arith::ExtSIOp,
            arith::ExtUIOp, arith::FPToSIOp, arith::FPToUIOp, arith::SIToFPOp,
            arith::UIToFPOp,
            // Special cases of these ops are directly allowed to sue
            // cooperative matrix types. Other cases can use a loop.
            arith::MulFOp>(op))
      return false;
  }

  // Look at operands to make sure we don't have inlined constants. Cooperative
  // matrix loads can only happen from StorageBuffer or Workgroup storage
  // classes.
  for (Value input : genericOp.getInputs()) {
    if (llvm::isa<TensorType>(input.getType())) {
      if (matchPattern(input, m_Constant()))
        return false;
      continue;
    }

    // For buffers we need to walk back the subview chain to see if it's
    // originally from a constant.
    while (auto subviewOp = input.getDefiningOp<memref::SubViewOp>()) {
      input = subviewOp.getViewSource();
    }
    if (auto toMemrefOp = input.getDefiningOp<bufferization::ToMemrefOp>()) {
      if (matchPattern(toMemrefOp.getTensor(), m_Constant()))
        return false;
    }
  }

  return true;
}

bool needToPrmoteCForCooperativeMatrix(linalg::LinalgOp matmulOp) {
  assert(matmulOp.hasPureTensorSemantics());
  Value result = matmulOp.getOperation()->getResult(0);
  if (!result.hasOneUse())
    return true; // Be conservative.
  Operation *user = *result.getUsers().begin();
  if (isa<IREE::Flow::DispatchTensorStoreOp>(user))
    return false;
  if (auto genericOp = dyn_cast<linalg::GenericOp>(user)) {
    return !isCooperativeMatrixFusable(genericOp);
  }
  return true; // Be conservative.
}

namespace detail {

LogicalResult setCooperativeMatrixConfig(
    const spirv::TargetEnv &targetEnv, linalg::LinalgOp op,
    const unsigned numSubgroupsPerWorkgroup,
    const unsigned numMNTilesPerSubgroup, unsigned softwarePipelineDepth,
    unsigned softwarePipelineStoreStage) {
  LLVM_DEBUG(llvm::dbgs() << "trying to matmul tensorcore config...\n");
  // This configuration is only for cooperative matrix.
  if (!targetEnv.allows(spirv::Capability::CooperativeMatrixKHR) ||
      !targetEnv.allows(spirv::Extension::SPV_KHR_cooperative_matrix)) {
    return failure();
  }

  if (op.hasDynamicShape())
    return failure();

  Value lhs = op.getDpsInputOperand(0)->get();
  Value rhs = op.getDpsInputOperand(1)->get();
  Value init = op.getDpsInitOperand(0)->get();

  int lastParallelDim = -1;
  const auto [bIndex, mIndex, nIndex, kIndex] =
      getMatmulBMNKIndex(op, &lastParallelDim);
  if (mIndex < 0 || nIndex < 0 || kIndex < 0)
    return failure();
  const bool isBM = bIndex >= 0;

  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();

  const int64_t dimM = loopRanges[mIndex];
  const int64_t dimK = loopRanges[kIndex];
  const int64_t dimN = loopRanges[nIndex];
  LLVM_DEBUG({
    llvm::dbgs() << "input matmul shape (B, M, N, K) = ("
                 << (bIndex >= 0 ? loopRanges[bIndex] : -1) << ", " << dimM
                 << ", " << dimN << ", " << dimK << ")\n";
  });

  // TODO: Cooperative matrix support is fairly restricted. We can only have
  // a curated list of fused element wise ops as defined in the extension
  // SPV_KHR_cooperative_matrix. Check that once we move bufferization after
  // vectorization.

  auto getElementType = [](Value v) {
    return llvm::cast<ShapedType>(v.getType()).getElementType();
  };

  Type lhsElem = getElementType(lhs);
  Type rhsElem = getElementType(rhs);
  Type initElem = getElementType(init);
  GPUMatmulShapeType problem(dimM, dimN, dimK, lhsElem, rhsElem, initElem);

  spirv::ResourceLimitsAttr limits = targetEnv.getResourceLimits();
  auto properties =
      limits.getCooperativeMatrixPropertiesKhr()
          .getAsRange<spirv::CooperativeMatrixPropertiesKHRAttr>();

  SmallVector<GPUMatmulShapeType> intrinsics;
  intrinsics.reserve(limits.getCooperativeMatrixPropertiesKhr().size());
  for (auto p : properties) {
    intrinsics.emplace_back(p.getMSize(), p.getNSize(), p.getKSize(),
                            p.getAType(), p.getBType(), p.getCType());
  }

  GPUMMAHeuristicSeeds seeds{numSubgroupsPerWorkgroup, numMNTilesPerSubgroup,
                             numTilesPerSubgroupDimK};

  std::optional<GPUMMASchedule> schedule =
      deduceMMASchedule(problem, intrinsics, seeds);
  if (!schedule)
    return failure();

  auto pipeline = CodeGenPipeline::SPIRVCooperativeMatrixVectorize;

  std::optional<int64_t> subgroupSize = limits.getSubgroupSize();
  // AMD RDNA architectures supports both wave32 and wave64 modes. Prefer to use
  // wave32 mode for better performance.
  if (targetEnv.getVendorID() == spirv::Vendor::AMD) {
    if (std::optional<int> minSize = limits.getMinSubgroupSize())
      subgroupSize = *minSize;
  }

  std::array<int64_t, 3> workgroupSize{schedule->nWarpCount * *subgroupSize,
                                       schedule->mWarpCount, 1};

  SmallVector<int64_t> vectorSizes(kIndex + 1, 0);
  if (isBM)
    vectorSizes[bIndex] = 1;
  vectorSizes[mIndex] = schedule->mSize;
  vectorSizes[nIndex] = schedule->nSize;
  vectorSizes[kIndex] = schedule->kSize;

  SmallVector<int64_t> subgroupTileSizes(lastParallelDim + 1, 0);
  if (isBM)
    subgroupTileSizes[bIndex] = 1;
  subgroupTileSizes[mIndex] = schedule->mTileCount * vectorSizes[mIndex];
  subgroupTileSizes[nIndex] = schedule->nTileCount * vectorSizes[nIndex];

  SmallVector<int64_t> workgroupTileSizes(lastParallelDim + 1, 0);
  if (isBM)
    workgroupTileSizes[bIndex] = 1;
  workgroupTileSizes[mIndex] = schedule->mWarpCount * subgroupTileSizes[mIndex];
  workgroupTileSizes[nIndex] = schedule->nWarpCount * subgroupTileSizes[nIndex];

  // Also create one level for reduction. This is needed because of
  // SPIRVTileAndPromotePass requires it.
  // TODO(#10499): Consolidate tiling configuration across different pipelines.
  SmallVector<int64_t> reductionTileSizes;
  reductionTileSizes.append(kIndex, 0);
  reductionTileSizes.push_back(schedule->kTileCount * schedule->kSize);

  TileSizesListType tileSizes;
  tileSizes.reserve(3);
  tileSizes.push_back(workgroupTileSizes);
  tileSizes.push_back(subgroupTileSizes);
  tileSizes.push_back(reductionTileSizes);
  tileSizes.push_back(vectorSizes);

  // Don't do multibuffering if the inner reduction loop is folded out.
  auto pipelineDepth = softwarePipelineDepth;
  auto storeStage = softwarePipelineStoreStage;
  if (schedule->kTileCount <= 1) {
    pipelineDepth = 0;
    storeStage = 0;
  }

  // Check if the C matrix will be promoted for computing shared memory usage.
  bool promoteC = needToPrmoteCForCooperativeMatrix(op);

  // Decrease pipeline depth until it fits in shared memory.
  const int maxBytes = limits.getMaxComputeSharedMemorySize();
  auto usedBytes =
      getTileBytes(workgroupTileSizes[mIndex], workgroupTileSizes[nIndex],
                   reductionTileSizes[kIndex],
                   IREE::Util::getTypeBitWidth(getElementType(lhs)), promoteC);

  while (pipelineDepth > 0 &&
         getMultiBufferMemoryUsage(usedBytes, pipelineDepth, storeStage) >
             maxBytes) {
    pipelineDepth--;
  }

  return setOpConfigAndEntryPointFnTranslation(
      op->getParentOfType<mlir::FunctionOpInterface>(), op, tileSizes, pipeline,
      workgroupSize, subgroupSize,
      getSoftwarePipeliningAttrDict(op->getContext(), pipelineDepth,
                                    storeStage));
}

} // namespace detail

//===----------------------------------------------------------------------===//
// FFT Default Configuration
//===----------------------------------------------------------------------===//

static LogicalResult setFftOpConfig(spirv::ResourceLimitsAttr limits,
                                    IREE::LinalgExt::FftOp op) {
  LLVM_DEBUG(llvm::dbgs() << "trying to deduce config as fft...\n");
  const int subgroupSize = limits.getSubgroupSize();
  auto pipeline = CodeGenPipeline::SPIRVBaseDistribute;

  std::array<int64_t, 3> workgroupSize = {subgroupSize, 1, 1};

  SmallVector<utils::IteratorType> loopIteratorTypes =
      op.getLoopIteratorTypes();
  unsigned loopDepth = loopIteratorTypes.size();
  SmallVector<int64_t> workgroupTileSize(loopDepth, 0);

  // Tiling along partitioned loops with size 1.
  for (auto [index, iteratorType] : llvm::enumerate(loopIteratorTypes)) {
    if (iteratorType == utils::IteratorType::parallel) {
      workgroupTileSize[index] = 1;
    }
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
      op->getParentOfType<mlir::FunctionOpInterface>(), op, tileSizes, pipeline,
      workgroupSize);
}

//===----------------------------------------------------------------------===//
// Winograd Default Configuration
//===----------------------------------------------------------------------===//

static LogicalResult setWinogradOpConfig(spirv::ResourceLimitsAttr limits,
                                         IREE::LinalgExt::LinalgExtOp op) {
  // Tiling is already done by tile and decompose, so we only set pipeline and
  // workgroup size. The tile sizes below are placeholders and were obtained
  // by manual tuning on the AMD Navi2 GPU on a small set of convolution
  // sizes found in the StableDiffusion model.
  auto pipeline = CodeGenPipeline::SPIRVWinogradVectorize;
  std::array<int64_t, 3> workgroupSize = {32, 4, 4};
  TileSizesListType tileSizes = {{1, 32}};
  return setOpConfigAndEntryPointFnTranslation(
      op->getParentOfType<mlir::FunctionOpInterface>(), op, tileSizes, pipeline,
      workgroupSize);
}

//===----------------------------------------------------------------------===//
// Reduction Default Configuration
//===----------------------------------------------------------------------===//

/// Set the configuration for reductions that can be mapped to warp reductions.
static LogicalResult setReductionConfig(const spirv::TargetEnv &targetEnv,
                                        linalg::LinalgOp op) {
  LLVM_DEBUG(llvm::dbgs() << "trying to deduce config as reduction...\n");

  // This pipeline eventually generates non-uniform group shuffle ops, which
  // requires special capability.
  if (!targetEnv.allows(spirv::Capability::GroupNonUniformShuffle))
    return failure();

  SmallVector<unsigned> parallelDims;
  SmallVector<unsigned> reductionDims;
  op.getParallelDims(parallelDims);
  op.getReductionDims(reductionDims);

  SmallVector<int64_t, 4> bounds = op.getStaticLoopRanges();
  int64_t numParallelDims = op.getNumParallelLoops();

  // We should have reduction dimensions.
  if (reductionDims.empty())
    return failure();

  // Make sure reduction dimensions are static and innermost ones.
  int64_t numDynamicReductionDims = 0;
  for (unsigned dim : reductionDims) {
    if (ShapedType::isDynamic(bounds[dim])) {
      numDynamicReductionDims++;
    }
    if (dim < numParallelDims) {
      LLVM_DEBUG(llvm::dbgs() << "failed: non-innermost reduction dims\n");
      return failure();
    }
  }

  // Distribution of multi-dim masked writes currently aren't fully supported.
  if (numDynamicReductionDims > 1) {
    return failure();
  }

  if (op.getRegionOutputArgs().size() != 1)
    return failure();

  // Only support projected permutation for now. This could be extended to
  // projected permutated with broadcast.
  if (llvm::any_of(op.getDpsInputOperands(), [&](OpOperand *input) {
        return !op.getMatchingIndexingMap(input).isProjectedPermutation();
      })) {
    return failure();
  }

  bool foundSingleReductionOutput = false;
  for (int64_t i = 0, e = op.getDpsInits().size(); i < e; i++) {
    // Only single combiner operations are supported for now.
    SmallVector<Operation *> combinerOps;
    if (matchReduction(op.getRegionOutputArgs(), i, combinerOps) &&
        combinerOps.size() == 1) {
      if (foundSingleReductionOutput)
        return failure();
      foundSingleReductionOutput = true;
      continue;
    }
    if (!op.getMatchingIndexingMap(op.getDpsInitOperand(i)).isIdentity()) {
      return failure();
    }
  }
  if (!foundSingleReductionOutput)
    return failure();

  const int subgroupSize = targetEnv.getResourceLimits().getSubgroupSize();

  // Tile all the parallel dimension to 1.
  SmallVector<unsigned> partitionedLoops =
      cast<PartitionableLoopsInterface>(op.getOperation())
          .getPartitionableLoops(kNumMaxParallelDims);
  llvm::SmallDenseSet<unsigned, 4> partitionedLoopsSet;
  partitionedLoopsSet.insert(partitionedLoops.begin(), partitionedLoops.end());
  size_t numLoops = partitionedLoops.empty() ? 0 : partitionedLoops.back() + 1;
  SmallVector<int64_t> workgroupTileSizes(numLoops, 1);

  // Without any bounds on dynamic reduction dims, we need specialization to
  // get peak performance. For now, just use the subgroup size.
  if (numDynamicReductionDims) {
    SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 0);
    reductionTileSizes[reductionDims[0]] = subgroupSize;
    TileSizesListType tileSizes;
    tileSizes.emplace_back(std::move(workgroupTileSizes)); // Workgroup level
    tileSizes.emplace_back(std::move(reductionTileSizes)); // Reduction level
    std::array<int64_t, 3> workgroupSize = {subgroupSize, 1, 1};
    if (failed(setOpConfigAndEntryPointFnTranslation(
            op->getParentOfType<mlir::FunctionOpInterface>(), op, tileSizes,
            CodeGenPipeline::SPIRVSubgroupReduce, workgroupSize))) {
      return failure();
    }

    // Set lowering configuration to drive tiling for other Linalg ops too---the
    // pipeline expects it.
    op->getParentOfType<FunctionOpInterface>().walk([&](linalg::LinalgOp op) {
      setLoweringConfig(op, IREE::Codegen::LoweringConfigAttr::get(
                                op.getContext(), tileSizes));
    });
    return success();
  }

  int64_t reductionSize = 1;
  for (int64_t dim : reductionDims)
    reductionSize *= bounds[dim];
  if (reductionSize % subgroupSize != 0)
    return failure();

  const Type elementType =
      llvm::cast<ShapedType>(op.getDpsInits()[0].getType()).getElementType();
  if (!elementType.isIntOrFloat())
    return failure();
  unsigned bitWidth = IREE::Util::getTypeBitWidth(elementType);
  // Reduction distribution only supports 8/16/32 bit types now.
  if (bitWidth != 32 && bitWidth != 16 && bitWidth != 8)
    return failure();

  // Let each thread handle `vectorSize` elements.
  unsigned vectorSize = kMaxVectorNumBits / bitWidth;
  while ((reductionSize / vectorSize) % subgroupSize != 0)
    vectorSize /= 2;

  // Deduce the workgroup size we should use for reduction. Currently a
  // workgroup processes all elements in reduction dimensions. Need to make sure
  // the workgroup size we use can divide the total reduction size, and it's
  // also within hardware limitations.
  const int64_t maxWorkgroupSize =
      targetEnv.getResourceLimits().getMaxComputeWorkgroupInvocations();
  int64_t groupSize = reductionSize / vectorSize;
  if (groupSize > maxWorkgroupSize) {
    groupSize = GreatestCommonDivisor(APInt(64, uint64_t(groupSize)),
                                      APInt(64, uint64_t(maxWorkgroupSize)))
                    .getZExtValue();
  }

  // Then we need to strike a balance--
  // 1) parallel dimensions are distributed to workgroups. If there are many
  //    workgroups dispatched, we'd want to have each GPU core hosting multiple
  //    of them for occupancy.
  // 2) we want each thread to read quite a few 128-bit vectors for better
  //    memory cache behavior.
  // Both means we cannot use a too large workgroup size.

  int64_t parallelSize = 1;
  for (int64_t dim : parallelDims) {
    if (!ShapedType::isDynamic(bounds[dim]))
      parallelSize *= bounds[dim];
  }
  // Total parallel size that can fill the GPU with enough workgorups.
  // TODO: query from the target device; roughly 2x hardware compute unit.
  int parallelThreshold = 256;
  // How many 128-bit vectors each thread should at least read.
  const int targetVectorCount = 8;
  while (parallelSize > parallelThreshold &&
         (groupSize / 2) % subgroupSize == 0 &&
         reductionSize / (groupSize * vectorSize) < targetVectorCount) {
    // Use less subgroups per workgroup..
    groupSize /= 2;
    // in order to host more workgroups per hardware compute unit.
    parallelSize /= 2;
  }

  // Current warp reduction pattern is a two step butterfly warp reduce.
  // First, do warp reductions along multiple subgroups.
  // Second, reduce results from multiple subgroups using single warp reduce.
  // The final warp reduce requires subgroup count <= subgroup size to work.
  if ((groupSize / subgroupSize) > subgroupSize)
    return failure();

  std::array<int64_t, 3> workgroupSize = {groupSize, 1, 1};

  SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 0);
  int64_t remaingGroupSize = groupSize;
  for (int i = reductionDims.size() - 1; i >= 0; --i) {
    int64_t dim = reductionDims[i];
    int64_t bound = bounds[dim];
    if (i == reductionDims.size() - 1)
      bound /= vectorSize;
    APInt size = GreatestCommonDivisor(APInt(64, uint64_t(remaingGroupSize)),
                                       APInt(64, uint64_t(bound)));
    reductionTileSizes[dim] = size.getSExtValue();
    if (i == reductionDims.size() - 1)
      reductionTileSizes[dim] *= vectorSize;
    remaingGroupSize /= size.getSExtValue();
  }

  TileSizesListType tileSizes;
  tileSizes.emplace_back(std::move(workgroupTileSizes)); // Workgroup level
  tileSizes.emplace_back(std::move(reductionTileSizes)); // reduction level
  if (failed(setOpConfigAndEntryPointFnTranslation(
          op->getParentOfType<mlir::FunctionOpInterface>(), op, tileSizes,
          CodeGenPipeline::SPIRVSubgroupReduce, workgroupSize))) {
    return failure();
  }

  // Set lowering configuration to drive tiling for other Linalg ops too---the
  // pipeline expects it.
  op->getParentOfType<FunctionOpInterface>().walk([&](linalg::LinalgOp op) {
    setLoweringConfig(
        op, IREE::Codegen::LoweringConfigAttr::get(op.getContext(), tileSizes));
  });
  return success();
}

//===----------------------------------------------------------------------===//
// Everything Default Configuration
//===----------------------------------------------------------------------===//

/// Returns a small tiling factor for the given reduction `dimSize`.
/// Returns 0 to avoid tiling.
static int getReductionTilingFactor(int64_t dimSize) {
  if (dimSize % 4 == 0)
    return 4;

  // Try to find the smallest prime factor as the tiling factor. As a trade off
  // between generated code size and compilation time, only look at prime
  // numbers less than 50 right now.
  static constexpr std::array<int, 15> primeNumbers = {
      2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
  for (int n : primeNumbers) {
    if (dimSize % n == 0)
      return n;
  }

  return 1; // Otherwise just tile with size 1.
}

/// Returns the minimal element bitwidth used in the operands and results of the
/// given Linalg op.
static int64_t getMinElementBitwidth(linalg::LinalgOp linalgOp) {
  unsigned bitwidth = std::numeric_limits<unsigned>::max();
  for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
    unsigned b =
        IREE::Util::getTypeBitWidth(getElementTypeOrSelf(operand->get()));
    bitwidth = std::min(bitwidth, b);
  }
  for (Value result : linalgOp.getDpsInits()) {
    unsigned b = IREE::Util::getTypeBitWidth(getElementTypeOrSelf(result));
    bitwidth = std::min(bitwidth, b);
  }
  return bitwidth;
};

static LogicalResult setDefaultOpConfig(spirv::ResourceLimitsAttr limits,
                                        Operation *op,
                                        bool allowVectorization = true) {
  LLVM_DEBUG(llvm::dbgs() << "trying to deduce as default op...\n");
  auto funcOp = op->getParentOfType<mlir::FunctionOpInterface>();
  auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);

  // Special case for not tiled ops.
  if (partitionedLoops.empty()) {
    // No tiled loops means we cannot tile (and distribute) at all. Use just one
    // single thread to run everything.
    auto pipeline = CodeGenPipeline::SPIRVBaseDistribute;
    std::array<int64_t, 3> workgroupSize = {1, 1, 1};
    return setOpConfigAndEntryPointFnTranslation(funcOp, op, {}, pipeline,
                                                 workgroupSize);
  }

  const int subgroupSize = limits.getSubgroupSize();
  const unsigned loopDepth = partitionedLoops.back() + 1;

  // Configurations we need to decide.
  std::array<int64_t, 3> workgroupSize;
  SmallVector<int64_t> workgroupTileSizes;
  SmallVector<int64_t> threadTileSizes;

  // Initialize the configuration.
  auto initConfiguration = [&]() {
    workgroupSize = {subgroupSize, 1, 1};
    workgroupTileSizes.resize(loopDepth, 0);
    threadTileSizes.resize(loopDepth, 0);

    // Initialize tiling along all partitioned loops with size 1.
    for (int64_t loopIndex : partitionedLoops) {
      workgroupTileSizes[loopIndex] = threadTileSizes[loopIndex] = 1;
    }
    // Override the innermost dimension to distribute to threads in a subgroup.
    workgroupTileSizes.back() = subgroupSize;
    threadTileSizes.back() = 1;
  };

  // Special case for non-linalg ops.
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || linalgOp.getNumDpsInits() != 1) {
    auto pipeline = CodeGenPipeline::SPIRVBaseDistribute;

    initConfiguration();
    TileSizesListType tileSizes;
    tileSizes.push_back(workgroupTileSizes);
    tileSizes.push_back(threadTileSizes);

    return setOpConfigAndEntryPointFnTranslation(funcOp, op, tileSizes,
                                                 pipeline, workgroupSize);
  }

  // Common case for all linalg ops.

  // The core idea is to distribute the partitioned loops to the workgroup
  // dimensions. The goal is to fill up the GPU as much as possible, which means
  // 1) distributing to as many threads as possible, and 2) avoid assigning too
  // many threads to handle out-of-bound elements (thus idle).

  auto elementHasPowerOfTwoBitwidth = [](Value operand) {
    Type elementType = getElementTypeOrSelf(operand.getType());
    return isa<IntegerType, FloatType>(elementType) &&
           llvm::isPowerOf2_64(IREE::Util::getTypeBitWidth(elementType));
  };

  // Whether we can try to use the vectorization pipeline.
  SmallVector<int64_t> loopBounds = linalgOp.getStaticLoopRanges();
  bool vectorizable =
      allowVectorization &&
      // The vectorization pipeline assumes tensor semantics for tiling.
      linalgOp.hasPureTensorSemantics() && !linalgOp.hasIndexSemantics() &&
      // Require all affine maps to be projected permutation so that we can
      // generate vector transfer ops.
      llvm::all_of(
          linalgOp.getIndexingMapsArray(),
          [](AffineMap map) { return map.isProjectedPermutation(); }) &&
      llvm::all_of(linalgOp->getOperands(), elementHasPowerOfTwoBitwidth) &&
      llvm::none_of(loopBounds, ShapedType::isDynamic);

  const unsigned minBitwidth = getMinElementBitwidth(linalgOp);
  // Make sure we use a tile size that results in some integral number of bytes.
  const unsigned scaleToByte = minBitwidth < 8 ? 8 / minBitwidth : 1;

  // Distribute workload to the given `numThreads` by allowing a potental loss.
  auto distributeToThreads = [&](int64_t numThreads,
                                 std::optional<int64_t> lossFactor =
                                     std::nullopt) {
    LLVM_DEBUG(llvm::dbgs() << "\nLoss factor: " << lossFactor << "\n");
    initConfiguration();
    // If there are more than 3 parallel dim try to tile the extra higher level
    // dimensions to 1 for extra dimensions.
    if (isa<linalg::GenericOp>(linalgOp.getOperation())) {
      for (int64_t i = 0, e = workgroupTileSizes.size(); i < e; i++) {
        if (workgroupTileSizes[i] != 0)
          break;
        if (loopBounds[i] != 1)
          workgroupTileSizes[i] = 1;
      }
    }
    // Scan from the innermost shape dimension and try to deduce the
    // configuration for the corresponding GPU workgroup dimension.
    int64_t wgDim = 0;
    for (auto shapeDim : llvm::reverse(partitionedLoops)) {
      int64_t loopBound = loopBounds[shapeDim];
      // Skip dynamic dimensions.
      if (ShapedType::isDynamic(loopBound))
        continue;

      // Try to find some power of two that can devide the current shape dim
      // size. This vector keeps the candidate tile sizes.
      SmallVector<int64_t, 8> candidates;

      // For the inner most workgroup dim, try to see if we can have 4
      // elements per thread. This enables vectorization.
      if (vectorizable && wgDim == 0 && !lossFactor) {
        candidates.push_back(4 * numThreads);
      }
      // Try all power of two numbers upto the subgroup size.
      for (unsigned i = numThreads; i >= 1; i >>= 1) {
        candidates.push_back(i);
      }
      LLVM_DEBUG({
        llvm::dbgs() << "Base candidate tile sizes: [";
        llvm::interleaveComma(candidates, llvm::dbgs());
        llvm::dbgs() << "]\n";
      });

      for (int64_t candidate : candidates) {
        int64_t scaledTileSize = candidate * scaleToByte;
        if (loopBound % scaledTileSize != 0) {
          if (!lossFactor)
            continue;
          // Skip this candidate if it causes many threads to be idle.
          int64_t idleThreads = candidate - (loopBound % scaledTileSize);
          if (idleThreads > candidate / *lossFactor)
            continue;
        }
        // If the workload is too small and we cannot distribute to more than 2
        // workgroups, try a smaller tile size to increase parallelism.
        if (partitionedLoops.size() == 1 && candidate > subgroupSize &&
            divideCeil(loopBound, scaledTileSize) <= 2) {
          continue;
        }

        // Found a suitable candidate. Try to let each thread handle 4
        // elements if this is the workgroup x dimension.
        workgroupTileSizes[shapeDim] = scaledTileSize;
        LLVM_DEBUG(llvm::dbgs()
                   << "Chosen workgroup tile size: " << scaledTileSize << "\n");
        if (vectorizable && wgDim == 0 && !lossFactor && candidate % 4 == 0) {
          // Use size-1 vectors to increase parallelism if larger ones causes
          // idle threads in the subgroup.
          bool hasIdleThreads =
              partitionedLoops.size() == 1 && candidate <= subgroupSize;
          int vectorSize = hasIdleThreads ? 1 : 4;
          LLVM_DEBUG(llvm::dbgs() << "Use vector size: " << vectorSize << "\n");
          threadTileSizes[shapeDim] = vectorSize * scaleToByte;
          workgroupSize[wgDim] = candidate / vectorSize;
          assert(numThreads % (candidate / vectorSize) == 0);
          numThreads /= candidate / vectorSize;
        } else {
          if (wgDim == 0)
            vectorizable = false;
          threadTileSizes[shapeDim] = scaleToByte;
          workgroupSize[wgDim] = candidate;
          assert(numThreads % candidate == 0);
          numThreads /= candidate;
        }
        assert(numThreads >= 1);
        break;
      }

      // Stop if we have distributed all threads.
      if (numThreads == 1)
        break;
      wgDim++;
    }
    return numThreads;
  };

  // First try to see if we can use up all threads without any loss.
  if (distributeToThreads(subgroupSize) != 1) {
    // Otherwise, allow larger and larger loss factor.

    // Threads for distribution. Use 32 at least.
    int64_t numThreads = std::max(subgroupSize, 32);
    // We can tolerate (1 / lossFactor) of threads in the workgroup to be idle.
    int64_t lossFactor = 32;

    for (; lossFactor >= 1; lossFactor >>= 1) {
      if (distributeToThreads(numThreads, lossFactor) == 1)
        break;
    }
  }

  auto pipeline = vectorizable ? CodeGenPipeline::SPIRVBaseVectorize
                               : CodeGenPipeline::SPIRVBaseDistribute;

  TileSizesListType tileSizes;
  tileSizes.push_back(workgroupTileSizes);
  tileSizes.push_back(threadTileSizes);

  if (vectorizable) {
    // Try to tile all reductions by some small factor, preferrably 4, when
    // possible. This gives us a chance to perform vector4 load if an input has
    // its innnermost dimension being reduction. It also avoids generating too
    // many instructions when unrolling vector later.
    SmallVector<int64_t> loopTileSizes(linalgOp.getNumLoops(), 0);
    for (const auto &[i, iter] :
         llvm::enumerate(linalgOp.getIteratorTypesArray())) {
      if (linalg::isReductionIterator(iter) || i >= workgroupTileSizes.size() ||
          workgroupTileSizes[i] == 0) {
        loopTileSizes[i] = getReductionTilingFactor(loopBounds[i]);
      }
    }
    if (llvm::any_of(loopTileSizes, [](int64_t s) { return s != 0; })) {
      tileSizes.push_back(loopTileSizes);
    }
  }

  return setOpConfigAndEntryPointFnTranslation(funcOp, op, tileSizes, pipeline,
                                               workgroupSize);
}

//===----------------------------------------------------------------------===//
// Transform Dialect Specialized Configurations
//===----------------------------------------------------------------------===//

static LogicalResult
setTransformDialectConfig(mlir::FunctionOpInterface entryPoint, Operation *op,
                          const spirv::TargetEnv &targetEnv) {
  if (!clSPIRVEnableTransformDialectJit) {
    return failure();
  }

  MLIRContext *context = entryPoint.getContext();
  auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      context, CodeGenPipeline::TransformDialectCodegen);

  spirv::ResourceLimitsAttr limits = targetEnv.getResourceLimits();

  // TODO: unify the target information into one structure.
  iree_compiler::gpu::GPUModel gpuModel;
  gpuModel.hasWarpShuffle =
      targetEnv.allows(spirv::Capability::GroupNonUniformShuffle);
  gpuModel.hasTF32TensorCore = false;
  gpuModel.hasMmaSync = false;
  gpuModel.hasTF32TensorCore = false;
  gpuModel.minSubgroupSize = limits.getMinSubgroupSize();
  gpuModel.maxSubgroupSize = limits.getMaxSubgroupSize();
  gpuModel.maxWorkGroupInvocations = limits.getMaxComputeWorkgroupInvocations();

  // Populates the supported WMMA fragment combinations from the target
  // environment. Infer tf32 support from the list of supported fragment types.
  auto properties =
      limits.getCooperativeMatrixPropertiesKhr()
          .getAsRange<spirv::CooperativeMatrixPropertiesKHRAttr>();
  for (auto property : properties) {
    if (property.getScope().getValue() != spirv::Scope::Subgroup)
      continue;
    gpuModel.supportedWMMAConfigs.emplace_back(iree_compiler::gpu::MMAConfig{
        property.getMSize(), property.getNSize(), property.getKSize(),
        property.getAType(), property.getBType(), property.getCType()});
  }

  if (failed(iree_compiler::gpu::matchAndSetTransformStrategy(entryPoint, op,
                                                              gpuModel)))
    return failure();
  return setTranslationInfo(entryPoint, translationInfo);
}

//===----------------------------------------------------------------------===//
// Configuration Dispatcher
//===----------------------------------------------------------------------===//

/// Sets the CodeGen configuration as attributes to the given `rootOp` if it's a
/// known Linalg matmul/convolution op with good configurations.
static LogicalResult setSPIRVOpConfig(const spirv::TargetEnv &targetEnv,
                                      mlir::FunctionOpInterface entryPointFn,
                                      Operation *rootOp) {
  // First try to see if there is a matching transform dialect configuration.
  if (succeeded(setTransformDialectConfig(entryPointFn, rootOp, targetEnv))) {
    return success();
  }

  // First try to find a proper CodeGen configuration to tile and vectorize for
  // the current target architecture.
  switch (targetEnv.getVendorID()) {
  case spirv::Vendor::AMD:
    if (succeeded(detail::setAMDCodeGenConfig(targetEnv, rootOp)))
      return success();
    break;
  case spirv::Vendor::Apple:
    if (succeeded(detail::setAppleCodeGenConfig(targetEnv, rootOp)))
      return success();
    break;
  case spirv::Vendor::ARM:
    if (succeeded(detail::setMaliCodeGenConfig(targetEnv, rootOp)))
      return success();
    break;
  case spirv::Vendor::NVIDIA:
    if (succeeded(detail::setNVIDIACodeGenConfig(targetEnv, rootOp)))
      return success();
    break;
  case spirv::Vendor::Qualcomm:
    if (succeeded(detail::setAdrenoCodeGenConfig(targetEnv, rootOp)))
      return success();
    break;
  default:
    break;
  }

  // Otherwise fallback to use a default configuration that tiles and
  // distributes/vectorizes.
  spirv::ResourceLimitsAttr limits = targetEnv.getResourceLimits();
  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::BatchMatmulOp, linalg::MatmulOp>([&](auto op) {
        // Try to tile and vectorize first. It's common to see 32 threads
        // per subgroup for GPUs.
        std::array<int64_t, 2> workgroupXY = {32, 2};
        std::array<int64_t, 3> threadMNK;
        auto inputType = llvm::cast<ShapedType>(op.getInputs()[0].getType());
        if (IREE::Util::getTypeBitWidth(inputType.getElementType()) == 16) {
          threadMNK = {8, 8, 8};
        } else {
          threadMNK = {8, 8, 4};
        }
        auto result =
            detail::setMatmulOpConfig(limits, op, workgroupXY, threadMNK);
        if (succeeded(result))
          return success();

        LLVM_DEBUG(llvm::dbgs()
                   << "failed to set matmul op config, trying reduction\n");
        if (succeeded(setReductionConfig(targetEnv, op)))
          return success();

        // If unsuccessful, try to tile and distribute.
        return setDefaultOpConfig(limits, op);
      })
      .Case<linalg::ConvolutionOpInterface>([limits](auto op) {
        // Use the result type in case of larger bitwidth for accumulators.
        auto type = cast<ShapedType>(op->getResult(0).getType());
        const int bitwidth = type.getElementTypeBitWidth();
        if (bitwidth <= 32) {
          const int multipler = 32 / bitwidth;
          const int bestTilingFactor = 32 * multipler;
          const int subgroupSize = 32;
          auto result = detail::setConvOpConfig(cast<linalg::LinalgOp>(*op),
                                                subgroupSize, bestTilingFactor);
          if (succeeded(result))
            return success();
        }

        // If unsuccessful, try to tile and distribute/vectorize.
        return setDefaultOpConfig(limits, op);
      })
      .Case<linalg::GenericOp>([&](linalg::GenericOp op) {
        LLVM_DEBUG(llvm::dbgs() << "figuring configuration for generic op\n");
        if (succeeded(setReductionConfig(targetEnv, op)))
          return success();

        // If a generic op has reduction iterator types, it can be treated as a
        // root op for configuration as well. Use the default configuration,
        // which will mark it as a root.
        if (op.getNumLoops() != op.getNumParallelLoops()) {
          return setDefaultOpConfig(limits, op);
        }
        return failure();
      })
      .Case<IREE::LinalgExt::FftOp>([limits](IREE::LinalgExt::FftOp op) {
        return setFftOpConfig(limits, op);
      })
      .Case<IREE::LinalgExt::WinogradInputTransformOp,
            IREE::LinalgExt::WinogradOutputTransformOp>(
          [&](auto op) { return setWinogradOpConfig(limits, op); })
      .Default([](Operation *) { return failure(); });
};

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

static LogicalResult setConfigForKernel(const spirv::TargetEnv &targetEnv,
                                        IREE::HAL::ExecutableExportOp exportOp,
                                        mlir::FunctionOpInterface funcOp) {
  // First check whether we already have workgroup count set--it's a "contract"
  // to indicate that we should bypass all tiling and distribution to go down
  // just the most basic lowering flow.
  if (Block *body = exportOp.getWorkgroupCountBody()) {
    auto retOp = cast<IREE::HAL::ReturnOp>(body->getTerminator());
    // For scalar dispatch cases--using just one thread of one workgroup.
    auto isOne = [](Value value) { return matchPattern(value, m_One()); };
    if (llvm::all_of(retOp.getOperands(), isOne)) {
      std::array<int64_t, 3> workgroupSize = {1, 1, 1};
      if (failed(setDispatchConfig(funcOp, workgroupSize, std::nullopt)))
        return failure();
      auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
          funcOp.getContext(), CodeGenPipeline::SPIRVBaseLowering);
      return setTranslationInfo(funcOp, translationInfo);
    }
  }

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  if (computeOps.empty()) {
    // No compute operations found. Allow to pass through without a config.
    return success();
  }

  // Try to find a configuration according to a matmul/convolution op, which as
  // at least one reduction dimension, and use it as the root op. So, skip all
  // fused parallel producer ops.
  ArrayRef roots(computeOps);
  while (roots.size() > 1) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(roots.front());
    if (!linalgOp)
      break;
    if (linalgOp.getNumParallelLoops() != linalgOp.getNumLoops())
      break;
    roots = roots.drop_front();
  }

  for (Operation *computeOp : roots) {
    if (succeeded(setSPIRVOpConfig(targetEnv, funcOp, computeOp)))
      return success();
  }

  Operation *computeOp = roots.back();
  spirv::ResourceLimitsAttr limits = targetEnv.getResourceLimits();
  // If there are still no root op, check for any linalg.generic op.
  if (succeeded(setDefaultOpConfig(limits, computeOp)))
    return success();

  // Check if the op configuration was set.
  return computeOp->emitOpError(
      "without known roots, the last compute operation in the tiled "
      "loop body is expected to be set as root");
}

LogicalResult initSPIRVLaunchConfig(ModuleOp module) {
  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(module);
  spirv::TargetEnvAttr targetEnvAttr = getSPIRVTargetEnvAttr(module);
  if (!targetEnvAttr) {
    return module.emitOpError(
        "expected parent hal.executable.variant to have spirv.target_env "
        "attribute");
  }
  spirv::TargetEnv targetEnv(targetEnvAttr);

  for (auto funcOp : module.getOps<mlir::FunctionOpInterface>()) {
    auto exportOp = exportOps.lookup(funcOp.getName());
    if (!exportOp)
      continue;
    if (getTranslationInfo(exportOp))
      continue;

    if (failed(setConfigForKernel(targetEnv, exportOp, funcOp))) {
      return failure();
    }
  }

  return success();
}

} // namespace mlir::iree_compiler
