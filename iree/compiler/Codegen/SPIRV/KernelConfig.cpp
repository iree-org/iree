// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"

#define DEBUG_TYPE "iree-spirv-kernel-config"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Convolution Default Configuration
//===----------------------------------------------------------------------===//

namespace detail {

LogicalResult setConvOpConfig(linalg::LinalgOp linalgOp,
                              const int64_t subgroupSize,
                              const int64_t bestTilingFactor) {
  ArrayRef<int64_t> inputShape = getUntiledShape(linalgOp.inputs()[0]);
  SmallVector<int64_t> outputShape = getUntiledResultShape(linalgOp, 0);
  if (llvm::any_of(inputShape, ShapedType::isDynamic)) return success();
  if (llvm::any_of(outputShape, ShapedType::isDynamic)) return success();

  int64_t ic = inputShape[3];
  int64_t oh = outputShape[1], ow = outputShape[2], oc = outputShape[3];

  // The conversion pipeline requires the input channel dimension to be some
  // multipler of four, or less than four.
  if (!(ic % 4 == 0 || ic < 4)) return success();

  // The core idea is to distribute the convolution OH/OW/OC dimension to the
  // workgroup Z/Y/X dimension, with each thread in a workgroup handling
  // multiple vector elements. We try to 1) utilize all threads in a subgroup,
  // and 2) handle an optimal tile size along each dimension.

  int64_t residualThreads = subgroupSize;
  int64_t residualTilingFactor = bestTilingFactor;

  SmallVector<int64_t, 3> workgroupSize(3, 1);     // (X, Y, Z)
  SmallVector<int64_t> workgroupTileSizes(4, 0);   // (N, OH, OW, OC)
  SmallVector<int64_t> invocationTileSizes(4, 0);  // (N, OH, OW, OC)

  // Deduce the configuration for the OC dimension.
  for (int64_t x = residualThreads; x >= 2; x >>= 1) {
    // Handle 4 elements per thread for the innermost dimension. We need this
    // for vectorized load.
    int64_t chosenTileSize = 4;
    if (oc % (x * chosenTileSize) == 0) {
      workgroupSize[0] = x;
      workgroupTileSizes[3] = x * chosenTileSize;
      invocationTileSizes[3] = chosenTileSize;
      residualThreads /= x;
      residualTilingFactor /= chosenTileSize;
      break;
    }
  }
  if (workgroupTileSizes[3] == 0) return success();

  // Deduce the configruation for the OW and OH dimension. Try to make them even
  // if possible given we typically have images with the same height and width.
  bool tileToSquare = false;
  unsigned log2Threads = llvm::Log2_64(residualThreads);
  if (ow == oh && residualThreads != 1 && log2Threads % 2 == 0) {
    int64_t yz = 1ll << (log2Threads / 2);

    int64_t chosenTileSize = 1ll << (llvm::Log2_64(residualTilingFactor) / 2);
    while (chosenTileSize >= 1 && ow % (yz * chosenTileSize) != 0) {
      chosenTileSize >>= 1;
    }

    if (chosenTileSize != 0) {
      workgroupSize[1] = workgroupSize[2] = yz;
      workgroupTileSizes[2] = workgroupTileSizes[1] = yz * chosenTileSize;
      invocationTileSizes[2] = invocationTileSizes[1] = chosenTileSize;
      tileToSquare = true;
    }
  }

  // Otherwise treat OW and OH separately to allow them to have different number
  // of threads and tiling size.
  if (!tileToSquare) {
    // Decide the tiling and distribution parameters for one dimension.
    auto decideOneDim = [&](int64_t inputDim, int64_t &wgDimSize,
                            int64_t &wgTileSize, int64_t &invoTileSize) {
      for (int64_t dim = residualThreads; dim >= 1; dim >>= 1) {
        int64_t chosenTileSize = 0;
        for (int64_t t = residualTilingFactor; t >= 1; t >>= 1) {
          if (inputDim % (dim * t) == 0) {
            chosenTileSize = t;
            break;
          }
        }
        if (chosenTileSize) {
          wgDimSize = dim;
          wgTileSize = dim * chosenTileSize;
          invoTileSize = chosenTileSize;
          residualThreads /= dim;
          residualTilingFactor /= chosenTileSize;
          return true;
        }
      }
      return false;
    };

    if (!decideOneDim(ow, workgroupSize[1], workgroupTileSizes[2],
                      invocationTileSizes[2]) ||
        !decideOneDim(oh, workgroupSize[2], workgroupTileSizes[1],
                      invocationTileSizes[1])) {
      return success();
    }
  }

  auto pipeline = IREE::Codegen::DispatchLoweringPassPipeline::SPIRVVectorize;
  TileSizesListType tileSizes;
  tileSizes.push_back(workgroupTileSizes);
  tileSizes.push_back(invocationTileSizes);
  // Tiling along reduction dimensions
  if (isa<linalg::Conv2DNhwcHwcfOp>(linalgOp)) {
    tileSizes.push_back({0, 0, 0, 0, 1, 1, 4});
  } else if (isa<linalg::DepthwiseConv2DNhwcHwcOp>(linalgOp)) {
    tileSizes.push_back({0, 0, 0, 0, 1, 1});
  } else {
    return success();
  }

  auto funcOp = linalgOp->getParentOfType<FuncOp>();
  return setOpConfigAndEntryPointFnTranslation(funcOp, linalgOp, tileSizes, {},
                                               pipeline, workgroupSize);
}

}  // namespace detail

//===----------------------------------------------------------------------===//
// Matmul Default Configuration
//===----------------------------------------------------------------------===//

namespace detail {

LogicalResult setMatmulOpConfig(linalg::LinalgOp op,
                                std::array<int64_t, 2> bestWorkgroupSizeXY,
                                std::array<int64_t, 3> bestThreadTileSizeMNK) {
  auto lhsType = op.inputs()[0].getType().cast<ShapedType>();
  auto elementBits = lhsType.getElementType().getIntOrFloatBitWidth();
  if (elementBits != 16 && elementBits != 32) return success();

  ArrayRef<int64_t> lhsShape = getUntiledShape(op.inputs()[0]);
  ArrayRef<int64_t> rhsShape = getUntiledShape(op.inputs()[1]);
  if (llvm::any_of(lhsShape, ShapedType::isDynamic)) return success();
  if (llvm::any_of(rhsShape, ShapedType::isDynamic)) return success();

  bool isBM = isa<linalg::BatchMatmulOp>(op);

  int64_t dimM = lhsShape[0 + isBM];
  int64_t dimK = lhsShape[1 + isBM];
  int64_t dimN = rhsShape[1 + isBM];

  // The core idea is to distribute the matmul M/N dimension to the workgroup
  // Y/X dimension, with each thread in a workgroup handling multiple vector
  // elements. We start from the best (X, Y) and the tiling sizes for (M, N, K)
  // and try different configurations by scaling them down until we find a
  // configuration that can perfectly tile the input matmul.

  const int64_t bestX = bestWorkgroupSizeXY[0], bestY = bestWorkgroupSizeXY[1];
  const int64_t bestThreadM = bestThreadTileSizeMNK[0],
                bestThreadN = bestThreadTileSizeMNK[1],
                bestThreadK = bestThreadTileSizeMNK[2];

  int64_t residualThreads = bestX * bestY;
  int64_t residualTilingFactor = (bestThreadM + bestThreadK) * bestThreadN;

  SmallVector<int64_t, 3> workgroupSize(3, 1);            // (X, Y, Z)
  SmallVector<int64_t> workgroupTileSizes(2 + isBM, 0);   // ([B,] M, N)
  SmallVector<int64_t> invocationTileSizes(2 + isBM, 0);  // ([B,] M, N)
  SmallVector<int64_t> reductionTileSizes(3 + isBM, 0);   // ([B,] M, N, K)

  if (isBM) workgroupTileSizes[0] = invocationTileSizes[0] = 1;

  // Deduce the configuration for the N dimension. Start with the best workgroup
  // X size, and reduce by a factor of two each time.
  for (int64_t x = bestX; x >= 2; x >>= 1) {
    // Handle 4 elements per thread for the innermost dimension. We need this
    // for vectorized load.
    int64_t chosenTileSize = bestThreadN;
    if (dimN % (x * chosenTileSize) == 0) {
      workgroupSize[0] = x;
      workgroupTileSizes[1 + isBM] = x * chosenTileSize;
      invocationTileSizes[1 + isBM] = chosenTileSize;
      residualThreads /= x;
      assert(residualTilingFactor % chosenTileSize == 0);
      residualTilingFactor /= chosenTileSize;
      break;
    }
  }
  if (workgroupTileSizes[1 + isBM] == 0) return success();

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
      workgroupSize[1] = y;
      workgroupTileSizes[0 + isBM] = y * chosenTileSize;
      invocationTileSizes[0 + isBM] = chosenTileSize;
      assert(residualTilingFactor > chosenTileSize);
      residualTilingFactor -= chosenTileSize;
      break;
    }
  }
  if (workgroupTileSizes[0 + isBM] == 0) return success();

  // Deduce the configuration for the K dimension. We need some power of two
  // here so that we can do vector load.
  for (int64_t t = llvm::PowerOf2Floor(residualTilingFactor); t >= 2; t >>= 1) {
    if (dimK % t == 0) {
      reductionTileSizes[2 + isBM] = t;
      break;
    }
  }
  if (reductionTileSizes[2 + isBM] == 0) return success();

  auto pipeline = IREE::Codegen::DispatchLoweringPassPipeline::SPIRVVectorize;
  TileSizesListType tileSizes;
  tileSizes.push_back(workgroupTileSizes);
  tileSizes.push_back(invocationTileSizes);
  tileSizes.push_back(reductionTileSizes);
  return setOpConfigAndEntryPointFnTranslation(op->getParentOfType<FuncOp>(),
                                               op, tileSizes, {}, pipeline,
                                               workgroupSize);
}

}  // namespace detail

//===----------------------------------------------------------------------===//
// FFT Default Configuration
//===----------------------------------------------------------------------===//

static LogicalResult setFftOpConfig(spirv::ResourceLimitsAttr limits,
                                    IREE::LinalgExt::FftOp op) {
  const int64_t subgroupSize = limits.subgroup_size().getValue().getSExtValue();
  auto pipeline = IREE::Codegen::DispatchLoweringPassPipeline::SPIRVDistribute;

  std::array<int64_t, 3> workgroupSize = {subgroupSize, 1, 1};

  auto interfaceOp = cast<IREE::Flow::PartitionableLoopsInterface>(*op);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);

  unsigned loopDepth = partitionedLoops.back() + 1;
  SmallVector<int64_t> workgroupTileSize(loopDepth, 0);

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
  return setOpConfigAndEntryPointFnTranslation(op->getParentOfType<FuncOp>(),
                                               op, tileSizes, {}, pipeline,
                                               workgroupSize);
}

//===----------------------------------------------------------------------===//
// Everything Default Configuration
//===----------------------------------------------------------------------===//

static LogicalResult setDefaultOpConfig(spirv::ResourceLimitsAttr limits,
                                        Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << "Using default config for op: " << *op << "\n");
  FuncOp funcOp = op->getParentOfType<FuncOp>();
  auto interfaceOp = cast<IREE::Flow::PartitionableLoopsInterface>(*op);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);

  // Special case for not tiled ops.
  if (partitionedLoops.empty()) {
    // No tiled loops means we cannot tile (and distribute) at all. Use just one
    // single thread to run everything.
    auto pipeline =
        IREE::Codegen::DispatchLoweringPassPipeline::SPIRVDistribute;
    std::array<int64_t, 3> workgroupSize = {1, 1, 1};
    return setOpConfigAndEntryPointFnTranslation(funcOp, op, {}, {}, pipeline,
                                                 workgroupSize);
  }

  const int subgroupSize = limits.subgroup_size().getValue().getSExtValue();
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
  if (!linalgOp || linalgOp.getNumOutputs() != 1) {
    auto pipeline =
        IREE::Codegen::DispatchLoweringPassPipeline::SPIRVDistribute;

    initConfiguration();
    TileSizesListType tileSizes;
    tileSizes.push_back(workgroupTileSizes);
    tileSizes.push_back(threadTileSizes);

    return setOpConfigAndEntryPointFnTranslation(funcOp, op, tileSizes, {},
                                                 pipeline, workgroupSize);
  }

  // Common case for all linalg ops.

  // The core idea is to distribute the partitioned loops to the workgroup
  // dimensions. The goal is to fill up the GPU as much as possible, which means
  // 1) distributing to as many threads as possible, and 2) avoid assigning too
  // many threads to handle out-of-bound elements (thus idle).

  SmallVector<LoopTilingAndDistributionInfo> tiledLoopInfo =
      getTiledAndDistributedLoopInfo(funcOp);
  // The number of linalg implicit loops to partition and tiled loops
  // surrounding the op should match. Otherwise, something is incorrect.
  assert(partitionedLoops.size() == tiledLoopInfo.size());

  // The upper bound for each implicit loop: 0 - untiled, negative - dynamic.
  SmallVector<int64_t> loopBounds(loopDepth, 0);
  // tiledLoopInfo uses the reverse order of partitionedLoops.
  for (auto pair : llvm::zip(llvm::reverse(partitionedLoops), tiledLoopInfo)) {
    unsigned loopIndex = std::get<0>(pair);
    const LoopTilingAndDistributionInfo &loopInfo = std::get<1>(pair);
    Optional<int64_t> attrValue =
        getConstantIntValue(loopInfo.untiledUpperBound);
    if (attrValue) {
      loopBounds[loopIndex] = *attrValue;
    } else {
      loopBounds[loopIndex] = ShapedType::kDynamicSize;
    }
  }

  // Returns true if the given `operand` has 32-bit element type.
  auto has32BitElementType = [](Value operand) {
    auto shapedType = operand.getType().dyn_cast<ShapedType>();
    Type elementType =
        (shapedType ? shapedType.getElementType() : operand.getType());
    return elementType.isa<FloatType>() || elementType.isInteger(32);
  };

  // Whether we can try to use the vectorization pipeline.
  auto untiledResultShape = getUntiledResultShape(linalgOp, 0);
  bool vectorizable =
      !linalgOp.hasIndexSemantics() &&
      // TODO: Skip vectorization for linalg.copy ops. Right now handling of
      // it still goes through the old bufferization-first pipeline, while
      // vectorization pipeline expects tensor-semantic ops.
      !isa<linalg::CopyOp>(op) &&
      // Skip vectorization for non-minor identity inputs as it generates
      // vector.transfer_read ops with permutation maps that we currently
      // cannot lower.
      // TODO: Remove this restriction once the lowering of the permutation
      // map is supported in core.
      llvm::all_of(linalgOp.getIndexingMaps(),
                   [](AffineMap &map) { return map.isMinorIdentity(); }) &&
      // TODO: Lowering of integers other than i32 may require emulation.
      // This is currently not supported for vector operation.
      llvm::all_of(linalgOp->getOperands(), has32BitElementType) &&
      !untiledResultShape.empty() &&
      llvm::none_of(untiledResultShape, ShapedType::isDynamic);

  LLVM_DEBUG({
    llvm::dbgs() << "Linalg op " << linalgOp << "\n  partitioned loops: [";
    llvm::interleaveComma(partitionedLoops, llvm::dbgs());
    llvm::dbgs() << "]\n  loop bounds: [";
    llvm::interleaveComma(loopBounds, llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  // Distribute workload to the given `numThreads` by allowing a potental loss.
  auto distributeToThreads = [&](int64_t numThreads,
                                 Optional<int64_t> lossFactor = llvm::None) {
    LLVM_DEBUG(llvm::dbgs() << "\nLoss factor: " << lossFactor << "\n");
    initConfiguration();

    // Scan from the innermost shape dimension and try to deduce the
    // configuration for the corresponding GPU workgroup dimension.
    for (auto p : llvm::zip(llvm::reverse(partitionedLoops), tiledLoopInfo)) {
      int shapeDim = std::get<0>(p);
      int wgDim = std::get<1>(p).processorDistributionDim;
      LLVM_DEBUG({
        llvm::dbgs() << "Remaining threads: " << numThreads << "\n";
        llvm::dbgs() << "Shape dim #" << shapeDim << "=";
        llvm::dbgs() << loopBounds[shapeDim] << "\n"
                     << "Workgroup dim #" << wgDim << "\n";
      });

      // Skip untiled or dynamic dimensions.
      // TODO: Skip size-1 dimensions in Flow level tiling and distribution.
      if (loopBounds[shapeDim] <= 0) continue;

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
        llvm::dbgs() << "Candidates tile sizes: [";
        llvm::interleaveComma(candidates, llvm::dbgs());
        llvm::dbgs() << "]\n";
      });

      for (int64_t candidate : candidates) {
        if (loopBounds[shapeDim] % candidate != 0) {
          if (!lossFactor) continue;
          // Skip this candidate if it causes many threads to be idle.
          int64_t idleThreads = candidate - (loopBounds[shapeDim] % candidate);
          if (idleThreads > candidate / *lossFactor) continue;
        }
        LLVM_DEBUG(llvm::dbgs() << "Chosen Candiate " << candidate << "\n");

        // Found a suitable candidate. Try to let each thread handle 4
        // elements if this is the workgroup x dimension.
        workgroupTileSizes[shapeDim] = candidate;
        if (vectorizable && wgDim == 0 && !lossFactor && candidate % 4 == 0) {
          threadTileSizes[shapeDim] = 4;
          workgroupSize[wgDim++] = candidate / 4;
          assert(numThreads % (candidate / 4) == 0);
          numThreads /= candidate / 4;
        } else {
          if (wgDim == 0) vectorizable = false;
          threadTileSizes[shapeDim] = 1;
          workgroupSize[wgDim++] = candidate;
          assert(numThreads % candidate == 0);
          numThreads /= candidate;
        }
        assert(numThreads >= 1);
        break;
      }

      // Stop if we have distributed all threads.
      if (numThreads == 1) break;
    }
    return numThreads;
  };

  // First try to see if we can use up all threads without any loss.
  if (distributeToThreads(subgroupSize) != 1) {
    // Otherwise, allow larger and larger loss factor.

    // Threads for distribution Use 32 at least.
    int64_t numThreads = std::max(subgroupSize, 32);
    // We can tolerate (1 / lossFactor) of threads in the workgroup to be idle.
    int64_t lossFactor = 32;

    for (; lossFactor >= 1; lossFactor >>= 1) {
      if (distributeToThreads(numThreads, lossFactor) == 1) break;
    }
  }

  auto pipeline =
      vectorizable
          ? IREE::Codegen::DispatchLoweringPassPipeline::SPIRVVectorize
          : IREE::Codegen::DispatchLoweringPassPipeline::SPIRVDistribute;

  TileSizesListType tileSizes;
  tileSizes.push_back(workgroupTileSizes);
  tileSizes.push_back(threadTileSizes);

  return setOpConfigAndEntryPointFnTranslation(funcOp, op, tileSizes, {},
                                               pipeline, workgroupSize);
}

//===----------------------------------------------------------------------===//
// Configuration Dispatcher
//===----------------------------------------------------------------------===//

/// Sets the CodeGen configuration as attributes to the given `rootOp` if it's a
/// known Linalg matmul/convolution op with good configurations.
static LogicalResult setSPIRVOpConfig(const spirv::TargetEnv &targetEnv,
                                      Operation *rootOp) {
  LogicalResult result = success();
  // First try to find a proper CodeGen configuration to tile and vectorize for
  // the current target architecture.
  switch (targetEnv.getVendorID()) {
    case spirv::Vendor::ARM:
      result = detail::setMaliCodeGenConfig(targetEnv, rootOp);
      break;
    case spirv::Vendor::NVIDIA:
      result = detail::setNVIDIACodeGenConfig(targetEnv, rootOp);
      break;
    case spirv::Vendor::Qualcomm:
      result = detail::setAdrenoCodeGenConfig(targetEnv, rootOp);
      break;
    default:
      break;
  }

  if (failed(result)) return result;
  // Check whether there is actually a configuration found. If so, it's done.
  if (getLoweringConfig(rootOp)) return result;

  // Otherwise fallback to use a default configuration that tiles and
  // distributes/vectorizes.
  spirv::ResourceLimitsAttr limits = targetEnv.getResourceLimits();
  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::BatchMatmulOp, linalg::MatmulOp>([limits](auto op) {
        // Try to tile and vectorize first.
        std::array<int64_t, 2> workgroupXY = {32, 2};
        std::array<int64_t, 3> threadMNK = {8, 8, 4};
        auto result = detail::setMatmulOpConfig(op, workgroupXY, threadMNK);
        if (failed(result)) return result;
        if (getLoweringConfig(op)) return result;

        // If unsuccessful, try to tile and distribute.
        return setDefaultOpConfig(limits, op);
      })
      .Case<linalg::Conv2DNhwcHwcfOp, linalg::DepthwiseConv2DNhwcHwcOp>(
          [limits](auto op) {
            // Try to tile and vectorize first. It's common to see 32 threads
            // per subgroup for GPUs.
            auto result = detail::setConvOpConfig(op, /*subgroupSize=*/32,
                                                  /*bestTilingFactor=*/32);
            if (failed(result)) return result;
            if (getLoweringConfig(op)) return result;

            // If unsuccessful, try to tile and distribute.
            return setDefaultOpConfig(limits, op);
          })
      .Case<IREE::LinalgExt::FftOp>([limits](IREE::LinalgExt::FftOp op) {
        return setFftOpConfig(limits, op);
      })
      .Case<linalg::GenericOp>([limits](linalg::GenericOp op) {
        // If a generic op has reduction iterator types, it can be treated as a
        // root op for configuration as well. Use the default configuration,
        // which will mark it as a root.
        if (op.getNumLoops() != op.getNumParallelLoops()) {
          return setDefaultOpConfig(limits, op);
        }
        return success();
      })
      .Default([](Operation *) { return success(); });
};

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult initSPIRVLaunchConfig(ModuleOp module) {
  llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> entryPointOps =
      getAllEntryPoints(module);
  spirv::TargetEnvAttr targetEnvAttr = getSPIRVTargetEnvAttr(module);
  if (!targetEnvAttr) {
    return module.emitOpError(
        "expected parent hal.executable.variant to have spv.target_env "
        "attribute");
  }
  spirv::TargetEnv targetEnv(targetEnvAttr);
  spirv::ResourceLimitsAttr limits = targetEnv.getResourceLimits();

  for (auto funcOp : module.getOps<FuncOp>()) {
    auto entryPointOp = entryPointOps.lookup(funcOp.getName());
    if (!entryPointOp) continue;

    SmallVector<Operation *> computeOps;
    SmallVector<LoopTilingAndDistributionInfo> tiledLoops;
    if (failed(getComputeOps(funcOp, computeOps, tiledLoops))) {
      return funcOp.emitOpError("failed to get compute ops");
    }

    Operation *rootOperation = nullptr;
    // Try to find a configuration according to a matmul/convolution op and use
    // it as the root op.
    for (Operation *computeOp : computeOps) {
      if (failed(setSPIRVOpConfig(targetEnv, computeOp))) return failure();

      // Check if the op configuration was set.
      if (!getLoweringConfig(computeOp)) continue;

      if (rootOperation) {
        return computeOp->emitOpError(
            "unhandled multiple roots in dispatch region");
      }
      rootOperation = computeOp;
    }

    // If there are still no root op, check for any linalg.generic op.
    if (!rootOperation) {
      for (Operation *computeOp : reverse(computeOps)) {
        if (failed(setDefaultOpConfig(limits, computeOp))) return failure();

        // Check if the op configuration was set.
        if (!getLoweringConfig(computeOp)) {
          return computeOp->emitOpError(
              "without known roots, the last compute operation in the tiled "
              "loop body is expected to be set as root");
        }
        rootOperation = computeOp;
        break;
      }
    }

    if (!rootOperation) {
      // If the tiled loops are not empty then this could be a corner case of
      // tensor.insert_slice being tiled and distributed, that just shows up as
      // a `flow.dispatch.tensor.load` and a `flow.dispatch.tensor.store` (or as
      // a copy). For now just treat the tiled loops not being empty as an
      // indicator of that. Need a better way of information flow from flow
      // dialect to hal.
      if (!tiledLoops.empty()) {
        // These configuration parameters will be overwritten by the
        // SPIRVDistributeCopy pipeline later.
        const int64_t subgroupSize =
            limits.subgroup_size().getValue().getSExtValue();
        std::array<int64_t, 3> workgroupSize = {subgroupSize, 1, 1};
        SmallVector<int64_t> workloadPerWorkgroup(tiledLoops.size(), 1);
        workloadPerWorkgroup.front() = subgroupSize;
        setTranslationInfo(
            funcOp,
            IREE::Codegen::DispatchLoweringPassPipeline::SPIRVDistributeCopy,
            workloadPerWorkgroup, workgroupSize);
        return success();
      }
      return funcOp.emitError("contains no root Linalg operation");
    }

    // Propogate the `lowering.config` attribute to the other ops.
    // TODO(ravishankarm, antiagainst): This is a very specific use (and
    // fragile). In general, this should not be needed. Things are already tiled
    // and distributed. The rest of the compilation must be structured to either
    // use `TileAndFuse` or they are independent configurations that are
    // determined based on the op.
    IREE::Codegen::LoweringConfigAttr config = getLoweringConfig(rootOperation);
    for (auto op : computeOps) {
      if (op == rootOperation) continue;
      setLoweringConfig(op, config);
    }
  }
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
