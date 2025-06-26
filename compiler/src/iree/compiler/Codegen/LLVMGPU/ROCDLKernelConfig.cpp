// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/ROCDLKernelConfig.h"

#include "compiler/src/iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUSelectUKernels.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

namespace {

using CodeGenPipeline = IREE::Codegen::DispatchLoweringPassPipeline;

//===----------------------------------------------------------------------===//
// Warp Reduction Configuration
//===----------------------------------------------------------------------===//

static bool isMatvecLike(linalg::LinalgOp linalgOp) {
  if (linalgOp.getNumParallelLoops() != 2)
    return false;

  if (linalgOp.getNumReductionLoops() != 1)
    return false;

  // TODO: Allow for matvec with fused dequantization.
  FailureOr<linalg::ContractionDimensions> dims =
      linalg::inferContractionDims(linalgOp);
  if (failed(dims))
    return false;

  // TODO: Support batch matvec.
  if (!dims->batch.empty())
    return false;

  for (ArrayRef indices : {dims->m, dims->n, dims->k}) {
    if (!llvm::hasSingleElement(indices))
      return false;
  }

  // Check if the first parallel dimension has bound 1, indicating we found a
  // vector shape.
  SmallVector<int64_t> bounds = linalgOp.getStaticLoopRanges();
  if (bounds[dims->m.front()] != 1)
    return false;

  return true;
}

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

  // Tile all the parallel dimension to 1.
  SmallVector<unsigned> partitionedLoops =
      cast<PartitionableLoopsInterface>(op.getOperation())
          .getPartitionableLoops(kNumMaxParallelDims);
  size_t numLoops = partitionedLoops.empty() ? 0 : partitionedLoops.back() + 1;
  SmallVector<int64_t> workgroupTileSizes(numLoops, 1);

  // Without any bounds on dynamic reduction dims, we need specialization to
  // get peak performance. For now, just use the warp size.
  if (numDynamicReductionDims) {
    SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 0);
    int64_t preferredSubgroupSize = target.getPreferredSubgroupSize();
    reductionTileSizes[reductionDims[0]] = preferredSubgroupSize;
    TileSizesListType tileSizes;
    tileSizes.emplace_back(std::move(workgroupTileSizes)); // Workgroup level
    tileSizes.emplace_back(std::move(reductionTileSizes)); // Reduction level
    std::array<int64_t, 3> workgroupSize = {preferredSubgroupSize, 1, 1};
    if (failed(setOpConfigAndEntryPointFnTranslation(
            entryPoint, op, tileSizes, CodeGenPipeline::LLVMGPUWarpReduction,
            workgroupSize))) {
      return failure();
    }
    return success();
  }

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

  const Type elementType =
      cast<ShapedType>(op.getDpsInitOperand(0)->get().getType())
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
  if (llvm::none_of(bounds, ShapedType::isDynamic) && isMatvecLike(op)) {
    int64_t lastParallelBound = bounds[parallelDims.back()];
    int64_t numParallelReductions = 1;
    const int64_t maxParallelFactor = groupSize / 4;
    for (int64_t parallelFactor = 2;
         (parallelFactor < maxParallelFactor) &&
         (lastParallelBound % parallelFactor == 0) &&
         (lastParallelBound > parallelFactor);
         parallelFactor *= 2) {
      numParallelReductions = parallelFactor;
    }
    workgroupTileSizes.back() = numParallelReductions;
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

//===----------------------------------------------------------------------===//
// Root Configuration
//===----------------------------------------------------------------------===//

static LogicalResult setRootConfig(IREE::GPU::TargetAttr target,
                                   mlir::FunctionOpInterface entryPointFn,
                                   Operation *computeOp) {
  IREE::GPU::UKernelConfigAttr ukernelConfig = selectUKernel(computeOp);
  if (succeeded(setDataTiledMultiMmaLoweringConfig(target, entryPointFn,
                                                   computeOp, ukernelConfig))) {
    return success();
  }
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(computeOp)) {
    if (succeeded(IREE::GPU::setMatmulLoweringConfig(target, entryPointFn,
                                                     linalgOp))) {
      return success();
    }
    if (succeeded(IREE::GPU::setIGEMMConvolutionLoweringConfig(
            target, entryPointFn, computeOp))) {
      return success();
    }
    if (succeeded(setWarpReductionConfig(target, entryPointFn, linalgOp))) {
      return success();
    }
    // TODO: Add configurations for matmul here too.
    if (succeeded(IREE::GPU::setTileAndFuseLoweringConfig(target, entryPointFn,
                                                          computeOp))) {
      return success();
    }
  }

  return failure();
}

// Propagates the configuration to the other ops.
static void propagateLoweringConfig(Operation *rootOp,
                                    ArrayRef<Operation *> computeOps) {
  if (IREE::Codegen::LoweringConfigAttrInterface config =
          getLoweringConfig(rootOp)) {
    for (auto op : computeOps) {
      if (op != rootOp)
        setLoweringConfig(op, config);
    }
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult initROCDLLaunchConfig(FunctionOpInterface funcOp) {
  IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
  if (!target)
    return funcOp.emitError("missing GPU target in #hal.executable.target");

  // First check whether we already have workgroup count set--it's a
  // "contract" to indicate that we should bypass all tiling and
  // distribution to go down just the most basic lowering flow.
  if (auto exportOp = getEntryPoint(funcOp)) {
    if (Block *body = exportOp->getWorkgroupCountBody()) {
      auto retOp = cast<IREE::HAL::ReturnOp>(body->getTerminator());
      // For scalar dispatch cases--using just one thread of one workgroup.
      auto isOne = [](Value value) { return matchPattern(value, m_One()); };
      if (llvm::all_of(retOp.getOperands(), isOne)) {
        std::array<int64_t, 3> workgroupSize = {1, 1, 1};
        auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
            funcOp.getContext(),
            IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUBaseLowering,
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
    // Currently ROCDL requires propagation of user lowering configs for
    // all pipelines except TileAndFuse.
    if (translationInfo.getDispatchLoweringPassPipeline() !=
        IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTileAndFuse) {
      for (auto op : computeOps) {
        if (getLoweringConfig(op)) {
          propagateLoweringConfig(op, computeOps);
          break;
        }
      }
    }
  }

  Operation *rootOp = nullptr;

  // Find the root operation. linalg.generic and linalg.fill are not root
  // operations if there are other compute operations present.
  for (Operation *op : llvm::reverse(computeOps)) {
    if (!isa<linalg::GenericOp, linalg::FillOp>(op)) {
      rootOp = op;
      break;
    }
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      // linalg.generic with `reduction` iterator types are roots as well.
      if (genericOp.getNumLoops() != genericOp.getNumParallelLoops()) {
        rootOp = op;
        break;
      }
    }
  }

  if (!rootOp) {
    for (Operation *op : llvm::reverse(computeOps)) {
      if (isa<linalg::GenericOp, linalg::FillOp>(op)) {
        rootOp = op;
        break;
      }
    }
  }

  if (!rootOp) {
    // No root operation found, set it to none.
    auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
        funcOp.getContext(), CodeGenPipeline::None);
    if (failed(setTranslationInfo(funcOp, translationInfo))) {
      return failure();
    }
    return success();
  }

  if (failed(setRootConfig(target, funcOp, rootOp)))
    return failure();

  if (getTranslationInfo(funcOp).getDispatchLoweringPassPipeline() !=
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTileAndFuse) {
    propagateLoweringConfig(rootOp, computeOps);
  }
  return success();
}

} // namespace mlir::iree_compiler
