// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/VMVX/KernelDispatch.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/IR/Attributes.h"

#define DEBUG_TYPE "vmvx-kernel-dispatch"

namespace mlir::iree_compiler {

constexpr int kDefaultDistTileSize = 64;

static SmallVector<int64_t>
getDefaultDistributionTileSizes(TilingInterface op) {
  unsigned numLoops = op.getLoopIteratorTypes().size();
  auto partitionedLoops = cast<PartitionableLoopsInterface>(op.getOperation())
                              .getPartitionableLoops(kNumMaxParallelDims);
  SmallVector<int64_t> distTileSizes(numLoops, kDefaultDistTileSize);
  llvm::DenseSet<unsigned> partitionedLoopsSet(partitionedLoops.begin(),
                                               partitionedLoops.end());
  for (auto dim : llvm::seq<int64_t>(0, distTileSizes.size())) {
    if (!partitionedLoopsSet.count(dim))
      distTileSizes[dim] = 0;
  }

  return distTileSizes;
}

static IREE::CPU::LoweringConfigAttr
getLoweringConfigWithDistributionTiles(MLIRContext *ctx,
                                       ArrayRef<int64_t> tileSizes) {
  SmallVector<NamedAttribute> items;
  items.emplace_back(
      IREE::CPU::getTilingLevelName(IREE::CPU::TilingLevel::DistributionTiles),
      IREE::CPU::LoweringConfigAttr::getTilingLevelAttr(ctx, tileSizes));
  return IREE::CPU::LoweringConfigAttr::get(ctx, items);
}

/// Sets the lowering configuration for dispatch region for linalg_ext.fft
/// root op.
static LogicalResult setRootConfig(mlir::FunctionOpInterface entryPointFn,
                                   IREE::LinalgExt::FftOp fftOp) {
  assert(!getLoweringConfig(fftOp) && "expected lowering_config is not set");
  SmallVector<int64_t> distTileSizes = getDefaultDistributionTileSizes(fftOp);
  auto rank = fftOp.getOperandRank();
  if (distTileSizes.size() >= rank && distTileSizes[rank - 1] != 0) {
    APInt value;
    if (matchPattern(fftOp.getStage(), m_ConstantInt(&value))) {
      distTileSizes[rank - 1] = 1ll << value.getSExtValue();
      distTileSizes[rank - 1] = std::max(
          distTileSizes[rank - 1], static_cast<int64_t>(kDefaultDistTileSize));
    } else {
      return fftOp.emitOpError("non-constant stage might not work for fft op");
    }
  }
  IREE::CPU::LoweringConfigAttr loweringConfig =
      getLoweringConfigWithDistributionTiles(fftOp.getContext(), distTileSizes);
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, fftOp, loweringConfig,
      IREE::Codegen::DispatchLoweringPassPipeline::VMVXDefault);
}

static LogicalResult setRootConfig(mlir::FunctionOpInterface entryPointFn,
                                   TilingInterface tilingInterfaceOp) {
  assert(!getLoweringConfig(tilingInterfaceOp) &&
         "expected lowering_config is not set");

  SmallVector<int64_t> distTileSizes =
      getDefaultDistributionTileSizes(tilingInterfaceOp);
  IREE::CPU::LoweringConfigAttr loweringConfig =
      getLoweringConfigWithDistributionTiles(tilingInterfaceOp.getContext(),
                                             distTileSizes);
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, tilingInterfaceOp, loweringConfig,
      IREE::Codegen::DispatchLoweringPassPipeline::VMVXDefault);
}

static LogicalResult
setVMVXRootConfigImpl(mlir::FunctionOpInterface entryPointFn, Operation *op) {
  auto setRootConfigFn = [&](Operation *op) -> LogicalResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<IREE::LinalgExt::FftOp>(
            [&](auto op) { return setRootConfig(entryPointFn, op); })
        .Case<TilingInterface>(
            [&](auto op) { return setRootConfig(entryPointFn, op); })
        .Default([&](Operation *op) { return success(); });
  };
  return setRootConfigFn(op);
}

static LogicalResult
lowerUsingVMVXDefaultPipeline(mlir::FunctionOpInterface op) {
  auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      op.getContext(),
      IREE::Codegen::DispatchLoweringPassPipeline::VMVXDefault);
  return setTranslationInfo(op, translationInfo);
}

/// Sets the translation information to use for a dispatch region.
static LogicalResult
setConfigForKernel(mlir::FunctionOpInterface entryPointFn) {
  SmallVector<Operation *> computeOps = getComputeOps(entryPointFn);
  if (computeOps.empty()) {
    return lowerUsingVMVXDefaultPipeline(entryPointFn);
  }

  FailureOr<Operation *> rootOp = getRootOperation(computeOps);
  if (failed(rootOp)) {
    return failure();
  }

  // Handle the case with no known root operation.
  Operation *rootOperation = rootOp.value();
  if (!rootOperation) {
    return lowerUsingVMVXDefaultPipeline(entryPointFn);
  }

  if (failed(setVMVXRootConfigImpl(entryPointFn, rootOperation))) {
    return failure();
  }

  return success();
}

LogicalResult initVMVXLaunchConfig(FunctionOpInterface funcOp) {
  if (getTranslationInfo(funcOp)) {
    return success();
  }

  if (failed(setConfigForKernel(funcOp))) {
    return failure();
  }

  return success();
}

} // namespace mlir::iree_compiler
