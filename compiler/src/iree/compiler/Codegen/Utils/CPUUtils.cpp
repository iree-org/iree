// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/CPUUtils.h"

#include <numeric>

#include "iree/compiler/Codegen/Dialect/IREECodegenAttrs.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#define DEBUG_TYPE "iree-codegen-cpu-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir {
namespace iree_compiler {

using IREE::Codegen::DispatchLoweringPassPipeline;

template <typename T>
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const llvm::SmallVectorImpl<T> &vector) {
  for (T element : vector) {
    os << element << " ";
  }

  return os;
}

FailureOr<Operation *> getRootOperation(ArrayRef<Operation *> computeOps) {
  Operation *rootOperation = nullptr;
  for (auto op : llvm::reverse(computeOps)) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      // Do not treat linalg ops that are all parallel as root operations in
      // this sweep.
      if (linalgOp.getNumLoops() == linalgOp.getNumParallelLoops())
        continue;

      // All other linalg ops are root ops.
      rootOperation = op;
      break;
    }

    if (isa<TilingInterface>(op) &&
        !isa<tensor::PadOp, tensor::PackOp, tensor::UnPackOp>(op)) {
      // All other operations that implement this interface are root ops.
      rootOperation = op;
      break;
    }
  }

  if (!rootOperation) {
    // Check for elementwise operations.
    for (auto op : llvm::reverse(computeOps)) {
      if (isa<linalg::LinalgOp>(op)) {
        rootOperation = op;
        break;
      }
    }
  }

  if (!rootOperation) {
    // Check for pad/pack/unpack ops by themselves.
    for (auto op : llvm::reverse(computeOps)) {
      if (isa<tensor::PadOp, tensor::PackOp, tensor::UnPackOp>(op)) {
        rootOperation = op;
        break;
      }
    }
  }

  return rootOperation;
}

LogicalResult adjustTileSizesForUnPackOp(func::FuncOp entryPointFn,
                                         Operation *rootOp) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp);
  if (!linalgOp)
    return success();

  auto loweringConfig = getLoweringConfig(linalgOp);
  TileSizesListType tileSizesList = loweringConfig.getTileSizeVals();

  bool foundUnPackOp = false;
  SmallVector<int64_t> alignedSizes(linalgOp.getNumLoops(), 1);
  for (OpOperand *opOperand : linalgOp.getDpsInputOperands()) {
    auto unpackOp = opOperand->get().getDefiningOp<tensor::UnPackOp>();
    if (!unpackOp)
      continue;

    foundUnPackOp = true;
    auto idxMap = linalgOp.getMatchingIndexingMap(opOperand);
    LLVM_DEBUG(DBGS() << "Find unpack op candidate: " << unpackOp << "\n"
                      << "The corresponding indexing map is: " << idxMap
                      << "\n");

    SmallVector<int64_t> innerTiles = unpackOp.getStaticTiles();
    ArrayRef<int64_t> dimPos = unpackOp.getInnerDimsPos();
    for (auto [pos, size] : llvm::zip_equal(dimPos, innerTiles)) {
      if (ShapedType::isDynamic(size))
        continue;
      auto dimExpr = dyn_cast<AffineDimExpr>(idxMap.getResult(pos));
      if (!dimExpr)
        return failure();
      int mappedPos = dimExpr.getPosition();
      alignedSizes[mappedPos] = std::lcm(alignedSizes[mappedPos], size);
    }
  }

  if (!foundUnPackOp)
    return success();

  LLVM_DEBUG(DBGS() << "The tile sizes for each dimension should be aligned to "
                    << alignedSizes);

  // Fixup for making tileSizes be multiple of inner_tile_sizes.
  for (SmallVectorImpl<int64_t> &tileSizes : tileSizesList) {
    for (auto idx : llvm::seq<int64_t>(0, tileSizes.size())) {
      if (tileSizes[idx] == 0)
        continue;
      tileSizes[idx] = llvm::alignTo(tileSizes[idx], alignedSizes[idx]);
    }
  }

  auto pipeline = getTranslationInfo(entryPointFn).getPassPipeline().getValue();
  if (pipeline == DispatchLoweringPassPipeline::CPUDoubleTilingPeelingExpert) {
    LLVM_DEBUG(DBGS() << "unpack fusion does not work with peeling, falling "
                         "back to non-peeling path");
    pipeline = DispatchLoweringPassPipeline::CPUDoubleTilingExpert;
  }

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, rootOp, tileSizesList,
      loweringConfig.getScalableTileFlagVals(), pipeline);
}

} // namespace iree_compiler
} // namespace mlir
