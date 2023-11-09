// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Utils.h"

#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#define DEBUG_TYPE "iree-llvmcpu-utils"

namespace mlir {
namespace iree_compiler {

bool preferIntrinsicsOverAsm(IREE::HAL::ExecutableTargetAttr targetAttr) {
  auto intrinsicsAttr =
      getConfigBoolAttr(targetAttr, "prefer_intrinsics_over_asm");
  return intrinsicsAttr && intrinsicsAttr->getValue();
}

bool hasAVX2Feature(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return hasFeature(targetAttr, "+avx2");
}

bool hasAVX512fFeature(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return hasFeature(targetAttr, "+avx512f");
}

bool hasVFeature(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return hasFeature(targetAttr, "+v");
}

bool hasZve32xFeature(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return hasFeature(targetAttr, "+zve32x");
}

bool hasZve32fFeature(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return hasFeature(targetAttr, "+zve32f");
}

bool hasZve64xFeature(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return hasFeature(targetAttr, "+zve64x");
}

bool hasAnySVEFeature(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return hasFeature(targetAttr, "+sve") || hasFeature(targetAttr, "+sve2");
}

bool hasSMEFeature(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return hasFeature(targetAttr, "+sme");
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

bool hasByteAlignedElementTypes(linalg::LinalgOp linalgOp) {
  return llvm::all_of(linalgOp->getOperands(), [](Value operand) {
    auto bitwidth =
        IREE::Util::getTypeBitWidth(getElementTypeOrSelf(operand.getType()));
    return bitwidth % 8 == 0;
  });
}

void setSCFTileSizes(scf::SCFTilingOptions &options, TilingInterface consumerOp,
                     SmallVector<int64_t> tileSizes,
                     SmallVector<bool> tileScalableFlags) {
  // scf::tileUsingSCFForOp expects the num of tile sizes = num of loops.
  int numLoops = consumerOp.getLoopIteratorTypes().size();
  tileSizes.resize(numLoops, /*default=*/0);
  tileScalableFlags.resize(numLoops, /*default=*/false);
  if (!llvm::is_contained(tileScalableFlags, true)) {
    // Non-scalable case: All constant tile sizes.
    options.setTileSizes(
        getAsIndexOpFoldResult(consumerOp.getContext(), tileSizes));
  } else {
    // Scalable case: Multiply scalable tile sizes by a vector.vscale op.
    options.setTileSizeComputationFunction(
        [=](OpBuilder &b, Operation *op) -> SmallVector<OpFoldResult> {
          auto loc = op->getLoc();
          return llvm::map_to_vector(
              llvm::zip(tileSizes, tileScalableFlags),
              [&](auto pair) -> OpFoldResult {
                auto [t, isScalable] = pair;
                Value size = b.create<arith::ConstantIndexOp>(loc, t);
                if (isScalable) {
                  Value vscale = b.create<vector::VectorScaleOp>(loc);
                  size = b.create<arith::MulIOp>(loc, size, vscale);
                }
                return size;
              });
        });
  }
}

std::optional<CastOpInterface>
getCastOpOfElementWiseCast(linalg::GenericOp genericOp) {
  if (!genericOp || genericOp.getNumDpsInputs() != 1 ||
      genericOp.getNumDpsInits() != 1 ||
      genericOp.getBody()->getOperations().size() != 2 ||
      !isElementwise(genericOp)) {
    return std::nullopt;
  }
  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
  auto castOp = yieldOp->getOperand(0).getDefiningOp<CastOpInterface>();
  if (!castOp) {
    return std::nullopt;
  }
  Value castIn = castOp->getOperand(0);
  if (castIn.isa<BlockArgument>() &&
      castIn.cast<BlockArgument>().getArgNumber() != 0) {
    return std::nullopt;
  }
  return castOp;
}

} // namespace iree_compiler
} // namespace mlir
