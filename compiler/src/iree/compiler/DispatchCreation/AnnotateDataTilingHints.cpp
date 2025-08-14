// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-annotate-data-tiling-hints"

namespace mlir::iree_compiler::DispatchCreation {
#define GEN_PASS_DEF_ANNOTATEDATATILINGHINTSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {
struct AnnotateDataTilingHintsPass final
    : impl::AnnotateDataTilingHintsPassBase<AnnotateDataTilingHintsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

/// Returns true iff the linalgOp has a body like a regular matmul, i.e.
/// yield(add(out, mul(cast(in0), cast(in1))))
static bool hasMatmulLikeBody(linalg::LinalgOp linalgOp) {
  auto outBlockArg =
      linalgOp.getMatchingBlockArgument(linalgOp.getDpsInitOperand(0));
  auto yieldOp =
      dyn_cast<linalg::YieldOp>(outBlockArg.getParentBlock()->getTerminator());
  if (!yieldOp) {
    return false;
  }
  Operation *addOp = yieldOp->getOperand(0).getDefiningOp();
  if (!addOp || !isa<arith::AddIOp, arith::AddFOp>(addOp)) {
    return false;
  }
  Value addLhs = addOp->getOperand(0);
  Value addRhs = addOp->getOperand(1);
  Operation *addLhsOp = addLhs.getDefiningOp();
  Operation *addRhsOp = addRhs.getDefiningOp();
  if (!(addLhsOp && addRhs == outBlockArg) &&
      !(addRhsOp && addLhs == outBlockArg)) {
    return false;
  }
  Operation *mulOp = addLhsOp ? addLhsOp : addRhsOp;
  if (!isa<arith::MulFOp, arith::MulIOp>(mulOp)) {
    return false;
  }
  Value mulLhs = mulOp->getOperand(0);
  Value mulRhs = mulOp->getOperand(1);
  auto mulLhsOp = mulLhs.getDefiningOp<CastOpInterface>();
  auto mulRhsOp = mulRhs.getDefiningOp<CastOpInterface>();
  if (!isa<BlockArgument>(mulLhs) && !mulLhsOp && !isa<BlockArgument>(mulRhs) &&
      !mulRhsOp) {
    return false;
  }
  if ((mulLhsOp && !isa<BlockArgument>(mulLhsOp->getOperand(0))) ||
      (mulRhsOp && !isa<BlockArgument>(mulRhsOp->getOperand(0)))) {
    return false;
  }
  return true;
}

/// Not all contractions are supported by data tiling, so return true if:
///   1) linalgOp has pure tensor semantics.
///   2) linalgOp does not have a preset compilation info.
///   3) The workgroup count is not present if linalgOp is wrapped within
///      Flow::DispatchRegionOp.
///   4) All the operands do not have encodings.
///   5) linalgOp has contraction indexingMaps.
///   6) There are not more than one of each contraction dimension.
///   7) There is an M or N dimension, and there is a K dimension.
///   8) linalgOp has the same body as an ordinary int or float matmul.
///
/// These restrictions are required because data tiling currently creates
/// an Mmt4DOp or BatchMmt4DOp on the packed inputs.
///
/// TODO(#16176): Loosen restrictions on contraction ops once data tiling
/// can support more cases.
static bool isSupportedContractionOp(linalg::LinalgOp linalgOp) {
  if (!linalgOp.hasPureTensorSemantics()) {
    return false;
  }
  if (getCompilationInfo(linalgOp)) {
    return false;
  }
  auto hasWorkgroupCounts = [](Operation *op) -> bool {
    auto parentDispatchOp = op->getParentOfType<IREE::Flow::DispatchRegionOp>();
    return parentDispatchOp && !parentDispatchOp.getWorkgroupCount().empty();
  };
  if (hasWorkgroupCounts(linalgOp)) {
    return false;
  }
  auto hasEncoding = [](Value operand) -> bool {
    auto type = llvm::dyn_cast<RankedTensorType>(operand.getType());
    return type && type.getEncoding();
  };
  if (llvm::any_of(linalgOp.getDpsInputs(), hasEncoding) ||
      llvm::any_of(linalgOp.getDpsInits(), hasEncoding)) {
    return false;
  }

  if (!linalg::isaContractionOpInterface(linalgOp)) {
    return false;
  }
  auto cDims = linalg::inferContractionDims(linalgOp);
  if (failed(cDims) || cDims->batch.size() > 1 || cDims->m.size() > 1 ||
      cDims->n.size() > 1 || cDims->k.size() > 1) {
    return false;
  }
  if ((cDims->n.empty() && cDims->m.empty()) || cDims->k.empty()) {
    return false;
  }
  if (!hasMatmulLikeBody(linalgOp)) {
    return false;
  }
  return true;
}

void AnnotateDataTilingHintsPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  SmallVector<Operation *> candidates;
  WalkResult result = funcOp.walk([&](Operation *op) -> WalkResult {
    if (IREE::Encoding::hasDataTilingHint(op)) {
      return WalkResult::interrupt();
    }
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (linalgOp && isSupportedContractionOp(linalgOp)) {
      candidates.push_back(op);
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return;
  }
  for (Operation *op : candidates) {
    IREE::Encoding::setDataTilingHint(op);
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
