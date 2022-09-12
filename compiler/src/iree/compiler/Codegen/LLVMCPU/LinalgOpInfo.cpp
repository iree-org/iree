// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/LinalgOpInfo.h"

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"

using namespace mlir::linalg;

namespace mlir {
namespace iree_compiler {

LinalgOpInfo::LinalgOpInfo(linalg::LinalgOp linalgOp) { computeInfo(linalgOp); }

/// Returns true if `map` is a tranposition.
// TODO(dcaballe): Discern between "memcopy" transposes and "shuffle"
// transposes.
// TODO(dcaballe): Shouldn't there be a utility for this somewhere in Affine?
static bool isTransposeMap(AffineMap map) {
  unsigned prevDim = 0;
  for (AffineExpr expr : map.getResults()) {
    if (expr.isa<AffineConstantExpr>()) {
      continue;
    }

    if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
      if (prevDim > dimExpr.getPosition()) {
        return true;
      }
      prevDim = dimExpr.getPosition();
      continue;
    }

    llvm_unreachable("Unexpected AffineExpr");
  }

  return false;
}

/// Returns true if a LinalgOp implements a transpose.
// TODO(dcaballe):
//   * Consider transpose + reductions.
//   * Consider input and output transposes.
static bool isTransposeLinalgOp(linalg::LinalgOp linalgOp) {
  // Reductions are not supported.
  if (linalgOp.getNumReductionLoops() > 0) {
    return false;
  }

  // Multiple outputs are not supported yet.
  if (linalgOp.getNumOutputs() != 1) {
    return false;
  }

  // Inverse map to use transfer op permutation logic.
  AffineMap outputInversedMap = inversePermutation(
      linalgOp.getTiedIndexingMap(linalgOp.getOutputOperand(0)));
  SmallVector<AffineMap> inputInversedMaps;
  for (OpOperand *linalgOperand : linalgOp.getInputOperands()) {
    inputInversedMaps.push_back(inverseAndBroadcastProjectedPermutation(
        linalgOp.getTiedIndexingMap(linalgOperand)));
  }

  bool isInputTransposed = llvm::any_of(
      inputInversedMaps, [](AffineMap map) { return isTransposeMap(map); });
  bool isOutputTransposed = isTransposeMap(outputInversedMap);

  return isInputTransposed || isOutputTransposed;
}

static bool computeTransposeInfo(LinalgOp linalgOp) {
  return isTransposeLinalgOp(linalgOp);
}

static bool computeReductionInfo(LinalgOp linalgOp) {
  return linalgOp.getNumReductionLoops() > 1;
}

void LinalgOpInfo::computeInfo(LinalgOp linalgOp) {
  transposeTrait = computeTransposeInfo(linalgOp);
  reductionTrait = computeReductionInfo(linalgOp);
}

}  // namespace iree_compiler
}  // namespace mlir
