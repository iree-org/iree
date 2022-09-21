// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/LinalgOpInfo.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"

using namespace mlir::linalg;

namespace mlir {
namespace iree_compiler {

/// Returns true if `map` is a tranpose. A transpose map is a projected
/// permutation with or without zeros in results where there exist at least two
/// dimensions di and dj such that di < dj and result_pos(di) > result_pos(dj).
/// Examples:
///
///  (d0, d1, d2) -> (d0, d2) is not a transpose map.
///  (d0, d1, d2) -> (d2, d0) is a transpose map.
///  (d0, d1, d2) -> (d1, d2) is not a transpose map.
///  (d0, d1, d2) -> (d0, 0, d1) is not a transpose map.
///  (d0, d1, d2) -> (d2, 0, d1) is a transpose map.
///  (d0, d1, d2) -> (d1, 0) is not a transpose map.
///
// TODO(dcaballe): Discern between "memcopy" transposes and "shuffle"
// transposes.
// TODO(dcaballe): Move to Affine utils?
static bool isTransposeMap(AffineMap map) {
  // A transpose map must be a projected permutation with or without
  // broadcasted/reduction dimensions.
  if (!map.isProjectedPermutation(/*allowZeroInResults=*/true)) {
    return false;
  }

  // Check that the projected permutation has at least two result dimensions
  // that are actually transposed by comparing its input position.
  unsigned prevDim = 0;
  for (AffineExpr expr : map.getResults()) {
    if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
      // Constant zero expression, guaranteed by 'allowZeroInResults' above.
      continue;
    } else if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
      if (prevDim > dimExpr.getPosition()) {
        return true;
      }
      prevDim = dimExpr.getPosition();
    } else {
      return false;
    }
  }

  return false;
}

/// The default filter passes all op operands.
static bool defaultTransposeMapFilter(AffineMap map) { return true; }

LinalgOpInfo::LinalgOpInfo(linalg::LinalgOp linalgOp)
    : transposeMapFilter(defaultTransposeMapFilter) {
  computeInfo(linalgOp);
}
LinalgOpInfo::LinalgOpInfo(linalg::LinalgOp linalgOp,
                           TransposeMapFilter transposeMapFilter)
    : transposeMapFilter(transposeMapFilter) {
  computeInfo(linalgOp);
}

/// Returns true if a LinalgOp implements a transpose.
// TODO(dcaballe):
//   * Consider transpose + reductions.
//   * Consider input and output transposes.
static SmallVector<OpOperand *> computeTransposeInfo(
    LinalgOp linalgOp, TransposeMapFilter transposeMapFilter) {
  SmallVector<OpOperand *> transposeOperands;

  // Reductions are not supported.
  if (linalgOp.getNumReductionLoops() > 0) {
    return transposeOperands;
  }

  // Multiple outputs are not supported yet.
  if (linalgOp.getNumOutputs() != 1) {
    return transposeOperands;
  }

  // Inverse map to use transfer op permutation logic.
  AffineMap outputInversedMap = inversePermutation(
      linalgOp.getTiedIndexingMap(linalgOp.getOutputOperand(0)));

  SmallVector<AffineMap> inputInversedMaps;
  for (OpOperand *linalgOperand : linalgOp.getInputOperands()) {
    auto map = linalgOp.getTiedIndexingMap(linalgOperand);
    if (!map.isProjectedPermutation(/*allowZeroInResults=*/true)) {
      return transposeOperands;
    }
    AffineMap inverseMap = inverseAndBroadcastProjectedPermutation(map);
    if (isTransposeMap(inverseMap) && transposeMapFilter(inverseMap)) {
      transposeOperands.push_back(linalgOperand);
    }
  }

  if (isTransposeMap(outputInversedMap) &&
      transposeMapFilter(outputInversedMap)) {
    transposeOperands.push_back(linalgOp.getOutputOperand(0));
  }

  return transposeOperands;
}

static bool computeReductionInfo(LinalgOp linalgOp) {
  return linalgOp.getNumReductionLoops() > 1;
}

static bool computeDynamicInfo(LinalgOp linalgOp) {
  return linalgOp.hasDynamicShape();
}

static bool computeTwoOrThreeLoopsInfo(LinalgOp linalgOp) {
  return linalgOp.getNumParallelLoops() >= 2 &&
         linalgOp.getNumParallelLoops() <= 3;
}

static bool computeGenericInfo(LinalgOp linalgOp) {
  return isa<GenericOp>(linalgOp);
}

void LinalgOpInfo::computeInfo(LinalgOp linalgOp) {
  transposeOperands = computeTransposeInfo(linalgOp, transposeMapFilter);
  reductionTrait = computeReductionInfo(linalgOp);
  dynamicTrait = computeDynamicInfo(linalgOp);
  genericTrait = computeGenericInfo(linalgOp);
  twoOrThreeLoopsTrait = computeTwoOrThreeLoopsInfo(linalgOp);
}

}  // namespace iree_compiler
}  // namespace mlir
