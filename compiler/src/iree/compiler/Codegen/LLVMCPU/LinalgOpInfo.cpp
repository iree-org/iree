// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/LinalgOpInfo.h"

//#include <numeric>

//#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
//#include "iree/compiler/Codegen/LLVMCPU/TargetMLTransformInfo.h"
//#include "iree/compiler/Codegen/Transforms/Transforms.h"
//#include "iree/compiler/Codegen/Utils/Utils.h"
//#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
//#include "llvm/ADT/TypeSwitch.h"
//#include "llvm/Support/CommandLine.h"
//#include "llvm/Support/TargetSelect.h"
//#include "mlir/Dialect/Func/IR/FuncOps.h"
//#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
//#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
//#include "mlir/Dialect/MemRef/IR/MemRef.h"
//#include "mlir/Dialect/MemRef/Transforms/Passes.h"
//#include "mlir/Dialect/Utils/StaticValueUtils.h"
//#include "mlir/IR/Matchers.h"
//#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

  // Inverse map to use transfer linalgOp permutation logic.
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
  hasTransposeTrait = computeTransposeInfo(linalgOp);
  hasReductionTrait = computeReductionInfo(linalgOp);
}

}  // namespace iree_compiler
}  // namespace mlir
