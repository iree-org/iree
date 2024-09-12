// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Utils.h"

#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#define DEBUG_TYPE "iree-llvmcpu-utils"

namespace mlir::iree_compiler {

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
  return hasFeature(targetAttr, "+sve") || hasFeature(targetAttr, "+sve2") ||
         hasFeature(targetAttr, "+v9a");
}

bool hasSMEFeature(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return hasFeature(targetAttr, "+sme");
}

bool hasI8mmFeature(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return hasFeature(targetAttr, "+i8mm");
}

bool isLinalgGeneric2DTranspose(linalg::GenericOp genericOp) {
  // Check op has 2 dimensions.
  if (genericOp.getNumLoops() != 2)
    return false;

  // Check op has single input and output.
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1)
    return false;

  // Check all iterators are parallel.
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops())
    return false;

  // Check that the two indexing maps are a permutation of each other.
  SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
  bool isTranspose =
      (indexingMaps[0].isPermutation() && indexingMaps[1].isIdentity()) ||
      (indexingMaps[1].isPermutation() && indexingMaps[0].isIdentity());
  if (!isTranspose)
    return false;

  // Make sure the region only contains a yield op.
  Block &body = genericOp.getRegion().front();
  if (!llvm::hasSingleElement(body))
    return false;

  auto yieldOp = cast<linalg::YieldOp>(body.getTerminator());

  // The yield op should return the block argument corresponding to the input.
  auto yieldArg = dyn_cast<BlockArgument>(yieldOp.getValues()[0]);
  if (!yieldArg || yieldArg.getArgNumber() != 0 || yieldArg.getOwner() != &body)
    return false;

  return true;
}

bool mayHaveUndefinedBehaviorInMasking(Operation *op) {
  // Those operations will be lowered to division or related instructions,
  // and they might result in divide-by-zero.
  if (isa<mlir::arith::RemSIOp, mlir::arith::RemUIOp, mlir::arith::DivSIOp,
          mlir::arith::DivUIOp, mlir::arith::CeilDivSIOp,
          mlir::arith::CeilDivUIOp, mlir::arith::FloorDivSIOp,
          mlir::arith::DivFOp, mlir::arith::RemFOp>(op)) {
    return true;
  }
  return false;
}

} // namespace mlir::iree_compiler
