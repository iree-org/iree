// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#define DEBUG_TYPE "iree-codegen-gpu-transforms"

namespace mlir::iree_compiler::IREE::PCF {

//===---------------------------------------------------------------------===//
// Forall Fusion
//===---------------------------------------------------------------------===//

FailureOr<PCF::LoopOp> convertForallToPCF(RewriterBase &rewriter,
                                          scf::ForallOp forallOp,
                                          PCF::ScopeAttr scope) {
  // TODO: Support non-normalized loops.
  if (!forallOp.isNormalized()) {
    return failure();
  }

  scf::InParallelOp terminator = forallOp.getTerminator();
  for (Operation &op : terminator.getBody()->getOperations()) {
    // Bail on terminator ops other than parallel insert slice since we don't
    // know how to convert it.
    auto insertSliceOp = dyn_cast<tensor::ParallelInsertSliceOp>(&op);
    if (!insertSliceOp) {
      return failure();
    }

    // Bail on non-shared outs destinations.
    auto bbArgDest = dyn_cast<BlockArgument>(insertSliceOp.getDest());
    if (!bbArgDest || bbArgDest.getOwner()->getParentOp() != forallOp) {
      return failure();
    }
  }

  MutableArrayRef<BlockArgument> bodySharedOuts = forallOp.getRegionIterArgs();

  for (BlockArgument bbArg : bodySharedOuts) {
    for (OpOperand &use : bbArg.getUses()) {
      // Skip users outside of the terminator. These are replaced with the init.
      if (use.getOwner()->getParentOp() != terminator) {
        continue;
      }

      // Bail if the use is not on the dest of the insert slice.
      auto insertSliceUser =
          cast<tensor::ParallelInsertSliceOp>(use.getOwner());
      if (use != insertSliceUser.getDestMutable()) {
        return failure();
      }
    }
  }

  // Replace non-insert slice users outside of the `scf.forall.in_parallel` with
  // the init values.
  ValueRange inits = forallOp.getDpsInits();
  for (auto [init, bbArg] : llvm::zip_equal(inits, bodySharedOuts)) {
    rewriter.replaceUsesWithIf(bbArg, init, [&](OpOperand &use) {
      return use.getOwner()->getParentOp() != terminator;
    });
  }

  Location loc = forallOp.getLoc();

  SmallVector<Value> iterationCounts =
      llvm::map_to_vector(forallOp.getMixedUpperBound(), [&](OpFoldResult b) {
        return getValueOrCreateConstantIndexOp(rewriter, loc, b);
      });

  auto loopOp = PCF::LoopOp::create(rewriter, loc, scope, iterationCounts,
                                    forallOp.getDpsInits());

  // Add parent only sync scope to the body arg types.
  Attribute syncScope = PCF::SyncOnParentAttr::get(rewriter.getContext());
  for (auto regionRefArg : loopOp.getRegionRefArgs()) {
    auto srefType = cast<PCF::ShapedRefType>(regionRefArg.getType());
    auto newSrefType = PCF::ShapedRefType::get(
        rewriter.getContext(), srefType.getShape(), srefType.getElementType(),
        srefType.getScope(), syncScope);
    regionRefArg.setType(newSrefType);
  }

  rewriter.setInsertionPoint(terminator);
  llvm::SmallDenseMap<Value, Value> argToReplacementMap;
  for (auto [bbArg, refArg] :
       llvm::zip_equal(bodySharedOuts, loopOp.getRegionRefArgs())) {
    argToReplacementMap[bbArg] = refArg;
  }

  // Iterate the insert_slice ops in the order to retain the order of writes.
  SmallVector<tensor::ParallelInsertSliceOp> insertOps(
      terminator.getBody()->getOps<tensor::ParallelInsertSliceOp>());
  for (tensor::ParallelInsertSliceOp insertSliceOp : insertOps) {
    PCF::WriteSliceOp::create(
        rewriter, insertSliceOp.getLoc(), insertSliceOp.getSource(),
        argToReplacementMap[insertSliceOp.getDest()],
        insertSliceOp.getMixedOffsets(), insertSliceOp.getMixedSizes(),
        insertSliceOp.getMixedStrides());
    rewriter.eraseOp(insertSliceOp);
  }

  // Replace the terminator with the new terminator kind.
  rewriter.replaceOpWithNewOp<PCF::ReturnOp>(terminator);

  SmallVector<Value> argReplacements(loopOp.getIdArgs());
  // Use the inits as the replacements for the shared outs bbargs to appease
  // `inlineBlockBefore`. By this point all of their users have been replaced
  // or erased so it doesn't matter what goes here.
  argReplacements.append(inits.begin(), inits.end());
  rewriter.inlineBlockBefore(forallOp.getBody(), loopOp.getBody(),
                             loopOp.getBody()->begin(), argReplacements);

  rewriter.replaceOp(forallOp, loopOp);
  return loopOp;
}

} // namespace mlir::iree_compiler::IREE::PCF
