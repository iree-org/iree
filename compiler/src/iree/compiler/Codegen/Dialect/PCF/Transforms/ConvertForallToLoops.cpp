// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "iree/compiler/Utils/RewriteUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"

#define DEBUG_TYPE "iree-pcf-convert-forall-to-loops"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_CONVERTFORALLTOLOOPSPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

struct ConvertForallToLoopsPass final
    : impl::ConvertForallToLoopsPassBase<ConvertForallToLoopsPass> {
  void runOnOperation() override;
};

void ConvertForallToLoopsPass::runOnOperation() {
  // Collect all mapping-less forall ops to convert to sequential pcf.loop ops.
  SmallVector<scf::ForallOp> opsToConvert;
  getOperation()->walk([&](scf::ForallOp forallOp) {
    std::optional<ArrayAttr> mapping = forallOp.getMapping();
    if (!mapping || mapping->empty()) {
      opsToConvert.push_back(forallOp);
    }
  });

  IRRewriter rewriter(getOperation());
  PCF::ScopeAttrInterface sequentialScope =
      PCF::SequentialAttr::get(&getContext());
  for (auto forallOp : opsToConvert) {
    rewriter.setInsertionPoint(forallOp);
    if (failed(convertForallToPCF(rewriter, forallOp, sequentialScope))) {
      forallOp->emitOpError("failed to convert forall");
      return signalPassFailure();
    }
  }
}

static SmallVector<int64_t> getMappingPermutation(ArrayAttr mapping) {
  auto mappingRange = mapping.getAsRange<DeviceMappingAttrInterface>();
  int64_t mappingBase =
      cast<DeviceMappingAttrInterface>(
          *std::min_element(mappingRange.begin(), mappingRange.end(),
                            [](auto a, auto b) {
                              return a.getMappingId() < b.getMappingId();
                            }))
          .getMappingId();
  return llvm::map_to_vector(
      mappingRange, [&](auto a) { return a.getMappingId() - mappingBase; });
}

static FailureOr<SmallVector<int64_t>>
getProcessorIdPermutation(scf::ForallOp forallOp) {
  std::optional<ArrayAttr> mappingAttr = forallOp.getMapping();
  // Unspecified mappings indicate sequential foralls which we can choose the
  // iteration order for.
  if (!mappingAttr) {
    return llvm::to_vector(llvm::reverse(llvm::seq(forallOp.getRank())));
  }
  // Empty mappings are unsupported at the moment. It's unclear when a forall
  // with an empty mapping would be useful or important.
  if (mappingAttr.value().empty()) {
    return {};
  }

  SmallVector<int64_t> perm = getMappingPermutation(mappingAttr.value());
  if (!isPermutationVector(perm)) {
    return failure();
  }
  return perm;
}

LogicalResult matchForallConversion(scf::ForallOp forallOp) {
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

  for (BlockArgument bbArg : forallOp.getRegionIterArgs()) {
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
  if (failed(getProcessorIdPermutation(forallOp))) {
    return failure();
  }
  return success();
}

PCF::LoopOp convertForallToPCFImpl(RewriterBase &rewriter,
                                   scf::ForallOp forallOp,
                                   PCF::ScopeAttrInterface scope,
                                   int64_t numIds) {
  assert(succeeded(matchForallConversion(forallOp)) &&
         "converting unsupported forall op");

  // Maps from fastest -> slowest to current order.
  SmallVector<int64_t> perm = *getProcessorIdPermutation(forallOp);

  // Maps from current order to fastest -> slowest.
  SmallVector<int64_t> invPerm = invertPermutationVector(perm);

  // Get the permuted ubs/lbs/steps and save them for later since we need them
  // to reconstruct the correct ids.
  SmallVector<OpFoldResult> mixedUbs = forallOp.getMixedUpperBound();
  applyPermutationToVector(mixedUbs, invPerm);
  SmallVector<OpFoldResult> mixedLbs = forallOp.getMixedLowerBound();
  applyPermutationToVector(mixedLbs, invPerm);
  SmallVector<OpFoldResult> mixedStep = forallOp.getMixedStep();
  applyPermutationToVector(mixedStep, invPerm);
  // Permute the ivs of the body back to the original order by permuting the
  // uses before we move it over to the new op.
  permuteValues(rewriter, forallOp.getLoc(), forallOp.getInductionVars(), perm);

  scf::InParallelOp terminator = forallOp.getTerminator();
  MutableArrayRef<BlockArgument> bodySharedOuts = forallOp.getRegionIterArgs();

  // Replace non-insert slice users outside of the `scf.forall.in_parallel` with
  // the init values.
  ValueRange inits = forallOp.getDpsInits();
  for (auto [init, bbArg] : llvm::zip_equal(inits, bodySharedOuts)) {
    rewriter.replaceUsesWithIf(bbArg, init, [&](OpOperand &use) {
      return use.getOwner()->getParentOp() != terminator;
    });
  }

  Location loc = forallOp.getLoc();

  AffineExpr s0, s1, s2;
  bindSymbols(rewriter.getContext(), s0, s1, s2);
  AffineExpr numIters = (s0 - s1).ceilDiv(s2);
  SmallVector<Value> iterationCounts;
  for (auto [ub, lb, step] : llvm::zip_equal(mixedUbs, mixedLbs, mixedStep)) {
    OpFoldResult iterCount = affine::makeComposedFoldedAffineApply(
        rewriter, loc, numIters, ArrayRef<OpFoldResult>{ub, lb, step});
    iterationCounts.push_back(
        getValueOrCreateConstantIndexOp(rewriter, loc, iterCount));
  }

  int64_t numDelinIds = numIds > 0 && iterationCounts.size() > numIds
                            ? iterationCounts.size() - numIds + 1
                            : 0;
  // Reverse the delinearization basis because affine.delinearize_index is from
  // slowest to fastest varying.
  SmallVector<Value> delinearizationBasis(
      llvm::reverse(ArrayRef<Value>(iterationCounts).take_back(numDelinIds)));
  if (!delinearizationBasis.empty()) {
    AffineExpr mul = s0 * s1;
    Value total = delinearizationBasis.front();
    total = std::accumulate(
        delinearizationBasis.begin() + 1, delinearizationBasis.end(), total,
        [&](Value l, Value r) {
          OpFoldResult acc =
              affine::makeComposedFoldedAffineApply(rewriter, loc, mul, {l, r});
          return getValueOrCreateConstantIndexOp(rewriter, loc, acc);
        });
    // Replace the first |numDelinIds| entries with their product.
    iterationCounts.erase(iterationCounts.end() - numDelinIds,
                          iterationCounts.end());
    iterationCounts.push_back(total);
  }

  auto loopOp = PCF::LoopOp::create(rewriter, loc, scope, iterationCounts,
                                    forallOp.getDpsInits());
  SmallVector<Value> argReplacements;
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(loopOp.getBody());
    // If/else here to avoid popping off the front of a vector.
    if (!delinearizationBasis.empty()) {
      auto delin = affine::AffineDelinearizeIndexOp::create(
          rewriter, loc, loopOp.getIdArgs().back(), delinearizationBasis,
          /*hasOuterBound=*/true);
      argReplacements.append(loopOp.getIdArgs().begin(),
                             loopOp.getIdArgs().end() - 1);
      // Add replacements in reverse to get fastest -> slowest.
      auto delinResultsReverse = llvm::reverse(delin.getResults());
      argReplacements.append(delinResultsReverse.begin(),
                             delinResultsReverse.end());
    } else {
      argReplacements.append(loopOp.getIdArgs().begin(),
                             loopOp.getIdArgs().end());
    }

    // id * step + lb.
    AffineExpr applyLbAndStep = s0 * s1 + s2;
    for (auto [id, step, lb] :
         llvm::zip_equal(argReplacements, mixedStep, mixedLbs)) {
      OpFoldResult newId = affine::makeComposedFoldedAffineApply(
          rewriter, loc, applyLbAndStep, {id, step, lb});
      id = getValueOrCreateConstantIndexOp(rewriter, loc, newId);
    }
  }

  // Add parent only sync scope to the body arg types.
  Attribute syncScope = PCF::SyncOnReturnAttr::get(rewriter.getContext());
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

  // Use the inits as the replacements for the shared outs bbargs to appease
  // `inlineBlockBefore`. By this point all of their users have been replaced
  // or erased so it doesn't matter what goes here.
  argReplacements.append(inits.begin(), inits.end());
  rewriter.inlineBlockBefore(forallOp.getBody(), loopOp.getBody(),
                             loopOp.getBody()->end(), argReplacements);

  rewriter.replaceOp(forallOp, loopOp);
  return loopOp;
}

} // namespace

//===---------------------------------------------------------------------===//
// scf.forall -> pcf.loop
//===---------------------------------------------------------------------===//

FailureOr<PCF::LoopOp> convertForallToPCF(RewriterBase &rewriter,
                                          scf::ForallOp forallOp,
                                          PCF::ScopeAttrInterface scope,
                                          int64_t numIds) {
  if (failed(matchForallConversion(forallOp))) {
    return failure();
  }
  return convertForallToPCFImpl(rewriter, forallOp, scope, numIds);
}

} // namespace mlir::iree_compiler::IREE::PCF
