// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/ConversionDialectInterface.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "iree/compiler/Utils/RewriteUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-pcf-lower-structural-pcf"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_LOWERSTRUCTURALPCFPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

class LoadDependentDialectExtension : public DialectExtensionBase {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoadDependentDialectExtension)

  LoadDependentDialectExtension() : DialectExtensionBase(/*dialectNames=*/{}) {}

  void apply(MLIRContext *context,
             MutableArrayRef<Dialect *> dialects) const final {
    for (Dialect *dialect : dialects) {
      auto *iface = dyn_cast<PCFConversionDialectInterface>(dialect);
      if (!iface) {
        continue;
      }
      iface->loadStructuralLoweringDependentDialects(context);
    }
  }

  /// Return a copy of this extension.
  std::unique_ptr<DialectExtensionBase> clone() const final {
    return std::make_unique<LoadDependentDialectExtension>(*this);
  }
};

struct LowerStructuralPCFPass final
    : impl::LowerStructuralPCFPassBase<LowerStructuralPCFPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    // Direct dialect deps.
    registry.insert<iree_compiler::IREE::PCF::PCFDialect, scf::SCFDialect,
                    cf::ControlFlowDialect>();
    registry.addExtensions<LoadDependentDialectExtension>();
  }
  void runOnOperation() override;
};

struct LowerGenericOp : public OpRewritePattern<IREE::PCF::GenericOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::PCF::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (genericOp->getNumResults() != 0) {
      return rewriter.notifyMatchFailure(
          genericOp, "unexpected attempt to resolve generic op with results");
    }

    if (genericOp.getSyncOnReturn()) {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointAfter(genericOp);
      if (failed(genericOp.getScope().addBarrier(rewriter))) {
        genericOp.emitOpError("failed to construct requested barrier");
        return failure();
      }
    }

    Location loc = genericOp.getLoc();
    SmallVector<Value> indexArgs = genericOp.getScope().getWorkerIDs(
        rewriter, loc, genericOp.getNumIterators());
    assert(indexArgs.size() == genericOp.getNumIterators() &&
           "expected num worker ids to match number of iterators");

    indexArgs.append(genericOp.getScope().getWorkerCounts(
        rewriter, loc, genericOp.getNumIterators()));
    assert(indexArgs.size() == genericOp.getNumIndexArgs() &&
           "expected num worker counts to match number of iterators");

    auto executeRegion =
        scf::ExecuteRegionOp::create(rewriter, loc, TypeRange());
    Block *entryBlock = &genericOp.getRegion().front();
    Block *newEntry = rewriter.createBlock(&executeRegion.getRegion(),
                                           executeRegion.getRegion().begin());
    // Inline the entry block into the new region.
    rewriter.inlineBlockBefore(entryBlock, newEntry, newEntry->begin(),
                               indexArgs);

    // Move the remaining blocks into the new region.
    SmallVector<Block *> blocksToMove = llvm::map_to_vector(
        genericOp.getRegion().getBlocks(), [](Block &b) { return &b; });
    for (Block *block : blocksToMove) {
      rewriter.moveBlockBefore(block, &executeRegion.getRegion(),
                               executeRegion.getRegion().end());
    }

    // Replace all pcf.return ops with scf.yield ops. We make this pattern
    // responsible for terminator conversion to try to keep IR valid between
    // pattern rewrites.
    for (Block &block : executeRegion.getRegion()) {
      auto pcfTerminator = dyn_cast<PCF::ReturnOp>(block.getTerminator());
      if (pcfTerminator) {
        rewriter.setInsertionPoint(pcfTerminator);
        rewriter.replaceOpWithNewOp<scf::YieldOp>(pcfTerminator);
      }
    }

    rewriter.eraseOp(genericOp);

    return success();
  }
};

struct LowerLoopOp final : OpRewritePattern<IREE::PCF::LoopOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::PCF::LoopOp loopOp,
                                PatternRewriter &rewriter) const override {
    if (loopOp->getNumResults() != 0) {
      return rewriter.notifyMatchFailure(
          loopOp, "unexpected attempt to resolve loop op with results");
    }

    if (loopOp.getSyncOnReturn()) {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointAfter(loopOp);
      if (failed(loopOp.getScope().addBarrier(rewriter))) {
        loopOp.emitOpError("failed to construct requested barrier");
        return failure();
      }
    }

    Location loc = loopOp.getLoc();
    SmallVector<Value> workerCounts =
        loopOp.getScope().getWorkerCounts(rewriter, loc, loopOp.getNumIdArgs());
    SmallVector<Value> ids =
        loopOp.getScope().getWorkerIDs(rewriter, loc, loopOp.getNumIdArgs());
    auto pcfTerminator = cast<PCF::ReturnOp>(loopOp.getBody()->getTerminator());

    assert(workerCounts.size() == loopOp.getNumIdArgs() &&
           "expected worker count to match number of id args");
    assert(ids.size() == loopOp.getNumIdArgs() &&
           "expected worker id count to match number of id args");

    // Create an scf.forall op with no mapping. This is moderately more
    // expressive than scf.for because it indicates that loop iterations
    // are independent of one another.
    //
    // `scf.forall` orders its loop parameters from slowest to fasted varying,
    // so reverse lbs/ubs/steps.
    SmallVector<OpFoldResult> lbs =
        llvm::to_vector_of<OpFoldResult>(reverse(ids));
    SmallVector<OpFoldResult> steps =
        llvm::to_vector_of<OpFoldResult>(reverse(workerCounts));
    SmallVector<OpFoldResult> ubs(llvm::reverse(loopOp.getCount()));
    auto forallOp = scf::ForallOp::create(
        rewriter, loc, lbs, ubs, steps, ValueRange(), /*mapping=*/std::nullopt);
    // We can take the body of the loop op directly since the index body args
    // already represent the tile ids.
    forallOp.getRegion().takeBody(loopOp.getRegion());
    Block *body = forallOp.getBody();

    // Since forall loop ids are ordered from slowest to fastest varying we need
    // to reverse the ordering of their uses.
    SmallVector<int64_t> reversePerm = llvm::to_vector(
        llvm::reverse(llvm::seq<int64_t>(body->getNumArguments())));
    permuteValues(rewriter, loc, body->getArguments(), reversePerm);

    // Replace pcf.return terminator with scf.forall.in_parallel.
    rewriter.setInsertionPoint(pcfTerminator);
    rewriter.replaceOpWithNewOp<scf::InParallelOp>(pcfTerminator);

    // Erase the old loop op.
    rewriter.eraseOp(loopOp);

    return success();
  }
};

static bool hasReturnOnlyBody(Block &b) {
  return llvm::hasSingleElement(b.getOperations()) &&
         isa<PCF::ReturnOp>(&b.getOperations().back());
}

static Block *getOrCreateReturnBlock(RewriterBase &rewriter, Location loc,
                                     Region *region, TypeRange argTypes) {
  assert(!region->getBlocks().empty() && "unexpected empty region");
  Block &endBlock = region->getBlocks().back();
  if (endBlock.getArgumentTypes() == argTypes && hasReturnOnlyBody(endBlock)) {
    return &endBlock;
  }

  // Insertion guard back to the original point. Creating a block sets the
  // insertion point to the end of the current block.
  OpBuilder::InsertionGuard g(rewriter);
  Block *newBlock =
      rewriter.createBlock(region, region->end(), argTypes,
                           SmallVector<Location>(argTypes.size(), loc));
  PCF::ReturnOp::create(rewriter, loc);
  return newBlock;
}

struct LowerBranchCondReturnOp final
    : OpRewritePattern<IREE::PCF::BranchCondReturnOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::PCF::BranchCondReturnOp branchOp,
                                PatternRewriter &rewriter) const override {
    Region *parentRegion = branchOp->getParentRegion();
    Block *returnBlock =
        getOrCreateReturnBlock(rewriter, branchOp.getLoc(), parentRegion,
                               branchOp.getDestOperands().getTypes());
    rewriter.setInsertionPoint(branchOp);
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        branchOp, branchOp.getCondition(), returnBlock,
        branchOp.getDestOperands(), branchOp.getDest(),
        branchOp.getDestOperands());
    return success();
  }
};

WalkResult verifyOperationLegality(Operation *op) {
  if (isa<PCF::GenericOp, PCF::LoopOp, PCF::BranchCondReturnOp>(op)) {
    return WalkResult::interrupt();
  }
  return WalkResult::advance();
}

void LowerStructuralPCFPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<LowerGenericOp, LowerLoopOp, LowerBranchCondReturnOp>(
      &getContext());

  // WalkAndApplyPatterns walks post-order, meaning children can safely assume
  // parents have not yet been converted and pull context from them if needed.
  walkAndApplyPatterns(getOperation(), std::move(patterns));

  if (getOperation()->walk(verifyOperationLegality).wasInterrupted()) {
    return signalPassFailure();
  }
}

} // namespace

} // namespace mlir::iree_compiler::IREE::PCF
