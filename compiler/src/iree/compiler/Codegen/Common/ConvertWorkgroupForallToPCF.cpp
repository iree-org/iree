// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTWORKGROUPFORALLTOPCFPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
struct ConvertWorkgroupForall final : OpRewritePattern<scf::ForallOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(scf::ForallOp op,
                                PatternRewriter &rewriter) const override;
};

/// Folds an scf.forall with split-reduction mapping containing a pcf.loop
/// with workgroup scope into a single pcf.generic. This handles the "split-k"
/// pattern where the outer forall represents additional parallelism from
/// K-dimension splitting and the inner pcf.loop represents the original
/// workgroup-level iteration.
struct FoldSplitKWorkgroupLoop : OpRewritePattern<scf::ForallOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(scf::ForallOp op,
                                PatternRewriter &rewriter) const override;
};

struct ConvertWorkgroupForallToPCFPass final
    : impl::ConvertWorkgroupForallToPCFPassBase<
          ConvertWorkgroupForallToPCFPass> {
  void runOnOperation() override;
  using Base::Base;
};

} // namespace

LogicalResult
ConvertWorkgroupForall::matchAndRewrite(scf::ForallOp op,
                                        PatternRewriter &rewriter) const {
  ArrayAttr mappingAttr = op.getMappingAttr();
  if (!mappingAttr || mappingAttr.empty() ||
      !llvm::all_of(mappingAttr,
                    llvm::IsaPred<IREE::Codegen::WorkgroupMappingAttr>)) {
    return failure();
  }
  // Linearize all ids down to 1 so that in cases when there are multiple
  // scf.foralls with incompatible delinearization bases. This technically
  // may be a small pessimization in very specific static cases, so if someone
  // ever finds they care they can try doing the analysis here to figure out
  // when it's ok not to linearize.
  //
  // Interface is implemented via external models hence the cast.
  auto scope = cast<IREE::PCF::ScopeAttrInterface>(
      IREE::Codegen::WorkgroupScopeAttr::get(rewriter.getContext(),
                                             /*linearize=*/true));
  FailureOr<IREE::PCF::LoopOp> res =
      convertForallToPCFLoop(rewriter, op, scope, 1);
  if (failed(res)) {
    return failure();
  }

  // Create a workgroup count hint to launch all workgroups along x.
  auto counts = llvm::to_vector_of<OpFoldResult>(res->getCount());
  rewriter.setInsertionPoint(*res);
  [[maybe_unused]] LogicalResult hintRes = createWorkgroupCountHint(
      rewriter, res->getLoc(), counts, /*maxWorkgroupParallelDims=*/1,
      /*reverse=*/false);
  assert(succeeded(hintRes) &&
         "Unexpected failure to construct workgroup count hint");
  rewriter.replaceOp(op, res->getResults());
  return success();
}

LogicalResult
FoldSplitKWorkgroupLoop::matchAndRewrite(scf::ForallOp op,
                                         PatternRewriter &rewriter) const {
  // Match scf.forall with split-reduction mapping.
  if (!forallOpHasMappingType<IREE::LinalgExt::SplitReductionMappingAttr>(op)) {
    return failure();
  }

  // Find pcf.loop with workgroup scope in the forall body.
  IREE::PCF::LoopOp loopOp = nullptr;
  for (Operation &bodyOp : op.getBody()->without_terminator()) {
    if (auto loop = dyn_cast<IREE::PCF::LoopOp>(&bodyOp)) {
      if (isa<IREE::Codegen::WorkgroupScopeAttr>(loop.getScope())) {
        loopOp = loop;
        break;
      }
    }
  }
  if (!loopOp) {
    return failure();
  }

  // Capture values needed for workgroup count computation before folding.
  // The fold erases the forall op, so any mixed bounds/steps must be
  // materialized up front.
  Location loc = op.getLoc();
  SmallVector<OpFoldResult> lowerBounds = op.getMixedLowerBound();
  SmallVector<OpFoldResult> upperBounds = op.getMixedUpperBound();
  SmallVector<OpFoldResult> steps = op.getMixedStep();
  SmallVector<Value> loopCounts(loopOp.getCount());

  // Fold forall + pcf.loop into pcf.generic.
  FailureOr<IREE::PCF::GenericOp> result =
      IREE::PCF::foldForallIntoPCFLoop(rewriter, op);
  if (failed(result)) {
    return failure();
  }

  // Compute total workgroup count after folding (forall iterations * loop
  // count). Generate IR before the pcf.generic.
  rewriter.setInsertionPoint(*result);

  AffineExpr s0, s1, s2;
  bindSymbols(rewriter.getContext(), s0, s1, s2);
  AffineExpr numItersExpr = (s0 - s1).ceilDiv(s2);

  Value forallCount = nullptr;
  for (int64_t i = 0, e = upperBounds.size(); i < e; ++i) {
    OpFoldResult lb = i < (int64_t)lowerBounds.size()
                          ? lowerBounds[i]
                          : rewriter.getIndexAttr(0);
    OpFoldResult ub = upperBounds[i];
    OpFoldResult step =
        i < (int64_t)steps.size() ? steps[i] : rewriter.getIndexAttr(1);

    Value iterCount = getValueOrCreateConstantIndexOp(
        rewriter, loc,
        affine::makeComposedFoldedAffineApply(rewriter, loc, numItersExpr,
                                              {ub, lb, step}));
    if (!forallCount) {
      forallCount = iterCount;
    } else {
      forallCount =
          arith::MulIOp::create(rewriter, loc, forallCount, iterCount);
    }
  }

  Value totalLoopCount = nullptr;
  for (Value count : loopCounts) {
    if (!totalLoopCount) {
      totalLoopCount = count;
    } else {
      totalLoopCount =
          arith::MulIOp::create(rewriter, loc, totalLoopCount, count);
    }
  }

  Value totalCount =
      arith::MulIOp::create(rewriter, loc, forallCount, totalLoopCount);

  // Create workgroup count hint for the folded generic.
  SmallVector<OpFoldResult> counts = {OpFoldResult(totalCount)};
  [[maybe_unused]] LogicalResult hintRes = createWorkgroupCountHint(
      rewriter, loc, counts, /*maxWorkgroupParallelDims=*/1,
      /*reverse=*/false);
  assert(succeeded(hintRes) &&
         "Unexpected failure to construct workgroup count hint");

  return success();
}

void ConvertWorkgroupForallToPCFPass::runOnOperation() {
  // First pass: Convert workgroup foralls to pcf.loop.
  {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertWorkgroupForall>(&getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }

  // Second pass: Fold split-k foralls containing pcf.loop into pcf.generic.
  {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldSplitKWorkgroupLoop>(&getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
}

} // namespace mlir::iree_compiler
