// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CommonExtensions.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgTransform/SimplePatternRewriter.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/ListenerCSE.h"
#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Interfaces/BufferizationInterfaces.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/IndexingUtils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE;

iree_compiler::IREE::transform_dialect::CommonExtensions::CommonExtensions() {
  registerTransformOps<
#define GET_OP_LIST
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensionsOps.cpp.inc"
      >();
}

void mlir::iree_compiler::registerTransformDialectCommonExtension(
    DialectRegistry &registry) {
  registry.addExtensions<
      mlir::iree_compiler::IREE::transform_dialect::CommonExtensions>();
}

// Return true if all the uses of op are either Store/transfer_write.
// There can be SubviewOp users as long as all its users are also
// StoreOp/transfer_write. If return true it also fills out the uses, if it
// returns false uses is unchanged.
static bool allUsesAreStores(Operation *op, std::vector<Operation *> &uses) {
  std::vector<Operation *> opUses;
  for (OpOperand &use : op->getUses()) {
    Operation *useOp = use.getOwner();
    if (isa<memref::DeallocOp, vector::TransferWriteOp, memref::StoreOp>(
            useOp) ||
        (isa<memref::SubViewOp>(useOp) && allUsesAreStores(useOp, opUses))) {
      opUses.push_back(useOp);
      continue;
    }
    return false;
  }
  uses.insert(uses.end(), opUses.begin(), opUses.end());
  return true;
}

// Track temporary allocations that are never read from. If this is the case
// it means both the allocations and associated stores can be removed.
static void eraseDeadAllocAndStores(RewriterBase &rewriter,
                                    Operation *parentOp) {
  std::vector<Operation *> opToErase;
  parentOp->walk([&](memref::AllocOp op) {
    if (allUsesAreStores(op, opToErase)) {
      opToErase.push_back(op.getOperation());
    }
  });
  for (Operation *op : opToErase) rewriter.eraseOp(op);
}

//===---------------------------------------------------------------------===//
// ApplyBufferOptimizationsOp
//===---------------------------------------------------------------------===//
DiagnosedSilenceableFailure
transform_dialect::ApplyBufferOptimizationsOp::applyToOne(
    Operation *target, transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  // Apply store to load forwarding and dead store elimination.
  IRRewriter rewriter(target->getContext());
  ErrorCheckingTrackingListener listener(state, *this);
  rewriter.setListener(&listener);
  vector::transferOpflowOpt(rewriter, target);
  eraseDeadAllocAndStores(rewriter, target);
  return listener.check(target->getLoc());
}

void transform_dialect::ApplyBufferOptimizationsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

void transform_dialect::ApplyBufferOptimizationsOp::build(
    OpBuilder &builder, OperationState &result, Value target) {
  result.addOperands(target);
}

//===---------------------------------------------------------------------===//
// ApplyPatternsOp
//===---------------------------------------------------------------------===//
void transform_dialect::ApplyPatternsOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    const ApplyPatternsOpPatterns &patterns) {
  result.addOperands(target);

  auto unitAttr = builder.getUnitAttr();

#define ADD_PATTERN(NAME, ATTR) \
  if (patterns.NAME)            \
    result.addAttribute(ApplyPatternsOp::ATTR(result.name), unitAttr);
  ///
  /// When touching something here, do not forget to update CommonExtensions.h.
  ///
  ADD_PATTERN(additionalIreePatterns, getAdditionalIreePatternsAttrName)
  ADD_PATTERN(bubbleCollapse, getBubbleCollapseAttrName)
  ADD_PATTERN(bubbleExpand, getBubbleExpandAttrName)
  ADD_PATTERN(bubblePackUnPack, getBubblePackUnPackAttrName)
  ADD_PATTERN(canonicalization, getCanonicalizationAttrName)
  ADD_PATTERN(cse, getCseAttrName)
  ADD_PATTERN(eraseUnnecessaryTensorOperands,
              getEraseUnnecessaryTensorOperandsAttrName)
  ADD_PATTERN(expandMemrefStridedMetadata,
              getExpandMemrefStridedMetadataAttrName)
  ADD_PATTERN(extractAddressComputations, getExtractAddressComputationsAttrName)
  ADD_PATTERN(foldMemrefAliases, getFoldMemrefAliasesAttrName)
  ADD_PATTERN(foldReassociativeReshapes, getFoldReassociativeReshapesAttrName)
  ADD_PATTERN(foldTensorEmptyExtract, getFoldTensorEmptyExtractAttrName)
  ADD_PATTERN(foldTensorSubsets, getFoldTensorSubsetsAttrName)
  ADD_PATTERN(licm, getLicmAttrName)
  ADD_PATTERN(linalgElementwiseGreedyFusion,
              getLinalgElementwiseGreedyFusionAttrName)
  ADD_PATTERN(lowerTransferOpPermutations,
              getLowerTransferOpPermutationsAttrName)
  ADD_PATTERN(lowerVectorMasks, getLowerVectorMasksAttrName)
  ADD_PATTERN(prepareVectorToMma, getPrepareVectorToMmaAttrName)
  ADD_PATTERN(rankReducingLinalg, getRankReducingLinalgAttrName)
  ADD_PATTERN(rankReducingLinalgViaReshapes,
              getRankReducingLinalgViaReshapesAttrName)
  ADD_PATTERN(rankReducingVector, getRankReducingVectorAttrName)
  ADD_PATTERN(swapPaddingElideConditional,
              getSwapPaddingElideConditionalAttrName)
  ADD_PATTERN(swappingPatterns, getSwappingPatternsAttrName)
  ADD_PATTERN(tilingCanonicalization, getTilingCanonicalizationAttrName)
  ADD_PATTERN(unrollVectorsGpuMmaSync, getUnrollVectorsGpuMmaSyncAttrName)
  ADD_PATTERN(unrollVectorsGpuWmma, getUnrollVectorsGpuWmmaAttrName)
#undef ADD_PATTERN
}

static void addOperands(Operation *op, SetVector<Value> &operandSet) {
  if (!op) return;
  TypeSwitch<Operation *, void>(op)
      .Case<linalg::LinalgOp>([&](linalg::LinalgOp linalgOp) {
        SmallVector<Value> inputOperands{linalgOp.getDpsInputOperands()};
        operandSet.insert(inputOperands.begin(), inputOperands.end());
      })
      .Default([&](Operation *operation) {
        operandSet.insert(operation->operand_begin(), operation->operand_end());
      });
}

template <int limit = 3>
static bool setFusedOpOperandLimit(OpOperand *fusedOperand) {
  Operation *producer = fusedOperand->get().getDefiningOp();
  if (!producer) return false;
  Operation *consumer = fusedOperand->getOwner();
  SetVector<Value> fusedOpOperands;
  if (producer->getNumResults() != 1) return false;
  addOperands(consumer, fusedOpOperands);
  fusedOpOperands.remove(producer->getResult(0));
  addOperands(producer, fusedOpOperands);
  return fusedOpOperands.size() <= limit;
}

namespace {
/// Rewrite a tensor.generate as an arith.constant when possible.
struct GenerateToConstant : public OpRewritePattern<tensor::GenerateOp> {
  using OpRewritePattern<tensor::GenerateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::GenerateOp generateOp,
                                PatternRewriter &rewriter) const final {
    auto tensorType = generateOp.getResult().getType().cast<RankedTensorType>();
    if (!tensorType.hasStaticShape()) return failure();
    auto terminatorOp =
        cast<tensor::YieldOp>(generateOp.getBody().front().getTerminator());
    if (terminatorOp->getNumOperands() > 1) return failure();
    auto constantOp =
        terminatorOp->getOperand(0).getDefiningOp<arith::ConstantOp>();
    if (!constantOp) return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        generateOp, tensorType,
        DenseElementsAttr::get(tensorType, constantOp.getValueAttr()));
    return success();
  }
};

/// Fold tensor.empty used by extract_slice if this the only use of
/// extract_slice and the result is static.
struct FoldTensorEmptyExtract
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractOp,
                                PatternRewriter &rewriter) const final {
    auto tensorEmpty = extractOp.getSource().getDefiningOp<tensor::EmptyOp>();
    if (!tensorEmpty || !extractOp.getType().hasStaticShape() ||
        !tensorEmpty->hasOneUse())
      return failure();
    rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
        extractOp, extractOp.getType().getShape(),
        extractOp.getType().getElementType());
    return success();
  }
};

/// Fold `tensor.pad(cst, tensor.extract*(linalg.fill(cst)))` into
/// `linalg.fill(cst, empty)` when the padding constant and the fill constant
/// are the same.
/// This seems generally desirable as a folding but may be too intrusive, so we
/// only apply it selectively for now.
// TODO: atm hardcoded on linalg.fill but we could take any result of any
// generic that yields a constant in that result.
struct FoldFillIntoPad : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const final {
    Operation *currentOp = padOp.getSource().getDefiningOp();
    auto maybeExtractSlice =
        dyn_cast_or_null<tensor::ExtractSliceOp>(currentOp);
    while (currentOp && maybeExtractSlice) {
      currentOp = maybeExtractSlice.getSource().getDefiningOp();
      maybeExtractSlice = dyn_cast_or_null<tensor::ExtractSliceOp>(currentOp);
    }
    auto fillOp = dyn_cast_or_null<linalg::FillOp>(currentOp);
    if (!fillOp) {
      return rewriter.notifyMatchFailure(
          padOp, "not coming from a linalg.fill op via tensor.extract_slice*");
    }

    Value padValue = padOp.getConstantPaddingValue();
    RankedTensorType resultType = padOp.getResultType();
    if (!padValue ||
        getAsOpFoldResult(padValue) !=
            getAsOpFoldResult(fillOp.getDpsInputOperand(0)->get())) {
      return rewriter.notifyMatchFailure(
          padOp, "not a constant value matching the fill value");
    }

    Location loc = padOp.getLoc();
    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        loc, resultType,
        linalg::createDynamicDimensions(rewriter, loc, padOp.getResult()));
    rewriter.replaceOpWithNewOp<linalg::FillOp>(padOp, padValue,
                                                emptyOp.getResult());

    return success();
  }
};
}  // namespace

static void addLowerTransferOpPermutationsPatterns(
    RewritePatternSet &patterns) {
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
}

static void addLowerVectorMasksPatterns(RewritePatternSet &patterns) {
  vector::populateVectorMaskLoweringPatternsForSideEffectingOps(patterns);
}

static void addExtractAddressComputationsPatterns(RewritePatternSet &patterns) {
  memref::populateExtractAddressComputationsPatterns(patterns);
}

static void addFoldMemrefAliasPatterns(RewritePatternSet &patterns) {
  memref::populateFoldMemRefAliasOpPatterns(patterns);
}

static void addFoldTensorEmptyExtract(RewritePatternSet &patterns) {
  patterns.add<FoldTensorEmptyExtract>(patterns.getContext());
}

static void addReassociativeReshapePatterns(RewritePatternSet &patterns) {
  tensor::populateReassociativeReshapeFoldingPatterns(patterns);
}

static void addFoldTensorSubsetsPatterns(RewritePatternSet &patterns) {
  tensor::populateFoldTensorSubsetOpPatterns(patterns);
  // TODO: upstream should move these to populateFoldTensorSubsetOpPatterns.
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
}

static void addEraseUnnecessaryTensorOperandsPatterns(
    RewritePatternSet &patterns) {
  linalg::populateEraseUnnecessaryInputsPatterns(patterns);
}

static void addPrepareVectorToMmaPatterns(RewritePatternSet &patterns) {
  populatePrepareVectorToMMAPatterns(patterns, /*useNvGpu=*/true);
}

static void addRankReducingLinalgPatterns(RewritePatternSet &patterns) {
  populateReshapeToInterfaceTensorPatterns(patterns);
  linalg::populateFoldUnitExtentDimsViaSlicesPatterns(patterns);
}

static void addRankReducingLinalgViaReshapesPatterns(
    RewritePatternSet &patterns) {
  populateReshapeToInterfaceTensorPatterns(patterns);
  linalg::populateFoldUnitExtentDimsViaReshapesPatterns(patterns);
}

static void addRankReducingVectorPatterns(RewritePatternSet &patterns) {
  vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
}

static void addSwappingPatterns(RewritePatternSet &patterns,
                                bool swapPaddingElideCornerCase) {
  patterns.add<linalg::ExtractSliceOfPadTensorSwapPattern>(
      patterns.getContext(),
      [&](tensor::ExtractSliceOp) -> std::optional<bool> {
        return !swapPaddingElideCornerCase;
      });
}

static void addTilingCanonicalizationPatterns(RewritePatternSet &patterns) {
  linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  /// This seems generally desirable as a folding but may be too intrusive, so
  /// we only apply it selectively for now.
  patterns.add<FoldFillIntoPad>(patterns.getContext());
}

static std::optional<SmallVector<int64_t>>
getGPUTensorCoreNativeMmaSyncVectorSize(Operation *op) {
  return getMmaNativeVectorSize(op);
}

static void addUnrollVectorsGpuMmaSyncPatterns(RewritePatternSet &patterns) {
  auto unrollOrder = [](Operation *op) -> std::optional<SmallVector<int64_t>> {
    auto contract = dyn_cast<vector::ContractionOp>(op);
    if (!contract) return std::nullopt;
    return mlir::iree_compiler::gpuMmaUnrollOrder(contract);
  };
  vector::populateVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShapeFn(getGPUTensorCoreNativeMmaSyncVectorSize)
                    .setUnrollTraversalOrderFn(unrollOrder));
}

static std::optional<SmallVector<int64_t>> getGPUTensorCoreNativeWmmaVectorSize(
    Operation *op) {
  return getWmmaNativeVectorSize(op);
}

static void addUnrollVectorsGpuWmmaPatterns(RewritePatternSet &patterns) {
  auto unrollOrder = [](Operation *op) -> std::optional<SmallVector<int64_t>> {
    auto contract = dyn_cast<vector::ContractionOp>(op);
    if (!contract) return std::nullopt;
    return mlir::iree_compiler::gpuMmaUnrollOrder(contract);
  };
  vector::populateVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShapeFn(getGPUTensorCoreNativeWmmaVectorSize)
                    .setUnrollTraversalOrderFn(unrollOrder));
}

static void addAdditionalIreePatterns(RewritePatternSet &patterns) {
  patterns.add<GenerateToConstant>(patterns.getContext());
}

static void addAllRegisteredCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  for (Dialect *dialect : ctx->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (RegisteredOperationName op : ctx->getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, ctx);
}

DiagnosedSilenceableFailure transform_dialect::ApplyPatternsOp::applyToOne(
    Operation *target, transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    return mlir::emitDefiniteFailure(
        target,
        "applies only to isolated-from-above targets because it needs to apply "
        "patterns greedily");
  }
  MLIRContext *ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  if (getAdditionalIreePatterns()) addAdditionalIreePatterns(patterns);
  if (getBubbleCollapse()) {
    linalg::populateFoldReshapeOpsByCollapsingPatterns(
        patterns, [](OpOperand *) { return true; });
  }
  if (getBubbleExpand()) {
    linalg::populateFoldReshapeOpsByExpansionPatterns(
        patterns, [](OpOperand *) { return true; });
  }
  if (getBubblePackUnPack())
    linalg::populateDataLayoutPropagationPatterns(
        patterns, [](Operation *op) { return true; });
  if (getCanonicalization()) addAllRegisteredCanonicalizationPatterns(patterns);
  if (getEraseUnnecessaryTensorOperands())
    addEraseUnnecessaryTensorOperandsPatterns(patterns);
  if (getExpandMemrefStridedMetadata())
    memref::populateExpandStridedMetadataPatterns(patterns);
  if (getExtractAddressComputations())
    addExtractAddressComputationsPatterns(patterns);
  if (getFoldMemrefAliases()) addFoldMemrefAliasPatterns(patterns);
  if (getFoldReassociativeReshapes()) addReassociativeReshapePatterns(patterns);
  if (getFoldTensorEmptyExtract()) addFoldTensorEmptyExtract(patterns);
  if (getFoldTensorSubsets()) addFoldTensorSubsetsPatterns(patterns);
  if (getLinalgElementwiseGreedyFusion())
    linalg::populateElementwiseOpsFusionPatterns(patterns,
                                                 setFusedOpOperandLimit<3>);
  if (getLowerTransferOpPermutations())
    addLowerTransferOpPermutationsPatterns(patterns);
  if (getLowerVectorMasks()) addLowerVectorMasksPatterns(patterns);
  if (getPrepareVectorToMma()) addPrepareVectorToMmaPatterns(patterns);
  if (getRankReducingLinalg()) addRankReducingLinalgPatterns(patterns);
  if (getRankReducingLinalgViaReshapes())
    addRankReducingLinalgViaReshapesPatterns(patterns);
  if (getRankReducingVector()) addRankReducingVectorPatterns(patterns);
  if (getSwappingPatterns())
    addSwappingPatterns(patterns, getSwapPaddingElideConditional());
  if (getTilingCanonicalization()) addTilingCanonicalizationPatterns(patterns);
  if (getUnrollVectorsGpuMmaSync())
    addUnrollVectorsGpuMmaSyncPatterns(patterns);
  if (getUnrollVectorsGpuWmma()) addUnrollVectorsGpuWmmaPatterns(patterns);

  Location loc = target->getLoc();
  ErrorCheckingTrackingListener listener(state, *this);
  GreedyRewriteConfig config;
  config.listener = &listener;
  // Manually gather list of ops because the other GreedyPatternRewriteDriver
  // overloads only accepts ops that are isolated from above.
  SmallVector<Operation *> ops;
  target->walk([&](Operation *nestedOp) {
    if (target != nestedOp) ops.push_back(nestedOp);
  });
  LogicalResult result =
      applyOpPatternsAndFold(ops, std::move(patterns), config);
  if (failed(result)) {
    return listener.check(
        loc, mlir::emitDefiniteFailure(target, "greedy patterns failed"));
  }

  auto diag = listener.check(loc);
  if (!diag.succeeded()) return diag;

  if (getLicm()) {
    target->walk([&](func::FuncOp funcOp) {
      // This assumes LICM never removes operations so we don't need tracking.
      // TODO: confirm / revisit this assumption and plumb a rewriter through
      // upstream moveLoopInvariantCode if necessary.
      funcOp->walk([](LoopLikeOpInterface loopLike) {
        moveLoopInvariantCode(loopLike);
      });
      // For now, put single loop promotion as part of licm. Underlying
      // implementations perform splice operations which shouldn't need
      // tracking.
      // TODO: confirm / revisit this assumption and plumb a rewriter through
      // upstream moveLoopInvariantCode if necessary.
      funcOp->walk([](Operation *op) {
        (void)llvm::TypeSwitch<Operation *, LogicalResult>(op)
            .Case<affine::AffineForOp, scf::ForOp>(
                [](auto loop) { return promoteIfSingleIteration(loop); })
            .Default([](Operation *) { return success(); });
      });
    });
  }

  if (getCse()) {
    func::FuncOp lastFuncVisited;
    auto walkResult = target->walk([&](func::FuncOp funcOp) -> WalkResult {
      lastFuncVisited = funcOp;
      result =
          eliminateCommonSubexpressions(funcOp, /*domInfo=*/nullptr, &listener);
      if (failed(result)) return WalkResult::interrupt();
      if (failed(listener.checkErrorState())) return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      if (failed(result)) {
        return mlir::emitDefiniteFailure(lastFuncVisited,
                                         "greedy patterns failed");
      }
      if (failed(listener.checkErrorState()))
        return mlir::emitDefiniteFailure(lastFuncVisited,
                                         "pattern listener tracker fail");
    }
  }

  return listener.check(loc);
}

void transform_dialect::ApplyPatternsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// HoistStaticAllocOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform_dialect::HoistStaticAllocOp::applyToOne(
    func::FuncOp target, transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  Location loc = target->getLoc();
  IRRewriter rewriter(target->getContext());
  ErrorCheckingTrackingListener listener(state, *this);
  rewriter.setListener(&listener);
  mlir::iree_compiler::hoistStaticallyBoundAllocationsInFunc<memref::AllocOp>(
      rewriter, target);
  return listener.check(loc);
}

void transform_dialect::HoistStaticAllocOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// ShareForallOperandsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::ShareForallOperandsOp::applyToOne(
    scf::ForallOp forallOp, transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  IRRewriter rewriter(getContext());
  SmallVector<int64_t> shareOperands(getShareOperands());
  // Empty case: consider all operands need to be shared.
  if (shareOperands.empty()) {
    shareOperands =
        llvm::to_vector(llvm::seq<int64_t>(0, forallOp.getOutputs().size()));
  }
  for (int64_t outputIdx : getShareOperands()) {
    if (outputIdx < 0 || outputIdx >= forallOp.getOutputs().size())
      return mlir::emitDefiniteFailure(forallOp, "operand idx overflow");
    Value toShare = forallOp.getOutputs()[outputIdx];
    if (std::distance(toShare.getUses().begin(), toShare.getUses().end()) !=
        2) {
      /*return mlir::emitSilenceableFailure(
          forallOp,
          "operand to share must have exactly 2 uses, the forall op "
          "and an extract_slice op.");*/
      continue;
    }
    tensor::ExtractSliceOp extractSliceOp;
    for (Operation *user : toShare.getUsers()) {
      extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
      if (extractSliceOp) break;
    }
    if (!extractSliceOp) {
      /*return mlir::emitSilenceableFailure(
        forallOp,
        "shared operands use must be extractSliceOp.");*/
      continue;
    }
    // Get the corresponding bbArg.
    BlockArgument bbArg = forallOp.getOutputBlockArguments()[outputIdx];

    // Check if the extract_slice has a matching parallel_insert_slice
    // (i.e., same source/target, offsets, sizes and strides).
    auto isMatchingParallelInsertSlice = [&](Operation &op) {
      auto insertSlice = dyn_cast<tensor::ParallelInsertSliceOp>(&op);
      if (!insertSlice) return false;
      if (insertSlice.getDest() != bbArg) return false;
      return llvm::equal(insertSlice.getMixedOffsets(),
                         extractSliceOp.getMixedOffsets()) &&
             llvm::equal(insertSlice.getMixedSizes(),
                         extractSliceOp.getMixedSizes()) &&
             llvm::equal(insertSlice.getMixedStrides(),
                         extractSliceOp.getMixedStrides());
    };
    if (llvm::none_of(forallOp.getTerminator().getYieldingOps(),
                      isMatchingParallelInsertSlice)) {
      continue;
    }

    // Promote extract_slice source to bbArg.
    rewriter.updateRootInPlace(extractSliceOp, [&]() {
      extractSliceOp.getSourceMutable().assign(bbArg);
    });
  }

  results.push_back(forallOp);
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// ForallToWorkgroupOp
//===---------------------------------------------------------------------===//

/// Populate the workgroup_count region of `dispatchOp`.
/// For now, this only supports constant index ops and empty workload
/// operands. Assumes the HAL::ExecutableExportOp is built with an empty
/// region.
static LogicalResult populateWorkgroupCountComputingRegion(
    RewriterBase &rewriter, scf::ForallOp forallOp,
    HAL::ExecutableExportOp exportOp) {
  Location loc = forallOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  Region &r = exportOp.getWorkgroupCount();
  assert(r.empty() && "expected block-less workgroup_count region");
  Block *block = rewriter.createBlock(&r);
  // The HAL::DeviceType argument is always the first argument.
  block->addArgument(HAL::DeviceType::get(rewriter.getContext()), loc);
  rewriter.setInsertionPointToStart(block);

  SmallVector<Value> results;
  // For now, this assumes that we only pull in constants.
  // TODO: Iteratively pull required operations.
  for (Value v : forallOp.getUpperBound(rewriter)) {
    auto op = dyn_cast_or_null<arith::ConstantIndexOp>(v.getDefiningOp());
    if (!op) return failure();
    results.push_back(
        cast<arith::ConstantIndexOp>(rewriter.clone(*op)).getResult());
  }
  // Pad to `3` to match assumptions hardcoded in IREE.
  for (unsigned i = results.size(); i < 3; ++i) {
    results.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
  }
  rewriter.create<HAL::ReturnOp>(loc, results);

  return success();
}

//===---------------------------------------------------------------------===//
// Patterns for ForallToWorkgroup rewrite.
//===---------------------------------------------------------------------===//

LogicalResult rewriteForallToWorkgroup(RewriterBase &rewriter,
                                       scf::ForallOp forallOp,
                                       IREE::HAL::ExecutableExportOp exportOp) {
  // Step 0. Target-specific verifications. There is no good place to anchor
  // those right now: the ForallOp is target-independent and the
  // transform op does not apply to individual ForallOp.
  MLIRContext *ctx = forallOp->getContext();
  Location loc = forallOp->getLoc();
  // TODO iree should have own device mapping like #hal.workgroup<x/y/z>
  Attribute bX = gpu::GPUBlockMappingAttr::get(ctx, gpu::Blocks::DimX);
  Attribute bY = gpu::GPUBlockMappingAttr::get(ctx, gpu::Blocks::DimY);
  Attribute bZ = gpu::GPUBlockMappingAttr::get(ctx, gpu::Blocks::DimZ);
  if (forallOp.getNumResults() > 0)
    return forallOp->emitError(
        "only bufferized scf.forall lowers to workgroup");
  if (forallOp.getRank() > 3)
    return forallOp->emitError(
        "scf.forall with rank > 3 does not lower to workgroup");

  if (!forallOp.getMapping().has_value())
    return forallOp->emitError("mapping must be present");
  SmallVector<Attribute> blockMapping =
      llvm::to_vector(forallOp.getMapping()->getValue());
  if (llvm::any_of(blockMapping, [](DeviceMappingAttrInterface map) {
        return !map.isa<gpu::GPUBlockMappingAttr>();
      })) {
    return forallOp->emitError("mapping must be #gpu.block<x/y/z/>");
  }

  // Step 1. Complete the blockMapping to a full mapping (with 1s) if
  // necessary.
  SmallVector<Value> numBlocks =
      llvm::to_vector(forallOp.getUpperBound(rewriter));
  // Ensure we have 3 block sizes, one for each id.
  Value one;
  for (auto attr : {bX, bY, bZ}) {
    if (!llvm::is_contained(blockMapping, attr)) {
      blockMapping.push_back(attr);
      one = one ? one : rewriter.create<arith::ConstantIndexOp>(loc, 1);
      numBlocks.push_back(one);
    }
  }
  // Step 2. sort the values by the corresponding GPUBlockMappingAttr.
  auto comparator = [](Attribute a, Attribute b) -> bool {
    return static_cast<int64_t>(a.cast<gpu::GPUBlockMappingAttr>().getBlock()) <
           static_cast<int64_t>(b.cast<gpu::GPUBlockMappingAttr>().getBlock());
  };
  SmallVector<Value> gridDimValues =
      getValuesSortedByKey(blockMapping, numBlocks, comparator);

  // Step 3. Outline the compute workload region and set up the workload
  // operands, if this has not been done already.
  // Using `transform.iree.tile_to_forall_and_workgroup_count_region` is
  // the preferred way to set up tiling and workgroup_count region **at the
  // same time**.
  //
  // The block of code below will be retired once there is enough confidence
  // we can do everything without it. This includes in particular providing
  // custom fusion heuristics at the flow level: at this time, the only way to
  // fully control fusion of more advanced cases is to use the transform
  // dialect at the flow level and explicitly match the ops we want to fuse.
  // Once fusion is customizable enough in perpetuity, we can retire this.
  if (exportOp.getWorkgroupCount().empty()) {
    if (llvm::any_of(forallOp.getUpperBound(rewriter), [](Value v) {
          return !v.getDefiningOp<arith::ConstantIndexOp>();
        })) {
      return forallOp->emitError(
          "unsupported dynamic workgroup_count atm --- need to slice out "
          "workgroup_count computation into "
          "ExecutableExport::workgroup_count."
          "\nThis region may require arbitrary computations and cannot "
          "magically match what the `stream.cmd.dispatch` has already "
          "imposed "
          "on us at a distance."
          "\nFor now we must specify the number of values properly when "
          "applying the topLevel tile_to_forall_op");
    }
    if (failed(populateWorkgroupCountComputingRegion(rewriter, forallOp,
                                                     exportOp))) {
      return forallOp->emitOpError(
                 "failed to populate workload region for dispatchOp: ")
             << exportOp;
    }
  }

  // Step 4. Create the workgroup id and count ops.
  IRMapping bvm;
  SmallVector<Value> workgroupIdOps, workgroupCountOps;
  for (Attribute attr : blockMapping) {
    auto idx =
        static_cast<int64_t>(attr.cast<gpu::GPUBlockMappingAttr>().getBlock());
    workgroupIdOps.push_back(
        rewriter.create<HAL::InterfaceWorkgroupIDOp>(loc, idx));
    workgroupCountOps.push_back(
        rewriter.create<HAL::InterfaceWorkgroupCountOp>(loc, idx));
  }
  bvm.map(forallOp.getInductionVars(), workgroupIdOps);
  bvm.map(forallOp.getUpperBound(rewriter), workgroupCountOps);

  // Step 5. Predicate omitted given unique topLevel scf::ForallOp.

  // Step 6. Move the body of forallOp.
  // Erase the terminator first, it will not be used since we are on buffers.
  rewriter.eraseOp(forallOp.getTerminator());
  Block *targetBlock;
  Block::iterator insertionPoint;
  targetBlock = forallOp->getBlock();
  insertionPoint = Block::iterator(forallOp);
  Block &sourceBlock = forallOp.getRegion().front();
  targetBlock->getOperations().splice(insertionPoint,
                                      sourceBlock.getOperations());

  // Step 7. RAUW thread indices to thread ops.
  for (Value blockIdx : forallOp.getInductionVars()) {
    for (Operation *user : llvm::make_early_inc_range(blockIdx.getUsers())) {
      rewriter.updateRootInPlace(user, [&]() {
        user->replaceUsesOfWith(blockIdx, bvm.lookup(blockIdx));
      });
    }
  }

  // Step 5. Barriers omitted given unique topLevel scf::ForallOp.

  // Step 6. Erase old op.
  rewriter.eraseOp(forallOp);

  return success();
}

//===---------------------------------------------------------------------===//
// IREE-specific transformations defined outside of iree_linalg_transform.
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform_dialect::ForallToWorkgroupOp::applyToOne(
    func::FuncOp target, transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  if (!isa<HAL::ExecutableOp, HAL::ExecutableVariantOp>(state.getTopLevel())) {
    return mlir::emitDefiniteFailure(
        state.getTopLevel(),
        "requires HAL::ExecutableOp or HAL::ExecutableVariantOp toplevel "
        "to attach the workgroup size information to a nested "
        "ExecutableExportOp");
  }

  IREE::HAL::ExecutableExportOp exportOp;
  state.getTopLevel()->walk([&](IREE::HAL::ExecutableExportOp op) {
    if (op.getSymName() == target.getName()) exportOp = op;
  });
  if (!exportOp) {
    return mlir::emitSilenceableFailure(
        target, "no IREE::HAL::ExecutableExportOp found");
  }

  scf::ForallOp topLevelForallOp;
  auto walkResult = target->walk([&](scf::ForallOp forallOp) {
    if (forallOp->getParentOfType<scf::ForallOp>())
      return WalkResult::advance();
    if (topLevelForallOp) return WalkResult::interrupt();
    topLevelForallOp = forallOp;
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    return mlir::emitSilenceableFailure(
        target, "could not find a unique topLevel scf.forall");
  }

  Location loc = target->getLoc();
  IRRewriter rewriter(topLevelForallOp->getContext());
  rewriter.setInsertionPoint(topLevelForallOp);
  ErrorCheckingTrackingListener listener(state, *this);
  rewriter.setListener(&listener);
  if (failed(rewriteForallToWorkgroup(rewriter, topLevelForallOp, exportOp))) {
    return listener.check(loc, mlir::emitDefiniteFailure(
                                   target, "rewriteForallToWorkgroup failed"));
  }

  return listener.check(loc);
}

void transform_dialect::ForallToWorkgroupOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// TileToForallAndWorkgroupCountRegionOp
//===---------------------------------------------------------------------===//

void transform_dialect::TileToForallAndWorkgroupCountRegionOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    ArrayRef<int64_t> staticTileSizes, transform::TileSizesSpec,
    ArrayAttr mappingAttr) {
  return build(builder, result, target,
               /*mixedTileSizes=*/
               getAsOpFoldResult(builder.getI64ArrayAttr(staticTileSizes)),
               /*_=*/transform::TileSizesSpec(), /*mapping=*/mappingAttr);
}

void transform_dialect::TileToForallAndWorkgroupCountRegionOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    ArrayRef<OpFoldResult> mixedTileSizes, transform::TileSizesSpec,
    ArrayAttr mappingAttr) {
  assert(result.name.isRegistered() && "not registered!!");
  SmallVector<int64_t> staticTileSizes;
  SmallVector<Value> dynamicTileSizes;
  dispatchIndexOpFoldResults(mixedTileSizes, dynamicTileSizes, staticTileSizes);
  // Call the default builder which sets up the proper operands segment sizes
  // attributes for multiple variadic operands. In the absence of this,
  // horrible bugs ensue.
  MLIRContext *ctx = builder.getContext();
  auto operationType = pdl::OperationType::get(ctx);

  build(builder, result,
        /*resultTypes=*/TypeRange{operationType, operationType},
        /*target=*/target,
        /*numThreads=*/ValueRange{},
        /*tileSizes=*/dynamicTileSizes,
        /*staticNumThreads=*/ArrayRef<int64_t>(),
        /*staticTileSizes=*/staticTileSizes,
        /*mapping=*/mappingAttr);
}

void transform_dialect::TileToForallAndWorkgroupCountRegionOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    ArrayRef<int64_t> staticNumThreads, transform::NumThreadsSpec,
    ArrayAttr mappingAttr) {
  return build(builder, result,
               /*target=*/target,
               /*mixedNumThreads=*/
               getAsOpFoldResult(builder.getI64ArrayAttr(staticNumThreads)),
               /*_=*/transform::NumThreadsSpec(),
               /*mapping=*/mappingAttr);
}

void transform_dialect::TileToForallAndWorkgroupCountRegionOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    ArrayRef<OpFoldResult> mixedNumThreads, transform::NumThreadsSpec,
    ArrayAttr mappingAttr) {
  assert(result.name.isRegistered() && "not registered!!");
  SmallVector<int64_t> staticNumThreads;
  SmallVector<Value> dynamicNumThreads;
  dispatchIndexOpFoldResults(mixedNumThreads, dynamicNumThreads,
                             staticNumThreads);
  // Call the default builder which sets up the proper operands segment sizes
  // attributes for multiple variadic operands. In the absence of this,
  // horrible bugs ensue.
  MLIRContext *ctx = builder.getContext();
  auto operationType = pdl::OperationType::get(ctx);
  build(builder, result,
        /*resultTypes=*/TypeRange{operationType, operationType},
        /*target=*/target,
        /*numThreads=*/dynamicNumThreads,
        /*tileSizes=*/ValueRange{},
        /*staticNumThreads=*/staticNumThreads,
        /*staticTileSizes=*/ArrayRef<int64_t>(),
        /*mapping=*/mappingAttr);
}

/// Lower the ops within the workgroup count region of `exportOp` that
/// represents the workgroup count calculation, to the actual
/// computation that returns the number of workgroups. For now
/// this lowers the `flow.dispatch.workgroup_count_from_dag_root` op
/// to `ceilDiv(workload, tileSizes)`.
/// Note: transform::TransformState &state is passed to allow  unpacking
/// pdl::OperationType handles on the fly.
static LogicalResult lowerWorkgroupCountComputingRegion(
    transform::TransformState &state, RewriterBase &rewriter, Location loc,
    HAL::ExecutableExportOp exportOp, ArrayRef<OpFoldResult> numThreads,
    ArrayRef<OpFoldResult> tileSizes, std::optional<ArrayAttr> mapping) {
  Region &r = exportOp.getWorkgroupCount();
  if (!r.hasOneBlock()) {
    return rewriter.notifyMatchFailure(exportOp,
                                       "expected export op to have a workgroup "
                                       "count region with a single block");
  }
  auto workgroupCountOps =
      r.front().getOps<IREE::Flow::DispatchWorkgroupCountFromDagRootOp>();
  if (!llvm::hasSingleElement(workgroupCountOps)) {
    return rewriter.notifyMatchFailure(
        exportOp,
        "expected region to have a single "
        "flow.dispatch.workgroup_count_from_dag_root op");
  }
  auto workgroupCountOp = *workgroupCountOps.begin();
  auto workload = workgroupCountOp.getOperands();

  bool useNumThreads = !numThreads.empty();
  ArrayRef<OpFoldResult> tileSizesOrNumThreads =
      useNumThreads ? numThreads : tileSizes;
  StringRef kindStr = useNumThreads ? "num thread" : "tile size";

  SmallVector<OpFoldResult> unpackedTileSizesOrNumThreads;
  int64_t numTiledDims = 0;
  for (auto ofr : tileSizesOrNumThreads) {
    if (ofr.is<Value>() &&
        ofr.get<Value>().getType().isa<pdl::OperationType>()) {
      for (Operation *sizeProducer : state.getPayloadOps(ofr.get<Value>())) {
        if (sizeProducer->getNumResults() != 1) {
          auto diag = mlir::emitDefiniteFailure(sizeProducer)
                      << "the operation producing " << kindStr
                      << " must have one result";
          diag.attachNote(loc) << "when applying this transform";
          return diag;
        }
        unpackedTileSizesOrNumThreads.push_back(sizeProducer->getResult(0));
      }
    } else {
      unpackedTileSizesOrNumThreads.push_back(ofr);
    }
    if (!isConstantIntValue(unpackedTileSizesOrNumThreads.back(), 0))
      ++numTiledDims;
  }

  if (unpackedTileSizesOrNumThreads.size() > workload.size()) {
    return rewriter.notifyMatchFailure(
        exportOp,
        "number of " + kindStr + "s overflow the dimension from the workload");
  }

  // Generate permutation of tiled dims based on the specified mapping.
  SmallVector<int64_t> mappingPermutation;
  if (mapping.has_value()) {
    if (numTiledDims != mapping->size()) {
      return rewriter.notifyMatchFailure(exportOp,
                                         "number of mapping elements must "
                                         "match number of non-zero " +
                                             kindStr + "s");
    }
    for (DeviceMappingAttrInterface map : mapping.value())
      mappingPermutation.push_back(map.getMappingId());
  } else {
    // No mapping specified: No permutation.
    for (int64_t i = 0; i < numTiledDims; ++i) mappingPermutation.push_back(i);
  }

  // Compute number of workgroups.
  SmallVector<OpFoldResult> workgroupCount(3, rewriter.getIndexAttr(1));
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(workgroupCountOp);
  loc = workgroupCountOp.getLoc();
  int64_t nextTiledDim = 0;
  for (int64_t workgroupsDim : mappingPermutation) {
    // Skip dims with tile size 0. These are not tiled.
    while (isConstantIntValue(unpackedTileSizesOrNumThreads[nextTiledDim], 0))
      ++nextTiledDim;
    if (useNumThreads) {
      workgroupCount[workgroupsDim] =
          unpackedTileSizesOrNumThreads[nextTiledDim];
    } else {
      AffineExpr s0, s1;
      bindSymbols(rewriter.getContext(), s0, s1);
      auto m = AffineMap::get(0, 2, s0.ceilDiv(s1));
      workgroupCount[workgroupsDim] = affine::makeComposedFoldedAffineApply(
          rewriter, loc, m,
          ArrayRef<OpFoldResult>{workload[nextTiledDim],
                                 unpackedTileSizesOrNumThreads[nextTiledDim]});
    }
    ++nextTiledDim;
  }

  rewriter.replaceOp(workgroupCountOp, getValueOrCreateConstantIndexOp(
                                           rewriter, loc, workgroupCount));
  return success();
}

SmallVector<OpFoldResult>
transform_dialect::TileToForallAndWorkgroupCountRegionOp::getMixedNumThreads() {
  Builder b(getContext());
  return getMixedValues(getStaticNumThreads(), getNumThreads(), b);
}

SmallVector<OpFoldResult>
transform_dialect::TileToForallAndWorkgroupCountRegionOp::getMixedTileSizes() {
  Builder b(getContext());
  return getMixedValues(getStaticTileSizes(), getTileSizes(), b);
}

LogicalResult
transform_dialect::TileToForallAndWorkgroupCountRegionOp::verify() {
  if (getMixedNumThreads().empty() == getMixedTileSizes().empty())
    return emitOpError("either num_threads or tile_sizes must be specified");
  return success();
}

void transform_dialect::TileToForallAndWorkgroupCountRegionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getTarget(), effects);
  transform::onlyReadsHandle(getTileSizes(), effects);
  transform::onlyReadsHandle(getNumThreads(), effects);
  transform::producesHandle(getResults(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure
transform_dialect::TileToForallAndWorkgroupCountRegionOp::apply(
    transform::TransformResults &transformResults,
    transform::TransformState &state) {
  ArrayRef<Operation *> targetOps = state.getPayloadOps(getTarget());
  if (targetOps.empty()) {
    transformResults.set(getForallOp().cast<OpResult>(), {});
    transformResults.set(getTiledOp().cast<OpResult>(), {});
    return DiagnosedSilenceableFailure::success();
  }
  if (targetOps.size() != 1) {
    return mlir::emitDefiniteFailure(
               state.getTopLevel(),
               "expected single target op in payload, got: ")
           << targetOps.size();
  }
  auto funcOp = targetOps.front()->getParentOfType<func::FuncOp>();
  FailureOr<IREE::HAL::ExecutableExportOp> exportOp = getEntryPoint(funcOp);
  if (failed(exportOp)) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "couldn't find export op for func");
  }

  /// Lower the workgroup count region in keeping with the way dispatch
  /// regions are created by default in IREEs compilation flow.
  IRRewriter rewriter(getContext());
  if (failed(lowerWorkgroupCountComputingRegion(
          state, rewriter, getLoc(), exportOp.value(), getMixedNumThreads(),
          getMixedTileSizes(), getMapping()))) {
    return mlir::emitDefiniteFailure(exportOp.value(),
                                     "failed to lower workgroup count region");
  }

  ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());

  // Result payload ops.
  SmallVector<Operation *> tileOps;
  SmallVector<Operation *> tiledOps;

  DiagnosedSilenceableFailure diag = transform::tileToForallOpImpl(
      rewriter, state, cast<transform::TransformOpInterface>(getOperation()),
      targets, getMixedNumThreads(), getMixedTileSizes(), getMapping(), tileOps,
      tiledOps);

  if (!diag.succeeded()) {
    transformResults.set(getForallOp().cast<OpResult>(),
                         SmallVector<mlir::Operation *>{});
    transformResults.set(getTiledOp().cast<OpResult>(),
                         SmallVector<mlir::Operation *>{});
    return diag;
  }

  transformResults.set(getForallOp().cast<OpResult>(), tileOps);
  transformResults.set(getTiledOp().cast<OpResult>(), tiledOps);
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// IREEBufferizeOp
//===---------------------------------------------------------------------===//

// Important note: this transform is load-bearing and is the glue between
// different dialects that want to operate on tensors.
// Originaly, it used to just call `addIREEComprehensiveBufferizePasses` but
// this introduces a lot of complexity in the registration process due to the
// use of nested pass pipelines, to a point that it is a major endeavor to
// connect a new dialect.
// Instead, avoid calling the passes and only take what we need from them.
//
// TODO: Maybe we need both a transform.iree.cpu.bufferize and a
// transform.iree.gpu.bufferize rather than a single common bufferize op?
//
// Note: This has become so specific that it may be worth it to separate in
// its own .cpp file.
using mlir::bufferization::BufferizationOptions;
using mlir::bufferization::OneShotAnalysisState;
using mlir::bufferization::OneShotBufferizationOptions;

void transform_dialect::IREEBufferizeOp::build(OpBuilder &builder,
                                               OperationState &result,
                                               Value target, bool targetGpu,
                                               bool testAnalysisOnly,
                                               bool printConflicts) {
  result.addOperands(target);
  if (targetGpu) {
    result.addAttribute(IREEBufferizeOp::getTargetGpuAttrName(result.name),
                        builder.getUnitAttr());
  }
  if (testAnalysisOnly) {
    result.addAttribute(
        IREEBufferizeOp::getTestAnalysisOnlyAttrName(result.name),
        builder.getUnitAttr());
  }
  if (printConflicts) {
    result.addAttribute(IREEBufferizeOp::getPrintConflictsAttrName(result.name),
                        builder.getUnitAttr());
  }
  MLIRContext *ctx = builder.getContext();
  result.addTypes(pdl::OperationType::get(ctx));
}

//===---------------------------------------------------------------------===//
// Default allocation functions for CPU backend
// TODO: register the bufferization behavior in a target-specific way.
// TODO: Maybe bufferize should have a separate cpu and a gpu version. This is
// unclear though: what happens on heterogeneous HW ?
//===---------------------------------------------------------------------===//

// Allocation callbacks to use with upstream comprehensive bufferization
static FailureOr<Value> cpuComprehensiveBufferizeAllocationFn(
    OpBuilder &builder, Location loc, MemRefType memRefType,
    ValueRange dynamicSizes, unsigned alignment) {
  return builder
      .create<memref::AllocaOp>(loc, memRefType, dynamicSizes,
                                builder.getI64IntegerAttr(alignment))
      .getResult();
}

static LogicalResult cpuComprehensiveBufferizeDeallocationFn(OpBuilder &builder,
                                                             Location loc,
                                                             Value allocation) {
  return success();
}

static LogicalResult cpuComprehensiveBufferizeCopyFn(OpBuilder &builder,
                                                     Location loc, Value from,
                                                     Value to) {
  // TODO: ideally we should use linalg.copy which was recently reintroduced
  // as an OpDSL named op. However, IREE-specific patterns to cleanup spurious
  // post-bufferization copies do not trigger properly.
  // So we keep using `createLinalgCopyOp` which builds a GenericOp.
  // builder.create<linalg::CopyOp>(loc, from, to);
  mlir::iree_compiler::createLinalgCopyOp(builder, loc, from, to);
  return success();
}

static FailureOr<Value> gpuComprehensiveBufferizeAllocationFn(
    OpBuilder &builder, Location loc, MemRefType memRefType,
    ValueRange dynamicSizes, unsigned alignment) {
  auto addressSpaceAttr = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  MemRefType allocType =
      MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                      AffineMap(), addressSpaceAttr);
  return builder
      .create<memref::AllocOp>(loc, allocType, dynamicSizes,
                               builder.getI64IntegerAttr(alignment))
      .getResult();
}

static LogicalResult gpuComprehensiveBufferizeDeallocationFn(OpBuilder &builder,
                                                             Location loc,
                                                             Value allocation) {
  builder.create<memref::DeallocOp>(loc, allocation);
  return success();
}

static LogicalResult gpuComprehensiveBufferizeCopyFn(OpBuilder &builder,
                                                     Location loc, Value from,
                                                     Value to) {
  // Insert barriers for copies from and to shared memory.
  bool needsBarrier = false;
  if (hasSharedMemoryAddressSpace(from.getType().cast<MemRefType>()) !=
      hasSharedMemoryAddressSpace(to.getType().cast<MemRefType>())) {
    needsBarrier = true;
  }
  if (needsBarrier) builder.create<gpu::BarrierOp>(loc);
  // TODO: ideally we should use linalg.copy which was recently reintroduced
  // as an OpDSL named op. However, IREE-specific patterns to cleanup spurious
  // post-bufferization copies do not trigger properly.
  // So we keep using `createLinalgCopyOp` which builds a GenericOp.
  // builder.create<linalg::CopyOp>(loc, from, to);
  mlir::iree_compiler::createLinalgCopyOp(builder, loc, from, to);
  if (needsBarrier) builder.create<gpu::BarrierOp>(loc);
  return success();
}

static IREEOneShotBufferizationOptions getBufferizationOptions() {
  IREEOneShotBufferizationOptions options;
  // options.testAnalysisOnly = testAnalysisOnly;
  // options.printConflicts = printConflicts;

  // bufferization.to_memref is used to bufferize constants in IREE. IREE has
  // it's own logic to handle constants. We'd like to leave the arith.constant
  // as is and insert bufferization.to_memref to convert the tensor to memref.
  options.opFilter.denyOperation<arith::ConstantOp>();
  options.opFilter.denyOperation<bufferization::ToMemrefOp>();

  // This type converter converts tensor types to memref types when no exact
  // memref type can be inferred from the context.
  options.unknownTypeConverterFn = [](Value value, Attribute memorySpace,
                                      const BufferizationOptions &options) {
    auto tensorType = value.getType().cast<TensorType>();

    // Special rule for ConstantOps: These always lower to some memref with a
    // static identity layout.
    if (value.getDefiningOp<arith::ConstantOp>())
      return bufferization::getMemRefTypeWithStaticIdentityLayout(tensorType,
                                                                  memorySpace);

    // Default case: Fully dynamic layout map for best compatibility.
    return bufferization::getMemRefTypeWithFullyDynamicLayout(tensorType,
                                                              memorySpace);
  };

  return options;
}

namespace {
/// Pattern to rewrite tensor.empty to tensor.alloc.
struct EmptyTensorLoweringPattern : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern<tensor::EmptyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::EmptyOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<bufferization::AllocTensorOp>(
        op, op.getType(), op.getDynamicSizes());
    return success();
  }
};
}  // namespace

DiagnosedSilenceableFailure transform_dialect::IREEBufferizeOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  ArrayRef<Operation *> payload = state.getPayloadOps(getTarget());
  if (payload.size() != 1 ||
      !isa<ModuleOp, HAL::ExecutableOp, HAL::ExecutableVariantOp>(
          payload.front())) {
    return mlir::emitDefiniteFailure(
        state.getTopLevel(),
        "requires exactly a single HAL::ExecutableOp or "
        "HAL::ExecutableVariantOp target op.");
  }

  //===-------------------------------------------------------------------===//
  // DO NOT JUST CALL `addIREEComprehensiveBufferizePasses` as this results in
  // a lot of registration issues due to nested pass pipeline mess.
  // Instead, take what we need from it.
  //===-------------------------------------------------------------------===//
  // Bufferize the dispatch.
  using mlir::bufferization::BufferizationOptions;
  BufferizationOptions::AllocationFn allocationFn =
      cpuComprehensiveBufferizeAllocationFn;
  BufferizationOptions::DeallocationFn deallocationFn =
      cpuComprehensiveBufferizeDeallocationFn;
  BufferizationOptions::MemCpyFn memCpyFn = cpuComprehensiveBufferizeCopyFn;
  if (getTargetGpu()) {
    allocationFn = gpuComprehensiveBufferizeAllocationFn;
    deallocationFn = gpuComprehensiveBufferizeDeallocationFn;
    memCpyFn = gpuComprehensiveBufferizeCopyFn;
  }

  Operation *target = payload.front();
  Location loc = target->getLoc();
  ErrorCheckingTrackingListener listener(state, *this);
  //   1. Rewrite tensor.empty to tensor.alloc, without the pass baggage.
  {
    RewritePatternSet patterns(getContext());
    patterns.add<EmptyTensorLoweringPattern>(patterns.getContext());
    GreedyRewriteConfig config;
    config.listener = &listener;
    // Manually gather list of ops because the other GreedyPatternRewriteDriver
    // overloads only accepts ops that are isolated from above.
    SmallVector<Operation *> ops;
    state.getTopLevel()->walk([&](Operation *nestedOp) {
      if (state.getTopLevel() != nestedOp) ops.push_back(nestedOp);
    });
    LogicalResult result =
        applyOpPatternsAndFold(ops, std::move(patterns), config);
    LogicalResult listenerResult = listener.checkErrorState();
    if (failed(result)) {
      return listener.check(
          loc, mlir::emitDefiniteFailure(state.getTopLevel(),
                                         "greedy pattern application failed"));
    }
    if (failed(listenerResult)) {
      return listener.check(
          loc, mlir::emitDefiniteFailure(state.getTopLevel(),
                                         "listener tracking failed"));
    }
  }

  //   2. Run one-shot-bufferize, without the pass baggage.
  IREEOneShotBufferizationOptions options = getBufferizationOptions();
  options.allocationFn = allocationFn;
  options.deallocationFn = deallocationFn;
  options.memCpyFn = memCpyFn;
  options.testAnalysisOnly = getTestAnalysisOnly();
  options.printConflicts = getPrintConflicts();
  if (getTargetGpu()) {
    // TODO(#12933): Because of regressions in CUDA backend, there is an
    // option to keep a legacy mode of not representing the offset in the
    // type. Remove once the bug is fixed.
    options.embedSubspanOffsetIntoMemRefType = false;
  }
  if (failed(runIREEOneShotBufferize(state.getTopLevel(), options)))
    return listener.check(loc, emitDefaultDefiniteFailure(target));

  // Early exit if test_analysis_only is set.
  if (getTestAnalysisOnly()) {
    results.set(getOperation()->getOpResult(0), payload.front());
    return listener.check(loc);
  }

  //   3. Post-bufferization passes are fine.
  PassManager pm(getContext());
  addIREEPostBufferizationPasses(pm);
  WalkResult res = state.getTopLevel()->walk([&](ModuleOp moduleOp) {
    if (failed(pm.run(moduleOp))) {
      getOperation()->emitError()
          << "failed to post-bufferization passes on module:\n"
          << *(moduleOp.getOperation()) << "\nunder top-level:\n"
          << *state.getTopLevel();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted())
    return listener.check(loc, emitDefaultDefiniteFailure(target));

  results.set(getOperation()->getOpResult(0), payload.front());
  return listener.check(loc);
}

//===---------------------------------------------------------------------===//
// IREEEliminateEmptyTensorsOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::IREEEliminateEmptyTensorsOp::applyToOne(
    ::mlir::Operation *target,
    ::mlir::transform::ApplyToEachResultList &results,
    ::mlir::transform::TransformState &state) {
  Location loc = target->getLoc();
  IRRewriter rewriter(target->getContext());
  ErrorCheckingTrackingListener listener(state, *this);
  rewriter.setListener(&listener);
  if (failed(
          eliminateEmptyTensors(rewriter, target, getBufferizationOptions()))) {
    getOperation()->emitError() << "failed to eliminate tensor.empty ops";
    return listener.check(loc, emitDefaultDefiniteFailure(target));
  }
  return listener.check(loc);
}

void transform_dialect::IREEEliminateEmptyTensorsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// EraseHALDescriptorTypeFromMemRef
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::IREEEraseHALDescriptorTypeFromMemRefOp::applyToOne(
    ::mlir::Operation *target,
    ::mlir::transform::ApplyToEachResultList &results,
    ::mlir::transform::TransformState &state) {
  if (!isa<func::FuncOp>(target)) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "expects a func::FuncOp as the target op");
  }
  auto funcOp = cast<func::FuncOp>(target);

  if (failed(eraseHALDescriptorTypeFromMemRef(funcOp))) {
    return mlir::emitDefiniteFailure(
        state.getTopLevel(),
        "failed to erase #hal.descriptor_type as MemRef memory space");
  }

  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::IREEEraseHALDescriptorTypeFromMemRefOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// ConvertConv2DToImg2ColAndAdjustWorkgroupCountOp
//===---------------------------------------------------------------------===//

// Rewrite the workgroup count compute region based on the specified collapsed
// dimensions on the output tensor. Only the dims that are expected to be tiled
// and distributed are adjusted, thus considering only the output is sufficient.
// TODO: This should be deprecated once workgroup count computation is
// refactored based on slices.
static LogicalResult adjustWorkgroupCountComputeRegionForImg2Col(
    transform::TransformState &state, RewriterBase &rewriter, Location loc,
    HAL::ExecutableExportOp exportOp, AffineMap outputMap,
    tensor::CollapseShapeOp collapseOp) {
  Region &r = exportOp.getWorkgroupCount();
  if (!r.hasOneBlock()) {
    return rewriter.notifyMatchFailure(exportOp,
                                       "expected export op to have a workgroup "
                                       "count region with a single block");
  }
  auto workgroupCountOps =
      r.front().getOps<IREE::Flow::DispatchWorkgroupCountFromDagRootOp>();
  if (!llvm::hasSingleElement(workgroupCountOps)) {
    return rewriter.notifyMatchFailure(
        exportOp,
        "expected region to have a single "
        "flow.dispatch.workgroup_count_from_dag_root op");
  }
  auto workgroupCountOp = *workgroupCountOps.begin();
  auto workload = workgroupCountOp.getOperands();

  // Extend the vector with workload values in range [l, r).
  auto pushRange = [&](SmallVector<Value> &vec, int l, int r) {
    for (; l < r; l++) vec.push_back(workload[l]);
  };

  SmallVector<unsigned> workloadDims;
  for (AffineExpr result : outputMap.getResults()) {
    workloadDims.push_back(result.cast<AffineDimExpr>().getPosition());
  }
  workloadDims.push_back(workload.size());
  SmallVector<ReassociationIndices> dimCollapseMapping =
      collapseOp.getReassociationIndices();

  // Adjust the workgroup count computation. This happens by taking the product
  // of workloads corresponding to groups of collapsed dims. For example,
  //    workload = [w1, w2, w3, w4, w5, w6, w7, w8]
  //    dimCollapseMapping = [[0], [1, 2], [3]]
  //    workloadDims = [1, 3, 4, 5, (8)]
  // will collapse to
  //    newWorkload = [w1, w2, w3, w4 * w5, w6, w7, w8]
  SmallVector<Value> newWorkload;
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(workgroupCountOp);
  loc = workgroupCountOp.getLoc();
  for (auto indices : dimCollapseMapping) {
    // If there is a single index we can just push the corresponding workload
    // arg.
    if (indices.size() == 1) {
      int64_t index = indices[0];
      pushRange(newWorkload, workloadDims[index], workloadDims[index + 1]);
      continue;
    }

    assert(indices.size());
    SmallVector<AffineExpr> syms(indices.size());
    bindSymbolsList(rewriter.getContext(), MutableArrayRef<AffineExpr>{syms});
    SmallVector<Value> originalSizes;
    AffineExpr product = syms[0];
    SmallVector<Value> skipped;
    for (auto [enIndex, dimIndex] : llvm::enumerate(indices)) {
      originalSizes.push_back(workload[workloadDims[dimIndex]]);
      if (enIndex > 0) {
        // Push any skipped workload values.
        pushRange(skipped, workloadDims[dimIndex - 1] + 1,
                  workloadDims[dimIndex]);
        product = product * syms[enIndex];
      }
    }

    // Make+push the collapsed workload value, followed by all values
    // until the next dim affected by the collapse_shape op.
    auto m = AffineMap::get(0, indices.size(), product);
    Value collapsedIndex =
        affine::makeComposedAffineApply(rewriter, loc, m, originalSizes);
    newWorkload.push_back(collapsedIndex);
    newWorkload.append(skipped);
    pushRange(newWorkload, workloadDims[indices.back()] + 1,
              workloadDims[indices.back() + 1]);
  }

  rewriter.replaceOpWithNewOp<IREE::Flow::DispatchWorkgroupCountFromDagRootOp>(
      workgroupCountOp, newWorkload);
  return success();
}

DiagnosedSilenceableFailure
transform_dialect::ConvertConv2DToImg2ColAndAdjustWorkgroupCountOp::applyToOne(
    linalg::LinalgOp target, transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  auto funcOp = target->getParentOfType<func::FuncOp>();

  // Get the output map for the target operation before rewriting.
  AffineMap outputMap = target.getIndexingMapsArray().back();

  IRRewriter rewriter(target->getContext());
  rewriter.setInsertionPoint(target);
  // TODO: Extend this to other convolution cases by handling cases beyond
  // collapse -> matmul -> expand from the patterns for 2D convolutions handled
  // here.
  auto maybeTransformed =
      TypeSwitch<Operation *, FailureOr<std::pair<Operation *, Operation *>>>(
          target)
          .Case([&](linalg::Conv2DNhwcHwcfOp op) {
            return rewriteInIm2Col(rewriter, op);
          })
          .Case([&](linalg::Conv2DNchwFchwOp op) {
            return rewriteInIm2Col(rewriter, op);
          })
          .Default([&](Operation *op) {
            return rewriter.notifyMatchFailure(op, "not supported");
          });
  if (failed(maybeTransformed)) return emitDefaultSilenceableFailure(target);

  Operation *im2col = maybeTransformed->first;
  Operation *convReplacement = maybeTransformed->second;

  // Push handles for the im2col operation and the operation that replaces the
  // original convolution.
  results.push_back(im2col);
  // Handle to the operation that replaces the original convolution.
  results.push_back(convReplacement);

  // If there is no export region, no adjustment needed.
  FailureOr<IREE::HAL::ExecutableExportOp> exportOp = getEntryPoint(funcOp);
  if (failed(exportOp)) return DiagnosedSilenceableFailure::success();

  // The result of the im2col pattern is expected to be an expand on the matmul
  // output.
  auto expandShapeOp = dyn_cast<tensor::ExpandShapeOp>(convReplacement);
  if (!expandShapeOp)
    return mlir::emitDefiniteFailure(target,
                                     "im2col matmul output not expanded");

  auto matmulOp = expandShapeOp.getSrc().getDefiningOp<linalg::LinalgOp>();
  if (!matmulOp || !isaContractionOpInterface(matmulOp))
    return mlir::emitDefiniteFailure(target, "im2col matmul not found");

  // If there is no collapse then the workgroup count compute region doesn't
  // need to be updated.
  auto outputCollapse = matmulOp.getDpsInitOperand(0)
                            ->get()
                            .getDefiningOp<tensor::CollapseShapeOp>();
  if (!outputCollapse) return DiagnosedSilenceableFailure::success();

  /// Lower the workgroup count region in keeping with the way dispatch
  /// regions are created by default in IREEs compilation flow.
  if (failed(adjustWorkgroupCountComputeRegionForImg2Col(
          state, rewriter, getLoc(), exportOp.value(), outputMap,
          outputCollapse))) {
    return mlir::emitDefiniteFailure(convReplacement,
                                     "failed to adjust workgroup count region");
  }
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::ConvertConv2DToImg2ColAndAdjustWorkgroupCountOp::build(
    OpBuilder &builder, OperationState &result, Value target) {
  result.addOperands(target);
  MLIRContext *ctx = builder.getContext();
  result.addTypes({pdl::OperationType::get(ctx), pdl::OperationType::get(ctx)});
}

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensionsOps.cpp.inc"
