// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Dialect/Util/Analysis/IntegerDivisibilityAnalysis.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-llvmcpu-split-reduction"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUSPLITREDUCTIONPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {

/// Returns the single combiner operation of the reduction body for the
/// `initIndex`-th output of `op`. Returns nullptr if the body does not reduce
/// with a single operation.
static Operation *matchSingleCombinerOp(linalg::LinalgOp op,
                                        unsigned initIndex) {
  SmallVector<Operation *, 4> combinerOps;
  if (!matchReduction(op.getRegionOutputArgs(), initIndex, combinerOps) ||
      combinerOps.size() != 1) {
    return nullptr;
  }
  return combinerOps.front();
}

/// Make sure that
/// - the pass has not been applied before
/// - has tensor semantics
/// - number of reduction loops == 1
/// - has exactly 1 output
/// - index map has only projected permutations
/// - is a linalg generic op
/// - has exactly 1 input
/// - if enableReductionReordering is not set, then operand is an int
/// - innermost dimension of the input operand is reduction
/// - the reduction body is a single combiner op with a known neutral element
/// TODO: support named ops, numInputs > 1, and modify lastDim check below
/// accordingly. If fpReductionReordering is not enabled by default, it must
/// be an integer or index type to proceed to allow associative reordering.
LogicalResult splitReductionPrecondition(Operation *op,
                                         bool fpReductionReordering) {
  linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);

  if (!linalgOp.hasPureTensorSemantics()) {
    LDBG() << "doesn't have tensor semantics";
    return failure();
  }
  if (linalgOp.getNumReductionLoops() != 1) {
    LDBG() << "number of reduction loops != 1";
    return failure();
  }
  if (linalgOp.getNumDpsInits() != 1) {
    LDBG() << "doesn't have exactly 1 output";
    return failure();
  }
  if (!linalgOp.hasOnlyProjectedPermutations()) {
    LDBG() << "index map doesn't have only projected permutations";
    return failure();
  }
  if (!isa<linalg::GenericOp>(op)) {
    LDBG() << "is not a generic op";
    return failure();
  }
  if (linalgOp.getNumDpsInputs() != 1) {
    LDBG() << "doesn't have exactly 1 input";
    return failure();
  }
  // The `linalg::splitReduction` method does not work for ops with indexing
  // semantics. See https://github.com/iree-org/iree/pull/14979
  if (linalgOp.hasIndexSemantics()) {
    LDBG()
        << "the split method used currently doesnt support indexing semantics";
    return failure();
  }

  auto elemType =
      getElementTypeOrSelf(linalgOp.getDpsInitOperand(0)->get().getType());
  if (!(fpReductionReordering || elemType.isIntOrIndex())) {
    LDBG() << "skipped because reduction reordering on FP is not enabled.";
    return failure();
  }

  SmallVector<unsigned> dims;
  linalgOp.getReductionDims(dims);
  AffineMap map =
      linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(0));
  unsigned lastIdx = map.getNumResults() - 1;
  unsigned lastDim = map.getDimPosition(lastIdx);
  if (lastDim != dims[0]) {
    LDBG() << "innermost dimension of the input operand is not reduction";
    return failure();
  }

  // The reduction body must consist of a single combiner op with a known
  // neutral element, since the rewrite materializes the split using that
  // combiner and initializes the partial accumulator with its neutral element.
  Operation *combinerOp = matchSingleCombinerOp(linalgOp, /*initIndex=*/0);
  if (!combinerOp) {
    LDBG() << "cannot match a single combiner op in the reduction body";
    return failure();
  }
  if (!arith::getNeutralElement(combinerOp).has_value()) {
    LDBG() << "unknown neutral element for the reduction combiner";
    return failure();
  }

  return success();
}

/// Returns the result position in the input operand's indexing map that
/// corresponds to the single reduction iterator dimension of `op`. Fails if
/// `op` does not have exactly one reduction dimension or if that dimension is
/// not referenced by the input indexing map.
static FailureOr<unsigned> getReductionOperandDimPosition(linalg::LinalgOp op) {
  SmallVector<unsigned> reductionDims;
  op.getReductionDims(reductionDims);
  if (reductionDims.size() != 1) {
    return failure();
  }

  AffineMap inputMap = op.getMatchingIndexingMap(op.getDpsInputOperand(0));
  for (auto [idx, dimExpr] : llvm::enumerate(inputMap.getResults())) {
    if (cast<AffineDimExpr>(dimExpr).getPosition() == reductionDims.front()) {
      return idx;
    }
  }
  return failure();
}

/// Returns the size of the input operand along its reduction dimension. Prefers
/// the static tensor shape when available.
static FailureOr<OpFoldResult> getReductionSize(linalg::LinalgOp op) {
  FailureOr<unsigned> reductionOperandDimPos =
      getReductionOperandDimPosition(op);
  if (failed(reductionOperandDimPos)) {
    return failure();
  }
  Value input = op.getDpsInputOperand(0)->get();
  auto inputType = cast<RankedTensorType>(input.getType());
  int64_t staticDim = inputType.getDimSize(*reductionOperandDimPos);
  if (ShapedType::isStatic(staticDim)) {
    return OpFoldResult(
        IntegerAttr::get(IndexType::get(op.getContext()), staticDim));
  }
  if (auto viewOp = input.getDefiningOp<OffsetSizeAndStrideOpInterface>()) {
    return viewOp.getMixedSizes()[*reductionOperandDimPos];
  }
  return failure();
}

/// Creates an `arith.constant` holding the neutral element of the reduction
/// combiner for the `initIndex`-th output of `op`. Fails if the combiner cannot
/// be matched as a single op or if it has no known neutral element.
static FailureOr<Value> createIdentityValue(OpBuilder &builder,
                                            linalg::LinalgOp op,
                                            unsigned initIndex) {
  Operation *reductionOp = matchSingleCombinerOp(op, initIndex);
  if (!reductionOp) {
    return failure();
  }
  std::optional<TypedAttr> identityAttr = arith::getNeutralElement(reductionOp);
  if (!identityAttr) {
    return failure();
  }
  Type elementType =
      getElementTypeOrSelf(op.getDpsInits()[initIndex].getType());
  return arith::ConstantOp::create(builder, op.getLoc(), elementType,
                                   *identityAttr)
      .getResult();
}

/// Returns true if the reduction dimension size of `op` is provably a multiple
/// of `splitSize`. For dynamic cases, returns false when  `solver` is null or
/// has no divisibility information.
static bool canSplitReductionWithSize(linalg::LinalgOp op, int64_t splitSize,
                                      DataFlowSolver *solver) {
  FailureOr<OpFoldResult> reductionSize = getReductionSize(op);
  if (failed(reductionSize)) {
    return false;
  }
  if (std::optional<int64_t> cstSize = getConstantIntValue(*reductionSize)) {
    return *cstSize % splitSize == 0;
  }

  if (!solver) {
    return false;
  }
  auto *lattice = solver->lookupState<IREE::Util::IntegerDivisibilityLattice>(
      cast<Value>(*reductionSize));
  if (!lattice || lattice->getValue().isUninitialized()) {
    return false;
  }
  return lattice->getValue().getValue().udiv() % splitSize == 0;
}

/// Rewrites `linalgOp` by splitting its single reduction dimension into an
/// outer reduction dimension and an inner parallel dimension of size
/// `splitSize`, yielding a partial reduction followed by a final reduction over
/// the new inner dimension. The split is materialized with
/// `tensor.expand_shape`; the inner split size is static while the outer
/// dimension may be dynamic when `reductionSize` is only bounded (not a
/// compile-time constant). `insertSplitIndex` controls where the new parallel
/// dimension is placed in the partial-output tensor. `reductionSize` is the
/// size of the reduction dimension, possibly a dynamic `Value`, and is assumed
/// to be a multiple of `splitSize`.
///
/// Only the inner-parallel variant is supported: the new parallel dimension is
/// always placed inside the reduction, matching the shape
///   `<..., outer, splitSize, ...>`
/// where `outer` is the original reduction dimension.
///
/// Assumes `splitReductionPrecondition` has been verified on `linalgOp`: a
/// single reduction dim, a single input and single output, a matchable combiner
/// with a known neutral element, and an input indexing map whose innermost
/// result is the reduction dim.
static linalg::SplitReductionResult splitReduction(linalg::LinalgOp linalgOp,
                                                   int64_t splitSize,
                                                   unsigned insertSplitIndex,
                                                   OpFoldResult reductionSize,
                                                   RewriterBase &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(linalgOp);

  SmallVector<unsigned> dims;
  linalgOp.getReductionDims(dims);
  unsigned reductionDim = dims[0];
  unsigned insertSplitDimension = reductionDim + 1;

  Operation *reductionOp = matchSingleCombinerOp(linalgOp, /*initIndex=*/0);
  assert(reductionOp &&
         "precondition guarantees a single combiner op in the reduction body");
  FailureOr<Value> identityValue =
      createIdentityValue(rewriter, linalgOp, /*initIndex=*/0);
  assert(succeeded(identityValue) &&
         "precondition guarantees a known neutral element for the combiner");

  Location loc = linalgOp.getLoc();
  MLIRContext *context = linalgOp.getContext();

  // Calculate the new shape and indexing map of the input operand.
  OpOperand *inputOperand = linalgOp.getDpsInputOperand(0);
  Value input = inputOperand->get();
  auto inputType = cast<RankedTensorType>(input.getType());
  AffineMap map = linalgOp.getMatchingIndexingMap(inputOperand);
  SmallVector<OpFoldResult> newShape;
  SmallVector<int64_t> newStaticShape;
  SmallVector<AffineExpr> exprs;
  SmallVector<ReassociationIndices> reassociation;
  unsigned index = 0;

  AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
  AffineMap outerDimMap = AffineMap::get(0, 1, s0.floorDiv(splitSize), context);
  for (unsigned idx : llvm::seq<unsigned>(0, map.getNumResults())) {
    unsigned dim = map.getDimPosition(idx);
    if (reductionDim == dim) {
      OpFoldResult outerSize = affine::makeComposedFoldedAffineApply(
          rewriter, loc, outerDimMap, ArrayRef<OpFoldResult>{reductionSize});
      newShape.push_back(outerSize);                        // reduce
      newShape.push_back(rewriter.getIndexAttr(splitSize)); // parallel (insert)
      newStaticShape.push_back(
          getConstantIntValue(outerSize).value_or(ShapedType::kDynamic));
      newStaticShape.push_back(splitSize);
      exprs.push_back(rewriter.getAffineDimExpr(
          dim < insertSplitDimension ? dim : dim + 1));
      exprs.push_back(rewriter.getAffineDimExpr(insertSplitDimension));
      reassociation.push_back({index++, index++});
      continue;
    }
    OpFoldResult dimSize = tensor::getMixedSize(rewriter, loc, input, idx);
    newShape.push_back(dimSize);
    newStaticShape.push_back(
        getConstantIntValue(dimSize).value_or(ShapedType::kDynamic));
    exprs.push_back(
        rewriter.getAffineDimExpr(dim < insertSplitDimension ? dim : dim + 1));
    reassociation.push_back({index++});
  }
  SmallVector<AffineMap> newMaps;
  newMaps.push_back(AffineMap::get(map.getNumDims() + 1, 0, exprs, context));

  auto newInputType =
      RankedTensorType::get(newStaticShape, inputType.getElementType());
  Value newInput = tensor::ExpandShapeOp::create(
      rewriter, loc, newInputType, input, reassociation, newShape);

  // Calculate the new output map and shape, we insert the new dimension based
  // on `insertSplitIndex`.
  SmallVector<OpFoldResult> newOutputShape;
  AffineMap oldOutputMap =
      linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(0));
  SmallVector<OpFoldResult> oldOutputShape = tensor::getMixedSizes(
      rewriter, loc, linalgOp.getDpsInitOperand(0)->get());
  SmallVector<AffineExpr> outputExpr;
  for (unsigned idx : llvm::seq<unsigned>(0, oldOutputShape.size() + 1)) {
    if (insertSplitIndex == idx) {
      newOutputShape.push_back(rewriter.getIndexAttr(splitSize));
      outputExpr.push_back(rewriter.getAffineDimExpr(insertSplitDimension));
    }
    if (idx < oldOutputShape.size()) {
      newOutputShape.push_back(oldOutputShape[idx]);
      unsigned dim = oldOutputMap.getDimPosition(idx);
      outputExpr.push_back(rewriter.getAffineDimExpr(
          dim < insertSplitDimension ? dim : dim + 1));
    }
  }
  Value emptyTensor =
      tensor::EmptyOp::create(rewriter, loc, newOutputShape,
                              linalgOp.getRegionOutputArgs()[0].getType());
  auto fillOp =
      linalg::FillOp::create(rewriter, loc, *identityValue, emptyTensor);
  Value identityTensor = fillOp.getResult(0);

  newMaps.push_back(
      AffineMap::get(oldOutputMap.getNumDims() + 1, 0, outputExpr, context));
  SmallVector<utils::IteratorType> newIteratorTypes;
  for (auto [index, iteratorType] :
       llvm::enumerate(linalgOp.getIteratorTypesArray())) {
    if (insertSplitDimension == index) {
      newIteratorTypes.push_back(utils::IteratorType::parallel);
    }
    newIteratorTypes.push_back(iteratorType);
  }
  if (insertSplitDimension == linalgOp.getIteratorTypesArray().size()) {
    newIteratorTypes.push_back(utils::IteratorType::parallel);
  }
  // Create the new op matching the original op with an extra parallel
  // dimension.
  auto genericOp = linalg::GenericOp::create(
      rewriter, loc, TypeRange({emptyTensor.getType()}), ValueRange({newInput}),
      ValueRange({identityTensor}), newMaps, newIteratorTypes);
  rewriter.inlineRegionBefore(linalgOp->getRegion(0), genericOp.getRegion(),
                              genericOp.getRegion().begin());

  // Then create a new reduction that only reduces the newly added dimension
  // from the previous op.
  unsigned intermRank = newOutputShape.size();
  AffineMap inputMap = rewriter.getMultiDimIdentityMap(intermRank);
  SmallVector<utils::IteratorType> reductionIteratorTypes;
  SmallVector<AffineExpr> finalExprs;
  for (unsigned i : llvm::seq<unsigned>(0, intermRank)) {
    if (insertSplitIndex == i) {
      reductionIteratorTypes.push_back(utils::IteratorType::reduction);
    } else {
      finalExprs.push_back(rewriter.getAffineDimExpr(i));
      reductionIteratorTypes.push_back(utils::IteratorType::parallel);
    }
  }
  AffineMap outputMap = AffineMap::get(intermRank, 0, finalExprs, context);
  SmallVector<AffineMap> reductionMaps = {inputMap, outputMap};

  auto reduction = linalg::GenericOp::create(
      rewriter, loc, linalgOp->getResultTypes(),
      ValueRange({genericOp.getResult(0)}), linalgOp.getDpsInits(),
      reductionMaps, reductionIteratorTypes,
      [reductionOp](OpBuilder &b, Location nestedLoc, ValueRange inputs) {
        Operation *clonedReductionOp = b.clone(*reductionOp);
        clonedReductionOp->setOperand(0, inputs[0]);
        clonedReductionOp->setOperand(1, inputs[1]);
        linalg::YieldOp::create(b, nestedLoc, clonedReductionOp->getResult(0));
      });
  rewriter.replaceOp(linalgOp, reduction.getResults());

  return linalg::SplitReductionResult{
      emptyTensor.getDefiningOp(), fillOp,
      cast<linalg::LinalgOp>(genericOp.getOperation()),
      cast<linalg::LinalgOp>(reduction.getOperation())};
}

/// Converts an inner-reduction into outer reduction + inner-parallel dimension,
/// followed by simple inner reduction.
LogicalResult splitReductionImpl(Operation *op, int64_t size,
                                 RewriterBase &rewriter) {
  IRRewriter::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(op);
  linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
  AffineMap map =
      linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(0));
  unsigned lastIdx = map.getNumResults() - 1;
  unsigned numLoops = linalgOp.getNumLoops();

  // 1) Tile to extract a single vector-length array.
  SmallVector<OpFoldResult> tileSizesSVFirst(numLoops,
                                             rewriter.getIndexAttr(1));
  tileSizesSVFirst[numLoops - 1] = rewriter.getIndexAttr(0);
  auto options = scf::SCFTilingOptions().setTileSizes(tileSizesSVFirst);
  FailureOr<scf::SCFTilingResult> tileResFirst = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(linalgOp.getOperation()), options);
  if (failed(tileResFirst)) {
    LDBG() << "failed on step 1 (SCFTiling)";
    return failure();
  }
  rewriter.replaceOp(linalgOp, tileResFirst->replacements);

  // 2) Apply splitReduction on the single vector-length array.
  // splitReduction already replaces the op.
  auto tiledOp = cast<linalg::LinalgOp>(tileResFirst->tiledOps.back());
  FailureOr<OpFoldResult> reductionSize = getReductionSize(tiledOp);
  if (failed(reductionSize)) {
    LDBG() << "failed to determine tiled reduction size";
    return success();
  }
  linalg::SplitReductionResult splitRes =
      splitReduction(tiledOp, size, lastIdx, *reductionSize, rewriter);

  // 3) Tile the first op generated by splitReduction with tile size of 1,
  // to essentially create a reduction loop. Note that
  // splitRes.splitLinalgOp.getNumLoops() = numLoops + 1.
  SmallVector<OpFoldResult> tileSizesSV(splitRes.splitLinalgOp.getNumLoops(),
                                        rewriter.getIndexAttr(0));
  // The reduction happens only in the penultimate dimension, which we now
  // tile.
  tileSizesSV[numLoops - 1] = rewriter.getIndexAttr(1);
  options = scf::SCFTilingOptions().setTileSizes(tileSizesSV);
  FailureOr<scf::SCFTilingResult> tileRes = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(splitRes.splitLinalgOp.getOperation()),
      options);
  if (failed(tileRes)) {
    LDBG() << "failed on step 3 (SCFTiling)";
    return failure();
  }
  rewriter.replaceOp(splitRes.splitLinalgOp, tileRes->replacements);
  return success();
}

/// Pass to splitReduce linalg operations.
class LLVMCPUSplitReductionPass
    : public impl::LLVMCPUSplitReductionPassBase<LLVMCPUSplitReductionPass> {
public:
  using Base::Base;
  explicit LLVMCPUSplitReductionPass(bool fpReductionReordering) {
    this->enableFpReductionReordering = fpReductionReordering;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }
  void runOnOperation() override;
};

void LLVMCPUSplitReductionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

  struct Candidate {
    linalg::GenericOp op;
    int64_t splitSize;
  };

  SmallVector<Candidate> candidates;
  SmallVector<Candidate> dynamicCandidatesNeedingProof;
  funcOp.walk([&](linalg::GenericOp op) {
    LDBG() << "candidate: " << op;
    if (failed(splitReductionPrecondition(op, enableFpReductionReordering))) {
      return;
    }

    IREE::Codegen::LoweringConfigAttrInterface maybeLoweringConfig =
        getLoweringConfig(op);
    if (!maybeLoweringConfig) {
      LDBG() << "can't find lowering_config, skip SplitReduction";
      return;
    }
    auto attr = cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
        maybeLoweringConfig.getTilingLevelAttr(static_cast<unsigned>(
            IREE::CPU::TilingLevel::VectorReductionTiles)));
    ArrayRef<bool> scalableDims = attr.getScalableFlags();
    if (scalableDims.back()) {
      LDBG() << "scalable reduction dimensions not yet supported, skip "
                "SplitReduction";
      return;
    }
    ArrayRef<int64_t> reductionSizes = attr.getSizes();
    if (reductionSizes.empty()) {
      LDBG()
          << "the list of reduction tiling sizes is empty, skip SplitReduction";
      return;
    }

    Candidate candidate = {op, reductionSizes.back()};
    FailureOr<OpFoldResult> reductionSize = getReductionSize(op);
    if (failed(reductionSize)) {
      return;
    }
    if (getConstantIntValue(*reductionSize)) {
      if (canSplitReductionWithSize(op, candidate.splitSize,
                                    /*solver=*/nullptr)) {
        candidates.push_back(candidate);
      }
      return;
    }
    dynamicCandidatesNeedingProof.push_back(candidate);
  });

  std::unique_ptr<DataFlowSolver> solver;
  if (!dynamicCandidatesNeedingProof.empty()) {
    solver = std::make_unique<DataFlowSolver>();
    solver->load<dataflow::SparseConstantPropagation>();
    solver->load<dataflow::DeadCodeAnalysis>();
    solver->load<IREE::Util::IntegerDivisibilityAnalysis>();
    if (failed(solver->initializeAndRun(funcOp))) {
      return signalPassFailure();
    }
  }

  for (const Candidate &candidate : dynamicCandidatesNeedingProof) {
    if (canSplitReductionWithSize(candidate.op, candidate.splitSize,
                                  solver.get())) {
      candidates.push_back(candidate);
    }
  }

  IRRewriter rewriter(context);
  for (const Candidate &candidate : candidates) {
    if (failed(
            splitReductionImpl(candidate.op, candidate.splitSize, rewriter))) {
      return signalPassFailure();
    }
  }
}
} // namespace
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUSplitReductionPass(const bool enableFpReductionReordering) {
  return std::make_unique<LLVMCPUSplitReductionPass>(
      enableFpReductionReordering);
}
} // namespace mlir::iree_compiler
