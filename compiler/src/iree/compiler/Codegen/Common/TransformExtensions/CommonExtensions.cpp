// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CommonExtensions.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgTransform/SimplePatternRewriter.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/ListenerGreedyPatternRewriteDriver.h"
#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Interfaces/BufferizationInterfaces.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
// #include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"

#define DEBUG_TYPE "iree-common-extensions-transforms"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE;

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")

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

//===---------------------------------------------------------------------===//
// ApplyPatternsOp
//===---------------------------------------------------------------------===//
void transform_dialect::ApplyPatternsOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    const ApplyPatternsOpPatterns &patterns) {
  MLIRContext *ctx = builder.getContext();
  result.addOperands(target);
  auto unitAttr = builder.getUnitAttr();
#define ADD_PATTERN(NAME, ATTR) \
  if (patterns.NAME)            \
    result.addAttribute(ApplyPatternsOp::ATTR(result.name), unitAttr);
  ADD_PATTERN(additionalIreePatterns, getAdditionalIreePatternsAttrName)
  ADD_PATTERN(bubbleCollapseExpand, getBubbleCollapseExpandAttrName)
  ADD_PATTERN(canonicalization, getCanonicalizationAttrName)
  ADD_PATTERN(eraseUnnecessaryTensorOperands,
              getEraseUnnecessaryTensorOperandsAttrName)
  ADD_PATTERN(foldMemrefAliases, getFoldMemrefAliasesAttrName)
  ADD_PATTERN(foldReassociativeReshapes, getFoldReassociativeReshapesAttrName)
  ADD_PATTERN(lowerTransferOpPermutations,
              getLowerTransferOpPermutationsAttrName)
  ADD_PATTERN(rankReducingLinalg, getRankReducingLinalgAttrName)
  ADD_PATTERN(rankReducingVector, getRankReducingVectorAttrName)
  ADD_PATTERN(expandMemrefStridedMetadata,
              getExpandMemrefStridedMetadataAttrName)
  ADD_PATTERN(swapPaddingElideConditional,
              getSwapPaddingElideConditionalAttrName)
  ADD_PATTERN(swappingPatterns, getSwappingPatternsAttrName)
#undef ADD_PATTERN
  result.addTypes({pdl::OperationType::get(ctx)});
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
}  // namespace

static void addLowerTransferOpPermutationsPatterns(
    RewritePatternSet &patterns) {
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
}

static void addFoldMemrefAliasPatterns(RewritePatternSet &patterns) {
  memref::populateFoldMemRefAliasOpPatterns(patterns);
}

static void addReassociativeReshapePatterns(RewritePatternSet &patterns) {
  tensor::populateReassociativeReshapeFoldingPatterns(patterns);
}

static void addEraseUnnecessaryTensorOperandsPatterns(
    RewritePatternSet &patterns) {
  linalg::populateEraseUnnecessaryInputsPatterns(patterns);
}

static void addRankReducingLinalgPatterns(RewritePatternSet &patterns) {
  populateReshapeToInterfaceTensorPatterns(patterns);
  linalg::populateFoldUnitExtentDimsViaSlicesPatterns(patterns);
}

static void addRankReducingVectorPatterns(RewritePatternSet &patterns) {
  vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
}

static void addSwappingPatterns(RewritePatternSet &patterns,
                                bool swapPaddingElideCornerCase) {
  patterns.add<linalg::ExtractSliceOfPadTensorSwapPattern>(
      patterns.getContext(),
      [&](tensor::ExtractSliceOp) -> llvm::Optional<bool> {
        return !swapPaddingElideCornerCase;
      });
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
  if (getCanonicalization()) addAllRegisteredCanonicalizationPatterns(patterns);
  if (getLowerTransferOpPermutations())
    addLowerTransferOpPermutationsPatterns(patterns);
  if (getEraseUnnecessaryTensorOperands())
    addEraseUnnecessaryTensorOperandsPatterns(patterns);
  if (getFoldMemrefAliases()) addFoldMemrefAliasPatterns(patterns);
  if (getFoldReassociativeReshapes()) addReassociativeReshapePatterns(patterns);
  if (getRankReducingLinalg()) addRankReducingLinalgPatterns(patterns);
  if (getRankReducingVector()) addRankReducingVectorPatterns(patterns);
  if (getExpandMemrefStridedMetadata())
    memref::populateExpandStridedMetadataPatterns(patterns);
  if (getSwappingPatterns())
    addSwappingPatterns(patterns, getSwapPaddingElideConditional());
  if (getAdditionalIreePatterns()) addAdditionalIreePatterns(patterns);
  if (getBubbleCollapseExpand()) {
    linalg::populateFoldReshapeOpsByExpansionPatterns(
        patterns, [](OpOperand *) { return true; });
  }

  TrackingListener listener(state);
  GreedyRewriteConfig config;
  LogicalResult result = applyPatternsAndFoldGreedily(
      target, std::move(patterns), config, &listener);
  LogicalResult listenerResult = listener.checkErrorState();
  if (failed(result)) {
    return mlir::emitDefiniteFailure(target,
                                     "greedy pattern application failed");
  }
  if (failed(listenerResult))
    return mlir::emitDefiniteFailure(target, "listener tracking failed");

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::ApplyPatternsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::producesHandle(getResult(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// ShareForeachThreadOperandsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::ShareForeachThreadOperandsOp::applyToOne(
    scf::ForeachThreadOp foreachThreadOp,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  IRRewriter rewriter(getContext());
  SmallVector<int64_t> shareOperands(getShareOperands());
  // Empty case: consider all operands need to be shared.
  if (shareOperands.empty()) {
    shareOperands = llvm::to_vector(
        llvm::seq<int64_t>(0, foreachThreadOp.getOutputs().size()));
  }
  for (int64_t outputIdx : getShareOperands()) {
    if (outputIdx < 0 || outputIdx >= foreachThreadOp.getOutputs().size())
      return mlir::emitDefiniteFailure(foreachThreadOp, "operand idx overflow");
    Value toShare = foreachThreadOp.getOutputs()[outputIdx];
    if (std::distance(toShare.getUses().begin(), toShare.getUses().end()) !=
        2) {
      /*return mlir::emitSilenceableFailure(
          foreachThreadOp,
          "operand to share must have exactly 2 uses, the foreach_thread op "
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
        foreachThreadOp,
        "shared operands use must be extractSliceOp.");*/
      continue;
    }
    // Get the corresponding bbArg.
    BlockArgument bbArg = foreachThreadOp.getOutputBlockArguments()[outputIdx];

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
    if (llvm::none_of(foreachThreadOp.getTerminator().getYieldingOps(),
                      isMatchingParallelInsertSlice)) {
      continue;
    }

    // Promote extract_slice source to bbArg.
    rewriter.updateRootInPlace(extractSliceOp, [&]() {
      extractSliceOp.getSourceMutable().assign(bbArg);
    });
  }

  results.push_back(foreachThreadOp);
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// ForeachThreadToWorkgroupOp
//===---------------------------------------------------------------------===//

void transform_dialect::ForeachThreadToWorkgroupOp::build(
    OpBuilder &builder, OperationState &result, Value target) {
  result.addOperands(target);
  MLIRContext *ctx = builder.getContext();
  result.addTypes({pdl::OperationType::get(ctx)});
}

/// Populate the workgroup_count region of `dispatchOp`.
/// For now, this only supports constant index ops and empty workload
/// operands. Assumes the HAL::ExecutableExportOp is built with an empty
/// region.
static LogicalResult populateWorkgroupCountComputingRegion(
    PatternRewriter &rewriter, scf::ForeachThreadOp foreachThreadOp,
    HAL::ExecutableExportOp exportOp) {
  Location loc = foreachThreadOp.getLoc();
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
  for (Value v : foreachThreadOp.getNumThreads()) {
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
// Patterns for ForeachThreadToWorkgroup rewrite.
//===---------------------------------------------------------------------===//

LogicalResult rewriteForeachThreadToWorkgroup(
    scf::ForeachThreadOp foreachThreadOp,
    IREE::HAL::ExecutableExportOp exportOp, PatternRewriter &rewriter) {
  // Step 0. Target-specific verifications. There is no good place to anchor
  // those right now: the ForeachThreadOp is target-independent and the
  // transform op does not apply to individual ForeachThreadOp.
  MLIRContext *ctx = foreachThreadOp->getContext();
  Location loc = foreachThreadOp->getLoc();
  // TODO iree should have own device mapping like #hal.workgroup<x/y/z>
  Attribute bX = gpu::GPUBlockMappingAttr::get(ctx, gpu::Blocks::DimX);
  Attribute bY = gpu::GPUBlockMappingAttr::get(ctx, gpu::Blocks::DimY);
  Attribute bZ = gpu::GPUBlockMappingAttr::get(ctx, gpu::Blocks::DimZ);
  if (foreachThreadOp.getNumResults() > 0)
    return foreachThreadOp->emitError(
        "only bufferized scf.foreach_thread lowers to workgroup");
  if (foreachThreadOp.getNumThreads().size() > 3)
    return foreachThreadOp->emitError(
        "scf.foreach_thread with rank > 3 does not lower to workgroup");

  if (!foreachThreadOp.getMapping().has_value())
    return foreachThreadOp->emitError("mapping must be present");
  SmallVector<Attribute> blockMapping =
      llvm::to_vector(foreachThreadOp.getMapping()->getValue());
  if (llvm::any_of(blockMapping, [](DeviceMappingAttrInterface map) {
        return !map.isa<gpu::GPUBlockMappingAttr>();
      })) {
    return foreachThreadOp->emitError("mapping must be #gpu.block<x/y/z/>");
  }

  // Step 1. Complete the blockMapping to a full mapping (with 1s) if necessary.
  SmallVector<Value> numBlocks =
      llvm::to_vector(foreachThreadOp.getNumThreads());
  // Ensure we have 3 block sizes, one for each id.
  Value one;
  for (auto attr : {bX, bY, bZ}) {
    if (std::find(blockMapping.begin(), blockMapping.end(), attr) ==
        blockMapping.end()) {
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
  SmallVector<Value> gridDimValues = scf::ForeachThreadOp::getValuesSortedByKey(
      blockMapping, numBlocks, comparator);

  // Step 3. Outline the compute workload region and set up the workload
  // operands, if this has not been done already.
  // Using `transform.iree.tile_to_foreach_thread_and_workgroup_count_region` is
  // the preferred way to set up tiling and workgroup_count region **at the same
  // time**.
  //
  // The block of code below will be retired once there is enough confidence we
  // can do everything without it. This includes in particular providing custom
  // fusion heuristics at the flow level: at this time, the only way to fully
  // control fusion of more advanced cases is to use the transform dialect at
  // the flow level and explicitly match the ops we want to fuse.
  // Once fusion is customizable enough in perpetuity, we can retire this.
  if (exportOp.getWorkgroupCount().empty()) {
    if (llvm::any_of(foreachThreadOp.getNumThreads(), [](Value v) {
          return !v.getDefiningOp<arith::ConstantIndexOp>();
        })) {
      return foreachThreadOp->emitError(
          "unsupported dynamic workgroup_count atm --- need to slice out "
          "workgroup_count computation into ExecutableExport::workgroup_count."
          "\nThis region may require arbitrary computations and cannot "
          "magically match what the `stream.cmd.dispatch` has already imposed "
          "on us at a distance."
          "\nFor now we must specify the number of values properly when "
          "applying the topLevel tile_to_foreach_thread_op");
    }
    if (failed(populateWorkgroupCountComputingRegion(rewriter, foreachThreadOp,
                                                     exportOp))) {
      return foreachThreadOp->emitOpError(
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
  bvm.map(foreachThreadOp.getThreadIndices(), workgroupIdOps);
  bvm.map(foreachThreadOp.getNumThreads(), workgroupCountOps);

  // Step 5. Predicate omitted given unique topLevel scf::ForeachThreadOp.

  // Step 6. Move the body of foreachThreadOp.
  // Erase the terminator first, it will not be used since we are on buffers.
  rewriter.eraseOp(foreachThreadOp.getTerminator());
  Block *targetBlock;
  Block::iterator insertionPoint;
  targetBlock = foreachThreadOp->getBlock();
  insertionPoint = Block::iterator(foreachThreadOp);
  Block &sourceBlock = foreachThreadOp.getRegion().front();
  targetBlock->getOperations().splice(insertionPoint,
                                      sourceBlock.getOperations());

  // Step 7. RAUW thread indices to thread ops.
  for (Value blockIdx : foreachThreadOp.getThreadIndices()) {
    for (Operation *user : llvm::make_early_inc_range(blockIdx.getUsers())) {
      rewriter.updateRootInPlace(user, [&]() {
        user->replaceUsesOfWith(blockIdx, bvm.lookup(blockIdx));
      });
    }
  }

  // Step 5. Barriers omitted given unique topLevel scf::ForeachThreadOp.

  // Step 6. Erase old op.
  rewriter.eraseOp(foreachThreadOp);

  return success();
}

//===---------------------------------------------------------------------===//
// PackGreedilyOp.
//===---------------------------------------------------------------------===//

namespace {
auto par = utils::IteratorType::parallel;
auto red = utils::IteratorType::reduction;
}  // namespace

static int64_t numResultsFunctionOf(AffineMap map, AffineDimExpr d) {
  int64_t count = 0;
  for (AffineExpr e : map.getResults())
    if (e.isFunctionOfDim(d.getPosition())) ++count;
  return count;
}

/// Return the set of AffineDimExpr
static DenseSet<int64_t> findPermutationsIndexingOperand(
    linalg::LinalgOp linalgOp, OpOperand *opOperand, utils::IteratorType iter) {
  DenseSet<int64_t> res;
  assert(linalgOp == opOperand->getOwner() && "expected linalgOp owner");
  AffineMap indexingMap = linalgOp.getMatchingIndexingMap(opOperand);
  for (AffineExpr e : indexingMap.getResults()) {
    if (auto d = e.dyn_cast<AffineDimExpr>()) {
      if (linalgOp.getIteratorTypesArray()[d.getPosition()] == iter &&
          numResultsFunctionOf(indexingMap, d) == 1)
        res.insert(d.getPosition());
    }
  }
  return res;
}

struct ContractionDimsForPacking {
  int64_t mPos, nPos, kPos;
};
/// Greedily look for 2 parallel (m and n) and 1 reduction (k) dimension that
/// form a contraction. Such dimensions are such that:
///   1. The m dimension is involved in an outer-product along LHS
///      (i.e. it is a permutation on RES and LHS and does not appear in RHS).
///   2. The n dimension is involved in an outer-product along RHS
///      (i.e. it is a permutation on RES and RHS and does not appear in LHS).
///   3. The k dimension appears as a permutation on LHS and RHS.
///   4. m, n and k appear only once in any given indexing.
///
/// This allows detecting that some contraction is embedded within `linalgOp`.
///
/// When multiple possibilities for selecting m, n and k appear, we just pick
/// an arbitrary one (i.e. the first in a DenseSet).
// TODO: Better heuristic (e.g pick dims based on packing-based metric).
static FailureOr<ContractionDimsForPacking> getContractionDims(
    linalg::LinalgOp linalgOp) {
  assert(linalgOp.getNumDpsInits() == 1 && "wrong number of dps inits");
  assert(linalgOp.getNumDpsInputs() == 2 && "wrong number of dps inputs");

  DenseSet<int64_t> a = findPermutationsIndexingOperand(
      linalgOp, linalgOp.getDpsInputOperand(0), par);
  DenseSet<int64_t> b = findPermutationsIndexingOperand(
      linalgOp, linalgOp.getDpsInputOperand(1), par);
  DenseSet<int64_t> c = findPermutationsIndexingOperand(
      linalgOp, linalgOp.getDpsInitOperand(0), par);

  // A & C - B are the iterators involved in an outer-product along A (the LHS).
  DenseSet<int64_t> ac = a;
  llvm::set_intersect(ac, c);
  llvm::set_subtract(ac, b);
  // B & C - A are the iterators involved in an outer-product along B (the RHS).
  DenseSet<int64_t> bc = b;
  llvm::set_intersect(bc, c);
  llvm::set_subtract(bc, a);

  // Note: if we ever need them, A & B & C would be "batch" dimensions.

  // A & B red are the reduction dimensions.
  DenseSet<int64_t> ra = findPermutationsIndexingOperand(
      linalgOp, linalgOp.getDpsInputOperand(0), red);
  DenseSet<int64_t> rb = findPermutationsIndexingOperand(
      linalgOp, linalgOp.getDpsInputOperand(1), red);
  llvm::set_intersect(ra, rb);

  if (ac.empty() || bc.empty() || ra.empty()) return failure();

  // Pick the first one in each set.
  // TODO: Better heuristic (e.g pick dims based on packing-based metric).
  return ContractionDimsForPacking{*ac.begin(), *bc.begin(), *ra.begin()};
}

/// Return a permutation vector of size permSize that would result in moving
/// positions into desiredPositions.
///
/// For example, permSize == 5, positions = {2, 4}, desiredPositions = {1, 0}
/// would result in a {4, 2, 0, 1, 3} permutation vector.
static SmallVector<int64_t> computePermutationVector(
    int64_t permSize, ArrayRef<int64_t> positions,
    ArrayRef<int64_t> desiredPositions) {
  SmallVector<int64_t> res(permSize, -1);
  DenseSet<int64_t> seen;
  for (auto [pos, desiredPos] : llvm::zip(positions, desiredPositions)) {
    res[desiredPos] = pos;
    seen.insert(pos);
  }
  int64_t nextPos = 0;
  for (int64_t &entry : res) {
    if (entry != -1) continue;
    while (seen.contains(nextPos)) ++nextPos;
    entry = nextPos;
    ++nextPos;
  }
  return res;
}

/// Pack a LinalgOp by greedily inferring 2-D contraction dimensions (m, n, k)
/// where m and n are proper parallel dimensions and k is a proper reduction
/// dimension.
/// Packing occurs by rewriting the op as a linalg.generic and calling
/// linalg::pack by `mnkPackingSizes`.
/// The order of the packed dimensions is customizable: the `mnkOrder` is a
/// permutation of {0, 1, 2} to reorder {m, n, k} into one of the 8 possible
/// forms.
/// The outer dimensions of the operands are not permuted at this time, this is
/// left for future work.
static LogicalResult packContractionGreedily(
    RewriterBase &rewriter, linalg::LinalgOp linalgOp,
    ArrayRef<OpFoldResult> mnkPackingSizes, ArrayRef<int64_t> mnkOrder) {
  assert(mnkPackingSizes.size() == 3 && "unexpected num of packing sizes");
  assert(mnkOrder.size() == 3 && "unexpected mnkOrder size");

  int64_t numLoops = linalgOp.getNumLoops();
  if (numLoops <= 2)
    return rewriter.notifyMatchFailure(linalgOp,
                                       "need 3+ loops to pack a contraction");

  // Locally adjust the desired iterator position of mnk and packing sizes.
  int64_t numPackedDims = mnkPackingSizes.size();
  SmallVector<int64_t> mmnnkkPos(numPackedDims);
  for (int64_t i = 0, e = numPackedDims; i < e; ++i)
    mmnnkkPos[i] = numLoops - numPackedDims + mnkOrder[i];
  SmallVector<OpFoldResult> packingSizes(mnkPackingSizes.size());
  for (int64_t i = 0, e = numPackedDims; i < e; ++i)
    packingSizes[mnkOrder[i]] = mnkPackingSizes[i];

  // 1. Infer dims that are important for contraction.
  FailureOr<ContractionDimsForPacking> res = getContractionDims(linalgOp);
  if (failed(res)) {
    return rewriter.notifyMatchFailure(linalgOp,
                                       "couldn't infer contraction iterators");
  }

  // 2. Normalize linalgOp to an kmn-matmul-like with [red, par, par] most
  // minor iterators. If we wanted a different normalization order, this is
  // where it would have to start.
  int64_t mPos = res->mPos, nPos = res->nPos, kPos = res->kPos;
  LLVM_DEBUG(DBGSNL(); DBGSNL(); DBGSNL();
             DBGS() << "Start packing generic op greedily with (m@" << mPos
                    << ", n@" << nPos << ", k@" << kPos << "): " << linalgOp
                    << "\n";);

  // 2.a. Rewrite as a generic.
  auto genericOp = dyn_cast<linalg::GenericOp>(linalgOp.getOperation());
  if (!genericOp) {
    FailureOr<linalg::GenericOp> generalizeResult =
        generalizeNamedOp(rewriter, linalgOp);
    assert(succeeded(generalizeResult) && "unexpected failure generalizing op");
    genericOp = *generalizeResult;
  }

  // 2.b. Interchange to move the dimensions (k, m, n) as most-minor iterators.
  // Note that this only normalized the iteration order and does not change the
  // indexings of any operand.
  SmallVector<int64_t> permutation =
      computePermutationVector(numLoops, {mPos, nPos, kPos}, mmnnkkPos);
  LLVM_DEBUG(llvm::interleaveComma(permutation, DBGS() << "perm: "); DBGSNL(););
  // Sign .. unsigned pollution.
  SmallVector<unsigned> unsignedPerm(permutation.begin(), permutation.end());
  FailureOr<linalg::GenericOp> interchangeResult =
      interchangeGenericOp(rewriter, genericOp, unsignedPerm);
  assert(succeeded(interchangeResult) && "unexpected failure interchanging op");
  genericOp = *interchangeResult;
  LLVM_DEBUG(DBGS() << "Generalized Op to pack: " << genericOp << "\n";);

  // At this point, the op iterators are normalized to {leading, k, m, n}.
  // The layouts induced by packing will always be:
  //   - LHS{leading_lhs, kk, mm}
  //   - RHS{leading_rhs, kk, nn}
  //   - RES{leading_res, mm, nn}
  // If we wanted to change the packed order, we would reorder (k, m, n) to
  // something else above.
  //
  // Additional permutations of the outer dims of the operands (i.e.
  // leading_lhs, leading_rhs and leading_res) could follow by computing the
  // desired outerPerm for each operand. This is left for future work.

  // Add leading zeros to match numLoops.
  SmallVector<OpFoldResult> adjustedPackingSizes(numLoops - packingSizes.size(),
                                                 rewriter.getIndexAttr(0));
  llvm::append_range(adjustedPackingSizes, packingSizes);

  // TODO: If we wanted to give the genericOp a name after packing, after
  // calling `pack` would be a good time.
  return linalg::pack(rewriter, genericOp, adjustedPackingSizes);
}

DiagnosedSilenceableFailure transform_dialect::PackGreedilyOp::applyToOne(
    func::FuncOp target, transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  IRRewriter rewriter(getContext());
  target->walk([&](linalg::LinalgOp linalgOp) {
    // linalgOp will be replaced and the insertion point may be invalidated if
    // we set it before -> set it after.
    rewriter.setInsertionPointAfter(linalgOp);
    // Failing to pack greedily is perfectly fine.
    // In the future we will want to order packings according to some metric.
    // For now we just pack contractions embedded in ops in the order:
    //   {kk, mm, nn} by size {32, 8, 16}.
    (void)packContractionGreedily(
        /*rewriter=*/rewriter,
        /*linalgOp=*/linalgOp,
        /*mnkPackingSizes=*/
        getAsOpFoldResult(
            rewriter.getI64ArrayAttr({/*m=*/8, /*n=*/16, /*k=*/32})),
        /*mnkOrder=*/{1, 2, 0});
  });
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// IREE-specific transformations defined outside of iree_linalg_transform.
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::ForeachThreadToWorkgroupOp::applyToOne(
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
    results.assign(1, nullptr);
    return mlir::emitSilenceableFailure(
        target, "no IREE::HAL::ExecutableExportOp found");
  }

  scf::ForeachThreadOp topLevelForeachThreadOp;
  auto walkResult = target->walk([&](scf::ForeachThreadOp foreachThreadOp) {
    if (foreachThreadOp->getParentOfType<scf::ForeachThreadOp>())
      return WalkResult::advance();
    if (topLevelForeachThreadOp) return WalkResult::interrupt();
    topLevelForeachThreadOp = foreachThreadOp;
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    results.assign(1, nullptr);
    return mlir::emitSilenceableFailure(
        target, "could not find a unique topLevel scf.foreach_thread");
  }

  SimplePatternRewriter rewriter(topLevelForeachThreadOp);
  if (failed(rewriteForeachThreadToWorkgroup(topLevelForeachThreadOp, exportOp,
                                             rewriter))) {
    return mlir::emitDefiniteFailure(target,
                                     "rewriteForeachThreadToWorkgroup failed");
  }

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// TileToForeachThreadAndWorkgroupCountRegionOp
//===---------------------------------------------------------------------===//

void transform_dialect::TileToForeachThreadAndWorkgroupCountRegionOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    ArrayRef<int64_t> staticTileSizes, transform::TileSizesSpec,
    ArrayAttr mappingAttr) {
  return build(builder, result, target,
               /*mixedTileSizes=*/
               getAsOpFoldResult(builder.getI64ArrayAttr(staticTileSizes)),
               /*_=*/transform::TileSizesSpec(), /*mapping=*/mappingAttr);
}

void transform_dialect::TileToForeachThreadAndWorkgroupCountRegionOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    ArrayRef<OpFoldResult> mixedTileSizes, transform::TileSizesSpec,
    ArrayAttr mappingAttr) {
  assert(result.name.isRegistered() && "not registered!!");
  SmallVector<int64_t> staticTileSizes;
  SmallVector<Value> dynamicTileSizes;
  dispatchIndexOpFoldResults(mixedTileSizes, dynamicTileSizes, staticTileSizes);
  // Call the default builder which sets up the proper operands segment
  // sizes attributes for multiple variadic operands. In the absence of
  // this, horrible bugs ensue.
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

void transform_dialect::TileToForeachThreadAndWorkgroupCountRegionOp::build(
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

void transform_dialect::TileToForeachThreadAndWorkgroupCountRegionOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    ArrayRef<OpFoldResult> mixedNumThreads, transform::NumThreadsSpec,
    ArrayAttr mappingAttr) {
  assert(result.name.isRegistered() && "not registered!!");
  SmallVector<int64_t> staticNumThreads;
  SmallVector<Value> dynamicNumThreads;
  dispatchIndexOpFoldResults(mixedNumThreads, dynamicNumThreads,
                             staticNumThreads);
  // Call the default builder which sets up the proper operands segment
  // sizes attributes for multiple variadic operands. In the absence of
  // this, horrible bugs ensue.
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
    HAL::ExecutableExportOp exportOp, ArrayRef<OpFoldResult> tileSizes,
    Optional<ArrayAttr> mapping) {
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

  SmallVector<OpFoldResult> unpackedTileSizes;
  int64_t numTiledDims = 0;
  for (auto ofr : tileSizes) {
    if (ofr.is<Value>() &&
        ofr.get<Value>().getType().isa<pdl::OperationType>()) {
      for (Operation *sizeProducer : state.getPayloadOps(ofr.get<Value>())) {
        if (sizeProducer->getNumResults() != 1) {
          auto diag =
              mlir::emitDefiniteFailure(sizeProducer)
              << "the operation producing tile size must have one result";
          diag.attachNote(loc) << "when applying this transform";
          return diag;
        }
        unpackedTileSizes.push_back(sizeProducer->getResult(0));
      }
    } else {
      unpackedTileSizes.push_back(ofr);
    }
    if (!isConstantIntValue(unpackedTileSizes.back(), 0)) ++numTiledDims;
  }

  if (unpackedTileSizes.size() > workload.size()) {
    return rewriter.notifyMatchFailure(
        exportOp,
        "number of tile sizes overflow the dimension from the workload");
  }

  // Generate permutation of tiled dims based on the specified mapping.
  SmallVector<int64_t> mappingPermutation;
  if (mapping.has_value()) {
    if (numTiledDims != mapping->size()) {
      return rewriter.notifyMatchFailure(exportOp,
                                         "number of mapping elements must "
                                         "match number of non-zero tile sizes");
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
    while (isConstantIntValue(unpackedTileSizes[nextTiledDim], 0))
      ++nextTiledDim;
    AffineExpr s0, s1;
    bindSymbols(rewriter.getContext(), s0, s1);
    auto m = AffineMap::get(0, 2, s0.ceilDiv(s1));
    workgroupCount[workgroupsDim] = makeComposedFoldedAffineApply(
        rewriter, loc, m,
        ArrayRef<OpFoldResult>{workload[nextTiledDim],
                               unpackedTileSizes[nextTiledDim]});
    ++nextTiledDim;
  }

  rewriter.replaceOp(workgroupCountOp, getValueOrCreateConstantIndexOp(
                                           rewriter, loc, workgroupCount));
  return success();
}

SmallVector<OpFoldResult> transform_dialect::
    TileToForeachThreadAndWorkgroupCountRegionOp::getMixedNumThreads() {
  Builder b(getContext());
  return getMixedValues(getStaticNumThreads(), getNumThreads(), b);
}

SmallVector<OpFoldResult> transform_dialect::
    TileToForeachThreadAndWorkgroupCountRegionOp::getMixedTileSizes() {
  Builder b(getContext());
  return getMixedValues(getStaticTileSizes(), getTileSizes(), b);
}

LogicalResult
transform_dialect::TileToForeachThreadAndWorkgroupCountRegionOp::verify() {
  if (getMixedNumThreads().empty() == getMixedTileSizes().empty())
    return emitOpError("either num_threads or tile_sizes must be specified");
  return success();
}

void transform_dialect::TileToForeachThreadAndWorkgroupCountRegionOp::
    getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getTarget(), effects);
  transform::onlyReadsHandle(getTileSizes(), effects);
  transform::onlyReadsHandle(getNumThreads(), effects);
  transform::producesHandle(getResults(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure
transform_dialect::TileToForeachThreadAndWorkgroupCountRegionOp::apply(
    transform::TransformResults &transformResults,
    transform::TransformState &state) {
  ArrayRef<Operation *> targetOps = state.getPayloadOps(getTarget());
  assert(targetOps.size() == 1 && "expected single target op in payload");
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
          state, rewriter, getLoc(), exportOp.value(), getMixedTileSizes(),
          getMapping()))) {
    return mlir::emitDefiniteFailure(exportOp.value(),
                                     "failed to lower workgroup count region");
  }

  ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());

  // Result payload ops.
  SmallVector<Operation *> tileOps;
  SmallVector<Operation *> tiledOps;

  DiagnosedSilenceableFailure diag = transform::tileToForeachThreadOpImpl(
      rewriter, state, cast<transform::TransformOpInterface>(getOperation()),
      targets, getMixedNumThreads(), getMixedTileSizes(), getMapping(), tileOps,
      tiledOps);

  if (!diag.succeeded()) {
    transformResults.set(getForeachThreadOp().cast<OpResult>(),
                         SmallVector<mlir::Operation *>{});
    transformResults.set(getTiledOp().cast<OpResult>(),
                         SmallVector<mlir::Operation *>{});
    return diag;
  }

  transformResults.set(getForeachThreadOp().cast<OpResult>(), tileOps);
  transformResults.set(getTiledOp().cast<OpResult>(), tiledOps);
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// IREEBufferizeOp
//===---------------------------------------------------------------------===//

// Important note: this transform is load-bearing and is the glue between
// different dialects that want to operate on tensors.
// Originaly, it used to just call `addIREEComprehensiveBufferizePasses` but
// this introduces a lot of complexity in the registration process due to
// the use of nested pass pipelines, to a point that it is a major endeavor
// to connect a new dialect. Instead, avoid calling the passes and only take
// what we need from them.
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
// TODO: Maybe bufferize should have a separate cpu and a gpu version. This
// is unclear though: what happens on heterogeneous HW ?
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
  // as an OpDSL named op. However, IREE-specific patterns to cleanup
  // spurious post-bufferization copies do not trigger properly. So we keep
  // using `createLinalgCopyOp` which builds a GenericOp.
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
  // TODO: ideally we should use linalg.copy which was recently reintroduced
  // as an OpDSL named op. However, IREE-specific patterns to cleanup
  // spurious post-bufferization copies do not trigger properly. So we keep
  // using `createLinalgCopyOp` which builds a GenericOp.
  // builder.create<linalg::CopyOp>(loc, from, to);
  mlir::iree_compiler::createLinalgCopyOp(builder, loc, from, to);
  return success();
}

static OneShotBufferizationOptions getBufferizationOptions() {
  OneShotBufferizationOptions options;
  // options.testAnalysisOnly = testAnalysisOnly;
  // options.printConflicts = printConflicts;

  // bufferization.to_memref is used to bufferize constants in IREE. IREE
  // has it's own logic to handle constants. We'd like to leave the
  // arith.constant as is and insert bufferization.to_memref to convert the
  // tensor to memref.
  options.opFilter.denyOperation<arith::ConstantOp>();
  options.opFilter.denyOperation<bufferization::ToMemrefOp>();

  // This type converter converts tensor types to memref types when no exact
  // memref type can be inferred from the context.
  options.unknownTypeConverterFn = [](Value value, Attribute memorySpace,
                                      const BufferizationOptions &options) {
    auto tensorType = value.getType().cast<TensorType>();

    // Special rule for ConstantOps: These always lower to some memref with
    // a static identity layout.
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
  // DO NOT JUST CALL `addIREEComprehensiveBufferizePasses` as this results
  // in a lot of registration issues due to nested pass pipeline mess.
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

  //   1. Rewrite tensor.empty to tensor.alloc, without the pass baggage.
  {
    RewritePatternSet patterns(getContext());
    patterns.add<EmptyTensorLoweringPattern>(patterns.getContext());
    TrackingListener listener(state);
    GreedyRewriteConfig config;
    LogicalResult result = applyPatternsAndFoldGreedily(
        state.getTopLevel(), std::move(patterns), config, &listener);
    LogicalResult listenerResult = listener.checkErrorState();
    if (failed(result)) {
      return mlir::emitDefiniteFailure(state.getTopLevel(),
                                       "greedy pattern application failed");
    }
    if (failed(listenerResult))
      return mlir::emitDefiniteFailure(state.getTopLevel(),
                                       "listener tracking failed");
  }

  //   2. Run one-shot-bufferize, without the pass baggage.
  OneShotBufferizationOptions options = getBufferizationOptions();
  options.allocationFn = allocationFn;
  options.deallocationFn = deallocationFn;
  options.memCpyFn = memCpyFn;
  options.testAnalysisOnly = getTestAnalysisOnly();
  options.printConflicts = getPrintConflicts();
  WalkResult res = state.getTopLevel()->walk([&](ModuleOp moduleOp) {
    if (failed(runIREEOneShotBufferize(moduleOp, options)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (res.wasInterrupted())
    return DiagnosedSilenceableFailure::definiteFailure();

  // Early exit if test_analysis_only is set.
  if (getTestAnalysisOnly()) {
    results.set(getOperation()->getOpResult(0), payload.front());
    return DiagnosedSilenceableFailure::success();
  }

  //   3. Post-bufferization passes are fine.
  PassManager pm(getContext());
  addIREEPostBufferizationPasses(pm);
  res = state.getTopLevel()->walk([&](ModuleOp moduleOp) {
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
    return DiagnosedSilenceableFailure::definiteFailure();

  results.set(getOperation()->getOpResult(0), payload.front());
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// IREEEliminateEmptyTensorsOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::IREEEliminateEmptyTensorsOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  ArrayRef<Operation *> payloads = state.getPayloadOps(getTarget());
  for (Operation *payload : payloads) {
    if (failed(eliminateEmptyTensors(payload, getBufferizationOptions()))) {
      getOperation()->emitError() << "failed to eliminate tensor.empty ops";
      return DiagnosedSilenceableFailure::definiteFailure();
    }
  }
  results.set(getOperation()->getOpResult(0), payloads);
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::IREEEliminateEmptyTensorsOp::build(
    OpBuilder &builder, OperationState &result, Value target) {
  result.addOperands(target);
  MLIRContext *ctx = builder.getContext();
  result.addTypes(pdl::OperationType::get(ctx));
}

//===---------------------------------------------------------------------===//
// ConfigExtractPart
//===---------------------------------------------------------------------===//
void transform_dialect::ConfigExtractPart::build(OpBuilder &builder,
                                                 OperationState &result,
                                                 Value target,
                                                 StringRef attrName,
                                                 Optional<int64_t> maybeLevel) {
  MLIRContext *ctx = builder.getContext();
  result.addOperands(target);
  result.addAttribute(ConfigExtractPart::getAttrNameAttrName(result.name),
                      builder.getStringAttr(attrName));
  if (maybeLevel) {
    result.addAttribute(ConfigExtractPart::getLevelAttrName(result.name),
                        builder.getI64IntegerAttr(*maybeLevel));
  }
  result.addTypes({pdl::OperationType::get(ctx)});
}

void transform_dialect::ConfigExtractPart::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::producesHandle(getResultConfigPart(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure transform_dialect::ConfigExtractPart::apply(
    transform::TransformResults &transformResults,
    transform::TransformState &state) {
  ArrayRef<Operation *> targetOps = state.getPayloadOps(getTarget());
  if (targetOps.empty()) {
    transformResults.set(getResultConfigPart().cast<OpResult>(), {});
    return DiagnosedSilenceableFailure::success();
  }

  assert(targetOps.size() == 1 && "expected single target op in payload");
  Operation *target = targetOps.front();
  auto config = iree_compiler::getLoweringConfig(target);
  if (!config) {
    transformResults.set(getResultConfigPart().cast<OpResult>(), {});
    return emitSilenceableFailure(target) << " has no IREE config";
  }

  // TODO: op verifier etc.
  if (getAttrName() != "tile_sizes")
    return emitDefiniteFailure("unsupported attr");

  if (!getLevel()) {
    transformResults.set(getResultConfigPart().cast<OpResult>(), {});
    return emitSilenceableFailure(target) << " level is required for tiling";
  }
  auto vals = config.getTileSizeVals(*getLevel());
  if (vals.empty()) {
    transformResults.set(getResultConfigPart().cast<OpResult>(), {});
    return emitSilenceableFailure(target) << " no tiling at level";
  }
  SmallVector<Value> values;
  SmallVector<Operation *> results;
  OpBuilder b(target);
  for (int64_t ts : vals) {
    results.push_back(b.create<arith::ConstantIndexOp>(target->getLoc(), ts));
    values.push_back(results.back()->getResult(0));
  }
  b.create<LinalgExt::DoNotDCEOperandsOp>(target->getLoc(), values);

  transformResults.set(getResultConfigPart().cast<OpResult>(), results);
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// EraseHALDescriptorTypeFromMemRef
//===---------------------------------------------------------------------===//

void transform_dialect::IREEEraseHALDescriptorTypeFromMemRefOp::build(
    OpBuilder &builder, OperationState &result, Value target) {
  result.addOperands(target);
  MLIRContext *ctx = builder.getContext();
  result.addTypes(pdl::OperationType::get(ctx));
}

DiagnosedSilenceableFailure
transform_dialect::IREEEraseHALDescriptorTypeFromMemRefOp::apply(
    transform::TransformResults &transformResults,
    transform::TransformState &state) {
  ArrayRef<Operation *> targetOps = state.getPayloadOps(getTarget());
  if (targetOps.size() != 1 || !isa<func::FuncOp>(targetOps.front())) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "expects a func::FuncOp as the target op");
  }
  auto funcOp = cast<func::FuncOp>(targetOps.front());

  if (failed(eraseHALDescriptorTypeFromMemRef(funcOp))) {
    return mlir::emitDefiniteFailure(
        state.getTopLevel(),
        "failed to erase #hal.descriptor_type as MemRef memory space");
  }

  transformResults.set(getOperation()->getOpResult(0), targetOps.front());
  return DiagnosedSilenceableFailure::success();
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
static void eraseDeadAllocAndStores(Operation *parentOp) {
  std::vector<Operation *> opToErase;
  parentOp->walk([&](memref::AllocOp op) {
    if (allUsesAreStores(op, opToErase)) {
      opToErase.push_back(op.getOperation());
    }
  });
  for (Operation *op : opToErase) {
    op->erase();
  }
}

DiagnosedSilenceableFailure
transform_dialect::ApplyBufferOptimizationsOp::applyToOne(
    Operation *target, transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  // Apply store to load forwarding and dead store elimination.
  vector::transferOpflowOpt(target);
  eraseDeadAllocAndStores(target);

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::ApplyBufferOptimizationsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::producesHandle(getResult(), effects);
  transform::modifiesPayload(effects);
}

void transform_dialect::ApplyBufferOptimizationsOp::build(
    OpBuilder &builder, OperationState &result, Value target) {
  result.addOperands(target);
  result.addTypes({pdl::OperationType::get(target.getContext())});
}
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensionsOps.cpp.inc"
