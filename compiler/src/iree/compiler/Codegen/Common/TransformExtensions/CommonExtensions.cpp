// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CommonExtensions.h"

#include <iree/compiler/Dialect/HAL/IR/HALOps.h>

#include "iree-dialects/Dialect/LinalgTransform/SimplePatternRewriter.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/ListenerGreedyPatternRewriteDriver.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Interfaces/BufferizationInterfaces.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/AllocTensorElimination.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

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

//===---------------------------------------------------------------------===//
// ApplyPatternsOp
//===---------------------------------------------------------------------===//
void transform_dialect::ApplyPatternsOp::build(OpBuilder &builder,
                                               OperationState &result,
                                               Value target,
                                               bool rankReducing) {
  MLIRContext *ctx = builder.getContext();
  result.addOperands(target);
  if (rankReducing) {
    result.addAttribute(ApplyPatternsOp::getRankReducingAttrName(result.name),
                        builder.getUnitAttr());
  }
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

struct PromoteCaptureToSharedOut
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSliceOp,
                                PatternRewriter &rewriter) const override {
    scf::ForeachThreadOp foreachThreadOp =
        extractSliceOp->getParentOfType<scf::ForeachThreadOp>();

    while (foreachThreadOp) {
      // Check if the extract_slice source is a shared output.
      auto outputIt =
          llvm::find(foreachThreadOp.getOutputs(), extractSliceOp.getSource());
      if (outputIt == foreachThreadOp.getOutputs().end()) {
        foreachThreadOp =
            foreachThreadOp->getParentOfType<scf::ForeachThreadOp>();
        continue;
      }

      // Get the corresponding bbArg of the loop body.
      BlockArgument bbArg =
          foreachThreadOp.getOutputBlockArguments()[std::distance(
              foreachThreadOp.getOutputs().begin(), outputIt)];

      // Check if the extract_slice has a matching parallel_insert_slice (i.e.,
      // same source/target, offsets, sizes and strides).
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
        foreachThreadOp =
            foreachThreadOp->getParentOfType<scf::ForeachThreadOp>();
        continue;
      }

      // Promote extract_slice source to bbArg.
      rewriter.updateRootInPlace(extractSliceOp, [&]() {
        extractSliceOp.getSourceMutable().assign(bbArg);
      });

      return success();
    }

    return failure();
  }
};
}  // namespace

static void addForeachThreadCapturePromotionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<PromoteCaptureToSharedOut>(patterns.getContext());
}

static void addRankReducingPatterns(RewritePatternSet &patterns) {
  populateReshapeToInterfaceTensorPatterns(patterns);
  vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
  linalg::populateFoldUnitExtentDimsPatterns(patterns);
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

static void addConverToDPSPatterns(RewritePatternSet &patterns) {
  populateReshapeToInterfaceTensorPatterns(patterns);
  vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
  linalg::populateFoldUnitExtentDimsPatterns(patterns);
}

DiagnosedSilenceableFailure transform_dialect::ApplyPatternsOp::applyToOne(
    Operation *target, SmallVectorImpl<Operation *> &results,
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
  if (getPromoteForeachThreadCaptureToShared())
    addForeachThreadCapturePromotionPatterns(patterns);
  if (getRankReducing()) addRankReducingPatterns(patterns);
  if (getSimplifyMemrefMetadata())
    memref::populateSimplifyExtractStridedMetadataOpPatterns(patterns);
  if (getSwappingPatterns())
    addSwappingPatterns(patterns, getSwapPaddingElideConditional());
  if (getAdditionalIreePatterns()) addAdditionalIreePatterns(patterns);

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

  results.assign({target});
  return DiagnosedSilenceableFailure(success());
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

/// Apply the permutation `perm` to `vals.
/// Return failure if perm is not a permutation.
// TODO: upstream as extraClassDeclaration once stabilized.
template <typename T>
static FailureOr<SmallVector<T>> permute(const SmallVector<T> &vals,
                                         ArrayRef<int64_t> perm) {
  if (vals.size() != perm.size()) return failure();
  SmallVector<T> result(vals.size());
  SmallVector<bool> seen(vals.size());
  for (const auto &it : llvm::zip(perm, vals)) {
    // Already seen, invalid thread_dim_mapping.
    if (seen[std::get<0>(it)]) return failure();
    result[std::get<0>(it)] = std::get<1>(it);
    seen[std::get<0>(it)] = true;
  }
  // Some not seen, invalid thread_dim_mapping.
  if (!llvm::all_of(seen, [](bool b) { return b; })) return failure();
  return result;
}

/// Helper to get apply the `thread_dim_mapping` permutation of a
/// `foreachThreadOp` to `values`.
// TODO: upstream as extraClassDeclaration once stabilized.
template <typename T>
static FailureOr<SmallVector<T>> getPermuted(
    scf::ForeachThreadOp foreachThreadOp, const SmallVector<T> &values) {
  // Apply mapping permutation if specified.
  auto mapping = foreachThreadOp.getThreadDimMapping();
  if (mapping && !mapping.empty()) {
    auto maybePermuted = permute(values, extractFromI64ArrayAttr(mapping));
    if (failed(maybePermuted))
      return foreachThreadOp->emitError("invalid permutation");
    return *maybePermuted;
  }
  return values;
}

/// Helper to get the `num_threads` of a `foreachThreadOp` after applying the
/// `thread_dim_mapping` permutation.
// TODO: upstream as extraClassDeclaration once stabilized.
static FailureOr<SmallVector<OpFoldResult>> getNumThreads(
    OpBuilder &b, scf::ForeachThreadOp foreachThreadOp) {
  SmallVector<OpFoldResult> threadCount = foreachThreadOp.getNumThreads();
  threadCount.resize(3, b.getIndexAttr(1));
  return getPermuted(foreachThreadOp, threadCount);
}

/// Helper to get the thread indices of a `foreachThreadOp` after applying the
/// `thread_dim_mapping` permutation.
// TODO: upstream as extraClassDeclaration once stabilized.
static FailureOr<SmallVector<Value>> getThreadIndices(
    OpBuilder &b, scf::ForeachThreadOp foreachThreadOp) {
  SmallVector<Value> threadCount = foreachThreadOp.getThreadIndices();
  threadCount.resize(3, Value());
  return getPermuted(foreachThreadOp, threadCount);
}

//===---------------------------------------------------------------------===//
// Patterns for ForeachThreadToWorkgroup rewrite.
//===---------------------------------------------------------------------===//

LogicalResult rewriteForeachThreadToWorkgroup(
    scf::ForeachThreadOp foreachThreadOp,
    IREE::HAL::ExecutableExportOp exportOp, PatternRewriter &rewriter) {
  if (foreachThreadOp.getNumResults() > 0)
    return foreachThreadOp->emitError(
        "only bufferized scf.foreach_thread lowers to workgroup");
  if (foreachThreadOp.getNumThreads().size() > 3)
    return foreachThreadOp->emitError(
        "scf.foreach_thread with rank > 3 does not lower to workgroup");

  // Step 0. Outline the compute workload region and set up the workload
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
    auto maybeWorkgroupCounts = getNumThreads(rewriter, foreachThreadOp);
    if (failed(maybeWorkgroupCounts) ||
        llvm::any_of(*maybeWorkgroupCounts, [](OpFoldResult ofr) {
          return !getConstantIntValue(ofr).has_value();
        }))
      return foreachThreadOp->emitError(
          "unsupported dynamic workgroup_count atm --- need to slice out "
          "workgroup_count computation into ExecutableExport::workgroup_count. "
          "This region may require arbitrary computations and cannot magically "
          "match what the `stream.cmd.dispatch` has already imposed on us at a "
          "distance. For now we must specify the number of values properly "
          "when applying the topLevel tile_to_foreach_thread_op");
    SmallVector<int64_t> workgroupCounts;
    for (OpFoldResult ofr : *maybeWorkgroupCounts)
      workgroupCounts.push_back(getConstantIntValue(ofr).value());
    if (failed(populateWorkgroupCountComputingRegion(rewriter, foreachThreadOp,
                                                     exportOp))) {
      return foreachThreadOp->emitOpError(
                 "failed to populate workload region for dispatchOp: ")
             << exportOp;
    }
  }

  // Step 1. Create the workgroup id and count ops.
  Location loc = foreachThreadOp.getLoc();
  BlockAndValueMapping bvm;
  SmallVector<Value, 8> workgroupIdOps, workgroupCountOps;
  for (int64_t rank : llvm::seq<int64_t>(0, 3)) {
    workgroupIdOps.push_back(
        rewriter.create<HAL::InterfaceWorkgroupIDOp>(loc, rank));
    workgroupCountOps.push_back(
        rewriter.create<HAL::InterfaceWorkgroupCountOp>(loc, rank));
  }
  bvm.map(foreachThreadOp.getThreadIndices(), workgroupIdOps);
  bvm.map(foreachThreadOp.getNumThreads(), workgroupCountOps);

  // Step 2. Predicate omitted given unique topLevel scf::ForeachThreadOp.

  // Step 3. Move the body of foreachThreadOp.
  // Erase the terminator first, it will not be used since we are on buffers.
  rewriter.eraseOp(foreachThreadOp.getTerminator());
  Block *targetBlock;
  Block::iterator insertionPoint;
  targetBlock = foreachThreadOp->getBlock();
  insertionPoint = Block::iterator(foreachThreadOp);
  Block &sourceBlock = foreachThreadOp.getRegion().front();
  targetBlock->getOperations().splice(insertionPoint,
                                      sourceBlock.getOperations());

  // Step 4. RAUW thread indices to thread ops.
  SmallVector<Value> threadIndices =
      *getThreadIndices(rewriter, foreachThreadOp);
  assert(workgroupIdOps.size() == 3 && "3 workgroup id ops are required");
  assert(threadIndices.size() == 3 && "3 thread id dimensions are required");
  for (auto it : llvm::zip(threadIndices, workgroupIdOps)) {
    Value val = std::get<0>(it);
    if (!val) continue;
    for (Operation *user : llvm::make_early_inc_range(val.getUsers())) {
      rewriter.updateRootInPlace(
          user, [&]() { user->replaceUsesOfWith(val, std::get<1>(it)); });
    }
  }

  // Step 5. Barriers omitted given unique topLevel scf::ForeachThreadOp.

  // Step 6. Erase old op.
  rewriter.eraseOp(foreachThreadOp);

  return success();
}

//===---------------------------------------------------------------------===//
// IREE-specific transformations defined outside of iree_linalg_transform.
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::ForeachThreadToWorkgroupOp::applyToOne(
    func::FuncOp target, SmallVectorImpl<Operation *> &results,
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

  results.assign({target});
  return DiagnosedSilenceableFailure(success());
}

//===---------------------------------------------------------------------===//
// TileToForeachThreadAndWorkgroupCountRegionOp
//===---------------------------------------------------------------------===//

void transform_dialect::TileToForeachThreadAndWorkgroupCountRegionOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    ArrayRef<int64_t> staticTileSizes, transform::TileSizesSpec,
    ArrayRef<int64_t> threadDimMapping) {
  return build(builder, result, target,
               getAsOpFoldResult(builder.getI64ArrayAttr(staticTileSizes)),
               transform::TileSizesSpec(), threadDimMapping);
}

void transform_dialect::TileToForeachThreadAndWorkgroupCountRegionOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    ArrayRef<OpFoldResult> mixedTileSizes, transform::TileSizesSpec,
    ArrayRef<int64_t> threadDimMapping) {
  assert(result.name.isRegistered() && "not registered!!");
  SmallVector<int64_t> staticTileSizes;
  SmallVector<Value> dynamicTileSizes;
  dispatchIndexOpFoldResults(mixedTileSizes, dynamicTileSizes, staticTileSizes,
                             ShapedType::kDynamicSize);
  // Call the default builder which sets up the proper operands segment sizes
  // attributes for multiple variadic operands. In the absence of this, horrible
  // bugs ensue.
  MLIRContext *ctx = builder.getContext();
  auto operationType = pdl::OperationType::get(ctx);
  auto staticTileSizesAttr = builder.getI64ArrayAttr(staticTileSizes);
  ArrayAttr threadDimMappingAttr;
  if (!threadDimMapping.empty())
    threadDimMappingAttr = builder.getI64ArrayAttr(threadDimMapping);
  build(builder, result, TypeRange{operationType, operationType}, target,
        /*numThreads=*/ValueRange{}, dynamicTileSizes,
        /*staticNumThreads=*/ArrayAttr(), staticTileSizesAttr,
        threadDimMappingAttr);
}

void transform_dialect::TileToForeachThreadAndWorkgroupCountRegionOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    ArrayRef<int64_t> staticNumThreads, transform::NumThreadsSpec,
    ArrayRef<int64_t> threadDimMapping) {
  return build(builder, result, target,
               getAsOpFoldResult(builder.getI64ArrayAttr(staticNumThreads)),
               transform::NumThreadsSpec(), threadDimMapping);
}

void transform_dialect::TileToForeachThreadAndWorkgroupCountRegionOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    ArrayRef<OpFoldResult> mixedNumThreads, transform::NumThreadsSpec,
    ArrayRef<int64_t> threadDimMapping) {
  assert(result.name.isRegistered() && "not registered!!");
  SmallVector<int64_t> staticNumThreads;
  SmallVector<Value> dynamicNumThreads;
  dispatchIndexOpFoldResults(mixedNumThreads, dynamicNumThreads,
                             staticNumThreads, ShapedType::kDynamicSize);
  // Call the default builder which sets up the proper operands segment sizes
  // attributes for multiple variadic operands. In the absence of this, horrible
  // bugs ensue.
  MLIRContext *ctx = builder.getContext();
  auto operationType = pdl::OperationType::get(ctx);
  auto staticNumThreadsAttr = builder.getI64ArrayAttr(staticNumThreads);
  ArrayAttr threadDimMappingAttr;
  if (!threadDimMapping.empty())
    threadDimMappingAttr = builder.getI64ArrayAttr(threadDimMapping);
  build(builder, result, TypeRange{operationType, operationType}, target,
        dynamicNumThreads, /*tileSizes=*/ValueRange{}, staticNumThreadsAttr,
        /*staticTileSizes=*/ArrayAttr(), threadDimMappingAttr);
}

/// Lower the ops within the workgroup count region of `exportOp` that
/// represents the workgroup count calculation, to the actual
/// computation that returns the number of workgroups. For now
/// this lowers the `flow.dispatch.workgroup_count_from_dag_root` op
/// to `ceilDiv(workload, tileSizes)`.
static LogicalResult lowerWorkgroupCountComputingRegion(
    RewriterBase &rewriter, HAL::ExecutableExportOp exportOp,
    ArrayRef<OpFoldResult> tileSizes) {
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

  if (tileSizes.size() > workload.size()) {
    return rewriter.notifyMatchFailure(
        exportOp,
        "number of tile sizes overflow the dimension from the workload");
  }

  SmallVector<OpFoldResult> workgroupCount;
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(workgroupCountOp);
  Location loc = workgroupCountOp.getLoc();
  for (auto tileSize : llvm::enumerate(tileSizes)) {
    if (isConstantIntValue(tileSize.value(), 0)) {
      workgroupCount.push_back(workload[tileSize.index()]);
      continue;
    }
    AffineExpr s0, s1;
    bindSymbols(rewriter.getContext(), s0, s1);
    auto m = AffineMap::get(0, 2, s0.ceilDiv(s1));
    OpFoldResult count = makeComposedFoldedAffineApply(
        rewriter, loc, m,
        ArrayRef<OpFoldResult>{workload[tileSize.index()], tileSize.value()});
    workgroupCount.push_back(count);
  }
  workgroupCount = llvm::to_vector(llvm::reverse(workgroupCount));
  workgroupCount.resize(3, rewriter.getIndexAttr(1));
  rewriter.replaceOp(workgroupCountOp, getValueOrCreateConstantIndexOp(
                                           rewriter, loc, workgroupCount));
  return success();
}

SmallVector<OpFoldResult> transform_dialect::
    TileToForeachThreadAndWorkgroupCountRegionOp::getMixedNumThreads() {
  return getMixedSizes(getStaticNumThreads(), getNumThreads());
}

SmallVector<OpFoldResult> transform_dialect::
    TileToForeachThreadAndWorkgroupCountRegionOp::getMixedTileSizes() {
  return getMixedSizes(getStaticTileSizes(), getTileSizes());
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

  SmallVector<OpFoldResult> mixedTileSizes = getMixedTileSizes();
  if (mixedTileSizes.empty()) {
    return mlir::emitDefiniteFailure(exportOp.value(),
                                     "require tile sizes to be specified");
  }

  /// Lower the workgroup count region in keeping with the way dispatch
  /// regions are created by default in IREEs compilation flow.
  IRRewriter rewriter(getContext());
  if (failed(lowerWorkgroupCountComputingRegion(rewriter, exportOp.value(),
                                                mixedTileSizes))) {
    return mlir::emitDefiniteFailure(exportOp.value(),
                                     "failed to lower workgroup count region");
  }

  ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());

  // Result payload ops.
  SmallVector<Operation *> tileOps;
  SmallVector<Operation *> tiledOps;

  DiagnosedSilenceableFailure diag = transform::tileToForeachThreadOpImpl(
      rewriter, state, cast<transform::TransformOpInterface>(getOperation()),
      targets, getMixedNumThreads(), getMixedTileSizes(), getThreadDimMapping(),
      tileOps, tiledOps);

  if (!diag.succeeded()) {
    transformResults.set(getForeachThreadOp().cast<OpResult>(),
                         SmallVector<mlir::Operation *>{});
    transformResults.set(getTiledOp().cast<OpResult>(),
                         SmallVector<mlir::Operation *>{});
    return diag;
  }

  transformResults.set(getForeachThreadOp().cast<OpResult>(), tileOps);
  transformResults.set(getTiledOp().cast<OpResult>(), tiledOps);
  return DiagnosedSilenceableFailure(success());
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
                                               Value target, bool targetGpu) {
  result.addOperands(target);
  if (targetGpu) {
    result.addAttribute(IREEBufferizeOp::getTargetGpuAttrName(result.name),
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
  // TODO: use gpu::GPUDialect::getWorkgroupAddressSpace() but this requires
  // moving out of CommonExtensions.
  MemRefType allocType = MemRefType::get(memRefType.getShape(),
                                         memRefType.getElementType(), {}, 3);
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
  // as an OpDSL named op. However, IREE-specific patterns to cleanup spurious
  // post-bufferization copies do not trigger properly.
  // So we keep using `createLinalgCopyOp` which builds a GenericOp.
  // builder.create<linalg::CopyOp>(loc, from, to);
  mlir::iree_compiler::createLinalgCopyOp(builder, loc, from, to);
  return success();
}

/// Temporarily copied from IREEComprehensiveBufferizePass.cpp to avoid buying
/// into nested pass pipeline mess.
static LogicalResult emptyTensorElimination(
    Operation *op, OneShotBufferizationOptions options) {
  // Analyze IR.
  options.testAnalysisOnly = false;
  options.printConflicts = false;
  OneShotAnalysisState state(op, options);
  if (failed(analyzeOp(op, state))) return failure();

  // Rewrite init_tensors that are anchored on specific ops.
  IRRewriter rewriter(op->getContext());
  if (failed(bufferization::insertSliceAnchoredAllocTensorEliminationStep(
          rewriter, op, state)))
    return failure();
  if (failed(
          storeTensorOpAnchoredInitTensorEliminationStep(rewriter, op, state)))
    return failure();

  return success();
}

/// Temporarily copied from IREEComprehensiveBufferizePass.cpp to avoid buying
/// into nested pass pipeline mess.
// The following is copied from bufferization::runOneShotBufferize with
// modifications.
static LogicalResult runIREEOneShotBufferize(
    Operation *op, const OneShotBufferizationOptions &options) {
  OneShotAnalysisState state(op, options);
  if (failed(analyzeOp(op, state))) return failure();
  if (options.testAnalysisOnly) return success();
  return bufferization::runOneShotBufferize(op, options);
}

/// Temporarily copied from IREEComprehensiveBufferizePass.cpp to avoid buying
/// into nested pass pipeline mess.
static LogicalResult runIREEBufferizeOnModule(
    ModuleOp moduleOp, BufferizationOptions::AllocationFn allocationFn,
    BufferizationOptions::DeallocationFn deallocationFn,
    BufferizationOptions::MemCpyFn memCpyFn) {
  OneShotBufferizationOptions options;
  options.allocationFn = allocationFn;
  options.deallocationFn = deallocationFn;
  options.memCpyFn = memCpyFn;
  // options.testAnalysisOnly = testAnalysisOnly;
  // options.printConflicts = printConflicts;

  // bufferization.to_memref is used to bufferize constants in IREE. IREE has
  // it's own logic to handle constants. We'd like to leave the arith.constant
  // as is and insert bufferization.to_memref to convert the tensor to memref.
  options.opFilter.denyOperation<arith::ConstantOp>();
  options.opFilter.denyOperation<bufferization::ToMemrefOp>();

  // This type converter converts tensor types to memref types when no exact
  // memref type can be inferred from the context.
  options.unknownTypeConverterFn = [](Value value, unsigned memorySpace,
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

  if (failed(emptyTensorElimination(moduleOp.getOperation(), options)))
    return failure();

  return runIREEOneShotBufferize(moduleOp, options);
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

  //   1. Rewrite tensor.empty to tensor.alloc, without the pass baggage.
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

  //   2. Run one-shot-bufferize, without the pass baggage.
  WalkResult res = state.getTopLevel()->walk([&](ModuleOp moduleOp) {
    if (failed(runIREEBufferizeOnModule(moduleOp, allocationFn, deallocationFn,
                                        memCpyFn)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (res.wasInterrupted())
    return DiagnosedSilenceableFailure::definiteFailure();

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

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensionsOps.cpp.inc"
