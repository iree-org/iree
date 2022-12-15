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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

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
  linalg::populateFoldUnitExtentDimsViaReshapesPatterns(patterns);
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
  linalg::populateFoldUnitExtentDimsViaReshapesPatterns(patterns);
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
  BlockAndValueMapping bvm;
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
  dispatchIndexOpFoldResults(mixedTileSizes, dynamicTileSizes, staticTileSizes,
                             ShapedType::kDynamic);
  // Call the default builder which sets up the proper operands segment sizes
  // attributes for multiple variadic operands. In the absence of this, horrible
  // bugs ensue.
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
                             staticNumThreads, ShapedType::kDynamic);
  // Call the default builder which sets up the proper operands segment sizes
  // attributes for multiple variadic operands. In the absence of this, horrible
  // bugs ensue.
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

static OneShotBufferizationOptions getBufferizationOptions() {
  OneShotBufferizationOptions options;
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

  //   1. Eliminate tensor.empty, without the pass baggage.
  WalkResult res = state.getTopLevel()->walk([&](ModuleOp moduleOp) {
    if (failed(eliminateEmptyTensors(moduleOp.getOperation(),
                                     getBufferizationOptions())))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (res.wasInterrupted())
    return DiagnosedSilenceableFailure::definiteFailure();

  //   2. Rewrite tensor.empty to tensor.alloc, without the pass baggage.
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

  //   3. Run one-shot-bufferize, without the pass baggage.
  OneShotBufferizationOptions options = getBufferizationOptions();
  options.allocationFn = allocationFn;
  options.deallocationFn = deallocationFn;
  options.memCpyFn = memCpyFn;
  options.testAnalysisOnly = getTestAnalysisOnly();
  options.printConflicts = getPrintConflicts();
  res = state.getTopLevel()->walk([&](ModuleOp moduleOp) {
    if (failed(runIREEOneShotBufferize(moduleOp, options)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (res.wasInterrupted())
    return DiagnosedSilenceableFailure::definiteFailure();

  //   4. Post-bufferization passes are fine.
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

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensionsOps.cpp.inc"
