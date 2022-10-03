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
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Pass/PassManager.h"

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
  registry.addExtensions<transform_dialect::CommonExtensions>();
}

//===---------------------------------------------------------------------===//
// ApplyPatternsOp
//===---------------------------------------------------------------------===//
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

DiagnosedSilenceableFailure transform_dialect::ApplyPatternsOp::applyToOne(
    Operation *target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    target->emitOpError(
        "applies only to isolated-from-above targets because it needs to apply "
        "patterns greedily");
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  }
  MLIRContext *ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  if (getCanonicalization()) addAllRegisteredCanonicalizationPatterns(patterns);
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
  if (failed(result) || failed(listenerResult))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  results.assign({target});
  return DiagnosedSilenceableFailure(success());
}

//===---------------------------------------------------------------------===//
// IREEBufferizeOp
//===---------------------------------------------------------------------===//

// TODO: Maybe we need both a transform.iree.cpu.bufferize and a
// transform.iree.gpu.bufferize rather than a single common bufferize op?
//
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

void transform_dialect::IREEBufferizeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getTarget(), effects);
  transform::producesHandle(getResult(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure transform_dialect::IREEBufferizeOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  ArrayRef<Operation *> payload = state.getPayloadOps(getTarget());
  if (payload.size() != 1 ||
      !isa<ModuleOp, HAL::ExecutableOp, HAL::ExecutableVariantOp>(
          payload.front())) {
    state.getTopLevel()->emitOpError(
        "requires exactly a single HAL::ExecutableOp or "
        "HAL::ExecutableVariantOp target op.");
    return DiagnosedSilenceableFailure(failure());
  }
  PassManager pm(getContext());
  // Bufferize the dispatch.
  using mlir::bufferization::BufferizationOptions;
  BufferizationOptions::AllocationFn allocationFn =
      cpuComprehensiveBufferizeAllocationFn;
  BufferizationOptions::DeallocationFn deallocationFn =
      cpuComprehensiveBufferizeDeallocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = cpuComprehensiveBufferizeCopyFn;
  if (getTargetGpu()) {
    allocationFn = gpuComprehensiveBufferizeAllocationFn;
    deallocationFn = gpuComprehensiveBufferizeDeallocationFn;
    memcpyFn = gpuComprehensiveBufferizeCopyFn;
  }
  mlir::iree_compiler::addIREEComprehensiveBufferizePasses(
      pm, allocationFn, deallocationFn, memcpyFn);
  WalkResult res = state.getTopLevel()->walk([&](ModuleOp moduleOp) {
    if (failed(pm.run(moduleOp))) {
      getOperation()->emitError()
          << "failed to bufferize ModuleOp:\n"
          << *(moduleOp.getOperation()) << "\nunder top-level:\n"
          << *state.getTopLevel();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  results.set(getOperation()->getOpResult(0), payload.front());
  return DiagnosedSilenceableFailure(failure(res.wasInterrupted()));
}
/// Populate the workgroup_count region of `dispatchOp`.
/// For now, this only supports constant index ops and empty workload operands.
/// Assumes the HAL::ExecutableExportOp is built with an empty region.
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
    state.getTopLevel()->emitOpError(
        "requires HAL::ExecutableOp or HAL::ExecutableVariantOp toplevel "
        "to attach the workgroup size information to a nested "
        "ExecutableExportOp");
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  }

  IREE::HAL::ExecutableExportOp exportOp;
  state.getTopLevel()->walk([&](IREE::HAL::ExecutableExportOp op) {
    if (op.getSymName() == target.getName()) exportOp = op;
  });
  if (!exportOp) {
    state.getTopLevel()->emitOpError("no IREE::HAL::ExecutableExportOp found");
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
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
    state.getTopLevel()->emitOpError(
        "could not find a unique topLevel scf.foreach_thread");
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  }

  SimplePatternRewriter rewriter(topLevelForeachThreadOp);
  if (failed(rewriteForeachThreadToWorkgroup(topLevelForeachThreadOp, exportOp,
                                             rewriter)))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));

  results.assign({target});

  return DiagnosedSilenceableFailure(success());
}

void transform_dialect::ForeachThreadToWorkgroupOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getTarget(), effects);
  transform::producesHandle(getTransformed(), effects);
}

//===---------------------------------------------------------------------===//
// TileToForeachThreadAndWorkgroupCountRegion
//===---------------------------------------------------------------------===//

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
    TileToForeachThreadAndWorkgroupCountRegion::getMixedNumThreads() {
  return getMixedSizes(getStaticNumThreads(), getNumThreads());
}

SmallVector<OpFoldResult> transform_dialect::
    TileToForeachThreadAndWorkgroupCountRegion::getMixedTileSizes() {
  return getMixedSizes(getStaticTileSizes(), getTileSizes());
}

LogicalResult
transform_dialect::TileToForeachThreadAndWorkgroupCountRegion::verify() {
  if (getMixedNumThreads().empty() == getMixedTileSizes().empty())
    return emitOpError("either num_threads or tile_sizes must be specified");
  return success();
}

void transform_dialect::TileToForeachThreadAndWorkgroupCountRegion::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getTarget(), effects);
  transform::onlyReadsHandle(getTileSizes(), effects);
  transform::onlyReadsHandle(getNumThreads(), effects);
  transform::producesHandle(getResults(), effects);
}

DiagnosedSilenceableFailure
transform_dialect::TileToForeachThreadAndWorkgroupCountRegion::apply(
    transform::TransformResults &transformResults,
    transform::TransformState &state) {
  ArrayRef<Operation *> targetOps = state.getPayloadOps(getTarget());
  assert(targetOps.size() == 1 && "expected single target op in payload");
  auto funcOp = targetOps.front()->getParentOfType<func::FuncOp>();
  FailureOr<IREE::HAL::ExecutableExportOp> exportOp = getEntryPoint(funcOp);
  if (failed(exportOp)) {
    state.getTopLevel()->emitOpError("couldn't find export op for func");
    return DiagnosedSilenceableFailure(reportUnknownTransformError(funcOp));
  }

  SmallVector<OpFoldResult> mixedTileSizes = getMixedTileSizes();
  if (mixedTileSizes.empty()) {
    exportOp.value()->emitOpError("require tile sizes to be specified");
    return DiagnosedSilenceableFailure(
        reportUnknownTransformError(exportOp.value()));
  }

  /// Lower the workgroup count region in keeping with the way dispatch
  /// regions are created by default in IREEs compilation flow.
  IRRewriter rewriter(getContext());
  if (failed(lowerWorkgroupCountComputingRegion(rewriter, exportOp.value(),
                                                mixedTileSizes))) {
    exportOp.value()->emitOpError("failed to lower workgroup count region");
    return DiagnosedSilenceableFailure(
        reportUnknownTransformError(exportOp.value()));
  }

  ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());

  // Result payload ops.
  SmallVector<Operation *> tileOps;
  SmallVector<Operation *> tiledOps;

  DiagnosedSilenceableFailure diag = transform::tileToForeachThreadOpImpl(
      rewriter, state, cast<transform::TransformOpInterface>(getOperation()),
      targets, getMixedNumThreads(), getMixedTileSizes(), getThreadDimMapping(),
      tileOps, tiledOps);

  if (!diag.succeeded()) return diag;

  transformResults.set(getForeachThreadOp().cast<OpResult>(), tileOps);
  transformResults.set(getTiledOp().cast<OpResult>(), tiledOps);

  return DiagnosedSilenceableFailure(success());
}

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensionsOps.cpp.inc"
