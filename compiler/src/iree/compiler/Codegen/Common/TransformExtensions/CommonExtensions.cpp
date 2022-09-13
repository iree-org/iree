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
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
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

static void addRankReducingPatterns(RewritePatternSet &patterns) {
  populateReshapeToInterfaceTensorPatterns(patterns);
  vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
  linalg::populateFoldUnitExtentDimsPatterns(patterns);
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

/// Lower the ops within the workgroup count region of `exportOp` that
/// represents the workgroup count calculation, to the actual
/// computation that returns the number of workgroups. For now
/// this lowers the `flow.dispatch.workgroup_count_from_dag_root` op
/// to `ceilDiv(workload, tileSizes)`.
static LogicalResult lowerWorkgroupCountComputingRegion(
    PatternRewriter &rewriter, scf::ForeachThreadOp foreachThreadOp,
    HAL::ExecutableExportOp exportOp, ArrayRef<OpFoldResult> tileSizes) {
  Region &r = exportOp.getWorkgroupCount();
  if (r.hasOneBlock()) {
    return rewriter.notifyMatchFailure(exportOp,
                                       "expected export op to have a workgroup "
                                       "count region with a single block");
  }
  auto workgroupCountOps =
      r.front().getOps<IREE::Flow::DispatchWorkgroupCountFromDagRootOp>();
  if (llvm::hasSingleElement(workgroupCountOps)) {
    return rewriter.notifyMatchFailure(
        exportOp,
        "expected region to have a single "
        "flow.dispatch.workgroup_count_from_dag_root op");
  }
  auto workgroupCountOp = *workgroupCountOps.begin();
  auto workload = workgroupCountOp.getOperands();

  if (tileSizes.size() > workload.size()) {
    return rewriter.notifyMatchFailure(
        exportOp, "tile sizes more than the workload captured");
  }

  SmallVector<Value> workgroupCount;
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
    Value tileSizeVal =
        getValueOrCreateConstantIndexOp(rewriter, loc, tileSize.value());
    Value count = rewriter.create<AffineApplyOp>(
        loc, m, ValueRange{workload[tileSize.index()], tileSizeVal});
    workgroupCount.push_back(count);
  }
  rewriter.replaceOp(workgroupCountOp, workgroupCount);
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
    IREE::HAL::ExecutableExportOp exportOp, ArrayRef<OpFoldResult> tileSizes,
    PatternRewriter &rewriter) {
  if (foreachThreadOp.getNumResults() > 0)
    return foreachThreadOp->emitError(
        "only bufferized scf.foreach_thread lowers to workgroup");
  if (foreachThreadOp.getNumThreads().size() > 3)
    return foreachThreadOp->emitError(
        "scf.foreach_thread with rank > 3 does not lower to workgroup");

  // Step 0. Populate the workgroup count region with the actual computation
  // that returns the workgroup count.
  if (failed(lowerWorkgroupCountComputingRegion(rewriter, foreachThreadOp,
                                                exportOp, tileSizes)))
    return foreachThreadOp->emitOpError(
               "failed to populate workload region for dispatchOp: ")
           << exportOp;

  // Step 1. Create the workgroup id and count ops.
  Location loc = foreachThreadOp.getLoc();
  BlockAndValueMapping bvm;
  SmallVector<Value, 8> workgroupIdOps, workgroupCountOps;
  for (int64_t rank :
       llvm::seq<int64_t>(0, foreachThreadOp.getThreadIndices().size())) {
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
  for (auto it : llvm::zip(threadIndices, workgroupIdOps)) {
    if (!std::get<0>(it)) continue;
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
  }

  // Step 5. Barriers omitted given unique topLevel scf::ForeachThreadOp.

  // Step 6. Erase old op.
  rewriter.eraseOp(foreachThreadOp);

  return success();
}

//===---------------------------------------------------------------------===//
// IREE-specific transformations defined outside of iree_linalg_transform.
//===---------------------------------------------------------------------===//

SmallVector<OpFoldResult>
transform_dialect::ForeachThreadToWorkgroupOp::getMixedTileSizes() {
  return getMixedSizes(getStaticTileSizes(), getTileSizes());
}

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
  if (!exportOp.getWorkgroupCount().empty())
    return emitDefaultSilenceableFailure(target)
           << "export op must have an empty workgroup count region that "
              "the transform fills --- the transform is not applied";

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
                                             getMixedTileSizes(), rewriter)))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));

  results.assign({target});

  return DiagnosedSilenceableFailure(success());
}

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensionsOps.cpp.inc"
