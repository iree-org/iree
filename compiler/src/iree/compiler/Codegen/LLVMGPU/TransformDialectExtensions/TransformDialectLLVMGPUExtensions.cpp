// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TransformDialectLLVMGPUExtensions.h"

#include "iree-dialects/Transforms/Functional.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Region.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE;

iree_compiler::IREE::transform_dialect::TransformDialectLLVMGPUExtensions::
    TransformDialectLLVMGPUExtensions() {
  registerTransformOps<
#define GET_OP_LIST
#include "iree/compiler/Codegen/LLVMGPU/TransformDialectExtensions/TransformDialectLLVMGPUExtensionsOps.cpp.inc"
      >();
}

void mlir::iree_compiler::registerTransformDialectLLVMGPUExtension(
    DialectRegistry &registry) {
  registry
      .addExtensions<transform_dialect::TransformDialectLLVMGPUExtensions>();
}

// TODO: Maybe we need both a transform.iree.cpu.bufferize and a
// transform.iree.gpu.bufferize rather than a single common bufferize op?

static FailureOr<SmallVector<int64_t>> joinStaticRanges(
    ArrayRef<scf::ForeachThreadOp> ops) {
  SmallVector<int64_t> joinedStaticRange = {1, 1, 1};
  for (auto op : ops) {
    for (auto en : llvm::enumerate(op.getNumThreads())) {
      auto maybeInt = getConstantIntValue(en.value());
      if (!maybeInt) {
        op->emitError(
            "scf.foreach_thread with non-constant workgroup size cannot be "
            "lowered into the translation_info attr");
        return failure();
      }
      joinedStaticRange[en.index()] =
          std::max(joinedStaticRange[en.index()], *maybeInt);
    }
  }
  return joinedStaticRange;
}

//===---------------------------------------------------------------------===//
// Patterns for ForeachThreadToGpu rewrite.
//===---------------------------------------------------------------------===//

struct ForeachThreadToGpuRewriter
    : public OpRewritePattern<scf::ForeachThreadOp> {
  ForeachThreadToGpuRewriter(ArrayRef<int64_t> globalWorkgroupSizes,
                             MLIRContext *context)
      : OpRewritePattern<scf::ForeachThreadOp>(context),
        globalWorkgroupSizes(globalWorkgroupSizes.begin(),
                             globalWorkgroupSizes.end()) {}

  FailureOr<SmallVector<Value>> returningMatchAndRewrite(
      scf::ForeachThreadOp foreachThreadOp, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(scf::ForeachThreadOp foreachThreadOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(foreachThreadOp, rewriter);
  }

  SmallVector<int64_t> globalWorkgroupSizes;
};

FailureOr<SmallVector<Value>>
ForeachThreadToGpuRewriter::returningMatchAndRewrite(
    scf::ForeachThreadOp foreachThreadOp, PatternRewriter &rewriter) const {
  if (foreachThreadOp.getNumResults() > 0)
    return foreachThreadOp->emitError(
        "only bufferized scf.foreach_thread lowers to gpu.thread");
  if (foreachThreadOp.getNumThreads().size() > 3)
    return foreachThreadOp->emitError(
        "scf.foreach_thread with rank > 3 does not lower to gpu.thread");

  auto maybeWorkgroupSizes = joinStaticRanges({foreachThreadOp});
  assert(succeeded(maybeWorkgroupSizes) && "unexpected dynamic workgroup size");

  auto workgroupSizes = *maybeWorkgroupSizes;

  // Step 1. Create the gpu.thread ops
  Location loc = foreachThreadOp.getLoc();
  IndexType indexType = rewriter.getIndexType();
  SmallVector<Value> threadCount = foreachThreadOp.getNumThreads();
  SmallVector<gpu::Dimension, 3> gpuDims{gpu::Dimension::x, gpu::Dimension::y,
                                         gpu::Dimension::z};
  SmallVector<Value, 3> threadOps;
  threadCount.resize(3, rewriter.create<arith::ConstantIndexOp>(loc, 1));
  for (int64_t idx : llvm::seq<int64_t>(0, threadCount.size())) {
    threadOps.push_back(
        rewriter.create<gpu::ThreadIdOp>(loc, indexType, gpuDims[idx]));
  }

  // Step 2. Maybe create conditionals to predicate the region.
  Value predicate;
  for (auto it : llvm::zip(threadOps, workgroupSizes, globalWorkgroupSizes)) {
    auto threadId = std::get<0>(it);
    auto workgroupSize = std::get<1>(it);
    auto globalWorkgroupSize = std::get<2>(it);
    assert(workgroupSize <= globalWorkgroupSize && "workgroup size overflow");
    if (workgroupSize == globalWorkgroupSize) continue;
    Value tmpPredicate = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, threadId,
        rewriter.create<arith::ConstantIndexOp>(loc, workgroupSize));
    predicate =
        predicate ? rewriter.create<arith::AndIOp>(loc, predicate, tmpPredicate)
                  : tmpPredicate;
  }

  // Step 3. Move the body of foreachThreadOp.
  // Erase the terminator first, it will not be used.
  rewriter.eraseOp(foreachThreadOp.getTerminator());
  Block *targetBlock;
  Block::iterator insertionPoint;
  if (predicate) {
    // Step 3.a. If predicated, move at the beginning.
    auto ifOp =
        rewriter.create<scf::IfOp>(loc, predicate, /*withElseRegion=*/false);
    targetBlock = ifOp.thenBlock();
    insertionPoint = ifOp.thenBlock()->begin();
  } else {
    // Step 3.a. Otherwise, move inline just before foreachThreadOp.
    targetBlock = foreachThreadOp->getBlock();
    insertionPoint = Block::iterator(foreachThreadOp);
  }
  Block &sourceBlock = foreachThreadOp.getRegion().front();
  targetBlock->getOperations().splice(insertionPoint,
                                      sourceBlock.getOperations());

  // Step 4. RAUW thread indices to thread ops.
  for (auto it : llvm::zip(foreachThreadOp.getThreadIndices(), threadOps))
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));

  // Step 5. syncthreads.
  rewriter.create<gpu::BarrierOp>(loc);

  // Step 6. Erase old op.
  rewriter.eraseOp(foreachThreadOp);

  return threadCount;
}

//===---------------------------------------------------------------------===//
// IREE-specific LLVMGPU transformations.
//===---------------------------------------------------------------------===//

// TODO: if the number of threads was wired like the workgroup_count, we could
// reuse most of the code and not require a static number of threads.
// TODO: synchronizations for imperfectly nested stuff.
DiagnosedSilenceableFailure
transform_dialect::ForeachThreadToGpuAndTranslationInfo::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  if (!isa<HAL::ExecutableVariantOp>(state.getTopLevel())) {
    emitOpError() << "applies only to HAL::ExecutableVariantOp targets";
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  DenseMap<func::FuncOp, SmallVector<scf::ForeachThreadOp>>
      foreachThreadsPerFunc;
  DenseMap<scf::ForeachThreadOp, func::FuncOp> foreachThreadToFunc;
  WalkResult walkResult = state.getTopLevel()->walk<WalkOrder::PostOrder>(
      [&](scf::ForeachThreadOp op) {
        if (op->getParentOfType<scf::ForeachThreadOp>())
          return WalkResult::interrupt();

        auto funcOp = op->getParentOfType<func::FuncOp>();
        foreachThreadToFunc.insert(std::make_pair(op, funcOp));
        auto it = foreachThreadsPerFunc.find(funcOp);
        if (it == foreachThreadsPerFunc.end())
          foreachThreadsPerFunc.insert(
              std::make_pair(funcOp, SmallVector<scf::ForeachThreadOp>{op}));
        else
          it->second.push_back(op);
        return WalkResult::advance();
      });
  if (walkResult.wasInterrupted())
    return DiagnosedSilenceableFailure::definiteFailure();

  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps;
  state.getTopLevel()->walk(
      [&](IREE::HAL::ExecutableExportOp op) { exportOps[op.sym_name()] = op; });

  walkResult =
      state.getTopLevel()->walk<WalkOrder::PostOrder>([&](func::FuncOp funcOp) {
        auto maybeJoinedRanges =
            joinStaticRanges(foreachThreadsPerFunc.lookup(funcOp));
        if (failed(maybeJoinedRanges)) return WalkResult::interrupt();
        for (auto foreachThreadOp : foreachThreadsPerFunc.lookup(funcOp)) {
          auto workgroupSize = functional::applyReturningPatternAt(
              ForeachThreadToGpuRewriter(*maybeJoinedRanges, getContext()),
              foreachThreadOp);
          if (failed(workgroupSize)) return WalkResult::interrupt();
        }
        auto exportOp = exportOps.lookup(funcOp.getName());
        OpBuilder b(exportOp);
        auto newAttr = b.getIndexArrayAttr(*maybeJoinedRanges);
        exportOp->setAttr(exportOp.workgroup_sizeAttrName(), newAttr);
        return WalkResult::advance();
      });

  return walkResult.wasInterrupted()
             ? DiagnosedSilenceableFailure::definiteFailure()
             : DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// VectorDistributionOp.
//===---------------------------------------------------------------------===//

/// Emit shared local memory allocation in case it is needed when lowering the
/// warp operations.
static Value allocateGlobalSharedMemory(Location loc, OpBuilder &builder,
                                        vector::WarpExecuteOnLane0Op warpOp,
                                        Type type) {
  MemRefType memrefType;
  if (auto vectorType = type.dyn_cast<VectorType>()) {
    memrefType =
        MemRefType::get(vectorType.getShape(), vectorType.getElementType(), {},
                        gpu::GPUDialect::getWorkgroupAddressSpace());
  } else {
    memrefType = MemRefType::get({1}, type, {},
                                 gpu::GPUDialect::getWorkgroupAddressSpace());
  }
  return builder.create<memref::AllocOp>(loc, memrefType);
}

/// Emit warp reduction code sequence for a given input.
static Value warpReduction(Location loc, OpBuilder &builder, Value input,
                           vector::CombiningKind kind, uint32_t size) {
  Value laneVal = input;
  // Parallel reduction using butterfly shuffles.
  for (uint64_t i = 1; i < size; i <<= 1) {
    Value shuffled = builder
                         .create<gpu::ShuffleOp>(loc, laneVal, i,
                                                 /*width=*/size,
                                                 /*mode=*/gpu::ShuffleMode::XOR)
                         .result();
    laneVal = makeArithReduction(builder, loc, kind, laneVal, shuffled);
  }
  return laneVal;
}

namespace {

/// Pattern to convert InsertElement to broadcast, this is a workaround until
/// MultiDimReduction distribution is supported.
class InsertElementToBroadcast final
    : public OpRewritePattern<vector::InsertElementOp> {
 public:
  using OpRewritePattern<vector::InsertElementOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertElementOp insertOp,
                                PatternRewriter &rewriter) const override {
    if (insertOp.getDestVectorType().getNumElements() != 1) return failure();
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        insertOp, insertOp.getDestVectorType(), insertOp.getSource());
    return success();
  }
};

}  // namespace

// TODO: Figure out the proper canonicalization and drop the complexity here.
// TODO: More sophisticated detection for matching
//   (threadIdx.x == 0 && other stuff not involving threadIdx.x)
static LogicalResult isThreadIdxxZeroPredicate(scf::IfOp ifOp) {
  if (!ifOp || ifOp.getNumResults() > 0 ||
      ifOp.getThenRegion().getBlocks().size() != 1 ||
      !ifOp.getElseRegion().empty())
    return failure();
  auto pred = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!pred) return failure();
  auto EQ = arith::CmpIPredicate::eq;
  auto SLT = arith::CmpIPredicate::slt;
  auto SLE = arith::CmpIPredicate::sle;
  auto ULT = arith::CmpIPredicate::ult;
  auto ULE = arith::CmpIPredicate::ule;
  if (auto threadIdOp = pred.getLhs().getDefiningOp<gpu::ThreadIdOp>()) {
    if (pred.getPredicate() == EQ && isConstantIntValue(pred.getRhs(), 0))
      return success();
    if (pred.getPredicate() == SLE && isConstantIntValue(pred.getRhs(), 0))
      return success();
    if (pred.getPredicate() == ULE && isConstantIntValue(pred.getRhs(), 0))
      return success();
    if (pred.getPredicate() == SLT && isConstantIntValue(pred.getRhs(), 1))
      return success();
    if (pred.getPredicate() == ULT && isConstantIntValue(pred.getRhs(), 1))
      return success();
  }
  auto SGT = arith::CmpIPredicate::sgt;
  auto SGE = arith::CmpIPredicate::sge;
  auto UGT = arith::CmpIPredicate::ugt;
  auto UGE = arith::CmpIPredicate::uge;
  if (auto threadIdOp = pred.getRhs().getDefiningOp<gpu::ThreadIdOp>()) {
    if (pred.getPredicate() == EQ && isConstantIntValue(pred.getLhs(), 0))
      return success();
    if (pred.getPredicate() == SGE && isConstantIntValue(pred.getLhs(), 0))
      return success();
    if (pred.getPredicate() == UGE && isConstantIntValue(pred.getLhs(), 0))
      return success();
    if (pred.getPredicate() == SGT && isConstantIntValue(pred.getLhs(), 1))
      return success();
    if (pred.getPredicate() == UGT && isConstantIntValue(pred.getLhs(), 1))
      return success();
  }
  return failure();
}

struct VectorDistributionResult {
  Operation *res;
};

static FailureOr<VectorDistributionResult> vectorDistribution(
    PatternRewriter &rewriter, Location loc, scf::IfOp ifOp,
    int64_t workgroupSizeX, int64_t warpSize) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(ifOp);

  // Bail if cond is not `if (threadIdx.x == 0)`.
  if (failed(isThreadIdxxZeroPredicate(ifOp)))
    return ifOp->emitError("unmet prerequisite: isThreadIdxxZeroPredicate");

  // All the code below will be executed on a single warp given a fixed
  // (threadIdxy, threadIdxz).
  Value threadIdxx = rewriter.create<gpu::ThreadIdOp>(
      loc, rewriter.getIndexType(), gpu::Dimension::x);

  assert(workgroupSizeX % warpSize == 0);
  if (workgroupSizeX != warpSize) {
    // Add a guard for `threadIdxx < warp size` around the WarpExecuteOnLane0Op.
    Value predicate = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, threadIdxx,
        rewriter.create<arith::ConstantIndexOp>(loc, warpSize));
    // Note: return-less IfOp is built with a terminator, no need to add one.
    auto newIfOp =
        rewriter.create<scf::IfOp>(loc, predicate, /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&newIfOp.getThenRegion().front());
  }
  auto warpOp = rewriter.create<vector::WarpExecuteOnLane0Op>(
      loc, TypeRange(), threadIdxx, warpSize);

  // Move the code from the previous ifOp to the WarpExecuteOnLane0Op.
  Block &sourceBlock = ifOp.getThenRegion().front();
  Block &targetBlock = warpOp.getWarpRegion().front();
  Block::iterator insertionPoint = targetBlock.begin();
  targetBlock.getOperations().splice(insertionPoint,
                                     sourceBlock.getOperations(),
                                     sourceBlock.without_terminator().begin(),
                                     sourceBlock.without_terminator().end());
  rewriter.setInsertionPointToEnd(&targetBlock);
  rewriter.create<vector::YieldOp>(loc);

  // Erase old op.
  rewriter.eraseOp(ifOp);

  // Hoist the scalar code outside of the warp region.
  // Note: moving code does not require a listener.
  vector::moveScalarUniformCode(warpOp);

  return VectorDistributionResult{warpOp};
}

static LogicalResult applyMultiReductionLoweringPatterns(Operation *target) {
  assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());

  MLIRContext *ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  vector::populateVectorMultiReductionLoweringPatterns(
      patterns, vector::VectorMultiReductionLowering::InnerReduction);
  patterns.add<InsertElementToBroadcast>(ctx);
  return applyPatternsAndFoldGreedily(target, std::move(patterns));
}

static LogicalResult applyVectorTransferWriteDistribution(Operation *target) {
  assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());

  auto distributionFn = [](vector::TransferWriteOp writeOp) {
    // Create a map (d0, d1) -> (d1) to distribute along the inner
    // dimension. Once we support n-d distribution we can add more
    // complex cases.
    int64_t vecRank = writeOp.getVectorType().getRank();
    OpBuilder builder(writeOp.getContext());
    auto map =
        AffineMap::get(vecRank, 0, builder.getAffineDimExpr(vecRank - 1));
    return map;
  };
  MLIRContext *ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  vector::populateDistributeTransferWriteOpPatterns(patterns, distributionFn);
  return applyPatternsAndFoldGreedily(target, std::move(patterns));
}

static LogicalResult applyPropagateVectorDistribution(Operation *target) {
  assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());

  MLIRContext *ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  vector::populatePropagateWarpVectorDistributionPatterns(patterns);
  vector::populateDistributeReduction(patterns, warpReduction);
  return applyPatternsAndFoldGreedily(target, std::move(patterns));
}

static LogicalResult applyWarpExecuteOnLane0ToScf(Operation *target) {
  assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());

  MLIRContext *ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  vector::WarpExecuteOnLane0LoweringOptions options;
  options.warpAllocationFn = allocateGlobalSharedMemory;
  options.warpSyncronizationFn = [](Location loc, OpBuilder &builder,
                                    vector::WarpExecuteOnLane0Op warpOp) {};
  vector::populateWarpExecuteOnLane0OpToScfForPattern(patterns, options);
  return applyPatternsAndFoldGreedily(target, std::move(patterns));
}

LogicalResult distributeWarpExecuteOnLane0(Operation *target) {
  assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());

  // TODO: Pass transform::TransformState &state and attach Listener with
  // auto &listener = state.addExtension<::detail::TrackingListener>();
  // auto detachListener = llvm::make_scope_exit(
  //   [&] { state.removeExtension<::detail::TrackingListener>(); });
  // if (failed(mapBlockArguments(state)))
  //   return DiagnosedSilenceableFailure::definiteFailure();

  // MultiReduction lowering is necessary until we have explicit support for
  // distributing that op.
  if (failed(applyMultiReductionLoweringPatterns(target))) return failure();
  if (failed(applyVectorTransferWriteDistribution(target))) return failure();
  if (failed(applyPropagateVectorDistribution(target))) return failure();
  if (failed(applyWarpExecuteOnLane0ToScf(target))) return failure();
  return success();
}

// TODO: Refactor in a generic util that can be reused.
static IREE::HAL::ExecutableExportOp getExecutableExportOpForFunc(
    IREE::HAL::ExecutableVariantOp halExecutableVariantOp,
    func::FuncOp funcOp) {
  IREE::HAL::ExecutableExportOp exportOp;
  halExecutableVariantOp->walk([&](IREE::HAL::ExecutableExportOp op) {
    if (op.sym_name() != funcOp.getName()) WalkResult::advance();
    exportOp = op;
    WalkResult::interrupt();
  });
  return exportOp;
}

// TODO: Upstream this.
template <typename OpTy>
static OpTy getSelfOrParentOfType(Operation *op) {
  auto opOfType = dyn_cast<OpTy>(op);
  if (!opOfType) opOfType = op->getParentOfType<OpTy>();
  return opOfType;
}

LogicalResult transform_dialect::VectorDistributionOp::applyToOne(
    Operation *target, transform::TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    InFlightDiagnostic diag = emitOpError()
                              << "applies only to isolated-from-above targets";
    diag.attachNote(target->getLoc()) << "non-isolated target";
    return diag;
  }

  auto halExecutableVariantOp =
      getSelfOrParentOfType<IREE::HAL::ExecutableVariantOp>(target);
  auto funcOp = getSelfOrParentOfType<func::FuncOp>(target);
  assert(funcOp);
  IREE::HAL::ExecutableExportOp exportOp =
      getExecutableExportOpForFunc(halExecutableVariantOp, funcOp);
  assert(exportOp && "missing export op");

  auto maybeAttr = exportOp.workgroup_size();
  if (!maybeAttr)
    return exportOp->emitError("export op must have workgroup_size attribute");

  int64_t workgroupSizeX = (*maybeAttr)[0].cast<IntegerAttr>().getInt();

  int64_t warpSize = getWarpSize();
  if (workgroupSizeX % warpSize != 0) {
    return exportOp->emitError()
           << "vector distribution requires workgroup size for x to be a "
           << "multiple of the warp size: " << workgroupSizeX << " vs "
           << warpSize;
  }

  WalkResult walkResult = target->walk([&](scf::IfOp ifOp) {
    functional::detail::SimpleRewriter rewriter(getContext());
    rewriter.setInsertionPoint(ifOp);
    FailureOr<VectorDistributionResult> vectorDistributionResult =
        vectorDistribution(rewriter, target->getLoc(), ifOp, workgroupSizeX,
                           warpSize);
    if (failed(vectorDistributionResult) ||
        failed(distributeWarpExecuteOnLane0(target))) {
      target->emitError("failed to apply");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted() ? failure() : success();
}

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/LLVMGPU/TransformDialectExtensions/TransformDialectLLVMGPUExtensionsOps.cpp.inc"
