// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "LLVMGPUExtensions.h"

#include "iree-dialects/Dialect/LinalgTransform/SimplePatternRewriter.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using llvm::dbgs;

#define DEBUG_TYPE "transform-llvmgpu-extensions"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(dbgs() << '[' << DEBUG_TYPE << "] " << X)

using namespace mlir;
using namespace mlir::iree_compiler::IREE;

iree_compiler::IREE::transform_dialect::LLVMGPUExtensions::LLVMGPUExtensions() {
  // CreateAsyncGroupsOp depends on the following two dialects.
  declareGeneratedDialect<gpu::GPUDialect>();
  declareGeneratedDialect<nvgpu::NVGPUDialect>();

  registerTransformOps<
#define GET_OP_LIST
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensionsOps.cpp.inc"
      >();
}

void mlir::iree_compiler::registerTransformDialectLLVMGPUExtension(
    DialectRegistry &registry) {
  registry.addExtensions<transform_dialect::LLVMGPUExtensions>();
}

//===---------------------------------------------------------------------===//
// IREE-specific LLVMGPU transformations.
//===---------------------------------------------------------------------===//

static LogicalResult checkNoMoreWarpMappingAttributes(
    scf::ForallOp forallOp,
    ArrayRef<DeviceMappingAttrInterface> warpMappingAttributes) {
  auto maybeMapping = forallOp.getMapping();
  // If no mapping, we are good.
  if (!maybeMapping) return success();

  // If the intersection is empty, we are good.
  SetVector<Attribute> s1(maybeMapping->begin(), maybeMapping->end());
  SetVector<Attribute> s2(warpMappingAttributes.begin(),
                          warpMappingAttributes.end());
  int64_t sizeBefore = s1.size();
  s1.set_subtract(s2);
  if (sizeBefore == s1.size()) return success();

  // Otherwise, fail.
  LDBG("In func:" << forallOp->getParentOfType<func::FuncOp>() << "\n");
  LDBG("--forall with warp attr was not mapped:" << forallOp << "\n");
  forallOp->emitOpError(
      "Mapping failed: is threadIdx.x a multiple of the warp size?");
  return failure();
}

void transform_dialect::MapNestedForallToGpuThreadsOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    ArrayRef<int64_t> workgroupSize) {
  result.addOperands(target);
  result.addAttribute(
      MapNestedForallToGpuThreadsOp::getWorkgroupSizeAttrName(result.name),
      builder.getI64ArrayAttr(workgroupSize));
  MLIRContext *ctx = builder.getContext();
  result.addTypes({pdl::OperationType::get(ctx)});
}

/// Return a flatten thread id for the workgroup with given sizes.
static OpFoldResult getLinearThreadId(RewriterBase &rewriter, Location loc,
                                      ArrayRef<OpFoldResult> threads,
                                      ArrayRef<OpFoldResult> workgroupSize) {
  assert(threads.size() == 3 && "expected 3 threads");
  assert(workgroupSize.size() == 3 && "expected 3 workgroup sizes");
  AffineExpr tx, ty, tz, BDX, BDY;
  bindDims(rewriter.getContext(), tx, ty, tz);
  bindSymbols(rewriter.getContext(), BDX, BDY);
  SmallVector<OpFoldResult> threadsAndWorkGroups(threads);
  llvm::append_range(threadsAndWorkGroups, workgroupSize);
  return makeComposedFoldedAffineApply(
      rewriter, loc, tx + ty * BDX + tz * BDX * BDY, threadsAndWorkGroups);
}

/// Rewrite scf.forall by distributing a flat id onto multiple dimensions.
// This is a derived version of mapNestedForallToThreadsImpl that handles flat
// ids to support distributing with a different shape than the block sizes.
static DiagnosedSilenceableFailure rewriteOneForallToGpuWithLinearThreadId(
    RewriterBase &rewriter, scf::ForallOp forallOp, int64_t numWarps,
    Value linearThreadId, bool syncAfterDistribute,
    std::optional<transform::TransformOpInterface> transformOp,
    const ArrayRef<DeviceMappingAttrInterface> &threadMappingAttributes) {
  // Step 0. Target-specific verifications. There is no good place to anchor
  // those right now: the ForallOp is target-independent and the
  // transform op does not apply to individual ForallOp.
  auto failureHelper =
      [&](const Twine &message) -> DiagnosedSilenceableFailure {
    if (transformOp.has_value()) {
      return transformOp->emitSilenceableError() << message;
    }
    return emitDefiniteFailure(forallOp, message);
  };
  Location loc = forallOp->getLoc();
  if (!forallOp.isNormalized())
    return failureHelper("unsupported non-normalized loops");
  if (forallOp.getNumResults() > 0)
    return failureHelper("only bufferized scf.forall lowers to gpu.thread_id");
  if (forallOp.getRank() > 3)
    return failureHelper(
        "scf.forall with rank > 3 does not lower to gpu.thread_id");
  if (llvm::any_of(forallOp.getMixedUpperBound(), [](OpFoldResult ofr) {
        return !getConstantIntValue(ofr).has_value();
      })) {
    return failureHelper("unsupported dynamic blockdim size");
  }
  if (!forallOp.getMapping().has_value())
    return failureHelper("mapping must be present");
  SmallVector<Attribute> threadMapping =
      llvm::to_vector(forallOp.getMapping()->getValue());

  // Step 1. Complete the threadMapping to a full mapping (with 1s) if
  // necessary.
  SmallVector<Value> numThreads = forallOp.getUpperBound(rewriter);
  // Ensure we have 3 block sizes, one for each id.
  Value one;
  for (auto attr : threadMappingAttributes) {
    if (std::find(threadMapping.begin(), threadMapping.end(), attr) ==
        threadMapping.end()) {
      threadMapping.push_back(attr);
      one = one ? one : rewriter.create<arith::ConstantIndexOp>(loc, 1);
      numThreads.push_back(one);
    }
  }

  LDBG("Start delinearizing forall: " << forallOp << "\n");

  // Step 2. sort the values by the corresponding DeviceMappingAttrInterface.
  auto comparator = [&](DeviceMappingAttrInterface a,
                        DeviceMappingAttrInterface b) -> bool {
    return a.getMappingId() < b.getMappingId();
  };
  SmallVector<Value> blockDimValues =
      getValuesSortedByKey(threadMapping, numThreads, comparator);
  SmallVector<int64_t> blockDims =
      llvm::to_vector(llvm::map_range(blockDimValues, [](Value v) {
        return v.getDefiningOp<arith::ConstantIndexOp>().value();
      }));

  // Step 3. Create the gpu.thread ops and map the induction variables to the
  // newly created ops.
  // blockDims is in [x, y, z] order, but we delinearize in [z, y, x] order.
  LLVM_DEBUG(llvm::interleaveComma(
                 blockDims, DBGS() << "--delinearizing with dims(x, y, z): ");
             dbgs() << "\n";);

  SmallVector<int64_t> reverseBlockDims(llvm::reverse(blockDims));
  reverseBlockDims.push_back(32);
  LLVM_DEBUG(
      llvm::interleaveComma(
          reverseBlockDims,
          DBGS() << "--delinearizing with reverse dims(wz, wy, wx, tx): ");
      dbgs() << "\n";);

  SmallVector<int64_t> strides = computeStrides(reverseBlockDims);
  AffineExpr d0;
  bindDims(rewriter.getContext(), d0);
  SmallVector<AffineExpr> delinearizingExprs = delinearize(d0, strides);
  LLVM_DEBUG(
      llvm::interleaveComma(
          strides, DBGS() << "--delinearizing strides(wz, wy, wx, tx): ");
      dbgs() << "\n"; llvm::interleaveComma(
          delinearizingExprs, DBGS()
                                  << "--delinearizing exprs(wz, wy, wx, tx): ");
      dbgs() << "\n";);
  SmallVector<Value> threadOpsUpdated;
  for (AffineExpr e : delinearizingExprs) {
    LDBG("----step func: " << forallOp->getParentOfType<func::FuncOp>()
                           << "\n");
    threadOpsUpdated.push_back(
        makeComposedAffineApply(rewriter, loc, e, linearThreadId));
  }
  LDBG("----step func: " << forallOp->getParentOfType<func::FuncOp>() << "\n");
  // At this point we have: (wz, wy, wx, tx), drop tx.
  threadOpsUpdated.resize(3);
  // And reverse to get to: (wx, wy, wz)
  threadOpsUpdated = SmallVector<Value>(llvm::reverse(threadOpsUpdated));

  // Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  int64_t flatblockDim = 1;
  for (int64_t dim : blockDims) {
    flatblockDim *= dim;
  }
  // int64_t dimUsed = 1;
  // for (int64_t dim : blockDims) {
  //   if (dim == 1) {
  //     threadOpsUpdated.push_back(zero);
  //     continue;
  //   }
  //   dimUsed *= dim;
  //   AffineExpr d0 = rewriter.getAffineDimExpr(0);
  //   Value dimId = dimUsed == flatblockDim
  //                     ? linearThreadId
  //                     : makeComposedAffineApply(rewriter, loc, d0
  //                     % dim,
  //                                               {linearThreadId});
  //   threadOpsUpdated.push_back(dimId);
  //   linearThreadId = makeComposedAffineApply(rewriter, loc,
  //   d0.floorDiv(dim),
  //                                            {linearThreadId});
  // }
  IRMapping bvm;
  for (auto [blockIdx, blockDim] :
       llvm::zip(forallOp.getInductionVars(), threadMapping)) {
    bvm.map(blockIdx,
            threadOpsUpdated[blockDim.cast<DeviceMappingAttrInterface>()
                                 .getMappingId()]);
  }

  // Step 4. Maybe create conditionals to predicate the region.
  Value predicate;
  if (flatblockDim > numWarps) {
    return failureHelper(
        "The requested GPU threads are fewer than the number of loop trip "
        "counts. Try to tile scf.forall before mapping or set "
        "small blockDim.");
  }
  if (flatblockDim != numWarps) {
    Value blockIdx = rewriter.create<arith::ConstantIndexOp>(loc, flatblockDim);
    predicate = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                               linearThreadId, blockIdx);
  }

  // Step 5. Move the body of forallOp.
  // Erase the terminator first, it will not be used.
  rewriter.eraseOp(forallOp.getTerminator());
  Block *targetBlock;
  Block::iterator insertionPoint;
  if (predicate) {
    // Step 5.a. If predicated, move at the beginning.
    auto ifOp = rewriter.create<scf::IfOp>(loc, predicate,
                                           /*withElseRegion=*/false);
    targetBlock = ifOp.thenBlock();
    insertionPoint = ifOp.thenBlock()->begin();
  } else {
    // Step 5.b. Otherwise, move inline just before forallOp.
    targetBlock = forallOp->getBlock();
    insertionPoint = Block::iterator(forallOp);
  }
  Block &sourceBlock = forallOp.getRegion().front();
  targetBlock->getOperations().splice(insertionPoint,
                                      sourceBlock.getOperations());

  // Step 6. RAUW thread indices to thread ops.
  for (Value loopIndex : forallOp.getInductionVars()) {
    Value threadIdx = bvm.lookup(loopIndex);
    rewriter.replaceAllUsesWith(loopIndex, threadIdx);
  }

  // Step 7. syncthreads.
  // TODO: Need warpsync
  if (syncAfterDistribute) rewriter.create<gpu::BarrierOp>(loc);

  // Step 8. Erase old op.
  rewriter.eraseOp(forallOp);

  return DiagnosedSilenceableFailure::success();
}

static LogicalResult rewriteOneForallToGpuWithLinearThreadId(
    RewriterBase &rewriter, scf::ForallOp forallOp, int64_t numWarps,
    ArrayRef<OpFoldResult> foldedThreads, ArrayRef<OpFoldResult> workgroupSize,
    bool syncAfterDistribute,
    std::optional<transform::TransformOpInterface> transformOp,
    const ArrayRef<DeviceMappingAttrInterface> &warpMappingAttributes) {
  Location loc = forallOp->getLoc();

  // Ignore forall that do not contain only warp mapping attributes.
  // Return a failure that encodes "failure to match".
  for (Attribute map : forallOp.getMapping()->getValue())
    if (!llvm::is_contained(warpMappingAttributes, map)) return failure();

  rewriter.setInsertionPoint(forallOp);

  OpFoldResult linearThreadId =
      getLinearThreadId(rewriter, loc, foldedThreads, workgroupSize);
  // AffineExpr d0 = rewriter.getAffineDimExpr(0);
  // linearThreadId = makeComposedFoldedAffineApply(
  //     rewriter, loc, d0.floorDiv(mlir::iree_compiler::kWarpSize),
  //     {linearThreadId});

  DiagnosedSilenceableFailure diagnosedFailure =
      rewriteOneForallToGpuWithLinearThreadId(
          rewriter, forallOp, numWarps,
          getValueOrCreateConstantIndexOp(rewriter, loc, linearThreadId),
          /*syncAfterDistribute=*/true, transformOp, warpMappingAttributes);

  return success(diagnosedFailure.succeeded());
}

// TODO: if the number of threads was wired like the workgroup_count, we could
// reuse most of the code and not require a static number of threads.
// TODO: synchronizations for imperfectly nested stuff.
DiagnosedSilenceableFailure
transform_dialect::MapNestedForallToGpuThreadsOp::applyToOne(
    func::FuncOp target, transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  if (!isa<HAL::ExecutableOp, HAL::ExecutableVariantOp>(state.getTopLevel())) {
    state.getTopLevel()->emitOpError(
        "requires HAL::ExecutableOp or HAL::ExecutableVariantOp "
        "toplevel to "
        "attach the workgroup size information to a nested "
        "ExecutableExportOp");
    return emitDefaultDefiniteFailure(target);
  }

  IREE::HAL::ExecutableExportOp exportOp;
  state.getTopLevel()->walk([&](IREE::HAL::ExecutableExportOp op) {
    if (op.getSymName() == target.getName()) exportOp = op;
  });
  if (!exportOp) {
    state.getTopLevel()->emitOpError("no IREE::HAL::ExecutableExportOp found");
    return emitDefaultDefiniteFailure(target);
  }

  SmallVector<int64_t> workgroupSize =
      extractFromI64ArrayAttr(getWorkgroupSize());
  // TODO: no magic constant but IREE uses this extensively.
  workgroupSize.resize(/*size=*/3, /*value=*/1);

  auto transformOp = cast<transform::TransformOpInterface>(getOperation());

  SimplePatternRewriter rewriter(target);
  // Just insert threadIds at the top of the function so we can
  // reuse.
  rewriter.setInsertionPointToStart(&target.getFunctionBody().front());
  Location loc = target->getLoc();
  auto indexType = rewriter.getIndexType();
  SmallVector<Value> threads{
      rewriter.create<gpu::ThreadIdOp>(loc, indexType, gpu::Dimension::x),
      rewriter.create<gpu::ThreadIdOp>(loc, indexType, gpu::Dimension::y),
      rewriter.create<gpu::ThreadIdOp>(loc, indexType, gpu::Dimension::z)};
  auto threadIdGenerator = [&](RewriterBase &rewriter, scf::ForallOp forallOp,
                               SmallVectorImpl<Value> &threadIds) {
    threadIds.assign(threads);
  };

  MLIRContext *ctx = target->getContext();
  SmallVector<DeviceMappingAttrInterface> threadMappingAttributes = {
      gpu::GPUThreadMappingAttr::get(ctx, gpu::Threads::DimX),
      gpu::GPUThreadMappingAttr::get(ctx, gpu::Threads::DimY),
      gpu::GPUThreadMappingAttr::get(ctx, gpu::Threads::DimZ)};
  DiagnosedSilenceableFailure diag =
      mlir::transform::gpu::mapNestedForallToThreadsImpl(
          rewriter, transformOp, target, workgroupSize, true,
          threadMappingAttributes, threadIdGenerator);

  if (diag.succeeded()) {
    auto newAttr = rewriter.getIndexArrayAttr(workgroupSize);
    // TODO: should really be:
    // exportOp.setWorkgroupSizeAttr(newAttr);
    exportOp->setAttr(exportOp.getWorkgroupSizeAttrName(), newAttr);
  }

  // Map warpIds, only warpSize divides the total number of threads.
  int64_t totalNumThreads =
      workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
  SmallVector<DeviceMappingAttrInterface> warpMappingAttributes = {
      gpu::GPUWarpMappingAttr::get(ctx, gpu::Warps::DimX),
      gpu::GPUWarpMappingAttr::get(ctx, gpu::Warps::DimY),
      gpu::GPUWarpMappingAttr::get(ctx, gpu::Warps::DimZ)};
  if (diag.succeeded() && (totalNumThreads % kWarpSize == 0)) {
    // Walk forall ops and try to map all warpMappingAttributes.
    WalkResult res = target->walk([&](scf::ForallOp forallOp) -> WalkResult {
      // Optimize away threadIds that are always zero.
      SmallVector<OpFoldResult> foldedThreads =
          getAsIndexOpFoldResult(ctx, {0, 0, 0});
      for (int64_t i = 0; i < 3; ++i) {
        if (workgroupSize[i] == 1) continue;
        foldedThreads[i] = threads[i];
      }
      int64_t numWarps = totalNumThreads / kWarpSize;
      LogicalResult res = rewriteOneForallToGpuWithLinearThreadId(
          rewriter, forallOp, numWarps, foldedThreads,
          getAsIndexOpFoldResult(ctx, workgroupSize),
          /*syncAfterDistribute=*/true, transformOp, warpMappingAttributes);

      // If any warp mapping attribute remains, interrupt and
      // fail hard.
      if (failed(res))
        return checkNoMoreWarpMappingAttributes(forallOp,
                                                warpMappingAttributes);

      return success();
    });

    // If any warp mapping attribute remains, fail hard.
    if (res.wasInterrupted()) return emitDefaultDefiniteFailure(target);
  }

  results.push_back(target);
  return diag;
}

//===---------------------------------------------------------------------===//
// VectorToWarpExecuteOnLane0Op.
//===---------------------------------------------------------------------===//
void transform_dialect::VectorToWarpExecuteOnLane0Op::build(
    OpBuilder &builder, OperationState &result, Value target,
    int64_t warpSize) {
  MLIRContext *ctx = builder.getContext();
  result.addOperands(target);
  result.addAttribute(
      VectorToWarpExecuteOnLane0Op::getWarpSizeAttrName(result.name),
      builder.getI64IntegerAttr(warpSize));
  result.addTypes({pdl::OperationType::get(ctx)});
}

/// Helper method to replace all uses of the laneId operand by the constant
/// 0 inside the region. This is a necessary prerequisite to perform any kind of
/// hoisting of IR that is inside the region.
/// Return success if any replacement occurred, failure otherwise.
// TODO: this is currently brittle, what we really need here is a scope-aware
// SCCP.
static LogicalResult replaceAllUsesOfLaneWithin(
    RewriterBase &b, vector::WarpExecuteOnLane0Op executeOp) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(executeOp);
  Value zero = b.create<arith::ConstantIndexOp>(executeOp.getLoc(), 0);
  b.setInsertionPointToStart(&executeOp.getWarpRegion().front());
  Value laneId = executeOp.getLaneid();
  bool applied = false;
  for (Operation *user : llvm::make_early_inc_range(laneId.getUsers())) {
    if (!executeOp->isProperAncestor(user)) continue;
    b.startRootUpdate(user);
    user->replaceUsesOfWith(laneId, zero);
    b.finalizeRootUpdate(user);
    applied = true;
  }
  return success(applied);
}

/// Return the gpu::ThreadIdOp for which the predicate if equivalent to
/// `if (threadIdx.x == 0)`.
// TODO: Figure out the proper canonicalization and drop the complexity here.
// TODO: More sophisticated detection for matching
//   (threadIdx.x == 0 && other stuff not involving threadIdx.x)
static FailureOr<gpu::ThreadIdOp> isThreadIdxxZeroPredicate(scf::IfOp ifOp) {
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
    if (threadIdOp.getDimension() != gpu::Dimension::x) return failure();
    if (pred.getPredicate() == EQ && isConstantIntValue(pred.getRhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == SLE && isConstantIntValue(pred.getRhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == ULE && isConstantIntValue(pred.getRhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == SLT && isConstantIntValue(pred.getRhs(), 1))
      return threadIdOp;
    if (pred.getPredicate() == ULT && isConstantIntValue(pred.getRhs(), 1))
      return threadIdOp;
  }
  auto SGT = arith::CmpIPredicate::sgt;
  auto SGE = arith::CmpIPredicate::sge;
  auto UGT = arith::CmpIPredicate::ugt;
  auto UGE = arith::CmpIPredicate::uge;
  if (auto threadIdOp = pred.getRhs().getDefiningOp<gpu::ThreadIdOp>()) {
    if (threadIdOp.getDimension() != gpu::Dimension::x) return failure();
    if (pred.getPredicate() == EQ && isConstantIntValue(pred.getLhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == SGE && isConstantIntValue(pred.getLhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == UGE && isConstantIntValue(pred.getLhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == SGT && isConstantIntValue(pred.getLhs(), 1))
      return threadIdOp;
    if (pred.getPredicate() == UGT && isConstantIntValue(pred.getLhs(), 1))
      return threadIdOp;
  }
  return failure();
}

struct VectorDistributionResult {
  vector::WarpExecuteOnLane0Op warpOp;
};

static FailureOr<VectorDistributionResult> rewriteScfIfAsWarpExecuteOnLane0(
    PatternRewriter &rewriter, Location loc, scf::IfOp ifOp,
    int64_t workgroupSizeX, int64_t warpSize) {
  // Bail if cond is not `if (threadIdx.x == 0)`.
  FailureOr<gpu::ThreadIdOp> maybeThreadIdxxOp =
      isThreadIdxxZeroPredicate(ifOp);
  if (failed(maybeThreadIdxxOp)) return failure();

  // All the code below will be executed on a single warp given a
  // fixed (threadIdxy, threadIdxz). Note, we reuse
  // `maybeThreadIdxxOp` here because we later want to replace this
  // op instance by 0 without relying on CSE or canonicalizations.
  Value threadIdxx = *maybeThreadIdxxOp;

  assert(workgroupSizeX % warpSize == 0);
  if (workgroupSizeX != warpSize) {
    // Add a guard for `threadIdxx < warp size` around the
    // WarpExecuteOnLane0Op.
    Value predicate = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, threadIdxx,
        rewriter.create<arith::ConstantIndexOp>(loc, warpSize));
    // Note: return-less IfOp is built with a terminator, no need to
    // add one.
    auto newIfOp =
        rewriter.create<scf::IfOp>(loc, predicate, /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&newIfOp.getThenRegion().front());
  }
  auto warpOp = rewriter.create<vector::WarpExecuteOnLane0Op>(
      loc, TypeRange(), threadIdxx, warpSize);

  // Move the code from the previous ifOp to the
  // WarpExecuteOnLane0Op.
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

  // This simple rewrite propagates zero in lieu of laneId within the
  // warp_execute_on_lane_0 op.
  // Atm, this **must** occur before any hoisting of code.
  // TODO: Replace this by a more robust scoped SCCP that will make
  // it more robust re. hoisting.
  (void)replaceAllUsesOfLaneWithin(rewriter, warpOp);

  // Hoist the scalar code outside of the warp region.
  // Note: moving code does not require a listener.
  vector::moveScalarUniformCode(warpOp);

  return VectorDistributionResult{warpOp};
}

// TODO: Refactor in a generic util that can be reused.
static HAL::ExecutableExportOp getExecutableExportOpForFunc(
    HAL::ExecutableVariantOp halExecutableVariantOp, func::FuncOp funcOp) {
  if (!halExecutableVariantOp || !funcOp) return {};
  HAL::ExecutableExportOp exportOp;
  halExecutableVariantOp->walk([&](HAL::ExecutableExportOp op) {
    if (op.getSymName() != funcOp.getName()) return WalkResult::advance();
    exportOp = op;
    return WalkResult::interrupt();
  });
  return exportOp;
}

DiagnosedSilenceableFailure
transform_dialect::VectorToWarpExecuteOnLane0Op::applyToOne(
    scf::IfOp target, transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  if (!isa<HAL::ExecutableOp, HAL::ExecutableVariantOp>(state.getTopLevel())) {
    results.assign(1, nullptr);
    return emitDefaultSilenceableFailure(state.getTopLevel())
           << "requires HAL::ExecutableOp or "
              "HAL::ExecutableVariantOp toplevel "
              "so that IR is properly isolated. This is required so "
              "we can "
              "safely inspect the HAL::ExecutableExportOp under "
              "multi-threaded "
              "pass assumptions.";
  }

  auto halExecutableVariantOp =
      target->getParentOfType<HAL::ExecutableVariantOp>();
  auto funcOp = target->getParentOfType<func::FuncOp>();
  HAL::ExecutableExportOp exportOp =
      getExecutableExportOpForFunc(halExecutableVariantOp, funcOp);
  if (!halExecutableVariantOp || !funcOp || !exportOp) {
    // Return a silenceable failure and set the expected 1 result to
    // nullptr.
    results.assign(1, nullptr);
    return emitDefaultSilenceableFailure(target)
           << "export op is missing --- the transform is not "
              "applied";
  }

  Optional<ArrayAttr> maybeAttr = exportOp.getWorkgroupSize();
  // TODO: Pervasive 3 constant in IREE.
  if (!maybeAttr || maybeAttr->size() != 3) {
    // Return a silenceable failure and set the expected 1 result to
    // nullptr.
    results.assign(1, nullptr);
    return emitDefaultSilenceableFailure(target)
           << "export op must have workgroup_size attribute set "
              "with 3 entries "
              "--- the transform is not applied";
  }

  int64_t workgroupSizeX = (*maybeAttr)[0].cast<IntegerAttr>().getInt();
  int64_t warpSize = getWarpSize();
  if (workgroupSizeX % warpSize != 0) {
    // Return a silenceable failure and set the expected 1 result to
    // nullptr.
    results.assign(1, nullptr);
    return emitDefaultSilenceableFailure(target)
           << "vector distribution requires workgroup size for x to "
              "be a "
           << "multiple of the warp size: " << workgroupSizeX << " vs "
           << warpSize << " --- the transform is not applied";
  }

  SimplePatternRewriter rewriter(target);
  FailureOr<VectorDistributionResult> vectorDistributionResult =
      rewriteScfIfAsWarpExecuteOnLane0(rewriter, target->getLoc(), target,
                                       workgroupSizeX, warpSize);
  if (failed(vectorDistributionResult)) {
    // Return a silenceable failure and set the expected 1 result to
    // nullptr.
    results.assign(1, nullptr);
    return emitDefaultSilenceableFailure(target)
           << "scf::ifOp needs to be predicated on threadIdx.x == 0 "
              "--- the "
              "transform is not applied";
  }
  results.push_back(vectorDistributionResult->warpOp);
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// VectorWarpDistributionOp.
//===---------------------------------------------------------------------===//
void transform_dialect::VectorWarpDistributionOp::build(OpBuilder &builder,
                                                        OperationState &result,
                                                        Value target) {
  result.addTypes(pdl::OperationType::get(builder.getContext()));
  result.addOperands(target);
}

/// Emit shared local memory allocation in case it is needed when lowering the
/// warp operations.
static Value allocateGlobalSharedMemory(Location loc, OpBuilder &builder,
                                        vector::WarpExecuteOnLane0Op warpOp,
                                        Type type) {
  MemRefType memrefType;
  auto addressSpaceAttr = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  if (auto vectorType = type.dyn_cast<VectorType>()) {
    memrefType =
        MemRefType::get(vectorType.getShape(), vectorType.getElementType(),
                        MemRefLayoutAttrInterface{}, addressSpaceAttr);
  } else {
    memrefType = MemRefType::get({1}, type, MemRefLayoutAttrInterface{},
                                 addressSpaceAttr);
  }
  return builder.create<memref::AllocOp>(loc, memrefType);
}

/// Return a value yielded by `warpOp` which statifies the filter lamdba
/// condition and is not dead.
static OpOperand *getWarpResult(vector::WarpExecuteOnLane0Op warpOp,
                                function_ref<bool(Operation *)> fn) {
  auto yield = cast<vector::YieldOp>(
      warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
  for (OpOperand &yieldOperand : yield->getOpOperands()) {
    Value yieldValues = yieldOperand.get();
    Operation *definedOp = yieldValues.getDefiningOp();
    if (definedOp && fn(definedOp)) {
      if (!warpOp.getResult(yieldOperand.getOperandNumber()).use_empty())
        return &yieldOperand;
    }
  }
  return {};
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

/// Sink out load op feeding into a warp op yield.
/// ```
/// %0 = vector.warp_execute_on_lane_0(%arg0) -> (f32) {
///   ...
//    %2 = memref.load %src[%c0] : memref<1024xf32>
///   vector.yield %2 : f32
/// }
/// ```
/// To
/// ```
/// %dead = vector.warp_execute_on_lane_0(%arg0) -> (f32) {
///   ...
//    %2 = memref.load %src[%c0] : memref<1024xf32>
///   vector.yield %2 : f32
/// }
/// gpu.synchronize
/// %0 = memref.load %src[%c0] : memref<1024xf32>
struct WarpOpLoad : public OpRewritePattern<vector::WarpExecuteOnLane0Op> {
  using OpRewritePattern<vector::WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand = getWarpResult(
        warpOp, [](Operation *op) { return isa<memref::LoadOp>(op); });
    if (!operand) return failure();
    auto load = operand->get().getDefiningOp<memref::LoadOp>();
    unsigned operandIndex = operand->getOperandNumber();
    Value distributedVal = warpOp.getResult(operandIndex);

    SmallVector<Value, 4> indices(load.getIndices().begin(),
                                  load.getIndices().end());
    if (!indices.empty()) return failure();

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(warpOp);
    // TODO: generalize this.
    // options.warpSyncronizationFn currently must take a
    // WarpExecuteOnLane0Op which we don't have here.
    rewriter.create<gpu::BarrierOp>(load.getLoc());
    Value newRead = rewriter.create<memref::LoadOp>(
        load.getLoc(), distributedVal.getType(), load.getMemref(), indices);

    // The result type of WarpExecuteOnLane0Op may or may not match
    // the yielded type depending on whether the op has "broadcast"
    // behavior (see the doc of WarpExecuteOnLane0Op).
    for (OpOperand &use : distributedVal.getUses()) {
      rewriter.startRootUpdate(use.getOwner());
      Value replacement = newRead;
      if (use.get().getType() != newRead.getType()) {
        replacement = rewriter.create<vector::BroadcastOp>(
            load.getLoc(), use.get().getType(), newRead);
      }
      use.getOwner()->setOperand(use.getOperandNumber(), replacement);
      rewriter.finalizeRootUpdate(use.getOwner());
    }
    return success();
  }
};

/// Shared memory allocations are representated as AllocOp in IREE but they
/// really have the semantic of global variables. Therefore hoisting them is
/// always correct for static allocations.
struct HoistSharedMemoryAlloc : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::AllocOp alloc,
                                PatternRewriter &rewriter) const override {
    auto addressSpaceAttr =
        alloc.getType().getMemorySpace().dyn_cast<gpu::AddressSpaceAttr>();
    if (!(addressSpaceAttr &&
          addressSpaceAttr.getValue() !=
              gpu::GPUDialect::getWorkgroupAddressSpace()) ||
        alloc.getNumOperands() != 0)
      return failure();
    auto warpParent = alloc->getParentOfType<vector::WarpExecuteOnLane0Op>();
    if (!warpParent) return failure();
    alloc->moveBefore(warpParent);
    // Conservatively move the dealloc after the warpOp. This may
    // extend the liverange of the allocation but is always correct.
    for (Operation *user : alloc->getUsers()) {
      if (isa<memref::DeallocOp>(user)) user->moveAfter(warpParent);
    }
    return success();
  }
};

}  // namespace

static void populateMultiReductionLoweringPatterns(Operation *target,
                                                   RewritePatternSet &patterns,
                                                   PatternBenefit benefit) {
  assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());

  vector::populateVectorMultiReductionLoweringPatterns(
      patterns, vector::VectorMultiReductionLowering::InnerReduction, benefit);
  patterns.add<InsertElementToBroadcast>(target->getContext(), benefit);
}

static AffineMap simpleDistributionFunction(Value val) {
  AffineMap map = AffineMap::get(val.getContext());
  auto vecType = val.getType().dyn_cast<VectorType>();
  if (!vecType) return map;
  // Create a map (d0, d1) -> (d1) to distribute along the inner
  // dimension. Once we support n-d distribution we can add more
  // complex cases.
  int64_t vecRank = vecType.getRank();
  OpBuilder builder(val.getContext());
  map = AffineMap::get(vecRank, 0, builder.getAffineDimExpr(vecRank - 1));
  return map;
}

static void populateVectorTransferWriteDistribution(Operation *target,
                                                    RewritePatternSet &patterns,
                                                    PatternBenefit benefit) {
  assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());
  vector::populateDistributeTransferWriteOpPatterns(
      patterns, simpleDistributionFunction, benefit);
}

static Value simpleWarpShuffleFunction(Location loc, OpBuilder &builder,
                                       Value val, Value srcIdx,
                                       int64_t warpSz) {
  assert((val.getType().isF32() || val.getType().isInteger(32)) &&
         "unsupported shuffle type");
  Type i32Type = builder.getIntegerType(32);
  Value srcIdxI32 = builder.create<arith::IndexCastOp>(loc, i32Type, srcIdx);
  Value warpSzI32 = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(i32Type, warpSz));
  Value result = builder
                     .create<gpu::ShuffleOp>(loc, val, srcIdxI32, warpSzI32,
                                             gpu::ShuffleMode::IDX)
                     .getResult(0);
  return result;
}

static void populatePropagateVectorDistribution(Operation *target,
                                                RewritePatternSet &patterns,
                                                PatternBenefit benefit) {
  auto groupReductionFn = [](Location loc, OpBuilder &builder, Value input,
                             vector::CombiningKind kind, uint32_t size) {
    return mlir::iree_compiler::emitGPUGroupReduction(loc, builder, input, kind,
                                                      size, 32);
  };
  assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());
  vector::populatePropagateWarpVectorDistributionPatterns(
      patterns, simpleDistributionFunction, simpleWarpShuffleFunction, benefit);
  vector::populateDistributeReduction(patterns, groupReductionFn, benefit);
  patterns.add<WarpOpLoad, HoistSharedMemoryAlloc>(target->getContext(),
                                                   benefit);
}

static void warpSyncronizationFn(Location loc, OpBuilder &builder,
                                 vector::WarpExecuteOnLane0Op warpOp) {
  builder.create<gpu::BarrierOp>(loc);
};

static void populateWarpExecuteOnLane0ToScf(
    Operation *target, RewritePatternSet &patterns,
    const vector::WarpExecuteOnLane0LoweringOptions &options,
    PatternBenefit benefit) {
  assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());
  vector::populateWarpExecuteOnLane0OpToScfForPattern(patterns, options,
                                                      benefit);
}

DiagnosedSilenceableFailure
transform_dialect::VectorWarpDistributionOp::applyToOne(
    Operation *target, transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    target->emitOpError(
        "applies only to isolated-from-above targets because it "
        "needs to apply "
        "patterns greedily");
    return emitDefaultDefiniteFailure(target);
  }

  // TODO: Hook up into the ApplyPatternOp in CommonExtensions.cpp to
  // automatically get listening capabilities.

  MLIRContext *ctx = target->getContext();
  // MultiReduction lowering is necessary until we have explicit
  // support for distributing that op.
  RewritePatternSet preProcessingPatterns(ctx);
  populateMultiReductionLoweringPatterns(target, preProcessingPatterns,
                                         /*benefit=*/1);
  vector::ShapeCastOp::getCanonicalizationPatterns(preProcessingPatterns, ctx);
  vector::BroadcastOp::getCanonicalizationPatterns(preProcessingPatterns, ctx);
  vector::ExtractOp::getCanonicalizationPatterns(preProcessingPatterns, ctx);
  if (failed(applyPatternsAndFoldGreedily(target,
                                          std::move(preProcessingPatterns)))) {
    return mlir::emitDefiniteFailure(target,
                                     "multi-reduce patterns failed to apply");
  }

  RewritePatternSet patterns(ctx);
  populateVectorTransferWriteDistribution(target, patterns,
                                          /*benefit=*/2);
  populatePropagateVectorDistribution(target, patterns,
                                      /*benefit=*/1);
  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns)))) {
    return mlir::emitDefiniteFailure(
        target, "warp distribution patterns failed to apply");
  }

  RewritePatternSet endPatterns(ctx);
  vector::WarpExecuteOnLane0LoweringOptions options;
  options.warpAllocationFn = allocateGlobalSharedMemory;
  options.warpSyncronizationFn = warpSyncronizationFn;
  populateWarpExecuteOnLane0ToScf(target, endPatterns, options,
                                  /*benefit=*/0);
  if (failed(applyPatternsAndFoldGreedily(target, std::move(endPatterns)))) {
    return mlir::emitDefiniteFailure(
        target, "warp execute on lane 0 to scf patterns failed to apply");
  }

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::VectorToMMAConversionOp::build(OpBuilder &builder,
                                                       OperationState &result,
                                                       Value target) {
  result.addOperands(target);
  result.addTypes({pdl::OperationType::get(builder.getContext())});
}

DiagnosedSilenceableFailure
transform_dialect::VectorToMMAConversionOp::applyToOne(
    Operation *target, transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    target->emitOpError(
        "applies only to isolated-from-above targets because it "
        "needs to apply "
        "patterns greedily");
    return emitDefaultDefiniteFailure(target);
  }

  auto funcOp = dyn_cast<func::FuncOp>(target);
  if (!funcOp) {
    target->emitOpError("Must apply to a func op");
    return emitDefaultDefiniteFailure(target);
  }

  if (!(getUseMmaSync() ^ getUseWmma())) {
    target->emitOpError(
        "Exactly one of use_mma_sync or use_wmma must be specified");
    return emitDefaultDefiniteFailure(target);
  }

  MLIRContext *ctx = target->getContext();

  // Unrolling to native vector size must have previously occurred.
  // TODO: Add pattern to propagate the extract through the scf.for
  // ops. Convert slice of contract operations to mma_sync/wmma ops.
  RewritePatternSet patterns(ctx);
  mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
  populatePrepareVectorToMMAPatterns(patterns, getUseMmaSync());
  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns)))) {
    target->emitOpError("vector to mma preparation patterns failed to apply");
    return emitDefaultDefiniteFailure(target);
  }

  IRRewriter rewriter(getContext());
  if (getUseWmma()) {
    if (failed(convertVectorToMMAOps(rewriter, target))) {
      target->emitOpError("vector to wmma patterns failed to apply");
      return emitDefaultDefiniteFailure(target);
    }
    results.push_back(target);
    return DiagnosedSilenceableFailure::success();
  }

  if (failed(convertVectorToNVVMCompatibleMMASync(rewriter, funcOp))) {
    target->emitOpError("vector to mma patterns failed to apply");
    return emitDefaultDefiniteFailure(target);
  }
  // Using TF32 for Float.
  RewritePatternSet f32ToTF32patterns(funcOp.getContext());
  nvgpu::populateMmaSyncF32ToTF32Patterns(f32ToTF32patterns,
                                          nvgpu::MmaSyncF32Lowering::TF32);
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(f32ToTF32patterns)))) {
    target->emitOpError("vector to mma F32ToTF32 patterns failed to apply");
    return emitDefaultDefiniteFailure(target);
  }
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// PromoteOperandsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform_dialect::PromoteOperandsOp::applyToOne(
    Operation *target, transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  IRRewriter rewriter(getContext());
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(target);
  SmallVector<int64_t> indices = llvm::to_vector(getIndices());
  int64_t numOperands = target->getNumOperands();

  results.push_back(target);
  bufferization::BufferizationOptions options;
  for (int64_t index : indices) {
    if ((index >= 0) && (index < numOperands)) {
      FailureOr<Value> ret = bufferization::allocateTensorForShapedValue(
          rewriter, target->getLoc(), target->getOperand(index), false, options,
          true);
      if (failed(ret)) {
        return emitDefaultDefiniteFailure(target)
               << "failed to promote operand";
      }
      target->setOperand(index, ret.value());
      results.push_back(ret.value().getDefiningOp());
    } else {
      return emitDefaultDefiniteFailure(target) << "invalid index specified";
    }
  }
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// PipelineSharedMemoryCopiesOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::PipelineSharedMemoryCopiesOp::applyToOne(
    scf::ForOp forOp, transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  IRRewriter rewriter(getContext());
  int64_t depth(getDepth());
  FailureOr<scf::ForOp> pipelinedFor = iree_compiler::pipelineSharedMemoryCopy(
      rewriter, forOp, PipeliningSchedulingStrategy::loadGlobalStage0, false,
      depth);
  if (failed(pipelinedFor)) return emitDefaultSilenceableFailure(forOp);
  results.push_back(pipelinedFor.value());
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// CreateAsyncGroupsOp.
//===---------------------------------------------------------------------===//
DiagnosedSilenceableFailure transform_dialect::CreateAsyncGroupsOp::applyToOne(
    func::FuncOp target, transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  iree_compiler::createAsyncGroups(cast<func::FuncOp>(target), getUseMmaSync());
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensionsOps.cpp.inc"
