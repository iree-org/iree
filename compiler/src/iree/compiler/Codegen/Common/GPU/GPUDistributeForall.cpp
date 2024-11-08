// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <functional>
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUDISTRIBUTEFORALLPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

struct GPUDistributeForallPass final
    : impl::GPUDistributeForallPassBase<GPUDistributeForallPass> {
  void runOnOperation() override;
};
} // namespace

LogicalResult resolveGPUMappedForallOp(RewriterBase &rewriter,
                                       scf::ForallOp forallOp,
                                       Value linearThreadId,
                                       int64_t flatWorkgroupSize,
                                       int64_t subgroupSize) {

  // Skip forall ops without mappings.
  if (!forallOp.getMapping()) {
    return success();
  }

  ArrayAttr mapping = forallOp.getMappingAttr();
  bool hasThreadMapping =
      llvm::all_of(mapping, llvm::IsaPred<gpu::GPUThreadMappingAttr>);
  bool hasWarpMapping =
      llvm::all_of(mapping, llvm::IsaPred<gpu::GPUWarpMappingAttr>);

  // Skip forall ops that are not mapped to GPU ids.
  if (!hasThreadMapping && !hasWarpMapping) {
    return success();
  }

  if (forallOp->getNumResults() != 0) {
    forallOp.emitOpError("Cannot distribute scf.forall op on tensors.");
    return failure();
  }

  if (!isDescendingRelativeMappingIndices(mapping.getValue())) {
    forallOp.emitOpError("Cannot distribute forall op with non-descending "
                         "relative iterator mapping");
    return failure();
  }

  if (!llvm::all_of(mapping, [](Attribute attr) {
        return cast<DeviceMappingAttrInterface>(attr).isLinearMapping();
      })) {
    forallOp.emitOpError("unimplemented: resolution of scf.forall ops without "
                         "linear id mappings.");
    return failure();
  }

  if (!forallOp.isNormalized()) {
    forallOp.emitOpError("scf.forall op must be normalized for distribution.");
    return failure();
  }

  MLIRContext *context = rewriter.getContext();
  Location loc = forallOp.getLoc();
  AffineExpr d0, d1;
  bindDims(context, d0, d1);

  // Divide the thread ID by the subgroup size if this loop is mapped to
  // subgroups.
  assert(!(hasThreadMapping && hasWarpMapping));
  Value flatId = linearThreadId;
  if (hasWarpMapping) {
    flatId = rewriter
                 .create<affine::AffineDelinearizeIndexOp>(
                     loc, flatId,
                     ArrayRef<int64_t>{flatWorkgroupSize / subgroupSize,
                                       subgroupSize})
                 .getResult(0);
  }

  SmallVector<OpFoldResult> delinSizes;
  OpFoldResult producerCount = rewriter.getIndexAttr(1);
  for (auto workerCount : forallOp.getMixedUpperBound()) {
    delinSizes.push_back(workerCount);
    producerCount = affine::makeComposedFoldedAffineApply(
        rewriter, loc, d0 * d1, {producerCount, workerCount});
  }

  // If the total number of producers doesn't evenly divide into
  int64_t flatTotalNumWorkers =
      hasWarpMapping ? flatWorkgroupSize / subgroupSize : flatWorkgroupSize;
  OpFoldResult newLoopTripCount = affine::makeComposedFoldedAffineApply(
      rewriter, loc, d0.floorDiv(flatTotalNumWorkers), producerCount);
  OpFoldResult remainingLanes = affine::makeComposedFoldedAffineApply(
      rewriter, loc, d0 % flatTotalNumWorkers, {producerCount});

  // If the loop isn't guaranteed to perfectly tile onto the workers,
  // we will run one more iteration of the loop on the workitems where it
  // needs to execute.
  std::optional<int64_t> remainingLanesCount =
      getConstantIntValue(remainingLanes);
  bool hasPostLoopTail =
      !remainingLanesCount || remainingLanesCount.value() != 0;
  OpFoldResult maxIteration =
      hasPostLoopTail
          ? affine::makeComposedFoldedAffineApply(
                rewriter, loc, d0.ceilDiv(flatTotalNumWorkers), {producerCount})
          : newLoopTripCount;

  // Step 3. Create the `scf.for` loop for the loop.
  Value lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value ub = getValueOrCreateConstantIndexOp(rewriter, loc, newLoopTripCount);
  Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  auto forLoop = rewriter.create<scf::ForOp>(loc, lb, ub, step, ValueRange{});
  Block *loopBody = forLoop.getBody();

  // Get the replacement IDs for the forall iterator ids.
  rewriter.setInsertionPointToStart(loopBody);
  Value newFlatProducerId = rewriter.create<affine::AffineLinearizeIndexOp>(
      loc, ValueRange{forLoop.getInductionVar(), flatId},
      ArrayRef<OpFoldResult>{maxIteration,
                             rewriter.getIndexAttr(flatTotalNumWorkers)},
      /*disjoint=*/true);

  // We require a descending relative mapping, so we can reuse the upper bound
  // sizes directly.
  auto delinearize = rewriter.create<affine::AffineDelinearizeIndexOp>(
      loc, newFlatProducerId, delinSizes);

  SmallVector<Value> newBlockArgs = delinearize.getResults();

  // Step 4. Inline the region of the forall op.
  Operation *forallTerminator = forallOp.getBody()->getTerminator();
  rewriter.inlineBlockBefore(forallOp.getBody(), loopBody->getTerminator(),
                             newBlockArgs);
  rewriter.eraseOp(forallTerminator);
  rewriter.eraseOp(forallOp);

  // Step 5. Create the post-loop code that only executes on some workitems.
  if (hasPostLoopTail) {
    rewriter.setInsertionPointAfter(forLoop);
    IRMapping cloneMap;
    Value willExecuteTail = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, flatId,
        getValueOrCreateConstantIndexOp(rewriter, loc, remainingLanes));
    auto tailIfOp = rewriter.create<scf::IfOp>(
        loc, TypeRange{}, willExecuteTail, /*addThenBlock=*/false,
        /*addElseBlock=*/false);
    cloneMap.map(forLoop.getInductionVar(), ub);
    // We're relying on the fact that `scf.for` and `scf.if` share the same
    // terminator.
    forLoop.getRegion().cloneInto(&tailIfOp.getThenRegion(), cloneMap);
  }
  return success();
}

void GPUDistributeForallPass::runOnOperation() {
  auto funcOp = getOperation();

  // First map all lane level forall loops to lanes.
  IRRewriter rewriter(funcOp->getContext());
  IREE::GPU::mapLaneForalls(rewriter, funcOp, /*insertBarrier=*/false);

  SmallVector<scf::ForallOp> forallOps;
  funcOp.walk([&](scf::ForallOp op) { forallOps.push_back(op); });
  // Early exit if no more forall ops to distribute.
  if (forallOps.empty()) {
    return;
  }

  std::optional<SmallVector<int64_t>> maybeWorkgroupSize =
      getWorkgroupSize(funcOp);
  if (!maybeWorkgroupSize) {
    funcOp.emitOpError(
        "unimplemented: Distribution with dynamic workgroup size.");
    return signalPassFailure();
  }
  SmallVector<int64_t> workgroupSize = maybeWorkgroupSize.value();

  std::optional<int64_t> maybeSubgroupSize = getSubgroupSize(funcOp);
  if (!maybeSubgroupSize) {
    funcOp.emitOpError(
        "unimplemented: Distribution with dynamic subgroup size.");
    return signalPassFailure();
  }

  int64_t flatWorkgroupSize =
      std::accumulate(workgroupSize.begin(), workgroupSize.end(), 1,
                      std::multiplies<int64_t>());
  int64_t subgroupSize = *maybeSubgroupSize;

  if (flatWorkgroupSize % subgroupSize != 0 &&
      llvm::any_of(forallOps, [](scf::ForallOp forall) {
        return forallOpHasMappingType<gpu::GPUWarpMappingAttr>(forall);
      })) {
    funcOp.emitOpError("Invalid workgroup size is not divisible by subgroup "
                       "size for warp distribution.");
    return signalPassFailure();
  }

  rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());
  SmallVector<Value> threadGrid = {rewriter.createOrFold<gpu::ThreadIdOp>(
                                       funcOp.getLoc(), gpu::Dimension::z),
                                   rewriter.createOrFold<gpu::ThreadIdOp>(
                                       funcOp.getLoc(), gpu::Dimension::y),
                                   rewriter.createOrFold<gpu::ThreadIdOp>(
                                       funcOp.getLoc(), gpu::Dimension::x)};
  SmallVector<int64_t> threadGridBasis = {workgroupSize[2], workgroupSize[1],
                                          workgroupSize[0]};

  Value linearThreadIdVal = rewriter.create<affine::AffineLinearizeIndexOp>(
      funcOp.getLoc(), threadGrid, threadGridBasis, /*disjoint=*/true);
  for (auto forall : forallOps) {
    rewriter.setInsertionPoint(forall);
    if (failed(resolveGPUMappedForallOp(rewriter, forall, linearThreadIdVal,
                                        flatWorkgroupSize, subgroupSize))) {
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler
