// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-vector-reduction-to-gpu"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_VECTORREDUCTIONTOGPUPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
static void debugPrint(Operation *op, const char *message) {
  LLVM_DEBUG({
    llvm::dbgs() << "//--- " << message << " ---//\n";
    op->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
}

/// Emit shared local memory allocation in case it is needed when lowering the
/// warp operations.
static Value allocateGlobalSharedMemory(Location loc, OpBuilder &builder,
                                        gpu::WarpExecuteOnLane0Op warpOp,
                                        Type type) {
  MemRefType memrefType;
  auto addressSpaceAttr = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  if (auto vectorType = llvm::dyn_cast<VectorType>(type)) {
    memrefType =
        MemRefType::get(vectorType.getShape(), vectorType.getElementType(),
                        MemRefLayoutAttrInterface{}, addressSpaceAttr);
  } else {
    memrefType = MemRefType::get({1}, type, MemRefLayoutAttrInterface{},
                                 addressSpaceAttr);
  }
  return builder.create<memref::AllocOp>(loc, memrefType);
}

/// Returns true if the given op is a memref.load from a uniform buffer or
/// read-only storage buffer.
static bool isUniformLoad(Operation *op) {
  using namespace IREE::HAL;

  auto loadOp = dyn_cast<memref::LoadOp>(op);
  if (!loadOp)
    return false;
  if (!hasGlobalMemoryAddressSpace(loadOp.getMemRefType()))
    return false;
  auto space = loadOp.getMemRefType().getMemorySpace();
  auto descTypeAttr = llvm::dyn_cast_if_present<DescriptorTypeAttr>(space);
  if (descTypeAttr && descTypeAttr.getValue() == DescriptorType::UniformBuffer)
    return true;

  auto subspan = loadOp.getMemRef().getDefiningOp<InterfaceBindingSubspanOp>();
  if (auto fatBufferCast =
          loadOp.getMemRef().getDefiningOp<amdgpu::FatRawBufferCastOp>()) {
    subspan =
        fatBufferCast.getSource().getDefiningOp<InterfaceBindingSubspanOp>();
  }
  if (!subspan)
    return false;

  descTypeAttr = dyn_cast_if_present<DescriptorTypeAttr>(
      cast<MemRefType>(subspan.getResult().getType()).getMemorySpace());
  if (descTypeAttr && descTypeAttr.getValue() == DescriptorType::UniformBuffer)
    return true;
  if (auto flags = subspan.getDescriptorFlags()) {
    if (bitEnumContainsAll(*flags, IREE::HAL::DescriptorFlags::ReadOnly))
      return true;
  }
  return false;
}

/// Hoist uniform operations as well as special hal operations that have side
/// effect but are safe to move out of the warp single lane region.
static void moveScalarAndBindingUniformCode(gpu::WarpExecuteOnLane0Op warpOp) {
  /// Hoist ops without side effect as well as special binding ops.
  auto canBeHoisted = [](Operation *op,
                         function_ref<bool(Value)> definedOutside) {
    if (op->getNumRegions() != 0)
      return false;
    if (!llvm::all_of(op->getOperands(), definedOutside))
      return false;
    if (isMemoryEffectFree(op))
      return true;

    if (isa<IREE::HAL::InterfaceBindingSubspanOp,
            IREE::HAL::InterfaceConstantLoadOp, memref::AssumeAlignmentOp>(op))
      return true;
    if (isUniformLoad(op))
      return true;
    // Shared memory is already scoped to the workgroup and can safely be
    // hoisted out of the the warp op.
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      if (hasSharedMemoryAddressSpace(allocOp.getType())) {
        return true;
      }
    }

    return false;
  };
  Block *body = warpOp.getBody();

  // Keep track of the ops we want to hoist.
  llvm::SmallSetVector<Operation *, 8> opsToMove;

  // Helper to check if a value is or will be defined outside of the region.
  auto isDefinedOutsideOfBody = [&](Value value) {
    auto *definingOp = value.getDefiningOp();
    return (definingOp && opsToMove.count(definingOp)) ||
           warpOp.isDefinedOutsideOfRegion(value);
  };

  // Do not use walk here, as we do not want to go into nested regions and hoist
  // operations from there.
  for (auto &op : body->without_terminator()) {
    bool hasVectorResult = llvm::any_of(op.getResults(), [](Value result) {
      return llvm::isa<VectorType>(result.getType());
    });
    if ((!hasVectorResult || isUniformLoad(&op)) &&
        canBeHoisted(&op, isDefinedOutsideOfBody)) {
      opsToMove.insert(&op);
    }
  }

  // Move all the ops marked as uniform outside of the region.
  for (Operation *op : opsToMove)
    op->moveBefore(warpOp);
}

/// Pattern to convert single element vector.insert to broadcast, this is a
/// workaround until MultiDimReduction distribution is supported.
struct InsertToBroadcast final : OpRewritePattern<vector::InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertOp insertOp,
                                PatternRewriter &rewriter) const override {
    if (insertOp.getDestVectorType().getNumElements() != 1)
      return failure();
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        insertOp, insertOp.getDestVectorType(), insertOp.getValueToStore());
    return success();
  }
};

/// Pattern to sink `gpu.barrier` ops out of a `warp_execute_on_lane_0` op.
struct WarpOpBarrier final : OpRewritePattern<gpu::WarpExecuteOnLane0Op> {
  using OpRewritePattern<gpu::WarpExecuteOnLane0Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    auto yield = cast<gpu::YieldOp>(
        warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
    Operation *lastNode = yield->getPrevNode();
    auto barrierOp = dyn_cast_or_null<gpu::BarrierOp>(lastNode);
    if (!barrierOp)
      return failure();

    rewriter.setInsertionPointAfter(warpOp);
    (void)rewriter.create<gpu::BarrierOp>(barrierOp.getLoc());
    rewriter.eraseOp(barrierOp);
    return success();
  }
};

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

struct VectorReductionToGPUPass final
    : impl::VectorReductionToGPUPassBase<VectorReductionToGPUPass> {
  VectorReductionToGPUPass(bool expandSubgroupReduction)
      : expandSubgroupReduction(expandSubgroupReduction) {}

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    debugPrint(funcOp, "after step #0: before vector reduction to gpu");

    // 1. Pre-process multiDimReductions.
    // TODO: Remove once MultiDimReduce is supported by distribute patterns.
    {
      RewritePatternSet patterns(ctx);
      vector::populateVectorMultiReductionLoweringPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerReduction);
      // Add clean up patterns after lowering of multidimreduce lowering.
      patterns.add<InsertToBroadcast>(ctx);
      vector::ShapeCastOp::getCanonicalizationPatterns(patterns, ctx);
      vector::BroadcastOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
      (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }

    debugPrint(funcOp, "after step #1: preprocessing reduction ops");

    std::optional<SmallVector<int64_t>> maybeWorkgroupSize =
        getWorkgroupSize(funcOp);
    if (!maybeWorkgroupSize) {
      funcOp->emitOpError(
          "expected workgroup size to be set as part of `translation_info`");
      return signalPassFailure();
    }
    SmallVector<int64_t> &workgroupSize = maybeWorkgroupSize.value();
    assert(workgroupSize[1] == 1 && workgroupSize[2] == 1);
    // 2. Create the warp op and move the function body into it.
    const int groupSize = workgroupSize[0];
    Location loc = funcOp.getLoc();
    OpBuilder builder(funcOp);
    auto threadX = builder.create<gpu::ThreadIdOp>(loc, builder.getIndexType(),
                                                   gpu::Dimension::x);
    auto cstGroupSize = builder.create<arith::ConstantIndexOp>(loc, groupSize);
    auto warpOp = builder.create<gpu::WarpExecuteOnLane0Op>(
        loc, TypeRange(), threadX.getResult(), groupSize);
    warpOp.getWarpRegion().takeBody(funcOp.getFunctionBody());
    Block &newBlock = funcOp.getFunctionBody().emplaceBlock();
    threadX->moveBefore(&newBlock, newBlock.end());
    cstGroupSize->moveBefore(&newBlock, newBlock.end());
    warpOp->moveBefore(&newBlock, newBlock.end());
    warpOp.getWarpRegion().getBlocks().back().back().moveBefore(&newBlock,
                                                                newBlock.end());
    builder.setInsertionPointToEnd(&warpOp.getWarpRegion().getBlocks().back());
    builder.create<gpu::YieldOp>(loc);

    debugPrint(funcOp, "after step #2: wrapping code with the warp execute op");

    // 3. Hoist the scalar code outside of the warp region.
    moveScalarAndBindingUniformCode(warpOp);

    debugPrint(funcOp, "after step #3: hosting uniform code");

    // 4. Distribute transfer write operations and propagate vector
    // distribution.
    {
      std::optional<int> subgroupSize = getGPUSubgroupSize(funcOp);
      if (!subgroupSize) {
        funcOp->emitOpError("missing subgroup size");
        return signalPassFailure();
      }
      auto groupReductionFn = [=](Location loc, OpBuilder &builder, Value input,
                                  vector::CombiningKind kind,
                                  uint32_t size) -> Value {
        return emitGPUGroupReduction(loc, builder, input, kind, size,
                                     *subgroupSize, expandSubgroupReduction);
      };
      auto distributionFn = [](Value val) {
        auto vecType = llvm::dyn_cast<VectorType>(val.getType());
        if (!vecType)
          return AffineMap::get(val.getContext());
        // Create an identity dim map of rank |vecRank|. This greedily divides
        // threads along the outermost vector dimensions to the innermost ones.
        int64_t vecRank = vecType.getRank();
        OpBuilder builder(val.getContext());
        return builder.getMultiDimIdentityMap(vecRank);
      };

      RewritePatternSet patterns(ctx);
      vector::populatePropagateWarpVectorDistributionPatterns(
          patterns, distributionFn, simpleWarpShuffleFunction);
      vector::populateDistributeReduction(patterns, groupReductionFn);

      // We don't want to sink large transfer writes to a single lane -- pick a
      // conservative value based on the group size.
      unsigned maxWriteElementsToExtract = std::max(groupSize / 4, 1);
      vector::populateDistributeTransferWriteOpPatterns(
          patterns, distributionFn, maxWriteElementsToExtract);
      patterns.add<WarpOpBarrier>(patterns.getContext(), 3);
      vector::ReductionOp::getCanonicalizationPatterns(patterns, ctx);
      (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }

    debugPrint(funcOp, "after step #4: propagating distribution");

    // 5. Lower the remaining WarpExecuteOnLane0 ops.
    {
      RewritePatternSet patterns(ctx);
      vector::WarpExecuteOnLane0LoweringOptions options;
      options.warpAllocationFn = allocateGlobalSharedMemory;
      options.warpSyncronizationFn = [](Location loc, OpBuilder &builder,
                                        gpu::WarpExecuteOnLane0Op warpOp) {
        builder.create<gpu::BarrierOp>(loc);
      };
      vector::populateWarpExecuteOnLane0OpToScfForPattern(patterns, options);
      (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }

    debugPrint(funcOp, "after step #5: lowering remaining ops");
  }

private:
  bool expandSubgroupReduction;
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createConvertVectorReductionToGPUPass(bool expandSubgroupReduction) {
  return std::make_unique<VectorReductionToGPUPass>(expandSubgroupReduction);
}

} // namespace mlir::iree_compiler
