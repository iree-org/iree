// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-reduction-distribution"

namespace mlir {
namespace iree_compiler {

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
                           vector::CombiningKind kind, uint32_t warpSize,
                           uint32_t numLaneToReduce) {
  Value laneVal = input;
  assert(llvm::isPowerOf2_32(numLaneToReduce));
  // Parallel reduction using butterfly shuffles.
  for (uint64_t i = 1; i < numLaneToReduce; i <<= 1) {
    Value shuffled = builder
                         .create<gpu::ShuffleOp>(loc, laneVal, i,
                                                 /*width=*/warpSize,
                                                 /*mode=*/gpu::ShuffleMode::XOR)
                         .getShuffleResult();
    laneVal = makeArithReduction(builder, loc, kind, laneVal, shuffled);
  }
  // Broadcast the result to all the lanes.
  if (warpSize != numLaneToReduce) {
    laneVal = builder
                  .create<gpu::ShuffleOp>(loc, laneVal, 0,
                                          /*width=*/warpSize,
                                          /*mode=*/gpu::ShuffleMode::IDX)
                  .getShuffleResult();
  }
  return laneVal;
}

// List of identity elements by operation.
// https://en.wikipedia.org/wiki/Identity_element
static Attribute getCombiningKindIdentity(OpBuilder &builder,
                                          vector::CombiningKind combiningKind,
                                          Type type) {
  switch (combiningKind) {
    case vector::CombiningKind::ADD:
      return builder.getZeroAttr(type);
    case vector::CombiningKind::MUL: {
      if (type.isIntOrIndex()) {
        return builder.getIntegerAttr(type, 1);
      }
      return builder.getFloatAttr(type, 1);
    }
    case vector::CombiningKind::MINUI:
    case vector::CombiningKind::MINSI:
      return builder.getIntegerAttr(type, std::numeric_limits<int64_t>::max());
    case vector::CombiningKind::MAXUI:
    case vector::CombiningKind::MAXSI:
      return builder.getIntegerAttr(type, std::numeric_limits<int64_t>::min());
    case vector::CombiningKind::AND:
      return builder.getIntegerAttr(type, 1);
    case vector::CombiningKind::OR:
    case vector::CombiningKind::XOR:
      return builder.getZeroAttr(type);
    case vector::CombiningKind::MINF: {
      auto posInfApFloat = APFloat::getInf(
          type.cast<FloatType>().getFloatSemantics(), /*Negative=*/false);
      return builder.getFloatAttr(type, posInfApFloat);
    }
    case vector::CombiningKind::MAXF: {
      auto negInfApFloat = APFloat::getInf(
          type.cast<FloatType>().getFloatSemantics(), /*Negative=*/true);
      return builder.getFloatAttr(type, negInfApFloat);
    }
  }
  return Attribute();
}

/// Emit reduction across a group for a given input.
static Value groupReduction(Location loc, OpBuilder &builder, Value input,
                            vector::CombiningKind kind, uint32_t size,
                            const int warpSize) {
  assert(
      size % warpSize == 0 &&
      "Group reduction only support for sizes aligned on warp size for now.");
  // First reduce on a single thread to get per lane reduction value.
  Value laneVal = builder.create<vector::ReductionOp>(loc, kind, input);
  laneVal = warpReduction(loc, builder, laneVal, kind, warpSize, warpSize);
  // if we have more than one warp, reduce across warps.
  if (size > warpSize) {
    uint32_t numWarp = size / warpSize;
    assert(numWarp <= warpSize &&
           "Only support 1 level, need to implement recursive/loop for this "
           "case.");
    MemRefType memrefType =
        MemRefType::get(numWarp, laneVal.getType(), {},
                        gpu::GPUDialect::getWorkgroupAddressSpace());
    Value alloc = builder.create<memref::AllocOp>(loc, memrefType);
    Value threadX = builder.create<gpu::ThreadIdOp>(loc, builder.getIndexType(),
                                                    gpu::Dimension::x);
    Value cstWarpSize = builder.create<arith::ConstantIndexOp>(loc, warpSize);
    Value warpId = builder.create<arith::DivUIOp>(loc, threadX, cstWarpSize);
    Value laneId = builder.create<arith::RemUIOp>(loc, threadX, cstWarpSize);
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value lane0 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                laneId, zero);
    // Store the reduction for each warp.
    SmallVector<Value> indices = {warpId};
    builder.create<scf::IfOp>(loc, lane0, [&](OpBuilder &b, Location l) {
      b.create<memref::StoreOp>(l, laneVal, alloc, indices);
      b.create<scf::YieldOp>(l, llvm::None);
    });
    builder.create<gpu::BarrierOp>(loc);
    // Further reduce the outputs from each warps with a single warp reduce.
    Value loadVal = builder.create<memref::LoadOp>(loc, alloc, laneId);
    Value cstNumWarp = builder.create<arith::ConstantIndexOp>(loc, numWarp);
    if (!llvm::isPowerOf2_32(numWarp)) {
      // Pad with identity element if numel < warpSize for valid warp reduction.
      Value useIdentityElement = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge, laneId, cstNumWarp);
      numWarp = llvm::PowerOf2Ceil(numWarp);
      Attribute identityAttr =
          getCombiningKindIdentity(builder, kind, loadVal.getType());
      assert(identityAttr && "Unknown identity value for the reduction");
      Value identity = builder.create<arith::ConstantOp>(loc, identityAttr);
      loadVal = builder.create<arith::SelectOp>(loc, useIdentityElement,
                                                identity, loadVal);
    }
    laneVal = warpReduction(loc, builder, loadVal, kind, warpSize, numWarp);
  }
  return laneVal;
}

/// Hoist uniform operations as well as special hal operations that have side
/// effect but are safe to move out of the warp single lane region.
static void moveScalarAndBindingUniformCode(
    vector::WarpExecuteOnLane0Op warpOp) {
  /// Hoist ops without side effect as well as special binding ops.
  auto canBeHoisted = [](Operation *op,
                         function_ref<bool(Value)> definedOutside) {
    return llvm::all_of(op->getOperands(), definedOutside) &&
           (isMemoryEffectFree(op) ||
            isa<IREE::HAL::InterfaceBindingSubspanOp,
                IREE::HAL::InterfaceConstantLoadOp, memref::AssumeAlignmentOp>(
                op)) &&
           op->getNumRegions() == 0;
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
      return result.getType().isa<VectorType>();
    });
    if (!hasVectorResult && canBeHoisted(&op, isDefinedOutsideOfBody))
      opsToMove.insert(&op);
  }

  // Move all the ops marked as uniform outside of the region.
  for (Operation *op : opsToMove) op->moveBefore(warpOp);
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

class VectorReduceToGPUPass
    : public VectorReduceToGPUBase<VectorReduceToGPUPass> {
 public:
  explicit VectorReduceToGPUPass(std::function<int(func::FuncOp)> getWarpSize)
      : getWarpSize(getWarpSize) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, memref::MemRefDialect, gpu::GPUDialect,
                    AffineDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    // 1. Pre-process multiDimReductions.
    // TODO: Remove once MultiDimReduce is supported by distribute patterns.
    {
      RewritePatternSet patterns(ctx);
      vector::populateVectorMultiReductionLoweringPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerReduction);
      // Add clean up patterns after lowering of multidimreduce lowering.
      patterns.add<InsertElementToBroadcast>(ctx);
      vector::ShapeCastOp::getCanonicalizationPatterns(patterns, ctx);
      vector::BroadcastOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs()
          << "\n--- After Step 1: Preprocessing of reduction ops ---\n";
      funcOp.dump();
      llvm::dbgs() << "\n\n";
    });

    auto workgroupSize = llvm::to_vector<4>(llvm::map_range(
        getEntryPoint(funcOp)->getWorkgroupSize().value(),
        [&](Attribute attr) { return attr.cast<IntegerAttr>().getInt(); }));
    assert(workgroupSize[1] == 1 && workgroupSize[2] == 1);
    // 2. Create the warp op and move the function body into it.
    const int groupSize = workgroupSize[0];
    Location loc = funcOp.getLoc();
    OpBuilder builder(funcOp);
    auto threadX = builder.create<gpu::ThreadIdOp>(loc, builder.getIndexType(),
                                                   gpu::Dimension::x);
    auto cstGroupSize = builder.create<arith::ConstantIndexOp>(loc, groupSize);
    auto warpOp = builder.create<vector::WarpExecuteOnLane0Op>(
        loc, TypeRange(), threadX.getResult(), groupSize);
    warpOp.getWarpRegion().takeBody(funcOp.getFunctionBody());
    Block &newBlock = funcOp.getFunctionBody().emplaceBlock();
    threadX->moveBefore(&newBlock, newBlock.end());
    cstGroupSize->moveBefore(&newBlock, newBlock.end());
    warpOp->moveBefore(&newBlock, newBlock.end());
    warpOp.getWarpRegion().getBlocks().back().back().moveBefore(&newBlock,
                                                                newBlock.end());
    builder.setInsertionPointToEnd(&warpOp.getWarpRegion().getBlocks().back());
    builder.create<vector::YieldOp>(loc);

    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "\n--- After Step 2: Adding the distribution op ---\n";
      funcOp.dump();
      llvm::dbgs() << "\n\n";
    });

    // 3. Hoist the scalar code outside of the warp region.
    moveScalarAndBindingUniformCode(warpOp);

    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "\n--- After Step 3: Hoist uniform code ---\n";
      funcOp.dump();
      llvm::dbgs() << "\n\n";
    });

    // 4. Distribute transfer write operations and propagate vector
    // distribution.
    {
      auto getWarpSize = this->getWarpSize ? this->getWarpSize
                                           : [](func::FuncOp) { return 32; };
      auto groupReductionFn = [&](Location loc, OpBuilder &builder, Value input,
                                  vector::CombiningKind kind, uint32_t size) {
        return groupReduction(loc, builder, input, kind, size,
                              getWarpSize(funcOp));
      };
      auto distributionFn = [](Value val) {
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
      };
      RewritePatternSet patterns(ctx);
      vector::populatePropagateWarpVectorDistributionPatterns(
          patterns, distributionFn, simpleWarpShuffleFunction);
      vector::populateDistributeReduction(patterns, groupReductionFn);
      vector::populateDistributeTransferWriteOpPatterns(patterns,
                                                        distributionFn);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "\n--- After Step 4: Propagate distribution ---\n";
      funcOp.dump();
      llvm::dbgs() << "\n\n";
    });

    // 5. Lower the remaining WarpExecuteOnLane0 ops.
    {
      RewritePatternSet patterns(ctx);
      vector::WarpExecuteOnLane0LoweringOptions options;
      options.warpAllocationFn = allocateGlobalSharedMemory;
      options.warpSyncronizationFn = [](Location loc, OpBuilder &builder,
                                        vector::WarpExecuteOnLane0Op warpOp) {
        builder.create<gpu::BarrierOp>(loc);
      };
      vector::populateWarpExecuteOnLane0OpToScfForPattern(patterns, options);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "\n--- After Step 5: Lower remaining ops ---\n";
      funcOp.dump();
      llvm::dbgs() << "\n\n";
    });
  }

 private:
  std::function<int(func::FuncOp)> getWarpSize;
};

}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertVectorReductionToGPUPass(
    std::function<int(func::FuncOp)> getWarpSize) {
  return std::make_unique<VectorReduceToGPUPass>(getWarpSize);
}

}  // namespace iree_compiler
}  // namespace mlir
