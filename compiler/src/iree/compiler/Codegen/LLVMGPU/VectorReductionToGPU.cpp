// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

/// Emit shared local memory allocation in case it is needed when lowering the
/// warp operations.
static Value allocateGlobalSharedMemory(Location loc, OpBuilder &builder,
                                        vector::WarpExecuteOnLane0Op warpOp,
                                        Type type) {
  builder.setInsertionPoint(warpOp);
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

/// Special case to hoist hal operations that have side effect but are safe to
/// move out of the warp single lane region.
static void hoistHalBindingOps(vector::WarpExecuteOnLane0Op warpOp) {
  Block *body = warpOp.getBody();

  // Keep track of the ops we want to hoist.
  llvm::SmallSetVector<Operation *, 8> opsToMove;

  // Do not use walk here, as we do not want to go into nested regions and hoist
  // operations from there.
  for (auto &op : body->without_terminator()) {
    if (!isa<IREE::HAL::InterfaceBindingSubspanOp>(&op)) continue;
    if (llvm::any_of(op.getOperands(), [&](Value operand) {
          return !warpOp.isDefinedOutsideOfRegion(operand);
        }))
      continue;
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

struct LLVMGPUReduceToGPUPass
    : public LLVMGPUReduceToGPUBase<LLVMGPUReduceToGPUPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    // 1. Pre-process multiDimReductions.
    // TODO: Remove once MultiDimReduce is supported by distribute patterns.
    {
      RewritePatternSet patterns(ctx);
      vector::populateVectorMultiReductionLoweringPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerReduction);
      patterns.add<InsertElementToBroadcast>(ctx);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    // 2. Create the warp op and move the function body into it.
    const int warpSize = 32;
    Location loc = funcOp.getLoc();
    OpBuilder builder(funcOp);
    auto threadX = builder.create<gpu::ThreadIdOp>(loc, builder.getIndexType(),
                                                   gpu::Dimension::x);
    auto cstWarpSize = builder.create<arith::ConstantIndexOp>(loc, warpSize);
    auto laneId =
        builder.create<arith::RemUIOp>(loc, threadX.getResult(), cstWarpSize);
    auto warpOp = builder.create<vector::WarpExecuteOnLane0Op>(
        loc, TypeRange(), laneId, warpSize);
    warpOp.getWarpRegion().takeBody(funcOp.getBody());
    Block &newBlock = funcOp.getBody().emplaceBlock();
    threadX->moveBefore(&newBlock, newBlock.end());
    cstWarpSize->moveBefore(&newBlock, newBlock.end());
    laneId->moveBefore(&newBlock, newBlock.end());
    warpOp->moveBefore(&newBlock, newBlock.end());
    warpOp.getWarpRegion().getBlocks().back().back().moveBefore(&newBlock,
                                                                newBlock.end());
    builder.setInsertionPointToEnd(&warpOp.getWarpRegion().getBlocks().back());
    builder.create<vector::YieldOp>(loc);

    // 3. Hoist the scalar code outside of the warp region.
    vector::moveScalarUniformCode(warpOp);
    hoistHalBindingOps(warpOp);
    vector::moveScalarUniformCode(warpOp);

    // 4. Distribute transfer write operations.
    {
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
      RewritePatternSet patterns(ctx);
      vector::populateDistributeTransferWriteOpPatterns(patterns,
                                                        distributionFn);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    // 4. Propagate vector distribution.
    {
      RewritePatternSet patterns(ctx);
      vector::populatePropagateWarpVectorDistributionPatterns(patterns);
      vector::populateDistributeReduction(patterns, warpReduction);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    // 4. Lower the remaining WarpExecuteOnLane0 ops.
    {
      RewritePatternSet patterns(ctx);
      vector::WarpExecuteOnLane0LoweringOptions options;
      options.warpAllocationFn = allocateGlobalSharedMemory;
      options.warpSyncronizationFn = [](Location loc, OpBuilder &builder,
                                        vector::WarpExecuteOnLane0Op warpOp) {};
      vector::populateWarpExecuteOnLane0OpToScfForPattern(patterns, options);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }
  }
};

}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertVectorReductionToGPUPass() {
  return std::make_unique<LLVMGPUReduceToGPUPass>();
}

}  // namespace iree_compiler
}  // namespace mlir