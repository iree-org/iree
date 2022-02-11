// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>

#include "iree/compiler/Codegen/LLVMGPU/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

namespace {

static Value createElementwiseOp(OpBuilder& b, Location loc, Value lhs,
                                 Value rhs, vector::CombiningKind kind,
                                 bool isInteger) {
  switch (kind) {
    case vector::CombiningKind::ADD:
      if (isInteger)
        return b.create<arith::AddIOp>(loc, lhs, rhs);
      else
        return b.create<arith::AddFOp>(loc, lhs, rhs);
    case vector::CombiningKind::MAXF:
      return b.create<arith::MaxFOp>(loc, lhs, rhs);
    case vector::CombiningKind::MINF:
      return b.create<arith::MinFOp>(loc, lhs, rhs);
    case vector::CombiningKind::MUL:
      if (isInteger)
        return b.create<arith::MulIOp>(loc, lhs, rhs);
      else
        return b.create<arith::MulFOp>(loc, lhs, rhs);
    default:
      break;
  }
  return nullptr;
}

static Value warpScan(OpBuilder& b, Location loc, Value val,
                      vector::CombiningKind kind, bool isInteger, Type elemType) {
  std::array<Type, 2> shuffleType = {val.getType(), b.getI1Type()};
  Value activeWidth =
      b.create<arith::ConstantIntOp>(loc, 32, b.getI32Type());
  Value value = val;
  Value zero = b.create<arith::ConstantOp>(loc, elemType, b.getZeroAttr(elemType));
  for (int i = 1; i < kWarpSize; i <<= 1) {
    Value offset = b.create<arith::ConstantIntOp>(loc, i, b.getI32Type());
    auto shuffleOp = b.create<gpu::ShuffleOp>(loc, shuffleType, value, offset,
                                              activeWidth, gpu::ShuffleMode::UP);
    auto selectOp = b.create<arith::SelectOp>(loc, shuffleOp->getResult(1), 
        shuffleOp->getResult(0), zero);
    value = createElementwiseOp(b, loc, value, selectOp->getResult(0), kind, isInteger);
  }
  return value;
}

static Value broadcastToAllLanes(OpBuilder& b, Location loc, Value val) {
  std::array<Type, 2> shuffleType = {val.getType(), b.getI1Type()};
  Value activeWidth =
      b.create<arith::ConstantIntOp>(loc, 32, b.getI32Type());
  Value zero = b.create<arith::ConstantIntOp>(loc, 31, b.getI32Type());
  return b
      .create<gpu::ShuffleOp>(loc, shuffleType, val, zero, activeWidth, gpu::ShuffleMode::IDX)
      .getResult(0);
}

struct ConvertScanToGPU final
    : public OpRewritePattern<vector::ScanOp> {
  using OpRewritePattern<vector::ScanOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ScanOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getSourceType().getNumElements() != kWarpSize)
      return failure();
    auto funcOp = op->getParentOfType<FuncOp>();
    if (!funcOp) return failure();

    rewriter.setInsertionPoint(&funcOp.front(), funcOp.front().begin());
    mlir::Value laneId = rewriter.create<mlir::gpu::ThreadIdOp>(
        op.getLoc(), rewriter.getIndexType(), gpu::Dimension::x);
    rewriter.setInsertionPoint(op);
    mlir::AffineExpr d0 = rewriter.getAffineDimExpr(0);
    laneId = mlir::makeComposedAffineApply(
        rewriter, op.getLoc(), d0 % rewriter.getAffineConstantExpr(kWarpSize),
        {laneId});

    auto vecType = VectorType::get(
        SmallVector<int64_t>(op.getSourceType().getRank(), 1),
        op.getSourceType().getElementType());
    // Distribute the value on the warp lanes.
    Value distributedVal = rewriter.create<vector::ExtractMapOp>(
        op.getLoc(), vecType, op.source(), laneId);
    distributedVal = rewriter.create<vector::ExtractOp>(
        op.getLoc(), distributedVal,
        SmallVector<int64_t>(vecType.getRank(), 0));

    auto initialVecType = VectorType::get(
        SmallVector<int64_t>(op.getInitialValueType().getRank(), 1),
        op.getInitialValueType().getElementType());
    Value distributedAccVal;
    int64_t initialVecRank = initialVecType.getRank();
    if (initialVecRank == 0) {
      distributedAccVal = rewriter.create<vector::ExtractElementOp>(op.getLoc(),
          op.initial_value());
    } else if (initialVecRank == 1) {
      Value zero = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);
      distributedAccVal = rewriter.create<vector::ExtractElementOp>(op.getLoc(),
          op.initial_value(), zero);
    } else {
      distributedAccVal = rewriter.create<vector::ExtractMapOp>(
          op.getLoc(), initialVecType, op.initial_value(), laneId);
      distributedAccVal = rewriter.create<vector::ExtractOp>(
          op.getLoc(), distributedVal,
          SmallVector<int64_t>(initialVecType.getRank(), 0));
    }

    bool isInteger = vecType.getElementType().isa<IntegerType>();

    Value v = warpScan(rewriter, op.getLoc(), distributedVal, op.kind(),
                       isInteger, vecType.getElementType());
    v = createElementwiseOp(rewriter, op.getLoc(), distributedAccVal, 
        v, op.kind(), isInteger);

    SmallVector<int64_t> broadcastShape(vecType.getRank(), 1);
    Value broadcastVec = rewriter.create<vector::BroadcastOp>(op.getLoc(),
        VectorType::get(broadcastShape, op.getSourceType().getElementType()), v);
    Value v1 = rewriter.create<vector::InsertMapOp>(op.getLoc(),
        broadcastVec, op.dest(), laneId);

    Value v2 = broadcastToAllLanes(rewriter, op.getLoc(), v);
    if (initialVecRank <= 1) {
      SmallVector<int64_t> shape;
      if (initialVecRank == 1) {
        shape.push_back(1);
      }
      v2 = rewriter.create<vector::BroadcastOp>(op.getLoc(),
          VectorType::get(shape, op.getSourceType().getElementType()), v2);
    }

    rewriter.replaceOp(op, {v1, v2});
    return success();
  }
};

/// Converts extract_map(broadcast) to broadcast(extract_map).
/// Example:
/// ```
/// %b = vector.broadcast %a : vector<32xf32> to vector<2x32xf32>
/// %e = vector.extract_map %b[%id0, %id1] : vector<2x32xf32> to vector<1x4xf32>
/// ```
/// to:
/// ```
/// %e = vector.extract_map %a[%id1] : vector<32xf32> to vector<4xf32>
/// %b = vector.broadcast %e : vector<4xf32> to vector<1x4xf32>
/// ```
struct ExtractMapBroadcastPattern final
    : public OpRewritePattern<vector::ExtractMapOp> {
  using OpRewritePattern<vector::ExtractMapOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractMapOp op,
                                PatternRewriter &rewriter) const override {
    auto broadcast = op.vector().getDefiningOp<vector::BroadcastOp>();
    if (!broadcast)
      return failure();
    auto srcVecType = broadcast.getSourceType().dyn_cast<VectorType>();
    if (!srcVecType) {
      // Special case if the source is a scalar we don't need to distribute
      // anything we can just broadcast the original source.
      rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, op.getResultType(),
                                                       broadcast.source());
      return success();
    }
    VectorType dstVecType = broadcast.getVectorType();
    SmallVector<int64_t> extractShape(srcVecType.getShape().begin(),
                                      srcVecType.getShape().end());
    SmallVector<Value> ids;
    int64_t rankDiff = dstVecType.getRank() - srcVecType.getRank();
    for (unsigned i : llvm::seq(unsigned(0), op.map().getNumResults())) {
      unsigned dim = op.map().getDimPosition(i);
      // If the dimension was broadcasted we don't need to distribute it.
      if (dim - rankDiff >= srcVecType.getRank() ||
          srcVecType.getDimSize(dim - rankDiff) != dstVecType.getDimSize(dim))
        continue;
      // It is not a broadcasted dimension and it is distributed by the
      // extract_map. We need to propagate the distribution id and ajust the
      // shape.
      extractShape[i] = op.getResultType().getDimSize(i);
      ids.push_back(op.ids()[i]);
    }
    Value source = broadcast.source();
    // If there are still any dimension distributed add a new extract_map.
    if (!ids.empty()) {
      VectorType newVecType =
          VectorType::get(extractShape, dstVecType.getElementType());
      source = rewriter.create<vector::ExtractMapOp>(op.getLoc(), newVecType,
                                                     source, ids);
    }
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, op.getResultType(),
                                                     source);
    return success();
  }
};

struct LLVMGPUScanToGPUPass
    : public LLVMGPUScanToGPUBase<LLVMGPUScanToGPUPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, AffineDialect>();
  }

  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    patterns.insert<ConvertScanToGPU, ExtractMapBroadcastPattern>(
        funcOp.getContext());
    vector::populatePropagateVectorDistributionPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

}  // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> createConvertVectorScanToGPUPass() {
  return std::make_unique<LLVMGPUScanToGPUPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
