// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-breakdown-subbyte-extend"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace iree_compiler {
namespace {

// Breaks down a chain of (load -> bitcast -> broadcast -> extui) ops.
//
// This pattern is meant to handle vectors with sub-byte element types
// that are loaded as non-subbyte element types and then bitcasted into
// subbyte types like:
//
// ```mlir
// %15 = vector.load %0[%14] : memref<22544384xi8>, vector<32xi8>
// %16 = vector.bitcast %15 : vector<32xi8> to vector<64xi4>
// %17 = vector.broadcast %16 : vector<64xi4> to vector<1x1x64xi4>
// %23 = arith.extui %17 : vector<1x1x64xi4> to vector<1x1x64xi32>
// ```
//
// The above op sequence does not move the data when casting to `i4` type,
// but x86 vector instructions cannot handle subbyte data, so the data
// needs to be moved. We handle the moving of the data here in a way that
// can be vectorized because the x86 backend does not handle this case well
//
// We can convert the above to:
// ```mlir
// %cst = arith.constant dense<[15, 240, 3840, 61440, 983040, 15728640,
// 251658240, -268435456, 15, ..]>
//    : vector<64xi32>
// %cst_0 = arith.constant dense<[0, 4, 8, 12, 16, 20, 24, 28, 0, ..]>
//    : vector<64xi32>
// %15 = vector.load %0[%14] : memref<22544384xi8>, vector<32xi8>
// %21 = vector.bitcast %15 : vector<32xi8> to vector<8xi32>
// %22 = vector.shuffle %21, %21 [0, 0, 0, 0, 0, 0, 0, 0, 1, ..]
//    : vector<8xi32>, vector<8xi32>
// %23 = arith.andi %22, %cst : vector<64xi32>
// %24 = arith.shrui %23, %cst_0 : vector<64xi32>
// %25 = vector.broadcast %24 : vector<64xi32> to vector<1x1x64xi32>
// ```
struct BreakDownSubbyteExtend final : OpRewritePattern<arith::ExtUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ExtUIOp extOp,
                                PatternRewriter &rewriter) const override {
    auto broadcastOp = extOp.getIn().getDefiningOp<vector::BroadcastOp>();
    if (!broadcastOp)
      return failure();

    LDBG("broadcastOp: " << broadcastOp);

    auto bitCastOp = broadcastOp.getSource().getDefiningOp<vector::BitCastOp>();
    if (!bitCastOp)
      return failure();

    LDBG("bitCastOp: " << bitCastOp);

    auto loadOp = bitCastOp.getSource().getDefiningOp<vector::LoadOp>();
    if (!loadOp)
      return failure();

    LDBG("loadOp: " << loadOp);

    VectorType bitCastSrcType = bitCastOp.getSourceVectorType();
    VectorType bitCastDstType = bitCastOp.getType();
    VectorType extuiDstType = llvm::dyn_cast<VectorType>(extOp.getType());
    VectorType loadDstType = llvm::dyn_cast<VectorType>(loadOp.getType());
    if (!extuiDstType || !loadDstType)
      return failure();
    // We only handle 1-D vectors for now
    if (loadDstType.getRank() != 1)
      return failure();
    // We only have power-of-two bitwidth cases for now.
    if (!llvm::isPowerOf2_64(loadDstType.getNumElements()))
      return failure();

    int64_t srcElemBitwidth = bitCastSrcType.getElementTypeBitWidth();
    int64_t dstElemBitwidth = bitCastDstType.getElementTypeBitWidth();
    int64_t finalElemBitwidth = extuiDstType.getElementTypeBitWidth();

    // We only handle i8->i4 types for now
    if (dstElemBitwidth != 4 || srcElemBitwidth != 8)
      return failure();

    auto maskType = VectorType::get({extuiDstType.getNumElements()},
                                    extuiDstType.getElementType());

    int32_t maskBase = (1u << dstElemBitwidth) - 1;
    SmallVector<int32_t> maskArray(maskType.getNumElements());
    for (int32_t elemNum = 0; elemNum < maskType.getNumElements(); elemNum++) {
      maskArray[elemNum] = maskBase
                           << (elemNum * dstElemBitwidth % finalElemBitwidth);
    }
    auto maskVals = rewriter.create<arith::ConstantOp>(
        extOp.getLoc(), maskType,
        DenseIntElementsAttr::get(maskType, maskArray));
    LDBG("maskVals: " << maskVals);

    SmallVector<int32_t> shruiArray(maskType.getNumElements());
    for (int32_t elemNum = 0; elemNum < maskType.getNumElements(); elemNum++) {
      shruiArray[elemNum] = elemNum * dstElemBitwidth % finalElemBitwidth;
    }
    auto shruiVals = rewriter.create<arith::ConstantOp>(
        extOp.getLoc(), maskType,
        DenseIntElementsAttr::get(maskType, shruiArray));
    LDBG("shruiVals: " << shruiVals);

    int64_t numBits = srcElemBitwidth * bitCastSrcType.getNumElements();
    auto newBitCastType = VectorType::get({numBits / finalElemBitwidth},
                                          extuiDstType.getElementType());
    Value newBitCastResult = rewriter.create<vector::BitCastOp>(
        extOp.getLoc(), newBitCastType, loadOp.getResult());
    LDBG("newBitCastResult: " << newBitCastResult);

    SmallVector<int64_t> shuffleArray(extuiDstType.getNumElements());
    for (int64_t elemNum = 0; elemNum < extuiDstType.getNumElements();
         elemNum++) {
      shuffleArray[elemNum] = elemNum / (extuiDstType.getNumElements() /
                                         newBitCastType.getNumElements());
    }

    Value shuffleResult = rewriter.create<vector::ShuffleOp>(
        extOp.getLoc(), newBitCastResult, newBitCastResult, shuffleArray);
    LDBG("shuffleResult: " << shuffleResult);

    Value andResult =
        rewriter.create<arith::AndIOp>(extOp.getLoc(), shuffleResult, maskVals);
    LDBG("andResult: " << andResult);

    Value shruiResult =
        rewriter.create<arith::ShRUIOp>(extOp.getLoc(), andResult, shruiVals);
    LDBG("shruiResult: " << shruiResult);

    Value newBroadcast = rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        extOp, extuiDstType, shruiResult);
    LDBG("newBroadcast: " << newBroadcast);

    return success();
  }
};

struct LLVMCPUBreakDownSubbyteExtendPass final
    : public LLVMCPUBreakDownSubbyteExtendBase<
          LLVMCPUBreakDownSubbyteExtendPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<BreakDownSubbyteExtend>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUBreakDownSubbyteExtendPass() {
  return std::make_unique<LLVMCPUBreakDownSubbyteExtendPass>();
}

} // namespace iree_compiler
} // namespace mlir
