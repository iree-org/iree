// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
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
struct BreakDownSubbyteInsertExtend final : OpRewritePattern<arith::ExtUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ExtUIOp extOp,
                                PatternRewriter &rewriter) const override {
    auto insertOp = extOp.getIn().getDefiningOp<vector::InsertOp>();
    if (!insertOp)
      return failure();

    auto insertSrcType = llvm::dyn_cast<VectorType>(insertOp.getSource().getType());
    if (!insertSrcType || insertSrcType.getRank() != 1) {
      return failure();
    }

    SmallVector<vector::InsertOp> insertOps;
    insertOps.push_back(insertOp);
    while (auto prevInsertOp = insertOp.getDest().getDefiningOp<vector::InsertOp>()){
      insertOps.push_back(prevInsertOp);
      insertOp = prevInsertOp;
    }

    VectorType extuiSrcType = llvm::dyn_cast<VectorType>(extOp.getIn().getType());
    VectorType extuiDstType = llvm::dyn_cast<VectorType>(extOp.getType());
    if (!extuiSrcType || !extuiDstType) {
      return failure();
    }
    LDBG("extuiSrcType: " << extuiSrcType);
    LDBG("extuiDstType: " << extuiDstType);

    // We only have power-of-two bitwidth cases for now.
    if (!llvm::isPowerOf2_64(extuiSrcType.getNumElements()))
      return failure();

    int64_t srcElemBitwidth = extuiSrcType.getElementTypeBitWidth();
    int64_t dstElemBitwidth = extuiDstType.getElementTypeBitWidth();
    LDBG("srcElemBitwidth: " << srcElemBitwidth);
    LDBG("dstElemBitwidth: " << dstElemBitwidth);

    Value newVector = rewriter.create<arith::ConstantOp>(
        insertOps[insertOps.size()-1].getLoc(), extuiDstType, rewriter.getZeroAttr(extuiDstType));

    auto maskType = VectorType::get({insertSrcType.getNumElements()},
                                    extuiDstType.getElementType());

    int32_t maskBase = (1u << srcElemBitwidth) - 1;
    SmallVector<int32_t> maskArray(maskType.getNumElements());
    for (int32_t elemNum = 0; elemNum < maskType.getNumElements(); elemNum++) {
      maskArray[elemNum] = maskBase
                           << (elemNum * srcElemBitwidth % dstElemBitwidth);
    }
    auto maskVals = rewriter.create<arith::ConstantOp>(
        extOp.getLoc(), maskType,
        DenseIntElementsAttr::get(maskType, maskArray));
    LDBG("maskVals: " << maskVals);

    SmallVector<int32_t> shruiArray(maskType.getNumElements());
    for (int32_t elemNum = 0; elemNum < maskType.getNumElements(); elemNum++) {
      shruiArray[elemNum] = elemNum * srcElemBitwidth % dstElemBitwidth;
    }
    auto shruiVals = rewriter.create<arith::ConstantOp>(
        extOp.getLoc(), maskType,
        DenseIntElementsAttr::get(maskType, shruiArray));
    LDBG("shruiVals: " << shruiVals);


    for (int64_t idx = insertOps.size()-1; idx >= 0; idx--) {
      auto insert = insertOps[idx];
      if (llvm::dyn_cast<VectorType>(insert.getSource().getType()) != insertSrcType) {
        return failure();
      }
      auto bitCast = insert.getSource().getDefiningOp<vector::BitCastOp>();
      if (!bitCast) {
        return failure();
      }

      VectorType bitCastSrcType = bitCast.getSourceVectorType();

      int64_t numBits = bitCastSrcType.getElementTypeBitWidth() * bitCastSrcType.getNumElements();
      auto newBitCastType = VectorType::get({numBits / dstElemBitwidth},
                                            extuiDstType.getElementType());
      Value newBitCastResult = rewriter.create<vector::BitCastOp>(
          insert.getLoc(), newBitCastType, bitCast.getSource());
      LDBG("newBitCastResult: " << newBitCastResult);

      SmallVector<int64_t> shuffleArray(numBits / srcElemBitwidth);
      for (int64_t elemNum = 0; elemNum < numBits / srcElemBitwidth;
          elemNum++) {
        shuffleArray[elemNum] = elemNum / (numBits / srcElemBitwidth /
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

      newVector = rewriter.create<vector::InsertOp>(insert.getLoc(), shruiResult, newVector, insert.getPosition());
    }
    rewriter.replaceOp(extOp, newVector);
    return success();
  }
};


struct BreakDownSubbyteExtend final : OpRewritePattern<arith::ExtUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ExtUIOp extOp,
                                PatternRewriter &rewriter) const override {
    VectorType extuiSrcType = llvm::dyn_cast<VectorType>(extOp.getIn().getType());
    VectorType extuiDstType = llvm::dyn_cast<VectorType>(extOp.getType());
    if (!extuiSrcType || !extuiDstType) {
      return failure();
    }
    LDBG("extuiSrcType: " << extuiSrcType);
    LDBG("extuiDstType: " << extuiDstType);

    // We only have power-of-two bitwidth cases for now.
    if (!llvm::isPowerOf2_64(extuiSrcType.getNumElements()))
      return failure();

    int64_t srcElemBitwidth = extuiSrcType.getElementTypeBitWidth();
    int64_t dstElemBitwidth = extuiDstType.getElementTypeBitWidth();
    LDBG("srcElemBitwidth: " << srcElemBitwidth);
    LDBG("dstElemBitwidth: " << dstElemBitwidth);

    auto srcShape = extuiSrcType.getShape();

    Value newVector = rewriter.create<arith::ConstantOp>(
        extOp.getLoc(), extuiDstType, rewriter.getZeroAttr(extuiDstType));

    for (int64_t i = 0; i < extuiSrcType.getNumElements()/srcShape[srcShape.size()-1]; i++) {
      SmallVector<int64_t> indices;
      int64_t numElems = i;
      for (int64_t shapeIdx = srcShape.size()-2; shapeIdx >= 0; shapeIdx--) {
        int64_t size = srcShape[shapeIdx];
        indices.push_back(numElems % size);
        numElems /= size;
      }
      SmallVector<int64_t> extractIndices;
      for (int64_t j = indices.size()-1; j >= 0; j--) {
        extractIndices.push_back(indices[j]);
      }

      auto extractOp = rewriter.create<vector::ExtractOp>(extOp.getLoc(), extOp.getIn(), extractIndices);
      LDBG("extractOp: " << extractOp);
      VectorType extractOpType = llvm::cast<VectorType>(extractOp.getType());
      if (extractOpType.getNumElements() * srcElemBitwidth / dstElemBitwidth < 1) {
        return failure();
      }
      auto bitCastType = VectorType::get({extractOpType.getNumElements() * srcElemBitwidth / dstElemBitwidth},
                                            extuiDstType.getElementType());
      Value bitCastResult = rewriter.create<vector::BitCastOp>(
          extOp.getLoc(), bitCastType, extractOp);
      LDBG("bitCastResult: " << bitCastResult);

      auto extuiShape = extuiDstType.getShape();
      SmallVector<int64_t> shuffleArray(extuiShape[extuiShape.size()-1]);
      for (int64_t elemNum = 0; elemNum < extuiShape[extuiShape.size()-1];
          elemNum++) {
        shuffleArray[elemNum] = elemNum / (extuiShape[extuiShape.size()-1] /
                                          bitCastType.getNumElements());
      }

      Value shuffleResult = rewriter.create<vector::ShuffleOp>(
          extOp.getLoc(), bitCastResult, bitCastResult, shuffleArray);
      LDBG("shuffleResult: " << shuffleResult);

      newVector = rewriter.create<vector::InsertOp>(extOp.getLoc(), shuffleResult, newVector, extractIndices);
    }

    int32_t maskBase = (1u << srcElemBitwidth) - 1;
    SmallVector<int32_t> maskArray(extuiDstType.getNumElements());
    for (int32_t elemNum = 0; elemNum < extuiDstType.getNumElements(); elemNum++) {
      maskArray[elemNum] = maskBase
                           << (elemNum * srcElemBitwidth % dstElemBitwidth);
    }
    auto maskVals = rewriter.create<arith::ConstantOp>(
        extOp.getLoc(), extuiDstType,
        DenseIntElementsAttr::get(extuiDstType, maskArray));
    LDBG("maskVals: " << maskVals);

    SmallVector<int32_t> shruiArray(extuiDstType.getNumElements());
    for (int32_t elemNum = 0; elemNum < extuiDstType.getNumElements(); elemNum++) {
      shruiArray[elemNum] = elemNum * srcElemBitwidth % dstElemBitwidth;
    }
    auto shruiVals = rewriter.create<arith::ConstantOp>(
        extOp.getLoc(), extuiDstType,
        DenseIntElementsAttr::get(extuiDstType, shruiArray));
    LDBG("shruiVals: " << shruiVals);

    Value andResult =
        rewriter.create<arith::AndIOp>(extOp.getLoc(), newVector, maskVals);
    LDBG("andResult: " << andResult);

    Value shruiResult =
        rewriter.replaceOpWithNewOp<arith::ShRUIOp>(extOp, andResult, shruiVals);
    LDBG("shruiResult: " << shruiResult);

    return success();
  }
};

struct BreakDownSubbyteExtendFlatten final : OpRewritePattern<arith::ExtUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ExtUIOp extOp,
                                PatternRewriter &rewriter) const override {
    VectorType extuiSrcType = llvm::dyn_cast<VectorType>(extOp.getIn().getType());
    VectorType extuiDstType = llvm::dyn_cast<VectorType>(extOp.getType());
    if (!extuiSrcType || !extuiDstType) {
      return failure();
    }
    LDBG("extuiSrcType: " << extuiSrcType);
    LDBG("extuiDstType: " << extuiDstType);

    // We only have power-of-two bitwidth cases for now.
    if (!llvm::isPowerOf2_64(extuiSrcType.getNumElements()))
      return failure();

    int64_t srcElemBitwidth = extuiSrcType.getElementTypeBitWidth();
    int64_t dstElemBitwidth = extuiDstType.getElementTypeBitWidth();
    LDBG("srcElemBitwidth: " << srcElemBitwidth);
    LDBG("dstElemBitwidth: " << dstElemBitwidth);

    int64_t numBits = srcElemBitwidth * extuiSrcType.getNumElements();
    if (numBits / dstElemBitwidth < 1) {
      return failure();
    }

    VectorType flattenedType = VectorType::get({extuiSrcType.getNumElements()}, extuiSrcType.getElementType());
    Value shapeCastFlatten = rewriter.create<vector::ShapeCastOp>(extOp.getLoc(), flattenedType, extOp.getIn());

    auto bitCastType = VectorType::get({numBits / dstElemBitwidth}, extuiDstType.getElementType());
    Value bitCastResult = rewriter.create<vector::BitCastOp>(
        extOp.getLoc(), bitCastType, shapeCastFlatten);
    LDBG("bitCastResult: " << bitCastResult);


    SmallVector<int64_t> shuffleArray(extuiDstType.getNumElements());
    for (int64_t elemNum = 0; elemNum < extuiDstType.getNumElements();
        elemNum++) {
      shuffleArray[elemNum] = elemNum / (extuiDstType.getNumElements() /
                                        bitCastType.getNumElements());
    }

    Value shuffleResult = rewriter.create<vector::ShuffleOp>(
        extOp.getLoc(), bitCastResult, bitCastResult, shuffleArray);
    LDBG("shuffleResult: " << shuffleResult);

    Value shapeCastUnflatten = rewriter.create<vector::ShapeCastOp>(extOp.getLoc(), extuiDstType, shuffleResult);

    int32_t maskBase = (1u << srcElemBitwidth) - 1;
    SmallVector<int32_t> maskArray(extuiDstType.getNumElements());
    for (int32_t elemNum = 0; elemNum < extuiDstType.getNumElements(); elemNum++) {
      maskArray[elemNum] = maskBase
                           << (elemNum * srcElemBitwidth % dstElemBitwidth);
    }
    auto maskVals = rewriter.create<arith::ConstantOp>(
        extOp.getLoc(), extuiDstType,
        DenseIntElementsAttr::get(extuiDstType, maskArray));
    LDBG("maskVals: " << maskVals);

    SmallVector<int32_t> shruiArray(extuiDstType.getNumElements());
    for (int32_t elemNum = 0; elemNum < extuiDstType.getNumElements(); elemNum++) {
      shruiArray[elemNum] = elemNum * srcElemBitwidth % dstElemBitwidth;
    }
    auto shruiVals = rewriter.create<arith::ConstantOp>(
        extOp.getLoc(), extuiDstType,
        DenseIntElementsAttr::get(extuiDstType, shruiArray));
    LDBG("shruiVals: " << shruiVals);

    Value andResult =
        rewriter.create<arith::AndIOp>(extOp.getLoc(), shapeCastUnflatten, maskVals);
    LDBG("andResult: " << andResult);

    Value shruiResult =
        rewriter.replaceOpWithNewOp<arith::ShRUIOp>(extOp, andResult, shruiVals);
    LDBG("shruiResult: " << shruiResult);

    return success();
  }
};


// struct BreakDownSubbyteLoadExtend final : OpRewritePattern<arith::ExtUIOp> {
//   using OpRewritePattern::OpRewritePattern;
//   LogicalResult matchAndRewrite(arith::ExtUIOp extOp,
//                                 PatternRewriter &rewriter) const override {
//     // LDBG("broadcastOp: " << broadcastOp);

//     // auto bitCastOp = broadcastOp.getSource().getDefiningOp<vector::BitCastOp>();
//     // if (!bitCastOp)
//     //   return failure();

//     // LDBG("bitCastOp: " << bitCastOp);

//     // auto loadOp = bitCastOp.getSource().getDefiningOp<vector::LoadOp>();
//     // if (!loadOp)
//     //   return failure();

//     // LDBG("loadOp: " << loadOp);

//     // VectorType bitCastSrcType = bitCastOp.getSourceVectorType();
//     // VectorType bitCastDstType = bitCastOp.getType();
//     // VectorType extuiDstType = llvm::dyn_cast<VectorType>(extOp.getType());
//     // VectorType loadDstType = llvm::dyn_cast<VectorType>(loadOp.getType());
//     // if (!extuiDstType || !loadDstType)
//     //   return failure();
//     // // We only handle 1-D vectors for now
//     // if (loadDstType.getRank() != 1)
//     //   return failure();
//     // // We only have power-of-two bitwidth cases for now.
//     // if (!llvm::isPowerOf2_64(loadDstType.getNumElements()))
//     //   return failure();

//     // int64_t srcElemBitwidth = bitCastSrcType.getElementTypeBitWidth();
//     // int64_t dstElemBitwidth = bitCastDstType.getElementTypeBitWidth();
//     // int64_t finalElemBitwidth = extuiDstType.getElementTypeBitWidth();

//     // // We only handle i8->i4 types for now
//     // if (dstElemBitwidth != 4 || srcElemBitwidth != 8)
//     //   return failure();

//     // auto maskType = VectorType::get({extuiDstType.getNumElements()},
//     //                                 extuiDstType.getElementType());

//     // int32_t maskBase = (1u << dstElemBitwidth) - 1;
//     // SmallVector<int32_t> maskArray(maskType.getNumElements());
//     // for (int32_t elemNum = 0; elemNum < maskType.getNumElements(); elemNum++) {
//     //   maskArray[elemNum] = maskBase
//     //                        << (elemNum * dstElemBitwidth % finalElemBitwidth);
//     // }
//     // auto maskVals = rewriter.create<arith::ConstantOp>(
//     //     extOp.getLoc(), maskType,
//     //     DenseIntElementsAttr::get(maskType, maskArray));
//     // LDBG("maskVals: " << maskVals);

//     // SmallVector<int32_t> shruiArray(maskType.getNumElements());
//     // for (int32_t elemNum = 0; elemNum < maskType.getNumElements(); elemNum++) {
//     //   shruiArray[elemNum] = elemNum * dstElemBitwidth % finalElemBitwidth;
//     // }
//     // auto shruiVals = rewriter.create<arith::ConstantOp>(
//     //     extOp.getLoc(), maskType,
//     //     DenseIntElementsAttr::get(maskType, shruiArray));
//     // LDBG("shruiVals: " << shruiVals);

//     // int64_t numBits = srcElemBitwidth * bitCastSrcType.getNumElements();
//     // auto newBitCastType = VectorType::get({numBits / finalElemBitwidth},
//     //                                       extuiDstType.getElementType());
//     // Value newBitCastResult = rewriter.create<vector::BitCastOp>(
//     //     extOp.getLoc(), newBitCastType, loadOp.getResult());
//     // LDBG("newBitCastResult: " << newBitCastResult);

//     // SmallVector<int64_t> shuffleArray(extuiDstType.getNumElements());
//     // for (int64_t elemNum = 0; elemNum < extuiDstType.getNumElements();
//     //      elemNum++) {
//     //   shuffleArray[elemNum] = elemNum / (extuiDstType.getNumElements() /
//     //                                      newBitCastType.getNumElements());
//     // }

//     // Value shuffleResult = rewriter.create<vector::ShuffleOp>(
//     //     extOp.getLoc(), newBitCastResult, newBitCastResult, shuffleArray);
//     // LDBG("shuffleResult: " << shuffleResult);

//     // Value andResult =
//     //     rewriter.create<arith::AndIOp>(extOp.getLoc(), shuffleResult, maskVals);
//     // LDBG("andResult: " << andResult);

//     // Value shruiResult =
//     //     rewriter.create<arith::ShRUIOp>(extOp.getLoc(), andResult, shruiVals);
//     // LDBG("shruiResult: " << shruiResult);

//     // Value newBroadcast = rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
//     //     extOp, extuiDstType, shruiResult);
//     // LDBG("newBroadcast: " << newBroadcast);

//     return success();
//   }
// };

struct LLVMCPUBreakDownSubbyteExtendPass final
    : public LLVMCPUBreakDownSubbyteExtendBase<
          LLVMCPUBreakDownSubbyteExtendPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    {
      RewritePatternSet patterns(context);
      patterns.add<BreakDownSubbyteExtend>(context);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // For the case when the innermost dimension of the src type is too small to
    // fill a single element of the dst type. Might not be worth doing anything in
    // this case
    {
      RewritePatternSet patterns(context);
      patterns.add<BreakDownSubbyteExtendFlatten>(context);
      vector::populateVectorShapeCastLoweringPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
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
