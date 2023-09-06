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

template <typename T>
static Value shuffleMaskShift(PatternRewriter &rewriter, Location loc,
                              SmallVector<Value> shuffleInputs,
                              int64_t srcBitWidth, int64_t vectorSize) {
  auto shuffleInType = llvm::cast<VectorType>(shuffleInputs[0].getType());
  auto shuffleResultType =
      VectorType::get({vectorSize}, shuffleInType.getElementType());
  int64_t dstBitWidth = shuffleInType.getElementTypeBitWidth();
  T maskBase = (1u << srcBitWidth) - 1;

  SmallVector<T> maskArray(shuffleResultType.getNumElements());
  for (T elemNum = 0; elemNum < shuffleResultType.getNumElements(); elemNum++) {
    maskArray[elemNum] = maskBase << (elemNum * srcBitWidth % dstBitWidth);
  }
  auto maskVals = rewriter.create<arith::ConstantOp>(
      loc, shuffleResultType,
      DenseIntElementsAttr::get(shuffleResultType, maskArray));
  LDBG("maskVals: " << maskVals);
  SmallVector<T> shruiArray(shuffleResultType.getNumElements());
  for (T elemNum = 0; elemNum < shuffleResultType.getNumElements(); elemNum++) {
    shruiArray[elemNum] = elemNum * srcBitWidth % dstBitWidth;
  }
  auto shruiVals = rewriter.create<arith::ConstantOp>(
      loc, shuffleResultType,
      DenseIntElementsAttr::get(shuffleResultType, shruiArray));
  LDBG("shruiVals: " << shruiVals);

  int64_t dstSize = vectorSize * shuffleInputs.size();
  auto newVectorType =
      VectorType::get({dstSize}, shuffleResultType.getElementType());
  Value newVector = rewriter.create<arith::ConstantOp>(
      loc, newVectorType, rewriter.getZeroAttr(newVectorType));

  for (auto shuffleIn : llvm::enumerate(shuffleInputs)) {
    SmallVector<int64_t> shuffleArray(vectorSize);
    for (int64_t elemNum = 0; elemNum < vectorSize; elemNum++) {
      shuffleArray[elemNum] =
          elemNum / (vectorSize / shuffleInType.getNumElements());
    }
    Value shuffleResult = rewriter.create<vector::ShuffleOp>(
        loc, shuffleIn.value(), shuffleIn.value(), shuffleArray);
    LDBG("shuffleResult: " << shuffleResult);

    Value andResult =
        rewriter.create<arith::AndIOp>(loc, shuffleResult, maskVals);
    LDBG("andResult: " << andResult);

    Value shruiResult =
        rewriter.create<arith::ShRUIOp>(loc, andResult, shruiVals);
    LDBG("shruiResult: " << shruiResult);

    int64_t offset = shuffleIn.index() * vectorSize;
    newVector = rewriter.create<vector::InsertStridedSliceOp>(
        loc, shruiResult, newVector, offset, 1);
  }
  return newVector;
}

static std::optional<SmallVector<Value>>
getLoadsForExtend(arith::ExtUIOp extOp) {
  Value extSource = extOp.getIn();
  auto shapeCastOp = extSource.getDefiningOp<vector::ShapeCastOp>();
  if (!shapeCastOp) {
    return std::nullopt;
  }
  Value shapeCastSource = shapeCastOp.getSource();
  auto insertOp = shapeCastSource.getDefiningOp<vector::InsertStridedSliceOp>();
  if (!insertOp) {
    return std::nullopt;
  }
  SmallVector<Value> loads;
  while (insertOp) {
    Value insert = insertOp.getSource();
    auto insertShapeCastOp = insert.getDefiningOp<vector::ShapeCastOp>();
    if (!insertShapeCastOp) {
      return std::nullopt;
    }
    auto loadOp = insertShapeCastOp.getSource().getDefiningOp<vector::LoadOp>();
    if (!loadOp) {
      return std::nullopt;
    }
    loads.push_back(loadOp.getResult());
    insertOp = insertOp.getDest().getDefiningOp<vector::InsertStridedSliceOp>();
  }
  return loads;
}

struct BreakDownSubbyteExtend final : OpRewritePattern<arith::ExtUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ExtUIOp extOp,
                                PatternRewriter &rewriter) const override {
    VectorType extuiSrcType =
        llvm::dyn_cast<VectorType>(extOp.getIn().getType());
    VectorType extuiDstType = llvm::dyn_cast<VectorType>(extOp.getType());
    if (!extuiSrcType || !extuiDstType) {
      return failure();
    }

    SmallVector<Value> sources{extOp.getIn()};
    if (auto loads = getLoadsForExtend(extOp)) {
      sources = *loads;
    }

    int64_t srcElemBitwidth = extuiSrcType.getElementTypeBitWidth();
    int64_t dstElemBitwidth = extuiDstType.getElementTypeBitWidth();
    // We only have power-of-two bitwidth cases for now.
    if (!llvm::isPowerOf2_64(dstElemBitwidth) || srcElemBitwidth != 4)
      return failure();

    if (dstElemBitwidth != 32 && dstElemBitwidth != 16) {
      return failure();
    }

    int64_t vectorSizeBits = 512;
    int64_t vectorSize = vectorSizeBits / dstElemBitwidth;
    int64_t shuffleInputSizeBits = vectorSize * srcElemBitwidth;
    int64_t shuffleInputSize = shuffleInputSizeBits / dstElemBitwidth;
    auto shuffleInputType =
        VectorType::get({shuffleInputSize}, extuiDstType.getElementType());
    Value shuffleInput = rewriter.create<arith::ConstantOp>(
        extOp.getLoc(), shuffleInputType,
        rewriter.getZeroAttr(shuffleInputType));
    SmallVector<Value> shuffleInputs;

    for (int sourceIdx = 0; sourceIdx < sources.size(); sourceIdx++) {
      Value source = sources[sourceIdx];
      VectorType sourceType = llvm::cast<VectorType>(source.getType());
      SmallVector<int64_t> sourceShape(sourceType.getShape());
      int64_t innerSize = sourceShape.back();
      if (!llvm::isPowerOf2_64(innerSize)) {
        return failure();
      }
      for (int64_t i = 0; i < sourceType.getNumElements() / innerSize; i++) {
        SmallVector<int64_t> indices;
        int64_t numElems = i;
        SmallVector<int64_t> sourceOuterShape(sourceShape.begin(),
                                              sourceShape.end() - 1);
        for (int64_t size : llvm::reverse(sourceOuterShape)) {
          indices.push_back(numElems % size);
          numElems /= size;
        }
        std::reverse(indices.begin(), indices.end());

        Value innerSlice;
        if (indices.size()) {
          innerSlice = rewriter.create<vector::ExtractOp>(extOp.getLoc(),
                                                          source, indices);
        } else {
          innerSlice = source;
        }
        VectorType innerSliceType =
            llvm::cast<VectorType>(innerSlice.getType());
        int64_t numExtractedBits =
            innerSliceType.getNumElements() * srcElemBitwidth;
        if (numExtractedBits / dstElemBitwidth < 1) {
          LDBG("extract not big enough: " << numExtractedBits /
                                                 dstElemBitwidth);
          return failure();
        }
        auto bitCastType = VectorType::get({numExtractedBits / dstElemBitwidth},
                                           extuiDstType.getElementType());
        Value bitCastResult = rewriter.create<vector::BitCastOp>(
            extOp.getLoc(), bitCastType, innerSlice);
        LDBG("innerSlice: " << innerSlice);
        // LDBG("bitCastResult: " << bitCastResult);

        if (numExtractedBits >= shuffleInputSizeBits) {
          for (int64_t extractOffset = 0;
               extractOffset < numExtractedBits / dstElemBitwidth;
               extractOffset += shuffleInputSize) {
            Value extractedSlice =
                rewriter.create<vector::ExtractStridedSliceOp>(
                    extOp.getLoc(), bitCastResult, extractOffset,
                    shuffleInputSize, 1);
            shuffleInputs.push_back(extractedSlice);
            LDBG("extractedSlice: " << extractedSlice);
            // vector =
            // rewriter.create<vector::InsertStridedSliceOp>(extOp.getLoc(),
            // extractedSlice, vector, SmallVector<int64_t>{offset},
            // SmallVector<int64_t>{1});
          }
        } else {
          int64_t offset =
              i * numExtractedBits / dstElemBitwidth % shuffleInputSize;
          shuffleInput = rewriter.create<vector::InsertStridedSliceOp>(
              extOp.getLoc(), bitCastResult, shuffleInput,
              SmallVector<int64_t>{offset}, SmallVector<int64_t>{1});
          if (offset + numExtractedBits / dstElemBitwidth == shuffleInputSize) {
            shuffleInputs.push_back(shuffleInput);
            shuffleInput = rewriter.create<arith::ConstantOp>(
                extOp.getLoc(), shuffleInputType,
                rewriter.getZeroAttr(shuffleInputType));
          }
        }
      }
    }

    Value newVector;
    if (dstElemBitwidth == 32) {
      newVector = shuffleMaskShift<int32_t>(
          rewriter, extOp.getLoc(), shuffleInputs, srcElemBitwidth, vectorSize);
    } else if (dstElemBitwidth == 16) {
      newVector = shuffleMaskShift<int16_t>(
          rewriter, extOp.getLoc(), shuffleInputs, srcElemBitwidth, vectorSize);
    }
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(extOp, extuiDstType,
                                                     newVector);

    return success();
  }
};

struct BreakDownSubbyteExtendFlatten final : OpRewritePattern<arith::ExtUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ExtUIOp extOp,
                                PatternRewriter &rewriter) const override {
    VectorType extuiSrcType =
        llvm::dyn_cast<VectorType>(extOp.getIn().getType());
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

    VectorType flattenedType = VectorType::get({extuiSrcType.getNumElements()},
                                               extuiSrcType.getElementType());
    Value shapeCastFlatten = rewriter.create<vector::ShapeCastOp>(
        extOp.getLoc(), flattenedType, extOp.getIn());

    auto bitCastType = VectorType::get({numBits / dstElemBitwidth},
                                       extuiDstType.getElementType());
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

    Value shapeCastUnflatten = rewriter.create<vector::ShapeCastOp>(
        extOp.getLoc(), extuiDstType, shuffleResult);
    Value maskVals, shruiVals;
    if (dstElemBitwidth == 32) {
      int32_t maskBase = (1u << srcElemBitwidth) - 1;
      SmallVector<int32_t> maskArray(extuiDstType.getNumElements());
      for (int32_t elemNum = 0; elemNum < extuiDstType.getNumElements();
           elemNum++) {
        maskArray[elemNum] = maskBase
                             << (elemNum * srcElemBitwidth % dstElemBitwidth);
      }
      maskVals = rewriter.create<arith::ConstantOp>(
          extOp.getLoc(), extuiDstType,
          DenseIntElementsAttr::get(extuiDstType, maskArray));
      LDBG("maskVals: " << maskVals);

      SmallVector<int32_t> shruiArray(extuiDstType.getNumElements());
      for (int32_t elemNum = 0; elemNum < extuiDstType.getNumElements();
           elemNum++) {
        shruiArray[elemNum] = elemNum * srcElemBitwidth % dstElemBitwidth;
      }
      shruiVals = rewriter.create<arith::ConstantOp>(
          extOp.getLoc(), extuiDstType,
          DenseIntElementsAttr::get(extuiDstType, shruiArray));
      LDBG("shruiVals: " << shruiVals);
    } else if (dstElemBitwidth == 16) {
      int16_t maskBase = (1u << srcElemBitwidth) - 1;
      SmallVector<int16_t> maskArray(extuiDstType.getNumElements());
      for (int16_t elemNum = 0; elemNum < extuiDstType.getNumElements();
           elemNum++) {
        maskArray[elemNum] = maskBase
                             << (elemNum * srcElemBitwidth % dstElemBitwidth);
      }
      maskVals = rewriter.create<arith::ConstantOp>(
          extOp.getLoc(), extuiDstType,
          DenseIntElementsAttr::get(extuiDstType, maskArray));
      LDBG("maskVals: " << maskVals);

      SmallVector<int16_t> shruiArray(extuiDstType.getNumElements());
      for (int16_t elemNum = 0; elemNum < extuiDstType.getNumElements();
           elemNum++) {
        shruiArray[elemNum] = elemNum * srcElemBitwidth % dstElemBitwidth;
      }
      shruiVals = rewriter.create<arith::ConstantOp>(
          extOp.getLoc(), extuiDstType,
          DenseIntElementsAttr::get(extuiDstType, shruiArray));
      LDBG("shruiVals: " << shruiVals);
    } else {
      return failure();
    }

    Value andResult = rewriter.create<arith::AndIOp>(
        extOp.getLoc(), shapeCastUnflatten, maskVals);
    LDBG("andResult: " << andResult);

    rewriter.replaceOpWithNewOp<arith::ShRUIOp>(extOp, andResult, shruiVals);

    return success();
  }
};

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
    // fill a single element of the dst type.
    // {
    //   RewritePatternSet patterns(context);
    //   patterns.add<BreakDownSubbyteExtendFlatten>(context);
    //   vector::populateVectorShapeCastLoweringPatterns(patterns);
    //   if (failed(applyPatternsAndFoldGreedily(getOperation(),
    //                                           std::move(patterns)))) {
    //     return signalPassFailure();
    //   }
    // }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUBreakDownSubbyteExtendPass() {
  return std::make_unique<LLVMCPUBreakDownSubbyteExtendPass>();
}

void populateLLVMCPUBreakDownSubbyteExtendPatterns(
    RewritePatternSet &patterns) {
  patterns.add<BreakDownSubbyteExtend>(patterns.getContext());
}

} // namespace iree_compiler
} // namespace mlir
