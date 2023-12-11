// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/SPIRV/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {

// Breaks down a chain of (bitcast -> extract -> extui) ops.
//
// This pattern is meant to handle large vectors with sub-byte element types in
// the middle like the following:
//
// ```mlir
// %bitcast = vector.bitcast %input : vector<1xi32> to vector<8xi4>
// %extract = vector.extract_strided_slice %bitcast {
//              offsets = [4], sizes = [4], strides = [1]}
//            : vector<8xi4> to vector<4xi4>
// %extend = arith.extui %extract : vector<4xi4> to vector<4xi32>
// ```
//
// The above op sequence would cause quite some pain when converting to SPIR-V,
// given that 1) we need to emulate the sub-byte types and 2) type conversion
// won't know about the signedness when emulating them. So here it's better to
// use one dedicated pattern to match the whole sequence and process them.
//
// We can convert the above to
//
// ``mlir
// %zero = arith.constant dense<0> : vector<4xi32>
// %base = vector.extract %input[0] : vector<1xi32>
// %shr0 = arith.shrui %base, %c0
// %and0 = arith.andi %shr0, %c15
// %ins0 = vector.insert %and0, %zero [0]
// %shr1 = arith.shrui %base, %4
// %and1 = arith.andi %shr1, %c15
// %ins1 = vector.insert %and1, %ins1 [1]
// %shr2 = arith.shrui %base, %c8
// %and2 = arith.andi %shr2, %c15
// %ins2 = vector.insert %and2, %ins2 [2]
// %shr3 = arith.shrui %base, %c12
// %and3 = arith.andi %shr3, %c15
// %extend = vector.insert %and2, %ins3 [3]
// ```
//
// Note that the above pattern assumes littlen-endian style encoding of the
// sub-byte elements. That is, for 8xi4 [A, B, C, D, E, F, G, H], they are
// stored in memory as [BA, DC, FE, HG], and read as an i32 HGFEDCBA. Therefore
// the first i4 element is the lest significant 4 bits.
struct BreakDownCastExtractExtend final : OpRewritePattern<arith::ExtUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ExtUIOp extOp,
                                PatternRewriter &rewriter) const override {
    auto extractOp =
        extOp.getIn().getDefiningOp<vector::ExtractStridedSliceOp>();
    if (!extractOp)
      return failure();

    auto bitCastOp = extractOp.getVector().getDefiningOp<vector::BitCastOp>();
    if (!bitCastOp)
      return failure();

    VectorType extractSrcType = extractOp.getSourceVectorType();
    VectorType extractDstType = extractOp.getType();
    // We expect high-D vectors are broken down into 1-D ones so here we only
    // handle 1-D vectors.
    if (extractSrcType.getRank() != 1 || extractDstType.getRank() != 1)
      return failure();
    // We only have power-of-two bitwidth cases for now.
    if (!llvm::isPowerOf2_64(extractSrcType.getNumElements()) ||
        !llvm::isPowerOf2_64(extractDstType.getNumElements()))
      return failure();
    // We only handle not directly supported vector sizes.
    if (extractSrcType.getNumElements() <= 4)
      return failure();

    int64_t srcElemBitwidth =
        bitCastOp.getSourceVectorType().getElementTypeBitWidth();
    int64_t midElemBitwidth = extractDstType.getElementTypeBitWidth();

    // Calculate which original element to extract and the base offset to
    // extract bits from.
    int64_t extractElemOffset =
        cast<IntegerAttr>(extractOp.getOffsets()[0]).getInt();
    int64_t extractBitOffset =
        extractElemOffset * extractDstType.getElementTypeBitWidth();
    int64_t srcElemIndex = extractBitOffset / srcElemBitwidth;
    int64_t srcElemOffset = extractBitOffset % srcElemBitwidth;

    Value srcElement = rewriter.create<vector::ExtractOp>(
        extractOp.getLoc(), bitCastOp.getSource(),
        ArrayRef<int64_t>{srcElemIndex});

    Value result = rewriter.create<arith::ConstantOp>(
        extractOp.getLoc(), extOp.getType(),
        rewriter.getZeroAttr(extOp.getType()));

    // Extract each elements assuming little-endian style encoding--lower bits
    // corresponds to earlier elements.
    auto dstElemType = cast<VectorType>(extOp.getType()).getElementType();
    auto mask = rewriter.create<arith::ConstantOp>(
        extOp.getLoc(), dstElemType,
        rewriter.getIntegerAttr(dstElemType, (1u << midElemBitwidth) - 1));
    int64_t shrSize = srcElemOffset;
    for (int i = 0; i < extractDstType.getNumElements(); ++i) {
      // Each time we extract midElemBitwidth bits from srcElement. We do that
      // by shift right first and then and a mask.
      Value shrVal = rewriter.create<arith::ConstantOp>(
          extractOp.getLoc(), dstElemType,
          rewriter.getIntegerAttr(dstElemType, shrSize));
      Value shr = rewriter.create<arith::ShRUIOp>(extractOp.getLoc(),
                                                  srcElement, shrVal);
      Value elem =
          rewriter.create<arith::AndIOp>(extractOp.getLoc(), shr, mask);
      result = rewriter.create<vector::InsertOp>(extractOp.getLoc(), elem,
                                                 result, i);
      shrSize += midElemBitwidth;
    }

    rewriter.replaceOp(extOp, result);

    return success();
  }
};

struct SPIRVBreakDownLargeVectorPass final
    : public SPIRVBreakDownLargeVectorBase<SPIRVBreakDownLargeVectorPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<BreakDownCastExtractExtend>(context, /*benefits=*/10);
    // Convert vector.extract_strided_slice into a chain of vector.extract and
    // then a chain of vector.insert ops. This helps to cancel with previous
    // vector.insert/extract ops, especially for fP16 cases where we have
    // mismatched vector size for transfer and compute.
    vector::populateVectorExtractStridedSliceToExtractInsertChainPatterns(
        patterns, [](vector::ExtractStridedSliceOp op) {
          return op.getSourceVectorType().getNumElements() > 4;
        });
    vector::populateBreakDownVectorBitCastOpPatterns(
        patterns, [](vector::BitCastOp op) {
          return op.getSourceVectorType().getNumElements() > 4;
        });
    vector::InsertOp::getCanonicalizationPatterns(patterns, context);
    vector::ExtractOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVBreakDownLargeVectorPass() {
  return std::make_unique<SPIRVBreakDownLargeVectorPass>();
}

} // namespace mlir::iree_compiler
