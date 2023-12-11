// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef IREE_COMPILER_CODEGEN_COMMON_EXTRACTADDRESSCOMPUTATION_H_
#define IREE_COMPILER_CODEGEN_COMMON_EXTRACTADDRESSCOMPUTATION_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace mlir::iree_compiler {

/// Rewrite a store/load-like op so that all its indices are zeros.
/// E.g., %ld = memref.load %base[%off0]...[%offN]
/// =>
/// %new_base = subview %base[%off0,.., %offN][1,..,1][1,..,1]
/// %ld = memref.load %new_base[0,..,0] :
///    memref<1x..x1xTy, strided<[1,..,1], offset: ?>>
///
/// `getSrcMemRef` returns the source memref for the given load-like operation.
///
/// Using the given rewriter, `rebuildOpFromAddressAndIndices` creates a new
/// StoreLoadLikeOp that reads from srcMemRef[indices].
/// The returned operation will be used to replace storeLoadOp.
template <typename StoreLoadLikeOp, Value (*getSrcMemRef)(StoreLoadLikeOp),
          StoreLoadLikeOp (*rebuildOpFromAddressAndIndices)(
              RewriterBase & /*rewriter*/, StoreLoadLikeOp /*storeLoadOp*/,
              Value /*srcMemRef*/, ArrayRef<Value> /*indices*/),
          SmallVector<OpFoldResult> (*getViewSizeForEachDim)(
              RewriterBase & /*rewriter*/, StoreLoadLikeOp /*storeLoadOp*/)>
struct StoreLoadLikeOpRewriter : public OpRewritePattern<StoreLoadLikeOp> {
  using OpRewritePattern<StoreLoadLikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StoreLoadLikeOp storeLoadLikeOp,
                                PatternRewriter &rewriter) const override {
    Value srcMemRef = getSrcMemRef(storeLoadLikeOp);
    auto ldTy = srcMemRef.getType().cast<MemRefType>();
    unsigned storeLoadRank = ldTy.getRank();
    // Don't waste compile time if there is nothing to rewrite.
    if (storeLoadRank == 0)
      return failure();

    // If our load already has only zeros as indices there is nothing
    // to do.
    SmallVector<OpFoldResult> indices =
        getAsOpFoldResult(storeLoadLikeOp.getIndices());
    if (std::all_of(indices.begin(), indices.end(),
                    [](const OpFoldResult &opFold) {
                      return isConstantIntValue(opFold, 0);
                    })) {
      return failure();
    }

    // Create the array of ones of the right size.
    SmallVector<OpFoldResult> ones(storeLoadRank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes =
        getViewSizeForEachDim(rewriter, storeLoadLikeOp);
    assert(sizes.size() == storeLoadRank &&
           "Expected one size per load dimension");
    Location loc = storeLoadLikeOp.getLoc();
    auto subview =
        rewriter.create<memref::SubViewOp>(loc, /*source=*/srcMemRef,
                                           /*offsets=*/indices,
                                           /*sizes=*/sizes, /*strides=*/ones);
    // Rewrite the load with the subview as the base pointer.
    SmallVector<Value> zeros(storeLoadRank,
                             rewriter.create<arith::ConstantIndexOp>(loc, 0));
    StoreLoadLikeOp newLoad = rebuildOpFromAddressAndIndices(
        rewriter, storeLoadLikeOp, subview.getResult(), zeros);
    rewriter.replaceOp(storeLoadLikeOp, newLoad->getResults());
    return success();
  }
};

/// Add patterns that do address computation extraction.
/// Address computation extraction consists in expressing memory accesses
/// from a simple base pointer and do the offsets computation and so on
/// before hand. In other words, the address computation is not part of
/// the memory access anymore.
void populateExtractAddressComputationPatterns(RewritePatternSet &patterns);
} // namespace mlir::iree_compiler
#endif // IREE_COMPILER_CODEGEN_COMMON_EXTRACTADDRESSCOMPUTATION_H_
