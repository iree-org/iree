// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

FailureOr<linalg::GenericOp> generalizeArgCompareOp(RewriterBase &rewriter,
                                                    ArgCompareOp argCompareOp) {
  Location loc = argCompareOp.getLoc();
  Value input = argCompareOp.getInputValue();
  Value outVal = argCompareOp.outputValue();
  Value outIdx = argCompareOp.outputIndex();
  int64_t reductionDim = argCompareOp.getDimension();

  ShapedType inputType = argCompareOp.getInputType();
  ShapedType outValType = argCompareOp.getOutputValueType();
  ShapedType outIdxType = argCompareOp.getOutputIndexType();

  Type idxElemType = outIdxType.getElementType();
  int64_t rank = inputType.getRank();

  SmallVector<AffineExpr> inputExprs, outputExprs;
  for (int64_t i = 0; i < rank; ++i) {
    inputExprs.push_back(rewriter.getAffineDimExpr(i));
    if (i != reductionDim) {
      outputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
  }

  MLIRContext *ctx = rewriter.getContext();
  AffineMap inputMap = AffineMap::get(rank, 0, inputExprs, ctx);
  AffineMap outputMap = AffineMap::get(rank, 0, outputExprs, ctx);
  SmallVector<AffineMap> indexingMaps = {inputMap, outputMap, outputMap};

  SmallVector<utils::IteratorType> iteratorTypes(rank,
                                                 utils::IteratorType::parallel);
  iteratorTypes[reductionDim] = utils::IteratorType::reduction;

  Block &srcBlock = argCompareOp.getRegion().front();
  auto genericOp = linalg::GenericOp::create(
      rewriter, loc, TypeRange{outValType, outIdxType}, ValueRange{input},
      ValueRange{outVal, outIdx}, indexingMaps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value inputElem = args[0];
        Value accVal = args[1];
        Value accIdx = args[2];

        Value idx =
            linalg::IndexOp::create(nestedBuilder, nestedLoc, reductionDim);
        // Add index_base offset if provided (index_base is of type index).
        if (Value indexBase = argCompareOp.getIndexBase()) {
          idx = arith::AddIOp::create(nestedBuilder, nestedLoc, idx, indexBase);
        }
        Value currentIdx = idx;
        if (!isa<IndexType>(idxElemType)) {
          currentIdx = arith::IndexCastOp::create(nestedBuilder, nestedLoc,
                                                  idxElemType, idx);
        }

        // Inline the comparator region.
        IRMapping regionMap;
        regionMap.map(srcBlock.getArgument(0), inputElem);
        regionMap.map(srcBlock.getArgument(1), accVal);
        for (Operation &op : srcBlock.without_terminator()) {
          nestedBuilder.clone(op, regionMap);
        }

        auto yieldOp = cast<IREE::LinalgExt::YieldOp>(srcBlock.getTerminator());
        Value cmp = regionMap.lookup(yieldOp.getOperand(0));

        Value newVal = arith::SelectOp::create(nestedBuilder, nestedLoc, cmp,
                                               inputElem, accVal);
        Value newIdx = arith::SelectOp::create(nestedBuilder, nestedLoc, cmp,
                                               currentIdx, accIdx);
        linalg::YieldOp::create(nestedBuilder, nestedLoc,
                                ValueRange{newVal, newIdx});
      });

  rewriter.replaceOp(argCompareOp, genericOp.getResults());
  return genericOp;
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
