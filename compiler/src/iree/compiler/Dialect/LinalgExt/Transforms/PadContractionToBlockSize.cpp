// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/Input/InputOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/PassDetail.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
namespace IREE = mlir::iree_compiler::IREE;
using namespace IREE::LinalgExt;

static Operation *sliceTensor(Location loc, Value expanded, Value original,
                              OpBuilder &builder) {
  SmallVector<OpFoldResult> sizes =
      tensor::getMixedSizes(builder, loc, original);
  SmallVector<OpFoldResult> offsets(sizes.size(), builder.getI64IntegerAttr(0));
  SmallVector<OpFoldResult> strides(sizes.size(), builder.getI64IntegerAttr(1));
  return builder.create<tensor::ExtractSliceOp>(loc, expanded, offsets, sizes,
                                                strides);
}

static bool padTensor(Location loc, OpOperand *operand,
                      ArrayRef<int64_t> alignments, OpBuilder &builder) {
  Value original = operand->get();
  auto type = original.getType().cast<RankedTensorType>();
  ArrayRef<int64_t> shape = type.getShape();
  assert(shape.size() == alignments.size() &&
         "expected shape and alignments to match");

  // New dimensions.
  SmallVector<int64_t> newStaticDims;
  newStaticDims.resize(shape.size(), ShapedType::kDynamic);
  SmallVector<OpFoldResult> newPaddingSizes(shape.size(),
                                            builder.getI64IntegerAttr(0));

  // Compute padded dims.
  bool needsPad = false;
  for (int i = 0, e = shape.size(); i < e; ++i) {
    auto inputDim = shape[i];
    auto alignment = alignments[i];
    if (inputDim >= 0) {
      // Static dim.
      if ((inputDim % alignment) == 0) {
        newStaticDims[i] = inputDim;
        continue;
      }
      int64_t alignedDim = (inputDim + (alignment - 1)) & ~(alignment - 1);
      newStaticDims[i] = alignedDim;
      newPaddingSizes[i] = builder.getI64IntegerAttr(alignedDim - inputDim);
      needsPad = true;
    } else {
      // Dynamic dim.
      Value inputDimValue = builder.create<tensor::DimOp>(loc, original, i);
      Value alignedDim =
          builder.create<IREE::Input::AlignOp>(loc, inputDimValue, alignment);
      newPaddingSizes[i] = alignedDim;
      needsPad = true;
    }
  }
  if (!needsPad)
    return false;

  auto resultType = RankedTensorType::get(newStaticDims, type.getElementType());
  Value zeroConstant = builder.create<arith::ConstantOp>(
      loc, builder.getZeroAttr(type.getElementType()));
  SmallVector<OpFoldResult> zeroStaticLow(shape.size(),
                                          builder.getI64IntegerAttr(0));
  SmallVector<Value> nullLow;
  Value padded = builder.create<tensor::PadOp>(loc, resultType, operand->get(),
                                               zeroStaticLow, newPaddingSizes,
                                               zeroConstant);
  operand->set(padded);
  return true;
}

namespace {

struct PadContractionToBlockSizePass
    : public PadContractionToBlockSizeBase<PadContractionToBlockSizePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Input::IREEInputDialect>();
  }

  void runOnOperation() override {
    getOperation()->walk([&](linalg::ContractionOpInterface op) {
      auto linalgOp = llvm::cast<linalg::LinalgOp>(op.getOperation());
      Location loc = op.getLoc();
      OpOperand *lhs = linalgOp.getDpsInputOperand(0);
      OpOperand *rhs = linalgOp.getDpsInputOperand(1);
      OpOperand *output = linalgOp.getDpsInitOperand(0);
      Value origOutput = output->get();
      OpResult result = op.getOperation()->getResult(0);

      bool insertSlice = false;
      OpBuilder builder(op.getOperation());
      if (op.isRowMajorMatmul()) {
        padTensor(loc, lhs, {rowAlignment, rowAlignment}, builder);
        padTensor(loc, rhs, {rowAlignment, columnAlignment}, builder);
        if (padTensor(loc, output, {rowAlignment, columnAlignment}, builder)) {
          result.setType(output->get().getType());
          insertSlice = true;
        }
      }

      // Insert an appropriate extract.
      if (insertSlice) {
        builder.setInsertionPointAfter(op.getOperation());
        Operation *slicedResult = sliceTensor(loc, result, origOutput, builder);
        result.replaceAllUsesExcept(slicedResult->getResult(0), slicedResult);
      }

      return WalkResult::advance();
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<>>
IREE::LinalgExt::createPadContractionToBlockSizePass() {
  return std::make_unique<PadContractionToBlockSizePass>();
}
