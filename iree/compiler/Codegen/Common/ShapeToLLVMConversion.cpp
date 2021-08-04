// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace mlir {
namespace iree_compiler {

namespace {

class RemoveMakeRankedShape : public ConvertToLLVMPattern {
 public:
  explicit RemoveMakeRankedShape(MLIRContext *context,
                                 LLVMTypeConverter &typeconverter)
      : ConvertToLLVMPattern(Shape::MakeRankedShapeOp::getOperationName(),
                             context, typeconverter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Check users are ops are going to be folded away.
    for (auto user : op->getUsers()) {
      if (!cast<Shape::TieShapeOp>(user) && !cast<Shape::RankedDimOp>(user))
        return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// Updates memref descriptors shape and strides informations and fold tie_shape
// into updated memref descriptor.
class ConvertTieShapePattern : public ConvertToLLVMPattern {
 public:
  explicit ConvertTieShapePattern(MLIRContext *context,
                                  LLVMTypeConverter &typeconverter)
      : ConvertToLLVMPattern(Shape::TieShapeOp::getOperationName(), context,
                             typeconverter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto tieShapeOp = cast<Shape::TieShapeOp>(op);
    auto loc = tieShapeOp.getLoc();
    MemRefDescriptor sourceMemRef(operands.front());
    auto makeRankedShapeOp =
        cast<Shape::MakeRankedShapeOp>(tieShapeOp.shape().getDefiningOp());
    auto rankedShapeType = makeRankedShapeOp.shape()
                               .getType()
                               .dyn_cast_or_null<Shape::RankedShapeType>();
    if (!rankedShapeType) return failure();

    auto shape = rankedShapeType.getAllDims();

    // Update memref descriptor shape and strides.
    // dynamic dim maybe in mid location, like shapex.ranked_shape<[128,?,128]
    int dynIdx = 0;
    for (int i = 0; i < shape.size(); ++i) {
      if (shape[i] == ShapedType::kDynamicSize) {
        sourceMemRef.setSize(rewriter, loc, i,
                             makeRankedShapeOp.dynamic_dimensions()[dynIdx++]);
      } else {
        sourceMemRef.setConstantSize(rewriter, loc, i, shape[i]);
      }
    }
    // Compute and update memref descriptor strides. Assumption here is memrefs
    // are row-major e.g following index linearization x[i, j, k] = i * x.dim[1]
    // * x.dim[2] + j * x.dim[2] + k
    sourceMemRef.setConstantStride(rewriter, loc, shape.size() - 1, 1);
    for (int i = shape.size() - 2; i >= 0; --i) {
      auto stride = sourceMemRef.stride(rewriter, loc, i + 1);
      auto dim = sourceMemRef.size(rewriter, loc, i + 1);
      Value strideVal = rewriter.create<LLVM::MulOp>(loc, stride, dim);
      sourceMemRef.setStride(rewriter, loc, i, strideVal);
    }
    rewriter.replaceOp(tieShapeOp, {sourceMemRef});
    return success();
  }
};
}  // namespace

void populateShapeToLLVMConversionPatterns(MLIRContext *context,
                                           TypeConverter *converter,
                                           OwningRewritePatternList &patterns) {
  LLVMTypeConverter &llvmConverter =
      *(static_cast<LLVMTypeConverter *>(converter));
  patterns.insert<ConvertTieShapePattern, RemoveMakeRankedShape>(context,
                                                                 llvmConverter);
  llvmConverter.addConversion(
      [](Shape::RankedShapeType, SmallVectorImpl<Type> &) {
        return success();
      });
}

}  // namespace iree_compiler
}  // namespace mlir
