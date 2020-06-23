// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

// Upateds memref descriptors shape and strides informations and fold tie_shape
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
    for (int i = 0; i < shape.size(); ++i) {
      if (shape[i] == ShapedType::kDynamicSize) {
        sourceMemRef.setSize(rewriter, loc, i,
                             makeRankedShapeOp.dynamic_dimensions()[i]);
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
};  // namespace iree_compiler

// Replace RankedDimOp with resolved index.
class ConvertRankedDimPattern : public ConvertToLLVMPattern {
 public:
  explicit ConvertRankedDimPattern(MLIRContext *context,
                                   LLVMTypeConverter &typeconverter)
      : ConvertToLLVMPattern(Shape::RankedDimOp::getOperationName(), context,
                             typeconverter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto rankedDimOp = dyn_cast_or_null<Shape::RankedDimOp>(op);
    if (!rankedDimOp) return failure();
    auto makeRankedShapeOp = dyn_cast_or_null<Shape::MakeRankedShapeOp>(
        rankedDimOp.shape().getDefiningOp());
    if (!makeRankedShapeOp) return failure();
    auto dimIndex = rankedDimOp.index();
    auto dynamicDims =
        makeRankedShapeOp.dynamic_dimensions()[dimIndex.getSExtValue()];
    rewriter.replaceOp(op, dynamicDims);
    return success();
  }
};

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

namespace {
struct ConvertToLLVMPass
    : public PassWrapper<ConvertToLLVMPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;
};

}  // namespace

void ConvertToLLVMPass::runOnOperation() {
  auto module = getOperation();
  OwningRewritePatternList patterns;
  LLVMTypeConverter converter(&getContext());
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());
  populateExpandTanhPattern(patterns, &getContext());
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateVectorToSCFConversionPatterns(patterns, &getContext());
  populateVectorToLLVMMatrixConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(converter, patterns);
  populateLinalgToLLVMConversionPatterns(converter, patterns, &getContext());
  // The following patterns resolves dynamic shapes by substituting tie_shape
  // ops with an updated memref descriptors and replacing RankDimOp with actual
  // index loaded from memref<?xi32> that holds all dynamic shapes
  // push constants.
  patterns.insert<ConvertRankedDimPattern, ConvertTieShapePattern,
                  RemoveMakeRankedShape>(&getContext(), converter);
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  if (failed(applyPartialConversion(module, target, patterns)))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertToLLVMPass() {
  return std::make_unique<ConvertToLLVMPass>();
}

static PassRegistration<ConvertToLLVMPass> pass(
    "iree-codegen-convert-to-llvm",
    "Perform final conversion from Linalg/HAL/Shape/Vector/Standard to LLVMIR "
    "dialect",
    [] { return std::make_unique<ConvertToLLVMPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
