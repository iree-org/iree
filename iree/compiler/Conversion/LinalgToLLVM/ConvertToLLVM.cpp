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
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

class ConvertTieShapePattern : public ConvertToLLVMPattern {
 public:
  explicit ConvertTieShapePattern(MLIRContext *context,
                                  LLVMTypeConverter &lowering_)
      : ConvertToLLVMPattern(Shape::TieShapeOp::getOperationName(), context,
                             lowering_) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto tieShapeOp = cast<Shape::TieShapeOp>(op);
    auto loc = tieShapeOp.getLoc();
    auto tieShapeMemrefType =
        tieShapeOp.operand().getType().dyn_cast_or_null<MemRefType>();
    if (!tieShapeMemrefType) return failure();
    auto targetDescTy = typeConverter.convertType(tieShapeMemrefType)
                            .dyn_cast_or_null<LLVM::LLVMType>();
    auto targetElementTy =
        typeConverter.convertType(tieShapeMemrefType.getElementType())
            .dyn_cast<LLVM::LLVMType>();
    if (!targetDescTy) return failure();

    // Update memory descriptor information.
    MemRefDescriptor sourceMemRef(operands.front());

    auto makeRankedShapeOp =
        cast<Shape::MakeRankedShapeOp>(tieShapeOp.shape().getDefiningOp());

    // Update descriptor shape information.
    for (int i = 0; i < makeRankedShapeOp.dynamic_dimensions().size(); ++i) {
      auto dynamicDims = makeRankedShapeOp.dynamic_dimensions()[i];
      sourceMemRef.setSize(rewriter, loc, i, dynamicDims);
    }

    rewriter.replaceOp(tieShapeOp, {sourceMemRef});
    rewriter.eraseOp(makeRankedShapeOp);
    return success();
  }
};  // namespace iree_compiler

class ConvertRankedDimPattern : public ConvertToLLVMPattern {
 public:
  explicit ConvertRankedDimPattern(MLIRContext *context,
                                   LLVMTypeConverter &lowering_)
      : ConvertToLLVMPattern(Shape::RankedDimOp::getOperationName(), context,
                             lowering_) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto rankedDimOp = cast<Shape::RankedDimOp>(op);
    auto makeRankedShapeOp =
        cast<Shape::MakeRankedShapeOp>(rankedDimOp.shape().getDefiningOp());
    auto dimIndex = rankedDimOp.index();
    auto dynamicDims =
        makeRankedShapeOp.dynamic_dimensions()[*dimIndex.getRawData()];
    rewriter.replaceOp(op, dynamicDims);
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
  // Convert to the LLVM IR dialect using the converter defined above.
  OwningRewritePatternList patterns;
  LLVMTypeConverter converter(&getContext());
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateVectorToSCFConversionPatterns(patterns, &getContext());
  populateVectorToLLVMMatrixConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(converter, patterns);
  patterns.insert<ConvertRankedDimPattern, ConvertTieShapePattern>(
      &getContext(), converter);
  populateLinalgToLLVMConversionPatterns(converter, patterns, &getContext());

  LLVMConversionTarget target(getContext());
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  if (failed(applyPartialConversion(module, target, patterns, &converter)))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertToLLVMPass() {
  return std::make_unique<ConvertToLLVMPass>();
}

static PassRegistration<ConvertToLLVMPass> pass(
    "iree-codegen-convert-to-llvm", "Convert Linalg to LLVM",
    [] { return std::make_unique<ConvertToLLVMPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
