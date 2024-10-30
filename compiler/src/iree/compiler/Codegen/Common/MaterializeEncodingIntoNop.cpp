// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_MATERIALIZEENCODINGINTONOPPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

using namespace IREE::Encoding;

namespace {

struct FoldAwayExtractSliceOnUnsetEncodingPattern
    : public OpConversionPattern<tensor::ExtractSliceOp> {
  using OpConversionPattern<tensor::ExtractSliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto unsetEncodingOp =
        op.getSource().getDefiningOp<IREE::Encoding::UnsetEncodingOp>();
    if (!unsetEncodingOp) {
      return rewriter.notifyMatchFailure(op,
                                         "source is not an unset_encoding op");
    }
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};

struct MaterializeEncodingIntoNopPass final
    : impl::MaterializeEncodingIntoNopPassBase<MaterializeEncodingIntoNopPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto operation = getOperation();

    auto materializeEncodingFn = [](RankedTensorType,
                                    IREE::HAL::ExecutableTargetAttr)
        -> FailureOr<MaterializeEncodingInfo> { return failure(); };
    auto materializeEncodingValueFn =
        [](RankedTensorType, OpBuilder &,
           Location) -> FailureOr<MaterializeEncodingValueInfo> {
      return failure();
    };

    RewritePatternSet materializeEncodingPattern(context);
    MaterializeEncodingTypeConverter typeConverter(
        materializeEncodingFn, IREE::HAL::ExecutableTargetAttr(),
        /*transposeNarrowN=*/false);
    MaterializeEncodingConversionTarget target(*context);
    populateMaterializeEncodingIntoPackUnPackPatterns(
        materializeEncodingPattern, typeConverter, materializeEncodingValueFn);
    populateShapeIndependentMaterializeEncodingPatterns(
        materializeEncodingPattern, target, typeConverter,
        materializeEncodingValueFn);

    // Fold the extract_slice ops away if the source is from unset_encoding op.
    target.addDynamicallyLegalOp<tensor::ExtractSliceOp>(
        [](tensor::ExtractSliceOp op) {
          return op.getSource().getDefiningOp<IREE::Encoding::UnsetEncodingOp>()
                     ? false
                     : true;
        });
    materializeEncodingPattern
        .insert<FoldAwayExtractSliceOnUnsetEncodingPattern>(context);

    if (failed(applyPartialConversion(operation, target,
                                      std::move(materializeEncodingPattern)))) {
      operation.emitOpError("materialization failed");
      return signalPassFailure();
    }

    // Add patterns to resolve dims ops and cleanups.
    {
      RewritePatternSet patterns(context);
      memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
      context->getOrLoadDialect<tensor::TensorDialect>()
          ->getCanonicalizationPatterns(patterns);
      if (failed(
              applyPatternsAndFoldGreedily(operation, std::move(patterns)))) {
        operation.emitOpError("folding patterns failed");
        return signalPassFailure();
      }
    }
  }
};
} // namespace

void addEncodingToNopPasses(FunctionLikeNest &passManager) {
  passManager.addPass(createMaterializeEncodingIntoNopPass)
      .addPass(createBufferizeCopyOnlyDispatchesPass)
      .addPass(createCanonicalizerPass);
}

} // namespace mlir::iree_compiler
