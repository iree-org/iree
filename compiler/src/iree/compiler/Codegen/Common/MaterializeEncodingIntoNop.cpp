// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

using namespace IREE::LinalgExt;

namespace {
struct MaterializeEncodingIntoNopPass
    : public MaterializeEncodingIntoNopBase<MaterializeEncodingIntoNopPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, func::FuncDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto operation = getOperation();

    auto materializeEncodingFn =
        [](RankedTensorType tensorType) -> FailureOr<MaterializeEncodingInfo> {
      return failure();
    };
    auto materializeEncodingValueFn =
        [](RankedTensorType, OpBuilder &,
           Location) -> FailureOr<MaterializeEncodingValueInfo> {
      return failure();
    };

    RewritePatternSet materializeEncodingPattern(context);
    MaterializeEncodingTypeConverter typeConverter(materializeEncodingFn);
    MaterializeEncodingConversionTarget target(*context);
    populateMaterializeEncodingIntoPackUnPackPatterns(
        materializeEncodingPattern, target, typeConverter,
        materializeEncodingValueFn);

    if (failed(applyPartialConversion(operation, target,
                                      std::move(materializeEncodingPattern)))) {
      operation.emitOpError("materialization failed");
      return signalPassFailure();
    }

    {
      RewritePatternSet patterns(context);
      populateMaterializeUpperBoundTileSizePatterns(patterns,
                                                    materializeEncodingFn);
      if (failed(
              applyPatternsAndFoldGreedily(operation, std::move(patterns)))) {
        operation.emitOpError(
            "encoding padding sizes materialization pattern failed");
        return signalPassFailure();
      }
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

std::unique_ptr<OperationPass<func::FuncOp>>
createMaterializeEncodingIntoNopPass() {
  return std::make_unique<MaterializeEncodingIntoNopPass>();
}

} // namespace mlir::iree_compiler
