// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#define DEBUG_TYPE "iree-global-opt-pack-storage"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_PACKSTORAGEPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"


static RankedTensorType appendAttributeToTensor(RankedTensorType type) {
  IntegerAttr bitwidthAttr =
      IntegerAttr::get(IntegerType::get(type.getContext(), 32),
                       type.getElementType().getIntOrFloatBitWidth());
  IREE::Encoding::PackedStorageAttr packedAttr =
      IREE::Encoding::PackedStorageAttr::get(type.getContext(), bitwidthAttr);
  auto newType = RankedTensorType::get(type.getShape(), type.getElementType(),
                               packedAttr);
  assert(mlir::iree_compiler::IREE::Encoding::hasPackedStorageAttr(newType));
  LLVM_DEBUG(llvm::dbgs() << " appending packed tensor type: " << newType << "\n");
  return newType;
}

struct PackAttributeSignaturePattern : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    TypeConverter::SignatureConversion convertedResult(
        funcOp.getNumArguments());
    if (failed(getTypeConverter()->convertSignatureArgs(
            funcOp.getArgumentTypes(), convertedResult)))
      return failure();
    rewriter.modifyOpInPlace(funcOp, [&] {
      rewriter.applySignatureConversion(&funcOp.getFunctionBody().front(),
                                        convertedResult);
    });
    return success();
  }
};


struct PackStoragePass : impl::PackStoragePassBase<PackStoragePass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect>();
  }
  void runOnOperation() override;

  static bool isPackStorageCandidate(RankedTensorType type) {
    auto elementType = type.getElementType();
    return elementType.isIntOrFloat() &&
           elementType.getIntOrFloatBitWidth() == 1;
  }
};

void PackStoragePass::runOnOperation() {
  auto funcOp = getOperation();
  LLVM_DEBUG(llvm::dbgs() << "== Running PackStoragePass on "
                          << funcOp.getName() << "\n");
  RewritePatternSet conversionPatterns(&getContext());
  conversionPatterns.add<PackAttributeSignaturePattern>(&getContext());

  if (failed(applyPatternsAndFoldGreedily(funcOp,
                                          std::move(conversionPatterns)))) {
    signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::GlobalOptimization
