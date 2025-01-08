// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Parser/Parser.h"
#include "mlir/AsmParser/AsmParser.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"

#define DEBUG_TYPE "iree-global-opt-pack-storage"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_PACKSTORAGEPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

static bool isPackStorageCandidate(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    auto elementType = tensorType.getElementType();
    return elementType.isIntOrFloat() &&
           elementType.getIntOrFloatBitWidth() == 1;
  }
  return false;
}

static RankedTensorType appendAttributeToTensor(RankedTensorType type) {
  if (!isPackStorageCandidate(type))
    return type;
  IREE::Encoding::PackedStorageAttr packedAttr = IREE::Encoding::PackedStorageAttr::get(type.getContext());
  auto newType =
      RankedTensorType::get(type.getShape(), type.getElementType(),
                            packedAttr);
  LLVM_DEBUG(llvm::dbgs() << " appending packed tensor type: " << newType
                          << "\n");
  return newType;
}

class PackedStorageConverter : public TypeConverter {
public:
  explicit PackedStorageConverter() {
    addConversion([](RankedTensorType ty) -> Type {
      if (isPackStorageCandidate(ty))
        return appendAttributeToTensor(ty);
      return ty;
    });
  }
};

struct PackAttributeSignaturePattern
    : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto converter = getTypeConverter();
    //auto &funcBlock = funcOp.getFunctionBody().front();
    TypeConverter::SignatureConversion newSignature(funcOp.getNumArguments());

    for (const auto [index, arg] : llvm::enumerate(funcOp.getArguments())) {
      if (true || isPackStorageCandidate(arg.getType())) {
        auto tensorType = cast<RankedTensorType>(arg.getType());
        newSignature.addInputs(index, appendAttributeToTensor(tensorType));
      }
    }
    //rewriter.applySignatureConversion(&funcBlock, newSignature, converter);
    rewriter.modifyOpInPlace(funcOp, [&] {
      funcOp.setType(rewriter.getFunctionType(newSignature.getConvertedTypes(),
                                              funcOp.getFunctionType().getResults()));
    });

    // Create a new FuncOp with the modified signature
    auto newFuncType =
        rewriter.getFunctionType(newSignature.getConvertedTypes(),
                                 funcOp.getFunctionType().getResults());
    auto newFuncOp = rewriter.create<func::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), newFuncType);

      // Copy attributes from the old FuncOp to the new one
    for (const auto &namedAttr : funcOp->getAttrs()) {
      if (namedAttr.getName() != "function_type")
        newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    // Convert the function body
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *converter,
                                           &newSignature)))
      return failure();

    // Replace the old FuncOp with the new one
    rewriter.replaceOp(funcOp, newFuncOp);
    rewriter.eraseOp(funcOp);
    return success();
  }
};

struct PackStoragePass : impl::PackStoragePassBase<PackStoragePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, mlir::iree_compiler::IREE::Encoding::IREEEncodingDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "== Running PackStoragePass on "
                            << funcOp->getName() << "\n");
    auto context = &getContext();

    PackedStorageConverter typeConverter;

    RewritePatternSet conversionPatterns(context);
    conversionPatterns.add<PackAttributeSignaturePattern>(typeConverter,
                                                          context);

    ConversionTarget target(*context);
    if (failed(applyPartialConversion(funcOp, target,
                                      std::move(conversionPatterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::iree_compiler::GlobalOptimization
