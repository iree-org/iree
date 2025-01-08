// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
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

static RankedTensorType appendAttributeToTensor(RankedTensorType type) {
  // IntegerAttr bitwidthAttr =
  //     IntegerAttr::get(IntegerType::get(type.getContext(), 32),
  //                      type.getElementType().getIntOrFloatBitWidth());
  IREE::Encoding::PackedStorageAttr packedAttr = IREE::Encoding::PackedStorageAttr::get(type.getContext());
  //    IREE::Encoding::PackedStorageAttr::get(type.getContext(), bitwidthAttr);

  //auto context = type.getContext();
  //size_t numRead = 0;
  //mlir::Attribute packedAttr = mlir::parseAttribute("#iree_encoding.packed_storage", context, Type(), &numRead);
  auto newType =
      RankedTensorType::get(type.getShape(), type.getElementType(),
                            packedAttr);

  //assert(mlir::iree_compiler::IREE::Encoding::hasPackedStorageAttr(newType));
  LLVM_DEBUG(llvm::dbgs() << " appending packed tensor type: " << newType
                          << "\n");
  return newType;
}

class PackedStorageConverter : public TypeConverter {
public:
  explicit PackedStorageConverter() {
    addConversion([](RankedTensorType ty) -> Type {
      return appendAttributeToTensor(ty);
    });
  }
};

struct PackAttributeSignaturePattern
    : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    TypeConverter::SignatureConversion converter(funcOp.getNumArguments());

    for (const auto [index, arg] : llvm::enumerate(funcOp.getArguments())) {
      if (isPackStorageCandidate(arg.getType())) {
        auto tensorType = cast<RankedTensorType>(arg.getType());
        converter.addInputs(index, appendAttributeToTensor(tensorType));
      }
    }
    rewriter.applySignatureConversion(&funcOp.getFunctionBody().front(),
                                      converter);
    // TODO: check this
    rewriter.modifyOpInPlace(funcOp, [&] {
      funcOp.setType(rewriter.getFunctionType(converter.getConvertedTypes(),
                                              std::nullopt));
    });
    return success();
  }

  static bool isPackStorageCandidate(Type type) {
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
      auto elementType = tensorType.getElementType();
      return elementType.isIntOrFloat() &&
             elementType.getIntOrFloatBitWidth() == 1;
    }
    return false;
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

    ConversionTarget target(*context);
    PackedStorageConverter typeConverter;

    RewritePatternSet conversionPatterns(context);
    conversionPatterns.add<PackAttributeSignaturePattern>(typeConverter,
                                                          context);

    if (failed(applyPartialConversion(funcOp, target,
                                      std::move(conversionPatterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::iree_compiler::GlobalOptimization
