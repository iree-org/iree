// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/UKernelOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/Util/Conversion/MemRefToUtil/Patterns.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/VMVX/Conversion/HALToVMVX/ConvertHALToVMVX.h"
#include "iree/compiler/Dialect/VMVX/Conversion/StandardToVMVX/ConvertStandardToVMVX.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXTypes.h"
#include "iree/compiler/Dialect/VMVX/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/VMVX/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VMVX {

namespace {

// Make `iree_codegen.get_base_pointer` a no-op during VMVX lowering.
struct ConvertGetBasePointerOp
    : public OpConversionPattern<IREE::Codegen::GetBasePointerOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::Codegen::GetBasePointerOp basePointerOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(basePointerOp, adaptor.getSource());
    return success();
  }
};

// Runs conversion with registered input dialects.
class ConversionPass : public ConversionBase<ConversionPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect, IREE::HAL::HALDialect,
                    IREE::VM::VMDialect, IREE::VMVX::VMVXDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();

    TypeConverter typeConverter;

    typeConverter.addConversion([](Type type) { return type; });

    // Run a pre-pass that updates the entry function signature.
    for (auto funcOp : getOperation().getOps<func::FuncOp>()) {
      if (funcOp.isPublic()) {
        if (failed(updateHALToVMVXEntryFuncOp(funcOp, typeConverter))) {
          return signalPassFailure();
        }
      }
    }

    // Ensure all input dialects go away.
    ConversionTarget conversionTarget(*context);
    conversionTarget.addIllegalDialect<tensor::TensorDialect>();
    conversionTarget.addIllegalDialect<IREE::Codegen::IREECodegenDialect>();
    conversionTarget.addLegalDialect<IREE::Util::UtilDialect>();
    conversionTarget.addLegalDialect<IREE::VMVX::VMVXDialect>();
    conversionTarget
        .addLegalDialect<mlir::func::FuncDialect, mlir::scf::SCFDialect,
                         mlir::arith::ArithDialect>();
    conversionTarget.addLegalDialect<mlir::AffineDialect>();
    conversionTarget.addLegalDialect<memref::MemRefDialect>();
    conversionTarget.addIllegalOp<mlir::UnrealizedConversionCastOp>();

    RewritePatternSet patterns(&getContext());
    populateHALToVMVXPatterns(context, conversionTarget, patterns,
                              typeConverter);
    populateStandardToVMVXPatterns(context, patterns, typeConverter);
    populateMemRefToUtilPatterns(context, conversionTarget, typeConverter,
                                 patterns,
                                 IREE::Util::BufferType::get(&getContext()));
    patterns.insert<ConvertGetBasePointerOp>(typeConverter, context);

    // Use the default 64-bit lowering for TOSA's ApplyScale operator:
    //   This lowering widens integer types to 64-bit an performs the non-fused
    //   operations, specifically multiply, add, and shift. Bit-widening
    //   is used to guarantee higher-order bits are not truncated during the
    //   multiply or add.
    //
    // TODO(suderman): remove the TOSA layering violation and lower to standard/
    // math ops instead.
    tosa::populateTosaRescaleToArithConversionPatterns(&patterns);

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(patterns)))) {
      getOperation().emitError() << "conversion to the VMVX dialect failed";
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createConversionPass() {
  return std::make_unique<ConversionPass>();
}

}  // namespace VMVX
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
