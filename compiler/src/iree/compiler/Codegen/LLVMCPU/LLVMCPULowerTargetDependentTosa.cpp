// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// Convert target-dependent TOSA ops to Arith.
class LLVMCPULowerTargetDependentTosaPass
    : public LLVMCPULowerTargetDependentTosaBase<
          LLVMCPULowerTargetDependentTosaPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithmeticDialect>();
  }

  void runOnOperation() override;
};

void LLVMCPULowerTargetDependentTosaPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  // Use the default 64-bit lowering for TOSA's ApplyScale operator:
  // This lowering widens integer types to 64-bit an performs the non-fused
  // operations, specifically multiply, add, and shift. Bit-widening
  // is used to guarantee higher-order bits are not truncated during the
  // multiply or add.
  bool use32BitImpl = false;
  auto variantOp = getExecutableVariantOp(funcOp);
  if (succeeded(variantOp) && isRISCV(*variantOp)) {
    // Use the 32-bit lowering for RISC-V if 'zve32x' is specified and there is
    // no 64-bit integer vector support.
    use32BitImpl = hasZve32xFeature(*variantOp) && !hasVFeature(*variantOp) &&
                   !hasZve64xFeature(*variantOp);
  }

  RewritePatternSet patterns(&getContext());
  tosa::populateTosaRescaleToArithConversionPatterns(&patterns, use32BitImpl);

  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithmeticDialect>();
  target.addIllegalDialect<tosa::TosaDialect>();

  if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
    signalPassFailure();
    return;
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPULowerTargetDependentTosaPass() {
  return std::make_unique<LLVMCPULowerTargetDependentTosaPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
