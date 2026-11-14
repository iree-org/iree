// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_LEGALIZEDEPRECATEDOPSPASS
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h.inc"

namespace {

struct LegalizeDeprecatedOpsPass final
    : impl::LegalizeDeprecatedOpsPassBase<LegalizeDeprecatedOpsPass> {
  void runOnOperation() override {
    // UnaryEinsum has no semantic verifier. The upstream replacement constructs
    // a scalar one and aborts for element types it cannot represent, including
    // quantized types. Check every unary einsum before invoking the upstream
    // pass.
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      WalkResult validation =
          function.walk([](mlir::stablehlo::UnaryEinsumOp op) {
            Type elementType = op.getOperand().getType().getElementType();
            if (isa<IntegerType, FloatType, ComplexType>(elementType)) {
              return WalkResult::advance();
            }
            op.emitOpError("unsupported element type for unary einsum: ")
                << elementType;
            return WalkResult::interrupt();
          });
      if (validation.wasInterrupted()) {
        return signalPassFailure();
      }
    }

    OpPassManager pipeline(ModuleOp::getOperationName());
    pipeline.addNestedPass<func::FuncOp>(
        ::mlir::stablehlo::createStablehloLegalizeDeprecatedOpsPass(
            ::mlir::stablehlo::StablehloLegalizeDeprecatedOpsPassOptions{
                // IREE has dedicated lowerings for both map and rng.
                /*failOnUnusedOps=*/false}));
    if (failed(runPipeline(pipeline, getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::stablehlo
