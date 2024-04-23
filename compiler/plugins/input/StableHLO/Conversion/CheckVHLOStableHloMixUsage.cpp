// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/PassDetail.h"
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"

namespace mlir::iree_compiler::stablehlo {
#define GEN_PASS_DEF_CHECKVHLOSTABLEHLOMIXUSAGE
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h.inc"

namespace {
struct CheckVHLOStableHloMixUsage final
    : impl::CheckVHLOStableHloMixUsageBase<CheckVHLOStableHloMixUsage> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    auto modOp = getOperation();
    Operation *lastStablehloOp = nullptr;
    Operation *lastVhloOp = nullptr;
    bool errorsFound = false;
    auto emitError = [&](Operation *vhloOp, Operation *stablehloOp) {
      vhloOp->emitOpError()
          << "Using VHLO and StableHLO Ops in the same module "
             "is not supported. ";
      stablehloOp->emitRemark() << "Last Stablehlo Op was found here: ";
      errorsFound = true;
    };
    modOp->walk([&](Operation *op) {
      auto opDialect = op->getDialect();
      if (opDialect == ctx->getLoadedDialect("stablehlo")) {
        if (lastVhloOp) {
          emitError(lastVhloOp, op);
          return WalkResult::interrupt();
        }
        lastStablehloOp = op;
      } else if (opDialect == ctx->getLoadedDialect("vhlo")) {
        if (lastStablehloOp) {
          emitError(op, lastStablehloOp);
          return WalkResult::interrupt();
        }
        lastVhloOp = op;
      }
      return WalkResult::advance();
    });
    if (errorsFound) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler::stablehlo
