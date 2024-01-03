// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/Input/InputOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

struct IREECheckKnownDialectsPass
    : public IREECheckKnownDialectsBase<IREECheckKnownDialectsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Input::IREEInputDialect, IREE::Flow::FlowDialect,
                    IREE::HAL::HALDialect, IREE::Util::UtilDialect,
                    mlir::func::FuncDialect, mlir::arith::ArithDialect>();
  }
  void runOnOperation() override {
    auto operation = getOperation();
    llvm::DenseSet<Operation *> errors;
    operation.walk([&](Operation *op) {
      if (!op->getDialect()) {
        errors.insert(op);
      }
    });

    InFlightDiagnostic errorDiag =
        emitError(getOperation().getLoc())
        << "one or more unknown operations were found in the compiler input "
           "(did you mean to pre-process through an IREE importer frontend?)";

    for (auto op : errors) {
      Diagnostic &note = errorDiag.attachNote(op->getLoc());
      note.append(op->getName());
    }

    signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createIREECheckKnownDialectsPass() {
  return std::make_unique<IREECheckKnownDialectsPass>();
}

} // namespace mlir::iree_compiler
