// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/TFL/PassDetail.h"
#include "iree_tf_compiler/TFL/Passes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace iree_integrations {
namespace TFL {
namespace {

class RetainCallOnceFuncsPass
    : public RetainCallOnceFuncsBase<RetainCallOnceFuncsPass> {
 public:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::TFL::TensorFlowLiteDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    llvm::DenseMap<StringRef, FuncOp> funcMap;
    for (auto func : moduleOp.getOps<mlir::FuncOp>()) {
      funcMap[func.sym_name()] = func;
    }

    for (auto func : moduleOp.getOps<mlir::FuncOp>()) {
      for (auto callOnce : func.getOps<mlir::TFL::CallOnceOp>()) {
        auto callFunc = funcMap[callOnce.session_init_function()];
        callOnce->setAttr("session_init_function_symbol",
                          SymbolRefAttr::get(callFunc));
      }
    }
  }
};

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> createRetainCallOnceFuncsPass() {
  return std::make_unique<RetainCallOnceFuncsPass>();
}

}  // namespace TFL
}  // namespace iree_integrations
}  // namespace mlir
