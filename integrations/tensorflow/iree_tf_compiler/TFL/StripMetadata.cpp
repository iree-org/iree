// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/TFL/PassDetail.h"
#include "iree_tf_compiler/TFL/Passes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_integrations {
namespace TFL {
namespace {

static bool isTFLAttr(NamedAttribute &namedAttr) {
  // NOTE: tflite mixes tf and tfl, for some reason.
  auto name = namedAttr.getName().strref();
  // Don't trim attributes from tf_saved_model---they carry ABI information.
  if (name.startswith("tf_saved_model.")) return false;

  if (name.startswith("tf.") || name.startswith("tf_") ||
      name.startswith("tfl.") || name.startswith("tfl_")) {
    return true;
  }
  StringRef attrNamespace = namedAttr.getValue().getDialect().getNamespace();
  return attrNamespace == "tf" || attrNamespace == "tfl";
}

class StripModuleMetadataPass
    : public StripModuleMetadataBase<StripModuleMetadataPass> {
 public:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
        moduleOp->getAttrs(),
        [](NamedAttribute namedAttr) { return isTFLAttr(namedAttr); }));
    for (auto namedAttr : stripAttrs) {
      moduleOp->removeAttr(namedAttr.getName());
    }
  }
};

class StripFunctionMetadataPass
    : public StripFunctionMetadataBase<StripFunctionMetadataPass> {
 public:
  void runOnOperation() override {
    auto funcOp = getOperation();
    auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
        funcOp->getAttrs(),
        [](NamedAttribute namedAttr) { return isTFLAttr(namedAttr); }));
    for (auto namedAttr : stripAttrs) {
      funcOp->removeAttr(namedAttr.getName());
    }

    for (int i = 0; i < funcOp.getNumArguments(); ++i) {
      auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
          mlir::function_interface_impl::getArgAttrs(funcOp, i),
          [](NamedAttribute namedAttr) { return isTFLAttr(namedAttr); }));
      for (auto namedAttr : stripAttrs) {
        funcOp.removeArgAttr(i, namedAttr.getName());
      }
    }

    for (int i = 0; i < funcOp.getNumResults(); ++i) {
      auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
          mlir::function_interface_impl::getResultAttrs(funcOp, i),
          [](NamedAttribute namedAttr) { return isTFLAttr(namedAttr); }));
      for (auto namedAttr : stripAttrs) {
        funcOp.removeResultAttr(i, namedAttr.getName());
      }
    }
  }
};

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> createStripModuleMetadataPass() {
  return std::make_unique<StripModuleMetadataPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> createStripFunctionMetadataPass() {
  return std::make_unique<StripFunctionMetadataPass>();
}

}  // namespace TFL
}  // namespace iree_integrations
}  // namespace mlir
