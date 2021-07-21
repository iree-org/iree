// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/TFL/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_integrations {
namespace TFL {

static bool isTFLAttr(NamedAttribute &namedAttr) {
  // NOTE: tflite mixes tf and tfl, for some reason.
  auto name = namedAttr.first.strref();
  if (name.startswith("tf.") || name.startswith("tf_") ||
      name.startswith("tfl.") || name.startswith("tfl_")) {
    return true;
  }
  StringRef attrNamespace = namedAttr.second.getDialect().getNamespace();
  return attrNamespace == "tf" || attrNamespace == "tfl";
}

class StripModuleMetadataPass
    : public PassWrapper<StripModuleMetadataPass, OperationPass<ModuleOp>> {
 public:
  StringRef getArgument() const override {
    return "iree-tflite-strip-module-metadata";
  }

  StringRef getDescription() const override {
    return "Remove unneeded TFLite attributes from module ops";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
        moduleOp->getAttrs(),
        [](NamedAttribute namedAttr) { return isTFLAttr(namedAttr); }));
    for (auto namedAttr : stripAttrs) {
      moduleOp->removeAttr(namedAttr.first);
    }
  }
};

class StripFunctionMetadataPass
    : public PassWrapper<StripFunctionMetadataPass, OperationPass<FuncOp>> {
 public:
  StringRef getArgument() const override {
    return "iree-tflite-strip-function-metadata";
  }

  StringRef getDescription() const override {
    return "Remove unneeded TFLite attributes from func ops";
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
        funcOp->getAttrs(),
        [](NamedAttribute namedAttr) { return isTFLAttr(namedAttr); }));
    for (auto namedAttr : stripAttrs) {
      funcOp->removeAttr(namedAttr.first);
    }

    for (int i = 0; i < funcOp.getNumArguments(); ++i) {
      auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
          funcOp.getArgAttrs(i),
          [](NamedAttribute namedAttr) { return isTFLAttr(namedAttr); }));
      for (auto namedAttr : stripAttrs) {
        funcOp.removeArgAttr(i, namedAttr.first);
      }
    }

    for (int i = 0; i < funcOp.getNumResults(); ++i) {
      auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
          funcOp.getResultAttrs(i),
          [](NamedAttribute namedAttr) { return isTFLAttr(namedAttr); }));
      for (auto namedAttr : stripAttrs) {
        funcOp.removeResultAttr(i, namedAttr.first);
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createStripModuleMetadataPass() {
  return std::make_unique<StripModuleMetadataPass>();
}

std::unique_ptr<OperationPass<FuncOp>> createStripFunctionMetadataPass() {
  return std::make_unique<StripFunctionMetadataPass>();
}

static PassRegistration<StripModuleMetadataPass> modulePass;
static PassRegistration<StripFunctionMetadataPass> funcPass;

}  // namespace TFL
}  // namespace iree_integrations
}  // namespace mlir
