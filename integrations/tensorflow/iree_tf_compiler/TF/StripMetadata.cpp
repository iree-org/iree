// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/TF/Passes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace iree_integrations {
namespace TF {

static bool isTFAttr(NamedAttribute &namedAttr) {
  auto name = namedAttr.getName().strref();
  if (name.startswith("tf.") || name.startswith("tf_")) {
    return true;
  }
  StringRef attrNamespace = namedAttr.getValue().getDialect().getNamespace();
  return attrNamespace == mlir::TF::TensorFlowDialect::getDialectNamespace() ||
         attrNamespace == mlir::tf_executor::TensorFlowExecutorDialect::
                              getDialectNamespace() ||
         attrNamespace ==
             mlir::tf_device::TensorFlowDeviceDialect::getDialectNamespace() ||
         attrNamespace == mlir::tf_saved_model::TensorFlowSavedModelDialect::
                              getDialectNamespace();
}

class StripModuleMetadataPass
    : public PassWrapper<StripModuleMetadataPass, OperationPass<ModuleOp>> {
 public:
  StringRef getArgument() const override {
    return "iree-tf-strip-module-metadata";
  }

  StringRef getDescription() const override {
    return "Remove unneeded TensorFlow attributes from module ops";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
        moduleOp->getAttrs(),
        [](NamedAttribute namedAttr) { return isTFAttr(namedAttr); }));
    for (auto namedAttr : stripAttrs) {
      moduleOp->removeAttr(namedAttr.getName());
    }
  }
};

class StripFunctionMetadataPass
    : public PassWrapper<StripFunctionMetadataPass,
                         OperationPass<func::FuncOp>> {
 public:
  StringRef getArgument() const override {
    return "iree-tf-strip-function-metadata";
  }

  StringRef getDescription() const override {
    return "Remove unneeded TensorFlow attributes from func ops";
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
        funcOp->getAttrs(),
        [](NamedAttribute namedAttr) { return isTFAttr(namedAttr); }));
    for (auto namedAttr : stripAttrs) {
      funcOp->removeAttr(namedAttr.getName());
    }

    for (int i = 0; i < funcOp.getNumArguments(); ++i) {
      auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
          mlir::function_interface_impl::getArgAttrs(funcOp, i),
          [](NamedAttribute namedAttr) { return isTFAttr(namedAttr); }));
      for (auto namedAttr : stripAttrs) {
        funcOp.removeArgAttr(i, namedAttr.getName());
      }
    }

    for (int i = 0; i < funcOp.getNumResults(); ++i) {
      auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
          mlir::function_interface_impl::getResultAttrs(funcOp, i),
          [](NamedAttribute namedAttr) { return isTFAttr(namedAttr); }));
      for (auto namedAttr : stripAttrs) {
        funcOp.removeResultAttr(i, namedAttr.getName());
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createStripModuleMetadataPass() {
  return std::make_unique<StripModuleMetadataPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> createStripFunctionMetadataPass() {
  return std::make_unique<StripFunctionMetadataPass>();
}

static PassRegistration<StripModuleMetadataPass> modulePass;

static PassRegistration<StripFunctionMetadataPass> funcPass;

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir
