// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/TF/Passes.h"
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
  auto name = namedAttr.first.strref();
  if (name.startswith("tf.") || name.startswith("tf_")) {
    return true;
  }
  StringRef attrNamespace = namedAttr.second.getDialect().getNamespace();
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
  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
        moduleOp->getAttrs(),
        [](NamedAttribute namedAttr) { return isTFAttr(namedAttr); }));
    for (auto namedAttr : stripAttrs) {
      moduleOp->removeAttr(namedAttr.first);
    }
  }
};

class StripFunctionMetadataPass
    : public PassWrapper<StripFunctionMetadataPass, OperationPass<FuncOp>> {
 public:
  void runOnOperation() override {
    auto funcOp = getOperation();
    auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
        funcOp->getAttrs(),
        [](NamedAttribute namedAttr) { return isTFAttr(namedAttr); }));
    for (auto namedAttr : stripAttrs) {
      funcOp->removeAttr(namedAttr.first);
    }

    for (int i = 0; i < funcOp.getNumArguments(); ++i) {
      auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
          funcOp.getArgAttrs(i),
          [](NamedAttribute namedAttr) { return isTFAttr(namedAttr); }));
      for (auto namedAttr : stripAttrs) {
        funcOp.removeArgAttr(i, namedAttr.first);
      }
    }

    for (int i = 0; i < funcOp.getNumResults(); ++i) {
      auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
          funcOp.getResultAttrs(i),
          [](NamedAttribute namedAttr) { return isTFAttr(namedAttr); }));
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

static PassRegistration<StripModuleMetadataPass> modulePass(
    "iree-tf-strip-module-metadata",
    "Remove unneeded TensorFlow attributes from module ops");

static PassRegistration<StripFunctionMetadataPass> funcPass(
    "iree-tf-strip-function-metadata",
    "Remove unneeded TensorFlow attributes from func ops");

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir
