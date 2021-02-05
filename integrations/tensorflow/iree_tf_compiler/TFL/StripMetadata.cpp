// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
        moduleOp.getAttrs(),
        [](NamedAttribute namedAttr) { return isTFLAttr(namedAttr); }));
    for (auto namedAttr : stripAttrs) {
      moduleOp.removeAttr(namedAttr.first);
    }
  }
};

class StripFunctionMetadataPass
    : public PassWrapper<StripFunctionMetadataPass, OperationPass<FuncOp>> {
 public:
  void runOnOperation() override {
    auto funcOp = getOperation();
    auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
        funcOp.getAttrs(),
        [](NamedAttribute namedAttr) { return isTFLAttr(namedAttr); }));
    for (auto namedAttr : stripAttrs) {
      funcOp.removeAttr(namedAttr.first);
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

static PassRegistration<StripModuleMetadataPass> modulePass(
    "iree-tflite-strip-module-metadata",
    "Remove unneeded TFLite attributes from module ops");

static PassRegistration<StripFunctionMetadataPass> funcPass(
    "iree-tflite-strip-function-metadata",
    "Remove unneeded TFLite attributes from func ops");

}  // namespace TFL
}  // namespace iree_integrations
}  // namespace mlir
