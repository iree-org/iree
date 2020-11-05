// Copyright 2020 Google LLC
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

// Main entry function for iree-tf-opt and derived binaries.
//
// Based on iree-opt with the addition of TF dialects and passes

#include "integrations/tensorflow/compiler/dialect/tf_strings/ir/dialect.h"
#include "integrations/tensorflow/compiler/dialect/tf_tensorlist/ir/tf_tensorlist_dialect.h"
#include "iree/tools/init_dialects.h"
#include "iree/tools/init_passes.h"
#include "iree/tools/init_targets.h"
#include "llvm/Support/InitLLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/MlirOptMain.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

void registerTFDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::TF::TensorFlowDialect,
                  mlir::tf_executor::TensorFlowExecutorDialect,
                  mlir::tf_device::TensorFlowDeviceDialect,
                  mlir::tf_saved_model::TensorFlowSavedModelDialect>();
}

void registerExtensionDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::iree_compiler::tf_strings::TFStringsDialect,
                  mlir::tf_tensorlist::TfTensorListDialect>();
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  mlir::DialectRegistry registry;
  mlir::iree_compiler::registerAllDialects(registry);
  registerTFDialects(registry);
  registerExtensionDialects(registry);

  mlir::iree_compiler::registerAllPasses();
  mlir::iree_compiler::registerHALTargetBackends();

  if (failed(MlirOptMain(argc, argv, "IREE-TF modular optimizer driver\n",
                         registry,
                         /*preloadDialectsInContext=*/false))) {
    return 1;
  }
  return 0;
}
