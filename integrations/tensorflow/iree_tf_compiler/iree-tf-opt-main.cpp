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
// This is a bare-bones, minimal *-opt just for testing the handful of local
// passes here. If you need something, add it, but add only what you need as
// each addition will likely end up on the build critical path.

#include "iree_tf_compiler/Passes.h"
#include "iree/tools/init_xla_dialects.h"
#include "llvm/Support/InitLLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/MlirOptMain.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  mlir::DialectRegistry registry;
  mlir::registerXLADialects(registry);

  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::iree_integrations::TF::registerAllDialects(registry);

  if (failed(MlirOptMain(argc, argv, "IREE-TF modular optimizer driver\n",
                         registry,
                         /*preloadDialectsInContext=*/false))) {
    return 1;
  }
  return 0;
}
