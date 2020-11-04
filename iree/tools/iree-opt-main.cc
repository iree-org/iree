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

// Main entry function for iree-opt and derived binaries.
//
// Based on mlir-opt but without registering passes and dialects we don't care
// about.

#include "iree/compiler/Conversion/HLOToLinalg/Passes.h"
#include "iree/compiler/Conversion/init_conversions.h"
#include "iree/compiler/Dialect/HAL/Conversion/Passes.h"
#include "iree/tools/init_compiler_modules.h"
#include "iree/tools/init_iree_dialects.h"
#include "iree/tools/init_iree_passes.h"
#include "iree/tools/init_mlir_dialects.h"
#include "iree/tools/init_mlir_passes.h"
#include "iree/tools/init_targets.h"
#include "iree/tools/init_xla_dialects.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"

#ifdef IREE_HAVE_EMITC_DIALECT
#include "emitc/InitDialect.h"
#endif  // IREE_HAVE_EMITC_DIALECT

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  mlir::DialectRegistry registry;
  mlir::registerMlirDialects(registry);
  mlir::registerMlirPasses();
#ifdef IREE_HAVE_EMITC_DIALECT
  mlir::registerEmitCDialect(registry);
#endif  // IREE_HAVE_EMITC_DIALECT
  mlir::registerXLADialects(registry);
  mlir::iree_compiler::registerIreeDialects(registry);
  mlir::iree_compiler::registerIreeCompilerModuleDialects(registry);
  mlir::iree_compiler::registerAllIreePasses();
  mlir::iree_compiler::registerHALConversionPasses();
  mlir::iree_compiler::registerHALTargetBackends();
  mlir::iree_compiler::registerLinalgToSPIRVPasses();
  mlir::iree_compiler::registerHLOToLinalgPasses();
  mlir::iree_compiler::registerLinalgToLLVMPasses();

  if (failed(MlirOptMain(argc, argv, "IREE modular optimizer driver\n",
                         registry,
                         /*preloadDialectsInContext=*/false))) {
    return 1;
  }
  return 0;
}
