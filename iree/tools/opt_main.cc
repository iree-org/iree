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

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));

static llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> verifyPasses(
    "verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    llvm::cl::desc("Allow operation with no registered dialects"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> showDialects(
    "show-dialects", llvm::cl::desc("Print the list of registered dialects"),
    llvm::cl::init(false));

int main(int argc, char **argv) {
  mlir::registerMlirDialects();
  mlir::registerMlirPasses();
#ifdef IREE_HAVE_EMITC_DIALECT
  mlir::registerEmitCDialect();
#endif  // IREE_HAVE_EMITC_DIALECT
  mlir::registerXLADialects();
  mlir::iree_compiler::registerIreeDialects();
  mlir::iree_compiler::registerIreeCompilerModuleDialects();
  mlir::iree_compiler::registerAllIreePasses();
  mlir::iree_compiler::registerHALConversionPasses();
  mlir::iree_compiler::registerHALTargetBackends();
  mlir::iree_compiler::registerLinalgToSPIRVPasses();
  mlir::iree_compiler::registerHLOToLinalgPasses();
  mlir::iree_compiler::registerLinalgToLLVMPasses();
  llvm::InitLLVM y(argc, argv);

  // Register MLIRContext command-line options like
  // -mlir-print-op-on-diagnostic.
  mlir::registerMLIRContextCLOptions();
  // Register assembly printer command-line options like
  // -mlir-print-op-generic.
  mlir::registerAsmPrinterCLOptions();
  // Register pass manager command-line options like -print-ir-*.
  mlir::registerPassManagerCLOptions();

  mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "IREE modular optimizer driver\n");

  if (showDialects) {
    llvm::outs() << "Registered Dialects:\n";
    mlir::MLIRContext context;
    for (mlir::Dialect *dialect : context.getRegisteredDialects()) {
      llvm::outs() << dialect->getNamespace() << "\n";
    }
    return 0;
  }

  // Set up the input file.
  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  return failed(mlir::MlirOptMain(output->os(), std::move(file), passPipeline,
                                  splitInputFile, verifyDiagnostics,
                                  verifyPasses, allowUnregisteredDialects));
}
