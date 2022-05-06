// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Custom translation main entry function.
// Based on the iree-compile main entry function (iree-compile-main.cc).
//
// We need this entry function because we want to register the custom
// dialect, which is missing in IREE's translation main entry function.

#include "iree/compiler/Dialect/VM/Target/init_targets.h"
#include "iree/compiler/Tools/init_compiler_modules.h"
#include "iree/compiler/Tools/init_iree_dialects.h"
#include "iree/compiler/Tools/init_mlir_dialects.h"
#include "iree/compiler/Tools/init_targets.h"
#include "iree/compiler/Tools/init_translations.h"
#include "iree/compiler/Tools/init_xla_dialects.h"
#include "iree_custom_modules/dialect/init_dialect.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/mlir-translate/Translation.h"

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));

static llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and "
                   "process each chunk independently"),
    llvm::cl::init(false));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  mlir::DialectRegistry registry;

  mlir::registerMlirDialects(registry);
  mlir::registerXLADialects(registry);
  mlir::iree_compiler::registerIreeDialects(registry);
  // Register the custom dialect
  mlir::iree_compiler::registerCustomDialect(registry);
  mlir::iree_compiler::registerIreeCompilerModuleDialects(registry);
  mlir::iree_compiler::registerHALTargetBackends();
  mlir::iree_compiler::registerVMTargets();
  mlir::registerMlirTranslations();
  mlir::iree_compiler::registerIreeTranslations();
  // Make sure command line options are registered.
  (void)mlir::iree_compiler::IREE::HAL::TargetOptions::FromFlags::get();

  // Register MLIRContext command-line options like
  // -mlir-print-op-on-diagnostic.
  mlir::registerMLIRContextCLOptions();
  // Register assembly printer command-line options like
  // -mlir-print-op-generic.
  mlir::registerAsmPrinterCLOptions();
  // Register pass manager command-line options like -mlir-print-ir-*.
  mlir::registerPassManagerCLOptions();

  // Add flags for all the registered translations.
  llvm::cl::opt<const mlir::TranslateFunction *, false, mlir::TranslationParser>
      translationRequested("", llvm::cl::desc("Translation to perform"),
                           llvm::cl::Required);

  llvm::cl::ParseCommandLineOptions(argc, argv, "IREE translation driver\n");

  std::string errorMessage;
  auto input = mlir::openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  /// Processes the memory buffer with a new MLIRContext.
  auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
                           llvm::raw_ostream &os) {
    mlir::MLIRContext context;
    context.appendDialectRegistry(registry);
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());
    mlir::SourceMgrDiagnosticHandler diagHandler(sourceMgr, &context);
    return (*translationRequested)(sourceMgr, os, &context);
  };

  if (splitInputFile) {
    if (failed(mlir::splitAndProcessBuffer(std::move(input), processBuffer,
                                           output->os())))
      return 1;
  } else {
    if (failed(processBuffer(std::move(input), output->os()))) return 1;
  }

  output->keep();
  return 0;
}
