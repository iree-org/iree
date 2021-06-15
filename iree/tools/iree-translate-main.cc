// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE translation main entry function.
//
// Note that this differs from mlir-translate and similar because we use the
// PassManger and do transformations on the IR before translating to other
// formats. Thus we use our own main entry function because we register
// Dialects and PassManager CLI options.

#include <functional>
#include <memory>
#include <string>
#include <type_traits>

#include "iree/compiler/Dialect/VM/Target/init_targets.h"
#include "iree/tools/init_compiler_modules.h"
#include "iree/tools/init_iree_dialects.h"
#include "iree/tools/init_mlir_dialects.h"
#include "iree/tools/init_targets.h"
#include "iree/tools/init_translations.h"
#include "iree/tools/init_xla_dialects.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Translation.h"

#ifdef IREE_HAVE_EMITC_DIALECT
#include "emitc/InitDialect.h"
#include "emitc/InitTranslation.h"
#endif  // IREE_HAVE_EMITC_DIALECT

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
  mlir::registerLLVMDialectTranslation(registry);
#ifdef IREE_HAVE_EMITC_DIALECT
  mlir::registerEmitCDialect(registry);
#endif  // IREE_HAVE_EMITC_DIALECT
  mlir::registerXLADialects(registry);
  mlir::iree_compiler::registerIreeDialects(registry);
  mlir::iree_compiler::registerIreeCompilerModuleDialects(registry);
  mlir::iree_compiler::registerHALTargetBackends();
  mlir::iree_compiler::registerVMTargets();
  mlir::registerMlirTranslations();
#ifdef IREE_HAVE_EMITC_DIALECT
  mlir::registerEmitCTranslation();
#endif  // IREE_HAVE_EMITC_DIALECT
  mlir::iree_compiler::registerIreeTranslations();
  // Make sure command line options are registered.
  (void)mlir::iree_compiler::IREE::HAL::getTargetOptionsFromFlags();

  // Register MLIRContext command-line options like
  // -mlir-print-op-on-diagnostic.
  mlir::registerMLIRContextCLOptions();
  // Register assembly printer command-line options like
  // -mlir-print-op-generic.
  mlir::registerAsmPrinterCLOptions();
  // Register pass manager command-line options like -print-ir-*.
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();

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
    context.allowUnregisteredDialects();
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
