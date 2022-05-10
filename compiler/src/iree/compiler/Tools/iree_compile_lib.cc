// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <functional>
#include <memory>
#include <string>
#include <type_traits>

#include "iree/compiler/ConstEval/Passes.h"
#include "iree/compiler/Dialect/VM/Target/init_targets.h"
#include "iree/compiler/Pipelines/Pipelines.h"
#include "iree/compiler/Tools/init_dialects.h"
#include "iree/compiler/Tools/init_llvmir_translations.h"
#include "iree/compiler/Tools/init_passes.h"
#include "iree/compiler/Tools/init_targets.h"
#include "iree/compiler/Tools/iree_translate_lib.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "iree/compiler/Utils/TracingUtils.h"
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
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#ifdef IREE_HAVE_EMITC_DIALECT
#include "iree/compiler/Dialect/VM/Target/C/CModuleTarget.h"
#include "iree/compiler/Dialect/VM/Target/C/TranslationFlags.h"
#endif  // IREE_HAVE_EMITC_DIALECT

namespace mlir {
namespace iree_compiler {

namespace {

enum class OutputFormat {
  none,
  vm_asm,
  vm_bytecode,
  vm_c,
};

IREEVMPipelineHooks &getHooks() {
  static IREEVMPipelineHooks hooks = {
      // buildConstEvalPassPipelineCallback =
      [](OpPassManager &pm) { pm.addPass(ConstEval::createJitGlobalsPass()); }};
  return hooks;
}

}  // namespace

}  // namespace iree_compiler
}  // namespace mlir

int mlir::iree_compiler::runIreecMain(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  static llvm::cl::OptionCategory mainOptions("IREE Main Options");

  // Global/static registrations.
  // Allegedly need to register passes to get good reproducers
  // TODO: Verify this (I think that this was fixed some time ago).
  mlir::iree_compiler::registerAllPasses();
  mlir::iree_compiler::registerHALTargetBackends();
  mlir::iree_compiler::registerVMTargets();

  // MLIRContext registration and hooks.
  mlir::DialectRegistry registry;
  mlir::iree_compiler::registerAllDialects(registry);
  mlir::iree_compiler::registerLLVMIRTranslations(registry);

  // Register MLIRContext command-line options like
  // -mlir-print-op-on-diagnostic.
  mlir::registerMLIRContextCLOptions();
  // Register assembly printer command-line options like
  // -mlir-print-op-generic.
  mlir::registerAsmPrinterCLOptions();
  // Register pass manager command-line options like -mlir-print-ir-*.
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();

  // Flag options structs (must resolve prior to CLI parsing).
  auto &bindingOptions = BindingOptions::FromFlags::get();
  auto &inputOptions = InputDialectOptions::FromFlags::get();
  auto &highLevelOptimizationOptions =
      HighLevelOptimizationOptions::FromFlags::get();
  auto &schedulingOptions = SchedulingOptions::FromFlags::get();
  auto &halTargetOptions = IREE::HAL::TargetOptions::FromFlags::get();
  auto &vmTargetOptions = IREE::VM::TargetOptions::FromFlags::get();
  auto &bytecodeTargetOptions =
      IREE::VM::BytecodeTargetOptions::FromFlags::get();

  // General command line flags.
  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file or '-' for stdin>"),
      llvm::cl::Required, llvm::cl::cat(mainOptions));

  llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"), llvm::cl::cat(mainOptions));

  // The output format flag is the master control for what we do with the
  // in-memory compiled form.
  llvm::cl::opt<OutputFormat> outputFormat(
      "output-format", llvm::cl::desc("Format of compiled output"),
      llvm::cl::values(
          clEnumValN(OutputFormat::vm_bytecode, "vm-bytecode",
                     "IREE VM Bytecode (default)"),
#ifdef IREE_HAVE_EMITC_DIALECT
          clEnumValN(OutputFormat::vm_c, "vm-c", "C source module"),
#endif
          clEnumValN(OutputFormat::vm_asm, "vm-asm", "IREE VM MLIR Assembly")),
      llvm::cl::init(OutputFormat::none), llvm::cl::cat(mainOptions));

  llvm::cl::opt<bool> legacyTranslateToCModule(
      "iree-mlir-to-vm-c-module",
      llvm::cl::desc("Alias for --output-format=c-module (deprecated)"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> legacyTranslateToVMBytecodeModule(
      "iree-mlir-to-vm-bytecode-module",
      llvm::cl::desc("Alias for --output-format=vm-bytecode (deprecated)"),
      llvm::cl::init(false));

  // Misc options.
  llvm::cl::opt<bool> splitInputFile(
      "split-input-file",
      llvm::cl::desc("Split the input file into pieces and "
                     "process each chunk independently"),
      llvm::cl::init(false));

// Optional output formats.
#ifdef IREE_HAVE_EMITC_DIALECT
  auto cTargetOptions = IREE::VM::getCTargetOptionsFromFlags();
#endif

  llvm::cl::ParseCommandLineOptions(argc, argv, "IREE compilation driver\n");

  // Post-process and select the correct outputFormat.
  if (legacyTranslateToCModule) {
    if (outputFormat != OutputFormat::none) {
      llvm::errs()
          << "Cannot specify --output-format= and --iree-mlir-to-vm-c-module\n";
      return 1;
    }
    outputFormat = OutputFormat::vm_c;
  }
  if (legacyTranslateToVMBytecodeModule) {
    if (outputFormat != OutputFormat::none) {
      llvm::errs() << "Cannot specify --output-format= and "
                      "--iree-mlir-to-vm-bytecode-module\n";
      return 1;
    }
    outputFormat = OutputFormat::vm_bytecode;
  }

  // Defualt output format.
  if (outputFormat == OutputFormat::none) {
    outputFormat = OutputFormat::vm_bytecode;
  }

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
                           llvm::raw_ostream &os) -> LogicalResult {
    mlir::MLIRContext context;
    context.allowUnregisteredDialects();
    context.appendDialectRegistry(registry);
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());
    mlir::SourceMgrDiagnosticHandler diagHandler(sourceMgr, &context);

    // Parse source.
    auto module = parseSourceFile<ModuleOp>(sourceMgr, &context);
    if (!module || failed(verify(*module))) {
      return failure();
    }

    // Main compilation pipeline.
    PassManager passManager(&context);
    mlir::applyPassManagerCLOptions(passManager);
    mlir::applyDefaultTimingPassManagerCLOptions(passManager);
    passManager.addInstrumentation(std::make_unique<PassTracing>());
    buildIREEVMTransformPassPipeline(bindingOptions, inputOptions,
                                     highLevelOptimizationOptions,
                                     schedulingOptions, halTargetOptions,
                                     vmTargetOptions, getHooks(), passManager);
    if (failed(passManager.run(module.get()))) {
      llvm::errs() << "compilation from source to vm failed\n";
      return failure();
    }

    // Switch based on output format.
    switch (outputFormat) {
      case OutputFormat::vm_asm:
        os << module.get();
        return success();
      case OutputFormat::vm_bytecode:
        return translateModuleToBytecode(module.get(), bytecodeTargetOptions,
                                         os);
#ifdef IREE_HAVE_EMITC_DIALECT
      case OutputFormat::vm_c:
        return mlir::iree_compiler::IREE::VM::translateModuleToC(
            module.get(), cTargetOptions, os);
#endif
      default:
        llvm::errs() << "INTERNAL ERROR: Unknown output format\n";
        return failure();
    }

    return failure();
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
