// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Reducer/iree_reduce_lib.h"
#include "iree/compiler/tool_entry_points_api.h"

#include "iree/compiler/Tools/init_dialects.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace llvm;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::Reducer;

#if defined(_MSC_VER)
#define fileno _fileno
#endif // _MSC_VER

// Parse and verify the input MLIR file. Returns null on error.
static OwningOpRef<Operation *> loadModule(MLIRContext &context,
                                           StringRef inputFilename) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  ParserConfig config(&context);
  sourceMgr->AddNewSourceBuffer(std::move(file), SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, config);
}

static LogicalResult ireeReduceMainFromCL(int argc, char **argv,
                                          MLIRContext &registry) {

  llvm::cl::OptionCategory ireeReduceCategory("iree-reduce options");

  llvm::cl::opt<std::string> testScript(cl::Positional, cl::Required,
                                        cl::desc("<test script>"),
                                        cl::cat(ireeReduceCategory));

  cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input file>"),
                                     cl::init("-"),
                                     llvm::cl::cat(ireeReduceCategory));

  cl::opt<std::string> outputFilename(
      "o", cl::desc("Output filename for the reduced test case."),
      cl::value_desc("filename"), cl::init("-"),
      llvm::cl::cat(ireeReduceCategory));

  cl::opt<bool> useBytecodeForTesting(
      "use-bytecode",
      cl::desc("Use bytecode as input to the interesting script."),
      cl::init(false), llvm::cl::cat(ireeReduceCategory));

  cl::opt<bool> outputAsBytecode(
      "output-bytecode", cl::desc("Output the final output as bytecode."),
      cl::init(false), llvm::cl::cat(ireeReduceCategory));

  llvm::cl::HideUnrelatedOptions(ireeReduceCategory);

  InitLLVM y(argc, argv);

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "IREE test case reduction tool.\n");

  // When reading from stdin and the input is a tty, it is often a user mistake
  // and the process "appears to be stuck". Print a message to let the user know
  // about it!
  if (inputFilename == "-" &&
      sys::Process::FileDescriptorIsDisplayed(fileno(stdin)))
    llvm::errs() << "(processing input from stdin now, hit ctrl-c/ctrl-d to "
                    "interrupt)\n";

  OwningOpRef<Operation *> module = loadModule(registry, inputFilename);

  std::string errorMessage;
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  ReducerConfig config(testScript, useBytecodeForTesting);
  Operation *newModule = ireeRunReducingStrategies(std::move(module), config);
  module = OwningOpRef<Operation *>(newModule);

  if (outputAsBytecode) {
    // Write bytecode to output file.
    BytecodeWriterConfig config;
    LogicalResult result =
        writeBytecodeToFile(module.get(), output->os(), config);
    if (failed(result)) {
      llvm::report_fatal_error("Failed to write bytecode to output file");
    }
  } else {
    // Write MLIR to file.
    module->print(output->os());
  }

  // Keep the output file if the invocation of MlirOptMain was successful.
  output->keep();
  return success();
}

int ireeReduceRunMain(int argc, char **argv) {
  llvm::setBugReportMsg(
      "Please report issues to https://github.com/openxla/iree/issues and "
      "include the crash backtrace.\n");

  mlir::DialectRegistry registry;
  mlir::iree_compiler::registerAllDialects(registry);

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  if (ireeReduceMainFromCL(argc, argv, context).failed()) {
    return 1;
  }

  return 0;
}
