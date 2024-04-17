// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Main entry function for iree-opt and derived binaries.
//
// Based on mlir-opt but registers the passes and dialects we care about.

#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/PluginManager.h"
#include "iree/compiler/Tools/init_dialects.h"
#include "iree/compiler/Tools/init_passes.h"
#include "iree/compiler/tool_entry_points_api.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Debug/Counter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace llvm;
using namespace mlir;

using mlir::iree_compiler::IREE::HAL::TargetBackendList;
using mlir::iree_compiler::IREE::HAL::TargetDeviceList;
using mlir::iree_compiler::IREE::HAL::TargetRegistry;

#if defined(_MSC_VER)
#define fileno _fileno
#endif // _MSC_VER

static LogicalResult ireeOptMainFromCL(int argc, char **argv,
                                       llvm::StringRef toolName,
                                       DialectRegistry &registry) {
  static cl::opt<std::string> inputFilename(
      cl::Positional, cl::desc("<input file>"), cl::init("-"));

  static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"),
                                             cl::init("-"));

  InitLLVM y(argc, argv);

  // We support a limited form of the PluginManager, allowing it to perform
  // global initialization and dialect registration.
  mlir::iree_compiler::PluginManager pluginManager;
  if (!pluginManager.loadAvailablePlugins()) {
    llvm::errs() << "error: Failed to initialize IREE compiler plugins\n";
    return failure();
  }
  // Initialization that applies to all global plugins must be done prior
  // to CL parsing.
  pluginManager.globalInitialize();
  pluginManager.registerPasses();
  pluginManager.registerGlobalDialects(registry);
  pluginManager.initializeCLI();

  // Register command line options. All passes must be registered at this point
  // to expose them through the tool.
  MlirOptMainConfig::registerCLOptions(registry);
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  tracing::DebugCounter::registerCLOptions();
  auto &pluginManagerOptions =
      mlir::iree_compiler::PluginManagerOptions::FromFlags::get();

  // Build the list of dialects as a header for the --help message.
  std::string helpHeader = (toolName + "\nAvailable Dialects: ").str();
  {
    llvm::raw_string_ostream os(helpHeader);
    interleaveComma(registry.getDialectNames(), os,
                    [&](auto name) { os << name; });
  }

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, helpHeader);
  MlirOptMainConfig config = MlirOptMainConfig::createFromCLOptions();

  // The local binder is meant for overriding session-level options, but for
  // tools like this it is unused.
  auto localBinder = mlir::iree_compiler::OptionsBinder::local();
  mlir::iree_compiler::PluginManagerSession pluginSession(
      pluginManager, localBinder, pluginManagerOptions);
  if (failed(pluginSession.initializePlugins()))
    return failure();
  pluginSession.registerDialects(registry);

  // In the normal compiler flow, activated plugins maintain a scoped registry
  // of target backends. However, no such layering exists for the opt tool.
  // Since it tests passes that are default initialized, we just configure the
  // global registry that such constructors depend on.
  TargetBackendList pluginTargetBackendList;
  pluginSession.populateHALTargetBackends(pluginTargetBackendList);
  const_cast<TargetRegistry &>(TargetRegistry::getGlobal())
      .mergeFrom(pluginTargetBackendList);
  TargetDeviceList pluginTargetDeviceList;
  pluginSession.populateHALTargetDevices(pluginTargetDeviceList);
  const_cast<TargetRegistry &>(TargetRegistry::getGlobal())
      .mergeFrom(pluginTargetDeviceList);

  // When reading from stdin and the input is a tty, it is often a user mistake
  // and the process "appears to be stuck". Print a message to let the user know
  // about it!
  if (inputFilename == "-" &&
      sys::Process::FileDescriptorIsDisplayed(fileno(stdin)))
    llvm::errs() << "(processing input from stdin now, hit ctrl-c/ctrl-d to "
                    "interrupt)\n";

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }
  if (failed(MlirOptMain(output->os(), std::move(file), registry, config)))
    return failure();

  // Keep the output file if the invocation of MlirOptMain was successful.
  output->keep();
  return success();
}

int ireeOptRunMain(int argc, char **argv) {
  llvm::setBugReportMsg(
      "Please report issues to https://github.com/openxla/iree/issues and "
      "include the crash backtrace.\n");

  mlir::DialectRegistry registry;
  mlir::iree_compiler::registerAllDialects(registry);
  mlir::iree_compiler::registerAllPasses();

  // Register the pass to drop embedded transform dialect IR.
  // TODO: this should be upstreamed.
  mlir::linalg::transform::registerDropSchedulePass();

  if (failed(ireeOptMainFromCL(argc, argv, "IREE modular optimizer driver\n",
                               registry))) {
    return 1;
  }
  return 0;
}
