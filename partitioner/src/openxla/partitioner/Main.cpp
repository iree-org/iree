// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "openxla/partitioner/GSPMDPipeline.h"
#include "stablehlo/dialect/Register.h"

// Upstream dialect deps.
// See: https://github.com/openxla/stablehlo/issues/1464
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

// Deps needed by the impl.
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"

using namespace mlir;

namespace openxla::partitioner {

namespace {
void llvmVersionPrinter(llvm::raw_ostream &os) {
  os << "OpenXLA Partitioner (https://openxla.github.io/iree)";
}
}  // namespace

// Errors are allocated and returned by pointer, transferring ownership to
// the caller. nullptr indicates no error.
struct Error {
  Error(std::string message) : message(std::move(message)) {}
  std::string message;
};

// Global initialization is encapsulated in the GlobalInit struct.
class GlobalInit {
 public:
  GlobalInit();
  void enableGlobalCL(int argc, const char **argv, const char *banner,
                      bool installSignalHandlers);
  const DialectRegistry &getRegistry() { return registry; }
  bool isGlobalCommandLineEnabled() { return usesCommandLine; }

 private:
  void registerCommandLineOptions();

  DialectRegistry registry;
  // Activates the global command line.
  // TODO: Port CL session scoping from IREE.
  bool usesCommandLine = false;
};

GlobalInit::GlobalInit() {
  registry.insert<cf::ControlFlowDialect, ml_program::MLProgramDialect,
                  quant::QuantizationDialect, func::FuncDialect,
                  tensor::TensorDialect, shape::ShapeDialect>();
  stablehlo::registerAllDialects(registry);
}

void GlobalInit::enableGlobalCL(int argc, const char **argv, const char *banner,
                                bool installSignalHandlers) {
  if (usesCommandLine) {
    llvm::errs() << "FATAL ERROR: ireeCompileParseCL called multiple times\n";
    abort();
  }

  usesCommandLine = true;
  registerCommandLineOptions();

  llvm::setBugReportMsg(
      "Please report issues to https://github.com/openxla/iree/issues and "
      "include the crash backtrace.\n");
  llvm::cl::SetVersionPrinter(llvmVersionPrinter);

  if (installSignalHandlers) {
    // A few other things from InitLLVM to setup default command-line signal
    // handlers. See InitLLVM::InitLLVM for the initialization sequence and
    // commentary, should it ever be necessary to revist this (it hasn't changed
    // in many years).
    llvm::sys::SetOneShotPipeSignalFunction(
        llvm::sys::DefaultOneShotPipeSignalHandler);
    static llvm::PrettyStackTraceProgram stackPrinter(argc, argv);
    llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
    llvm::install_out_of_memory_new_handler();
  }

  llvm::cl::ParseCommandLineOptions(argc, argv, banner);
}

void GlobalInit::registerCommandLineOptions() {
  // Register MLIRContext command-line options like
  // -mlir-print-op-on-diagnostic.
  mlir::registerMLIRContextCLOptions();
  // Register assembly printer command-line options like
  // -mlir-print-op-generic.
  mlir::registerAsmPrinterCLOptions();
  // Register pass manager command-line options like -mlir-print-ir-*.
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
}

// Sessions bring together an initialized context and set of flags.
struct Session {
  Session(GlobalInit &globalInit);

  GlobalInit &globalInit;
  MLIRContext context;
};

Session::Session(GlobalInit &globalInit) : globalInit(globalInit) {
  context.appendDialectRegistry(globalInit.getRegistry());
}

// A source is instantiated against a session and is used to access an llvm
// MemoryBuffer suitable for parsing. It is rooted on the session because
// advanced use cases can involve special setup for accessing sources.
struct Source {
  Source(Session &session) : session(session) {}

  std::unique_ptr<Error> openFile(const char *filePath);
  // Error *wrapBuffer(const char *bufferName, const char *buffer, size_t
  // length,
  //                   bool isNullTerminated);
  // Error *split(void (*callback)(iree_compiler_source_t *source, void
  // *userData),
  //              void *userData);
  const llvm::MemoryBuffer *getMemoryBuffer() {
    return sourceMgr.getMemoryBuffer(1);
  }

  Session &session;
  llvm::SourceMgr sourceMgr;
};

std::unique_ptr<Error> Source::openFile(const char *filePath) {
  std::string err;
  std::unique_ptr<llvm::MemoryBuffer> memoryBuffer =
      mlir::openInputFile(llvm::StringRef(filePath, strlen(filePath)), &err);
  if (!memoryBuffer) {
    return std::make_unique<Error>(std::move(err));
  }
  sourceMgr.AddNewSourceBuffer(std::move(memoryBuffer), llvm::SMLoc());
  return nullptr;
}

class Invocation {
 public:
  Invocation(Session &session);
  bool parseSource(Source &source);
  bool runGSPMDPipeline();

  // TODO: Wire up Output API.
  void dump();

 private:
  Session &session;
  PassManager passManager;
  OwningOpRef<Operation *> parsedModule;
  bool enableVerifier = true;
};

Invocation::Invocation(Session &session)
    : session(session), passManager(&session.context) {
  if (session.globalInit.isGlobalCommandLineEnabled()) {
    if (failed(mlir::applyPassManagerCLOptions(passManager))) {
      emitError(UnknownLoc::get(&session.context))
          << "failed to apply pass manager CL options";
    }
    mlir::applyDefaultTimingPassManagerCLOptions(passManager);
  }
  // Get instrumentation from IREE.
  // passManager.addInstrumentation(std::make_unique<PassTracing>());
}

void Invocation::dump() { (*parsedModule)->dump(); }

bool Invocation::parseSource(Source &source) {
  parsedModule =
      mlir::parseSourceFile<ModuleOp>(source.sourceMgr, &session.context);
  if (!parsedModule || failed(mlir::verify(*parsedModule))) {
    return false;
  }
  return true;
}

bool Invocation::runGSPMDPipeline() {
  buildGSPMDPipeline(passManager);
  passManager.enableVerifier(enableVerifier);
  if (failed(passManager.run(parsedModule.get()))) {
    return false;
  }
  return true;
}

}  // namespace openxla::partitioner

using namespace openxla::partitioner;

// This `main` is temporary and is using the C++ objects directly. To match
// how it is done in IREE, it should be using the C API, once established.
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  static llvm::cl::opt<std::string> inputPath(
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::Required);

  GlobalInit globalInit;
  globalInit.enableGlobalCL(argc, const_cast<const char **>(argv),
                            "OpenXLA Partitioner CLI",
                            /*installSignalHandlers=*/true);
  Session session(globalInit);
  Source source(session);
  if (auto err = source.openFile(inputPath.c_str())) {
    llvm::errs() << "ERROR: " << err->message << "\n";
    return 1;
  }
  Invocation inv(session);
  if (!inv.parseSource(source)) {
    return 1;
  }
  if (!inv.runGSPMDPipeline()) {
    return 1;
  }

  inv.dump();
  return 0;
}
