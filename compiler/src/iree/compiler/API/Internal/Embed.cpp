// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Platform detect for memfd_create and mmap support.
#if __linux__
// On Linux, memfd_create is available for GLIBC >= 2.27.
// Notably, this excludes RHEL7 (and correspondingingly manylinux2014).
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#if __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 27
#define IREE_COMPILER_USE_MEMFD_CREATE 1
#else
#define IREE_COMPILER_USE_MEMFD_CREATE 0
#endif
#define IREE_COMPILER_USE_MMAP 1
#elif defined(_WIN32)
// On Windows, we don't support either memfd_create or the use of mmap.
// The latter could be relaxes in the future by using platform specific
// APIs.
#define IREE_COMPILER_USE_MEMFD_CREATE 0
#define IREE_COMPILER_USE_MMAP 0
#else
// Default to mmap supported but no memfd_create.
#define IREE_COMPILER_USE_MEMFD_CREATE 0
#define IREE_COMPILER_USE_MMAP 1
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <stdio.h>
#include <stdlib.h>

#include <atomic>
#include <limits>

#include "iree/compiler/API/Internal/Diagnostics.h"
#include "iree/compiler/ConstEval/Passes.h"
#include "iree/compiler/Dialect/VM/Target/init_targets.h"
#include "iree/compiler/Pipelines/Pipelines.h"
#include "iree/compiler/PluginAPI/PluginManager.h"
#include "iree/compiler/Tools/init_dialects.h"
#include "iree/compiler/Tools/init_llvmir_translations.h"
#include "iree/compiler/Tools/init_passes.h"
#include "iree/compiler/Tools/init_targets.h"
#include "iree/compiler/Tools/version.h"
#include "iree/compiler/Utils/TracingUtils.h"
#include "iree/compiler/embedding_api.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#ifdef IREE_HAVE_C_OUTPUT_FORMAT
#include "iree/compiler/Dialect/VM/Target/C/CModuleTarget.h"
#include "iree/compiler/Dialect/VM/Target/C/TranslationFlags.h"
#endif  // IREE_HAVE_C_OUTPUT_FORMAT

#ifdef _WIN32
#include "llvm/Support/Windows/WindowsSupport.h"
#endif

namespace mlir::iree_compiler::embed {
namespace {

struct Error {
  Error(std::string message) : message(std::move(message)) {}
  std::string message;
};

struct GlobalInit {
  GlobalInit();
  ~GlobalInit() { llvm::llvm_shutdown(); }
  void registerCommandLineOptions();

  // Reference count of balanced calls to ireeCompilerGlobalInitialize
  // and ireeCompilerGlobalShutdown. Upon reaching zero, must be deleted.
  std::atomic<int> refCount{1};
  llvm::BumpPtrAllocator alloc;
  mlir::DialectRegistry registry;
  PluginManager pluginManager;

  // Command line handling.
  bool usesCommandLine = false;
  // Populated and retained if we have to copy and handle our own permuted
  // argv (i.e. Windows). Otherwise, not used.
  llvm::SmallVector<const char *> retainedArgv;

  // Our session options can optionally be bound to the global command-line
  // environment. If that is not the case, then these will be nullptr, and
  // they should be default initialized at the session level.
  PluginManagerOptions *clPluginManagerOptions = nullptr;
  BindingOptions *clBindingOptions = nullptr;
  InputDialectOptions *clInputOptions = nullptr;
  PreprocessingOptions *clPreprocessingOptions = nullptr;
  HighLevelOptimizationOptions *clHighLevelOptimizationOptions = nullptr;
  SchedulingOptions *clSchedulingOptions = nullptr;
  IREE::HAL::TargetOptions *clHalTargetOptions = nullptr;
  IREE::VM::TargetOptions *clVmTargetOptions = nullptr;
  IREE::VM::BytecodeTargetOptions *clBytecodeTargetOptions = nullptr;
};

GlobalInit::GlobalInit() {
  // Global/static registrations.
  // Allegedly need to register passes to get good reproducers
  // TODO: Verify this (I think that this was fixed some time ago).
  mlir::iree_compiler::registerAllPasses();
  mlir::iree_compiler::registerHALTargetBackends();
  mlir::iree_compiler::registerVMTargets();

  // MLIRContext registration and hooks.
  mlir::iree_compiler::registerAllDialects(registry);
  mlir::iree_compiler::registerLLVMIRTranslations(registry);

  if (!pluginManager.loadAvailablePlugins()) {
    fprintf(stderr, "Failed to initialize IREE compiler plugins.\n");
    abort();
  }
  pluginManager.globalInitialize();
  pluginManager.registerPasses();
  pluginManager.registerGlobalDialects(registry);
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

  // Bind session options to the command line environment.
  clPluginManagerOptions = &PluginManagerOptions::FromFlags::get();
  clBindingOptions = &BindingOptions::FromFlags::get();
  clInputOptions = &InputDialectOptions::FromFlags::get();
  clPreprocessingOptions = &PreprocessingOptions::FromFlags::get();
  clHighLevelOptimizationOptions =
      &HighLevelOptimizationOptions::FromFlags::get();
  clSchedulingOptions = &SchedulingOptions::FromFlags::get();
  clHalTargetOptions = &IREE::HAL::TargetOptions::FromFlags::get();
  clVmTargetOptions = &IREE::VM::TargetOptions::FromFlags::get();
  clBytecodeTargetOptions = &IREE::VM::BytecodeTargetOptions::FromFlags::get();

  pluginManager.initializeCLI();
}

struct Session {
  Session(GlobalInit &globalInit);

  Error *setFlags(int argc, const char *const *argv) {
    std::string errorMessage;
    auto callback = [&](llvm::StringRef message) {
      if (errorMessage.empty()) {
        errorMessage = "Error parsing flags:";
      }
      errorMessage.append("\n  ");
      errorMessage.append(message.data(), message.size());
    };

    if (failed(binder.parseArguments(argc, argv, callback))) {
      return new Error(std::move(errorMessage));
    }
    return nullptr;
  }

  void getFlags(bool nonDefaultOnly,
                void (*onFlag)(const char *flag, size_t length, void *),
                void *userData) {
    auto flagVector = binder.printArguments(nonDefaultOnly);
    for (std::string &value : flagVector) {
      onFlag(value.c_str(), value.size(), userData);
    }
  }

  LogicalResult activatePluginsOnce() {
    if (!pluginsActivated) {
      pluginsActivated = true;
      if (failed(pluginSession.initializePlugins())) {
        pluginActivationStatus = failure();
      } else {
        DialectRegistry registry;
        pluginSession.registerDialects(registry);
        context.appendDialectRegistry(registry);
        pluginActivationStatus = pluginSession.activatePlugins(&context);
      }
    }
    return pluginActivationStatus;
  }

  GlobalInit &globalInit;
  OptionsBinder binder;
  MLIRContext context;
  // PluginManagerOptions must initialize first because the session depends on
  // it.
  PluginManagerOptions pluginManagerOptions;
  PluginManagerSession pluginSession;
  // We lazily activate plugins on the first invocation. This allows plugin
  // activation to be configured at the session level via the API, if
  // desired.
  bool pluginsActivated = false;
  LogicalResult pluginActivationStatus{failure()};

  BindingOptions bindingOptions;
  InputDialectOptions inputOptions;
  PreprocessingOptions preprocessingOptions;
  HighLevelOptimizationOptions highLevelOptimizationOptions;
  SchedulingOptions schedulingOptions;
  IREE::HAL::TargetOptions halTargetOptions;
  IREE::VM::TargetOptions vmTargetOptions;
  IREE::VM::BytecodeTargetOptions bytecodeTargetOptions;
#ifdef IREE_HAVE_C_OUTPUT_FORMAT
  IREE::VM::CTargetOptions cTargetOptions;
#endif
};

Session::Session(GlobalInit &globalInit)
    : globalInit(globalInit),
      binder(OptionsBinder::local()),
      pluginSession(globalInit.pluginManager, binder, pluginManagerOptions) {
  context.allowUnregisteredDialects();
  context.appendDialectRegistry(globalInit.registry);

  // Bootstrap session options from the cl environment, if enabled.
  if (globalInit.usesCommandLine) {
    pluginManagerOptions = *globalInit.clPluginManagerOptions;
    bindingOptions = *globalInit.clBindingOptions;
    inputOptions = *globalInit.clInputOptions;
    preprocessingOptions = *globalInit.clPreprocessingOptions;
    highLevelOptimizationOptions = *globalInit.clHighLevelOptimizationOptions;
    schedulingOptions = *globalInit.clSchedulingOptions;
    halTargetOptions = *globalInit.clHalTargetOptions;
    vmTargetOptions = *globalInit.clVmTargetOptions;
    bytecodeTargetOptions = *globalInit.clBytecodeTargetOptions;
    // TODO: Make C target options like the others.
#ifdef IREE_HAVE_C_OUTPUT_FORMAT
    cTargetOptions = IREE::VM::getCTargetOptionsFromFlags();
#endif
  }

  // Register each options struct with the binder so we can manipulate
  // mnemonically via the API.
  bindingOptions.bindOptions(binder);
  preprocessingOptions.bindOptions(binder);
  inputOptions.bindOptions(binder);
  highLevelOptimizationOptions.bindOptions(binder);
  schedulingOptions.bindOptions(binder);
  halTargetOptions.bindOptions(binder);
  vmTargetOptions.bindOptions(binder);
  bytecodeTargetOptions.bindOptions(binder);
  // TODO: Fix binder support for cTargetOptions.
}

struct Source {
  Source(Session &session) : session(session) {}

  Error *openFile(const char *filePath);
  Error *wrapBuffer(const char *bufferName, const char *buffer, size_t length,
                    bool isNullTerminated);
  Error *split(void (*callback)(iree_compiler_source_t *source, void *userData),
               void *userData);
  const llvm::MemoryBuffer *getMemoryBuffer() {
    return sourceMgr.getMemoryBuffer(1);
  }

  Session &session;
  llvm::SourceMgr sourceMgr;
};

Error *Source::openFile(const char *filePath) {
  std::string err;
  std::unique_ptr<llvm::MemoryBuffer> memoryBuffer =
      mlir::openInputFile(llvm::StringRef(filePath, strlen(filePath)), &err);
  if (!memoryBuffer) {
    return new Error(std::move(err));
  }
  sourceMgr.AddNewSourceBuffer(std::move(memoryBuffer), llvm::SMLoc());
  return nullptr;
}

Error *Source::wrapBuffer(const char *bufferName, const char *buffer,
                          size_t length, bool isNullTerminated) {
  std::unique_ptr<llvm::MemoryBuffer> memoryBuffer;
  if (isNullTerminated) {
    // Sharp edge: MemoryBuffer::getMemBuffer will peek one past the passed
    // length to verify a null terminator, but this makes the API really hard to
    // ensure memory safety for. For our API, we just require that the buffer is
    // null terminated and that the null is included in the length. We then
    // subtract by 1 when constructing the underlying MemoryBuffer. This is
    // quite sad :(
    if (length == 0 || buffer[length - 1] != 0) {
      return new Error("expected null terminated buffer");
    }
    memoryBuffer = llvm::MemoryBuffer::getMemBuffer(
        StringRef(buffer, length - 1),
        StringRef(bufferName, strlen(bufferName)),
        /*RequiresNullTerminator=*/true);
  } else {
    // Not a null terminated buffer.
    memoryBuffer = llvm::MemoryBuffer::getMemBuffer(
        StringRef(buffer, length), StringRef(bufferName, strlen(bufferName)),
        /*RequiresNullTerminator=*/false);
  }
  sourceMgr.AddNewSourceBuffer(std::move(memoryBuffer), llvm::SMLoc());
  return nullptr;
}

Error *Source::split(void (*callback)(iree_compiler_source_t *source,
                                      void *userData),
                     void *userData) {
  const char splitMarkerConst[] = "// -----";
  StringRef splitMarker(splitMarkerConst);

  // This code is adapted from splitAndProcessBuffer which needs to be
  // refactored to be usable. It omits the fuzzy check for near misses
  // because it is very complicated and it is not obvious what it is doing.
  auto *origMemBuffer = getMemoryBuffer();
  SmallVector<StringRef, 8> rawSubBuffers;
  // Split dropping the last checkLen chars to enable flagging near misses.
  origMemBuffer->getBuffer().split(rawSubBuffers, splitMarker);
  if (rawSubBuffers.empty()) return nullptr;

  for (StringRef subBuffer : rawSubBuffers) {
    auto splitLoc = SMLoc::getFromPointer(subBuffer.data());
    unsigned splitLine = sourceMgr.getLineAndColumn(splitLoc).first;
    auto subMemBuffer = llvm::MemoryBuffer::getMemBufferCopy(
        subBuffer, Twine("within split at ") +
                       origMemBuffer->getBufferIdentifier() + ":" +
                       Twine(splitLine) + " offset ");

    Source *subSource = new Source(session);
    subSource->sourceMgr.AddNewSourceBuffer(std::move(subMemBuffer),
                                            llvm::SMLoc());
    callback((iree_compiler_source_t *)subSource, userData);
  }

  return nullptr;
}

struct Output {
  enum class Type {
    None,
    File,
    Membuffer,
  };

  ~Output();
  Error *openFile(const char *filePath);
  Error *openFD(int fd);
  Error *openMembuffer();
  void keep();

  Error *getWriteError() {
    if (type == Type::File && outputFile->os().has_error()) {
      return new Error(outputFile->os().error().message());
    }
    return nullptr;
  }

  Error *mapMemory(void **data, uint64_t *size) {
    if (type == Type::Membuffer) {
      stringOutputStream->flush();
      *data = static_cast<void *>(&outputString[0]);
      *size = outputString.size();
      return nullptr;
    } else if (type == Type::File) {
#if !IREE_COMPILER_USE_MMAP
      return new Error(
          "Mapping memory of a compiler output not created via "
          "ireeCompilerOutputOpenMembuffer is not supported on this platform");
#else
      if (!this->mapped_data) {
        if (!backingFileDescriptor) {
          return new Error(
              "Output was not opened against a file descriptor and cannot be "
              "mapped");
        }
        outputFile->os().flush();
        if (auto *error = getWriteError()) {
          return error;
        }

        this->mapped_size = lseek(*backingFileDescriptor, 0, SEEK_END);
        if (this->mapped_size == -1 ||
            this->mapped_size >= std::numeric_limits<size_t>::max()) {
          return new Error(
              "Failed to determine size of backing file descriptor");
        }
        void *mmap_result =
            mmap(nullptr, static_cast<size_t>(mapped_size), PROT_READ,
                 MAP_SHARED, *backingFileDescriptor, 0);
        if (mmap_result == MAP_FAILED) {
          return new Error("Failed to mmap file descriptor");
        }
        this->mapped_data = mmap_result;
      }
      *data = this->mapped_data;
      *size = this->mapped_size;
      return nullptr;
#endif
    } else {
      return new Error("Output was not opened in a way that supports mapping");
    }
  }

  Type type = Type::None;
  // Description of the output. If a file, this will be the file path.
  // Otherwise, it will be some debug-quality description.
  std::string description;
  llvm::raw_ostream *outputStream = nullptr;

  // If we have mapped the output, the mapping is stashed here.
  void *mapped_data = nullptr;
  uint64_t mapped_size = 0;

 private:
  // Fields for Type::File.
  // If the output was configured to a file, this is it.
  std::unique_ptr<llvm::ToolOutputFile> outputFile;
  // File descriptor if opened in a way where one was provided.
  std::optional<int> backingFileDescriptor;

  // Fields for Type::Memory.
  std::string outputString;
  std::optional<llvm::raw_string_ostream> stringOutputStream;
};

Output::~Output() {
#if IREE_COMPILER_USE_MMAP
  if (mapped_data) {
    munmap(mapped_data, static_cast<size_t>(mapped_size));
  }
#endif
}

Error *Output::openFile(const char *filePath) {
  std::string err;
  type = Type::File;
  description = filePath;
  outputFile = mlir::openOutputFile(description, &err);
  if (!outputFile) {
    return new Error(std::move(err));
  }
  outputStream = &outputFile->os();
  return nullptr;
}

Error *Output::openFD(int fd) {
  type = Type::File;
  description = "fd-";
  description.append(std::to_string(fd));
  outputFile = std::make_unique<llvm::ToolOutputFile>(description, fd);
  // Don't try to delete, etc.
  outputFile->keep();
  outputStream = &outputFile->os();
  this->backingFileDescriptor = fd;
  return nullptr;
}

Error *Output::openMembuffer() {
#if IREE_COMPILER_USE_MEMFD_CREATE
  int fd = memfd_create("iree_output.bin", 0);
  if (fd == -1) {
    return new Error("Error creating membuffer output via memfd_create");
  }
  return openFD(fd);
#else
  // Fallback to an std::string based accumulator if no platform support
  // for memfiles.
  type = Type::Membuffer;
  stringOutputStream.emplace(outputString);
  outputStream = &(*stringOutputStream);
  return nullptr;
#endif
}

void Output::keep() {
  if (outputFile) outputFile->keep();
}

// Invocation corresponds to iree_compiler_invocation_t
struct Invocation {
  Invocation(Session &session);
  bool parseSource(Source &source);
  bool runPipeline(enum iree_compiler_pipeline_t pipeline);
  Error *outputIR(Output &output);
  Error *outputVMBytecode(Output &output);
  Error *outputVMCSource(Output &output);
  Error *outputHALExecutable(Output &output);

  Session &session;
  PassManager passManager;
  IREEVMPipelineHooks pipelineHooks;

  // Diagnostic handlers are instantiated upon parsing the source (when we
  // have the SrcMgr) and held for the duration of the invocation. Each will
  // de-register upon destruction if set.
  std::optional<SourceMgrDiagnosticHandler> consoleDiagnosticHandler;
  std::optional<FormattingDiagnosticHandler> callbackDiagnosticHandler;

  OwningOpRef<Operation *> parsedModule;

  // Run options.
  std::string compileToPhaseName{"end"};
  bool enableVerifier = true;

  // Diagnostic options.
  bool enableConsoleDiagnosticHandler = false;
  void (*diagnosticCallback)(enum iree_compiler_diagnostic_severity_t severity,
                             const char *message, size_t messageSize,
                             void *userData) = nullptr;
  void *diagnosticCallbackUserData = nullptr;
  int diagnosticCallbackFlags = 0;
};

Invocation::Invocation(Session &session)
    : session(session), passManager(&session.context) {
  if (session.globalInit.usesCommandLine) {
    mlir::applyPassManagerCLOptions(passManager);
    mlir::applyDefaultTimingPassManagerCLOptions(passManager);
  }
  passManager.addInstrumentation(std::make_unique<PassTracing>());

  // Since the jitter invokes much of the top-level compiler recursively,
  // it must be injected at the top-level here vs in the pass pipeline
  // (or else the circular dependency cannot be resolved).
  pipelineHooks.buildConstEvalPassPipelineCallback = [](OpPassManager &pm) {
    pm.addPass(ConstEval::createJitGlobalsPass());
  };
  // The PluginSession implements PipelineExtensions and delegates it to
  // activated plugins.
  pipelineHooks.pipelineExtensions = &session.pluginSession;
}

bool Invocation::parseSource(Source &source) {
  // Initialize diagnostics.
  if (enableConsoleDiagnosticHandler && !consoleDiagnosticHandler) {
    consoleDiagnosticHandler.emplace(source.sourceMgr, &session.context);
  }
  if (diagnosticCallback && !callbackDiagnosticHandler) {
    callbackDiagnosticHandler.emplace(
        &session.context,
        [this](DiagnosticSeverity severity, std::string_view message) {
          iree_compiler_diagnostic_severity_t cSeverity;
          switch (severity) {
            case DiagnosticSeverity::Note:
              cSeverity = IREE_COMPILER_DIAGNOSTIC_SEVERITY_NOTE;
              break;
            case DiagnosticSeverity::Warning:
              cSeverity = IREE_COMPILER_DIAGNOSTIC_SEVERITY_NOTE;
              break;
            case DiagnosticSeverity::Error:
              cSeverity = IREE_COMPILER_DIAGNOSTIC_SEVERITY_NOTE;
              break;
            case DiagnosticSeverity::Remark:
              cSeverity = IREE_COMPILER_DIAGNOSTIC_SEVERITY_NOTE;
              break;
            default:
              cSeverity = IREE_COMPILER_DIAGNOSTIC_SEVERITY_ERROR;
          }
          diagnosticCallback(cSeverity, message.data(), message.size(),
                             diagnosticCallbackUserData);
        });
  }

  // Now that diagnostics are enabled, try to activate plugins.
  if (failed(session.activatePluginsOnce())) {
    return false;
  }

  parsedModule =
      mlir::parseSourceFile<ModuleOp>(source.sourceMgr, &session.context);
  if (!parsedModule || failed(mlir::verify(*parsedModule))) {
    return false;
  }
  return true;
}

bool Invocation::runPipeline(enum iree_compiler_pipeline_t pipeline) {
  switch (pipeline) {
    case IREE_COMPILER_PIPELINE_STD: {
      // Parse the compile to phase name.
      std::optional<IREEVMPipelinePhase> compileToPhase;
      enumerateIREEVMPipelinePhases(
          [&](IREEVMPipelinePhase phase, StringRef mnemonic, StringRef desc) {
            if (mnemonic == compileToPhaseName) {
              compileToPhase = phase;
            }
          });
      if (!compileToPhase) {
        (*parsedModule)->emitError()
            << "unrecognized compile-to phase name: " << compileToPhaseName;
        return false;
      }

      buildIREEVMTransformPassPipeline(
          session.bindingOptions, session.inputOptions,
          session.preprocessingOptions, session.highLevelOptimizationOptions,
          session.schedulingOptions, session.halTargetOptions,
          session.vmTargetOptions, pipelineHooks, passManager, *compileToPhase);
      break;
    }
    case IREE_COMPILER_PIPELINE_HAL_EXECUTABLE: {
      auto &bodyBlock = (*parsedModule)->getRegion(0).front();
      auto executableOps =
          llvm::to_vector<4>(bodyBlock.getOps<IREE::HAL::ExecutableOp>());
      auto sourceOps =
          llvm::to_vector<4>(bodyBlock.getOps<IREE::HAL::ExecutableSourceOp>());
      size_t usableOpCount = executableOps.size() + sourceOps.size();
      if (usableOpCount != 1) {
        (*parsedModule)->emitError()
            << "HAL executable translation requires "
               "exactly 1 top level hal.executable/hal.executable.source "
               "op";
        return false;
      }
      IREE::HAL::buildHALTransformPassPipeline(passManager,
                                               session.halTargetOptions);
      break;
    }
    default:
      (*parsedModule)->emitError()
          << "unsupported pipeline type " << (int)pipeline;
      return false;
  }

  passManager.enableVerifier(enableVerifier);
  if (failed(passManager.run(parsedModule.get()))) {
    return false;
  }
  return true;
}

Error *Invocation::outputIR(Output &output) {
  (*output.outputStream) << *parsedModule.get();
  return output.getWriteError();
}

Error *Invocation::outputVMBytecode(Output &output) {
  auto vmModule = llvm::dyn_cast<IREE::VM::ModuleOp>(*parsedModule);
  auto builtinModule = llvm::dyn_cast<mlir::ModuleOp>(*parsedModule);
  LogicalResult result = failure();
  if (vmModule) {
    result = translateModuleToBytecode(vmModule, session.vmTargetOptions,
                                       session.bytecodeTargetOptions,
                                       *output.outputStream);
  } else if (builtinModule) {
    result = translateModuleToBytecode(builtinModule, session.vmTargetOptions,
                                       session.bytecodeTargetOptions,
                                       *output.outputStream);
  } else {
    (*parsedModule)->emitError() << "expected a vm.module or builtin.module";
  }
  if (failed(result)) {
    return new Error("failed to generate bytecode");
  }
  output.outputStream->flush();
  return output.getWriteError();
}

Error *Invocation::outputVMCSource(Output &output) {
#ifndef IREE_HAVE_C_OUTPUT_FORMAT
  return new Error("VM C source output not enabled");
#else
  auto vmModule = llvm::dyn_cast<IREE::VM::ModuleOp>(*parsedModule);
  auto builtinModule = llvm::dyn_cast<mlir::ModuleOp>(*parsedModule);
  LogicalResult result = failure();

  if (vmModule) {
    result = mlir::iree_compiler::IREE::VM::translateModuleToC(
        vmModule, session.cTargetOptions, *output.outputStream);
  } else if (builtinModule) {
    result = mlir::iree_compiler::IREE::VM::translateModuleToC(
        builtinModule, session.cTargetOptions, *output.outputStream);
  } else {
    (*parsedModule)->emitError() << "expected a vm.module or builtin.module";
  }
  if (failed(result)) {
    return new Error("failed to generate bytecode");
  }
  output.outputStream->flush();
  return output.getWriteError();
#endif
}

Error *Invocation::outputHALExecutable(Output &output) {
  // Extract the serialized binary representation from the executable.
  auto &block = (*parsedModule)->getRegion(0).front();
  auto executableOp = *(block.getOps<IREE::HAL::ExecutableOp>().begin());
  auto binaryOps =
      llvm::to_vector<4>(executableOp.getOps<IREE::HAL::ExecutableBinaryOp>());
  if (binaryOps.size() != 1) {
    executableOp.emitError() << "executable translation failed to "
                                "produce exactly 1 binary for "
                                "the input executable";
    return new Error("not a valid HAL executable");
  }
  auto binaryOp = binaryOps.front();
  auto rawData = binaryOp.getData().getRawData();
  output.outputStream->write(rawData.data(), rawData.size());
  output.outputStream->flush();
  return output.getWriteError();
}

}  // namespace
}  // namespace mlir::iree_compiler::embed

using namespace mlir::iree_compiler::embed;

namespace {
GlobalInit *globalInit = nullptr;
bool isShutdown = false;

void llvmVersionPrinter(llvm::raw_ostream &os) {
  os << "IREE (https://openxla.github.io/iree):\n  ";
  std::string version = mlir::iree_compiler::getIreeRevision();
  if (version.empty()) {
    version = "(unknown)";
  }
  os << "IREE compiler version " << version << "\n  ";
  os << "LLVM version " << LLVM_VERSION_STRING << "\n  ";
#if LLVM_IS_DEBUG_BUILD
  os << "DEBUG build";
#else
  os << "Optimized build";
#endif
#ifndef NDEBUG
  os << " with assertions";
#endif
#if LLVM_VERSION_PRINTER_SHOW_HOST_TARGET_INFO
  std::string CPU = std::string(sys::getHostCPUName());
  if (CPU == "generic") CPU = "(unknown)";
  os << ".\n"
     << "  Default target: " << sys::getDefaultTargetTriple() << '\n'
     << "  Host CPU: " << CPU;
#endif
  os << '\n';
}

//===----------------------------------------------------------------------===//
// Internal to ABI type casters.
//===----------------------------------------------------------------------===//

Error *unwrap(iree_compiler_error_t *error) { return (Error *)error; }

iree_compiler_error_t *wrap(Error *error) {
  return (iree_compiler_error_t *)error;
}

Session *unwrap(iree_compiler_session_t *session) { return (Session *)session; }

iree_compiler_session_t *wrap(Session *session) {
  return (iree_compiler_session_t *)session;
}

Invocation *unwrap(iree_compiler_invocation_t *inv) {
  return (Invocation *)inv;
}

iree_compiler_invocation_t *wrap(Invocation *inv) {
  return (iree_compiler_invocation_t *)inv;
}

Source *unwrap(iree_compiler_source_t *source) { return (Source *)source; }

iree_compiler_source_t *wrap(Source *source) {
  return (iree_compiler_source_t *)source;
}

Output *unwrap(iree_compiler_output_t *output) { return (Output *)output; }

iree_compiler_output_t *wrap(Output *output) {
  return (iree_compiler_output_t *)output;
}

}  // namespace

//===----------------------------------------------------------------------===//
// C API implementation
//===----------------------------------------------------------------------===//

void ireeCompilerEnumerateRegisteredHALTargetBackends(
    void (*callback)(const char *backend, void *userData), void *userData) {
  auto registeredTargetBackends =
      mlir::iree_compiler::IREE::HAL::getRegisteredTargetBackends();
  for (auto &b : registeredTargetBackends) {
    callback(b.c_str(), userData);
  }
}

void ireeCompilerErrorDestroy(iree_compiler_error_t *error) {
  delete unwrap(error);
}

const char *ireeCompilerErrorGetMessage(iree_compiler_error_t *error) {
  return unwrap(error)->message.c_str();
}

int ireeCompilerGetAPIVersion() { return 0; }

void ireeCompilerGetProcessCLArgs(int *argc, const char ***argv) {
#ifdef _WIN32
  // See the Windows command line processing in InitLLVM.cpp. It hasn't
  // changed in forever.
  std::string banner = std::string((*argv)[0]) + ": ";
  llvm::ExitOnError exitOnError(banner);
  exitOnError(
      llvm::errorCodeToError(llvm::sys::windows::GetCommandLineArguments(
          globalInit->retainedArgv, globalInit->alloc)));
  // GetCommandLineArguments doesn't terminate with a nullptr per what argv
  // expects, so do that.
  globalInit->retainedArgv.push_back(nullptr);
  *argc = globalInit->retainedArgv.size() - 1;
  *argv = globalInit->retainedArgv.data();
#endif
}

void ireeCompilerSetupGlobalCL(int argc, const char **argv, const char *banner,
                               bool installSignalHandlers) {
  if (globalInit->usesCommandLine) {
    fprintf(stderr, "FATAL ERROR: ireeCompileParseCL called multiple times\n");
    abort();
  }
  globalInit->usesCommandLine = true;
  globalInit->registerCommandLineOptions();

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

void ireeCompilerGlobalInitialize() {
  if (globalInit) {
    globalInit->refCount.fetch_add(1);
    return;
  }
  if (isShutdown) {
    fprintf(stderr,
            "FATAL ERROR: ireeCompilerGlobalInitialize called after the final "
            "ireeCompilerGlobalShutdown\n");
    abort();
  }
  globalInit = new GlobalInit();
}

void ireeCompilerGlobalShutdown() {
  if (!globalInit || isShutdown) {
    fprintf(stderr,
            "FATAL ERROR: ireeCompilerGlobalShutdown called when not "
            "initialized\n");
    abort();
  }
  if (globalInit) {
    if (globalInit->refCount.fetch_sub(1) == 1) {
      delete globalInit;
      isShutdown = true;
      globalInit = nullptr;
      return;
    }
  }
}

iree_compiler_session_t *ireeCompilerSessionCreate() {
  if (!globalInit) {
    fprintf(stderr, "FATAL ERROR: Not initialized\n");
    abort();
  }
  return wrap(new Session(*globalInit));
}

void ireeCompilerSessionDestroy(iree_compiler_session_t *session) {
  delete unwrap(session);
}

iree_compiler_error_t *ireeCompilerSessionSetFlags(
    iree_compiler_session_t *session, int argc, const char *const *argv) {
  return wrap(unwrap(session)->setFlags(argc, argv));
}

void ireeCompilerSessionGetFlags(
    iree_compiler_session_t *session, bool nonDefaultOnly,
    void (*onFlag)(const char *flag, size_t length, void *), void *userData) {
  unwrap(session)->getFlags(nonDefaultOnly, onFlag, userData);
}

iree_compiler_invocation_t *ireeCompilerInvocationCreate(
    iree_compiler_session_t *session) {
  return wrap(new Invocation(*unwrap(session)));
}

void ireeCompilerInvocationEnableCallbackDiagnostics(
    iree_compiler_invocation_t *inv, int flags,
    void (*callback)(enum iree_compiler_diagnostic_severity_t severity,
                     const char *message, size_t messageSize, void *userData),
    void *userData) {
  unwrap(inv)->diagnosticCallbackFlags = flags;
  unwrap(inv)->diagnosticCallback = callback;
  unwrap(inv)->diagnosticCallbackUserData = userData;
}

void ireeCompilerInvocationEnableConsoleDiagnostics(
    iree_compiler_invocation_t *inv) {
  unwrap(inv)->enableConsoleDiagnosticHandler = true;
}

void ireeCompilerInvocationDestroy(iree_compiler_invocation_t *inv) {
  delete unwrap(inv);
}

void ireeCompilerInvocationSetCrashHandler(
    iree_compiler_invocation_t *inv, bool genLocalReproducer,
    iree_compiler_error_t *(*onCrashCallback)(
        iree_compiler_output_t **outOutput, void *userData),
    void *userData) {
  struct StreamImpl : public mlir::PassManager::ReproducerStream {
    StreamImpl(iree_compiler_output_t *output) : output(output) {
      unwrap(output)->keep();
    }
    ~StreamImpl() { ireeCompilerOutputDestroy(output); }

    llvm::StringRef description() override {
      return unwrap(output)->description;
    }

    llvm::raw_ostream &os() override { return *unwrap(output)->outputStream; }

    iree_compiler_output_t *output;
  };

  unwrap(inv)->passManager.enableCrashReproducerGeneration(
      [=](std::string &errorMessage)
          -> std::unique_ptr<mlir::PassManager::ReproducerStream> {
        iree_compiler_output_t *output = nullptr;
        auto error = onCrashCallback(&output, userData);
        if (error) {
          errorMessage = ireeCompilerErrorGetMessage(error);
          return nullptr;
        }

        if (!output) {
          errorMessage = "callback did not set output";
          return nullptr;
        }

        return std::make_unique<StreamImpl>(output);
      },
      /*genLocalReproducer=*/genLocalReproducer);
}

bool ireeCompilerInvocationParseSource(iree_compiler_invocation_t *inv,
                                       iree_compiler_source_t *source) {
  return unwrap(inv)->parseSource(*unwrap(source));
}

void ireeCompilerInvocationSetCompileToPhase(iree_compiler_invocation_t *inv,
                                             const char *phase) {
  unwrap(inv)->compileToPhaseName = std::string(phase);
}

void ireeCompilerInvocationSetVerifyIR(iree_compiler_invocation_t *inv,
                                       bool enable) {
  unwrap(inv)->enableVerifier = enable;
}

bool ireeCompilerInvocationPipeline(iree_compiler_invocation_t *inv,
                                    enum iree_compiler_pipeline_t pipeline) {
  return unwrap(inv)->runPipeline(pipeline);
}

void ireeCompilerSourceDestroy(iree_compiler_source_t *source) {
  delete unwrap(source);
}

iree_compiler_error_t *ireeCompilerSourceOpenFile(
    iree_compiler_session_t *session, const char *filePath,
    iree_compiler_source_t **out_source) {
  auto source = new Source(*unwrap(session));
  *out_source = wrap(source);
  return wrap(source->openFile(filePath));
}

iree_compiler_error_t *ireeCompilerSourceWrapBuffer(
    iree_compiler_session_t *session, const char *bufferName,
    const char *buffer, size_t length, bool isNullTerminated,
    iree_compiler_source_t **out_source) {
  auto source = new Source(*unwrap(session));
  *out_source = wrap(source);
  return wrap(source->wrapBuffer(bufferName, buffer, length, isNullTerminated));
}

iree_compiler_error_t *ireeCompilerSourceSplit(
    iree_compiler_source_t *source,
    void (*callback)(iree_compiler_source_t *source, void *userData),
    void *userData) {
  return wrap(unwrap(source)->split(callback, userData));
}

void ireeCompilerOutputDestroy(iree_compiler_output_t *output) {
  delete unwrap(output);
}

iree_compiler_error_t *ireeCompilerOutputOpenFile(
    const char *filePath, iree_compiler_output_t **out_output) {
  auto output = new Output();
  *out_output = wrap(output);
  return wrap(output->openFile(filePath));
}

iree_compiler_error_t *ireeCompilerOutputOpenFD(
    int fd, iree_compiler_output_t **out_output) {
  auto output = new Output();
  *out_output = wrap(output);
  return wrap(output->openFD(fd));
}

iree_compiler_error_t *ireeCompilerOutputOpenMembuffer(
    iree_compiler_output_t **out_output) {
  auto output = new Output();
  *out_output = wrap(output);
  return wrap(output->openMembuffer());
}

iree_compiler_error_t *ireeCompilerOutputMapMemory(
    iree_compiler_output_t *output, void **contents, uint64_t *size) {
  return wrap(unwrap(output)->mapMemory(contents, size));
}

void ireeCompilerOutputKeep(iree_compiler_output_t *output) {
  unwrap(output)->keep();
}

iree_compiler_error_t *ireeCompilerOutputWrite(iree_compiler_output_t *output,
                                               const void *data,
                                               size_t length) {
  llvm::raw_ostream *os = unwrap(output)->outputStream;
  if (!os) {
    return wrap(new Error("output not open for streaming"));
  }
  os->write((const char *)data, length);
  return wrap(unwrap(output)->getWriteError());
}

iree_compiler_error_t *ireeCompilerInvocationOutputIR(
    iree_compiler_invocation_t *inv, iree_compiler_output_t *output) {
  return wrap(unwrap(inv)->outputIR(*unwrap(output)));
}

iree_compiler_error_t *ireeCompilerInvocationOutputVMBytecode(
    iree_compiler_invocation_t *inv, iree_compiler_output_t *output) {
  return wrap(unwrap(inv)->outputVMBytecode(*unwrap(output)));
}

iree_compiler_error_t *ireeCompilerInvocationOutputVMCSource(
    iree_compiler_invocation_t *inv, iree_compiler_output_t *output) {
  return wrap(unwrap(inv)->outputVMCSource(*unwrap(output)));
}

iree_compiler_error_t *ireeCompilerInvocationOutputHALExecutable(
    iree_compiler_invocation_t *inv, iree_compiler_output_t *output) {
  return wrap(unwrap(inv)->outputHALExecutable(*unwrap(output)));
}
