// Copyright 2023 The OpenXLA Authors
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
#define OPENXLA_PARTITIONER_USE_MEMFD_CREATE 1
#else
#define OPENXLA_PARTITIONER_USE_MEMFD_CREATE 0
#endif
#define OPENXLA_PARTITIONER_USE_MMAP 1
#elif defined(_WIN32)
// On Windows, we don't support either memfd_create or the use of mmap.
// The latter could be relaxes in the future by using platform specific
// APIs.
#define OPENXLA_PARTITIONER_USE_MEMFD_CREATE 0
#define OPENXLA_PARTITIONER_USE_MMAP 0
#else
// Default to mmap supported but no memfd_create.
#define OPENXLA_PARTITIONER_USE_MEMFD_CREATE 0
#define OPENXLA_PARTITIONER_USE_MMAP 1
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <atomic>

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
#include "openxla/partitioner/embedding_api.h"
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

#define OPENXLA_PARTITIONER_API_MAJOR 1
#define OPENXLA_PARTITIONER_API_MINOR 1

using namespace mlir;

namespace openxla::partitioner::embed {

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
struct GlobalInit {
  GlobalInit();
  void enableGlobalCL(int argc, const char **argv, const char *banner,
                      bool installSignalHandlers);
  const DialectRegistry &getRegistry() { return registry; }
  bool isGlobalCommandLineEnabled() { return usesCommandLine; }
  void registerCommandLineOptions();

  std::atomic<int> refCount{1};
  DialectRegistry registry;
  // Activates the global command line.
  // TODO: Port CL session scoping from IREE.
  bool usesCommandLine = false;
  // Stash the revision for the life of the instance.
  // TODO: Get release revision stamp.
  std::string revision;
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

  Error *openFile(const char *filePath);
  Error *wrapBuffer(const char *bufferName, const char *buffer, size_t length,
                    bool isNullTerminated);
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
#if !OPENXLA_PARTITIONER_USE_MMAP
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
#if OPENXLA_PARTITIONER_USE_MMAP
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
#if OPENXLA_PARTITIONER_USE_MEMFD_CREATE
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

struct Invocation {
 public:
  Invocation(Session &session);
  bool parseSource(Source &source);
  bool runPipeline(llvm::StringRef pipeline);
  bool runGSPMDPipeline();
  Error *outputIR(Output &output);

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

bool Invocation::parseSource(Source &source) {
  parsedModule =
      mlir::parseSourceFile<ModuleOp>(source.sourceMgr, &session.context);
  if (!parsedModule || failed(mlir::verify(*parsedModule))) {
    return false;
  }
  return true;
}

bool Invocation::runPipeline(llvm::StringRef pipeline) {
  if (pipeline == "gspmd") {
    return runGSPMDPipeline();
  } else {
    emitError(UnknownLoc::get(&session.context))
        << "unknown pipeline requested: '" << pipeline << "'";
    return false;
  }
}

bool Invocation::runGSPMDPipeline() {
  buildGSPMDPipeline(passManager);
  passManager.enableVerifier(enableVerifier);
  if (failed(passManager.run(parsedModule.get()))) {
    return false;
  }
  return true;
}

Error *Invocation::outputIR(Output &output) {
  (*output.outputStream) << *parsedModule.get() << "\n";
  return output.getWriteError();
}

}  // namespace openxla::partitioner::embed

using namespace openxla::partitioner::embed;

//===----------------------------------------------------------------------===//
// Internal to ABI type casters.
//===----------------------------------------------------------------------===//
namespace {
Error *unwrap(openxla_partitioner_error_t *error) { return (Error *)error; }

openxla_partitioner_error_t *wrap(Error *error) {
  return (openxla_partitioner_error_t *)error;
}

Session *unwrap(openxla_partitioner_session_t *session) {
  return (Session *)session;
}

openxla_partitioner_session_t *wrap(Session *session) {
  return (openxla_partitioner_session_t *)session;
}

Invocation *unwrap(openxla_partitioner_invocation_t *inv) {
  return (Invocation *)inv;
}

openxla_partitioner_invocation_t *wrap(Invocation *inv) {
  return (openxla_partitioner_invocation_t *)inv;
}

Source *unwrap(openxla_partitioner_source_t *source) {
  return (Source *)source;
}

openxla_partitioner_source_t *wrap(Source *source) {
  return (openxla_partitioner_source_t *)source;
}

Output *unwrap(openxla_partitioner_output_t *output) {
  return (Output *)output;
}

openxla_partitioner_output_t *wrap(Output *output) {
  return (openxla_partitioner_output_t *)output;
}
}  // namespace

//===----------------------------------------------------------------------===//
// C API implementation
//===----------------------------------------------------------------------===//

namespace {
GlobalInit *globalInit = nullptr;
bool isShutdown = false;
}  // namespace

void openxlaPartitionerErrorDestroy(openxla_partitioner_error_t *error) {
  delete unwrap(error);
}

const char *openxlaPartitionerErrorGetMessage(
    openxla_partitioner_error_t *error) {
  return unwrap(error)->message.c_str();
}

int openxlaPartitionerGetAPIVersion() {
  static_assert(OPENXLA_PARTITIONER_API_MINOR >= 0 &&
                    OPENXLA_PARTITIONER_API_MINOR < 65536,
                "illegal api minor version");
  static_assert(OPENXLA_PARTITIONER_API_MAJOR >= 0 &&
                    OPENXLA_PARTITIONER_API_MAJOR < 65536,
                "illegal api minor version");
  return OPENXLA_PARTITIONER_API_MAJOR << 16 | OPENXLA_PARTITIONER_API_MINOR;
}

void openxlaPartitionerSetupGlobalCL(int argc, const char **argv,
                                     const char *banner,
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

void openxlaPartitionerGlobalInitialize() {
  if (globalInit) {
    globalInit->refCount.fetch_add(1);
    return;
  }
  if (isShutdown) {
    fprintf(stderr,
            "FATAL ERROR: openxlaPartitionerGlobalInitialize called after the "
            "final "
            "openxlaPartitionerGlobalShutdown\n");
    abort();
  }
  globalInit = new GlobalInit();
}

const char *openxlaPartitionerGetRevision() {
  if (!globalInit) {
    fprintf(stderr, "FATAL ERROR: Not initialized\n");
    abort();
  }
  return globalInit->revision.c_str();
}

void openxlaPartitionerGlobalShutdown() {
  if (!globalInit || isShutdown) {
    fprintf(stderr,
            "FATAL ERROR: openxlaPartitionerGlobalShutdown called when not "
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

openxla_partitioner_session_t *openxlaPartitionerSessionCreate() {
  if (!globalInit) {
    fprintf(stderr, "FATAL ERROR: Not initialized\n");
    abort();
  }
  return wrap(new Session(*globalInit));
}

void openxlaPartitionerSessionDestroy(openxla_partitioner_session_t *session) {
  delete unwrap(session);
}

// TODO: Finish implementing
// openxla_partitioner_error_t *openxlaPartitionerSessionSetFlags(
//     openxla_partitioner_session_t *session, int argc, const char *const
//     *argv) {
//   return wrap(unwrap(session)->setFlags(argc, argv));
// }

// TODO: Finish implementing
// void openxlaPartitionerSessionGetFlags(
//     openxla_partitioner_session_t *session, bool nonDefaultOnly,
//     void (*onFlag)(const char *flag, size_t length, void *), void *userData)
//     {
//   unwrap(session)->getFlags(nonDefaultOnly, onFlag, userData);
// }

openxla_partitioner_invocation_t *openxlaPartitionerInvocationCreate(
    openxla_partitioner_session_t *session) {
  return wrap(new Invocation(*unwrap(session)));
}

// TODO: Finish implementing
// void openxlaPartitionerInvocationEnableCallbackDiagnostics(
//     openxla_partitioner_invocation_t *inv, int flags,
//     void (*callback)(enum openxla_partitioner_diagnostic_severity_t severity,
//                      const char *message, size_t messageSize, void
//                      *userData),
//     void *userData) {
//   unwrap(inv)->diagnosticCallbackFlags = flags;
//   unwrap(inv)->diagnosticCallback = callback;
//   unwrap(inv)->diagnosticCallbackUserData = userData;
// }
// void openxlaPartitionerInvocationEnableConsoleDiagnostics(
//     openxla_partitioner_invocation_t *inv) {
//   unwrap(inv)->enableConsoleDiagnosticHandler = true;
// }

void openxlaPartitionerInvocationDestroy(
    openxla_partitioner_invocation_t *inv) {
  delete unwrap(inv);
}

// void openxlaPartitionerInvocationSetCrashHandler(
//     openxla_partitioner_invocation_t *inv, bool genLocalReproducer,
//     openxla_partitioner_error_t *(*onCrashCallback)(
//         openxla_partitioner_output_t **outOutput, void *userData),
//     void *userData) {
//   struct StreamImpl : public mlir::PassManager::ReproducerStream {
//     StreamImpl(openxla_partitioner_output_t *output) : output(output) {
//       unwrap(output)->keep();
//     }
//     ~StreamImpl() { openxlaPartitionerOutputDestroy(output); }

//     llvm::StringRef description() override {
//       return unwrap(output)->description;
//     }

//     llvm::raw_ostream &os() override { return *unwrap(output)->outputStream;
//     }

//     openxla_partitioner_output_t *output;
//   };

//   unwrap(inv)->passManager.enableCrashReproducerGeneration(
//       [=](std::string &errorMessage)
//           -> std::unique_ptr<mlir::PassManager::ReproducerStream> {
//         openxla_partitioner_output_t *output = nullptr;
//         auto error = onCrashCallback(&output, userData);
//         if (error) {
//           errorMessage = openxlaPartitionerErrorGetMessage(error);
//           return nullptr;
//         }

//         if (!output) {
//           errorMessage = "callback did not set output";
//           return nullptr;
//         }

//         return std::make_unique<StreamImpl>(output);
//       },
//       /*genLocalReproducer=*/genLocalReproducer);
// }

bool openxlaPartitionerInvocationParseSource(
    openxla_partitioner_invocation_t *inv,
    openxla_partitioner_source_t *source) {
  return unwrap(inv)->parseSource(*unwrap(source));
}

void openxlaPartitionerInvocationSetVerifyIR(
    openxla_partitioner_invocation_t *inv, bool enable) {
  unwrap(inv)->enableVerifier = enable;
}

bool openxlaPartitionerInvocationPipeline(openxla_partitioner_invocation_t *inv,
                                          const char *pipeline) {
  return unwrap(inv)->runPipeline(pipeline);
}

void openxlaPartitionerSourceDestroy(openxla_partitioner_source_t *source) {
  delete unwrap(source);
}

openxla_partitioner_error_t *openxlaPartitionerSourceOpenFile(
    openxla_partitioner_session_t *session, const char *filePath,
    openxla_partitioner_source_t **out_source) {
  auto source = new Source(*unwrap(session));
  *out_source = wrap(source);
  return wrap(source->openFile(filePath));
}

openxla_partitioner_error_t *openxlaPartitionerSourceWrapBuffer(
    openxla_partitioner_session_t *session, const char *bufferName,
    const char *buffer, size_t length, bool isNullTerminated,
    openxla_partitioner_source_t **out_source) {
  auto source = new Source(*unwrap(session));
  *out_source = wrap(source);
  return wrap(source->wrapBuffer(bufferName, buffer, length, isNullTerminated));
}

void openxlaPartitionerOutputDestroy(openxla_partitioner_output_t *output) {
  delete unwrap(output);
}

openxla_partitioner_error_t *openxlaPartitionerOutputOpenFile(
    const char *filePath, openxla_partitioner_output_t **out_output) {
  auto output = new Output();
  *out_output = wrap(output);
  return wrap(output->openFile(filePath));
}

openxla_partitioner_error_t *openxlaPartitionerOutputOpenFD(
    int fd, openxla_partitioner_output_t **out_output) {
  auto output = new Output();
  *out_output = wrap(output);
  return wrap(output->openFD(fd));
}

openxla_partitioner_error_t *openxlaPartitionerOutputOpenMembuffer(
    openxla_partitioner_output_t **out_output) {
  auto output = new Output();
  *out_output = wrap(output);
  return wrap(output->openMembuffer());
}

openxla_partitioner_error_t *openxlaPartitionerOutputMapMemory(
    openxla_partitioner_output_t *output, void **contents, uint64_t *size) {
  return wrap(unwrap(output)->mapMemory(contents, size));
}

void openxlaPartitionerOutputKeep(openxla_partitioner_output_t *output) {
  unwrap(output)->keep();
}

openxla_partitioner_error_t *openxlaPartitionerOutputWrite(
    openxla_partitioner_output_t *output, const void *data, size_t length) {
  llvm::raw_ostream *os = unwrap(output)->outputStream;
  if (!os) {
    return wrap(new Error("output not open for streaming"));
  }
  os->write((const char *)data, length);
  return wrap(unwrap(output)->getWriteError());
}

openxla_partitioner_error_t *openxlaPartitionerInvocationOutputIR(
    openxla_partitioner_invocation_t *inv,
    openxla_partitioner_output_t *output) {
  return wrap(unwrap(inv)->outputIR(*unwrap(output)));
}
