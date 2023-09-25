// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Tools/iree_compile_lib.h"

#include <functional>
#include <memory>
#include <string>
#include <type_traits>

#include "iree/compiler/Pipelines/Pipelines.h"
#include "iree/compiler/embedding_api.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace iree_compiler {

namespace {

enum class OutputFormat {
  none,
  vm_asm,
  vm_bytecode,
  vm_c,
  // Non-user exposed output formats.
  hal_executable,
  precompile,
};

enum class CompileMode {
  // IREE's full compilation pipeline.
  std,
  // Compile from VM IR (currently this does nothing but may do more in the
  // future).
  vm,
  // Translates an MLIR module containing a single hal.executable into a
  // target-specific binary form (such as an ELF file or a flatbuffer containing
  // a SPIR-V blob).
  hal_executable,
  // IREE's precompilation pipeline, which does input preprocessing and
  // pre-fusion global optimizations.
  precompile,
};

struct BytecodeVersionParser : public llvm::cl::parser<std::optional<int64_t>> {
  BytecodeVersionParser(llvm::cl::Option &O)
      : llvm::cl::parser<std::optional<int64_t>>(O) {}
  bool parse(llvm::cl::Option &O, StringRef /*argName*/, StringRef arg,
             std::optional<int64_t> &v) {
    long long w;
    if (llvm::getAsSignedInteger(arg, 10, w))
      return O.error("Invalid argument '" + arg +
                     "', only integer is supported.");
    v = w;
    return false;
  }
};

} // namespace

} // namespace iree_compiler
} // namespace mlir

int mlir::iree_compiler::runIreecMain(int argc, char **argv) {
  static llvm::cl::OptionCategory mainOptions("IREE Main Options");

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
#ifdef IREE_HAVE_C_OUTPUT_FORMAT
          clEnumValN(OutputFormat::vm_c, "vm-c", "C source module"),
#endif // IREE_HAVE_C_OUTPUT_FORMAT
          clEnumValN(OutputFormat::vm_asm, "vm-asm", "IREE VM MLIR Assembly")),
      llvm::cl::init(OutputFormat::vm_bytecode), llvm::cl::cat(mainOptions));

  llvm::cl::opt<CompileMode> compileMode(
      "compile-mode", llvm::cl::desc("IREE compilation mode"),
      llvm::cl::values(
          clEnumValN(CompileMode::std, "std", "Standard compilation"),
          clEnumValN(CompileMode::vm, "vm", "Compile from VM IR"),
          clEnumValN(
              CompileMode::hal_executable, "hal-executable",
              "Compile an MLIR module containing a single hal.executable into "
              "a target-specific binary form (such as an ELF file or a "
              "flatbuffer containing a SPIR-V blob)"),
          clEnumValN(CompileMode::precompile, "precompile",
                     "Precompilation pipeline which does input conversion and "
                     "global optimizations.")),
      llvm::cl::init(CompileMode::std), llvm::cl::cat(mainOptions));

  // Debugging/diagnostics.
  llvm::cl::opt<bool> verifyIR(
      "verify",
      llvm::cl::desc("Verifies the IR for correctness throughout compilation."),
      llvm::cl::init(true));

  llvm::cl::opt<IREEVMPipelinePhase> compileFrom(
      "compile-from",
      llvm::cl::desc("Compilation phase to resume from, starting with the "
                     "following phase."),
      llvm::cl::init(IREEVMPipelinePhase::Start));
  SmallVector<std::string> compileFromPhases;
  enumerateIREEVMPipelinePhases(
      [&](IREEVMPipelinePhase phase, StringRef name, StringRef desc) {
        compileFrom.getParser().addLiteralOption(name, phase, desc);
        compileFromPhases.push_back(name.str());
      });

  llvm::cl::opt<IREEVMPipelinePhase> compileTo(
      "compile-to",
      llvm::cl::desc(
          "Compilation phase to run up until before emitting output."),
      llvm::cl::init(IREEVMPipelinePhase::End));
  SmallVector<std::string> compileToPhases;
  enumerateIREEVMPipelinePhases(
      [&](IREEVMPipelinePhase phase, StringRef name, StringRef desc) {
        compileTo.getParser().addLiteralOption(name, phase, desc);
        compileToPhases.push_back(name.str());
      });

  llvm::cl::opt<bool> emitMLIRBytecode(
      "emit-mlir-bytecode",
      llvm::cl::desc(
          "Emit bytecode when generating compile-to or VM MLIR output."),
      llvm::cl::init(false));
  llvm::cl::opt<std::optional<int64_t>, /*ExternalStorage=*/false,
                mlir::iree_compiler::BytecodeVersionParser>
      emitMLIRBytecodeVersion(
          "emit-mlir-bytecode-version",
          llvm::cl::desc("Use specified bytecode version when "
                         "generating compile-to or VM MLIR output."),
          llvm::cl::init(std::nullopt));

  // Misc options.
  llvm::cl::opt<bool> splitInputFile(
      "split-input-file",
      llvm::cl::desc("Split the input file into pieces and "
                     "process each chunk independently."),
      llvm::cl::init(false));
  llvm::cl::opt<bool> listHalTargets(
      "iree-hal-list-target-backends",
      llvm::cl::desc(
          "Lists all registered target backends for executable compilation."),
      llvm::cl::init(false), llvm::cl::ValueDisallowed,
      llvm::cl::callback([&](const bool &) {
        llvm::outs() << "Registered target backends:\n";
        ireeCompilerEnumerateRegisteredHALTargetBackends(
            [](const char *backend, void *userData) {
              llvm::outs() << "  " << backend << "\n";
            },
            nullptr);
        exit(0);
      }));
  llvm::cl::opt<bool> listPlugins(
      "iree-list-plugins", llvm::cl::desc("Lists all loaded plugins."),
      llvm::cl::init(false), llvm::cl::ValueDisallowed,
      llvm::cl::callback([&](const bool &) {
        llvm::outs() << "Loaded plugins:\n";
        ireeCompilerEnumeratePlugins(
            [](const char *pluginName, void *userData) {
              llvm::outs() << "  " << pluginName << "\n";
            },
            nullptr);
        exit(0);
      }));

  ireeCompilerGlobalInitialize();
  ireeCompilerGetProcessCLArgs(&argc, const_cast<const char ***>(&argv));
  ireeCompilerSetupGlobalCL(argc, const_cast<const char **>(argv),
                            "IREE compilation driver\n",
                            /*installSignalHandlers=*/true);

  // Stash our globals in an RAII instance.
  struct MainState {
    iree_compiler_session_t *session = ireeCompilerSessionCreate();
    iree_compiler_source_t *source = nullptr;
    iree_compiler_output_t *output = nullptr;
    SmallVector<iree_compiler_source_t *> splitSources;

    ~MainState() {
      for (auto *splitSource : splitSources) {
        ireeCompilerSourceDestroy(splitSource);
      }
      ireeCompilerOutputDestroy(output);
      ireeCompilerSourceDestroy(source);
      ireeCompilerSessionDestroy(session);
      ireeCompilerGlobalShutdown();
    }
    void handleError(iree_compiler_error_t *error) {
      const char *msg = ireeCompilerErrorGetMessage(error);
      llvm::errs() << "error opening input file: " << msg << "\n";
      ireeCompilerErrorDestroy(error);
    }
  };
  MainState s;

  // Open input and output files.
  if (auto error = ireeCompilerSourceOpenFile(s.session, inputFilename.c_str(),
                                              &s.source)) {
    s.handleError(error);
    return 1;
  }
  if (auto error =
          ireeCompilerOutputOpenFile(outputFilename.c_str(), &s.output)) {
    s.handleError(error);
    return 1;
  }

  auto processBuffer = [&](iree_compiler_source_t *source) -> bool {
    // Stash per-invocation state in an RAII instance.
    struct InvState {
      InvState(MainState &s) { inv = ireeCompilerInvocationCreate(s.session); }
      ~InvState() { ireeCompilerInvocationDestroy(inv); }
      iree_compiler_invocation_t *inv;
    };
    InvState r(s);

    ireeCompilerInvocationEnableConsoleDiagnostics(r.inv);
    ireeCompilerInvocationSetCompileFromPhase(
        r.inv,
        compileFromPhases[static_cast<int>(compileFrom.getValue())].c_str());
    ireeCompilerInvocationSetCompileToPhase(
        r.inv, compileToPhases[static_cast<int>(compileTo.getValue())].c_str());
    ireeCompilerInvocationSetVerifyIR(r.inv, verifyIR);
    if (!ireeCompilerInvocationParseSource(r.inv, source))
      return false;

    // Switch on compileMode to choose a pipeline to run.
    switch (compileMode) {
    case CompileMode::std:
      if (!ireeCompilerInvocationPipeline(r.inv, IREE_COMPILER_PIPELINE_STD))
        return false;
      break;
    case CompileMode::vm:
      break;
    case CompileMode::hal_executable: {
      // Compiling a HAL executable, it is only valid to output in that form.
      outputFormat = OutputFormat::hal_executable;
      if (!ireeCompilerInvocationPipeline(
              r.inv, IREE_COMPILER_PIPELINE_HAL_EXECUTABLE))
        return false;
      break;
    }
    case CompileMode::precompile: {
      outputFormat = OutputFormat::precompile;
      if (!ireeCompilerInvocationPipeline(r.inv,
                                          IREE_COMPILER_PIPELINE_PRECOMPILE))
        return false;
      break;
    }
    default:
      llvm::errs() << "INTERNAL ERROR: unknown compile mode\n";
      return false;
    }

    auto outputMLIR = [&](iree_compiler_invocation_t *inv,
                          iree_compiler_output_t *output) {
      if (emitMLIRBytecode) {
        return ireeCompilerInvocationOutputIRBytecode(
            inv, output, emitMLIRBytecodeVersion.value_or(-1));
      } else {
        return ireeCompilerInvocationOutputIR(inv, output);
      }
    };

    // Ending early and just emitting IR.
    if (compileTo != IREEVMPipelinePhase::End) {
      if (auto error = outputMLIR(r.inv, s.output)) {
        s.handleError(error);
        return false;
      }
      return true;
    }

    // Switch based on output format.
    iree_compiler_error_t *outputError = nullptr;
    switch (outputFormat) {
    case OutputFormat::vm_asm:
      outputError = outputMLIR(r.inv, s.output);
      break;
    case OutputFormat::vm_bytecode:
      outputError = ireeCompilerInvocationOutputVMBytecode(r.inv, s.output);
      break;
#ifdef IREE_HAVE_C_OUTPUT_FORMAT
    case OutputFormat::vm_c:
      outputError = ireeCompilerInvocationOutputVMCSource(r.inv, s.output);
      break;
#endif // IREE_HAVE_C_OUTPUT_FORMAT
    case OutputFormat::hal_executable: {
      outputError = ireeCompilerInvocationOutputHALExecutable(r.inv, s.output);
      break;
    }
    case OutputFormat::precompile: {
      outputError = outputMLIR(r.inv, s.output);
      break;
    }
    default:
      llvm::errs() << "INTERNAL ERROR: Unknown output format\n";
      return false;
    }

    if (outputError) {
      s.handleError(outputError);
      return false;
    }
    return true;
  };

  // Process buffers, either via splitting or all at once.
  if (splitInputFile) {
    if (auto error = ireeCompilerSourceSplit(
            s.source,
            [](iree_compiler_source_t *source, void *userData) {
              MainState *userState = static_cast<MainState *>(userData);
              userState->splitSources.push_back(source);
            },
            static_cast<void *>(&s))) {
      s.handleError(error);
      return 1;
    }

    bool hadFailure = false;
    for (auto *splitSource : s.splitSources) {
      if (!processBuffer(splitSource)) {
        hadFailure = true;
      }
    }
    if (hadFailure) {
      return 1;
    }
  } else {
    if (!processBuffer(s.source))
      return 1;
  }

  ireeCompilerOutputKeep(s.output);
  return 0;
}
