// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#define DEBUG_TYPE "iree-tools"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

static StringRef fixupArg(StringRef arg) {
  // HACK: pass pipeline parsing doesn't handle strings with spaces and the only
  // way to get them through (I could find) is to double quote them. This
  // unfortunately breaks native path tokenization for single executable quoted
  // paths.
  if (arg.starts_with("\"") && arg.ends_with("\"")) {
    arg = arg.drop_front(1).drop_back(1);
  }
  return arg;
}

static LogicalResult buildPassPipeline(StringRef rawPipelineStr,
                                       OpPassManager &passManager) {
  auto pipelineStr = fixupArg(rawPipelineStr);

  // Strip the `builtin.module(...)` that surrounds the pass pipeline
  // description. On failure an assertion is triggered, but in release builds
  // it just will silently return and not raise an error. There is no
  // way to handle the error in caller currently.
  StringRef text(pipelineStr);
  size_t pos = text.find_first_of("(");
  if (pos == StringRef::npos) {
    llvm::errs() << "ERROR: expected preprocessing pass pipeline string to be "
                    "nested within `builtin.module(..)`; got `"
                 << pipelineStr << "`\n";
    return failure();
  }
  if (text.substr(0, pos) != "builtin.module") {
    llvm::errs() << "ERROR: expected preprocessing pass pipeline string to be "
                    "nested within `builtin.module(..)`\n";
    return failure();
  }
  if (text.back() != ')') {
    llvm::errs() << "ERROR: mismatched parenthesis in pass pipeline `"
                 << pipelineStr << "`\n";
    return failure();
  }
  text = text.substr(pos + 1);
  if (failed(parsePassPipeline(text.drop_back(), passManager))) {
    llvm::errs() << "ERROR: failed to parse textual pass pipeline `"
                 << pipelineStr << "`\n";
    return failure();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "Preprocessing pass pipeline : ";
    passManager.printAsTextualPipeline(llvm::dbgs());
  });
  return success();
}

// Replaces the contents and attributes on |executableOp| with those of the
// given |replacementOp|.
static void replaceExecutableContents(IREE::HAL::ExecutableOp executableOp,
                                      IREE::HAL::ExecutableOp replacementOp) {
  // Drop all dialect attrs from the original and use those of the replacement.
  for (auto attr :
       llvm::make_early_inc_range(executableOp->getDialectAttrs())) {
    executableOp->removeAttr(attr.getName());
  }
  executableOp->setDialectAttrs(replacementOp->getDialectAttrs());

  // Drop the original body and take the replacement one.
  executableOp.getBody().takeBody(replacementOp.getBody());
}

static LogicalResult preprocessWithCommand(IREE::HAL::ExecutableOp executableOp,
                                           StringRef rawCommand) {
  auto command = fixupArg(rawCommand);

  // Setup IO redirects used to pass around the executable MLIR contents.
  SmallString<32> stdinFile, stdoutFile, stderrFile;
  int inputFd = 0;
  llvm::sys::fs::createTemporaryFile("executable-preprocessor-stdin", "",
                                     inputFd, stdinFile);
  llvm::sys::fs::createTemporaryFile("executable-preprocessor-stdout", "",
                                     stdoutFile);
  llvm::sys::fs::createTemporaryFile("executable-preprocessor-stderr", "",
                                     stderrFile);
  llvm::FileRemover stdinRemover(stdinFile.c_str());
  llvm::FileRemover stdoutRemover(stdoutFile.c_str());
  llvm::FileRemover stderrRemover(stderrFile.c_str());
  std::optional<StringRef> redirects[] = {
      stdinFile.str(),
      stdoutFile.str(),
      stderrFile.str(),
  };

  // Serialize the executable contents.
  // NOTE: this is currently being done in text format as it's easier to work
  // with. We'll probably want to flip to binary or make it an option if we
  // ever want to support versioning.
  {
    llvm::raw_fd_ostream inputStream(inputFd, /*shouldClose=*/true);
    executableOp.print(inputStream,
                       OpPrintingFlags().useLocalScope().enableDebugInfo());
    inputStream << "\n";  // newline at end of file
  }

  // LLVM wants all the args split up to launch the command so we tokenize here.
  // This is exactly how the LLVM command line parser does it with a macro
  // switch.
  llvm::BumpPtrAllocator scratchAllocator;
  llvm::StringSaver stringSaver(scratchAllocator);
  SmallVector<const char *> rawArgs;
#ifdef _WIN32
  auto Tokenize = llvm::cl::TokenizeWindowsCommandLine;
#else
  auto Tokenize = llvm::cl::TokenizeGNUCommandLine;
#endif  // _WIN32
  Tokenize(command, stringSaver, rawArgs, /*MarkEOLs=*/false);
  SmallVector<StringRef> args;
  for (auto rawArg : rawArgs) args.push_back(StringRef(rawArg));

  // Try to find the tool either by absolute path or by looking it up in env.
  auto tool = findTool(args[0].str());
  if (tool.empty()) {
    llvm::errs() << "ERROR: failed to find tool `" << args[0] << "` in PATH\n";
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Launching hal.executable preprocessor: ";
    for (auto arg : args) llvm::dbgs() << arg << " ";
    llvm::dbgs() << " 1> " << stdoutFile.str() << " 2> " << stderrFile.str()
                 << "\n";
  });

  // Launch the preprocessing tool. Note that this may fail for tons of reasons
  // (bad program path, bad command line, bad system state, bad IO, bad
  // preprocessor, etc).
  std::string errorMessage;
  int runResult = llvm::sys::ExecuteAndWait(
      unescapeCommandLineComponent(tool), args, /*Env=*/std::nullopt,
      /*Redirects=*/redirects,
      /*SecondsToWait=*/0, /*MemoryLimit=*/0, /*ErrMsg=*/&errorMessage);
  if (runResult != 0) {
    llvm::errs() << "ERROR: preprocessor invocation failed: " << errorMessage
                 << "\n";
    llvm::errs() << "ERROR: tool stderr preserved at " << stderrFile.str()
                 << "\n";
    stderrRemover.releaseFile();
    return failure();
  }

  // NOTE: we could check for empty stdout and quickly skip replacement.

  // Deserialize the resulting contents.
  mlir::ParserConfig parserConfig(executableOp.getContext());
  auto parsedOpRef = mlir::parseSourceFile(stdoutFile.str(), parserConfig);
  if (!parsedOpRef) {
    llvm::errs() << "ERROR: preprocessor failed to parse command output\n";
    llvm::errs() << "ERROR: tool stdout preserved at " << stdoutFile.str()
                 << "\n";
    stdoutRemover.releaseFile();
    return failure();
  }

  // Find the expected executable. This may come back as either an executable
  // nested in a module or the executable itself.
  IREE::HAL::ExecutableOp replacementOp;
  if (auto tryCast = dyn_cast<IREE::HAL::ExecutableOp>(*parsedOpRef)) {
    replacementOp = tryCast;
  } else if (auto moduleOp = dyn_cast<mlir::ModuleOp>(*parsedOpRef)) {
    auto executableOps = moduleOp.getOps<IREE::HAL::ExecutableOp>();
    if (!executableOps.empty()) {
      replacementOp = *executableOps.begin();
    }
  }
  if (!replacementOp) {
    llvm::errs()
        << "ERROR: preprocessor did not output a hal.executable as expected\n";
    llvm::errs() << "ERROR: tool stdout preserved at " << stdoutFile.str()
                 << "\n";
    stdoutRemover.releaseFile();
    return failure();
  }

  // Replace the executable with the contents of the file.
  replaceExecutableContents(executableOp, replacementOp);

  return success();
}

class PreprocessExecutablesPass
    : public PassWrapper<PreprocessExecutablesPass,
                         OperationPass<IREE::HAL::ExecutableOp>> {
 public:
  PreprocessExecutablesPass() = default;
  PreprocessExecutablesPass(const PreprocessExecutablesPass &pass) {}
  PreprocessExecutablesPass(std::optional<std::string> pipeline,
                            std::optional<std::string> command) {
    if (pipeline.has_value()) {
      this->pipeline = std::move(pipeline).value();
    } else if (command.has_value()) {
      this->command = std::move(command).value();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    if (pipeline.hasValue()) {
      OpPassManager passManager(IREE::HAL::ExecutableOp::getOperationName());
      // Can't signal failure here; things will fail during pass execution where
      // we can signalPassFailure.
      if (succeeded(buildPassPipeline(pipeline, passManager))) {
        passManager.getDependentDialects(registry);
      }
    }
  }

  StringRef getArgument() const override {
    return "iree-hal-preprocess-executables";
  }

  StringRef getDescription() const override {
    return "Preprocesses each executable with a pass pipeline or external "
           "tool.";
  }

  void runOnOperation() override {
    auto executableOp = getOperation();
    if (pipeline.hasValue()) {
      OpPassManager passManager(executableOp.getOperationName());
      if (failed(buildPassPipeline(pipeline, passManager))) {
        llvm::errs() << "ERROR: failed to parse preprocessing pipeline `"
                     << pipeline << "`\n";
        return signalPassFailure();
      }
      if (failed(runPipeline(passManager, executableOp))) {
        llvm::errs() << "ERROR: failed to preprocess executable `"
                     << executableOp.getName() << "` using pipeline `"
                     << pipeline << "`\n";
        return signalPassFailure();
      }
    } else if (command.hasValue()) {
      if (failed(preprocessWithCommand(executableOp, command))) {
        llvm::errs() << "ERROR: failed to preprocess executable `"
                     << executableOp.getName() << "` using command `" << command
                     << "`\n";
        return signalPassFailure();
      }
    }
  }

 private:
  Option<std::string> pipeline{
      *this,
      "pipeline",
      llvm::cl::desc("Pass pipeline used to preprocess the executable."),
      llvm::cl::init(""),
  };
  Option<std::string> command{
      *this,
      "command",
      llvm::cl::desc("Shell command used to preprocess the executable."),
      llvm::cl::init(""),
  };
};

std::unique_ptr<OperationPass<IREE::HAL::ExecutableOp>>
createPreprocessExecutablesPass(std::string rawCommand) {
  auto command = fixupArg(rawCommand);
  if (command.starts_with("builtin.module")) {
    return createPreprocessExecutablesWithPipelinePass(command.str());
  } else {
    return createPreprocessExecutablesWithToolPass(command.str());
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableOp>>
createPreprocessExecutablesWithPipelinePass(std::string pipeline) {
  return std::make_unique<PreprocessExecutablesPass>(std::move(pipeline),
                                                     std::nullopt);
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableOp>>
createPreprocessExecutablesWithToolPass(std::string command) {
  return std::make_unique<PreprocessExecutablesPass>(std::nullopt,
                                                     std::move(command));
}

static PassRegistration<PreprocessExecutablesPass> pass([] {
  return std::make_unique<PreprocessExecutablesPass>();
});

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
