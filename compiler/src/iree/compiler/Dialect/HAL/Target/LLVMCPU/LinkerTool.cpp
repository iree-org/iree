// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/LinkerTool.h"

#include "iree/compiler/Utils/StringUtils.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"

#define DEBUG_TYPE "iree-tools"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// static
Artifact Artifact::fromFile(StringRef path) { return {path.str(), nullptr}; }

// static
Artifact Artifact::createTemporary(StringRef prefix, StringRef suffix) {
  auto sanitizedPrefix = sanitizeFileName(prefix);
  auto sanitizedSuffix = sanitizeFileName(suffix);

  llvm::SmallString<32> filePath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          sanitizedPrefix, sanitizedSuffix, filePath)) {
    llvm::errs() << "failed to generate temporary file: " << error.message();
    return {};
  }
  std::error_code error;
  auto file = std::make_unique<llvm::ToolOutputFile>(filePath, error,
                                                     llvm::sys::fs::OF_None);
  if (error) {
    llvm::errs() << "failed to open temporary file '" << filePath
                 << "': " << error.message();
    return {};
  }
  return {filePath.str().str(), std::move(file)};
}

// static
Artifact Artifact::createVariant(StringRef basePath, StringRef suffix) {
  SmallString<32> filePath(basePath);
  llvm::sys::path::replace_extension(filePath, suffix);
  std::error_code error;
  auto file = std::make_unique<llvm::ToolOutputFile>(filePath, error,
                                                     llvm::sys::fs::OF_Append);
  if (error) {
    llvm::errs() << "failed to open temporary file '" << filePath
                 << "': " << error.message();
    return {};
  }
  return {filePath.str().str(), std::move(file)};
}

void Artifact::keep() const {
  if (outputFile) outputFile->keep();
}

Optional<std::vector<int8_t>> Artifact::read() const {
  auto fileData = llvm::MemoryBuffer::getFile(path);
  if (!fileData) {
    llvm::errs() << "failed to load library output file '" << path << "'";
    return std::nullopt;
  }
  auto sourceBuffer = fileData.get()->getBuffer();
  std::vector<int8_t> resultBuffer(sourceBuffer.size());
  std::memcpy(resultBuffer.data(), sourceBuffer.data(), sourceBuffer.size());
  return resultBuffer;
}

bool Artifact::readInto(raw_ostream &targetStream) const {
  // NOTE: we could make this much more efficient if we read in the file a
  // chunk at a time and piped it along to targetStream. I couldn't find
  // anything in LLVM that did this, for some crazy reason, but since we are
  // dealing with binaries that can be 10+MB here it'd be nice if we could avoid
  // reading them all into memory.
  auto fileData = llvm::MemoryBuffer::getFile(path);
  if (!fileData) {
    llvm::errs() << "failed to load library output file '" << path << "'";
    return false;
  }
  auto sourceBuffer = fileData.get()->getBuffer();
  targetStream.write(sourceBuffer.data(), sourceBuffer.size());
  return true;
}

void Artifact::close() { outputFile->os().close(); }

void Artifacts::keepAllFiles() {
  libraryFile.keep();
  debugFile.keep();
  for (auto &file : otherFiles) {
    file.keep();
  }
}

std::string LinkerTool::getSystemToolPath() const {
  // Always use the --iree-llvmcpu-system-linker-path flag when specified as
  // it's explicitly telling us what to use.
  if (!targetOptions.systemLinkerPath.empty()) {
    return targetOptions.systemLinkerPath;
  }

  // Allow users to override the automatic search with an environment variable.
  char *linkerPath = std::getenv("IREE_LLVM_SYSTEM_LINKER_PATH");
  if (linkerPath) {
    return std::string(linkerPath);
  }

  // Fallback to other searches as specified by the LinkerTool implementation.
  return "";
}

LogicalResult LinkerTool::runLinkCommand(std::string commandLine,
                                         StringRef env) {
  LLVM_DEBUG(llvm::dbgs() << "Running linker command:\n"
                          << env << " " << commandLine << "\n");
  if (!env.empty()) {
#if defined(_WIN32)
    commandLine = ("set " + env + " && " + commandLine).str();
#else
    commandLine = (env + " " + commandLine).str();
#endif  // _WIN32
  } else {
    commandLine = escapeCommandLineComponent(commandLine);
  }
  int exitCode = system(commandLine.c_str());
  if (exitCode == 0) return success();
  llvm::errs() << "Linking failed; escaped command line returned exit code "
               << exitCode << ":\n\n"
               << commandLine << "\n\n";
  return failure();
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
