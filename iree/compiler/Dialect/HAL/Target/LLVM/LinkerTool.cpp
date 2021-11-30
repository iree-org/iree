// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVM/LinkerTool.h"

#include "llvm/Support/Process.h"

#define DEBUG_TYPE "llvmaot-linker"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Sanitizes potentially user provided portions of a file name by replacing
// all but a small set of alpha numeric and safe punctuation characters with
// '_'. This is intended for components of temporary files that are uniqued
// independently, where the input is meant to aid debugability but does not
// need to be retained verbatim.
static void sanitizeFilePart(llvm::SmallVectorImpl<char> &part) {
  for (char &c : part) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.')
      continue;
    c = '_';
  }
}

// static
Artifact Artifact::createTemporary(StringRef prefix, StringRef suffix) {
  llvm::SmallString<8> prefixCopy(prefix);
  llvm::SmallString<8> suffixCopy(suffix);
  sanitizeFilePart(prefixCopy);
  sanitizeFilePart(suffixCopy);

  llvm::SmallString<32> filePath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          prefixCopy, suffixCopy, filePath)) {
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

Optional<std::vector<int8_t>> Artifact::read() const {
  auto fileData = llvm::MemoryBuffer::getFile(path);
  if (!fileData) {
    llvm::errs() << "failed to load library output file '" << path << "'";
    return llvm::None;
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
  if (libraryFile.outputFile) libraryFile.outputFile->keep();
  if (debugFile.outputFile) debugFile.outputFile->keep();
  for (auto &file : otherFiles) {
    file.outputFile->keep();
  }
}

std::string LinkerTool::getSystemToolPath() const {
  // Always use the -iree-llvm-system-linker-path flag when specified as it's
  // explicitly telling us what to use.
  if (!targetOptions.linkerPath.empty()) {
    return targetOptions.linkerPath;
  }

  // Allow users to override the automatic search with an environment variable.
  char *linkerPath = std::getenv("IREE_LLVMAOT_SYSTEM_LINKER_PATH");
  if (linkerPath) {
    return std::string(linkerPath);
  }

  // Fallback to other searches as specified by the LinkerTool implementation.
  return "";
}

// It's easy to run afoul of quoting rules on Windows, such as when using
// spaces in the linker environment variable.
// See: https://stackoverflow.com/a/9965141
static std::string escapeCommandLineComponent(const std::string &commandLine) {
#if defined(_MSC_VER)
  return "\"" + commandLine + "\"";
#else
  return commandLine;
#endif  // _MSC_VER
}

static std::string normalizeToolNameForPlatform(const std::string &toolName) {
#if defined(_MSC_VER)
  return toolName + ".exe";
#else
  return toolName;
#endif  // _MSC_VER
}

static std::string findToolAtPath(
    SmallVector<std::string, 4> normalizedToolNames, const Twine &path) {
  LLVM_DEBUG(llvm::dbgs() << "Searching for tool at path '" << path << "'\n");
  for (auto toolName : normalizedToolNames) {
    SmallString<256> pathStorage;
    llvm::sys::path::append(pathStorage, path, toolName);

    if (llvm::sys::fs::exists(pathStorage)) {
      llvm::sys::fs::make_absolute(pathStorage);
      (void)llvm::sys::path::remove_dots(pathStorage, /*remove_dot_dot=*/true);
      return escapeCommandLineComponent(std::string(pathStorage));
    }
  }

  return "";
}

LogicalResult LinkerTool::runLinkCommand(std::string commandLine,
                                         StringRef env) {
  LLVM_DEBUG(llvm::dbgs() << "Running linker command:\n"
                          << env << " " << commandLine);
  if (!env.empty()) {
#if defined(_MSC_VER)
    commandLine = ("set " + env + " && " + commandLine).str();
#else
    commandLine = (env + " " + commandLine).str();
#endif  // _MSC_VER
  }
  int exitCode = system(commandLine.c_str());
  if (exitCode == 0) return success();
  llvm::errs() << "Linking failed; escaped command line returned exit code "
               << exitCode << ":\n\n"
               << commandLine << "\n\n";
  return failure();
}

std::string LinkerTool::findToolInEnvironment(
    SmallVector<std::string, 4> toolNames) const {
  SmallVector<std::string, 4> normalizedToolNames;
  normalizedToolNames.reserve(toolNames.size());
  for (auto toolName : toolNames) {
    normalizedToolNames.push_back(normalizeToolNameForPlatform(toolName));
  }
  LLVM_DEBUG(llvm::dbgs() << "Searching environment for one of these tools: [";
             llvm::interleaveComma(normalizedToolNames, llvm::dbgs());
             llvm::dbgs() << "]\n");

  std::string mainExecutablePath =
      llvm::sys::fs::getMainExecutable(nullptr, nullptr);
  SmallString<256> mainExecutableDir(mainExecutablePath);
  llvm::sys::path::remove_filename(mainExecutableDir);

  // First search the current executable's directory. This should find tools
  // within the install directory (through CMake or binary distributions).
  std::string toolPath = findToolAtPath(normalizedToolNames, mainExecutableDir);
  if (!toolPath.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Found tool in executable's directory at path "
                            << toolPath << "\n");
    return toolPath;
  }

  // Next search around in the CMake build tree.
  toolPath = findToolAtPath(
      normalizedToolNames,
      mainExecutableDir + "/../../third_party/llvm-project/llvm/bin/");
  if (!toolPath.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Found tool in build tree at path " << toolPath << "\n");
    return toolPath;
  }

  // Finally search the environment path.
  for (auto toolName : normalizedToolNames) {
    if (auto result = llvm::sys::Process::FindInEnvPath("PATH", toolName)) {
      LLVM_DEBUG(llvm::dbgs() << "Found tool on environment PATH at path "
                              << toolPath << "\n");
      return escapeCommandLineComponent(std::string(*result));
    }
  }

  return "";
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
