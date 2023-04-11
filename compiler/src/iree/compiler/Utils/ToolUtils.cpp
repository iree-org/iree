// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/ToolUtils.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

#define DEBUG_TYPE "iree-tools"

namespace mlir {
namespace iree_compiler {

std::string escapeCommandLineComponent(const std::string &component) {
#if defined(_WIN32)
  return "\"" + component + "\"";
#else
  return component;
#endif  // _WIN32
}

StringRef unescapeCommandLineComponent(StringRef component) {
#if defined(_WIN32)
  if (component.starts_with("\"") && component.ends_with("\"")) {
    return component.drop_front(1).drop_back(1);
  }
#endif  // _WIN32
  return component;
}

static std::string normalizeToolNameForPlatform(const std::string &toolName) {
#if defined(_WIN32)
  return toolName + ".exe";
#else
  return toolName;
#endif  // _WIN32
}

static std::string findToolAtPath(SmallVector<std::string> normalizedToolNames,
                                  const Twine &path) {
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

static SmallVector<std::string> normalizeToolNames(
    SmallVector<std::string> toolNames) {
  SmallVector<std::string> normalizedToolNames;
  normalizedToolNames.reserve(toolNames.size());
  for (auto toolName : toolNames) {
    normalizedToolNames.push_back(normalizeToolNameForPlatform(toolName));
  }
  return normalizedToolNames;
}

std::string findToolFromExecutableDir(SmallVector<std::string> toolNames) {
  const auto &normalizedToolNames = normalizeToolNames(toolNames);
  std::string mainExecutablePath =
      llvm::sys::fs::getMainExecutable(nullptr, nullptr);
  SmallString<256> mainExecutableDir(mainExecutablePath);
  llvm::sys::path::remove_filename(mainExecutableDir);
  LLVM_DEBUG({
    llvm::dbgs() << "Searching from the executable directory "
                 << mainExecutableDir << " for one of these tools: [";
    llvm::interleaveComma(normalizedToolNames, llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  // First search the current executable's directory. This should find tools
  // within the install directory (through CMake or binary distributions).
  std::string toolPath = findToolAtPath(normalizedToolNames, mainExecutableDir);
  if (!toolPath.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Found tool in executable's directory at path "
                            << toolPath << "\n");
    return toolPath;
  }

  // Next search around in the CMake build tree.
  toolPath = findToolAtPath(normalizedToolNames,
                            mainExecutableDir + "/../llvm-project/bin/");
  if (!toolPath.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Found tool in build tree at path " << toolPath << "\n");
    return toolPath;
  }

  LLVM_DEBUG(llvm::dbgs() << "Tool not found.\n");
  return "";
}

std::string findToolInEnvironment(SmallVector<std::string> toolNames) {
  const auto &normalizedToolNames = normalizeToolNames(toolNames);
  LLVM_DEBUG({
    llvm::dbgs() << "Searching environment PATH for one of these tools: [";
    llvm::interleaveComma(normalizedToolNames, llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  for (auto toolName : normalizedToolNames) {
    if (auto result = llvm::sys::Process::FindInEnvPath("PATH", toolName)) {
      LLVM_DEBUG(llvm::dbgs() << "Found tool on environment PATH at path "
                              << result << "\n");
      return escapeCommandLineComponent(std::string(*result));
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Tool not found.\n");
  return "";
}

std::string findTool(SmallVector<std::string> toolNames) {
  // TODO(benvanik): add a test for IREE_[toolName]_PATH.

  // Search the install or build dir.
  std::string executableDirPath = findToolFromExecutableDir(toolNames);
  if (!executableDirPath.empty()) return executableDirPath;

  // Currently fall back on searching the environment.
  std::string environmentPath = findToolInEnvironment(toolNames);
  if (!environmentPath.empty()) return environmentPath;

  return "";
}

std::string findTool(std::string toolName) {
  SmallVector<std::string> toolNames = {toolName};
  return findTool(toolNames);
}

}  // namespace iree_compiler
}  // namespace mlir
