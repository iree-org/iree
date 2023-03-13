// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_TOOLUTILS_H_
#define IREE_COMPILER_UTILS_TOOLUTILS_H_

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {

// Escapes a command line component where required.
// It's easy to run afoul of quoting rules on Windows, such as when using
// spaces in the linker environment variable.
// See: https://stackoverflow.com/a/9965141
std::string escapeCommandLineComponent(const std::string &component);

// Removes escaping from a command line component if present.
StringRef unescapeCommandLineComponent(StringRef component);

// Returns the path to the first tool in |toolNames| found in the process
// executable directory (plus some hard-coded relative paths from there,
// reflecting our build structure with the LLVM submodule) or empty string if no
// tool was found.
std::string findToolFromExecutableDir(SmallVector<std::string> toolNames);

// Returns the path to the first tool in |toolNames| found in the environment,
// or empty string if no tool was found.
std::string findToolInEnvironment(SmallVector<std::string> toolNames);

// Returns the path to the first tool in |toolNames| found in the environment
// PATH or the process executable directory. Returns empty string if no tool
// was found.
std::string findTool(SmallVector<std::string> toolNames);
std::string findTool(std::string toolName);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_TOOLUTILS_H_
