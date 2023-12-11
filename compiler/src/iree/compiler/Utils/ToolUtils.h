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

namespace mlir::iree_compiler {

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

// Finds a bundled directory containing platform libraries for the given
// platform name, returning an empty string if not found. We store bundled
// platform libraries in a directory like:
//   iree_platform_libs/{platformName}
// adjacent to the shared library hosting the compiler (i.e. this entry
// point). On a Posix system, this will typically be in a "lib" dir and
// on Windows, it will be adjacent to executables (i.e. a "bin" or "tools"
// dir). Note that if installed to a system library directory on a Posix
// system, this would be under something like:
//   /usr/lib/iree_platform_libs/{platformName}
// This is not atypical to how other dependencies are located in a qualified
// lib directory.
std::string findPlatformLibDirectory(StringRef platformName);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_TOOLUTILS_H_
