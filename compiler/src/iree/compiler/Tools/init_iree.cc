// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.

#include "iree/compiler/Tools/init_iree.h"

#include "iree/compiler/Tools/version.h"
#include "llvm/Support/CommandLine.h"

static void versionPrinter(llvm::raw_ostream &os) {
  os << "IREE (https://openxla.github.io/):\n  ";
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

mlir::iree_compiler::InitIree::InitIree(int &argc, char **&argv)
    : init_llvm_(argc, argv) {
  llvm::setBugReportMsg(
      "Please report issues to https://github.com/openxla/iree/issues and "
      "include the crash backtrace.\n");
  llvm::cl::SetVersionPrinter(versionPrinter);
}
