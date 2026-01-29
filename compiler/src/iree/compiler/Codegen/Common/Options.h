// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_OPTIONS_H_
#define IREE_COMPILER_CODEGEN_COMMON_OPTIONS_H_

#include "iree/compiler/Utils/OptionUtils.h"

namespace mlir::iree_compiler {

// Options controlling codegen tuning spec configuration.
struct TuningSpecOptions {
  // File path to a module containing a tuning spec (transform dialect library).
  std::string tuningSpecPath = "";

  // Whether to enable default tuning spec transform libraries shipped with the
  // compiler.
  bool enableDefaultTuningSpecs = false;

  // Dump the final tuning spec modules to the specified directory.
  // When set to '-', prints the tuning spec to stdout.
  std::string tuningSpecDumpDir = "";

  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<TuningSpecOptions>;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_OPTIONS_H_
