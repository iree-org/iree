// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Options.h"

IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::TuningSpecOptions);

namespace mlir::iree_compiler {

void TuningSpecOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory category(
      "IREE codegen tuning spec options",
      "Options for controlling codegen tuning spec loading and configuration.");

  binder.opt<std::string>(
      "iree-codegen-tuning-spec-path", tuningSpecPath,
      llvm::cl::desc("File path to a module containing a tuning spec "
                     "(transform dialect library)."),
      llvm::cl::cat(category));

  binder.opt<bool>(
      "iree-codegen-enable-default-tuning-specs", enableDefaultTuningSpecs,
      llvm::cl::desc("Whether to enable default tuning spec transform "
                     "libraries shipped with the compiler."),
      llvm::cl::cat(category));

  binder.opt<std::string>(
      "iree-codegen-dump-tuning-specs-to", tuningSpecDumpDir,
      llvm::cl::desc("Dump the final tuning spec modules to the specified "
                     "directory. When set to '-', prints the tuning spec to "
                     "stdout."),
      llvm::cl::cat(category));
}

} // namespace mlir::iree_compiler
