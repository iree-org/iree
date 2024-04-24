// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_TARGETOPTIONS_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_TARGETOPTIONS_H_

#include <string>
#include <vector>

#include "iree/compiler/Utils/OptionUtils.h"

namespace mlir::iree_compiler::IREE::HAL {

// TODO(benvanik): remove this and replace with the pass pipeline options.
// Controls executable translation targets.
struct TargetOptions {
  // TODO(benvanik): multiple targets of the same type, etc.
  std::vector<std::string> targets;

  // Coarse debug level for executable translation across all targets.
  // Each target backend can use this to control its own flags, with values
  // generally corresponding to the gcc-style levels 0-3:
  //   0: no debug information
  //   1: minimal debug information
  //   2: default debug information
  //   3: maximal debug information
  int debugLevel;

  // Default path to write executable files into.
  std::string executableFilesPath;

  // A path to write individual executable source listings into (before
  // configuration).
  std::string executableSourcesPath;

  // A path to write individual executable source listings into (after
  // configuration).
  std::string executableConfigurationsPath;

  // A path to write standalone executable benchmarks into.
  std::string executableBenchmarksPath;

  // A path to write executable intermediates into.
  std::string executableIntermediatesPath;

  // A path to write translated and serialized executable binaries into.
  std::string executableBinariesPath;

  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<TargetOptions>;
};

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_TARGETOPTIONS_H_
