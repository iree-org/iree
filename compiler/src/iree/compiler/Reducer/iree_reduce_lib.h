// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_TOOLS_IREE_REDUCER_LIB_H
#define IREE_COMPILER_TOOLS_IREE_REDUCER_LIB_H

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::iree_compiler::Reducer {

struct ReducerConfig {
  ReducerConfig() = delete;

  explicit ReducerConfig(StringRef testScript, bool useBytecode)
      : testScript(testScript), useBytecode(useBytecode) {}

  // Path to the test script to run on the reduced program.
  StringRef testScript;
  // Flag to indicate whether the test script can use bytecode or not.
  bool useBytecode;
};

Operation *ireeRunReducingStrategies(OwningOpRef<Operation *> module,
                                     ReducerConfig &config);

} // namespace mlir::iree_compiler::Reducer

#endif // IREE_COMPILER_TOOLS_IREE_REDUCER_LIB_H
