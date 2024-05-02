// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_TRANSFORMDIALECTUTILS_H_
#define IREE_COMPILER_UTILS_TRANSFORMDIALECTUTILS_H_

#include "mlir/IR/BuiltinOps.h"

namespace mlir::iree_compiler {

enum StrategyRunResult {
  Success = 0,
  NotFound = 1,
  Failed = 2,
};

StrategyRunResult runTransformConfigurationStrategy(Operation *payloadRoot,
                                                    StringRef entryPointName,
                                                    ModuleOp &transformLibrary);

FailureOr<std::pair<std::optional<std::string>, std::optional<std::string>>>
parseTransformLibraryFileNameAndEntrySequence(std::string input);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_TRANSFORMDIALECTUTILS_H_
