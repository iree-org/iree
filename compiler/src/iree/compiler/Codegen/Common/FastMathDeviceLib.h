// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_FASTMATHDEVICELIB_H_
#define IREE_COMPILER_CODEGEN_COMMON_FASTMATHDEVICELIB_H_

#include "mlir/IR/PatternMatch.h"
#include <functional>

namespace mlir::iree_compiler {

// Populates patterns that implement device library versions of math operations
void populateDeviceLibMathPatterns(
    RewritePatternSet &patterns,
    const std::function<bool(StringRef)> &predicate);

} // namespace mlir::iree_compiler

#endif  // IREE_COMPILER_CODEGEN_COMMON_FASTMATHDEVICELIB_H_ 
