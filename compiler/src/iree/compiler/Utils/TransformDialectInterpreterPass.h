// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_TRANSFORMDIALECTINTERPRETERPASS_H_
#define IREE_COMPILER_UTILS_TRANSFORMDIALECTINTERPRETERPASS_H_

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {

/// Create an IREE-specific Transform dialect interpreter pass with all
/// registrations necessary for IREE.
std::unique_ptr<Pass> createTransformDialectInterpreterPass(
    llvm::StringRef transformFileName = llvm::StringRef(),
    llvm::StringRef debugPayloadRootTag = llvm::StringRef(),
    llvm::StringRef debugTransformRootTag = llvm::StringRef());

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_UTILS_TRANSFORMDIALECTINTERPRETERPASS_H_
