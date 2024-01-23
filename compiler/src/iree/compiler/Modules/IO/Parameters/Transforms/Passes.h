// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_MODULES_IO_PARAMETERS_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_MODULES_IO_PARAMETERS_TRANSFORMS_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::IREE::IO::Parameters {

//// Moves all global initial values to a parameter archive.
// std::unique_ptr<Pass>
// createParameterizeGlobalsPass(std::string archivePath = "",
//                               std::string parameterNamespace = "");

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "iree/compiler/Modules/IO/Parameters/Transforms/Passes.h.inc" // IWYU pragma: keep

void registerParametersPasses();

} // namespace mlir::iree_compiler::IREE::IO::Parameters

#endif // IREE_COMPILER_MODULES_IO_PARAMETERS_TRANSFORMS_PASSES_H_
