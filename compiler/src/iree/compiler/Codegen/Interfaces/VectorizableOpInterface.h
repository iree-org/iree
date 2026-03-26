// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_INTERFACES_VECTORIZABLE_OP_INTERFACE_H_
#define IREE_COMPILER_CODEGEN_INTERFACES_VECTORIZABLE_OP_INTERFACE_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

// clang-format off
#include "iree/compiler/Codegen/Interfaces/VectorizableOpInterface.h.inc" // IWYU pragma: export
// clang-format on

namespace mlir::iree_compiler {

/// Registers external models for VectorizableOpInterface.
void registerVectorizableOpInterfaceExternalModels(DialectRegistry &registry);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_INTERFACES_VECTORIZABLE_OP_INTERFACE_H_
