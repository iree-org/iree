// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_INTERFACES_TENSOR_MASKING_OP_INTERFACE_H_
#define IREE_COMPILER_CODEGEN_INTERFACES_TENSOR_MASKING_OP_INTERFACE_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

// clang-format off
#include "iree/compiler/Codegen/Interfaces/TensorMaskingOpInterface.h.inc" // IWYU pragma: export
// clang-format on

namespace mlir::iree_compiler {

void registerTensorMaskingOpInterface(DialectRegistry &registry);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_INTERFACES_TENSOR_MASKING_OP_INTERFACE_H_
