// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_TRANSFORMOPS_UTILTRANSFORMOPS_H_
#define IREE_COMPILER_DIALECT_UTIL_TRANSFORMOPS_UTILTRANSFORMOPS_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Util/TransformOps/UtilTransformOps.h.inc"

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace mlir::iree_compiler::IREE::Util {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace mlir::iree_compiler::IREE::Util

#endif // IREE_COMPILER_DIALECT_UTIL_TRANSFORMOPS_UTILTRANSFORMOPS_H_
