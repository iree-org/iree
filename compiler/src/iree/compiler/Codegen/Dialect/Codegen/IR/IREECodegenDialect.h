// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_IREECODEGEN_DIALECT_H_
#define IREE_COMPILER_CODEGEN_DIALECT_IREECODEGEN_DIALECT_H_

#include <mutex>

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

// clang-format off: must be included after all LLVM/MLIR eaders
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler {

void registerUKernelBufferizationInterface(DialectRegistry &registry);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_DIALECT_IREECODEGEN_DIALECT_H_
