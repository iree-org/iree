// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_MIPS_IR_MIPSDIALECT_H_
#define IREE_COMPILER_DIALECT_MIPS_IR_MIPSDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

// clang-format off
// MIPSDialect.h.inc is generated from MIPSOps.td via:
//   --dialect=mips --gen-dialect-decls MIPSDialect.h.inc
#include "iree/compiler/Dialect/MIPS/IR/MIPSDialect.h.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::MIPS {

// Register external BufferizableOpInterface models for MIPS ops.
// Call this from registerIreeDialects() before bufferization runs.
void registerMIPSBufferizableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry);

} // namespace mlir::iree_compiler::IREE::MIPS

#endif // IREE_COMPILER_DIALECT_MIPS_IR_MIPSDIALECT_H_
