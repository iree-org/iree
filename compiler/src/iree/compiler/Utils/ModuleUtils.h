// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_MODULEUTILS_H_
#define IREE_COMPILER_UTILS_MODULEUTILS_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace iree_compiler {

// Destructively merges |sourceOp| into |targetOp| using |targetBuilder|.
// |targetSymbolTable| is updated with the new symbols.
//
// If a private symbol in |sourceOp| conflicts with another symbol
// (public or private) in |targetOp|, it will be renamed.
//
// Fails if a public symbol in the |sourceOp| conflicts with another public
// symbol in the |targetOp|.
LogicalResult mergeModuleInto(Operation *sourceOp, Operation *targetOp,
                              SymbolTable &targetSymbolTable,
                              OpBuilder &targetBuilder);

// Merges an MLIR module text in |source| into the |targetOp| using
// |targetBuilder|. See mergeModuleInto for details on symbol renaming.
LogicalResult mergeSourceModuleInto(Location loc, StringRef source,
                                    Operation *targetOp,
                                    SymbolTable &targetSymbolTable,
                                    OpBuilder &targetBuilder);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_MODULEUTILS_H_
