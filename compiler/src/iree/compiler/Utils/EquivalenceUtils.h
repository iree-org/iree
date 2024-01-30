// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_EQUIVALENCEUTILS_H_
#define IREE_COMPILER_UTILS_EQUIVALENCEUTILS_H_

#include "mlir/IR/Operation.h"

namespace mlir::iree_compiler {

// Recursively compares two regions for structural equivalence.
//
// Structural equivalence ensures that operations in both regions
// |lhs| and |rhs| have the same attributes and same use-def structure.
bool isStructurallyEquivalentTo(Region &lhs, Region &rhs);

// Recursively compares two operations for structural equivalence.
//
// Structural equivalence ensures that operations in the regions of both the
// |lhs| and |rhs| have the same attributes and same use-def structure.
bool isStructurallyEquivalentTo(Operation &lhs, Operation &rhs);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_EQUIVALENCEUTILS_H_
