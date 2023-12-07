// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UTILS_CPUUTILS_H_
#define IREE_COMPILER_CODEGEN_UTILS_CPUUTILS_H_

#include "iree/compiler/Codegen/Utils/Utils.h"

namespace mlir::iree_compiler {

/// Find the root operation for the dispatch region. The priority is:
///   1. A Linalg operation that has reduction loops.
///   2. Any other Linalg op or LinalgExt op.
///   3. An operation that implements TilingInterface.
/// If there are multiple operations meeting the same priority, the one closer
/// to the end of the function is the root op.
FailureOr<Operation *> getRootOperation(ArrayRef<Operation *> computeOps);

// Finds the MMT4D ukernel type listed in:
// runtime/src/iree/builtins/ukernel/exported_bits.h
// Returns IREE_UK_FLAG_MMT4D_TYPE_NONE if not found.
uint32_t findMmt4dUkernelType(Type lhsElemType, Type rhsElemType,
                              Type outElemType);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_UTILS_CPUUTILS_H_
