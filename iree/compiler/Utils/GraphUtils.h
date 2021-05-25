// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_GRAPHUTILS_H_
#define IREE_COMPILER_UTILS_GRAPHUTILS_H_

#include <vector>

#include "llvm/ADT/SetVector.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace iree_compiler {

// Puts all of the |unsortedOps| into |sortedOps| in an arbitrary topological
// order.
// https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
//
// Preconditions: |unsortedOps| has no cycles within the set of ops.
std::vector<Operation *> sortOpsTopologically(
    const llvm::SetVector<Operation *> &unsortedOps);
template <int N>
SmallVector<Operation *, N> sortOpsTopologically(
    const SmallVector<Operation *, N> &unsortedOps) {
  auto result = sortOpsTopologically(
      llvm::SetVector<Operation *>(unsortedOps.begin(), unsortedOps.end()));
  return SmallVector<Operation *, N>(result.begin(), result.end());
}

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_GRAPHUTILS_H_
