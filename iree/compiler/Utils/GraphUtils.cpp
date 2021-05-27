// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/GraphUtils.h"

#include <algorithm>
#include <functional>

namespace mlir {
namespace iree_compiler {

std::vector<Operation *> sortOpsTopologically(
    const llvm::SetVector<Operation *> &unsortedOps) {
  llvm::SetVector<Operation *> unmarkedOps;
  unmarkedOps.insert(unsortedOps.begin(), unsortedOps.end());
  llvm::SetVector<Operation *> markedOps;

  using VisitFn = std::function<void(Operation * op)>;
  VisitFn visit = [&](Operation *op) {
    if (markedOps.count(op) > 0) return;
    for (auto result : op->getResults()) {
      for (auto *user : result.getUsers()) {
        // Don't visit ops not in our set.
        if (unsortedOps.count(user) == 0) continue;
        visit(user);
      }
    }
    markedOps.insert(op);
  };

  while (!unmarkedOps.empty()) {
    auto *op = unmarkedOps.pop_back_val();
    visit(op);
  }

  auto sortedOps = markedOps.takeVector();
  std::reverse(sortedOps.begin(), sortedOps.end());
  return sortedOps;
}

}  // namespace iree_compiler
}  // namespace mlir
