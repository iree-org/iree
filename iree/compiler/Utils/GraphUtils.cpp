// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
    for (auto *result : op->getResults()) {
      for (auto *user : result->getUsers()) {
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
