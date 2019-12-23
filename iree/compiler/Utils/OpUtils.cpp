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

#include "iree/compiler/Utils/OpUtils.h"

namespace mlir {
namespace iree_compiler {

void removeDeadOperations(llvm::SetVector<Operation *> &deadOperations) {
  while (!deadOperations.empty()) {
    auto *op = deadOperations.front();
    deadOperations.erase(deadOperations.begin());
    for (auto operand : op->getOperands()) {
      // TODO(benvanik): add check for op side effects.
      if (operand->hasOneUse()) {
        deadOperations.insert(operand->getDefiningOp());
      }
    }
    op->erase();
  }
}

void replaceSubsequentUses(Operation *userOp, ValuePtr oldValue,
                           ValuePtr newValue) {
  for (auto &use : oldValue->getUses()) {
    if (userOp->isBeforeInBlock(use.getOwner())) {
      use.set(newValue);
    }
  }
}

}  // namespace iree_compiler
}  // namespace mlir
