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

#ifndef IREE_COMPILER_UTILS_OPUTILS_H_
#define IREE_COMPILER_UTILS_OPUTILS_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "third_party/llvm/llvm/include/llvm/ADT/None.h"
#include "third_party/llvm/llvm/include/llvm/ADT/Optional.h"
#include "third_party/llvm/llvm/include/llvm/ADT/SetVector.h"

namespace mlir {
namespace iree_compiler {

// Recursively removes the given operations and all of their inputs that become
// unused.
void removeDeadOperations(llvm::SetVector<Operation *> &deadOperations);

// Replaces all uses of |oldValue| with |newValue| that are after |userOp|
// within the same block.
void replaceSubsequentUses(Operation *userOp, Value *oldValue, Value *newValue);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_OPUTILS_H_
