// Copyright 2020 Google LLC
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

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

// Clones an operation with new result types.
// The original operation will be erased and a new operation constructed
// in its place.
Operation *cloneWithNewResultTypes(Operation *op, TypeRange newResultTypes);

// Utility class to optimize a "closure" op, which maintains a variadic
// list of operands corresponding to entry block arguments.
class ClosureOpDce {
 public:
  ClosureOpDce(Operation *closureOp, Block &entryBlock,
               unsigned variadicOffset);

  bool needsOptimization() { return needsOperandElision || needsResultElision; }

  // Whether the operation needs to be replaced.
  bool needsNewOperation() { return needsResultElision; }

  // Performs the optimization. If the optional eraseOriginal=false and
  // needsNewOperation(), then the original will not be erased, leaving that
  // to the caller (which is needed in some pattern rewriting scenarios).
  // TODO(laurenzo): Fix OpBuilder upstream so that this eraseOriginal
  // workaround is not required to write a safe rewriter pattern that uses this
  // utility.
  Operation *optimize(OpBuilder &builder, bool eraseOriginal = true) {
    if (needsResultElision) elideUnusedResults(builder, eraseOriginal);
    if (needsOperandElision) elideUnusedOperands(builder);
    return closureOp;
  }

 private:
  void elideUnusedOperands(OpBuilder &builder);
  void elideUnusedResults(OpBuilder &builder, bool eraseOriginal);

  Operation *closureOp;
  Block &entryBlock;
  unsigned variadicOffset;
  llvm::SmallVector<llvm::Optional<BlockArgument>, 8> blockArgReplacements;
  llvm::SmallMapVector<Value, BlockArgument, 8> argToBlockMap;
  bool needsOperandElision = false;
  bool needsResultElision = false;
};

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
