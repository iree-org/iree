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

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

//------------------------------------------------------------------------------
// Closure optimization
//------------------------------------------------------------------------------

// Modifies in-place the operand results vectors for a closure operation.
// |excludedOperandIndices| and |excludedResultIndices| are sets containing the
// operands and results in the lists to remove.
void excludeClosureOperandsAndResults(SmallVector<Value, 4> &operandValues,
                                      ArrayRef<unsigned> excludedOperandIndices,
                                      SmallVector<Type, 4> &resultTypes,
                                      ArrayRef<unsigned> excludedResultIndices);
void excludeClosureOperandsAndResults(SmallVector<Value, 4> &operandValues,
                                      SmallVector<Value, 4> &operandDims,
                                      ArrayRef<unsigned> excludedOperandIndices,
                                      SmallVector<Type, 4> &resultTypes,
                                      SmallVector<Value, 4> &resultDims,
                                      ArrayRef<unsigned> excludedResultIndices);

// Erases the given result indices from terminators in the given region.
void eraseRegionResults(Region &region,
                        ArrayRef<unsigned> excludedResultIndices);

// Optimizes closure |closureOp| to remove duplicate operands and unused
// results. The op may be mutated, destroyed, or replaced with a new one. If an
// optional |rewriter| is provided then it will be notified of the operations
// performed on the op. Returns true if the op was optimized.
bool optimizeClosureLikeOp(ClosureOpInterface &closureOp,
                           PatternRewriter *rewriter = nullptr);
template <typename T>
inline bool optimizeClosureOp(T &op, PatternRewriter *rewriter = nullptr) {
  auto closureOp = cast<ClosureOpInterface>(op.getOperation());
  bool didOptimize = optimizeClosureLikeOp(closureOp, rewriter);
  op = dyn_cast_or_null<DispatchRegionOp>(closureOp.getOperation());
  return didOptimize;
}

// A pattern that optimizes the given region-containing op T (CSE, DCE, etc).
// Duplicate operands will be combined and unused operands and results will be
// removed.
//
// T must implement the IREE::Flow::ClosureOpInterface.
template <typename T>
struct ClosureOptimizationPattern : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    auto closureOp = cast<ClosureOpInterface>(op.getOperation());
    return optimizeClosureLikeOp(closureOp, &rewriter) ? success() : failure();
  }
};

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
