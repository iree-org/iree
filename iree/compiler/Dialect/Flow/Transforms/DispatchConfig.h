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

#include "iree/compiler/Dialect/Flow/Analysis/Dispatchability.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

// Queries dispatch options for an operation.
// This is presently mostly a hard-coded set of heuristics but should expand
// to be based on both queries of new op interfaces and a cost model.
class OpDispatchPolicy {
 public:
  // The benefit that selecting an anchor is expected to provide. Anchors
  // with higher benefit should be formed first.
  using AnchorBenefit = int;
  enum class FusionType {
    // Fusion is disallowed.
    DISABLED = 0,
    // The operation should be duped into the dispatch region.
    CLONE_INTO = 1,
    // The operation should be cloned into the dispatch region and have
    // uses be redirected to the dispatch region.
    MOVE_INTO = 3,
  };

  OpDispatchPolicy(Dispatchability &dispatchability)
      : dispatchability(dispatchability) {}

  // Returns true if the given |op| can be dispatched in all cases.
  // Other passes may handle special cases of these ops but this initial
  // identification is conservative.
  bool isDispatchable(Operation *op);

  // Returns true if the op is an "identity metadata" op that must be preserved
  // at use-def boundaries. Such ops are non-executalbe, with >= 1 operands
  // and one result where the result is assumed to be operand(0) with any
  // op-specific metadata attached.
  bool isIdentityMetadata(Operation *op);

  // Returns the benefit of treating the given op as an anchor to form a
  // dispatch region around, where <= 0 disables the ability of the op to
  // be an anchor.
  // Anchors are identified greedily by sorting in descending order of
  // anchor benefit and ascending topological order (i.e. all ops with the
  // highest benefit have a dispatch region greedily formed around them
  // prior to proceeding to ops with the next lowest benefit).
  //
  // It is only valid to call this for dispatchable ops.
  AnchorBenefit getAnchorBenefit(Operation *op);

  // Returns the type of fusion that can be done for an input op that feeds
  // into a given anchor op.
  FusionType fuseInput(Operation *anchorOp, Operation *inputOp);

  // Returns the type of fusion that can be done for an output op that
  // follows an anchor op.
  FusionType fuseOutput(Operation *anchorOp, Operation *outputOp);

 private:
  Dispatchability &dispatchability;
};

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
