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

#include <algorithm>

#include "iree/compiler/Dialect/Flow/Analysis/Dispatchability.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Utils/DispatchUtils.h"
#include "iree/compiler/Dialect/Flow/Utils/WorkloadUtils.h"
#include "iree/compiler/Utils/GraphUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

// Returns true if the given |op| can be dispatched in all cases.
// Other passes may handle special cases of these ops but this initial
// identification is conservative.
bool isDispatchableOp(Operation *op, Dispatchability &dispatchability) {
  // TODO(b/144530470): replace with tablegen attributes/interfaces.
  if (FlowDialect::isDialectOp(op)) {
    // Ignore things we've already produced as they should only relate to
    // sequencer operations.
    return false;
  } else if (op->isKnownTerminator()) {
    // Currently we skip all terminators as we want to leave them in the block
    // to keep it valid. Future folding passes may take care of them if they are
    // worth bringing into the dispatch region.
    return false;
  } else if (auto callOp = dyn_cast<CallOp>(op)) {
    return dispatchability.isDispatchable(callOp.getCallee());
  } else if (isa<CallIndirectOp>(op)) {
    // Indirect calls are not supported in dispatch code.
    return false;
  } else if (isa<ConstantOp>(op)) {
    // Constants are handled in the RematerializeDispatchConstants pass.
    // We do that independently so that we can more easily see the use of
    // constants across all dispatches instead of just on an individual basis
    // as we do here.
    return false;
  } else if (op->getNumResults() &&
             !op->getResult(0).getType().isa<ShapedType>()) {
    // We don't put scalar manipulation into dispatch regions.
    return false;
  } else if (!isOpOfKnownDialect(op)) {
    // Probably a custom op.
    return false;
  }
  return true;
}

// Returns true if the given |op| can have other ops fused into it.
// This is sketchy and it'd be nice to define this as an op property instead.
//
// What we are looking for in foldable ops is whether the execution of the op
// when fused has some possible benefit (or at least, a non-negative cost).
// Eventually we want to allow backends to vote on this and allow multiple
// folding strategies within the same executable. For now we just hardcode what
// we know for the ops we have.
//
// Preconditions: isDispatchableOp(op) == true.
bool isFusionRootOp(Operation *op) {
  // TODO(b/144530470): replace with tablegen attributes/interfaces.
  if (isa<xla_hlo::DotOp>(op) || isa<xla_hlo::ConvOp>(op)) {
    // We have hand-written kernels for these right now we want to stand alone.
    // When we do a bit more magic we should allow these ops to fold.
    return false;
  }
  return true;
}

// Returns true if the given |op| can be fused into other ops.
//
// Ops that perform narrowing on shapes (such as reduction ops) should not
// generally be fused with other downstream ops (probably...). This avoids
// potential oversampling and indexing issues and allows backends to perform
// more efficient rooted cascading reduction dispatches.
//
// Preconditions: isDispatchableOp(op) == true.
bool isFusableOp(Operation *op) {
  // TODO(b/144530470): replace with tablegen attributes/interfaces.
  if (isa<xla_hlo::DotOp>(op) || isa<xla_hlo::ConvOp>(op)) {
    return false;
  } else if (isa<xla_hlo::ReduceOp>(op)) {
    // Reduction is usually a dedicated root operation - we can shove things in
    // the front of it but not behind.
    return false;
  }
  return true;
}

// Recursively traverses the IR DAG along the operand edges to find ops we are
// able to fuse and appends them to |subgraph|.
void gatherFusionOps(Operation *op, Dispatchability &dispatchability,
                     llvm::SetVector<Operation *> *subgraph) {
  // Skip ops that are used outside of the subgraph we are building.
  for (auto result : op->getResults()) {
    if (result.use_empty() || result.hasOneUse()) continue;
    for (auto *user : result.getUsers()) {
      if (subgraph->count(user) == 0) {
        // Op that consumes the result is not (yet) in the subgraph.
        // For now we'll ignore these as it may represent a fork that we don't
        // want to join too early.
        return;
      }
    }
  }

  // Walk backward up to ops providing our input operands.
  for (auto operand : op->getOperands()) {
    auto *sourceOp = operand.getDefiningOp();
    if (!sourceOp) continue;
    if (subgraph->count(sourceOp) == 0) {
      if (isDispatchableOp(sourceOp, dispatchability) &&
          isFusableOp(sourceOp)) {
        gatherFusionOps(sourceOp, dispatchability, subgraph);
      }
    }
  }

  subgraph->insert(op);
}

// Finds all ops that can be fused together with the given |rootOp| by searching
// backwards in the op order through input edges.
// Returns a topologically sorted list of all fused ops with |rootOp| at the
// end.
std::vector<Operation *> findFusionSubgraphFromRoot(
    Operation *rootOp, Dispatchability &dispatchability) {
  if (!isFusionRootOp(rootOp)) {
    return {rootOp};
  }
  llvm::SetVector<Operation *> subgraph;
  subgraph.insert(rootOp);
  gatherFusionOps(rootOp, dispatchability, &subgraph);
  return sortOpsTopologically(subgraph);
}

// Identifies ranges of dispatchable ops and moves them into dispatch regions.
LogicalResult identifyBlockDispatchRegions(Block *block,
                                           Dispatchability &dispatchability) {
  // Fixed point iteration until we can no longer fuse anything.
  bool didFindAnyNewRegions;
  do {
    // Iterate in reverse so we root further along in the op list.
    didFindAnyNewRegions = false;
    for (auto &rootOp : llvm::reverse(*block)) {
      if (!isDispatchableOp(&rootOp, dispatchability)) {
        // Op should remain at the sequencer level.
        continue;
      }

      // Attempt to find all operations, including rootOp, that can be fused.
      // The ops will be sorted in topological order with rootOp as the last op.
      // Worst case we may end up with a subgraph of only the rootOp.
      auto fusedSubgraph = findFusionSubgraphFromRoot(&rootOp, dispatchability);

      // Compute the workload based on the output shape.
      // When variadic all output shapes match so we can just take the first.
      auto workload = calculateWorkload(&rootOp, rootOp.getResult(0));

      // Try to build a dispatch region from this root.
      if (failed(buildDispatchRegion(block, workload, fusedSubgraph))) {
        return failure();
      }

      // Successfully created a dispatch region from the ops and we must now
      // start over again as we've likely trashed the whole block structure.
      didFindAnyNewRegions = true;
      break;
    }
  } while (didFindAnyNewRegions);
  return success();
}

}  // namespace

// Identifies dispatchable ops and moves them into dispatch regions.
// Some ops, such as call, will be deferred until following passes.
class IdentifyDispatchRegionsPass
    : public FunctionPass<IdentifyDispatchRegionsPass> {
 public:
  void runOnFunction() override {
    // NOTE: we require the DispatchabilityAnalysisPass to have run first.
    auto dispatchability = getCachedParentAnalysis<Dispatchability>();
    if (!dispatchability.hasValue()) {
      getFunction().emitError()
          << "dispatchability analysis not performed "
             "on module; run -iree-flow-dispatchability-analysis first";
      return signalPassFailure();
    }

    for (auto &block : getFunction()) {
      if (failed(identifyBlockDispatchRegions(&block,
                                              dispatchability.getValue()))) {
        return signalPassFailure();
      }
    }
  }
};

std::unique_ptr<OpPassBase<FuncOp>> createIdentifyDispatchRegionsPass() {
  return std::make_unique<IdentifyDispatchRegionsPass>();
}

static PassRegistration<IdentifyDispatchRegionsPass> pass(
    "iree-flow-identify-dispatch-regions",
    "Conservatively identifies dispatch regions in functions");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
