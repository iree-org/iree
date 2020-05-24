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

#include <algorithm>

#include "iree/compiler/Dialect/Flow/Analysis/Dispatchability.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/DispatchConfig.h"
#include "iree/compiler/Dialect/Flow/Utils/WorkloadUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-dispatch"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

// Clones an operation with new result types.
Operation *cloneWithNewResultTypes(
    Operation *op, llvm::SmallVectorImpl<Type> &newResultTypes) {
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(op->getOperands());
  state.addTypes(newResultTypes);
  state.addSuccessors(op->getSuccessors());
  state.addAttributes(op->getAttrs());
  for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i) {
    state.addRegion();
  }
  Operation *newOp = Operation::create(state);
  for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i) {
    newOp->getRegion(i).takeBody(op->getRegion(i));
  }
  return newOp;
}

struct DispatchableOp {
  OpDispatchPolicy::AnchorBenefit anchorBenefit;
  size_t index;
  Operation *op;

  bool operator<(const DispatchableOp &other) const {
    // Note inverted index: this is so that traversing a sorted list in
    // reverse yields a topological ordering for each anchorBenefit.
    return std::tie(anchorBenefit, other.index) <
           std::tie(other.anchorBenefit, index);
  }
};

struct DispatchRegion {
  IREE::Flow::DispatchRegionOp op;
  Operation *anchorOp;

  Block &getEntryBlock() { return op.body().front(); }

  // Appends results to the dispatch region. This will re-allocate the
  // DispatchRegionOp itself but preserve the contained body block.
  // Returns a ResultRange for the new dispatch region op's results
  // corresponding to addlResults.
  ResultRange appendResults(llvm::SmallVectorImpl<Value> &addlResults) {
    Location loc = op.getLoc();
    Block &block = getEntryBlock();

    unsigned origNumResults = op.getNumResults();
    llvm::SmallVector<Type, 4> newTypes(op.getResultTypes().begin(),
                                        op.getResultTypes().end());
    for (auto r : addlResults) newTypes.push_back(r.getType());

    // Changing the arity of the results requires replacing the dispatch region.
    OpBuilder builder(op);
    auto newDrOp = llvm::cast<IREE::Flow::DispatchRegionOp>(
        builder.insert(cloneWithNewResultTypes(op, newTypes)));
    op.replaceAllUsesWith(ResultRange(newDrOp, 0, origNumResults));
    op.erase();
    op = newDrOp;

    // Add results to the terminator.
    auto terminator = getEntryBlock().getTerminator();
    llvm::SmallVector<Value, 4> returns(terminator->getOperands());
    returns.append(addlResults.begin(), addlResults.end());
    terminator->setOperands(returns);

    return ResultRange(op, origNumResults, addlResults.size());
  }
};

// Utility class to optimize a "closure" op, which maintains a variadic
// list of operands corresponding to entry block arguments.
class ClosureOpOptimizer {
 public:
  ClosureOpOptimizer(Operation *closureOp, Block &entryBlock,
                     unsigned variadicOffset)
      : closureOp(closureOp),
        entryBlock(entryBlock),
        variadicOffset(variadicOffset),
        blockArgReplacements(entryBlock.getNumArguments()) {
    assert(closureOp->getNumOperands() ==
           entryBlock.getNumArguments() + variadicOffset);

    // Build data structure for unused operand elision.
    for (auto it : llvm::enumerate(entryBlock.getArguments())) {
      BlockArgument blockArg = it.value();
      Value opArg = closureOp->getOperand(it.index() + variadicOffset);
      if (blockArg.getUses().empty()) {
        // Not used - Drop.
        needsOperandElision = true;
        blockArgReplacements[it.index()] = BlockArgument();
        continue;
      }
      auto existingIt = argToBlockMap.find(opArg);
      if (existingIt == argToBlockMap.end()) {
        // Not found - Record for deduping.
        argToBlockMap.insert(std::make_pair(opArg, blockArg));
      } else {
        // Found - Replace.
        needsOperandElision = true;
        blockArgReplacements[it.index()] = existingIt->second;
      }
    }

    // Check for unused results.
    for (auto result : closureOp->getResults()) {
      if (result.getUses().empty()) {
        needsResultElision = true;
        break;
      }
    }
  }

  bool getNeedsOptimization() {
    return needsOperandElision || needsResultElision;
  }

  Operation *optimize() {
    if (needsResultElision) elideUnusedResults();
    if (needsOperandElision) elideUnusedOperands();
    return closureOp;
  }

 private:
  void elideUnusedOperands() {
    llvm::SmallVector<Value, 8> newOperands(
        closureOp->operand_begin(),
        closureOp->operand_begin() + variadicOffset);
    unsigned blockArgIndex = 0;
    for (auto it : llvm::enumerate(blockArgReplacements)) {
      llvm::Optional<BlockArgument> replacement = it.value();
      Value currentOpArg = closureOp->getOperand(it.index() + variadicOffset);
      if (!replacement) {
        // No change.
        newOperands.push_back(currentOpArg);
        blockArgIndex++;
        continue;
      } else if (!replacement.getValue()) {
        // Drop.
        entryBlock.eraseArgument(blockArgIndex);
        continue;
      } else {
        // Replace.
        BlockArgument currentBlockArg = entryBlock.getArgument(blockArgIndex);
        currentBlockArg.replaceAllUsesWith(*replacement);
        entryBlock.eraseArgument(blockArgIndex);
      }
    }

    closureOp->setOperands(newOperands);
  }

  void elideUnusedResults() {
    // Determine the result signature transform needed.
    llvm::SmallVector<unsigned, 4> resultIndexMap;
    llvm::SmallVector<Type, 4> newResultTypes;
    for (auto it : llvm::enumerate(closureOp->getResults())) {
      if (!it.value().getUses().empty()) {
        newResultTypes.push_back(it.value().getType());
        resultIndexMap.push_back(it.index());
      }
    }

    // Re-allocate the op.
    OpBuilder builder(closureOp);
    Operation *newOp =
        builder.insert(cloneWithNewResultTypes(closureOp, newResultTypes));

    // Remap all returns.
    llvm::SmallVector<Value, 4> newReturns(resultIndexMap.size());
    newOp->walk([&](IREE::Flow::ReturnOp terminator) {
      for (unsigned i = 0, e = resultIndexMap.size(); i < e; ++i) {
        newReturns[i] = terminator.getOperand(resultIndexMap[i]);
      }
      terminator.getOperation()->setOperands(newReturns);
    });

    // Replace original uses.
    for (unsigned i = 0, e = resultIndexMap.size(); i < e; ++i) {
      closureOp->getResult(resultIndexMap[i])
          .replaceAllUsesWith(newOp->getResult(i));
    }
    closureOp->erase();
    closureOp = newOp;
  }

  Operation *closureOp;
  Block &entryBlock;
  unsigned variadicOffset;
  llvm::SmallVector<llvm::Optional<BlockArgument>, 8> blockArgReplacements;
  llvm::SmallMapVector<Value, BlockArgument, 8> argToBlockMap;
  bool needsOperandElision = false;
  bool needsResultElision = false;
};

// TODO: Make a method on DispatchRegionOP.
IREE::Flow::DispatchRegionOp optimizeDispatchRegion(
    IREE::Flow::DispatchRegionOp drOp) {
  ClosureOpOptimizer opt(drOp, drOp.body().front(), 1);
  Operation *newOp = opt.optimize();
  return llvm::cast<IREE::Flow::DispatchRegionOp>(newOp);
}

// Clones and hoists any identity metadata ops from the operands and results
// of the dispatch region back out into the surrounding block.
// This function is not general purpose: it only knows how to undo sinking
// done by dispatch region formation.
void hoistDispatchRegionMetadataOps(DispatchRegion &dr, OpDispatchPolicy &policy) {
  BlockAndValueMapping mapping;
  Block &block = dr.getEntryBlock();
  for (unsigned i = 0, e = block.getNumArguments(); i < e; ++i) {
    mapping.map(block.getArgument(i), dr.op.args()[i]);
  }

  // Hoist metadata ops from the operand edge.
  for (auto it : llvm::enumerate(block.getArguments())) {
    auto &blockArg = it.value();
    for (auto &blockUse : blockArg.getUses()) {
      Operation *useOp = blockUse.getOwner();
      if (!policy.isIdentityMetadata(useOp) ||
          useOp->getOperand(0) != blockArg) continue;
      OpBuilder builder(dr.op);
      Operation *newOp = builder.clone(*useOp, mapping);
      dr.op.argsMutable().slice(it.index(), 1).assign(newOp->getResult(0));
    }
  }

  // Hoist metadata ops from the result edge.
  // Since initial formation can only have a single block, this is safe.
  auto *terminator = block.getTerminator();  
  for (auto it : llvm::enumerate(terminator->getOperands())) {
    Operation *defOp = it.value().getDefiningOp();
    if (!defOp || !policy.isIdentityMetadata(defOp)) continue;
    OpBuilder builder(dr.op.getContext());
    builder.setInsertionPointAfter(dr.op);
    Operation *newOp = builder.clone(*defOp, mapping);
    dr.op.getResult(it.index()).replaceAllUsesWith(newOp->getResult(0));
    newOp->setOperand(0, dr.op.getResult(it.index()));
  }
}

void findDispatchableAnchorOps(Block &block, OpDispatchPolicy &policy,
                               OpDispatchPolicy::AnchorBenefit maxAnchorBenefit,
                               llvm::SmallVectorImpl<DispatchableOp> &ops) {
  for (auto it : llvm::enumerate(block.getOperations())) {
    Operation *op = &it.value();
    // Skip any already formed dispatch regions and non dispatchable ops.
    if (isa<IREE::Flow::DispatchRegionOp>(op)) continue;
    if (!policy.isDispatchable(op)) continue;
    OpDispatchPolicy::AnchorBenefit anchorBenefit = policy.getAnchorBenefit(op);
    if (anchorBenefit > maxAnchorBenefit || anchorBenefit <= 0) continue;
    ops.push_back({anchorBenefit, it.index(), op});
  }
}

llvm::Optional<DispatchRegion> formDispatchRegion(Block &block,
                                                  Operation *anchorOp) {
  OpBuilder b(anchorOp);
  auto loc = anchorOp->getLoc();
  if (anchorOp->getNumResults() < 1) {
    emitError(loc) << "dispatch anchor op must have at least one result: "
                   << *anchorOp;
    return llvm::None;
  }
  Value result = anchorOp->getResult(0);
  Value workload = calculateWorkload(anchorOp, result);
  if (!workload) return llvm::None;

  // Map anchor into new dispatch region.
  llvm::SmallVector<Value, 4> capturedInputs(anchorOp->getOperands());
  llvm::SmallVector<Type, 1> types(anchorOp->getResultTypes().begin(),
                                   anchorOp->getResultTypes().end());
  auto drOp = b.create<IREE::Flow::DispatchRegionOp>(loc, types, workload,
                                                     capturedInputs);
  auto *drBlock = new Block();
  drOp.body().push_back(drBlock);
  BlockAndValueMapping mapping;
  for (Value capturedInput : capturedInputs) {
    auto blockArg = drBlock->addArgument(capturedInput.getType());
    mapping.map(capturedInput, blockArg);
  }

  // Create new body.
  OpBuilder drBuilder = OpBuilder::atBlockEnd(drBlock);
  auto *newAnchorOp = drBuilder.clone(*anchorOp, mapping);
  drBuilder.create<IREE::Flow::ReturnOp>(loc, newAnchorOp->getResults());

  // Replace anchor uses with region result.
  for (auto it : llvm::enumerate(anchorOp->getResults())) {
    it.value().replaceAllUsesWith(drOp.getResult(it.index()));
  }
  anchorOp->erase();
  return DispatchRegion{drOp, newAnchorOp};
}

Operation *inlineDispatchOp(DispatchRegion &dispatchRegion, Operation *origOp,
                            OpBuilder &builder) {
  auto drOp = dispatchRegion.op;
  Location loc = origOp->getLoc();

  // Map existing dr args.
  BlockAndValueMapping mapping;
  Block &block = dispatchRegion.getEntryBlock();
  for (unsigned i = 0, e = block.getNumArguments(); i < e; ++i) {
    mapping.map(drOp.args()[i], block.getArgument(i));
  }

  // Also map any terminator operands to support inlining at the end.
  for (auto it : llvm::enumerate(block.getTerminator()->getOperands())) {
    mapping.map(drOp.getResult(it.index()), it.value());
  }

  // Remember the values corresponding to original op results.
  llvm::SmallVector<Value, 4> origOpResultValues;
  for (Value result : origOp->getResults()) {
    origOpResultValues.push_back(mapping.lookupOrNull(result));
  }

  // Add arguments for any op arguments that need to be captured.
  for (Value newArgument : origOp->getOperands()) {
    if (mapping.contains(newArgument)) continue;
    drOp.getOperation()->insertOperands(drOp.getNumOperands(), {newArgument});
    Value newBlockArgument = block.addArgument(newArgument.getType());
    mapping.map(newArgument, newBlockArgument);
  }

  // Clone the op.
  Operation *inlinedOp = builder.clone(*origOp, mapping);

  // Replace any results from the orig with results from the clone.
  for (unsigned i = 0, e = origOp->getNumResults(); i < e; ++i) {
    Value resultFrom = origOp->getResult(i);
    Value resultTo = origOpResultValues[i];
    if (resultTo) {
      resultTo.replaceAllUsesWith(inlinedOp->getResult(i));
    }
  }

  return inlinedOp;
}

// After a call to inlineDispatchOp, adds the results of the inlined op to
// the dispatch region's results and redirects any uses outside of the dispatch
// region.
void returnAndReplaceUses(DispatchRegion &dr, Operation *origOp,
                          Operation *inlinedOp) {
  // Extend the arity of the dispatch region.
  llvm::SmallVector<Value, 4> addlResults(inlinedOp->getResults());
  origOp->replaceAllUsesWith(dr.appendResults(addlResults));
}

// Returns whether the op has no uses on all of its results.
bool opHasNoUses(Operation *op) {
  for (auto result : op->getResults()) {
    if (!result.use_empty()) return false;
  }
  return true;
}

// Maintains a worklist of operations that are potential fusion candidates.
// By default, items are popped in inverse topological order. An operation
// can only be added to a worklist once and later additions will be ignored.
class FusionWorklist {
 public:
  FusionWorklist(Block *block, bool inverseTopological = true)
      : block(block), inverseTopological(inverseTopological) {}

  // Adds defining ops of operands to the worklist.
  void addOperandDefs(OperandRange operands) {
    for (Value operand : operands) {
      Operation *def = operand.getDefiningOp();
      if (!def) continue;
      if (def->getBlock() != block) continue;
      if (!visited.insert(def).second) continue;
      worklist.push_back(def);
      dirty = true;
    }
  }

  // Adds uses.
  void addResultUses(ResultRange results) {
    for (auto result : results) {
      for (auto &use : result.getUses()) {
        Operation *def = use.getOwner();
        if (def->isKnownTerminator()) continue;
        if (def->getBlock() != block) continue;
        if (!visited.insert(def).second) continue;
        worklist.push_back(def);
        llvm::dbgs() << "  ** DISCOVER RESULT USE: " << *def << "\n";
        dirty = true;
      }
    }
  }

  // Pops the next operation or nullptr if empty.
  Operation *popNext() {
    if (worklist.empty()) return nullptr;
    if (dirty) sort();
    return worklist.pop_back_val();
  }

 private:
  // Sorts worklist items such that popNext() values pop in inverse
  // topological order.
  void sort() {
    if (inverseTopological) {
      llvm::sort(worklist, [](Operation *left, Operation *right) {
        return left->isBeforeInBlock(right);
      });
    } else {
      llvm::sort(worklist, [](Operation *left, Operation *right) {
        return right->isBeforeInBlock(left);
      });
    }
  }

  Block *block;
  llvm::SmallVector<Operation *, 4> worklist;
  llvm::SmallDenseSet<Operation *, 4> visited;
  bool inverseTopological;
  bool dirty = false;
};

LogicalResult fuseInputs(DispatchRegion &dispatchRegion,
                         OpDispatchPolicy &policy) {
  LLVM_DEBUG(llvm::dbgs() << "++ FUSING INPUTS\n");

  FusionWorklist worklist(dispatchRegion.op.getOperation()->getBlock());
  worklist.addOperandDefs(dispatchRegion.op.getOperands());

  while (Operation *nextOp = worklist.popNext()) {
    if (!policy.isDispatchable(nextOp)) continue;
    auto action = policy.fuseInput(dispatchRegion.anchorOp, nextOp);
    LLVM_DEBUG(llvm::dbgs().indent(2));
    if (action == OpDispatchPolicy::FusionType::MOVE_INTO) {
      return nextOp->emitError() << "cannot fuse input with MOVE_INTO action";
    } else if (action == OpDispatchPolicy::FusionType::DISABLED) {
      LLVM_DEBUG(llvm::dbgs()
                 << "- SKIP NON FUSABLE INPUT: " << *nextOp << "\n");
      continue;
    }

    // Always inline inputs at the top of the block. Since we are processing
    // the worklist in inverse topological order, this preserves the original
    // ordering.
    LLVM_DEBUG(llvm::dbgs() << "- FUSABLE INPUT(" << static_cast<int>(action)
                            << "): " << *nextOp << "\n");
    Block &entryBlock = dispatchRegion.getEntryBlock();
    auto builder = OpBuilder::atBlockBegin(&entryBlock);
    auto *inlinedOp = inlineDispatchOp(dispatchRegion, nextOp, builder);
    if (!inlinedOp) {
      return failure();
    }
    worklist.addOperandDefs(nextOp->getOperands());

    // Erase the op if it has no uses. This keeps it from forming regions
    // that will be dce'd later (or getting in the way of the benefit
    // scheme). Note that dispatchable ops have no side effects, which
    // makes this simple check safe.
    // The dispatch region must be optimized to remove unused arguments
    // resulting from this fusion.
    dispatchRegion.op = optimizeDispatchRegion(dispatchRegion.op);
    if (opHasNoUses(nextOp)) {
      nextOp->erase();
    }
  }

  return success();
}

LogicalResult fuseOutputs(DispatchRegion &dispatchRegion,
                          OpDispatchPolicy &policy) {
  LLVM_DEBUG(llvm::dbgs() << "++ FUSING OUTPUT\n");

  FusionWorklist worklist(dispatchRegion.op.getOperation()->getBlock(),
                          /*inverseTopological=*/false);
  worklist.addResultUses(dispatchRegion.op.getResults());

  while (Operation *nextOp = worklist.popNext()) {
    if (!policy.isDispatchable(nextOp)) continue;
    auto action = policy.fuseOutput(dispatchRegion.anchorOp, nextOp);
    LLVM_DEBUG(llvm::dbgs().indent(2));
    if (action == OpDispatchPolicy::FusionType::DISABLED) {
      LLVM_DEBUG(llvm::dbgs()
                 << "- SKIP NON FUSABLE INPUT: " << *nextOp << "\n");
      continue;
    }
    if (action != OpDispatchPolicy::FusionType::MOVE_INTO) {
      return nextOp->emitError()
             << "cannot fuse output except with MOVE_INTO action";
    }
    LLVM_DEBUG(llvm::dbgs() << "- FUSABLE OUTPUT(" << static_cast<int>(action)
                            << "): " << *nextOp << "\n");
    // Since results will be redirected to the region results, need to scan
    // for worklist items before changing use-def chain.
    worklist.addResultUses(nextOp->getResults());
    Block &entryBlock = dispatchRegion.getEntryBlock();
    auto builder = OpBuilder::atBlockTerminator(&entryBlock);
    auto *inlinedOp = inlineDispatchOp(dispatchRegion, nextOp, builder);
    if (!inlinedOp) {
      return failure();
    }
    returnAndReplaceUses(dispatchRegion, nextOp, inlinedOp);
    if (opHasNoUses(nextOp)) {
      nextOp->erase();
    }
  }

  return success();
}

LogicalResult processBlock(Block &block, OpDispatchPolicy &policy) {
  int maxAnchorBenefit =
      std::numeric_limits<OpDispatchPolicy::AnchorBenefit>::max();
  // Maps DispatchRegionOp to the anchor op.
  llvm::DenseMap<Operation *, Operation *> dispatchRegions;
  // Per iteration scratch.
  llvm::SmallVector<DispatchableOp, 10> dispatchableOps;

  // Loop backwards from high anchor benefit to low.
  for (;;) {
    dispatchableOps.clear();
    // Enumerate un-dispatched ops.
    findDispatchableAnchorOps(block, policy, maxAnchorBenefit, dispatchableOps);
    if (dispatchableOps.empty()) break;
    llvm::sort(dispatchableOps);

    // Traversing from back->front will produce ops in [anchorPriority, index]
    // order.
    auto &d = dispatchableOps.back();
    if (d.anchorBenefit <= 0) break;
    LLVM_DEBUG(llvm::dbgs() << "FORM DISPATCH REGION(" << d.index << ":"
                            << d.anchorBenefit << "): " << *d.op << "\n");
    auto dispatchRegion = formDispatchRegion(block, d.op);
    if (!dispatchRegion) return failure();
    dispatchRegions.insert(
        std::make_pair(dispatchRegion->op, dispatchRegion->anchorOp));

    // Fuse outputs prior to inputs, since they can yield more things to
    // evaluate for input fusion.
    if (failed(fuseOutputs(*dispatchRegion, policy))) return failure();
    if (failed(fuseInputs(*dispatchRegion, policy))) return failure();

    // Ensure all unused operands and results are dce'd.
    dispatchRegion->op = optimizeDispatchRegion(dispatchRegion->op);
    hoistDispatchRegionMetadataOps(*dispatchRegion, policy);
  }
  return success();
}

// Identifies dispatchable ops and moves them into dispatch regions.
// Some ops, such as call, will be deferred until following passes.
class IdentifyDispatchRegions2Pass
    : public PassWrapper<IdentifyDispatchRegions2Pass, FunctionPass> {
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

    OpDispatchPolicy policy(*dispatchability);
    for (auto &block : getFunction()) {
      if (failed(processBlock(block, policy))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createIdentifyDispatchRegions2Pass() {
  return std::make_unique<IdentifyDispatchRegions2Pass>();
}

static PassRegistration<IdentifyDispatchRegions2Pass> pass(
    "iree-flow-identify-dispatch-regions2",
    "Conservatively identifies dispatch regions in functions (v2)");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
