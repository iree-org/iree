// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/Constant/ConstExpr.h"

#include "iree/compiler/Dialect/Util/Analysis/Constant/OpOracle.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GraphWriter.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define DEBUG_TYPE "iree-constexpr"

namespace mlir::iree_compiler::IREE::Util {

//===----------------------------------------------------------------------===//
// ConstExprAnalysis
//===----------------------------------------------------------------------===//

static OpOperand *findOperandFor(Operation *op, Value input) {
  for (OpOperand &operand : op->getOpOperands()) {
    if (operand.get() == input)
      return &operand;
  }
  return nullptr;
}

bool ConstExprAnalysis::isConstExprOperation(Operation *queryOp) const {
  if (queryOp->getNumResults() == 0) {
    bool hasNoMemoryEffects = false;
    if (auto effectOp = dyn_cast<MemoryEffectOpInterface>(queryOp))
      hasNoMemoryEffects = effectOp.hasNoEffect();
    if (hasNoMemoryEffects && queryOp->hasTrait<OpTrait::ReturnLike>())
      return true;
    return false;
  }
  // NOTE: this only checks the first result as all results are added to the map
  // with the same value. If we supported ops with only some results being
  // constant we'd need to change this and not look at the op at all.
  return isConstExprValue(queryOp->getResult(0));
}

bool ConstExprAnalysis::ConstValueInfo::hasNonAnalyzedConsumer() const {
  // The analysis cannot represent zero-result operations, so detect that
  // and return.
  for (Operation *user : getOperation()->getUsers()) {
    if (user->getNumResults() == 0) {
      return true;
    }
  }
  return false;
}

void ConstExprAnalysis::tryExpandToUseOrUseParent(Operation *definingOp,
                                                  Operation *useOp) {
  if (definingOp->getParentOp() != useOp->getParentOp()) {
    auto parentOp = useOp->getParentOp();
    if (parentOp && definingOp->getParentOp() == parentOp->getParentOp()) {
      expandToOp(parentOp);
    }
    return;
  }
  expandToOp(useOp);
}

ConstExprAnalysis::ConstExprAnalysis(Operation *rootOp)
    : asmState(rootOp,
               OpPrintingFlags().elideLargeElementsAttrs().skipRegions()) {
  Explorer explorer(rootOp, TraversalAction::SHALLOW);
  explorer.initialize();

  // Populate the constant roots for globals.
  // NOTE: these may be _run-time_ constant and not _compile-time_ constant,
  // such as if they are initialized based on values only available at runtime.
  explorer.forEachGlobal([&](const Explorer::GlobalInfo *info) {
    // Rely on globals having been canonicalized to immutable correctly.
    if (info->isIndirect || info->op.isGlobalMutable())
      return;
    if (!isLegalConstExprRootType(info->op.getGlobalType()))
      return;
    for (auto loadOp : info->getLoads())
      constantRoots[loadOp.getLoadedGlobalValue()] = loadOp;
  });

  // Populate the constant roots for all inline constants in the program.
  explorer.forEachFunctionLikeOp([&](FunctionOpInterface funcOp) {
    funcOp.walk([&](Operation *op) {
      if (!op->hasTrait<OpTrait::ConstantLike>())
        return;
      for (auto resultType : op->getResultTypes()) {
        if (!isLegalConstExprRootType(resultType))
          return;
      }
      for (auto result : op->getResults())
        constantRoots[result] = op;
    });
  });

  // Prime the const value map with known roots. This must be done first
  // so that traversal up the dag terminates if it hits one.
  for (auto it : constantRoots) {
    Value constValue = it.first;

    // Note the root in the ConstValueState so that we can do quick hit
    // detection when traversing.
    auto rootInfo = addInfo(constValue);
    rootInfo->isRoot = true;
    rootInfo->state = ConstValueInfo::CONSTANT;
    rootInfo->roots.insert(constValue);
    LLVM_DEBUG({
      llvm::dbgs() << "[ConstExprAnalysis] mark constant root: ";
      constValue.print(llvm::dbgs(), asmState);
      llvm::dbgs() << "\n";
    });
  }

  // Now go over each constant root again and expand the frontier to include
  // its consumers.
  for (auto it : constantRoots) {
    Operation *constOp = it.second;
    for (Operation *user : constOp->getUsers()) {
      tryExpandToUseOrUseParent(constOp, user);
    }
  }

  // Process worklist until all resolved.
  ConstValueWorklist iterWorklist;
  while (!worklist.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "[ConstExprAnalysis] process worklist:\n");
    iterWorklist.clear();
    iterWorklist.swap(worklist);
    for (ConstValueInfo *info : iterWorklist) {
      if (info->state != ConstValueInfo::UNKNOWN)
        continue;
      bool allConstants = true;
      for (ConstValueInfo *producerInfo : info->producers) {
        assert(producerInfo->state != ConstValueInfo::UNANALYZED &&
               "Producers of unknown value must be all analyzed.");

        if (producerInfo->state == ConstValueInfo::UNKNOWN) {
          // Producers unknown. No further progress until next iteration.
          worklist.push_back(info);
          allConstants = false;
          break;
        }

        if (producerInfo->state == ConstValueInfo::NON_CONSTANT) {
          // We have to be non constant too.
          info->state = ConstValueInfo::NON_CONSTANT;
          LLVM_DEBUG({
            llvm::dbgs() << "[ConstExprAnalysis]   - resolved NON_CONSTANT: ";
            info->constValue.print(llvm::dbgs(), asmState);
            llvm::dbgs() << "\n";
          });
          allConstants = false;
          break;
        }
      }

      // Fall-through. See if we have a resolution.
      if (allConstants) {
        // Finalize it.
        info->state = ConstValueInfo::CONSTANT;
        LLVM_DEBUG({
          llvm::dbgs() << "[ConstExprAnalysis]   + resolved CONSTANT: ";
          info->constValue.print(llvm::dbgs(), asmState);
          llvm::dbgs() << "\n";
        });

        // Now that all of its producers are known, record its roots.
        for (ConstValueInfo *producerInfo : info->producers) {
          info->roots.insert(producerInfo->roots.begin(),
                             producerInfo->roots.end());
        }

        // And expand the frontier.
        Operation *definingOp = info->constValue.getDefiningOp();
        assert(definingOp && "const values should have defining op");
        for (Operation *user : definingOp->getUsers()) {
          tryExpandToUseOrUseParent(definingOp, user);
        }
      }
    }
  }

  // Go through and populate all consumer sets now that producers are known.
  for (auto it : constInfoMap) {
    ConstValueInfo *consumer = it.second;
    for (ConstValueInfo *producer : consumer->producers) {
      producer->consumers.insert(consumer);
    }
  }

  LLVM_DEBUG(print(llvm::dbgs()));
}

ConstExprAnalysis::ConstValueInfo *
ConstExprAnalysis::addInfo(Value constValue) {
  auto info = std::make_unique<ConstValueInfo>(constValue);
  constInfoMap[constValue] = info.get();
  allocedConstInfos.push_back(std::move(info));
  return allocedConstInfos.back().get();
}

void ConstExprAnalysis::expandToOp(Operation *op) {
  SmallVector<Operation *> expandWorklist;
  expandWorklist.push_back(op);
  do {
    expandToOpStep(expandWorklist.pop_back_val(), expandWorklist);
  } while (!expandWorklist.empty());
}

void ConstExprAnalysis::expandToOpStep(
    Operation *op, SmallVectorImpl<Operation *> &expandWorklist) {
  ConstExprOpInfo opInfo = ConstExprOpInfo::getForOp(op);
  for (auto result : op->getResults()) {
    auto *valueInfo = constInfoMap.lookup(result);
    if (valueInfo && valueInfo->state != ConstValueInfo::UNANALYZED)
      continue;

    // Generate new info record.
    if (!valueInfo)
      valueInfo = addInfo(result);

    // Update the producers first as we might early-return below.
    for (Value producer : opInfo.producers) {
      if (ConstValueInfo *producerInfo = constInfoMap.lookup(producer)) {
        valueInfo->producers.insert(producerInfo);
        continue;
      }
      // Create an unanalyzed value info as a placeholder. The info might be
      // analyzed later if we are interested in it.
      ConstValueInfo *producerInfo = addInfo(producer);
      valueInfo->producers.insert(producerInfo);
      // If the producer is a multi-result operation, then add ConstValueInfo
      // for all of the op's results. This initialization ensures that no ops
      // will have a mix of results with and without ConstValueInfo.
      Operation *producerOp = producer.getDefiningOp();
      if (!producerOp) {
        continue;
      }
      for (Value producerOpResult : producerOp->getResults()) {
        if (!constInfoMap.lookup(producerOpResult)) {
          (void)addInfo(producerOpResult);
        }
      }
    }

    if (!opInfo.isEligible) {
      // Put it in a NON_CONSTANT state and bail. This is terminal.
      valueInfo->state = ConstValueInfo::NON_CONSTANT;
      LLVM_DEBUG({
        llvm::dbgs()
            << "[ConstExprAnalysis]   - expand to NON_CONSTANT (ineligible): ";
        result.print(llvm::dbgs(), asmState);
        llvm::dbgs() << "\n";
      });
      continue;
    }

    // If here, then an unknown state.
    valueInfo->state = ConstValueInfo::UNKNOWN;
    LLVM_DEBUG({
      llvm::dbgs() << "[ConstExprAnalysis]   ? expand to UNKNOWN: ";
      result.print(llvm::dbgs(), asmState);
      llvm::dbgs() << "\n";
    });
    worklist.push_back(valueInfo);

    // Process producers.
    for (auto producer : opInfo.producers) {
      Operation *definingOp = producer.getDefiningOp();
      if (!definingOp) {
        // Consider crossing out of block to be non-const.
        valueInfo->state = ConstValueInfo::NON_CONSTANT;
        break;
      }
      expandWorklist.push_back(definingOp);
    }
  }
}

void ConstExprAnalysis::print(raw_ostream &os) const {
  os << "[ConstExprAnalysis] found constants:\n";
  for (auto &info : allocedConstInfos) {
    if (info->state != ConstValueInfo::CONSTANT || info->isRoot)
      continue;
    if (!info->roots.empty()) {
      os << "\n[ConstExprAnalysis] constexpr ";
      info->constValue.print(os, asmState);
      os << "\n";
      os << "   + roots:\n";
      for (Value root : info->roots) {
        os << "      ";
        root.print(os, asmState);
        os << "\n";
      }
      os << "   + producers:\n";
      for (ConstValueInfo *producerInfo : info->producers) {
        os << "      ";
        producerInfo->constValue.print(os, asmState);
        os << "\n";
      }
    }
  }
}

void ConstExprAnalysis::dump() const { print(llvm::dbgs()); }

//===----------------------------------------------------------------------===//
// ConstExprHoistingPolicy
//===----------------------------------------------------------------------===//

ConstExprHoistingPolicy::ConstExprHoistingPolicy(
    const ConstExprAnalysis &analysis, int64_t threshold)
    : analysis(analysis), constExprMaxSizeIncreaseThreshold(threshold),
      decisions(analysis.allocedConstInfos.size()) {
  for (auto &it : analysis.allocedConstInfos) {
    decisions[it.get()] = {};
  }
}

void ConstExprHoistingPolicy::initialize() {
  // Bootstrap the worklist in analysis order, which is topological (def, use)
  // order.
  // TODO: Do a secondary sort?
  Worklist worklist;
  worklist.reserve(analysis.allocedConstInfos.size());
  for (auto &it : analysis.allocedConstInfos) {
    auto *info = it.get();
    // Skip unanalyzed values.
    if (info->state == ConstExprAnalysis::ConstValueInfo::UNANALYZED)
      continue;
    worklist.push_back(info);
  }

  // Since just initializing invariants, which are local, iteration order
  // doesn't matter.
  for (auto *info : worklist) {
    Decision *decision = getDecision(info);
    makeInvariantDecision(info, decision);
    LLVM_DEBUG({
      Outcome postDecisionOutcome = decision->getOutcome();
      if (postDecisionOutcome != UNDECIDED) {
        llvm::dbgs() << "[ConstExprHoistPolicy] invariant ";
        if (postDecisionOutcome == ENABLE_HOIST) {
          llvm::dbgs() << "ENABLE_HOIST";
        } else if (postDecisionOutcome == DISABLE_HOIST) {
          llvm::dbgs() << "DISABLE_HOIST";
        }
        llvm::dbgs() << ": ";
        info->constValue.print(llvm::dbgs(), analysis.getAsmState());
        llvm::dbgs() << "\n";
      }
    });
  }

  // Work iteratively until converged.
  for (int i = 0;; ++i) {
    (void)i;
    bool madeChange = false;
    for (auto *info : worklist) {
      Decision *decision = getDecision(info);
      if (decision->getOutcome() != UNDECIDED)
        continue;
      makeDecision(info, decision);

      if (decision->getOutcome() != UNDECIDED) {
        madeChange = true;
        LLVM_DEBUG({
          llvm::dbgs() << "[ConstExprHoistPolicy(" << i << ")] ";
          if (decision->getOutcome() == ENABLE_HOIST) {
            llvm::dbgs() << "ENABLE_HOIST";
          } else if (decision->getOutcome() == DISABLE_HOIST) {
            llvm::dbgs() << "DISABLE_HOIST";
          }
          llvm::dbgs() << ": ";
          info->constValue.print(llvm::dbgs(), analysis.getAsmState());
          llvm::dbgs() << "\n";
        });
      }
    }

    if (!madeChange) {
      LLVM_DEBUG({
        llvm::dbgs() << "[ConstExprHoistPolicy(" << i << ")] converged!\n";
      });
      break;
    }
  }

  for (auto *info : worklist) {
    Decision *decision = getDecision(info);
    if (decision->getOutcome() == UNDECIDED) {
      LLVM_DEBUG({
        llvm::dbgs() << "[ConstExprHoistPolicy] value did not converge: ";
        info->constValue.print(llvm::dbgs(), analysis.getAsmState());
        llvm::dbgs() << "\n";
      });
    }
  }
}

static bool doesHoistingIncreaseSizeSignificantly(
    const ConstExprAnalysis::ConstValueInfo *info, int64_t threshold) {
  int64_t inSize = 0;
  for (Value root : info->roots) {
    // TODO: Are there any other types we care about here?
    if (auto type = dyn_cast<ShapedType>(root.getType())) {
      int64_t elementCount = 1;
      for (int64_t dim : type.getShape()) {
        // Conservatively treat dynamic values as 1, to find a lower bound on
        // input size.
        if (!ShapedType::isDynamic(dim)) {
          elementCount *= dim;
        }
      }
      inSize +=
          getRoundedPhysicalStorageSize(elementCount, type.getElementType());
    }
  }

  int64_t outSize = 0;
  if (auto type = dyn_cast<ShapedType>(info->constValue.getType())) {
    int64_t elementCount = 1;
    for (int64_t dim : type.getShape()) {
      if (ShapedType::isDynamic(dim)) {
        // Dynamic values can lead to an unbounded increase in size, treat this
        // as a significant increase.
        return true;
      }
      elementCount *= dim;
    }
    outSize =
        getRoundedPhysicalStorageSize(elementCount, type.getElementType());
  }

  return outSize > inSize + threshold;
}

void ConstExprHoistingPolicy::makeInvariantDecision(
    const ConstExprAnalysis::ConstValueInfo *info, Decision *decision) {
  // Check 1: Is it not const-expr.
  if (!info->isConstExpr()) {
    return decision->disableHoist();
  }

  // Check 2: Is it a root (these are already hoisted).
  if (info->isRoot) {
    return decision->disableHoist();
  }

  // Check 3: Is the op itself a valid "leaf" that can become a global.
  if (!isHoistableConstExprLeaf(info)) {
    return decision->disableHoist();
  }

  // Check 4: Does hoisting this value significantly increase the size of the
  // module?
  if (doesHoistingIncreaseSizeSignificantly(
          info, constExprMaxSizeIncreaseThreshold)) {
    return decision->disableHoist();
  }
}

void ConstExprHoistingPolicy::makeDecision(
    const ConstExprAnalysis::ConstValueInfo *info, Decision *decision) {
  // A const-expr value has a legal escape if:
  //   - Has a non analyzed consumer
  //   - It has an analyzed consumer that:
  //     - Has been marked as DISABLE_HOIST (must feed into something that is
  //       not being hoisted).
  //     - Is consumed by a hoistable operand or no operand (signals implicit
  //       capture).
  bool hasLegalEscape = info->hasNonAnalyzedConsumer();
  if (!hasLegalEscape) {
    for (auto *consumerInfo : info->consumers) {
      Decision *consumerDecision = getDecision(consumerInfo);
      if (consumerDecision->getOutcome() != DISABLE_HOIST)
        continue;

      Operation *consumerOp = consumerInfo->getOperation();
      OpOperand *consumerOperand = findOperandFor(consumerOp, info->constValue);
      if (!consumerOperand) {
        // Must be an implicit capture.
        hasLegalEscape = true;
        break;
      } else if (isHoistableConstExprConsumingOperand(consumerOperand)) {
        hasLegalEscape = true;
      }
    }
  }

  // If there is no legal escape, we can concretely disable.
  if (!hasLegalEscape) {
    decision->disableHoist();
    return;
  }

  // Otherwise, we can conditionally enable hoisting (based on cost model, etc).
  // TODO: Implement further conditions.
  decision->enableHoist();
}

void ConstExprHoistingPolicy::printDotGraph(raw_ostream &os) const {
  WriteGraph(os, this);
}

void ConstExprHoistingPolicy::dumpDotGraph() const {
  printDotGraph(llvm::errs());
}

} // namespace mlir::iree_compiler::IREE::Util

namespace llvm {
using mlir::iree_compiler::IREE::Util::ConstExprAnalysis;
using mlir::iree_compiler::IREE::Util::ConstExprHoistingPolicy;
template <>
struct DOTGraphTraits<const ConstExprHoistingPolicy *>
    : public DefaultDOTGraphTraits {
  explicit DOTGraphTraits(bool isSimple = false)
      : DefaultDOTGraphTraits(isSimple) {}

  std::string getNodeLabel(const ConstExprAnalysis::ConstValueInfo *Node,
                           const ConstExprHoistingPolicy *g) {
    std::string label;
    llvm::raw_string_ostream os(label);
    os << Node->constValue.getType();
    return label;
  }

  static bool isNodeHidden(const ConstExprAnalysis::ConstValueInfo *Node,
                           const ConstExprHoistingPolicy *g) {
    // Only display nodes that the analysis has determined to be const-expr.
    return !Node->isConstExpr();
  }

  static std::string
  getNodeAttributes(const ConstExprAnalysis::ConstValueInfo *Node,
                    const ConstExprHoistingPolicy *g) {
    // Roots are colored red.
    if (Node->isRoot)
      return "fillcolor=red,style=filled";

    // Hoisted values are colored green.
    ConstExprHoistingPolicy::Outcome outcome = g->getOutcome(Node);
    if (outcome == ConstExprHoistingPolicy::Outcome::ENABLE_HOIST)
      return "fillcolor=green,style=filled";

    return "";
  }

  static void
  addCustomGraphFeatures(const ConstExprHoistingPolicy *g,
                         GraphWriter<const ConstExprHoistingPolicy *> &GW) {}
};
}; // namespace llvm
