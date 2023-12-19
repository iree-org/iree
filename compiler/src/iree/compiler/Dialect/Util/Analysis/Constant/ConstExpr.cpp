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
#include "mlir/Dialect/Arith/IR/Arith.h"

#define DEBUG_TYPE "iree-constexpr"

using llvm::dbgs;

using namespace mlir::iree_compiler::IREE::Util;

namespace mlir::iree_compiler::IREE::Util {

//===----------------------------------------------------------------------===//
// ConstExprAnalysis
//===----------------------------------------------------------------------===//

namespace {
OpOperand *findOperandFor(Operation *op, Value input) {
  for (OpOperand &operand : op->getOpOperands()) {
    if (operand.get() == input)
      return &operand;
  }
  return nullptr;
}

bool isHoistableToRootOp(Operation *rootOp, Operation *constOp) {
  if (constOp->isRegistered()) {
    Operation *syms = SymbolTable::getNearestSymbolTable(constOp);
    if (syms != rootOp)
      return false;
  } else {
    // Special handling for unregistered testing ops.
    // Since these are unregistered, SymbolTable::getNearestSymbolTable can't
    // check if they define new symbol tables.
    auto opName = constOp->getName().getStringRef();
    if (opName != "iree_unregistered.var_expr" &&
        opName != "iree_unregistered.non_leaf_const_expr" &&
        opName != "iree_unregistered.const_expr") {
      // Returns false for unknown unregistered ops.
      return false;
    }
    // Check if unregistered ops' parents are in the same symbol table.
    // For now we don't handle the case where unregistered ops' parent is
    // another unregistered op (will return false in this case).
    Operation *syms =
        SymbolTable::getNearestSymbolTable(constOp->getParentOp());
    if (syms != rootOp)
      return false;
  }

  Operation *parentOp = constOp->getParentOp();
  while (parentOp != rootOp) {
    assert(parentOp && "constOp is not a descendant of the rootOp.");
    // For now only hoist descendants from functions and nested control flow
    // regions.
    if (!(parentOp->hasTrait<RegionBranchOpInterface::Trait>() ||
          parentOp->hasTrait<FunctionOpInterface::Trait>())) {
      return false;
    }
    parentOp = parentOp->getParentOp();
  }

  return true;
}

} // namespace

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

ConstExprAnalysis::ConstExprAnalysis(Operation *rootOp) {
  Explorer explorer(rootOp, TraversalAction::SHALLOW);
  explorer.initialize();

  // Populate the constant roots for globals.
  explorer.forEachGlobal([&](const Explorer::GlobalInfo *info) {
    // Rely on globals having been canonicalized to immutable correctly.
    if (info->op.isGlobalMutable())
      return;
    if (info->isIndirect)
      return;
    if (!isLegalConstExprRootType(info->op.getGlobalType()))
      return;
    for (auto *use : info->uses) {
      auto loadOp = llvm::dyn_cast<GlobalLoadOp>(use);
      if (!loadOp)
        continue;
      if (!isHoistableToRootOp(rootOp, loadOp))
        continue;
      constantRoots[loadOp.getResult()] = loadOp;
    }
  });

  // Populate the constant roots for all inline constants in the program.
  rootOp->walk([&](arith::ConstantOp constOp) {
    if (isHoistableToRootOp(rootOp, constOp) &&
        isLegalConstExprRootType(constOp.getResult().getType()))
      constantRoots[constOp.getResult()] = constOp;
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
    LLVM_DEBUG(dbgs() << "CONSTANT ROOT: " << constValue << "\n");
  }

  // Now go over each constant root again and expand the frontier to include
  // its consumers.
  for (auto it : constantRoots) {
    Operation *constOp = it.second;
    for (auto &use : constOp->getUses()) {
      Operation *useOp = use.getOwner();
      if (isHoistableToRootOp(rootOp, useOp))
        expandToOp(useOp);
    }
  }

  // Process worklist until all resolved.
  ConstValueWorklist iterWorklist;
  while (!worklist.empty()) {
    LLVM_DEBUG(dbgs() << "PROCESS WORKLIST:\n");
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
          LLVM_DEBUG(dbgs() << "  RESOLVED AS NON_CONSTANT: "
                            << info->constValue << "\n");
          allConstants = false;
          break;
        }
      }

      // Fall-through. See if we have a resolution.
      if (allConstants) {
        // Finalize it.
        info->state = ConstValueInfo::CONSTANT;
        LLVM_DEBUG(dbgs() << "  RESOLVED AS CONSTANT: " << info->constValue
                          << "\n");

        // Now that all of its producers are known, record its roots.
        for (ConstValueInfo *producerInfo : info->producers) {
          info->roots.insert(producerInfo->roots.begin(),
                             producerInfo->roots.end());
        }

        // And expand the frontier.
        Operation *definingOp = info->constValue.getDefiningOp();
        assert(definingOp && "const values should have defining op");
        for (auto &use : definingOp->getUses()) {
          Operation *useOp = use.getOwner();
          if (isHoistableToRootOp(rootOp, useOp))
            expandToOp(useOp);
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
}

ConstExprAnalysis::ConstValueInfo *
ConstExprAnalysis::addInfo(Value constValue) {
  auto info = std::make_unique<ConstValueInfo>(constValue);
  constInfoMap[constValue] = info.get();
  allocedConstInfos.push_back(std::move(info));
  return allocedConstInfos.back().get();
}

void ConstExprAnalysis::expandToOp(Operation *op) {
  ConstExprOpInfo opInfo = ConstExprOpInfo::getForOp(op);
  for (auto result : op->getResults()) {
    auto *valueInfo = constInfoMap.lookup(result);
    if (valueInfo && valueInfo->state != ConstValueInfo::UNANALYZED)
      continue;

    // Generate new info record.
    if (!valueInfo)
      valueInfo = addInfo(result);

    // Update the producers first as we might early-return below.
    for (auto producer : opInfo.producers) {
      ConstValueInfo *producerInfo = constInfoMap.lookup(producer);
      if (!producerInfo) {
        // Create an unanalyzed value info as a placeholder. The info might be
        // analyzed later if we are interested in it.
        producerInfo = addInfo(producer);
      }
      valueInfo->producers.insert(producerInfo);
    }

    if (!opInfo.isEligible) {
      // Put it in a NON_CONSTANT state and bail. This is terminal.
      valueInfo->state = ConstValueInfo::NON_CONSTANT;
      LLVM_DEBUG(dbgs() << "  EXPAND TO INELIGIBLE: " << result << "\n");
      continue;
    }

    // If here, then an unknown state.
    valueInfo->state = ConstValueInfo::UNKNOWN;
    LLVM_DEBUG(dbgs() << "  EXPAND TO UNKNOWN: " << result << "\n");
    worklist.push_back(valueInfo);

    // Process producers.
    for (auto producer : opInfo.producers) {
      Operation *definingOp = producer.getDefiningOp();
      if (!definingOp) {
        // Consider crossing out of block to be non-const.
        valueInfo->state = ConstValueInfo::NON_CONSTANT;
        break;
      }
      expandToOp(definingOp);
    }
  }
}

void ConstExprAnalysis::print(raw_ostream &os) const {
  os << "\nFOUND CONSTANTS:\n----------------\n";
  for (auto &info : allocedConstInfos) {
    if (info->state != ConstValueInfo::CONSTANT || info->isRoot)
      continue;
    if (!info->roots.empty()) {
      os << "\n::" << info->constValue << "\n";
      os << "    WITH ROOTS:\n";
      for (Value root : info->roots) {
        os << "      " << root << "\n";
      }
      os << "    WITH PRODUCERS:\n";
      for (ConstValueInfo *producerInfo : info->producers) {
        os << "      " << producerInfo->constValue << "\n";
      }
    }
  }
}

void ConstExprAnalysis::dump() const { print(llvm::errs()); }

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
    Outcome postDecisionOutcome = decision->getOutcome();
    if (postDecisionOutcome != UNDECIDED) {
      LLVM_DEBUG(dbgs() << "ConstExprHoistPolicy(INVARIANT, ");
      if (postDecisionOutcome == ENABLE_HOIST) {
        LLVM_DEBUG(dbgs() << "ENABLE_HOIST");
      } else if (postDecisionOutcome == DISABLE_HOIST) {
        LLVM_DEBUG(dbgs() << "DISABLE_HOIST");
      }
      LLVM_DEBUG(dbgs() << "): " << info->constValue << "\n");
    }
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
        LLVM_DEBUG(dbgs() << "ConstExprHoistPolicy(" << i << ", ");
        if (decision->getOutcome() == ENABLE_HOIST) {
          LLVM_DEBUG(dbgs() << "ENABLE_HOIST");
        } else if (decision->getOutcome() == DISABLE_HOIST) {
          LLVM_DEBUG(dbgs() << "DISABLE_HOIST");
        }
        LLVM_DEBUG(dbgs() << "): " << info->constValue << "\n");
      }
    }

    if (!madeChange) {
      LLVM_DEBUG(dbgs() << "ConstExprHoistPolicy(" << i << ", CONVERGED)\n");
      break;
    }
  }

  for (auto *info : worklist) {
    Decision *decision = getDecision(info);
    if (decision->getOutcome() == UNDECIDED) {
      LLVM_DEBUG(dbgs() << "ConstExprHoistPolicy: Value did not converge: "
                        << info->constValue << "\n");
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
  //   - It has an anlyzed consumer that:
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
