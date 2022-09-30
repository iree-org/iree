// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/Constant/ConstExpr.h"

#include "iree/compiler/Dialect/Util/Analysis/Constant/OpOracle.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#define DEBUG_TYPE "iree-const-expr-analysis"

using llvm::dbgs;

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

//===----------------------------------------------------------------------===//
// ConstExprAnalysis
//===----------------------------------------------------------------------===//

namespace {
OpOperand *findOperandFor(Operation *op, Value input) {
  for (OpOperand &operand : op->getOpOperands()) {
    if (operand.get() == input) return &operand;
  }
  return nullptr;
}

}  // namespace

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
    if (info->op.isGlobalMutable()) return;
    if (info->isIndirect) return;
    for (auto *use : info->uses) {
      auto loadOp = llvm::dyn_cast<GlobalLoadOp>(use);
      if (!loadOp) continue;
      constantRoots[loadOp.getResult()] = loadOp;
    }
  });

  // Populate the constant roots for all inline constants in the program.
  rootOp->walk([&](arith::ConstantOp constOp) {
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
      if (info->state != ConstValueInfo::UNKNOWN) continue;
      bool allConstants = true;
      for (ConstValueInfo *producerInfo : info->producers) {
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

ConstExprAnalysis::ConstValueInfo *ConstExprAnalysis::addInfo(
    Value constValue) {
  auto info = std::make_unique<ConstValueInfo>(constValue);
  constInfoMap[constValue] = info.get();
  allocedConstInfos.push_back(std::move(info));
  return allocedConstInfos.back().get();
}

void ConstExprAnalysis::expandToOp(Operation *op) {
  ConstExprOpInfo opInfo = ConstExprOpInfo::getForOp(op);
  for (auto result : op->getResults()) {
    auto foundIt = constInfoMap.find(result);
    if (foundIt != constInfoMap.end()) continue;

    // Generate new info record.
    auto *valueInfo = addInfo(result);
    if (!opInfo.isEligible) {
      // Put it in a NON_CONSTANT state and bail. This is terminal.
      valueInfo->state = ConstValueInfo::NON_CONSTANT;
      LLVM_DEBUG(dbgs() << "  EXPAND TO INELIGIBLE: " << result << "\n");
      continue;
    }

    // If here, then an unknown state.
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

      ConstValueInfo *producerInfo = constInfoMap.lookup(producer);
      assert(producerInfo && "should have producer info in map");
      valueInfo->producers.insert(producerInfo);
    }
  }
}

void ConstExprAnalysis::print(raw_ostream &os) const {
  os << "\nFOUND CONSTANTS:\n----------------\n";
  for (auto &info : allocedConstInfos) {
    if (info->state != ConstValueInfo::CONSTANT || info->isRoot) continue;
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
    const ConstExprAnalysis &analysis)
    : analysis(analysis), decisions(analysis.allocedConstInfos.size()) {
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
    worklist.push_back(it.get());
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
      if (decision->getOutcome() != UNDECIDED) continue;
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
      if (consumerDecision->getOutcome() != DISABLE_HOIST) continue;

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

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
