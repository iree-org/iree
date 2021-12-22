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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

#define DEBUG_TYPE "iree-const-expr-analysis"

using llvm::dbgs;

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

ConstExprAnalysis::ConstExprAnalysis(Operation *rootOp) {
  Explorer explorer(rootOp, TraversalAction::SHALLOW);
  explorer.initialize();

  // Populate the constant roots for globals.
  explorer.forEachGlobal([&](const Explorer::GlobalInfo *info) {
    // Rely on globals having been canonicalized to immutable correctly.
    if (info->op.is_mutable()) return;
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

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
