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

#include "iree/compiler/Dialect/Flow/Analysis/Dispatchability.h"

#include <list>

#include "iree/compiler/Dialect/Flow/Utils/DispatchUtils.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace iree_compiler {

// static
LogicalResult Dispatchability::annotateIR(ModuleOp moduleOp) {
  Dispatchability dispatchability;
  if (failed(dispatchability.recalculate(moduleOp))) {
    moduleOp.emitOpError()
        << "failed to analyze dispatchability for the module";
    return failure();
  }

  Builder builder(moduleOp.getContext());
  SymbolTable symbolTable(moduleOp);
  for (auto &funcDispatchability : dispatchability.funcDispatchability_) {
    auto funcOp = symbolTable.lookup<FuncOp>(funcDispatchability.first);
    funcOp.setAttr("dispatchable",
                   builder.getBoolAttr(funcDispatchability.second));
  }

  return success();
}

LogicalResult Dispatchability::recalculate(ModuleOp moduleOp) {
  funcDispatchability_.clear();
  funcCloneModuleOp_ = ModuleOp::create(UnknownLoc::get(moduleOp.getContext()));
  funcClones_.clear();

  // Run through all functions until we are able to compute their
  // dispatchability. We do this so that we can determine if calls are allowed.
  OpBuilder cloneBuilder(funcCloneModuleOp_.get());
  std::vector<FuncOp> nextWorklist(moduleOp.getOps<FuncOp>().begin(),
                                   moduleOp.getOps<FuncOp>().end());
  std::vector<FuncOp> worklist;
  bool anyChanged;
  do {
    anyChanged = false;
    worklist.swap(nextWorklist);
    nextWorklist.clear();
    for (auto funcOp : worklist) {
      auto isDispatchable = computeDispatchability(funcOp);
      if (isDispatchable.hasValue()) {
        funcDispatchability_[funcOp.getName()] = isDispatchable.getValue();
        if (isDispatchable.getValue()) {
          auto clonedFuncOp = cast<FuncOp>(cloneBuilder.clone(*funcOp));
          funcClones_[funcOp.getName()] = clonedFuncOp;
          funcCloneModuleOp_->push_back(clonedFuncOp);
        }
        anyChanged = true;
      } else {
        nextWorklist.push_back(funcOp);
      }
    }
  } while (anyChanged);
  if (!nextWorklist.empty()) {
    return moduleOp.emitError() << "cycle detected in dispatchability analysis";
  }

  return success();
}

Optional<bool> Dispatchability::computeDispatchability(FuncOp funcOp) {
  if (funcOp.isExternal()) {
    // We assume all imports have side-effects right now, but that may not be
    // the case. We should add an attribute and check for it.
    return false;
  }

  // TODO(b/144530470): replace with tablegen attributes/interfaces.
  for (auto &block : funcOp.getBlocks()) {
    for (auto &op : block.getOperations()) {
      if (!IREE::Flow::isOpOfKnownDialect(&op)) {
        // Custom dialects aren't dispatchable (yet).
        return false;
      } else if (auto callOp = dyn_cast<CallOp>(op)) {
        if (callOp.getCallee() == funcOp.getName()) {
          // Recursion.
          continue;
        }
        auto it = funcDispatchability_.find(callOp.callee());
        if (it == funcDispatchability_.end()) {
          // Not yet calculated - yield.
          return llvm::None;
        }
        return it->second;
      } else if (isa<CallIndirectOp>(op)) {
        // Indirect calls are not supported and must first be devirtualized.
        return false;
      } else if (isa<mlir::ReturnOp>(op)) {
        // TODO(benvanik): widen to all known terminators? sometimes they may
        // have side-effects.
        continue;
      } else if (isa<mhlo::DotOp>(op) || isa<mhlo::ConvOp>(op)) {
        // Some unfusable ops must remain on their own.
        return false;
      } else if (isa<mhlo::ReduceOp>(op) || isa<mhlo::ReduceWindowOp>(op)) {
        // Reductions always become flow ops.
        return false;

        // TODO: Properly handle region side effects.
      } else if (!MemoryEffectOpInterface::hasNoEffect(&op) ||
                 op.getNumRegions() != 0) {
        // Ops with side-effects cannot be dispatched as we must be able to
        // exactly model I/O.
        return false;
      }
    }
  }

  // All cases not handled above are (probably) dispatchable. This makes what we
  // do here a blocklist, though as we move towards more frontend dialects that
  // may not be the best idea.
  return true;
}

void Dispatchability::walkDispatchableOps(
    function_ref<void(FuncOp funcOp)> fn) {
  for (auto funcOp : funcClones_) {
    fn(funcOp.second);
  }
}

bool Dispatchability::isDispatchable(StringRef funcName) {
  return funcDispatchability_[funcName];
}

bool Dispatchability::isDispatchable(FuncOp funcOp) {
  return isDispatchable(funcOp.getName());
}

bool Dispatchability::isInvalidated(
    const AnalysisManager::PreservedAnalyses &pa) {
  return false;
}

}  // namespace iree_compiler
}  // namespace mlir
