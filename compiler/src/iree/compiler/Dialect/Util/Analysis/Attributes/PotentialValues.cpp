// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/Attributes/PotentialValues.h"

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"

#define DEBUG_TYPE "iree-util-attributes"
using llvm::dbgs;

namespace mlir::iree_compiler::IREE::Util {

static std::string
getPVSAsStr(const DFX::PotentialConstantIntValuesState &pvs) {
  std::string str;
  llvm::raw_string_ostream sstream(str);
  sstream << "pvs: ";
  if (pvs.isValidState()) {
    sstream << "[";
    if (pvs.isUndefContained()) {
      sstream << "undef, ";
    }
    llvm::interleaveComma(pvs.getAssumedSet(), sstream,
                          [&](APInt value) { value.print(sstream, false); });
    sstream << "]";
  } else {
    sstream << "(invalid)";
  }
  sstream.flush();
  return str;
}

//===----------------------------------------------------------------------===//
// ConstantAttributePVS
//===----------------------------------------------------------------------===//

const char ConstantAttributePVS::ID = 0;

void ConstantAttributePVS::initializeValue(Value value, DFX::Solver &solver) {
  Attribute staticValue;
  if (matchPattern(value, m_Constant(&staticValue))) {
    LLVM_DEBUG(dbgs() << "ConstantAttributePVS: Match constant\n");
    unionAssumed(staticValue);
    indicateOptimisticFixpoint();
  }
}

ChangeStatus ConstantAttributePVS::updateValue(Value value,
                                               DFX::Solver &solver) {
  StateType newState;
  if (solver.getExplorer().walkDefiningOps(value, [&](OpResult result) {
        Attribute staticValue;
        if (matchPattern(value, m_Constant(&staticValue))) {
          unionAssumed(staticValue);
          return WalkResult::advance();
        }

        // TODO: Walk joins.

        // Dynamic ops we can't see through.
        newState.unionAssumedWithUndef();
        newState.indicatePessimisticFixpoint();
        return WalkResult::advance();
      }) == TraversalResult::INCOMPLETE) {
    // Incomplete traversal.
    newState.unionAssumedWithUndef();
    newState.indicatePessimisticFixpoint();
  }

  return DFX::clampStateAndIndicateChange(getState(), newState);
}

const std::string ConstantAttributePVS::getAsStr(AsmState &asmState) const {
  std::string str;
  llvm::raw_string_ostream sstream(str);
  sstream << "pvs: ";
  if (isValidState()) {
    sstream << "[";
    if (isUndefContained()) {
      sstream << "undef, ";
    }
    llvm::interleaveComma(getAssumedSet(), sstream,
                          [&](Attribute value) { value.print(sstream); });
    sstream << "]";
  } else {
    sstream << "(invalid)";
  }
  sstream.flush();
  return str;
}

//===----------------------------------------------------------------------===//
// GlobalPVS
//===----------------------------------------------------------------------===//

const char GlobalPVS::ID = 0;

void GlobalPVS::initializeOperation(IREE::Util::GlobalOp globalOp,
                                    DFX::Solver &solver) {
  auto *globalInfo = solver.getExplorer().getGlobalInfo(globalOp);
  if (!globalInfo || globalInfo->isIndirect) {
    // Cannot perform analysis.
    indicatePessimisticFixpoint();
  } else if (globalInfo) {
    if (auto initialValue = llvm::dyn_cast_if_present<IntegerAttr>(
            globalOp.getInitialValueAttr())) {
      // Initial value is available for use; stored values from the rest of the
      // program will come during iteration.
      unionAssumed(initialValue.getValue());
    }
  }
}

ChangeStatus GlobalPVS::updateOperation(IREE::Util::GlobalOp globalOp,
                                        DFX::Solver &solver) {
  StateType newState;
  auto *globalInfo = solver.getExplorer().getGlobalInfo(globalOp);
  for (auto use : globalInfo->uses) {
    auto storeOp = dyn_cast<IREE::Util::GlobalStoreOpInterface>(use);
    if (!storeOp)
      continue;
    auto value = solver.getElementFor<IntValuePVS>(
        *this, Position::forValue(storeOp.getStoredGlobalValue()),
        DFX::Resolution::REQUIRED);
    if (value.isValidState()) {
      newState.unionAssumed(value);
    } else {
      newState.unionAssumedWithUndef();
      newState.indicatePessimisticFixpoint();
    }
  }
  return DFX::clampStateAndIndicateChange(getState(), newState);
}

const std::string GlobalPVS::getAsStr(AsmState &asmState) const {
  return getPVSAsStr(getState());
}

//===----------------------------------------------------------------------===//
// IntValuePVS
//===----------------------------------------------------------------------===//

const char IntValuePVS::ID = 0;

void IntValuePVS::initializeValue(Value value, DFX::Solver &solver) {
  APInt staticValue;
  if (matchPattern(value, m_ConstantInt(&staticValue))) {
    LLVM_DEBUG(dbgs() << "IntValuePVS: Match constant " << staticValue << "\n");
    unionAssumed(staticValue);
    indicateOptimisticFixpoint();
  }
}

ChangeStatus IntValuePVS::updateValue(Value value, DFX::Solver &solver) {
  StateType newState;
  if (solver.getExplorer().walkDefiningOps(value, [&](OpResult result) {
        APInt staticValue;
        if (matchPattern(result, m_ConstantInt(&staticValue))) {
          newState.unionAssumed(staticValue);
          return WalkResult::advance();
        }

        if (auto loadOp = dyn_cast<IREE::Util::GlobalLoadOpInterface>(
                result.getDefiningOp())) {
          auto *globalInfo = solver.getExplorer().queryGlobalInfoFrom(
              loadOp.getGlobalName(), loadOp);
          auto global = solver.getElementFor<GlobalPVS>(
              *this, Position::forOperation(globalInfo->op),
              DFX::Resolution::REQUIRED);
          if (global.isValidState()) {
            newState.unionAssumed(global);
            return WalkResult::advance();
          }
        }

        // TODO(benvanik): more ops supported for joining. We could for
        // example walk the lhs/rhs of elementwise ops and perform the set
        // operations (so addi %lhs, %rhs could produce a PVS of all of %lhs
        // summed to all of %rhs). May not be worth it, though.
        // TODO(benvanik): move select op walking to the explorer.
        if (auto selectOp =
                dyn_cast<mlir::arith::SelectOp>(result.getDefiningOp())) {
          auto lhs = solver.getElementFor<IntValuePVS>(
              *this, Position::forValue(selectOp.getTrueValue()),
              DFX::Resolution::REQUIRED);
          auto rhs = solver.getElementFor<IntValuePVS>(
              *this, Position::forValue(selectOp.getFalseValue()),
              DFX::Resolution::REQUIRED);
          if (!lhs.isValidState() || !rhs.isValidState()) {
            newState.unionAssumedWithUndef();
            newState.indicatePessimisticFixpoint();
          } else {
            newState.unionAssumed(lhs);
            newState.unionAssumed(rhs);
          }
          return WalkResult::advance();
        }

        // Some other dynamic value we can't analyze (yet).
        newState.unionAssumedWithUndef();
        newState.indicatePessimisticFixpoint();
        return WalkResult::advance();
      }) == TraversalResult::INCOMPLETE) {
    newState.unionAssumedWithUndef();
    newState.indicatePessimisticFixpoint();
  }
  return DFX::clampStateAndIndicateChange(getState(), newState);
}

const std::string IntValuePVS::getAsStr(AsmState &asmState) const {
  return getPVSAsStr(getState());
}

} // namespace mlir::iree_compiler::IREE::Util
