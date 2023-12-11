// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/Attributes/PotentialValues.h"

#include "llvm/Support/Debug.h"
#include "mlir/IR/Matchers.h"

#define DEBUG_TYPE "iree-util-attributes"
using llvm::dbgs;

namespace mlir::iree_compiler::IREE::Util {

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

} // namespace mlir::iree_compiler::IREE::Util
