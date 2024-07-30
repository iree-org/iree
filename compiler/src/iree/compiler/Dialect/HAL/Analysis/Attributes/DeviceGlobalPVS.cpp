// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Analysis/Attributes/DeviceGlobalPVS.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-hal-device-analysis"

namespace mlir::iree_compiler::IREE::HAL {

//===----------------------------------------------------------------------===//
// DeviceGlobalValuePVS
//===----------------------------------------------------------------------===//

const char DeviceGlobalValuePVS::ID = 0;

void DeviceGlobalValuePVS::initializeValue(Value value, DFX::Solver &solver) {
  assert(isa<IREE::HAL::DeviceType>(value.getType()) &&
         "only initialize on values of type !hal.device");

  // If the value is a function arg of a public function then we'll never be
  // able to know (today). We could look for attributes defining device
  // properties but we can't recover a DeviceTargetAttr from them.
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (auto funcOp =
            dyn_cast<FunctionOpInterface>(blockArg.getOwner()->getParentOp())) {
      if (funcOp.isPublic()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "DeviceGlobalValuePVS: argument to a public function - "
                      "treating as undefined\n");
        unionAssumedWithUndef();
        indicatePessimisticFixpoint();
        return;
      }
    }
  }
}

ChangeStatus DeviceGlobalValuePVS::updateValue(Value value,
                                               DFX::Solver &solver) {
  StateType newState;
  auto traversalResult = TraversalResult::COMPLETE;

  // Walk into all producers of the SSA value.
  // Note that we may end up at multiple global loads of different globals
  // by walking up through calls/branches/etc.
  traversalResult |=
      solver.getExplorer().walkDefiningOps(value, [&](OpResult result) {
        updateFromDefiningOp(value, result, newState, solver);
        return WalkResult::advance();
      });

  if (traversalResult == TraversalResult::INCOMPLETE) {
    // Incomplete traversal because of external call graph edges or pointers.
    newState.unionAssumedWithUndef();
    newState.indicatePessimisticFixpoint();
  }
  return DFX::clampStateAndIndicateChange(getState(), newState);
}

void DeviceGlobalValuePVS::updateFromDefiningOp(Value value, OpResult result,
                                                StateType &newState,
                                                DFX::Solver &solver) {
  TypeSwitch<Operation *, void>(result.getOwner())
      .Case([&](mlir::arith::SelectOp op) {
        auto &truePVS = solver.getElementFor<DeviceGlobalValuePVS>(
            *this, Position::forValue(op.getTrueValue()),
            DFX::Resolution::REQUIRED);
        auto &falsePVS = solver.getElementFor<DeviceGlobalValuePVS>(
            *this, Position::forValue(op.getFalseValue()),
            DFX::Resolution::REQUIRED);
        newState ^= truePVS.getState();
        newState ^= falsePVS.getState();
      })
      .Case([&](IREE::Util::OptimizationBarrierOp op) {
        auto &sourcePVS = solver.getElementFor<DeviceGlobalValuePVS>(
            *this, Position::forValue(op.getOperand(0)),
            DFX::Resolution::REQUIRED);
        newState ^= sourcePVS.getState();
      })
      .Case([&](IREE::Util::GlobalLoadOpInterface op) {
        auto *globalInfo =
            solver.getExplorer().queryGlobalInfoFrom(op.getGlobalName(), op);
        newState.unionAssumed(globalInfo->op);
      })
      .Default([&](Operation *op) {});
}

const std::string DeviceGlobalValuePVS::getAsStr(AsmState &asmState) const {
  std::string str;
  llvm::raw_string_ostream sstream(str);
  sstream << "pvs: ";
  if (isValidState()) {
    sstream << "[";
    if (isUndefContained()) {
      sstream << "undef, ";
    }
    llvm::interleaveComma(getAssumedSet(), sstream,
                          [&](IREE::Util::GlobalOpInterface value) {
                            value.print(sstream, asmState);
                          });
    sstream << "]";
  } else {
    sstream << "(invalid)";
  }
  sstream.flush();
  return str;
}

} // namespace mlir::iree_compiler::IREE::HAL
