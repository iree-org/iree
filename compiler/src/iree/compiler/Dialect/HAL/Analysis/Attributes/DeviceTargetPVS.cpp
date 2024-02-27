// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Analysis/Attributes/DeviceTargetPVS.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-hal-device-analysis"

namespace mlir::iree_compiler::IREE::HAL {

//===----------------------------------------------------------------------===//
// DeviceTargetGlobalPVS
//===----------------------------------------------------------------------===//

const char DeviceTargetGlobalPVS::ID = 0;

void DeviceTargetGlobalPVS::initializeOperation(IREE::Util::GlobalOp globalOp,
                                                DFX::Solver &solver) {
  assert(isa<IREE::HAL::DeviceType>(globalOp.getType()) &&
         "only initialize on globals of type !hal.device");

  // We only support immutable initialized device globals.
  // We could track usage up through stores to handle the mutable case but
  // the compiler does not generate such programs today.
  auto *globalInfo = solver.getExplorer().getGlobalInfo(globalOp);
  if (!globalInfo || globalInfo->isIndirect || globalOp.isGlobalMutable()) {
    LLVM_DEBUG(llvm::dbgs()
               << "DeviceTargetGlobalPVS: mutable device globals or those used "
                  "indirectly are not yet implemented\n");
    unionAssumedWithUndef();
    indicatePessimisticFixpoint();
    return;
  }

  // Use the initial value to populate the potential value set.
  std::function<bool(Attribute)> unionAttr;
  unionAttr = [&](Attribute attr) -> bool {
    return TypeSwitch<Attribute, bool>(attr)
        .Case<IREE::HAL::DeviceTargetAttr>([&](auto targetAttr) {
          LLVM_DEBUG({
            llvm::dbgs() << "DeviceTargetGlobalPVS: unioning with target: ";
            attr.print(llvm::dbgs());
            llvm::dbgs() << "\n";
          });
          unionAssumed(targetAttr);
          return true;
        })
        .Case<IREE::HAL::DeviceFallbackAttr>([&](auto fallbackAttr) {
          LLVM_DEBUG({
            llvm::dbgs() << "DeviceTargetGlobalPVS: unioning with fallback: ";
            attr.print(llvm::dbgs());
            llvm::dbgs() << "\n";
          });
          auto *fallbackInfo = solver.getExplorer().queryGlobalInfoFrom(
              fallbackAttr.getName().getValue(), globalOp);
          if (!fallbackInfo) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "DeviceTargetGlobalPVS: !! failed to find fallback global "
                << fallbackAttr.getName().getValue() << "\n");
            return false;
          }
          auto fallbackPVS =
              solver.getOrCreateElementFor<DeviceTargetGlobalPVS>(
                  Position::forOperation(fallbackInfo->op));
          if (fallbackPVS.isUndefContained()) {
            LLVM_DEBUG(llvm::dbgs()
                       << "DeviceTargetGlobalPVS: !! fallback is undefined\n");
            return false;
          }
          unionAssumed(fallbackPVS.getState());
          return true;
        })
        .Case<IREE::HAL::DeviceSelectAttr>([&](auto selectAttr) {
          LLVM_DEBUG({
            llvm::dbgs() << "DeviceTargetGlobalPVS: unioning with selected "
                            "child devices: ";
            attr.print(llvm::dbgs());
            llvm::dbgs() << "\n";
          });
          for (auto childAttr : selectAttr.getDevices()) {
            if (!unionAttr(childAttr)) {
              return false;
            }
          }
          return true;
        })
        .Default(
            [&](auto attr) {
              LLVM_DEBUG(
                  llvm::dbgs()
                  << "DeviceTargetGlobalPVS: !! unknown initial value type\n");
              return false;
            });
  };
  if (auto initialValueAttr = globalOp.getInitialValueAttr()) {
    if (unionAttr(initialValueAttr)) {
      indicateOptimisticFixpoint();
    } else {
      unionAssumedWithUndef();
      indicatePessimisticFixpoint();
    }
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "DeviceTargetGlobalPVS: no initial value, dynamically "
                  "configure devices not yet implemented\n");
    unionAssumedWithUndef();
    indicatePessimisticFixpoint();
  }
}

ChangeStatus
DeviceTargetGlobalPVS::updateOperation(IREE::Util::GlobalOp globalOp,
                                       DFX::Solver &solver) {
  // We only support running on initialized globals today.
  // We could support walking store/load or other things, though.
  return ChangeStatus::UNCHANGED;
}

const std::string DeviceTargetGlobalPVS::getAsStr(AsmState &asmState) const {
  std::string str;
  llvm::raw_string_ostream sstream(str);
  sstream << "pvs: ";
  if (isValidState()) {
    sstream << "[";
    if (isUndefContained()) {
      sstream << "undef, ";
    }
    llvm::interleaveComma(getAssumedSet(), sstream,
                          [&](IREE::HAL::DeviceTargetAttr value) {
                            cast<Attribute>(value).print(sstream);
                          });
    sstream << "]";
  } else {
    sstream << "(invalid)";
  }
  sstream.flush();
  return str;
}

//===----------------------------------------------------------------------===//
// DeviceTargetValuePVS
//===----------------------------------------------------------------------===//

const char DeviceTargetValuePVS::ID = 0;

void DeviceTargetValuePVS::initializeValue(Value value, DFX::Solver &solver) {
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
                   << "DeviceTargetValuePVS: argument to a public function - "
                      "treating as undefined\n");
        unionAssumedWithUndef();
        indicatePessimisticFixpoint();
        return;
      }
    }
  }
}

ChangeStatus DeviceTargetValuePVS::updateValue(Value value,
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

void DeviceTargetValuePVS::updateFromDefiningOp(Value value, OpResult result,
                                                StateType &newState,
                                                DFX::Solver &solver) {
  TypeSwitch<Operation *, void>(result.getOwner())
      .Case([&](mlir::arith::SelectOp op) {
        auto &truePVS = solver.getElementFor<DeviceTargetValuePVS>(
            *this, Position::forValue(op.getTrueValue()),
            DFX::Resolution::REQUIRED);
        auto &falsePVS = solver.getElementFor<DeviceTargetValuePVS>(
            *this, Position::forValue(op.getFalseValue()),
            DFX::Resolution::REQUIRED);
        newState ^= truePVS.getState();
        newState ^= falsePVS.getState();
      })
      .Case([&](IREE::Util::OptimizationBarrierOp op) {
        auto &sourcePVS = solver.getElementFor<DeviceTargetValuePVS>(
            *this, Position::forValue(op.getOperand(0)),
            DFX::Resolution::REQUIRED);
        newState ^= sourcePVS.getState();
      })
      .Case([&](IREE::Util::GlobalLoadOpInterface op) {
        auto *globalInfo =
            solver.getExplorer().queryGlobalInfoFrom(op.getGlobalName(), op);
        auto &globalPVS = solver.getElementFor<DeviceTargetGlobalPVS>(
            *this, Position::forOperation(globalInfo->op),
            DFX::Resolution::REQUIRED);
        newState ^= globalPVS.getState();
      })
      .Default([&](Operation *op) {});
}

const std::string DeviceTargetValuePVS::getAsStr(AsmState &asmState) const {
  std::string str;
  llvm::raw_string_ostream sstream(str);
  sstream << "pvs: ";
  if (isValidState()) {
    sstream << "[";
    if (isUndefContained()) {
      sstream << "undef, ";
    }
    llvm::interleaveComma(getAssumedSet(), sstream,
                          [&](IREE::HAL::DeviceTargetAttr value) {
                            cast<Attribute>(value).print(sstream);
                          });
    sstream << "]";
  } else {
    sstream << "(invalid)";
  }
  sstream.flush();
  return str;
}

} // namespace mlir::iree_compiler::IREE::HAL
