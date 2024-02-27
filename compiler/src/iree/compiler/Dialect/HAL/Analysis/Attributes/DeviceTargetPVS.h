// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_ANALYSIS_ATTRIBUTES_DEVICETARGETPVS_H_
#define IREE_COMPILER_DIALECT_HAL_ANALYSIS_ATTRIBUTES_DEVICETARGETPVS_H_

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::HAL {

//===----------------------------------------------------------------------===//
// DeviceTargetGlobalPVS
//===----------------------------------------------------------------------===//

// Set of potential IREE::HAL::DeviceTargetAttr values for an initialized
// !hal.device global. When defined the device global may take on the traits of
// any of the potential values.
class DeviceTargetGlobalPVS
    : public DFX::StateWrapper<
          DFX::PotentialValuesState<IREE::HAL::DeviceTargetAttr>,
          DFX::TypedOperationElement<IREE::Util::GlobalOp>> {
public:
  using BaseType =
      DFX::StateWrapper<DFX::PotentialValuesState<IREE::HAL::DeviceTargetAttr>,
                        DFX::TypedOperationElement<IREE::Util::GlobalOp>>;
  using BaseType::BaseType;

  static DeviceTargetGlobalPVS &createForPosition(const Position &pos,
                                                  DFX::Solver &solver) {
    return *(new (solver.getAllocator()) DeviceTargetGlobalPVS(pos));
  }

  // Identity definitions.
  const std::string getName() const override { return "DeviceTargetGlobalPVS"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override;

private:
  void initializeOperation(IREE::Util::GlobalOp globalOp,
                           DFX::Solver &solver) override;
  ChangeStatus updateOperation(IREE::Util::GlobalOp globalOp,
                               DFX::Solver &solver) override;
};

//===----------------------------------------------------------------------===//
// DeviceTargetValuePVS
//===----------------------------------------------------------------------===//

// Set of potential values for a !hal.device SSA value.
// When defined the value may take on the traits of any of the potential values.
class DeviceTargetValuePVS
    : public DFX::StateWrapper<
          DFX::PotentialValuesState<IREE::HAL::DeviceTargetAttr>,
          DFX::ValueElement> {
public:
  using BaseType =
      DFX::StateWrapper<DFX::PotentialValuesState<IREE::HAL::DeviceTargetAttr>,
                        DFX::ValueElement>;
  using BaseType::BaseType;

  static DeviceTargetValuePVS &createForPosition(const Position &pos,
                                                 DFX::Solver &solver) {
    return *(new (solver.getAllocator()) DeviceTargetValuePVS(pos));
  }

  // Identity definitions.
  const std::string getName() const override { return "DeviceTargetValuePVS"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override;

private:
  void initializeValue(Value value, DFX::Solver &solver) override;
  ChangeStatus updateValue(Value value, DFX::Solver &solver) override;
  void updateFromDefiningOp(Value value, OpResult result, StateType &newState,
                            DFX::Solver &solver);
};

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_ANALYSIS_ATTRIBUTES_DEVICETARGETPVS_H_
