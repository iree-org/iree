// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_ANALYSIS_ATTRIBUTES_DEVICEGLOBALPVS_H_
#define IREE_COMPILER_DIALECT_HAL_ANALYSIS_ATTRIBUTES_DEVICEGLOBALPVS_H_

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::HAL {

//===----------------------------------------------------------------------===//
// DeviceGlobalValuePVS
//===----------------------------------------------------------------------===//

// Set of potential globals that provide a !hal.device SSA value.
// A set size of 1 indicates that the device SSA value is a particular device.
// Multiple entries indicate that multiple code paths may route to the value
// with different devices selected.
class DeviceGlobalValuePVS
    : public DFX::StateWrapper<
          DFX::PotentialValuesState<IREE::Util::GlobalOpInterface>,
          DFX::ValueElement> {
public:
  using BaseType = DFX::StateWrapper<
      DFX::PotentialValuesState<IREE::Util::GlobalOpInterface>,
      DFX::ValueElement>;
  using BaseType::BaseType;

  static DeviceGlobalValuePVS &createForPosition(const Position &pos,
                                                 DFX::Solver &solver) {
    return *(new (solver.getAllocator()) DeviceGlobalValuePVS(pos));
  }

  // Identity definitions.
  const std::string getName() const override { return "DeviceGlobalValuePVS"; }
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

#endif // IREE_COMPILER_DIALECT_HAL_ANALYSIS_ATTRIBUTES_DEVICEGLOBALPVS_H_
