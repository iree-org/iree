// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_ANALYSIS_DEVICETARGET_H_
#define IREE_COMPILER_DIALECT_HAL_ANALYSIS_DEVICETARGET_H_

#include <optional>

#include "iree/compiler/Dialect/HAL/Analysis/DeviceSet.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::HAL {

//===----------------------------------------------------------------------===//
// DeviceAnalysis
//===----------------------------------------------------------------------===//

// Performs whole-program analysis of device traits (limits, configuration, etc)
// and allows for queries against `!hal.device` values for known traits.
//
// Though safe to run at any time this may not provide meaningful results until
// after devices have been materialized and the program has been converted into
// the HAL dialect.
class DeviceAnalysis {
public:
  explicit DeviceAnalysis(Operation *rootOp);
  ~DeviceAnalysis();

  Explorer &getExplorer() { return explorer; }

  // Runs analysis and populates the device traits map.
  // May fail if analysis cannot be completed due to unsupported or unknown IR.
  LogicalResult run();

  // Returns a set of all !hal.device globals in the analyzed root op in the
  // order they are declared in the root op.
  ArrayRef<IREE::Util::GlobalOpInterface> getDeviceGlobals() {
    return deviceGlobals;
  }

  // Returns a set of possible device globals of the given `!hal.device` value,
  // if analyzed.
  std::optional<SetVector<IREE::Util::GlobalOpInterface>>
  lookupDeviceGlobals(Value deviceValue);

  // Returns a set of possible targets of the given `!hal.device` value, if
  // analyzed.
  std::optional<DeviceSet> lookupDeviceTargets(Value deviceValue);

private:
  Explorer explorer;
  llvm::BumpPtrAllocator allocator;
  DFX::Solver solver;
  std::optional<DeviceSet> defaultDeviceSet;
  SmallVector<IREE::Util::GlobalOpInterface> deviceGlobals;
};

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_ANALYSIS_DEVICETARGET_H_
