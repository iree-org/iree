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
  std::optional<SmallVector<IREE::Util::GlobalOpInterface>>
  lookupDeviceGlobals(Value deviceValue);

  // Returns a set of possible targets of the given `!hal.device` global, if
  // analyzed.
  std::optional<DeviceSet>
  lookupDeviceTargets(IREE::Util::GlobalOpInterface deviceGlobalOp);

  // Returns a set of possible targets of the given `!hal.device` global, if
  // analyzed.
  std::optional<DeviceSet> lookupDeviceTargets(SymbolRefAttr deviceGlobalAttr);

  // Returns a set of possible targets of the given `!hal.device` value, if
  // analyzed.
  std::optional<DeviceSet> lookupDeviceTargets(Value deviceValue);

  // Gathers all possible device targets in the root op.
  // Ordering is undefined.
  void
  gatherAllDeviceTargets(SetVector<IREE::HAL::DeviceTargetAttr> &resultSet);

  // Gathers the set of device targets potentially referenced by the given
  // affinity. Targets are ordered by most likely to least likely.
  void gatherDeviceAffinityTargets(
      IREE::Stream::AffinityAttr affinityAttr, Operation *fromOp,
      SetVector<IREE::HAL::DeviceTargetAttr> &resultSet);

  // Gathers all executable targets from all devices in the root op.
  // This should generally be avoided and the scoped
  // gatherRequiredExecutableTargets gather should be used instead.
  void gatherAllExecutableTargets(
      SetVector<IREE::HAL::ExecutableTargetAttr> &resultSet);

  // Gathers all executable targets that may be required by the given host op.
  // This should be called on the most narrowly scoped op possible as multiple
  // devices may be used within the same function-like op and have different
  // requirements. This may return a set with more targets than expected.
  void gatherRequiredExecutableTargets(
      Operation *forOp, SetVector<IREE::HAL::ExecutableTargetAttr> &resultSet);

  // Gathers all executable targets that may be required for the given affinity.
  // This should be called on the most narrowly scoped op possible as multiple
  // devices may be used within the same function-like op and have different
  // requirements. This may return a set with more targets than expected.
  void gatherRequiredExecutableTargets(
      IREE::Stream::AffinityAttr affinityAttr, Operation *fromOp,
      SetVector<IREE::HAL::ExecutableTargetAttr> &resultSet);

private:
  // Recursively resolves the referenced device into targets.
  void gatherDeviceTargets(Attribute rootAttr, Operation *fromOp,
                           SetVector<IREE::HAL::DeviceTargetAttr> &resultSet);

  Explorer explorer;
  llvm::BumpPtrAllocator allocator;
  DFX::Solver solver;
  std::optional<DeviceSet> defaultDeviceSet;
  SmallVector<IREE::Util::GlobalOpInterface> deviceGlobals;
};

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_ANALYSIS_DEVICETARGET_H_
