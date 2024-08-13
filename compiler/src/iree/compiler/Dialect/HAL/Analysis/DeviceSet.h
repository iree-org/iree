// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_ANALYSIS_DEVICESET_H_
#define IREE_COMPILER_DIALECT_HAL_ANALYSIS_DEVICESET_H_

#include <optional>

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::HAL {

// Provides configuration queries over a set of devices.
class DeviceSet {
public:
  DeviceSet() = default;
  explicit DeviceSet(ArrayAttr targetsAttr);
  explicit DeviceSet(ArrayRef<IREE::HAL::DeviceTargetAttr> targetAttrs);
  explicit DeviceSet(const DenseSet<IREE::HAL::DeviceTargetAttr> &targetAttrs);
  ~DeviceSet();

  // Returns zero or more executable targets that may be used by any device.
  std::optional<SmallVector<IREE::HAL::ExecutableTargetAttr>>
  getExecutableTargets() const;

  // Returns true if there is any UnitAttr with |name| in any device.
  bool hasConfigAttrAny(StringRef name) const;

  // Returns true if all device configurations have a UnitAttr with |name|.
  bool hasConfigAttrAll(StringRef name) const;

  // Returns the AND of boolean attributes of |name| in all devices.
  // Returns nullopt if any config does not have the key defined indicating
  // that it's not statically known/runtime dynamic.
  std::optional<bool> getConfigAttrAnd(StringRef name) const;

  // Returns the OR of boolean attributes of |name| in all devices.
  // Returns nullopt if any config does not have the key defined indicating
  // that it's not statically known/runtime dynamic.
  std::optional<bool> getConfigAttrOr(StringRef name) const;

  // Returns the range of integer attributes of |name| in all devices.
  // Returns nullopt if any config does not have the key defined indicating
  // that it's not statically known/runtime dynamic.
  std::optional<StaticRange<APInt>> getConfigAttrRange(StringRef name) const;

private:
  DenseSet<IREE::HAL::DeviceTargetAttr> targetAttrs;
};

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_ANALYSIS_DEVICESET_H_
