// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_RESOURCEACCESS_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_RESOURCEACCESS_H_

#include <cstdint>

namespace mlir::iree_compiler::IREE::HAL {
class DeviceTargetAttr;
class TargetRegistry;

/// Minimum units, in bytes, of physical accesses beyond logical resource
/// ranges. Units must be positive powers of two. A unit of one adds no
/// constraint beyond byte ranges. Devices describe transfer implementations;
/// executable backends describe kernel storage implementations.
struct ResourceAccessRequirements {
  int64_t storageInBytes = 1;
  int64_t fillInBytes = 1;
  int64_t copyInBytes = 1;
};

/// Returns the intersection of requirements modelled by registered providers.
/// Providers without a requirements query retain their existing configuration.
ResourceAccessRequirements
getRequiredResourceAccess(DeviceTargetAttr target,
                          const TargetRegistry &registry);

/// Builds a target whose configuration also satisfies its providers' minima.
/// Preserves stricter and unrelated configuration. Intended for default target
/// construction; explicit target configuration is validated separately.
DeviceTargetAttr
buildDeviceTargetWithRequiredResourceAccess(DeviceTargetAttr target,
                                            const TargetRegistry &registry);

} // namespace mlir::iree_compiler::IREE::HAL
#endif // IREE_COMPILER_DIALECT_HAL_TARGET_RESOURCEACCESS_H_
