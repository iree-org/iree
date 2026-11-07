// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/HAL/Target/ResourceAccess.h"

#include <algorithm>

#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler::IREE::HAL {

static ResourceAccessRequirements
intersectResourceAccessRequirements(ResourceAccessRequirements lhs,
                                    ResourceAccessRequirements rhs) {
  return {std::max(lhs.storageInBytes, rhs.storageInBytes),
          std::max(lhs.fillInBytes, rhs.fillInBytes),
          std::max(lhs.copyInBytes, rhs.copyInBytes)};
}

ResourceAccessRequirements
getRequiredResourceAccess(DeviceTargetAttr target,
                          const TargetRegistry &registry) {
  ResourceAccessRequirements result;
  if (auto device = registry.getTargetDevice(target.getDeviceID().getValue())) {
    if (auto required = device->getResourceAccessRequirements(target)) {
      result = intersectResourceAccessRequirements(result, *required);
    }
  }
  for (ExecutableTargetAttr executable : target.getExecutableTargets()) {
    if (auto backend =
            registry.getTargetBackend(executable.getBackend().getValue())) {
      if (auto required = backend->getResourceAccessRequirements(executable)) {
        result = intersectResourceAccessRequirements(result, *required);
      }
    }
  }
  return result;
}

DeviceTargetAttr
buildDeviceTargetWithRequiredResourceAccess(DeviceTargetAttr target,
                                            const TargetRegistry &registry) {
  auto required = getRequiredResourceAccess(target, registry);
  Builder builder(target.getContext());
  NamedAttrList config;
  if (auto existing = target.getConfiguration()) {
    config.append(existing.getValue());
  }
  for (auto [name, unit] :
       {std::pair<StringRef, int64_t>{"stream.min_access_granularity",
                                      required.storageInBytes},
        {"stream.min_fill_granularity", required.fillInBytes},
        {"stream.min_copy_granularity", required.copyInBytes}}) {
    auto existing = dyn_cast_if_present<IntegerAttr>(config.get(name));
    if (unit > (existing ? existing.getInt() : 1)) {
      config.set(name, builder.getI64IntegerAttr(unit));
    }
  }
  auto configuration = config.getDictionary(target.getContext());
  if (configuration == target.getConfiguration() ||
      (!target.getConfiguration() && configuration.empty())) {
    return target;
  }
  return DeviceTargetAttr::get(target.getContext(), target.getDeviceID(),
                               configuration, target.getExecutableTargets());
}

} // namespace mlir::iree_compiler::IREE::HAL
