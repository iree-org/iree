// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Analysis/DeviceSet.h"

namespace mlir::iree_compiler::IREE::HAL {

//===----------------------------------------------------------------------===//
// DeviceSet
//===----------------------------------------------------------------------===//

DeviceSet::DeviceSet(ArrayAttr targetsAttr) {
  for (auto targetAttr :
       targetsAttr.getAsRange<IREE::HAL::DeviceTargetAttr>()) {
    targetAttrs.insert(targetAttr);
  }
}

DeviceSet::DeviceSet(ArrayRef<IREE::HAL::DeviceTargetAttr> targetAttrs) {
  for (auto targetAttr : targetAttrs) {
    this->targetAttrs.insert(targetAttr);
  }
}

DeviceSet::DeviceSet(const DenseSet<IREE::HAL::DeviceTargetAttr> &targetAttrs)
    : targetAttrs(targetAttrs) {}

DeviceSet::~DeviceSet() = default;

std::optional<SmallVector<IREE::HAL::ExecutableTargetAttr>>
DeviceSet::getExecutableTargets() const {
  if (targetAttrs.empty()) {
    return std::nullopt;
  }
  SetVector<IREE::HAL::ExecutableTargetAttr> resultAttrs;
  for (auto targetAttr : targetAttrs) {
    targetAttr.getExecutableTargets(resultAttrs);
  }
  return llvm::to_vector(resultAttrs);
}

template <typename AttrT>
static std::optional<typename AttrT::ValueType> joinConfigAttrs(
    const DenseSet<IREE::HAL::DeviceTargetAttr> &targetAttrs, StringRef name,
    std::function<typename AttrT::ValueType(typename AttrT::ValueType,
                                            typename AttrT::ValueType)>
        join) {
  if (targetAttrs.empty()) {
    return std::nullopt;
  }
  std::optional<typename AttrT::ValueType> result;
  for (auto targetAttr : targetAttrs) {
    auto configAttr = targetAttr.getConfiguration();
    if (!configAttr) {
      return std::nullopt;
    }
    auto valueAttr = configAttr.getAs<AttrT>(name);
    if (!valueAttr) {
      return std::nullopt;
    } else if (!result) {
      result = valueAttr.getValue();
    } else {
      result = join(result.value(), valueAttr.getValue());
    }
  }
  return result;
}

template <typename AttrT>
static std::optional<StaticRange<typename AttrT::ValueType>>
joinConfigStaticRanges(const DenseSet<IREE::HAL::DeviceTargetAttr> &targetAttrs,
                       StringRef name,
                       std::function<StaticRange<typename AttrT::ValueType>(
                           StaticRange<typename AttrT::ValueType>,
                           StaticRange<typename AttrT::ValueType>)>
                           join) {
  if (targetAttrs.empty()) {
    return std::nullopt;
  }
  std::optional<StaticRange<typename AttrT::ValueType>> result;
  for (auto targetAttr : targetAttrs) {
    auto configAttr = targetAttr.getConfiguration();
    if (!configAttr) {
      return std::nullopt;
    }
    auto valueAttr = configAttr.getAs<AttrT>(name);
    if (!valueAttr) {
      return std::nullopt;
    } else if (!result) {
      result = valueAttr.getValue();
    } else {
      result =
          join(result.value(),
               StaticRange<typename AttrT::ValueType>{valueAttr.getValue()});
    }
  }
  return result;
}

bool DeviceSet::hasConfigAttrAny(StringRef name) const {
  for (auto targetAttr : targetAttrs) {
    if (auto configAttr = targetAttr.getConfiguration()) {
      if (configAttr.get(name)) {
        return true;
      }
    }
  }
  return false;
}

bool DeviceSet::hasConfigAttrAll(StringRef name) const {
  for (auto targetAttr : targetAttrs) {
    auto configAttr = targetAttr.getConfiguration();
    if (!configAttr || !configAttr.get(name)) {
      return false;
    }
  }
  return true;
}

std::optional<bool> DeviceSet::getConfigAttrAnd(StringRef name) const {
  return joinConfigAttrs<BoolAttr>(
      targetAttrs, name, [](bool lhs, bool rhs) { return lhs && rhs; });
}

std::optional<bool> DeviceSet::getConfigAttrOr(StringRef name) const {
  return joinConfigAttrs<BoolAttr>(
      targetAttrs, name, [](bool lhs, bool rhs) { return lhs || rhs; });
}

std::optional<StaticRange<APInt>>
DeviceSet::getConfigAttrRange(StringRef name) const {
  return joinConfigStaticRanges<IntegerAttr>(
      targetAttrs, name, [](StaticRange<APInt> lhs, StaticRange<APInt> rhs) {
        return StaticRange<APInt>{
            llvm::APIntOps::smin(lhs.min, rhs.min),
            llvm::APIntOps::smax(lhs.max, rhs.max),
        };
      });
}

} // namespace mlir::iree_compiler::IREE::HAL
