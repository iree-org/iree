// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_ASSIGNTARGETDEVICESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-assign-target-devices
//===----------------------------------------------------------------------===//

// Strips leading and trailing whitespace from |value|.
static StringRef stripWhitespace(StringRef value) {
  while (!value.empty() && llvm::isSpace(value.front())) {
    value = value.substr(1);
  }
  while (!value.empty() && llvm::isSpace(value.back())) {
    value = value.substr(0, value.size() - 1);
  }
  return value;
}

// Strips leading and trailing double quotes from |value| if both exist.
static StringRef stripQuotes(StringRef value) {
  value = stripWhitespace(value);
  if (!value.empty() && value.front() == '"' && value.back() == '"') {
    return stripWhitespace(value.substr(1, value.size() - 2));
  }
  return value;
}

// Consumes a leading `name=` literal.
// Returns the `name` and leaves remaining characters after `=` in |value|.
// Returns an empty string if no name literal is present.
static StringRef consumeNameLiteral(StringRef &value) {
  value = stripWhitespace(value);
  const size_t splitIdx = value.find('=');
  if (splitIdx == std::string::npos) {
    return "";
  }
  for (size_t i = 0; i < splitIdx; ++i) {
    const char c = value[i];
    if (!llvm::isAlnum(c) && c != '_') {
      return value;
    }
  }
  const StringRef name = value.substr(0, splitIdx);
  value = stripWhitespace(value.substr(splitIdx + 1));
  return stripWhitespace(name);
}

// Consumes the first portion of |value| corresponding to a device alias.
// Expects: `abc` or `abc[123]` (and allows `"abc"[123]`).
// Only valid literals will be parsed (a-z0-9_).
// Returns the device ID and optional ordinal. All other unconsumed characters
// will remain in |value| upon return.
static std::pair<StringRef, std::optional<int64_t>>
consumeAliasLiteral(StringRef &value) {
  value = stripWhitespace(value);
  const size_t splitIdx = value.find(',');
  StringRef part =
      splitIdx == std::string::npos ? value : value.substr(0, splitIdx);

  StringRef deviceID = part;
  std::optional<int64_t> ordinal;

  const size_t ordinalIdx = part.find('[');
  if (ordinalIdx != std::string::npos) {
    deviceID = part.substr(0, ordinalIdx);
    StringRef ordinalStr = part.substr(ordinalIdx + 1);
    if (ordinalStr.ends_with(']')) {
      ordinalStr = ordinalStr.substr(0, ordinalStr.size() - 1);
    }
    int64_t ordinalI64 = 0;
    if (!ordinalStr.getAsInteger(10, ordinalI64)) {
      ordinal = ordinalI64;
    }
  }

  value = stripWhitespace(value.substr(part.size()));
  return std::make_pair(stripQuotes(deviceID), ordinal);
}

struct TargetSpec {
  StringAttr name;
  TypedAttr attr;
};

// Parses the user-provided string into a target spec.
//
// Supports attributes:
//  #hal.device.alias<...>
//  #hal.device.target<...>
//  #hal.device.select<...>
//  #hal.device.fallback<...>
// Supports convenience shorthand:
//  ...,... -> #hal.device.select<[...,...]>
//  target -> #hal.device.alias<"target">
//  target[0] -> #hal.device.alias<"target"[0]>
//  "target"[0] -> #hal.device.alias<"target"[0]>
// Supports name= prefixes:
//  name=... -> ...
static FailureOr<TargetSpec> parseTargetSpec(Location loc,
                                             StringRef targetSpecStr) {
  auto *context = loc.getContext();
  targetSpecStr = stripQuotes(targetSpecStr);

  // Check for a name prefix and strip it from the spec.
  StringRef name = consumeNameLiteral(targetSpecStr);
  StringAttr nameAttr =
      name.empty() ? StringAttr{} : StringAttr::get(context, name);

  // Parse the spec attributes.
  SmallVector<Attribute> attrs;
  while (!targetSpecStr.empty()) {
    TypedAttr typedAttr;
    if (targetSpecStr.starts_with('#')) {
      // MLIR attribute.
      size_t numRead = 0;
      auto parsedAttr = mlir::parseAttribute(targetSpecStr, context,
                                             /*type=*/nullptr, &numRead);
      if (!parsedAttr) {
        return mlir::emitError(loc) << "failed to parse target spec prefix `"
                                    << targetSpecStr << "`";
      }
      typedAttr = dyn_cast<TypedAttr>(parsedAttr);
      if (!typedAttr) {
        return mlir::emitError(loc) << "unexpected target attribute type: "
                                       "expected a `!hal.device` but got `"
                                    << parsedAttr << "`";
      }
      targetSpecStr = stripWhitespace(targetSpecStr.substr(numRead));
    } else {
      // Alias string.
      auto [deviceID, ordinal] = consumeAliasLiteral(targetSpecStr);
      typedAttr = IREE::HAL::DeviceAliasAttr::get(
          context, IREE::HAL::DeviceType::get(context),
          StringAttr::get(context, deviceID), ordinal, DictionaryAttr{});
    }

    if (!typedAttr || !isa<IREE::HAL::DeviceType>(typedAttr.getType())) {
      return mlir::emitError(loc) << "unexpected target attribute type: "
                                     "expected a `!hal.device` but got `"
                                  << typedAttr.getType() << "`";
    }
    attrs.push_back(typedAttr);

    if (targetSpecStr.empty()) {
      break; // done
    } else if (!targetSpecStr.starts_with(',')) {
      return mlir::emitError(loc)
             << "unexpected additional characters after parsing an element: `"
             << targetSpecStr << "`";
    }
    targetSpecStr = targetSpecStr.substr(1); // strip ,
  }

  if (attrs.empty()) {
    return mlir::emitError(loc) << "expected one or more target attributes";
  } else if (attrs.size() == 1) {
    return TargetSpec{nameAttr, cast<TypedAttr>(attrs.front())};
  } else {
    return TargetSpec{nameAttr,
                      IREE::HAL::DeviceSelectAttr::get(context, attrs)};
  }
}

struct AssignTargetDevicesPass
    : public IREE::HAL::impl::AssignTargetDevicesPassBase<
          AssignTargetDevicesPass> {
  using IREE::HAL::impl::AssignTargetDevicesPassBase<
      AssignTargetDevicesPass>::AssignTargetDevicesPassBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // If no targets are specified we can't do anything - another pass earlier
    // in the pipeline will have had to add the targets.
    if (targetDevices.empty()) {
      return;
    }

    // Check to see if targets are already specified and if so then no-op the
    // pass so that we don't mess with whatever the user intended.
    if (moduleOp->hasAttr("hal.device.targets")) {
      return;
    }

    // If there are any device globals declared then bail as it means the user
    // has already materialized the devices they want.
    for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOpInterface>()) {
      if (isa<IREE::HAL::DeviceType>(globalOp.getGlobalType())) {
        return;
      }
    }

    // Parse each spec and validate correctness.
    bool hasAnyNamed = false;
    bool hasAnyUnnamed = false;
    SmallVector<TargetSpec> targetSpecs;
    for (auto &targetDevice : targetDevices) {
      auto targetSpecOr = parseTargetSpec(moduleOp.getLoc(), targetDevice);
      if (failed(targetSpecOr)) {
        return signalPassFailure();
      }
      if (targetSpecOr->name) {
        hasAnyNamed = true;
      } else {
        hasAnyUnnamed = true;
      }
      targetSpecs.push_back(*targetSpecOr);
    }

    // If any spec has a name assigned then all must have names assigned.
    if (hasAnyNamed && hasAnyUnnamed) {
      emitError(moduleOp.getLoc())
          << "if any target device spec has a name then all must be named";
      return signalPassFailure();
    }

    if (hasAnyNamed) {
      // NOTE: we allow duplicate names to override assignment.
      llvm::MapVector<StringAttr, Attribute> deviceAttrMap;
      for (auto targetSpec : targetSpecs) {
        assert(targetSpec.name && "all devices must be named");
        deviceAttrMap[targetSpec.name] = targetSpec.attr;
      }
      SmallVector<NamedAttribute> deviceAttrs;
      for (auto [name, value] : deviceAttrMap) {
        deviceAttrs.push_back(NamedAttribute(name, value));
      }
      moduleOp->setAttr(
          "hal.device.targets",
          DictionaryAttr::get(moduleOp.getContext(), deviceAttrs));
    } else {
      SmallVector<Attribute> deviceAttrs;
      for (auto [name, value] : targetSpecs) {
        assert(!name && "no devices may have names");
        deviceAttrs.push_back(value);
      }
      moduleOp->setAttr("hal.device.targets",
                        ArrayAttr::get(moduleOp.getContext(), deviceAttrs));
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
