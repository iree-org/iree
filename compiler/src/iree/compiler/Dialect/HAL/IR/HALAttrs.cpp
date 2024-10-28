// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/HAL/IR/HALAttrs.cpp.inc" // IWYU pragma: keep
#include "iree/compiler/Dialect/HAL/IR/HALEnums.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::HAL {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

template <typename AttrType>
static LogicalResult parseEnumAttr(AsmParser &parser, StringRef attrName,
                                   AttrType &attr) {
  Attribute genericAttr;
  auto loc = parser.getCurrentLocation();
  if (failed(parser.parseAttribute(genericAttr,
                                   parser.getBuilder().getNoneType()))) {
    return parser.emitError(loc)
           << "failed to parse '" << attrName << "' enum string value";
  }
  auto stringAttr = llvm::dyn_cast<StringAttr>(genericAttr);
  if (!stringAttr) {
    return parser.emitError(loc)
           << "expected " << attrName << " attribute specified as string";
  }
  auto symbolized =
      symbolizeEnum<typename AttrType::ValueType>(stringAttr.getValue());
  if (!symbolized.hasValue()) {
    return parser.emitError(loc)
           << "failed to parse '" << attrName << "' enum value";
  }
  attr = AttrType::get(parser.getBuilder().getContext(), symbolized.getValue());
  return success();
}

template <typename AttrType>
static LogicalResult parseOptionalEnumAttr(AsmParser &parser,
                                           StringRef attrName, AttrType &attr) {
  if (succeeded(parser.parseOptionalQuestion())) {
    // Special case `?` to indicate any/none/undefined/etc.
    attr = AttrType::get(parser.getBuilder().getContext(), 0);
    return success();
  }
  return parseEnumAttr<AttrType>(parser, attrName, attr);
}

//===----------------------------------------------------------------------===//
// #hal.collective<*>
//===----------------------------------------------------------------------===//

// See the iree/hal/command_buffer.h iree_hal_collective_op_t for details.
uint32_t CollectiveAttr::getEncodedValue() const {
  union {
    uint32_t packed; // packed value
    struct {
      uint8_t kind;
      uint8_t reduction;
      uint8_t elementType;
      uint8_t reserved;
    };
  } value = {0};
  value.kind = static_cast<uint8_t>(getKind());
  value.reduction = static_cast<uint8_t>(
      getReduction().value_or(CollectiveReductionOp::None));
  value.elementType = static_cast<uint8_t>(getElementType());
  return value.packed;
}

//===----------------------------------------------------------------------===//
// #hal.pipeline.layout<*>
//===----------------------------------------------------------------------===//

PipelineBindingAttr PipelineLayoutAttr::getBinding(int64_t ordinal) const {
  assert(ordinal >= 0 && ordinal < getBindings().size() &&
         "binding ordinal out of bounds");
  return getBindings()[ordinal];
}

//===----------------------------------------------------------------------===//
// #hal.executable.target<*>
//===----------------------------------------------------------------------===//

// static
ExecutableTargetAttr ExecutableTargetAttr::get(MLIRContext *context,
                                               StringRef backend,
                                               StringRef format) {
  return get(context, StringAttr::get(context, backend),
             StringAttr::get(context, format), DictionaryAttr::get(context));
}

// static
Attribute ExecutableTargetAttr::parse(AsmParser &p, Type type) {
  StringAttr backendAttr;
  StringAttr formatAttr;
  DictionaryAttr configurationAttr;
  // `<"backend", "format"`
  if (failed(p.parseLess()) || failed(p.parseAttribute(backendAttr)) ||
      failed(p.parseComma()) || failed(p.parseAttribute(formatAttr))) {
    return {};
  }
  // `, {config}`
  if (succeeded(p.parseOptionalComma()) &&
      failed(p.parseAttribute(configurationAttr))) {
    return {};
  }
  // `>`
  if (failed(p.parseGreater())) {
    return {};
  }
  return get(p.getContext(), backendAttr, formatAttr, configurationAttr);
}

void ExecutableTargetAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  p.printAttribute(getBackend());
  os << ", ";
  p.printAttribute(getFormat());
  auto config = getConfiguration();
  if (config && !config.empty()) {
    os << ", ";
    p.printAttribute(config);
  }
  os << ">";
}

std::string ExecutableTargetAttr::getSymbolNameFragment() const {
  return sanitizeSymbolName(getFormat().getValue().lower());
}

bool ExecutableTargetAttr::hasConfigurationAttr(StringRef name) {
  auto configAttr = getConfiguration();
  return configAttr && configAttr.get(name);
}

// For now this is very simple: if there are any specified fields that are
// present in this attribute they must match. We could allow target backends
// to customize this via attribute interfaces in the future if we needed.
bool ExecutableTargetAttr::isGenericOf(
    IREE::HAL::ExecutableTargetAttr specificAttr) {
  if (getBackend() != specificAttr.getBackend() ||
      getFormat() != specificAttr.getFormat()) {
    // Totally different backends and binary formats.
    // There may be cases where we want to share things - such as when targeting
    // both DLLs and dylibs or something - but today almost all of these are
    // unique situations.
    return false;
  }

  // If the config is empty on either we can quickly match.
  // This is the most common case for users manually specifying targets.
  auto genericConfigAttr = getConfiguration();
  auto specificConfigAttr = specificAttr.getConfiguration();
  if (!genericConfigAttr || !specificConfigAttr)
    return true;

  // Ensure all fields in specificConfigAttr either don't exist or match.
  for (auto expectedAttr : specificConfigAttr.getValue()) {
    auto actualValue = genericConfigAttr.getNamed(expectedAttr.getName());
    if (!actualValue) {
      continue; // ignore, not present in generic
    }
    if (actualValue->getValue() != expectedAttr.getValue()) {
      return false; // mismatch, both have values but they differ
    }
  }

  // Ensure all fields in genericConfigAttr exist in the specific one.
  // If missing then the generic is _more_ specific and can't match.
  for (auto actualAttr : genericConfigAttr.getValue()) {
    if (!specificConfigAttr.getNamed(actualAttr.getName())) {
      return false; // mismatch, present in generic but not specific
    }
  }

  // All fields match or are omitted in the generic version.
  return true;
}

// static
ExecutableTargetAttr ExecutableTargetAttr::lookup(Operation *op) {
  auto *context = op->getContext();
  auto attrId = StringAttr::get(context, "hal.executable.target");
  while (op) {
    // Take directly from the enclosing variant.
    if (auto variantOp = llvm::dyn_cast<IREE::HAL::ExecutableVariantOp>(op)) {
      return variantOp.getTarget();
    }
    // Use an override if specified.
    auto attr = op->getAttrOfType<IREE::HAL::ExecutableTargetAttr>(attrId);
    if (attr)
      return attr;
    // Continue walk.
    op = op->getParentOp();
  }
  // No target found during walk. No default to provide so fail and let the
  // caller decide what to do (assert/fallback/etc).
  return nullptr;
}

//===----------------------------------------------------------------------===//
// #hal.executable.object<*>
//===----------------------------------------------------------------------===//

// static
Attribute ExecutableObjectAttr::parse(AsmParser &p, Type type) {
  NamedAttrList dict;
  // `<{` dict `}>`
  if (failed(p.parseLess()) || failed(p.parseOptionalAttrDict(dict)) ||
      failed(p.parseGreater())) {
    return {};
  }
  auto pathAttr = llvm::dyn_cast_if_present<StringAttr>(dict.get("path"));
  auto dataAttr =
      llvm::dyn_cast_if_present<DenseIntElementsAttr>(dict.get("data"));
  return get(p.getContext(), pathAttr, dataAttr);
}

void ExecutableObjectAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<{";
  if (auto pathAttr = getPath()) {
    os << "path = ";
    p.printAttribute(getPath());
  }
  if (auto dataAttr = getData()) {
    os << ", data = ";
    p.printAttribute(getData());
  }
  os << "}>";
}

// static
void ExecutableObjectAttr::filterObjects(
    ArrayAttr objectAttrs, ArrayRef<StringRef> extensions,
    SmallVectorImpl<IREE::HAL::ExecutableObjectAttr> &filteredAttrs) {
  if (!objectAttrs)
    return;
  for (auto objectAttr :
       objectAttrs.getAsRange<IREE::HAL::ExecutableObjectAttr>()) {
    auto path = objectAttr.getPath();
    auto ext = llvm::sys::path::extension(path);
    if (llvm::is_contained(extensions, ext)) {
      filteredAttrs.push_back(objectAttr);
    }
  }
}

// Tries to find |filePath| on disk either at its absolute path or joined with
// any of the specified |searchPaths| in order.
// Returns the absolute file path when found or a failure if there are no hits.
static FailureOr<std::string>
findFileInPaths(StringRef filePath, ArrayRef<std::string> searchPaths) {
  // First try to see if it's an absolute path - we don't want to perform any
  // additional processing on top of that.
  if (llvm::sys::path::is_absolute(filePath)) {
    if (llvm::sys::fs::exists(filePath))
      return filePath.str();
    return failure();
  }

  // Try a relative lookup from the current working directory.
  if (llvm::sys::fs::exists(filePath))
    return filePath.str();

  // Search each path in turn for a file that exists.
  // It doesn't mean we can open it but we'll get a better error out of the
  // actual open attempt than what we could produce here.
  for (auto searchPath : searchPaths) {
    SmallVector<char> tryPath{searchPath.begin(), searchPath.end()};
    llvm::sys::path::append(tryPath, filePath);
    if (llvm::sys::fs::exists(Twine(tryPath)))
      return Twine(tryPath).str();
  }

  // Not found in either the user-specified absolute path, cwd, or the search
  // paths.
  return failure();
}

static llvm::cl::list<std::string> clExecutableObjectSearchPath(
    "iree-hal-executable-object-search-path",
    llvm::cl::desc("Additional search paths for resolving "
                   "#hal.executable.object file references."),
    llvm::cl::ZeroOrMore);

FailureOr<std::string> ExecutableObjectAttr::getAbsolutePath() {
  auto pathAttr = getPath();
  if (!pathAttr)
    return failure(); // not a file reference
  return findFileInPaths(pathAttr.getValue(), clExecutableObjectSearchPath);
}

std::optional<std::string> ExecutableObjectAttr::loadData() {
  if (auto dataAttr = getData()) {
    // This is shady but so is using this feature.
    // TODO(benvanik): figure out a way to limit the attribute to signless int8.
    // We could share the attribute -> byte array code with the VM constant
    // serialization if we wanted.
    auto rawData = dataAttr.getRawData();
    return std::string(rawData.data(), rawData.size());
  } else if (auto pathAttr = getPath()) {
    // Search for file and try to load it if found.
    auto filePath =
        findFileInPaths(pathAttr.getValue(), clExecutableObjectSearchPath);
    if (failed(filePath)) {
      llvm::errs()
          << "ERROR: referenced object file not found on any path; use "
             "--iree-hal-executable-object-search-path= to add search paths: "
          << *this << "\n";
      return std::nullopt;
    }
    auto file = llvm::MemoryBuffer::getFile(*filePath);
    if (!file)
      return std::nullopt;
    return std::string((*file)->getBuffer());
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// #hal.executable.objects<*>
//===----------------------------------------------------------------------===//

// static
LogicalResult ExecutableObjectsAttr::verify(
    function_ref<mlir::InFlightDiagnostic()> emitError, ArrayAttr targetsAttr,
    ArrayAttr targetObjectsAttr) {
  if (targetsAttr.size() != targetObjectsAttr.size()) {
    return emitError() << "targets and objects must be 1:1";
  }
  for (auto targetAttr : targetsAttr) {
    if (!llvm::isa<IREE::HAL::ExecutableTargetAttr>(targetAttr)) {
      return emitError()
             << "target keys must be #hal.executable.target attributes";
    }
  }
  for (auto objectsAttr : targetObjectsAttr) {
    auto objectsArrayAttr = llvm::dyn_cast<ArrayAttr>(objectsAttr);
    if (!objectsArrayAttr) {
      return emitError() << "target objects must be an array of "
                            "#hal.executable.object attributes";
    }
  }
  return success();
}

// static
Attribute ExecutableObjectsAttr::parse(AsmParser &p, Type type) {
  // `<{` target = [objects, ...], ... `}>`
  SmallVector<Attribute> targetAttrs;
  SmallVector<Attribute> objectsAttrs;
  if (failed(p.parseLess()))
    return {};
  if (succeeded(p.parseLBrace()) && !succeeded(p.parseOptionalRBrace())) {
    do {
      Attribute targetAttr;
      ArrayAttr objectsAttr;
      if (failed(p.parseAttribute(targetAttr)) || failed(p.parseEqual()) ||
          failed(p.parseAttribute(objectsAttr))) {
        return {};
      }
      targetAttrs.push_back(targetAttr);
      objectsAttrs.push_back(objectsAttr);
    } while (succeeded(p.parseOptionalComma()));
    if (failed(p.parseRBrace()))
      return {};
  }
  if (failed(p.parseGreater()))
    return {};
  return get(p.getContext(), ArrayAttr::get(p.getContext(), targetAttrs),
             ArrayAttr::get(p.getContext(), objectsAttrs));
}

void ExecutableObjectsAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<{";
  llvm::interleaveComma(llvm::zip_equal(getTargets(), getTargetObjects()), os,
                        [&](std::tuple<Attribute, Attribute> keyValue) {
                          p.printAttribute(std::get<0>(keyValue));
                          os << " = ";
                          p.printAttributeWithoutType(std::get<1>(keyValue));
                        });
  os << "}>";
}

std::optional<ArrayAttr> ExecutableObjectsAttr::getApplicableObjects(
    IREE::HAL::ExecutableTargetAttr specificTargetAttr) {
  SmallVector<Attribute> allObjectAttrs;
  for (auto [targetAttr, objectsAttr] :
       llvm::zip_equal(getTargets(), getTargetObjects())) {
    auto genericTargetAttr =
        llvm::cast<IREE::HAL::ExecutableTargetAttr>(targetAttr);
    if (genericTargetAttr.isGenericOf(specificTargetAttr)) {
      auto objectsArrayAttr = llvm::cast<ArrayAttr>(objectsAttr);
      allObjectAttrs.append(objectsArrayAttr.begin(), objectsArrayAttr.end());
    }
  }
  if (allObjectAttrs.empty())
    return std::nullopt;
  return ArrayAttr::get(specificTargetAttr.getContext(), allObjectAttrs);
}

//===----------------------------------------------------------------------===//
// #hal.device.alias<*>
//===----------------------------------------------------------------------===//

// static
DeviceAliasAttr DeviceAliasAttr::get(MLIRContext *context, StringRef deviceID) {
  return get(context, IREE::HAL::DeviceType::get(context),
             StringAttr::get(context, deviceID), std::nullopt,
             DictionaryAttr::get(context));
}

//===----------------------------------------------------------------------===//
// #hal.device.target<*>
//===----------------------------------------------------------------------===//

// static
DeviceTargetAttr DeviceTargetAttr::get(MLIRContext *context,
                                       StringRef deviceID) {
  // TODO(benvanik): query default configuration from the target backend.
  return get(context, StringAttr::get(context, deviceID),
             DictionaryAttr::get(context), {});
}

// static
Attribute DeviceTargetAttr::parse(AsmParser &p, Type type) {
  StringAttr deviceIDAttr;
  DictionaryAttr configAttr;
  SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
  // `<"device-id"`
  if (failed(p.parseLess()) || failed(p.parseAttribute(deviceIDAttr))) {
    return {};
  }
  // `, `
  if (succeeded(p.parseOptionalComma())) {
    if (succeeded(p.parseOptionalLSquare())) {
      // `[targets, ...]` (optional)
      do {
        IREE::HAL::ExecutableTargetAttr executableTargetAttr;
        if (failed(p.parseAttribute(executableTargetAttr))) {
          return {};
        }
        executableTargetAttrs.push_back(executableTargetAttr);
      } while (succeeded(p.parseOptionalComma()));
      if (failed(p.parseRSquare())) {
        return {};
      }
    } else {
      // `{config dict}` (optional)
      if (failed(p.parseAttribute(configAttr))) {
        return {};
      }
      // `, [targets, ...]` (optional)
      if (succeeded(p.parseOptionalComma())) {
        if (failed(p.parseLSquare())) {
          return {};
        }
        do {
          IREE::HAL::ExecutableTargetAttr executableTargetAttr;
          if (failed(p.parseAttribute(executableTargetAttr))) {
            return {};
          }
          executableTargetAttrs.push_back(executableTargetAttr);
        } while (succeeded(p.parseOptionalComma()));
        if (failed(p.parseRSquare())) {
          return {};
        }
      }
    }
  }
  // `>`
  if (failed(p.parseGreater())) {
    return {};
  }
  return get(p.getContext(), deviceIDAttr, configAttr, executableTargetAttrs);
}

void DeviceTargetAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  p.printAttribute(getDeviceID());
  auto configAttr = getConfiguration();
  if (configAttr && !configAttr.empty()) {
    os << ", ";
    p.printAttribute(configAttr);
  }
  auto executableTargetAttrs = getExecutableTargets();
  if (!executableTargetAttrs.empty()) {
    os << ", [";
    llvm::interleaveComma(executableTargetAttrs, os,
                          [&](auto executableTargetAttr) {
                            p.printAttribute(executableTargetAttr);
                          });
    os << "]";
  }
  os << ">";
}

std::string DeviceTargetAttr::getSymbolNameFragment() {
  std::string name = getDeviceID().getValue().lower();
  if (auto ordinalAttr =
          dyn_cast_if_present<IntegerAttr>(getConfigurationAttr("ordinal"))) {
    name += "_";
    name += std::to_string(ordinalAttr.getInt());
    name += "_"; // can't have trailing numbers
  }
  return sanitizeSymbolName(name);
}

bool DeviceTargetAttr::hasConfigurationAttr(StringRef name) {
  auto configAttr = getConfiguration();
  return configAttr && configAttr.get(name);
}

Attribute DeviceTargetAttr::getConfigurationAttr(StringRef name) {
  if (auto configAttr = getConfiguration()) {
    return configAttr.get(name);
  }
  return {};
}

void DeviceTargetAttr::getExecutableTargets(
    SetVector<IREE::HAL::ExecutableTargetAttr> &resultAttrs) {
  for (auto attr : getExecutableTargets()) {
    resultAttrs.insert(attr);
  }
}

void IREE::HAL::DeviceTargetAttr::printStatusDescription(
    llvm::raw_ostream &os) const {
  mlir::cast<Attribute>(this)->print(os, /*elideType=*/true);
}

// Produces a while-loop that enumerates each device available and tries to
// match it against the target information. SCF is... not very wieldy, but this
// is effectively:
// ```
//   %device_count = hal.devices.count : index
//   %result:3 = scf.while(%i = 0, %match_ordinal = 0, %device = null) {
//     %is_null = util.cmp.eq %device, null : !hal.device
//     %in_bounds = arith.cmpi slt %i, %device_count : index
//     %continue_while = arith.andi %is_null, %in_bounds : i1
//     scf.condition(%continue_while) %i, %match_ordinal %device
//         : index, index, !hal.device
//   } do {
//     %device_i = hal.devices.get %i : !hal.device
//     %device_match = <<buildDeviceMatch>>(%device_i)
//     %ordinal_match = arith.cmpi eq %match_ordinal, %device_ordinal : index
//     %is_match = arith.andi %device_match, %ordinal_match : i1
//     %try_device = arith.select %is_match, %device_i, null : !hal.device
//     %next_i = arith.addi %i, %c1 : index
//     %match_adv = arith.select %device_match, %c1, %c0 : index
//     %next_match_ordinal = arith.addi %match_ordinal, %match_adv : index
//     scf.yield %next_i, %next_match_ordinal, %try_device
//         : index, index !hal.device
//   }
// ```
// Upon completion %result#1 contains the device (or null).
// If the target had an ordinal specified we skip matches until a match with the
// specified ordinal is reached.
Value IREE::HAL::DeviceTargetAttr::buildDeviceEnumeration(
    Location loc, IREE::HAL::BuildDeviceTargetMatchFn buildDeviceTargetMatch,
    OpBuilder &builder) const {
  // Device configuration can control selection beyond just the match
  // expression.
  auto configAttr = getConfiguration();
  IntegerAttr deviceOrdinalAttr =
      configAttr ? configAttr.getAs<IntegerAttr>("ordinal") : IntegerAttr{};

  // Defers to the target backend to build the device match or does a simple
  // fallback for unregistered backends (usually for testing, but may be used
  // as a way to bypass validation for out-of-tree experiments).
  auto buildDeviceMatch = [&](Location loc, Value device,
                              OpBuilder &builder) -> Value {
    // Ask the target backend to build the match expression. It may opt to
    // let the default handling take care of things.
    Value match = buildDeviceTargetMatch(loc, device, *this, builder);
    if (match)
      return match;
    return IREE::HAL::DeviceTargetAttr::buildDeviceIDAndExecutableFormatsMatch(
        loc, device, getDeviceID(), getExecutableTargets(), builder);
  };

  // Enumerate all devices and match the first one found (if any).
  Type indexType = builder.getIndexType();
  Type deviceType = builder.getType<IREE::HAL::DeviceType>();
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value nullDevice = builder.create<IREE::Util::NullOp>(loc, deviceType);
  Value deviceOrdinal = deviceOrdinalAttr
                            ? builder.create<arith::ConstantIndexOp>(
                                  loc, deviceOrdinalAttr.getInt())
                            : c0;
  Value deviceCount = builder.create<IREE::HAL::DevicesCountOp>(loc, indexType);
  auto whileOp = builder.create<scf::WhileOp>(
      loc,
      TypeRange{
          /*i=*/indexType,
          /*match_ordinal=*/indexType,
          /*device=*/deviceType,
      },
      ValueRange{
          /*i=*/c0,
          /*match_ordinal=*/c0,
          /*device=*/nullDevice,
      },
      [&](OpBuilder &beforeBuilder, Location loc, ValueRange operands) {
        Value isNull = beforeBuilder.create<IREE::Util::CmpEQOp>(
            loc, operands[/*device=*/2], nullDevice);
        Value inBounds = beforeBuilder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, operands[/*i=*/0], deviceCount);
        Value continueWhile =
            beforeBuilder.create<arith::AndIOp>(loc, isNull, inBounds);
        beforeBuilder.create<scf::ConditionOp>(loc, continueWhile, operands);
      },
      [&](OpBuilder &afterBuilder, Location loc, ValueRange operands) {
        // Check whether the device is a match.
        Value device = afterBuilder.create<IREE::HAL::DevicesGetOp>(
            loc, deviceType, operands[/*i=*/0]);
        Value isDeviceMatch = buildDeviceMatch(loc, device, afterBuilder);

        // Check whether whether this matching device ordinal is the requested
        // ordinal out of all matching devices.
        Value isOrdinalMatch = afterBuilder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, operands[/*match_ordinal=*/1],
            deviceOrdinal);
        Value nextMatchOrdinal = afterBuilder.create<arith::AddIOp>(
            loc, operands[/*match_ordinal=*/1],
            afterBuilder.create<arith::SelectOp>(loc, isDeviceMatch, c1, c0));

        // Break if the device and ordinal match, otherwise continue with null.
        Value isMatch = afterBuilder.create<arith::AndIOp>(loc, isDeviceMatch,
                                                           isOrdinalMatch);
        Value tryDevice = afterBuilder.create<arith::SelectOp>(
            loc, isMatch, device, nullDevice);

        Value nextI =
            afterBuilder.create<arith::AddIOp>(loc, operands[/*i=*/0], c1);
        afterBuilder.create<scf::YieldOp>(
            loc, ValueRange{
                     /*i=*/nextI,
                     /*match_ordinal=*/nextMatchOrdinal,
                     /*device=*/tryDevice,
                 });
      });
  return whileOp.getResult(/*device=*/2);
}

// static
Value DeviceTargetAttr::buildDeviceIDAndExecutableFormatsMatch(
    Location loc, Value device, StringRef deviceIDPattern,
    ArrayRef<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs,
    OpBuilder &builder) {
  // Match first on the device ID, as that's the top-level filter.
  Value idMatch = IREE::HAL::DeviceQueryOp::createI1(
      loc, device, "hal.device.id", deviceIDPattern, builder);

  // If there are executable formats defined we should check at least one of
  // them is supported.
  if (executableTargetAttrs.empty()) {
    return idMatch; // just device ID
  } else {
    auto ifOp = builder.create<scf::IfOp>(loc, builder.getI1Type(), idMatch,
                                          true, true);
    auto thenBuilder = ifOp.getThenBodyBuilder();
    Value anyFormatMatch = buildExecutableFormatMatch(
        loc, device, executableTargetAttrs, thenBuilder);
    thenBuilder.create<scf::YieldOp>(loc, anyFormatMatch);
    auto elseBuilder = ifOp.getElseBodyBuilder();
    Value falseValue = elseBuilder.create<arith::ConstantIntOp>(loc, 0, 1);
    elseBuilder.create<scf::YieldOp>(loc, falseValue);
    return ifOp.getResult(0);
  }
}

// static
Value DeviceTargetAttr::buildExecutableFormatMatch(
    Location loc, Value device,
    ArrayRef<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs,
    OpBuilder &builder) {
  if (executableTargetAttrs.empty())
    return builder.create<arith::ConstantIntOp>(loc, 1, 1);
  Value anyFormatMatch;
  for (auto executableTargetAttr : executableTargetAttrs) {
    Value formatMatch = IREE::HAL::DeviceQueryOp::createI1(
        loc, device, "hal.executable.format",
        executableTargetAttr.getFormat().getValue(), builder);
    if (!anyFormatMatch) {
      anyFormatMatch = formatMatch;
    } else {
      anyFormatMatch =
          builder.create<arith::OrIOp>(loc, anyFormatMatch, formatMatch);
    }
  }
  return anyFormatMatch;
}

//===----------------------------------------------------------------------===//
// #hal.device.ordinal<*>
//===----------------------------------------------------------------------===//

void IREE::HAL::DeviceOrdinalAttr::printStatusDescription(
    llvm::raw_ostream &os) const {
  mlir::cast<Attribute>(this)->print(os, /*elideType=*/true);
}

Value IREE::HAL::DeviceOrdinalAttr::buildDeviceEnumeration(
    Location loc, IREE::HAL::BuildDeviceTargetMatchFn buildDeviceTargetMatch,
    OpBuilder &builder) const {
  return builder.create<IREE::HAL::DevicesGetOp>(
      loc, getType(),
      builder.create<arith::ConstantIndexOp>(loc, getOrdinal()));
}

//===----------------------------------------------------------------------===//
// #hal.device.fallback<*>
//===----------------------------------------------------------------------===//

void IREE::HAL::DeviceFallbackAttr::printStatusDescription(
    llvm::raw_ostream &os) const {
  mlir::cast<Attribute>(this)->print(os, /*elideType=*/true);
}

Value IREE::HAL::DeviceFallbackAttr::buildDeviceEnumeration(
    Location loc, IREE::HAL::BuildDeviceTargetMatchFn buildDeviceTargetMatch,
    OpBuilder &builder) const {
  // TODO(benvanik): hal.device.cast if needed - may need to look up the global
  // to do it as we don't encode what the device is here in a way that is
  // guaranteed to be consistent.
  return builder.create<IREE::Util::GlobalLoadOp>(loc, getType(),
                                                  getName().getValue());
}

//===----------------------------------------------------------------------===//
// #hal.device.select<*>
//===----------------------------------------------------------------------===//

// static
DeviceSelectAttr DeviceSelectAttr::get(MLIRContext *context,
                                       ArrayRef<Attribute> values) {
  return DeviceSelectAttr::get(context, IREE::HAL::DeviceType::get(context),
                               ArrayAttr::get(context, values));
}

// static
LogicalResult
DeviceSelectAttr::verify(function_ref<mlir::InFlightDiagnostic()> emitError,
                         Type type, ArrayAttr devicesAttr) {
  if (devicesAttr.empty())
    return emitError() << "must have at least one device to select";
  for (auto deviceAttr : devicesAttr) {
    if (!mlir::isa<IREE::HAL::DeviceAliasAttr>(deviceAttr) &&
        !mlir::isa<IREE::HAL::DeviceInitializationAttrInterface>(deviceAttr)) {
      return emitError() << "can only select between #hal.device.alias, "
                            "#hal.device.target, #hal.device.ordinal, "
                            "#hal.device.fallback, or other device "
                            "initialization attributes";
    }
  }
  // TODO(benvanik): when !hal.device is parameterized we should check that the
  // type is compatible with the entries.
  return success();
}

void IREE::HAL::DeviceSelectAttr::printStatusDescription(
    llvm::raw_ostream &os) const {
  // TODO(benvanik): print something easier to read (newline per device, etc).
  mlir::cast<Attribute>(this)->print(os, /*elideType=*/true);
}

// Builds a recursive nest of try-else blocks for each device specified.
Value IREE::HAL::DeviceSelectAttr::buildDeviceEnumeration(
    Location loc, IREE::HAL::BuildDeviceTargetMatchFn buildDeviceTargetMatch,
    OpBuilder &builder) const {
  Type deviceType = builder.getType<IREE::HAL::DeviceType>();
  Value nullDevice = builder.create<IREE::Util::NullOp>(loc, deviceType);
  std::function<Value(ArrayRef<IREE::HAL::DeviceInitializationAttrInterface>,
                      OpBuilder &)>
      buildTry;
  buildTry =
      [&](ArrayRef<IREE::HAL::DeviceInitializationAttrInterface> deviceAttrs,
          OpBuilder &tryBuilder) -> Value {
    auto deviceAttr = deviceAttrs.front();
    Value tryDevice = deviceAttr.buildDeviceEnumeration(
        loc, buildDeviceTargetMatch, tryBuilder);
    if (deviceAttrs.size() == 1)
      return tryDevice; // termination case
    Value isNull =
        tryBuilder.create<IREE::Util::CmpEQOp>(loc, tryDevice, nullDevice);
    auto ifOp =
        tryBuilder.create<scf::IfOp>(loc, deviceType, isNull, true, true);
    auto thenBuilder = ifOp.getThenBodyBuilder();
    Value tryChainDevice = buildTry(deviceAttrs.drop_front(1), thenBuilder);
    thenBuilder.create<scf::YieldOp>(loc, tryChainDevice);
    auto elseBuilder = ifOp.getElseBodyBuilder();
    elseBuilder.create<scf::YieldOp>(loc, tryDevice);
    return ifOp.getResult(0);
  };
  SmallVector<IREE::HAL::DeviceInitializationAttrInterface> deviceAttrs(
      getDevices().getAsRange<IREE::HAL::DeviceInitializationAttrInterface>());
  return buildTry(deviceAttrs, builder);
}

//===----------------------------------------------------------------------===//
// #hal.device.affinity<*>
//===----------------------------------------------------------------------===//

// static
Attribute DeviceAffinityAttr::parse(AsmParser &p, Type type) {
  // `<@device`
  StringAttr deviceName;
  int64_t queueMask = -1;
  if (failed(p.parseLess()) || failed(p.parseSymbolName(deviceName)))
    return {};
  if (succeeded(p.parseOptionalComma())) {
    // `[`queue_bit[, ...] `]`
    queueMask = 0;
    if (failed(p.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
          int64_t i = 0;
          if (failed(p.parseInteger(i)))
            return failure();
          queueMask |= 1ll << i;
          return success();
        }))) {
      return {};
    }
  }
  // `>`
  if (failed(p.parseGreater()))
    return {};
  return get(p.getContext(), FlatSymbolRefAttr::get(deviceName), queueMask);
}

void DeviceAffinityAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  os << getDevice();
  int64_t queueMask = getQueueMask();
  if (queueMask != -1) {
    os << ", [";
    for (int i = 0, j = 0; i < sizeof(queueMask) * 8; ++i) {
      if (queueMask & (1ll << i)) {
        if (j++ > 0)
          os << ", ";
        os << i;
      }
    }
    os << "]";
  }
  os << ">";
}

bool DeviceAffinityAttr::isExecutableWith(
    IREE::Stream::AffinityAttr other) const {
  if (!other)
    return true;
  // Only compatible with the same exact devices today. We could support a
  // peering model to allow operations to move across devices in a peered set
  // but that may be best done at higher levels and avoided once we get to the
  // "are these the same device" stage.
  auto otherAffinityAttr = llvm::dyn_cast_if_present<DeviceAffinityAttr>(other);
  if (!otherAffinityAttr || getDevice() != otherAffinityAttr.getDevice())
    return false;
  // If this affinity is a subset of the target affinity then it can execute
  // with it.
  if ((getQueueMask() & otherAffinityAttr.getQueueMask()) == getQueueMask())
    return true;
  // Otherwise not compatible.
  return false;
}

IREE::Stream::AffinityAttr
DeviceAffinityAttr::joinOR(IREE::Stream::AffinityAttr other) const {
  if (!other)
    return *this;
  if (!IREE::Stream::AffinityAttr::canExecuteTogether(*this, other)) {
    return nullptr;
  }
  auto otherAffinityAttr = llvm::dyn_cast_if_present<DeviceAffinityAttr>(other);
  return DeviceAffinityAttr::get(getContext(), getDevice(),
                                 getQueueMask() |
                                     otherAffinityAttr.getQueueMask());
}

IREE::Stream::AffinityAttr
DeviceAffinityAttr::joinAND(IREE::Stream::AffinityAttr other) const {
  if (!other)
    return *this;
  if (!IREE::Stream::AffinityAttr::canExecuteTogether(*this, other)) {
    return nullptr;
  }
  auto otherAffinityAttr = llvm::dyn_cast_if_present<DeviceAffinityAttr>(other);
  return DeviceAffinityAttr::get(getContext(), getDevice(),
                                 getQueueMask() &
                                     otherAffinityAttr.getQueueMask());
}

bool DeviceAffinityAttr::isLegalToInline(Operation *inlineSite,
                                         Operation *inlinable) const {
  // Look up the affinity of the inlining target site and only allow inlining if
  // it matches exactly. We could make a decision as to whether we allow
  // inlining when queues are subsets (so if the target site allows any queue
  // and the inlinable allows queue 2 then allow, etc). In the future we may
  // want to allow util.scope restrictions within the inline target to keep
  // queue specification tighter but today most queue masks are wildcarded
  // anyway.
  auto targetAffinityAttr = IREE::Stream::AffinityAttr::lookup(inlineSite);
  return *this == targetAffinityAttr;
}

//===----------------------------------------------------------------------===//
// #hal.device.promise<*>
//===----------------------------------------------------------------------===//

// static
Attribute DevicePromiseAttr::parse(AsmParser &p, Type type) {
  // `<@device`
  StringAttr deviceName;
  int64_t queueMask = -1;
  if (failed(p.parseLess()) || failed(p.parseSymbolName(deviceName)))
    return {};
  if (succeeded(p.parseOptionalComma())) {
    // `[`queue_bit[, ...] `]`
    queueMask = 0;
    if (failed(p.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
          int64_t i = 0;
          if (failed(p.parseInteger(i)))
            return failure();
          queueMask |= 1ll << i;
          return success();
        }))) {
      return {};
    }
  }
  // `>`
  if (failed(p.parseGreater()))
    return {};
  return get(p.getContext(), deviceName, queueMask);
}

void DevicePromiseAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<@";
  os << getDevice().getValue();
  int64_t queueMask = getQueueMask();
  if (queueMask != -1) {
    os << ", [";
    for (int i = 0, j = 0; i < sizeof(queueMask) * 8; ++i) {
      if (queueMask & (1ll << i)) {
        if (j++ > 0)
          os << ", ";
        os << i;
      }
    }
    os << "]";
  }
  os << ">";
}

bool DevicePromiseAttr::isExecutableWith(
    IREE::Stream::AffinityAttr other) const {
  if (!other)
    return true;
  // Only compatible with the same exact devices today. We could support a
  // peering model to allow operations to move across devices in a peered set
  // but that may be best done at higher levels and avoided once we get to the
  // "are these the same device" stage.
  auto otherPromiseAttr = llvm::dyn_cast_if_present<DevicePromiseAttr>(other);
  if (!otherPromiseAttr || getDevice() != otherPromiseAttr.getDevice())
    return false;
  // If this affinity is a subset of the target affinity then it can execute
  // with it.
  if ((getQueueMask() & otherPromiseAttr.getQueueMask()) == getQueueMask())
    return true;
  // Otherwise not compatible.
  return false;
}

IREE::Stream::AffinityAttr
DevicePromiseAttr::joinOR(IREE::Stream::AffinityAttr other) const {
  if (!other)
    return *this;
  if (!IREE::Stream::AffinityAttr::canExecuteTogether(*this, other)) {
    return nullptr;
  }
  auto otherPromiseAttr = llvm::dyn_cast_if_present<DevicePromiseAttr>(other);
  return DevicePromiseAttr::get(getContext(), getDevice(),
                                getQueueMask() |
                                    otherPromiseAttr.getQueueMask());
}

IREE::Stream::AffinityAttr
DevicePromiseAttr::joinAND(IREE::Stream::AffinityAttr other) const {
  if (!other)
    return *this;
  if (!IREE::Stream::AffinityAttr::canExecuteTogether(*this, other)) {
    return nullptr;
  }
  auto otherPromiseAttr = llvm::dyn_cast_if_present<DevicePromiseAttr>(other);
  return DevicePromiseAttr::get(getContext(), getDevice(),
                                getQueueMask() &
                                    otherPromiseAttr.getQueueMask());
}

bool DevicePromiseAttr::isLegalToInline(Operation *inlineSite,
                                        Operation *inlinable) const {
  // Look up the affinity of the inlining target site and only allow inlining if
  // it matches exactly. We could make a decision as to whether we allow
  // inlining when queues are subsets (so if the target site allows any queue
  // and the inlinable allows queue 2 then allow, etc). In the future we may
  // want to allow util.scope restrictions within the inline target to keep
  // queue specification tighter but today most queue masks are wildcarded
  // anyway.
  auto targetAffinityAttr = IREE::Stream::AffinityAttr::lookup(inlineSite);
  return *this == targetAffinityAttr;
}

//===----------------------------------------------------------------------===//
// IREE::HAL::HALDialect
//===----------------------------------------------------------------------===//

// At the end so it can use functions above:
#include "iree/compiler/Dialect/HAL/IR/HALAttrInterfaces.cpp.inc"

void HALDialect::registerAttributes() {
  // Register command line flags:
  (void)clExecutableObjectSearchPath;

  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Dialect/HAL/IR/HALAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

Attribute HALDialect::parseAttribute(DialectAsmParser &parser,
                                     Type type) const {
  StringRef mnemonic;
  Attribute genAttr;
  OptionalParseResult parseResult =
      generatedAttributeParser(parser, &mnemonic, type, genAttr);
  if (parseResult.has_value())
    return genAttr;
  parser.emitError(parser.getNameLoc())
      << "unknown HAL attribute: " << mnemonic;
  return {};
}

void HALDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
  TypeSwitch<Attribute>(attr).Default([&](Attribute) {
    if (failed(generatedAttributePrinter(attr, p))) {
      assert(false && "unhandled HAL attribute kind");
    }
  });
}

} // namespace mlir::iree_compiler::IREE::HAL
