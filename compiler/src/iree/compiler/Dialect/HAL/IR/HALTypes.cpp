// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/HAL/IR/HALAttrs.cpp.inc" // IWYU pragma: keep
#include "iree/compiler/Dialect/HAL/IR/HALEnums.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::HAL {

//===----------------------------------------------------------------------===//
// Enum utilities
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
// Element types
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
// Alignment
//===----------------------------------------------------------------------===//

llvm::MaybeAlign commonAlignment(llvm::MaybeAlign lhs, llvm::MaybeAlign rhs) {
  if (!lhs.has_value() || !rhs.has_value())
    return std::nullopt;
  return llvm::MaybeAlign(
      llvm::MinAlign(lhs.value().value(), rhs.value().value()));
}

// TODO(benvanik): share with align op folder and analysis.
// May need an interface for querying the alignment from ops that can carry it.
std::optional<uint64_t> lookupOffsetOrAlignment(Value value) {
  APInt constantValue;
  if (matchPattern(value, m_ConstantInt(&constantValue))) {
    // Value is constant and we can just treat that as if it were an alignment.
    return constantValue.getZExtValue();
  }

  auto op = value.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (auto alignmentAttr = op->getAttrOfType<IntegerAttr>("stream.alignment")) {
    // The op has an alignment tagged on it we can use directly.
    return alignmentAttr.getValue().getZExtValue();
  }

  // TODO(benvanik): walk other pass-through. These are the most common in our
  // programs today.
  if (auto loadOp = dyn_cast<IREE::HAL::InterfaceConstantLoadOp>(op)) {
    // Push constants have an optional value alignment.
    auto alignment = loadOp.getAlignment();
    if (alignment.has_value()) {
      return alignment.value().getZExtValue();
    }
  } else if (auto castOp = dyn_cast<arith::IndexCastUIOp>(op)) {
    return lookupOffsetOrAlignment(castOp.getOperand());
  }

  // TODO(benvanik): more searching using util.align and other ops.
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Object types
//===----------------------------------------------------------------------===//

Value BufferType::inferSizeFromValue(Location loc, Value value,
                                     OpBuilder &builder) const {
  return builder.createOrFold<BufferLengthOp>(loc, builder.getIndexType(),
                                              value);
}

Value BufferViewType::inferSizeFromValue(Location loc, Value value,
                                         OpBuilder &builder) const {
  return builder.createOrFold<BufferLengthOp>(
      loc, builder.getIndexType(),
      builder.createOrFold<BufferViewBufferOp>(
          loc, builder.getType<IREE::HAL::BufferType>(), value));
}

// static
Value DeviceType::resolveAny(Location loc, OpBuilder &builder) {
  Value deviceIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
  return builder.create<IREE::HAL::DevicesGetOp>(
      loc, builder.getType<IREE::HAL::DeviceType>(), deviceIndex);
}

//===----------------------------------------------------------------------===//
// #hal.device.target
//===----------------------------------------------------------------------===//

// static
DeviceTargetAttr DeviceTargetAttr::get(MLIRContext *context,
                                       StringRef deviceID) {
  // TODO(benvanik): query default configuration from the target backend.
  return get(context, StringAttr::get(context, deviceID),
             DictionaryAttr::get(context));
}

// static
Attribute DeviceTargetAttr::parse(AsmParser &p, Type type) {
  StringAttr deviceIDAttr;
  DictionaryAttr configAttr;
  // `<"device-id"`
  if (failed(p.parseLess()) || failed(p.parseAttribute(deviceIDAttr))) {
    return {};
  }
  // `, {config}`
  if (succeeded(p.parseOptionalComma()) &&
      failed(p.parseAttribute(configAttr))) {
    return {};
  }
  // `>`
  if (failed(p.parseGreater())) {
    return {};
  }
  return get(p.getContext(), deviceIDAttr, configAttr);
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
  os << ">";
}

std::string DeviceTargetAttr::getSymbolNameFragment() {
  return sanitizeSymbolName(getDeviceID().getValue().lower());
}

bool DeviceTargetAttr::hasConfigurationAttr(StringRef name) {
  auto configAttr = getConfiguration();
  return configAttr && configAttr.get(name);
}

SmallVector<ExecutableTargetAttr, 4> DeviceTargetAttr::getExecutableTargets() {
  SmallVector<ExecutableTargetAttr, 4> resultAttrs;
  auto configAttr = getConfiguration();
  if (configAttr) {
    auto targetsAttr = configAttr.getAs<ArrayAttr>("executable_targets");
    if (targetsAttr) {
      for (auto attr : targetsAttr.getValue()) {
        resultAttrs.push_back(llvm::dyn_cast<ExecutableTargetAttr>(attr));
      }
    }
  }
  return resultAttrs;
}

// static
SmallVector<IREE::HAL::DeviceTargetAttr, 4>
DeviceTargetAttr::lookup(Operation *op) {
  auto attrId = mlir::StringAttr::get(op->getContext(), "hal.device.targets");
  while (op) {
    auto targetsAttr = op->getAttrOfType<ArrayAttr>(attrId);
    if (targetsAttr) {
      SmallVector<IREE::HAL::DeviceTargetAttr, 4> result;
      for (auto targetAttr : targetsAttr) {
        result.push_back(llvm::cast<IREE::HAL::DeviceTargetAttr>(targetAttr));
      }
      return result;
    }
    op = op->getParentOp();
  }
  return {}; // No devices found; let caller decide what to do.
}

// Returns a set of all configuration attributes from all device targets with
// a configuration set. Targets with no configuration set are ignored.
static SmallVector<DictionaryAttr> lookupOptionalConfigAttrs(Operation *op) {
  auto targetAttrs = IREE::HAL::DeviceTargetAttr::lookup(op);
  if (targetAttrs.empty())
    return {};
  SmallVector<DictionaryAttr> configAttrs;
  for (auto targetAttr : targetAttrs) {
    auto configAttr = targetAttr.getConfiguration();
    if (configAttr)
      configAttrs.push_back(configAttr);
  }
  return configAttrs;
}

// Returns a set of all configuration attributes from all device targets.
// Returns nullopt if any target is missing a configuration attribute.
static std::optional<SmallVector<DictionaryAttr>>
lookupRequiredConfigAttrs(Operation *op) {
  auto targetAttrs = IREE::HAL::DeviceTargetAttr::lookup(op);
  if (targetAttrs.empty())
    return std::nullopt;
  SmallVector<DictionaryAttr> configAttrs;
  for (auto targetAttr : targetAttrs) {
    auto configAttr = targetAttr.getConfiguration();
    if (!configAttr)
      return std::nullopt;
    configAttrs.push_back(configAttr);
  }
  return configAttrs;
}

template <typename AttrT>
static std::optional<typename AttrT::ValueType> joinConfigAttrs(
    ArrayRef<DictionaryAttr> configAttrs, StringRef name,
    std::function<typename AttrT::ValueType(typename AttrT::ValueType,
                                            typename AttrT::ValueType)>
        join) {
  if (configAttrs.empty())
    return std::nullopt;
  auto firstValue = configAttrs.front().getAs<AttrT>(name);
  if (!firstValue)
    return std::nullopt;
  auto result = firstValue.getValue();
  for (auto configAttr : configAttrs.drop_front(1)) {
    auto value = configAttr.getAs<AttrT>(name);
    if (!value)
      return std::nullopt;
    result = join(result, value.getValue());
  }
  return result;
}

template <typename AttrT>
static std::optional<StaticRange<typename AttrT::ValueType>>
joinConfigStaticRanges(ArrayRef<DictionaryAttr> configAttrs, StringRef name,
                       std::function<StaticRange<typename AttrT::ValueType>(
                           StaticRange<typename AttrT::ValueType>,
                           StaticRange<typename AttrT::ValueType>)>
                           join) {
  if (configAttrs.empty())
    return std::nullopt;
  auto firstValue = configAttrs.front().getAs<AttrT>(name);
  if (!firstValue)
    return std::nullopt;
  StaticRange<typename AttrT::ValueType> result{firstValue.getValue()};
  for (auto configAttr : configAttrs.drop_front(1)) {
    auto value = configAttr.getAs<AttrT>(name);
    if (!value)
      return std::nullopt;
    result =
        join(result, StaticRange<typename AttrT::ValueType>{value.getValue()});
  }
  return result;
}

// static
bool DeviceTargetAttr::lookupConfigAttrAny(Operation *op, StringRef name) {
  auto configAttrs = lookupOptionalConfigAttrs(op);
  if (configAttrs.empty())
    return false;
  for (auto configAttr : configAttrs) {
    if (configAttr.get(name))
      return true;
  }
  return false;
}

// static
bool DeviceTargetAttr::lookupConfigAttrAll(Operation *op, StringRef name) {
  auto configAttrs = lookupRequiredConfigAttrs(op);
  if (!configAttrs)
    return false;
  for (auto configAttr : *configAttrs) {
    if (!configAttr.get(name))
      return false;
  }
  return true;
}

// static
std::optional<bool> DeviceTargetAttr::lookupConfigAttrAnd(Operation *op,
                                                          StringRef name) {
  auto configAttrs = lookupRequiredConfigAttrs(op);
  if (!configAttrs)
    return std::nullopt;
  return joinConfigAttrs<BoolAttr>(
      configAttrs.value(), name, [](bool lhs, bool rhs) { return lhs && rhs; });
}

// static
std::optional<bool> DeviceTargetAttr::lookupConfigAttrOr(Operation *op,
                                                         StringRef name) {
  auto configAttrs = lookupRequiredConfigAttrs(op);
  if (!configAttrs)
    return std::nullopt;
  return joinConfigAttrs<BoolAttr>(
      configAttrs.value(), name, [](bool lhs, bool rhs) { return lhs || rhs; });
}

// static
std::optional<StaticRange<APInt>>
DeviceTargetAttr::lookupConfigAttrRange(Operation *op, StringRef name) {
  auto configAttrs = lookupRequiredConfigAttrs(op);
  if (!configAttrs)
    return std::nullopt;
  return joinConfigStaticRanges<IntegerAttr>(
      configAttrs.value(), name,
      [](StaticRange<APInt> lhs, StaticRange<APInt> rhs) {
        return StaticRange<APInt>{
            llvm::APIntOps::smin(lhs.min, rhs.min),
            llvm::APIntOps::smax(lhs.max, rhs.max),
        };
      });
}

// static
SmallVector<ExecutableTargetAttr, 4>
DeviceTargetAttr::lookupExecutableTargets(Operation *op) {
  SmallVector<ExecutableTargetAttr, 4> resultAttrs;
  for (auto deviceTargetAttr : lookup(op)) {
    for (auto executableTargetAttr : deviceTargetAttr.getExecutableTargets()) {
      if (!llvm::is_contained(resultAttrs, executableTargetAttr)) {
        resultAttrs.push_back(executableTargetAttr);
      }
    }
  }
  return resultAttrs;
}

//===----------------------------------------------------------------------===//
// #hal.executable.target
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
    if (auto variantOp = llvm::dyn_cast<ExecutableVariantOp>(op)) {
      return variantOp.getTarget();
    }
    // Use an override if specified.
    auto attr = op->getAttrOfType<ExecutableTargetAttr>(attrId);
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
// #hal.executable.object
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
    SmallVectorImpl<ExecutableObjectAttr> &filteredAttrs) {
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
// #hal.executable.objects
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
// #hal.affinity.queue
//===----------------------------------------------------------------------===//

// static
Attribute AffinityQueueAttr::parse(AsmParser &p, Type type) {
  int64_t mask = 0;
  // `<`
  if (failed(p.parseLess()))
    return {};
  // `*` (any)
  if (succeeded(p.parseOptionalStar())) {
    mask = -1;
  } else {
    // `[`queue_bit[, ...] `]`
    if (failed(p.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
          int64_t i = 0;
          if (failed(p.parseInteger(i)))
            return failure();
          mask |= 1ll << i;
          return success();
        }))) {
      return {};
    }
  }
  // `>`
  if (failed(p.parseGreater()))
    return {};
  return get(p.getContext(), mask);
}

void AffinityQueueAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  int64_t mask = getMask();
  if (mask == -1) {
    os << "*";
  } else {
    os << "[";
    for (int i = 0, j = 0; i < sizeof(mask) * 8; ++i) {
      if (mask & (1ll << i)) {
        if (j++ > 0)
          os << ", ";
        os << i;
      }
    }
    os << "]";
  }
  os << ">";
}

bool AffinityQueueAttr::isExecutableWith(
    IREE::Stream::AffinityAttr other) const {
  if (!other)
    return true;
  // Only compatible with other queue affinities today. When we extend the
  // attributes to specify device targets we'd want to check here.
  auto otherQueueAttr = llvm::dyn_cast_if_present<AffinityQueueAttr>(other);
  if (!otherQueueAttr)
    return false;
  // If this affinity is a subset of the target affinity then it can execute
  // with it.
  if ((getMask() & otherQueueAttr.getMask()) == getMask())
    return true;
  // Otherwise not compatible.
  return false;
}

IREE::Stream::AffinityAttr
AffinityQueueAttr::joinOR(IREE::Stream::AffinityAttr other) const {
  if (!other)
    return *this;
  if (!IREE::Stream::AffinityAttr::canExecuteTogether(*this, other)) {
    return nullptr;
  }
  auto otherQueueAttr = llvm::dyn_cast_if_present<AffinityQueueAttr>(other);
  return AffinityQueueAttr::get(getContext(),
                                getMask() | otherQueueAttr.getMask());
}

IREE::Stream::AffinityAttr
AffinityQueueAttr::joinAND(IREE::Stream::AffinityAttr other) const {
  if (!other)
    return *this;
  if (!IREE::Stream::AffinityAttr::canExecuteTogether(*this, other)) {
    return nullptr;
  }
  auto otherQueueAttr = llvm::dyn_cast_if_present<AffinityQueueAttr>(other);
  return AffinityQueueAttr::get(getContext(),
                                getMask() & otherQueueAttr.getMask());
}

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/HAL/IR/HALAttrInterfaces.cpp.inc"
#include "iree/compiler/Dialect/HAL/IR/HALOpInterfaces.cpp.inc"
#include "iree/compiler/Dialect/HAL/IR/HALTypeInterfaces.cpp.inc"

void HALDialect::registerAttributes() {
  // Register command line flags:
  (void)clExecutableObjectSearchPath;

  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Dialect/HAL/IR/HALAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

void HALDialect::registerTypes() {
  addTypes<AllocatorType, BufferType, BufferViewType, ChannelType,
           CommandBufferType, DescriptorSetLayoutType, DeviceType, EventType,
           ExecutableType, FenceType, FileType, PipelineLayoutType,
           SemaphoreType>();
}

//===----------------------------------------------------------------------===//
// Attribute printing and parsing
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Type printing and parsing
//===----------------------------------------------------------------------===//

Type HALDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeKind;
  if (parser.parseKeyword(&typeKind))
    return {};
  auto type =
      llvm::StringSwitch<Type>(typeKind)
          .Case("allocator", AllocatorType::get(getContext()))
          .Case("buffer", BufferType::get(getContext()))
          .Case("buffer_view", BufferViewType::get(getContext()))
          .Case("channel", ChannelType::get(getContext()))
          .Case("command_buffer", CommandBufferType::get(getContext()))
          .Case("descriptor_set_layout",
                DescriptorSetLayoutType::get(getContext()))
          .Case("device", DeviceType::get(getContext()))
          .Case("event", EventType::get(getContext()))
          .Case("executable", ExecutableType::get(getContext()))
          .Case("fence", FenceType::get(getContext()))
          .Case("file", FileType::get(getContext()))
          .Case("pipeline_layout", PipelineLayoutType::get(getContext()))
          .Case("semaphore", SemaphoreType::get(getContext()))
          .Default(nullptr);
  if (!type) {
    parser.emitError(parser.getCurrentLocation())
        << "unknown HAL type: " << typeKind;
  }
  return type;
}

void HALDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (llvm::isa<AllocatorType>(type)) {
    p << "allocator";
  } else if (llvm::isa<BufferType>(type)) {
    p << "buffer";
  } else if (llvm::isa<BufferViewType>(type)) {
    p << "buffer_view";
  } else if (llvm::isa<ChannelType>(type)) {
    p << "channel";
  } else if (llvm::isa<CommandBufferType>(type)) {
    p << "command_buffer";
  } else if (llvm::isa<DescriptorSetLayoutType>(type)) {
    p << "descriptor_set_layout";
  } else if (llvm::isa<DeviceType>(type)) {
    p << "device";
  } else if (llvm::isa<EventType>(type)) {
    p << "event";
  } else if (llvm::isa<ExecutableType>(type)) {
    p << "executable";
  } else if (llvm::isa<FenceType>(type)) {
    p << "fence";
  } else if (llvm::isa<FileType>(type)) {
    p << "file";
  } else if (llvm::isa<PipelineLayoutType>(type)) {
    p << "pipeline_layout";
  } else if (llvm::isa<SemaphoreType>(type)) {
    p << "semaphore";
  } else {
    assert(false && "unknown HAL type");
  }
}

} // namespace mlir::iree_compiler::IREE::HAL
