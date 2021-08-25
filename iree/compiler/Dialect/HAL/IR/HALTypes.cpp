// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/HAL/IR/HALAttrs.cpp.inc"    // IWYU pragma: keep
#include "iree/compiler/Dialect/HAL/IR/HALEnums.cpp.inc"    // IWYU pragma: keep
#include "iree/compiler/Dialect/HAL/IR/HALStructs.cpp.inc"  // IWYU pragma: keep
// clang-format on

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

//===----------------------------------------------------------------------===//
// Enum utilities
//===----------------------------------------------------------------------===//

template <typename AttrType>
static LogicalResult parseEnumAttr(DialectAsmParser &parser, StringRef attrName,
                                   AttrType &attr) {
  Attribute genericAttr;
  auto loc = parser.getCurrentLocation();
  if (failed(parser.parseAttribute(genericAttr,
                                   parser.getBuilder().getNoneType()))) {
    return parser.emitError(loc)
           << "failed to parse '" << attrName << "' enum string value";
  }
  auto stringAttr = genericAttr.dyn_cast<StringAttr>();
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
static LogicalResult parseOptionalEnumAttr(DialectAsmParser &parser,
                                           StringRef attrName, AttrType &attr) {
  if (succeeded(parser.parseOptionalQuestion())) {
    // Special case `?` to indicate any/none/undefined/etc.
    attr = AttrType::get(parser.getBuilder().getContext(), 0);
    return success();
  }
  return parseEnumAttr<AttrType>(parser, attrName, attr);
}

static LogicalResult parseMemoryType(DialectAsmParser &parser,
                                     Attribute &attr) {
  if (succeeded(parser.parseOptionalQuestion())) {
    attr = parser.getBuilder().getI32IntegerAttr(0);
    return success();
  }

  std::string fullString;
  if (succeeded(parser.parseOptionalString(&fullString))) {
    auto symbolized = symbolizeEnum<MemoryTypeBitfield>(fullString);
    if (!symbolized.hasValue()) {
      return parser.emitError(parser.getCurrentLocation())
             << "failed to parse memory type enum value";
    }
    attr = parser.getBuilder().getI32IntegerAttr(
        static_cast<int32_t>(symbolized.getValue()));
    return success();
  }

  StringRef shortString;
  if (failed(parser.parseKeyword(&shortString))) {
    return parser.emitError(parser.getCurrentLocation())
           << "failed to find memory type short string";
  }
  MemoryTypeBitfield memoryType = MemoryTypeBitfield::None;
  for (char c : shortString) {
    switch (c) {
      case 'T':
        memoryType = memoryType | MemoryTypeBitfield::Transient;
        break;
      case 'h':
        memoryType = memoryType | MemoryTypeBitfield::HostVisible;
        break;
      case 'H':
        memoryType = memoryType | MemoryTypeBitfield::HostLocal;
        break;
      case 'c':
        memoryType = memoryType | MemoryTypeBitfield::HostCoherent;
        break;
      case 'C':
        memoryType = memoryType | MemoryTypeBitfield::HostCached;
        break;
      case 'd':
        memoryType = memoryType | MemoryTypeBitfield::DeviceVisible;
        break;
      case 'D':
        memoryType = memoryType | MemoryTypeBitfield::DeviceLocal;
        break;
      default:
        return parser.emitError(parser.getCurrentLocation())
               << "unknown memory type short-form char: " << c;
    }
  }
  attr =
      parser.getBuilder().getI32IntegerAttr(static_cast<int32_t>(memoryType));
  return success();
}

static void printMemoryType(DialectAsmPrinter &printer,
                            MemoryTypeBitfield memoryType) {
  if (memoryType == MemoryTypeBitfield::None) {
    printer << '?';
    return;
  }
  if (allEnumBitsSet(memoryType, MemoryTypeBitfield::Transient)) {
    printer << 't';
  }
  if (allEnumBitsSet(memoryType, MemoryTypeBitfield::HostLocal)) {
    printer << 'H';
  } else if (allEnumBitsSet(memoryType, MemoryTypeBitfield::HostVisible)) {
    printer << 'h';
  }
  if (allEnumBitsSet(memoryType, MemoryTypeBitfield::HostCoherent)) {
    printer << 'c';
  }
  if (allEnumBitsSet(memoryType, MemoryTypeBitfield::HostCached)) {
    printer << 'C';
  }
  if (allEnumBitsSet(memoryType, MemoryTypeBitfield::DeviceLocal)) {
    printer << 'D';
  } else if (allEnumBitsSet(memoryType, MemoryTypeBitfield::DeviceVisible)) {
    printer << 'd';
  }
}

static LogicalResult parseMemoryAccess(DialectAsmParser &parser,
                                       Attribute &attr) {
  if (succeeded(parser.parseOptionalQuestion())) {
    attr = parser.getBuilder().getI32IntegerAttr(0);
    return success();
  }

  std::string fullString;
  if (succeeded(parser.parseOptionalString(&fullString))) {
    auto symbolized = symbolizeEnum<MemoryAccessBitfield>(fullString);
    if (!symbolized.hasValue()) {
      return parser.emitError(parser.getCurrentLocation())
             << "failed to parse memory access enum value";
    }
    attr = parser.getBuilder().getI32IntegerAttr(
        static_cast<int32_t>(symbolized.getValue()));
    return success();
  }

  StringRef shortString;
  if (failed(parser.parseKeyword(&shortString))) {
    return parser.emitError(parser.getCurrentLocation())
           << "failed to find memory access short string";
  }
  MemoryAccessBitfield memoryAccess = MemoryAccessBitfield::None;
  for (char c : shortString) {
    switch (c) {
      case 'R':
        memoryAccess = memoryAccess | MemoryAccessBitfield::Read;
        break;
      case 'W':
        memoryAccess = memoryAccess | MemoryAccessBitfield::Write;
        break;
      case 'D':
        memoryAccess = memoryAccess | MemoryAccessBitfield::Discard;
        break;
      case 'A':
        memoryAccess = memoryAccess | MemoryAccessBitfield::MayAlias;
        break;
      default:
        return parser.emitError(parser.getCurrentLocation())
               << "unknown memory access short-form char: " << c;
    }
  }
  attr =
      parser.getBuilder().getI32IntegerAttr(static_cast<int32_t>(memoryAccess));
  return success();
}

static void printMemoryAccess(DialectAsmPrinter &printer,
                              MemoryAccessBitfield memoryAccess) {
  if (memoryAccess == MemoryAccessBitfield::None) {
    printer << '?';
    return;
  }
  if (allEnumBitsSet(memoryAccess, MemoryAccessBitfield::Read)) {
    printer << 'R';
  }
  if (allEnumBitsSet(memoryAccess, MemoryAccessBitfield::Discard)) {
    printer << 'D';
  }
  if (allEnumBitsSet(memoryAccess, MemoryAccessBitfield::Write)) {
    printer << 'W';
  }
  if (allEnumBitsSet(memoryAccess, MemoryAccessBitfield::MayAlias)) {
    printer << 'A';
  }
}

static LogicalResult parseBufferUsage(DialectAsmParser &parser,
                                      Attribute &attr) {
  if (succeeded(parser.parseOptionalQuestion())) {
    attr = parser.getBuilder().getI32IntegerAttr(0);
    return success();
  }

  std::string fullString;
  if (succeeded(parser.parseOptionalString(&fullString))) {
    auto symbolized = symbolizeEnum<BufferUsageBitfield>(fullString);
    if (!symbolized.hasValue()) {
      return parser.emitError(parser.getCurrentLocation())
             << "failed to parse buffer usage enum value";
    }
    attr = parser.getBuilder().getI32IntegerAttr(
        static_cast<int32_t>(symbolized.getValue()));
    return success();
  }

  StringRef shortString;
  if (failed(parser.parseKeyword(&shortString))) {
    return parser.emitError(parser.getCurrentLocation())
           << "failed to find buffer usage short string";
  }
  BufferUsageBitfield usage = BufferUsageBitfield::None;
  for (char c : shortString) {
    switch (c) {
      case 'C':
        usage = usage | BufferUsageBitfield::Constant;
        break;
      case 'T':
        usage = usage | BufferUsageBitfield::Transfer;
        break;
      case 'M':
        usage = usage | BufferUsageBitfield::Mapping;
        break;
      case 'D':
        usage = usage | BufferUsageBitfield::Dispatch;
        break;
      default:
        return parser.emitError(parser.getCurrentLocation())
               << "unknown buffer usage short-form char: " << c;
    }
  }
  attr = parser.getBuilder().getI32IntegerAttr(static_cast<int32_t>(usage));
  return success();
}

static void printBufferUsage(DialectAsmPrinter &printer,
                             BufferUsageBitfield usage) {
  if (usage == BufferUsageBitfield::None) {
    printer << '?';
    return;
  }
  if (allEnumBitsSet(usage, BufferUsageBitfield::Constant)) {
    printer << 'C';
  }
  if (allEnumBitsSet(usage, BufferUsageBitfield::Transfer)) {
    printer << 'T';
  }
  if (allEnumBitsSet(usage, BufferUsageBitfield::Mapping)) {
    printer << 'M';
  }
  if (allEnumBitsSet(usage, BufferUsageBitfield::Dispatch)) {
    printer << 'D';
  }
}

//===----------------------------------------------------------------------===//
// Element types
//===----------------------------------------------------------------------===//

// Keep these in sync with iree/hal/api.h
namespace {
enum class NumericalType : uint32_t {
  kUnknown = 0x00,
  kIntegerSigned = 0x01,
  kIntegerUnsigned = 0x02,
  // TODO(benvanik): specialize with semantics from APFloat.
  kFloatIEEE = 0x03,
};
constexpr inline int32_t makeElementTypeValue(NumericalType numericalType,
                                              int32_t bitCount) {
  return (static_cast<uint32_t>(numericalType) << 24) | bitCount;
}
}  // namespace
llvm::Optional<int32_t> getElementTypeValue(Type type) {
  if (auto intType = type.dyn_cast_or_null<IntegerType>()) {
    // TODO(benvanik): add signed/unsigned check when landed in MLIR.
    return makeElementTypeValue(NumericalType::kIntegerSigned,
                                intType.getWidth());
  } else if (auto floatType = type.dyn_cast_or_null<FloatType>()) {
    switch (APFloat::SemanticsToEnum(floatType.getFloatSemantics())) {
      case APFloat::S_IEEEhalf:
      case APFloat::S_IEEEsingle:
      case APFloat::S_IEEEdouble:
      case APFloat::S_IEEEquad:
        return makeElementTypeValue(NumericalType::kFloatIEEE,
                                    floatType.getWidth());
      default:
        return llvm::None;
    }
  }
  return llvm::None;
}

IntegerAttr getElementTypeAttr(Type type) {
  auto elementType = getElementTypeValue(type);
  if (!elementType) return {};
  return IntegerAttr::get(IntegerType::get(type.getContext(), 32),
                          elementType.getValue());
}

size_t getElementBitCount(IntegerAttr elementType) {
  return static_cast<size_t>((elementType.getValue().getZExtValue()) & 0xFF);
}

Value getElementBitCount(Location loc, Value elementType, OpBuilder &builder) {
  return builder.createOrFold<AndOp>(
      loc,
      builder.createOrFold<IndexCastOp>(loc, builder.getIndexType(),
                                        elementType),
      builder.createOrFold<ConstantIndexOp>(loc, 0xFF));
}

size_t getElementByteCount(IntegerAttr elementType) {
  return (getElementBitCount(elementType) + 8 - 1) / 8;
}

Value getElementByteCount(Location loc, Value elementType, OpBuilder &builder) {
  auto c1 = builder.createOrFold<ConstantIndexOp>(loc, 1);
  auto c8 = builder.createOrFold<ConstantIndexOp>(loc, 8);
  auto bitCount = getElementBitCount(loc, elementType, builder);
  return builder.createOrFold<UnsignedDivIOp>(
      loc,
      builder.createOrFold<SubIOp>(
          loc, builder.createOrFold<AddIOp>(loc, bitCount, c8), c1),
      c8);
}

llvm::Optional<int32_t> getEncodingTypeValue(Attribute attr) {
  // TODO(#6762): encoding attribute handling/mapping to enums.
  assert(!attr && "encoding types other than default not yet supported");
  // Default to IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR for now.
  return 1;
}

IntegerAttr getEncodingTypeAttr(Attribute attr, MLIRContext *context) {
  auto encodingType = getEncodingTypeValue(attr);
  if (!encodingType) return {};
  return IntegerAttr::get(IntegerType::get(context, 32),
                          encodingType.getValue());
}

//===----------------------------------------------------------------------===//
// Size-aware type utils
//===----------------------------------------------------------------------===//

// Returns the SSA value containing the size of the given |value|.
static Value lookupValueSize(Value value) {
  assert(value.getType().isa<IREE::Util::SizeAwareTypeInterface>());

  auto definingOp = value.getDefiningOp();
  if (!definingOp) {
    return {};  // Not yet implemented.
  }

  // Skip do-not-optimize ops.
  if (auto dnoOp = dyn_cast<IREE::Util::DoNotOptimizeOp>(definingOp)) {
    return lookupValueSize(dnoOp.getOperand(0));
  }

  // Query size from the size-aware op that defined the value, as it knows how
  // to get/build the right value.
  unsigned resultIndex = -1;
  for (unsigned i = 0; i < definingOp->getNumResults(); ++i) {
    if (definingOp->getResult(i) == value) {
      resultIndex = i;
      break;
    }
  }
  assert(resultIndex != -1 && "result not in results");
  auto sizeAwareOp = dyn_cast<IREE::Util::SizeAwareOpInterface>(definingOp);
  if (!sizeAwareOp) return {};
  return sizeAwareOp.getResultSize(resultIndex);
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
  return builder.createOrFold<BufferViewByteLengthOp>(loc, value);
}

//===----------------------------------------------------------------------===//
// Struct types
//===----------------------------------------------------------------------===//

BufferConstraintsAttr intersectBufferConstraints(BufferConstraintsAttr lhs,
                                                 BufferConstraintsAttr rhs) {
  if (!lhs) return rhs;
  if (!rhs) return lhs;
  Builder b(lhs.getContext());
  return BufferConstraintsAttr::get(
      b.getIndexAttr(std::min(lhs.max_allocation_size().getSExtValue(),
                              rhs.max_allocation_size().getSExtValue())),
      b.getIndexAttr(
          std::max(lhs.min_buffer_offset_alignment().getSExtValue(),
                   rhs.min_buffer_offset_alignment().getSExtValue())),
      b.getIndexAttr(std::min(lhs.max_buffer_range().getSExtValue(),
                              rhs.max_buffer_range().getSExtValue())),
      b.getIndexAttr(
          std::max(lhs.min_buffer_range_alignment().getSExtValue(),
                   rhs.min_buffer_range_alignment().getSExtValue())));
}

// TODO(benvanik): runtime buffer constraint queries from the allocator.
// We can add folders for those when the allocator is strongly-typed with
// #hal.buffer_constraints and otherwise leave them for runtime queries.
BufferConstraintsAdaptor::BufferConstraintsAdaptor(Location loc,
                                                   Value allocator)
    : loc_(loc), allocator_(allocator) {
  // Picked to represent what we kind of want on CPU today.
  uint64_t maxAllocationSize = 1 * 1024 * 1024 * 1024ull;
  uint64_t minBufferOffsetAlignment = 16ull;
  uint64_t maxBufferRange = 1 * 1024 * 1024 * 1024ull;
  uint64_t minBufferRangeAlignment = 16ull;
  Builder b(loc.getContext());
  bufferConstraints_ = BufferConstraintsAttr::get(
      b.getIndexAttr(maxAllocationSize),
      b.getIndexAttr(minBufferOffsetAlignment), b.getIndexAttr(maxBufferRange),
      b.getIndexAttr(minBufferRangeAlignment));
}

Value BufferConstraintsAdaptor::getMaxAllocationSize(OpBuilder &builder) {
  return builder.createOrFold<mlir::ConstantOp>(
      loc_, bufferConstraints_.max_allocation_sizeAttr());
}

Value BufferConstraintsAdaptor::getMinBufferOffsetAlignment(
    OpBuilder &builder) {
  return builder.createOrFold<mlir::ConstantOp>(
      loc_, bufferConstraints_.min_buffer_offset_alignmentAttr());
}

Value BufferConstraintsAdaptor::getMaxBufferRange(OpBuilder &builder) {
  return builder.createOrFold<mlir::ConstantOp>(
      loc_, bufferConstraints_.max_buffer_rangeAttr());
}

Value BufferConstraintsAdaptor::getMinBufferRangeAlignment(OpBuilder &builder) {
  return builder.createOrFold<mlir::ConstantOp>(
      loc_, bufferConstraints_.min_buffer_range_alignmentAttr());
}

//===----------------------------------------------------------------------===//
// Attribute printing and parsing
//===----------------------------------------------------------------------===//

// static
Attribute BufferConstraintsAttr::parse(DialectAsmParser &p) {
  auto b = p.getBuilder();
  if (failed(p.parseLess())) return {};

  IntegerAttr maxAllocationSizeAttr;
  IntegerAttr minBufferOffsetAlignmentAttr;
  IntegerAttr maxBufferRangeAttr;
  IntegerAttr minBufferRangeAlignmentAttr;
  if (failed(p.parseKeyword("max_allocation_size")) || failed(p.parseEqual()) ||
      failed(p.parseAttribute(maxAllocationSizeAttr, b.getIndexType())) ||
      failed(p.parseComma()) ||
      failed(p.parseKeyword("min_buffer_offset_alignment")) ||
      failed(p.parseEqual()) ||
      failed(
          p.parseAttribute(minBufferOffsetAlignmentAttr, b.getIndexType())) ||
      failed(p.parseComma()) || failed(p.parseKeyword("max_buffer_range")) ||
      failed(p.parseEqual()) ||
      failed(p.parseAttribute(maxBufferRangeAttr, b.getIndexType())) ||
      failed(p.parseComma()) ||
      failed(p.parseKeyword("min_buffer_range_alignment")) ||
      failed(p.parseEqual()) ||
      failed(p.parseAttribute(minBufferRangeAlignmentAttr, b.getIndexType()))) {
    return {};
  }

  if (failed(p.parseGreater())) return {};
  return BufferConstraintsAttr::get(
      maxAllocationSizeAttr, minBufferOffsetAlignmentAttr, maxBufferRangeAttr,
      minBufferRangeAlignmentAttr);
}

void BufferConstraintsAttr::print(DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  os << getKindName() << "<";
  os << "max_allocation_size = " << max_allocation_size() << ", ";
  os << "min_buffer_offset_alignment = " << min_buffer_offset_alignment()
     << ", ";
  os << "max_buffer_range = " << max_buffer_range() << ", ";
  os << "min_buffer_range_alignment = " << min_buffer_range_alignment();
  os << ">";
}

// static
Attribute ByteRangeAttr::parse(DialectAsmParser &p) {
  auto b = p.getBuilder();
  if (failed(p.parseLess())) return {};

  // TODO(benvanik): support the range syntax; the dialect asm parser fights
  // with it though by checking for proper []/() nesting.

  // Try first the range style: byte_range<[start..end)>
  bool startInclusive;
  if (succeeded(p.parseOptionalLSquare())) {  // [...
    startInclusive = true;
  } else if (succeeded(p.parseOptionalLParen())) {  // (...
    startInclusive = false;
  } else {
    // byte_range<offset, length>
    IntegerAttr offsetAttr;
    IntegerAttr lengthAttr;
    if (failed(p.parseAttribute(offsetAttr, b.getIndexType())) ||
        failed(p.parseComma()) ||
        failed(p.parseAttribute(lengthAttr, b.getIndexType())) ||
        failed(p.parseGreater())) {
      return {};
    }
    return get(offsetAttr, lengthAttr);
  }

  IntegerAttr startAttr;
  IntegerAttr endAttr;
  if (failed(p.parseAttribute(startAttr, b.getIndexType())) ||
      failed(p.parseKeyword("to")) ||
      failed(p.parseAttribute(endAttr, b.getIndexType()))) {
    return {};
  }

  bool endInclusive;
  if (succeeded(p.parseOptionalRSquare())) {  // ...]
    endInclusive = true;
  } else if (succeeded(p.parseOptionalRParen())) {  // ...)
    endInclusive = false;
  } else {
    p.emitError(p.getCurrentLocation()) << "expected ] or ) to end range";
    return {};
  }

  if (failed(p.parseGreater())) return {};

  startAttr = startInclusive
                  ? startAttr
                  : b.getIndexAttr((startAttr.getValue() + 1).getSExtValue());
  endAttr = endInclusive
                ? endAttr
                : b.getIndexAttr((endAttr.getValue() - 1).getSExtValue());

  IntegerAttr offsetAttr = startAttr;
  IntegerAttr lengthAttr = b.getIndexAttr(
      (endAttr.getValue() - startAttr.getValue()).getSExtValue());
  return get(offsetAttr, lengthAttr);
}

void ByteRangeAttr::print(DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  os << getKindName() << "<";
  os << offset();
  os << ", ";
  os << length();
  os << ">";
}

// static
Attribute DescriptorSetLayoutBindingAttr::parse(DialectAsmParser &p) {
  auto b = p.getBuilder();
  IntegerAttr bindingAttr;
  DescriptorTypeAttr typeAttr;
  MemoryAccessBitfieldAttr accessAttr;
  if (failed(p.parseLess()) ||
      failed(p.parseAttribute(bindingAttr, b.getIntegerType(32))) ||
      failed(p.parseComma()) || failed(parseEnumAttr(p, "type", typeAttr)) ||
      failed(p.parseComma()) || failed(parseMemoryAccess(p, accessAttr)) ||
      failed(p.parseGreater())) {
    return {};
  }
  return get(bindingAttr, typeAttr, accessAttr);
}

void DescriptorSetLayoutBindingAttr::print(DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  os << getKindName() << "<";
  os << binding() << ", ";
  os << "\"" << stringifyDescriptorType(type()) << "\", ";
  printMemoryAccess(p, access());
  os << ">";
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
Attribute DeviceTargetAttr::parse(MLIRContext *context, DialectAsmParser &p,
                                  Type type) {
  auto b = p.getBuilder();
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
  return get(b.getContext(), deviceIDAttr, configAttr);
}

void DeviceTargetAttr::print(DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  os << getMnemonic() << "<";
  p.printAttribute(getDeviceID());
  auto configAttr = getConfiguration();
  if (configAttr && !configAttr.empty()) {
    os << ", ";
    p.printAttribute(configAttr);
  }
  os << ">";
}

std::string DeviceTargetAttr::getSymbolNameFragment() {
  auto deviceName = getDeviceID().getValue().lower();
  std::replace(deviceName.begin(), deviceName.end(), '-', '_');
  return deviceName;
}

Attribute DeviceTargetAttr::getMatchExpression() {
  return DeviceMatchIDAttr::get(*this);
}

BufferConstraintsAttr DeviceTargetAttr::getBufferConstraints() {
  BufferConstraintsAttr constraintsAttr;
  auto configAttr = getConfiguration();
  if (configAttr) {
    constraintsAttr =
        configAttr.getAs<BufferConstraintsAttr>("buffer_constraints");
  }
  return constraintsAttr ? constraintsAttr
                         : getDefaultHostBufferConstraints(getContext());
}

SmallVector<ExecutableTargetAttr, 4> DeviceTargetAttr::getExecutableTargets() {
  SmallVector<ExecutableTargetAttr, 4> resultAttrs;
  auto configAttr = getConfiguration();
  if (configAttr) {
    auto targetsAttr = configAttr.getAs<ArrayAttr>("executable_targets");
    if (targetsAttr) {
      for (auto attr : targetsAttr.getValue()) {
        resultAttrs.push_back(attr.dyn_cast<ExecutableTargetAttr>());
      }
    }
  }
  return resultAttrs;
}

// static
SmallVector<IREE::HAL::DeviceTargetAttr, 4> DeviceTargetAttr::lookup(
    Operation *op) {
  auto attrId = mlir::Identifier::get("hal.device.targets", op->getContext());
  while (op) {
    auto targetsAttr = op->getAttrOfType<ArrayAttr>(attrId);
    if (targetsAttr) {
      SmallVector<IREE::HAL::DeviceTargetAttr, 4> result;
      for (auto targetAttr : targetsAttr) {
        result.push_back(targetAttr.cast<IREE::HAL::DeviceTargetAttr>());
      }
      return result;
    }
    op = op->getParentOp();
  }
  return {};  // No devices found; let caller decide what to do.
}

// static
BufferConstraintsAttr DeviceTargetAttr::getDefaultHostBufferConstraints(
    MLIRContext *context) {
  // Picked to represent what we kind of want on CPU today.
  uint64_t maxAllocationSize = 1 * 1024 * 1024 * 1024ull;
  uint64_t minBufferOffsetAlignment = 16ull;
  uint64_t maxBufferRange = 1 * 1024 * 1024 * 1024ull;
  uint64_t minBufferRangeAlignment = 16ull;
  Builder b(context);
  return BufferConstraintsAttr::get(b.getIndexAttr(maxAllocationSize),
                                    b.getIndexAttr(minBufferOffsetAlignment),
                                    b.getIndexAttr(maxBufferRange),
                                    b.getIndexAttr(minBufferRangeAlignment));
}

// static
BufferConstraintsAttr DeviceTargetAttr::lookupConservativeBufferConstraints(
    Operation *op) {
  BufferConstraintsAttr resultAttr = {};
  for (auto targetAttr : IREE::HAL::DeviceTargetAttr::lookup(op)) {
    auto configAttr = targetAttr.getConfiguration();
    if (!configAttr) continue;
    auto targetConstraintsAttr =
        configAttr.getAs<BufferConstraintsAttr>("buffer_constraints");
    if (!targetConstraintsAttr) continue;
    resultAttr = intersectBufferConstraints(resultAttr, targetConstraintsAttr);
  }
  return resultAttr ? resultAttr
                    : getDefaultHostBufferConstraints(op->getContext());
}

// static
SmallVector<ExecutableTargetAttr, 4> DeviceTargetAttr::lookupExecutableTargets(
    Operation *op) {
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
Attribute ExecutableTargetAttr::parse(MLIRContext *context, DialectAsmParser &p,
                                      Type type) {
  auto b = p.getBuilder();
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
  return get(b.getContext(), backendAttr, formatAttr, configurationAttr);
}

void ExecutableTargetAttr::print(DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  os << getMnemonic() << "<";
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

std::string ExecutableTargetAttr::getSymbolNameFragment() {
  auto format = getFormat().getValue().lower();
  std::replace(format.begin(), format.end(), '-', '_');
  return format;
}

Attribute ExecutableTargetAttr::getMatchExpression() {
  return DeviceMatchExecutableFormatAttr::get(getContext(), getFormat());
}

//===----------------------------------------------------------------------===//
// #hal.match.*
//===----------------------------------------------------------------------===//

Value MatchAlwaysAttr::buildConditionExpression(Location loc, Value value,
                                                OpBuilder builder) const {
  // #hal.match.always -> true
  return builder.createOrFold<ConstantIntOp>(loc, /*value=*/1, /*width=*/1);
}

static ArrayAttr parseMultiMatchAttrArray(DialectAsmParser &p) {
  auto b = p.getBuilder();
  SmallVector<Attribute, 4> conditionAttrs;
  if (failed(p.parseLess()) || failed(p.parseLSquare())) {
    return {};
  }
  do {
    Attribute conditionAttr;
    if (failed(p.parseAttribute(conditionAttr))) {
      return {};
    }
    conditionAttrs.push_back(conditionAttr);
  } while (succeeded(p.parseOptionalComma()));
  if (failed(p.parseRSquare()) || failed(p.parseGreater())) {
    return {};
  }
  return b.getArrayAttr(conditionAttrs);
}

static void printMultiMatchAttrList(ArrayAttr conditionAttrs,
                                    DialectAsmPrinter &p) {
  auto &os = p.getStream();
  os << "<[";
  interleaveComma(conditionAttrs, os,
                  [&](Attribute condition) { os << condition; });
  os << "]>";
}

// static
Attribute MatchAnyAttr::parse(MLIRContext *context, DialectAsmParser &p,
                              Type type) {
  return get(context, parseMultiMatchAttrArray(p));
}

void MatchAnyAttr::print(DialectAsmPrinter &p) const {
  p << getMnemonic();
  printMultiMatchAttrList(getConditions(), p);
}

Value MatchAnyAttr::buildConditionExpression(Location loc, Value value,
                                             OpBuilder builder) const {
  // #hal.match.any<[a, b, c]> -> or(or(a, b), c)
  if (getConditions().empty()) {
    // Empty returns false (no conditions match).
    return builder.create<ConstantIntOp>(loc, /*value=*/0, /*width=*/1);
  }
  auto conditionValues =
      llvm::map_range(getConditions(), [&](MatchAttrInterface attr) {
        return attr.buildConditionExpression(loc, value, builder);
      });
  Value resultValue;
  for (auto conditionValue : conditionValues) {
    resultValue = resultValue ? builder.createOrFold<OrOp>(loc, resultValue,
                                                           conditionValue)
                              : conditionValue;
  }
  return resultValue;
}

// static
Attribute MatchAllAttr::parse(MLIRContext *context, DialectAsmParser &p,
                              Type type) {
  return get(context, parseMultiMatchAttrArray(p));
}

void MatchAllAttr::print(DialectAsmPrinter &p) const {
  p << getMnemonic();
  printMultiMatchAttrList(getConditions(), p);
}

Value MatchAllAttr::buildConditionExpression(Location loc, Value value,
                                             OpBuilder builder) const {
  // #hal.match.all<[a, b, c]> -> and(and(a, b), c)
  if (getConditions().empty()) {
    // Empty returns true (all 0 conditions match).
    return builder.create<ConstantIntOp>(loc, /*value=*/1, /*width=*/1);
  }
  auto conditionValues =
      llvm::map_range(getConditions(), [&](MatchAttrInterface attr) {
        return attr.buildConditionExpression(loc, value, builder);
      });
  Value resultValue;
  for (auto conditionValue : conditionValues) {
    resultValue = resultValue ? builder.createOrFold<AndOp>(loc, resultValue,
                                                            conditionValue)
                              : conditionValue;
  }
  return resultValue;
}

// static
Attribute DeviceMatchIDAttr::parse(MLIRContext *context, DialectAsmParser &p,
                                   Type type) {
  StringAttr patternAttr;
  if (failed(p.parseLess()) || failed(p.parseAttribute(patternAttr)) ||
      failed(p.parseGreater())) {
    return {};
  }
  return get(context, patternAttr);
}

void DeviceMatchIDAttr::print(DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  os << getMnemonic() << "<";
  p.printAttribute(getPattern());
  os << ">";
}

Value DeviceMatchIDAttr::buildConditionExpression(Location loc, Value device,
                                                  OpBuilder builder) const {
  auto i1Type = builder.getI1Type();
  return builder
      .create<IREE::HAL::DeviceQueryOp>(
          loc, i1Type, i1Type, device, builder.getStringAttr("hal.device.id"),
          getPattern(), builder.getZeroAttr(i1Type))
      .value();
}

// static
Attribute DeviceMatchFeatureAttr::parse(MLIRContext *context,
                                        DialectAsmParser &p, Type type) {
  StringAttr patternAttr;
  if (failed(p.parseLess()) || failed(p.parseAttribute(patternAttr)) ||
      failed(p.parseGreater())) {
    return {};
  }
  return get(context, patternAttr);
}

void DeviceMatchFeatureAttr::print(DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  os << getMnemonic() << "<";
  p.printAttribute(getPattern());
  os << ">";
}

Value DeviceMatchFeatureAttr::buildConditionExpression(
    Location loc, Value device, OpBuilder builder) const {
  auto i1Type = builder.getI1Type();
  return builder
      .create<IREE::HAL::DeviceQueryOp>(
          loc, i1Type, i1Type, device,
          builder.getStringAttr("hal.device.feature"), getPattern(),
          builder.getZeroAttr(i1Type))
      .value();
}

// static
Attribute DeviceMatchArchitectureAttr::parse(MLIRContext *context,
                                             DialectAsmParser &p, Type type) {
  StringAttr patternAttr;
  if (failed(p.parseLess()) || failed(p.parseAttribute(patternAttr)) ||
      failed(p.parseGreater())) {
    return {};
  }
  return get(context, patternAttr);
}

void DeviceMatchArchitectureAttr::print(DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  os << getMnemonic() << "<";
  p.printAttribute(getPattern());
  os << ">";
}

Value DeviceMatchArchitectureAttr::buildConditionExpression(
    Location loc, Value device, OpBuilder builder) const {
  auto i1Type = builder.getI1Type();
  return builder
      .create<IREE::HAL::DeviceQueryOp>(
          loc, i1Type, i1Type, device,
          builder.getStringAttr("hal.device.architecture"), getPattern(),
          builder.getZeroAttr(i1Type))
      .value();
}

// static
Attribute DeviceMatchExecutableFormatAttr::parse(MLIRContext *context,
                                                 DialectAsmParser &p,
                                                 Type type) {
  StringAttr patternAttr;
  if (failed(p.parseLess()) || failed(p.parseAttribute(patternAttr)) ||
      failed(p.parseGreater())) {
    return {};
  }
  return get(context, patternAttr);
}

void DeviceMatchExecutableFormatAttr::print(DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  os << getMnemonic() << "<";
  p.printAttribute(getPattern());
  os << ">";
}

Value DeviceMatchExecutableFormatAttr::buildConditionExpression(
    Location loc, Value device, OpBuilder builder) const {
  auto i1Type = builder.getI1Type();
  return builder
      .create<IREE::HAL::DeviceQueryOp>(
          loc, i1Type, i1Type, device,
          builder.getStringAttr("hal.executable.format"), getPattern(),
          builder.getZeroAttr(i1Type))
      .value();
}

//===----------------------------------------------------------------------===//
// Experimental interface plumbing
//===----------------------------------------------------------------------===//

// static
Attribute ExConstantStorageAttr::parse(DialectAsmParser &p) {
  StringAttr bindingAttr;
  StringAttr storageAttr;
  IntegerAttr offsetAttr;
  IntegerAttr lengthAttr;
  if (failed(p.parseLess()) || failed(p.parseAttribute(bindingAttr)) ||
      failed(p.parseComma()) || failed(p.parseAttribute(storageAttr)) ||
      failed(p.parseComma()) || failed(p.parseAttribute(offsetAttr)) ||
      failed(p.parseComma()) || failed(p.parseAttribute(lengthAttr)) ||
      failed(p.parseGreater())) {
    return {};
  }
  return get(bindingAttr, storageAttr, offsetAttr, lengthAttr);
}

void ExConstantStorageAttr::print(DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  os << getKindName() << "<";
  p.printAttribute(bindingAttr());
  os << ", ";
  p.printAttribute(storageAttr());
  os << ", ";
  p.printAttribute(offsetAttr());
  os << ", ";
  p.printAttribute(lengthAttr());
  os << ">";
}

// static
Attribute ExPushConstantAttr::parse(DialectAsmParser &p) {
  IntegerAttr ordinalAttr;
  IntegerAttr operandAttr;
  if (failed(p.parseLess()) || failed(p.parseAttribute(ordinalAttr)) ||
      failed(p.parseComma()) || failed(p.parseAttribute(operandAttr)) ||
      failed(p.parseGreater())) {
    return {};
  }
  return get(ordinalAttr, operandAttr);
}

void ExPushConstantAttr::print(DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  os << getKindName() << "<";
  p.printAttribute(ordinalAttr());
  os << ", ";
  p.printAttribute(operandAttr());
  os << ">";
}

// static
Attribute ExOperandBufferAttr::parse(DialectAsmParser &p) {
  StringAttr bindingAttr;
  IntegerAttr operandAttr;
  if (failed(p.parseLess()) || failed(p.parseAttribute(bindingAttr)) ||
      failed(p.parseComma()) || failed(p.parseAttribute(operandAttr)) ||
      failed(p.parseGreater())) {
    return {};
  }
  return get(bindingAttr, operandAttr);
}

void ExOperandBufferAttr::print(DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  os << getKindName() << "<";
  p.printAttribute(bindingAttr());
  os << ", ";
  p.printAttribute(operandAttr());
  os << ">";
}

// static
Attribute ExResultBufferAttr::parse(DialectAsmParser &p) {
  StringAttr bindingAttr;
  IntegerAttr resultAttr;
  if (failed(p.parseLess()) || failed(p.parseAttribute(bindingAttr)) ||
      failed(p.parseComma()) || failed(p.parseAttribute(resultAttr)) ||
      failed(p.parseGreater())) {
    return {};
  }
  return get(bindingAttr, resultAttr);
}

void ExResultBufferAttr::print(DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  os << getKindName() << "<";
  p.printAttribute(bindingAttr());
  os << ", ";
  p.printAttribute(resultAttr());
  os << ">";
}

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/HAL/IR/HALAttrInterfaces.cpp.inc"
#include "iree/compiler/Dialect/HAL/IR/HALOpInterfaces.cpp.inc"
#include "iree/compiler/Dialect/HAL/IR/HALTypeInterfaces.cpp.inc"

void HALDialect::registerAttributes() {
  addAttributes<BufferConstraintsAttr, ByteRangeAttr,
                DescriptorSetLayoutBindingAttr,
                // Experimental:
                ExConstantStorageAttr, ExPushConstantAttr, ExOperandBufferAttr,
                ExResultBufferAttr>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Dialect/HAL/IR/HALAttrs.cpp.inc"  // IWYU pragma: keep
      >();
}

void HALDialect::registerTypes() {
  addTypes<AllocatorType, BufferType, BufferViewType, CommandBufferType,
           DescriptorSetType, DescriptorSetLayoutType, DeviceType, EventType,
           ExecutableType, ExecutableLayoutType, RingBufferType,
           SemaphoreType>();
}

//===----------------------------------------------------------------------===//
// Attribute printing and parsing
//===----------------------------------------------------------------------===//

Attribute HALDialect::parseAttribute(DialectAsmParser &parser,
                                     Type type) const {
  StringRef mnemonic;
  if (failed(parser.parseKeyword(&mnemonic))) return {};
  Attribute genAttr;
  OptionalParseResult parseResult =
      generatedAttributeParser(getContext(), parser, mnemonic, type, genAttr);
  if (parseResult.hasValue()) return genAttr;
  if (mnemonic == BufferConstraintsAttr::getKindName()) {
    return BufferConstraintsAttr::parse(parser);
  } else if (mnemonic == ByteRangeAttr::getKindName()) {
    return ByteRangeAttr::parse(parser);
  } else if (mnemonic == DescriptorSetLayoutBindingAttr::getKindName()) {
    return DescriptorSetLayoutBindingAttr::parse(parser);
  } else if (mnemonic == ExConstantStorageAttr::getKindName()) {
    return ExConstantStorageAttr::parse(parser);
  } else if (mnemonic == ExPushConstantAttr::getKindName()) {
    return ExPushConstantAttr::parse(parser);
  } else if (mnemonic == ExOperandBufferAttr::getKindName()) {
    return ExOperandBufferAttr::parse(parser);
  } else if (mnemonic == ExResultBufferAttr::getKindName()) {
    return ExResultBufferAttr::parse(parser);
  }
  parser.emitError(parser.getNameLoc())
      << "unknown HAL attribute: " << mnemonic;
  return {};
}

void HALDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
  TypeSwitch<Attribute>(attr)
      .Case<BufferConstraintsAttr, ByteRangeAttr,
            DescriptorSetLayoutBindingAttr,
            // Experimental:
            ExConstantStorageAttr, ExPushConstantAttr, ExOperandBufferAttr,
            ExResultBufferAttr>([&](auto typedAttr) { typedAttr.print(p); })
      .Default([&](Attribute) {
        if (failed(generatedAttributePrinter(attr, p))) {
          llvm_unreachable("unhandled HAL attribute kind");
        }
      });
}

//===----------------------------------------------------------------------===//
// Type printing and parsing
//===----------------------------------------------------------------------===//

Type HALDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeKind;
  if (parser.parseKeyword(&typeKind)) return {};
  auto type =
      llvm::StringSwitch<Type>(typeKind)
          .Case("allocator", AllocatorType::get(getContext()))
          .Case("buffer", BufferType::get(getContext()))
          .Case("buffer_view", BufferViewType::get(getContext()))
          .Case("command_buffer", CommandBufferType::get(getContext()))
          .Case("descriptor_set", DescriptorSetType::get(getContext()))
          .Case("descriptor_set_layout",
                DescriptorSetLayoutType::get(getContext()))
          .Case("device", DeviceType::get(getContext()))
          .Case("event", EventType::get(getContext()))
          .Case("executable", ExecutableType::get(getContext()))
          .Case("executable_layout", ExecutableLayoutType::get(getContext()))
          .Case("ring_buffer", RingBufferType::get(getContext()))
          .Case("semaphore", SemaphoreType::get(getContext()))
          .Default(nullptr);
  if (!type) {
    parser.emitError(parser.getCurrentLocation())
        << "unknown HAL type: " << typeKind;
  }
  return type;
}

void HALDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (type.isa<AllocatorType>()) {
    p << "allocator";
  } else if (type.isa<BufferType>()) {
    p << "buffer";
  } else if (type.isa<BufferViewType>()) {
    p << "buffer_view";
  } else if (type.isa<CommandBufferType>()) {
    p << "command_buffer";
  } else if (type.isa<DescriptorSetType>()) {
    p << "descriptor_set";
  } else if (type.isa<DescriptorSetLayoutType>()) {
    p << "descriptor_set_layout";
  } else if (type.isa<DeviceType>()) {
    p << "device";
  } else if (type.isa<EventType>()) {
    p << "event";
  } else if (type.isa<ExecutableType>()) {
    p << "executable";
  } else if (type.isa<ExecutableLayoutType>()) {
    p << "executable_layout";
  } else if (type.isa<RingBufferType>()) {
    p << "ring_buffer";
  } else if (type.isa<SemaphoreType>()) {
    p << "semaphore";
  } else {
    llvm_unreachable("unknown HAL type");
  }
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
