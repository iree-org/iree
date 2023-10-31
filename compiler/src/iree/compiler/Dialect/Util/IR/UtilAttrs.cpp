// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Parser/Parser.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/Util/IR/UtilAttrs.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

static llvm::cl::opt<bool> clZeroFillElidedAttrs(
    "iree-util-zero-fill-elided-attrs",
    llvm::cl::desc("Fills elided attributes with zeros when serializing."),
    llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// ostream utilities
//===----------------------------------------------------------------------===//

// ostream wrapper that writes to an existing buffer allocation.
// Assumes that no more data will be written than is allocated in the provided
// storage buffer.
class raw_inplace_ostream : public llvm::raw_pwrite_stream {
public:
  explicit raw_inplace_ostream(ArrayRef<char> storage) : storage(storage) {
    SetUnbuffered();
  }
  ~raw_inplace_ostream() override = default;

  void flush() = delete;

  void reserveExtraSpace(uint64_t extraSize) override {}

private:
  uint64_t current_pos() const override { return offset; }

  void write_impl(const char *ptr, size_t size) override {
    std::memcpy((char *)storage.data() + offset, ptr, size);
    offset += size;
  }

  void pwrite_impl(const char *ptr, size_t size, uint64_t poffset) override {
    std::memcpy((char *)storage.data() + poffset, ptr, size);
  }

  ArrayRef<char> storage;
  size_t offset = 0;
};

// Returns true if the raw data of the attribute matches our expected output
// format. This allows the use of the attribute getRawData() method.
static bool canUseRawData(DenseElementsAttr attr,
                          llvm::support::endianness endian) {
  Type elementType = attr.getElementType();
  if (!isa<IntegerType, FloatType, ComplexType>(elementType)) {
    // We cannot assume composite element types have the raw layout we want,
    // other than ComplexType, which is effectively vector<2xfN> and goes
    // through the later logic.
    return false;
  }

  int32_t bitWidth = getTypeBitWidth(elementType);
  if (bitWidth == 8) {
    // Don't care about endianness at all for single-byte data.
    return true;
  } else if (bitWidth % 8 != 0) {
    // Any non-byte aligned bit width is stored byte aligned.
    return false;
  } else if (endian != llvm::support::endian::system_endianness()) {
    // Can't use raw data if the endianness of the system doesn't match the
    // endianness of the target.
    return false;
  }
  return true;
}

// Appends the raw bytes of |value| in the given endianness to |buffer|.
// Non-byte-aligned types are rounded up to the next power of two byte-aligned
// bit width (i1 -> i8, i4 -> i8, i17 -> i32, etc).
static LogicalResult serializeAPIntRawData(Location loc, APInt value,
                                           uint64_t bitWidth,
                                           llvm::support::endianness endian,
                                           SmallVectorImpl<char> &buffer) {
  // Round up to 8-bit aligned bytes.
  uint64_t byteAligned = llvm::divideCeil(bitWidth, 8);
  // Round up to the next power of two (unless already a power of two).
  uint64_t byteWidth = llvm::PowerOf2Ceil(byteAligned);
  // Storage is in aligned bytes.
  buffer.resize(byteWidth);
  // Extract up to the declared bit width and pad.
  switch (byteWidth) {
  case 1: {
    uint8_t rawValue = llvm::support::endian::byte_swap<uint8_t>(
        value.extractBitsAsZExtValue(bitWidth, 0), endian);
    std::memcpy(buffer.data(), &rawValue, sizeof(rawValue));
    return success();
  }
  case 2: {
    uint16_t rawValue = llvm::support::endian::byte_swap<uint16_t>(
        value.extractBitsAsZExtValue(bitWidth, 0), endian);
    std::memcpy(buffer.data(), &rawValue, sizeof(rawValue));
    return success();
  }
  case 4: {
    uint32_t rawValue = llvm::support::endian::byte_swap<uint32_t>(
        value.extractBitsAsZExtValue(bitWidth, 0), endian);
    std::memcpy(buffer.data(), &rawValue, sizeof(rawValue));
    return success();
  }
  case 8: {
    uint64_t rawValue = llvm::support::endian::byte_swap<uint64_t>(
        value.extractBitsAsZExtValue(bitWidth, 0), endian);
    std::memcpy(buffer.data(), &rawValue, sizeof(rawValue));
    return success();
  }
  default:
    return emitError(loc) << "unhandled byte width in serializeAPIntRawData: "
                          << byteWidth;
  }
}

// Appends the raw bytes of |value| in the given endianness to |buffer|.
static LogicalResult serializeAPFloatRawData(Location loc, APFloat value,
                                             size_t bitWidth,
                                             llvm::support::endianness endian,
                                             SmallVectorImpl<char> &buffer) {
  buffer.resize(bitWidth / 8);
  switch (bitWidth) {
  case 8: {
    uint8_t rawValue = llvm::support::endian::byte_swap<uint8_t>(
        value.bitcastToAPInt().extractBitsAsZExtValue(8, 0), endian);
    std::memcpy(buffer.data(), &rawValue, sizeof(rawValue));
    return success();
  }
  case 16: {
    uint16_t rawValue = llvm::support::endian::byte_swap<uint16_t>(
        value.bitcastToAPInt().extractBitsAsZExtValue(16, 0), endian);
    std::memcpy(buffer.data(), &rawValue, sizeof(rawValue));
    return success();
  }
  case 32: {
    float rawValue =
        llvm::support::endian::byte_swap<float>(value.convertToFloat(), endian);
    std::memcpy(buffer.data(), &rawValue, sizeof(rawValue));
    return success();
  }
  case 64: {
    double rawValue = llvm::support::endian::byte_swap<double>(
        value.convertToDouble(), endian);
    std::memcpy(buffer.data(), &rawValue, sizeof(rawValue));
    return success();
  }
  default:
    return emitError(loc) << "unhandled bitWidth in serializeAPFloatRawData: "
                          << bitWidth;
  }
}

// Serializes |count| copies of |splatAttr| to |os|.
// Significantly faster than the generic ElementsAttr path that needs to perform
// conversion of the same splat value |count| times.
static LogicalResult serializeSplatValue(Location loc, Attribute splatAttr,
                                         int64_t count,
                                         llvm::support::endianness endian,
                                         llvm::raw_ostream &os) {
  // Get the encoded byte contents of the splat element.
  SmallVector<char> elementBuffer;
  if (auto attr = llvm::dyn_cast<SerializableAttrInterface>(splatAttr)) {
    if (failed(attr.serializeToVector(loc, endian, elementBuffer))) {
      return failure();
    }
  } else if (auto attr = llvm::dyn_cast<IntegerAttr>(splatAttr)) {
    if (failed(serializeAPIntRawData(loc, attr.getValue(),
                                     attr.getType().getIntOrFloatBitWidth(),
                                     endian, elementBuffer))) {
      return failure();
    }
  } else if (auto attr = llvm::dyn_cast<FloatAttr>(splatAttr)) {
    if (failed(serializeAPFloatRawData(loc, attr.getValue(),
                                       attr.getType().getIntOrFloatBitWidth(),
                                       endian, elementBuffer))) {
      return failure();
    }
  } else {
    assert(false && "unhandled serializable splat value");
    return failure();
  }

  // Write the splat value contents |count| times.
  for (int64_t i = 0; i < count; ++i) {
    os.write(elementBuffer.data(), elementBuffer.size());
  }
  return success();
}

// Serializes the raw data of the given |elementsAttr| to |os|.
// Assumes that the caller knows what they are doing; the raw data must be in
// the expected endianness and be densely packed.
static LogicalResult serializeRawData(Location loc,
                                      DenseElementsAttr elementsAttr,
                                      llvm::raw_ostream &os) {
  auto rawData = elementsAttr.getRawData();
  os.write(rawData.data(), rawData.size());
  return success();
}

// Serializes the raw data of the given |resourceElementsAttr| to |os|.
// Assumes that the caller knows what they are doing; the raw data must be in
// the expected endianness and be densely packed.
static LogicalResult serializeResourceRawData(Location loc,
                                      DenseResourceElementsAttr resourceElementsAttr,
                                      llvm::raw_ostream &os) {
  auto rawData = resourceElementsAttr.getRawHandle().getBlob()->getData();
  os.write(rawData.data(), rawData.size());
  return success();
}

// Stream writer that supports bit packing.
// In the initial state the writer starts at a byte-aligned offset 0 and as new
// values of |logicalBitWidth| are written they will be appended in
// |physicalBitWidth| chunks in the specified endianness. When completed any
// additional bits remaining in the last |physicalBitWidth| chunk will be padded
// with zeros.
//
// Note that the logical bit width may not evenly divide the physical bit width:
// a logical width of 3 and a physical width of 8 will always include 2 bits of
// zero padding.
template <typename physicalType,
          unsigned physicalBitWidth = sizeof(physicalType) * 8>
class PackedWriter {
public:
  explicit PackedWriter(unsigned logicalBitWidth,
                        llvm::support::endianness endian, llvm::raw_ostream &os)
      : logicalBitWidth(logicalBitWidth), endian(endian), os(os) {}

  void write(const uint64_t value) {
    if (bitOffset + logicalBitWidth > physicalBitWidth)
      flush();
    physicalBuffer |= value << bitOffset;
    bitOffset += logicalBitWidth;
  }

  void flush() {
    if (bitOffset == 0)
      return;
    physicalType physicalValue =
        llvm::support::endian::byte_swap<physicalType>(physicalBuffer, endian);
    os.write((const char *)&physicalValue, sizeof(physicalValue));
    physicalBuffer = 0;
    bitOffset = 0;
  }

private:
  const unsigned logicalBitWidth;
  const llvm::support::endianness endian;
  llvm::raw_ostream &os;
  unsigned bitOffset = 0;
  physicalType physicalBuffer = 0;
};

static LogicalResult
serializeSubByteIntegerElements(Location loc, DenseIntElementsAttr attr,
                                llvm::support::endianness endian,
                                llvm::raw_ostream &os) {
  const unsigned logicalBitWidth =
      attr.getElementType().getIntOrFloatBitWidth();
  // Round up to the next power of two (unless already a power of two) of the
  // 8-bit aligned logical bit width.
  const unsigned physicalBitWidth =
      getTypePhysicalStorageBitWidth(attr.getElementType());
  switch (physicalBitWidth) {
  case 8: {
    PackedWriter<uint8_t> writer(logicalBitWidth, endian, os);
    for (const auto &value : attr.getValues<APInt>()) {
      writer.write(value.getZExtValue());
    }
    writer.flush();
    return success();
  }
  case 16: {
    PackedWriter<uint16_t> writer(logicalBitWidth, endian, os);
    for (const auto &value : attr.getValues<APInt>()) {
      writer.write(value.getZExtValue());
    }
    writer.flush();
    return success();
  }
  case 32: {
    PackedWriter<uint32_t> writer(logicalBitWidth, endian, os);
    for (const auto &value : attr.getValues<APInt>()) {
      writer.write(value.getZExtValue());
    }
    writer.flush();
    return success();
  }
  case 64: {
    PackedWriter<uint64_t> writer(logicalBitWidth, endian, os);
    for (const auto &value : attr.getValues<APInt>()) {
      writer.write(value.getZExtValue());
    }
    writer.flush();
    return success();
  }
  default:
    return emitError(loc) << "unhandled packed integer physical bit width "
                          << physicalBitWidth << " for type " << attr.getType();
  }
}

template <typename elementType, unsigned numBits = sizeof(elementType) * 8>
static LogicalResult
serializeGenericIntegerElements(DenseIntElementsAttr attr,
                                llvm::support::endianness endian,
                                llvm::raw_ostream &os) {
  for (const APInt &value : attr.getValues<APInt>()) {
    elementType rawValue = llvm::support::endian::byte_swap<elementType>(
        value.extractBitsAsZExtValue(numBits, 0), endian);
    os.write((const char *)&rawValue, sizeof(rawValue));
  }
  return success();
}

template <typename elementType, unsigned numBits = sizeof(elementType) * 8>
static LogicalResult
serializeGenericFloatElements(DenseFPElementsAttr attr,
                              llvm::support::endianness endian,
                              llvm::raw_ostream &os) {
  for (const APFloat &value : attr.getValues<APFloat>()) {
    elementType rawValue = llvm::support::endian::byte_swap<elementType>(
        value.bitcastToAPInt().extractBitsAsZExtValue(numBits, 0), endian);
    os.write((const char *)&rawValue, sizeof(rawValue));
  }
  return success();
}


// Expands 8-values per byte raw data from DenseIntElementsAttr to 0/1 byte
// values in the output.
static LogicalResult serializeBitIntegerValuesAsBytes(DenseIntElementsAttr attr,
                                                      llvm::raw_ostream &os) {
  auto rawData = attr.getRawData();
  char bytes[8];
  for (size_t i = 0; i < rawData.size(); ++i) {
    int32_t bits = rawData[i];
    bytes[i * 8 + 0] = bits & 0x1;
    bytes[i * 8 + 1] = (bits & 0x2) >> 1;
    bytes[i * 8 + 2] = (bits & 0x4) >> 2;
    bytes[i * 8 + 3] = (bits & 0x8) >> 3;
    bytes[i * 8 + 4] = (bits & 0x10) >> 4;
    bytes[i * 8 + 5] = (bits & 0x20) >> 5;
    bytes[i * 8 + 6] = (bits & 0x40) >> 6;
    bytes[i * 8 + 7] = (bits & 0x80) >> 7;
  }
  os.write(bytes, sizeof(bytes));
  return success();
}

// Performs slow generic serialization of all of the elements in |elementsAttr|.
// Respects the target |endian| setting, performing byte swaps if required.
static LogicalResult
serializeGenericElementData(Location loc, DenseElementsAttr elementsAttr,
                            llvm::support::endianness endian,
                            llvm::raw_ostream &os) {
  if (auto attr = llvm::dyn_cast<DenseIntElementsAttr>(elementsAttr)) {
    // Don't hoist bitWidth given `getElementTypeBitWidth()` asserts if the
    // element type is not integer or floating-point.
    unsigned bitWidth = attr.getType().getElementTypeBitWidth();
    switch (bitWidth) {
    case 1: {
      // NOTE: i1 is treated as i8 in a lot of places in MLIR/IREE and will need
      // a larger cleanup to serialize as a sub-byte value like the others.
      // In this one case, we know that DenseIntElementsAttr has been
      // prematurely optimized to densely pack bit values ala std::vector<bool>.
      // Further, it packs them linearly, regardless of shape, so we have to
      // do a simple expansion.
      return serializeBitIntegerValuesAsBytes(attr, os);
    }
    case 8:
      return serializeRawData(loc, attr, os);
    case 16:
      return serializeGenericIntegerElements<uint16_t>(attr, endian, os);
    case 32:
      return serializeGenericIntegerElements<uint32_t>(attr, endian, os);
    case 64:
      return serializeGenericIntegerElements<uint64_t>(attr, endian, os);
    default:
      if (bitWidth < 64) {
        // Special case for bit-packing of sub-byte aligned types.
        // This could be extended to handle larger widths (i33, etc) but they
        // are rare today.
        return serializeSubByteIntegerElements(loc, attr, endian, os);
      }
      return emitError(loc)
             << "unhandled integer element bit width " << bitWidth
             << " for type " << elementsAttr.getType();
    }
  } else if (auto attr = llvm::dyn_cast<DenseFPElementsAttr>(elementsAttr)) {
    // Don't hoist bitWidth given `getElementTypeBitWidth()` asserts if the
    // element type is not integer or floating-point.
    unsigned bitWidth = attr.getType().getElementTypeBitWidth();
    switch (bitWidth) {
    case 8:
      // TODO(benvanik): see if serializeRawData works for f8 types.
      return serializeGenericFloatElements<uint8_t>(attr, endian, os);
    case 16:
      return serializeGenericFloatElements<uint16_t>(attr, endian, os);
    case 32:
      return serializeGenericFloatElements<uint32_t>(attr, endian, os);
    case 64:
      return serializeGenericFloatElements<uint64_t>(attr, endian, os);
    default:
      return emitError(loc) << "unhandled float element bit width " << bitWidth
                            << " for type " << elementsAttr.getType();
    }
  }
  return emitError(loc) << "unhandled constant type " << elementsAttr.getType();
}

// Performs slow generic serialization of all of the elements in |resourceElementsAttr|.
// Respects the target |endian| setting, performing byte swaps if required.
static LogicalResult
serializeGenericResourceElementData(Location loc, DenseResourceElementsAttr resourceElementsAttr,
                            llvm::support::endianness endian,
                            llvm::raw_ostream &os) {
  if (llvm::isa<IntegerType>(resourceElementsAttr.getType().getElementType())) {
    // Don't hoist bitWidth given `getElementTypeBitWidth()` asserts if the
    // element type is not integer or floating-point.
    // TODO(aviator19941): test i1
    unsigned bitWidth = resourceElementsAttr.getType().getElementTypeBitWidth();
    switch (bitWidth) {
    case 8:
      return serializeResourceRawData(loc, resourceElementsAttr, os);
    case 16:
      return serializeResourceRawData(loc, resourceElementsAttr, os);
    case 32:
      return serializeResourceRawData(loc, resourceElementsAttr, os);
    case 64:
      return serializeResourceRawData(loc, resourceElementsAttr, os);
    default:
      return emitError(loc)
             << "unhandled integer element bit width " << bitWidth
             << " for type " << resourceElementsAttr.getType();
    }
  }
  else if (llvm::isa<FloatType>(resourceElementsAttr.getType().getElementType())) {
    // Don't hoist bitWidth given `getElementTypeBitWidth()` asserts if the
    // element type is not integer or floating-point.
    // TODO(saienduri): implement float64 support (not neccesary now)
    unsigned bitWidth = resourceElementsAttr.getType().getElementTypeBitWidth();
    switch (bitWidth) {
    case 16:
      return serializeResourceRawData(loc, resourceElementsAttr, os);
    case 32:
      return serializeResourceRawData(loc, resourceElementsAttr, os);
    default:
      return emitError(loc) << "unhandled float element bit width " << bitWidth
                            << " for type " << resourceElementsAttr.getType();
    }
  }
  return emitError(loc) << "unhandled constant type " << resourceElementsAttr.getType();
}

//===----------------------------------------------------------------------===//
// #util.byte_pattern
//===----------------------------------------------------------------------===//

int64_t BytePatternAttr::getStorageSize() const {
  if (auto shapedType = getType().dyn_cast<ShapedType>()) {
    return IREE::Util::getRoundedPhysicalStorageSize(shapedType);
  } else {
    return IREE::Util::getTypePhysicalStorageBitWidth(getType());
  }
}

LogicalResult
BytePatternAttr::serializeToBuffer(Location loc,
                                   llvm::support::endianness endian,
                                   ArrayRef<char> buffer) const {
  const uint8_t byte = static_cast<uint8_t>(getPattern() % 256);
  std::memset(const_cast<char *>(buffer.data()), byte, buffer.size());
  return success();
}

LogicalResult
BytePatternAttr::serializeToStream(Location loc,
                                   llvm::support::endianness endian,
                                   llvm::raw_ostream &os) const {
  const uint8_t byte = static_cast<uint8_t>(getPattern() % 256);
  const char bytes[256] = {static_cast<char>(byte)};
  int64_t remaining = getStorageSize();
  while (remaining) {
    int64_t write_length =
        std::min(remaining, static_cast<int64_t>(sizeof(bytes)));
    os.write(bytes, static_cast<size_t>(write_length));
    remaining -= write_length;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// #util.byte_range
//===----------------------------------------------------------------------===//

Attribute ByteRangeAttr::parse(AsmParser &p, Type type) {
  if (failed(p.parseLess()))
    return {};

  // TODO(benvanik): support the range syntax; the dialect asm parser fights
  // with it though by checking for proper []/() nesting.

  // Try first the range style: byte_range<[start..end)>
  bool startInclusive;
  if (succeeded(p.parseOptionalLSquare())) { // [...
    startInclusive = true;
  } else if (succeeded(p.parseOptionalLParen())) { // (...
    startInclusive = false;
  } else {
    // byte_range<offset, length>
    int64_t offset;
    int64_t length;
    if (failed(p.parseInteger(offset)) || failed(p.parseComma()) ||
        failed(p.parseInteger(length)) || failed(p.parseGreater())) {
      return {};
    }
    return get(p.getContext(), offset, length);
  }

  int64_t start;
  int64_t end;
  if (failed(p.parseInteger(start)) || failed(p.parseKeyword("to")) ||
      failed(p.parseInteger(end))) {
    return {};
  }

  bool endInclusive;
  if (succeeded(p.parseOptionalRSquare())) { // ...]
    endInclusive = true;
  } else if (succeeded(p.parseOptionalRParen())) { // ...)
    endInclusive = false;
  } else {
    p.emitError(p.getCurrentLocation()) << "expected ] or ) to end range";
    return {};
  }

  if (failed(p.parseGreater()))
    return {};

  start = startInclusive ? start : start + 1;
  end = endInclusive ? end : end - 1;

  int64_t offset = start;
  int64_t length = end - start;
  return get(p.getContext(), offset, length);
}

void ByteRangeAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  os << getOffset();
  os << ", ";
  os << getLength();
  os << ">";
}

//===----------------------------------------------------------------------===//
// #util.composite
//===----------------------------------------------------------------------===//

// static
CompositeAttr CompositeAttr::get(MLIRContext *context,
                                 ArrayRef<Attribute> valueAttrs) {
  int64_t calculatedLength = 0;
  for (auto valueAttr : valueAttrs) {
    if (auto serializableAttr =
            llvm::dyn_cast<SerializableAttrInterface>(valueAttr)) {
      calculatedLength += serializableAttr.getStorageSize();
    } else {
      return {};
    }
  }
  return get(context, calculatedLength, ArrayAttr::get(context, valueAttrs));
}

// static
LogicalResult
CompositeAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                      int64_t totalLength, ArrayAttr valueAttrs) {
  int64_t calculatedLength = 0;
  for (auto valueAttr : valueAttrs) {
    if (auto serializableAttr =
            llvm::dyn_cast<SerializableAttrInterface>(valueAttr)) {
      calculatedLength += serializableAttr.getStorageSize();
    } else {
      return emitError() << "value is not serializable: " << valueAttr;
    }
  }
  if (calculatedLength != totalLength) {
    return emitError() << "total length mismatch: calculated size of values is "
                       << calculatedLength << " but composite reports "
                       << totalLength;
  }
  return success();
}

Attribute CompositeAttr::parse(AsmParser &parser, Type type) {
  SmallVector<int64_t> dims;
  if (failed(parser.parseLess()) ||
      failed(parser.parseDimensionList(dims, /*allowDynamic=*/false)) ||
      dims.size() != 1) {
    parser.emitError(parser.getCurrentLocation(), "invalid length specifier");
    return {};
  }
  int64_t totalLength = dims.front();

  Type elementType;
  if (failed(parser.parseType(elementType)) || !elementType.isInteger(8) ||
      failed(parser.parseComma()) || failed(parser.parseLSquare())) {
    parser.emitError(parser.getCurrentLocation(),
                     "invalid type specifier; expected i8");
    return {};
  }

  SmallVector<Attribute> valueAttrs;
  while (failed(parser.parseOptionalRSquare())) {
    Attribute valueAttr;
    if (failed(parser.parseAttribute(valueAttr))) {
      parser.emitError(parser.getCurrentLocation(), "invalid value attribute");
    }
    valueAttrs.push_back(valueAttr);
    if (failed(parser.parseOptionalComma())) {
      // List termination with no trailing comma.
      if (failed(parser.parseRSquare())) {
        parser.emitError(parser.getCurrentLocation(),
                         "unterminated value list");
        return {};
      }
      break;
    }
  }
  if (failed(parser.parseGreater())) {
    parser.emitError(parser.getCurrentLocation(), "unterminated value list");
    return {};
  }
  return get(parser.getContext(), totalLength,
             ArrayAttr::get(parser.getContext(), valueAttrs));
}

void CompositeAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<" << getTotalLength() << "xi8, [";
  if (getTotalLength() > 0) {
    os << "\n";
    for (auto valueAttr : getValues()) {
      // NOTE: there's no way to get a context-aware indent on attr printers.
      // We just guess based on what IR is seen the most in text form.
      os << "    ";
      p.printAttribute(valueAttr);
      os << ",\n";
    }
  }
  os << "]>";
}

int64_t CompositeAttr::getStorageSize() const { return getTotalLength(); }

LogicalResult CompositeAttr::serializeToBuffer(Location loc,
                                               llvm::support::endianness endian,
                                               ArrayRef<char> buffer) const {
  raw_inplace_ostream os(buffer);
  return serializeToStream(loc, endian, os);
}

LogicalResult CompositeAttr::serializeToStream(Location loc,
                                               llvm::support::endianness endian,
                                               llvm::raw_ostream &os) const {
  for (auto valueAttr : getValues()) {
    auto serializableAttr =
        llvm::dyn_cast<SerializableAttrInterface>(valueAttr);
    if (!serializableAttr) {
      return emitError(loc)
             << "unable to serialize a non-serializable attribute: "
             << valueAttr;
    }
    if (failed(serializableAttr.serializeToStream(loc, endian, os))) {
      return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SerializableAttrInterface implementations
//===----------------------------------------------------------------------===//

// External interface applied to ElementsAttrs so that we can serialize them to
// byte buffers.
struct SerializableDenseElementsAttrModel
    : public SerializableAttrInterface::ExternalModel<
          SerializableDenseElementsAttrModel, DenseIntOrFPElementsAttr> {
  int64_t getStorageSize(Attribute baseAttr) const {
    auto attr = llvm::cast<ElementsAttr>(baseAttr);
    return IREE::Util::getRoundedPhysicalStorageSize(
        attr.getNumElements(),
        cast<ShapedType>(attr.getType()).getElementType());
  }

  LogicalResult serializeToVector(Attribute baseAttr, Location loc,
                                  llvm::support::endianness endian,
                                  SmallVectorImpl<char> &buffer) const {
    buffer.resize(getStorageSize(baseAttr));
    return serializeToBuffer(baseAttr, loc, endian, buffer);
  }

  LogicalResult serializeToBuffer(Attribute baseAttr, Location loc,
                                  llvm::support::endianness endian,
                                  ArrayRef<char> buffer) const {
    raw_inplace_ostream os(buffer);
    return serializeToStream(baseAttr, loc, endian, os);
  }

  LogicalResult serializeToStream(Attribute baseAttr, Location loc,
                                  llvm::support::endianness endian,
                                  llvm::raw_ostream &os) const {
    // NOTE: not all ostream implementations handle this but for buffering ones
    // it can really help.
    os.reserveExtraSpace(getStorageSize(baseAttr));

    auto elementsAttr = llvm::cast<DenseElementsAttr>(baseAttr);
    if (elementsAttr.isSplat()) {
      // Fast-path for splat (no need to convert the value a bunch).
      return serializeSplatValue(loc, elementsAttr.getSplatValue<Attribute>(),
                                 elementsAttr.getNumElements(), endian, os);
    }

    if (canUseRawData(elementsAttr, endian)) {
      // Fast-path for bulk data copies that don't require endianness handling.
      // This relies on DenseElementsAttr storing 8-bit values as 8-bit values;
      // other sized types are stored in an opaque format.
      return serializeRawData(loc, elementsAttr, os);
    } else {
      // Slow-path that performs expensive conversion.
      return serializeGenericElementData(loc, elementsAttr, endian, os);
    }
  }
};

// External interface applied to ElementsAttrs so that we can serialize them to
// byte buffers.
struct SerializableDenseResourceElementsAttrModel
    : public SerializableAttrInterface::ExternalModel<
          SerializableDenseResourceElementsAttrModel,
          DenseResourceElementsAttr> {
  int64_t getStorageSize(Attribute baseAttr) const {
    auto attr = llvm::cast<DenseResourceElementsAttr>(baseAttr);
    return IREE::Util::getRoundedPhysicalStorageSize(
        attr.getNumElements(), attr.getType().getElementType());
  }

  LogicalResult serializeToVector(Attribute baseAttr, Location loc,
                                  llvm::support::endianness endian,
                                  SmallVectorImpl<char> &buffer) const {
    buffer.resize(getStorageSize(baseAttr));
    return serializeToBuffer(baseAttr, loc, endian, buffer);
  }

  LogicalResult serializeToBuffer(Attribute baseAttr, Location loc,
                                  llvm::support::endianness endian,
                                  ArrayRef<char> buffer) const {
    raw_inplace_ostream os(buffer);
    return serializeToStream(baseAttr, loc, endian, os);
  }

  LogicalResult serializeToStream(Attribute baseAttr, Location loc,
                                  llvm::support::endianness endian,
                                  llvm::raw_ostream &os) const {
    auto attr = llvm::cast<DenseResourceElementsAttr>(baseAttr);
    auto handle = attr.getRawHandle();

    // Special testing path for elided attributes. We want this to be an
    // error in normal circumstances as the output will produce garbage
    // results if executed but it can be useful when building reproducers.
    if (handle.getKey() == "__elided__") {
      if (!clZeroFillElidedAttrs) {
        return mlir::emitError(loc)
               << "elided attributes cannot be serialized; provide non-elided "
                  "values or pass --iree-util-zero-fill-elided-attrs for "
                  "testing and expect invalid execution results";
      }
      os.write_zeros(getStorageSize(baseAttr));
      return success();
    } else {
      os.reserveExtraSpace(getStorageSize(baseAttr));
      return serializeGenericResourceElementData(loc, attr, endian, os);
    }

    return mlir::emitError(loc)
           << "DenseResourceElementsAttr not yet supported for serialization";
  }
};

// External interface applied to string attrs so that we can serialize them to
// byte buffers. We don't include NUL terminators as it's 2022.
struct SerializableStringAttrModel
    : public SerializableAttrInterface::ExternalModel<
          SerializableStringAttrModel, StringAttr> {
  int64_t getStorageSize(Attribute baseAttr) const {
    auto attr = llvm::cast<StringAttr>(baseAttr);
    return attr.getValue().size();
  }

  LogicalResult serializeToVector(Attribute baseAttr, Location loc,
                                  llvm::support::endianness endian,
                                  SmallVectorImpl<char> &buffer) const {
    buffer.resize(getStorageSize(baseAttr));
    return serializeToBuffer(baseAttr, loc, endian, buffer);
  }

  LogicalResult serializeToBuffer(Attribute baseAttr, Location loc,
                                  llvm::support::endianness endian,
                                  ArrayRef<char> buffer) const {
    raw_inplace_ostream os(buffer);
    return serializeToStream(baseAttr, loc, endian, os);
  }

  LogicalResult serializeToStream(Attribute baseAttr, Location loc,
                                  llvm::support::endianness endian,
                                  llvm::raw_ostream &os) const {
    // NOTE: not all ostream implementations handle this but for buffering ones
    // it can really help.
    os.reserveExtraSpace(getStorageSize(baseAttr));
    auto stringAttr = llvm::cast<StringAttr>(baseAttr);
    os.write(stringAttr.data(), stringAttr.size());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// IREE::Util::UtilDialect
//===----------------------------------------------------------------------===//

// At the end so it can use functions above:
#include "iree/compiler/Dialect/Util/IR/UtilAttrInterfaces.cpp.inc"

void UtilDialect::registerAttributes() {
  // Register command line flags:
  (void)clZeroFillElidedAttrs;

  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Dialect/Util/IR/UtilAttrs.cpp.inc" // IWYU pragma: keep
      >();

  // NOTE: we only handle dense elements today; sparse will require a separate
  // serialization mechanism and may be something we want to handle much higher
  // up in the stack - things that end up here are generally already in a target
  // encoding.
  auto &context = *getContext();
  DenseIntElementsAttr::attachInterface<SerializableDenseElementsAttrModel>(
      context);
  DenseFPElementsAttr::attachInterface<SerializableDenseElementsAttrModel>(
      context);
  DenseResourceElementsAttr::attachInterface<
      SerializableDenseResourceElementsAttrModel>(context);
  StringAttr::attachInterface<SerializableStringAttrModel>(context);
}

} // namespace Util
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
