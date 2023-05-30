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
#include "iree/compiler/Dialect/Util/IR/UtilAttrs.cpp.inc"  // IWYU pragma: keep
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

  int32_t bitwidth = getTypeBitWidth(elementType);
  if (bitwidth == 8) {
    // Don't care about endianness at all for single-byte data.
    return true;
  } else if (bitwidth % 8 != 0) {
    // Any non-byte aligned bitwidth is stored byte aligned.
    return false;
  } else if (endian != llvm::support::endian::system_endianness()) {
    // Can't use raw data if the endianness of the system doesn't match the
    // endianness of the target.
    return false;
  }
  return true;
}

// Appends the raw bytes of |value| in the given endianness to |buffer|.
static LogicalResult getAPIntRawData(APInt value, size_t bitWidth,
                                     llvm::support::endianness endian,
                                     SmallVectorImpl<char> &buffer) {
  buffer.resize(bitWidth / 8);
  switch (bitWidth) {
    case 8: {
      uint8_t rawValue = llvm::support::endian::byte_swap<uint8_t>(
          value.extractBitsAsZExtValue(8, 0), endian);
      std::memcpy(buffer.data(), &rawValue, sizeof(rawValue));
      return success();
    }
    case 16: {
      uint16_t rawValue = llvm::support::endian::byte_swap<uint16_t>(
          value.extractBitsAsZExtValue(16, 0), endian);
      std::memcpy(buffer.data(), &rawValue, sizeof(rawValue));
      return success();
    }
    case 32: {
      uint32_t rawValue = llvm::support::endian::byte_swap<uint32_t>(
          value.extractBitsAsZExtValue(32, 0), endian);
      std::memcpy(buffer.data(), &rawValue, sizeof(rawValue));
      return success();
    }
    case 64: {
      uint64_t rawValue = llvm::support::endian::byte_swap<uint64_t>(
          value.extractBitsAsZExtValue(64, 0), endian);
      std::memcpy(buffer.data(), &rawValue, sizeof(rawValue));
      return success();
    }
    default:
      return failure();
  }
}

// Appends the raw bytes of |value| in the given endianness to |buffer|.
static LogicalResult getAPFloatRawData(APFloat value, size_t bitWidth,
                                       llvm::support::endianness endian,
                                       SmallVectorImpl<char> &buffer) {
  buffer.resize(bitWidth / 8);
  switch (bitWidth) {
    case 16: {
      uint16_t rawValue = llvm::support::endian::byte_swap<uint16_t>(
          value.bitcastToAPInt().extractBitsAsZExtValue(16, 0), endian);
      std::memcpy(buffer.data(), &rawValue, sizeof(rawValue));
      return success();
    }
    case 32: {
      float rawValue = llvm::support::endian::byte_swap<float>(
          value.convertToFloat(), endian);
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
      return failure();
  }
}

// Serializes |count| copies of |splatAttr| to |os|.
// Significantly faster than the generic ElementsAttr path that needs to perform
// conversion of the same splat value |count| times.
static LogicalResult serializeSplatValue(Attribute splatAttr, int64_t count,
                                         llvm::support::endianness endian,
                                         llvm::raw_ostream &os) {
  // Get the encoded byte contents of the splat element.
  SmallVector<char> elementBuffer;
  if (auto attr =
          llvm::dyn_cast<IREE::Util::SerializableAttrInterface>(splatAttr)) {
    if (failed(attr.serializeToVector(endian, elementBuffer))) {
      return failure();
    }
  } else if (auto attr = llvm::dyn_cast<IntegerAttr>(splatAttr)) {
    if (failed(getAPIntRawData(attr.getValue(),
                               attr.getType().getIntOrFloatBitWidth(), endian,
                               elementBuffer))) {
      return failure();
    }
  } else if (auto attr = llvm::dyn_cast<FloatAttr>(splatAttr)) {
    if (failed(getAPFloatRawData(attr.getValue(),
                                 attr.getType().getIntOrFloatBitWidth(), endian,
                                 elementBuffer))) {
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
static LogicalResult serializeRawData(DenseElementsAttr elementsAttr,
                                      llvm::raw_ostream &os) {
  auto rawData = elementsAttr.getRawData();
  os.write(rawData.data(), rawData.size());
  return success();
}

template <typename elementType, unsigned numBits = sizeof(elementType) * 8>
static LogicalResult serializeGenericIntElements(
    DenseIntElementsAttr attr, llvm::support::endianness endian,
    llvm::raw_ostream &os) {
  for (const APInt &value : attr.getValues<APInt>()) {
    elementType rawValue = llvm::support::endian::byte_swap<elementType>(
        value.extractBitsAsZExtValue(numBits, 0), endian);
    os.write((char *)&rawValue, sizeof(rawValue));
  }
  return success();
}

static LogicalResult serializeGenericF16Elements(
    DenseFPElementsAttr attr, llvm::support::endianness endian,
    llvm::raw_ostream &os) {
  for (const APFloat &value : attr.getValues<APFloat>()) {
    uint16_t rawValue = llvm::support::endian::byte_swap<uint16_t>(
        value.bitcastToAPInt().extractBitsAsZExtValue(16, 0), endian);
    os.write((char *)&rawValue, sizeof(rawValue));
  }
  return success();
}

static LogicalResult serializeGenericF32Elements(
    DenseFPElementsAttr attr, llvm::support::endianness endian,
    llvm::raw_ostream &os) {
  for (const APFloat &value : attr.getValues<APFloat>()) {
    float rawValue =
        llvm::support::endian::byte_swap<float>(value.convertToFloat(), endian);
    os.write((char *)&rawValue, sizeof(rawValue));
  }
  return success();
}

static LogicalResult serializeGenericF64Elements(
    DenseFPElementsAttr attr, llvm::support::endianness endian,
    llvm::raw_ostream &os) {
  for (const APFloat &value : attr.getValues<APFloat>()) {
    double rawValue = llvm::support::endian::byte_swap<double>(
        value.convertToDouble(), endian);
    os.write((char *)&rawValue, sizeof(rawValue));
  }
  return success();
}

// Performs slow generic serialization of all of the elements in |elementsAttr|.
// Respects the target |endian| setting, performing byte swaps if required.
static LogicalResult serializeGenericElementData(
    DenseElementsAttr elementsAttr, llvm::support::endianness endian,
    llvm::raw_ostream &os) {
  if (auto attr = llvm::dyn_cast<DenseIntElementsAttr>(elementsAttr)) {
    // Don't hoist |bitwidth| given `getElementTypeBitWidth()` asserts if the
    // element type is not integer or floating-point.
    int32_t bitwidth = attr.getType().getElementTypeBitWidth();
    switch (bitwidth) {
      case 8:
        return serializeRawData(attr, os);
      case 16:
        return serializeGenericIntElements<uint16_t>(attr, endian, os);
      case 32:
        return serializeGenericIntElements<uint32_t>(attr, endian, os);
      case 64:
        return serializeGenericIntElements<uint64_t>(attr, endian, os);
      default:
        return emitError(UnknownLoc::get(elementsAttr.getContext()))
               << "unhandled integer element bitwidth " << bitwidth
               << " for type " << elementsAttr.getType();
    }
  } else if (auto attr = llvm::dyn_cast<DenseFPElementsAttr>(elementsAttr)) {
    // Don't hoist |bitwidth| given `getElementTypeBitWidth()` asserts if the
    // element type is not integer or floating-point.
    int32_t bitwidth = attr.getType().getElementTypeBitWidth();
    switch (bitwidth) {
      case 16:
        return serializeGenericF16Elements(attr, endian, os);
      case 32:
        return serializeGenericF32Elements(attr, endian, os);
      case 64:
        return serializeGenericF64Elements(attr, endian, os);
      default:
        return emitError(UnknownLoc::get(elementsAttr.getContext()))
               << "unhandled float element bitwidth " << bitwidth
               << " for type " << elementsAttr.getType();
    }
  }
  return emitError(UnknownLoc::get(elementsAttr.getContext()))
         << "unhandled constant type " << elementsAttr.getType();
}

//===----------------------------------------------------------------------===//
// Buffer attributes
//===----------------------------------------------------------------------===//

Attribute ByteRangeAttr::parse(AsmParser &p, Type type) {
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
  if (succeeded(p.parseOptionalRSquare())) {  // ...]
    endInclusive = true;
  } else if (succeeded(p.parseOptionalRParen())) {  // ...)
    endInclusive = false;
  } else {
    p.emitError(p.getCurrentLocation()) << "expected ] or ) to end range";
    return {};
  }

  if (failed(p.parseGreater())) return {};

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
LogicalResult CompositeAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, int64_t totalLength,
    ArrayAttr valueAttrs) {
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

LogicalResult CompositeAttr::serializeToBuffer(llvm::support::endianness endian,
                                               ArrayRef<char> buffer) const {
  raw_inplace_ostream os(buffer);
  return serializeToStream(endian, os);
}

LogicalResult CompositeAttr::serializeToStream(llvm::support::endianness endian,
                                               llvm::raw_ostream &os) const {
  for (auto valueAttr : getValues()) {
    auto serializableAttr =
        llvm::dyn_cast<SerializableAttrInterface>(valueAttr);
    if (!serializableAttr) {
      llvm::errs() << "unable to serialize a non-serializable attribute: "
                   << valueAttr << "\n";
      return failure();
    }
    if (failed(serializableAttr.serializeToStream(endian, os))) {
      return failure();
    }
  }
  return success();
}

// External interface applied to ElementsAttrs so that we can serialize them to
// byte buffers.
struct SerializableDenseElementsAttrModel
    : public SerializableAttrInterface::ExternalModel<
          SerializableDenseElementsAttrModel, DenseIntOrFPElementsAttr> {
  int64_t getStorageSize(Attribute baseAttr) const {
    auto attr = llvm::cast<ElementsAttr>(baseAttr);
    return attr.getNumElements() *
           IREE::Util::getRoundedElementByteWidth(
               cast<ShapedType>(attr.getType()).getElementType());
  }

  LogicalResult serializeToVector(Attribute baseAttr,
                                  llvm::support::endianness endian,
                                  SmallVectorImpl<char> &buffer) const {
    buffer.resize(getStorageSize(baseAttr));
    return serializeToBuffer(baseAttr, endian, buffer);
  }

  LogicalResult serializeToBuffer(Attribute baseAttr,
                                  llvm::support::endianness endian,
                                  ArrayRef<char> buffer) const {
    raw_inplace_ostream os(buffer);
    return serializeToStream(baseAttr, endian, os);
  }

  LogicalResult serializeToStream(Attribute baseAttr,
                                  llvm::support::endianness endian,
                                  llvm::raw_ostream &os) const {
    // NOTE: not all ostream implementations handle this but for buffering ones
    // it can really help.
    os.reserveExtraSpace(getStorageSize(baseAttr));

    auto elementsAttr = llvm::cast<DenseElementsAttr>(baseAttr);
    if (elementsAttr.isSplat()) {
      // Fast-path for splat (no need to convert the value a bunch).
      return serializeSplatValue(elementsAttr.getSplatValue<Attribute>(),
                                 elementsAttr.getNumElements(), endian, os);
    }

    if (canUseRawData(elementsAttr, endian)) {
      // Fast-path for bulk data copies that don't require endianness handling.
      // This relies on DenseElementsAttr storing 8-bit values as 8-bit values;
      // other sized types are stored in an opaque format.
      return serializeRawData(elementsAttr, os);
    } else {
      // Slow-path that performs expensive conversion.
      return serializeGenericElementData(elementsAttr, endian, os);
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
    return attr.getNumElements() * IREE::Util::getRoundedElementByteWidth(
                                       attr.getType().getElementType());
  }

  LogicalResult serializeToVector(Attribute baseAttr,
                                  llvm::support::endianness endian,
                                  SmallVectorImpl<char> &buffer) const {
    buffer.resize(getStorageSize(baseAttr));
    return serializeToBuffer(baseAttr, endian, buffer);
  }

  LogicalResult serializeToBuffer(Attribute baseAttr,
                                  llvm::support::endianness endian,
                                  ArrayRef<char> buffer) const {
    raw_inplace_ostream os(buffer);
    return serializeToStream(baseAttr, endian, os);
  }

  LogicalResult serializeToStream(Attribute baseAttr,
                                  llvm::support::endianness endian,
                                  llvm::raw_ostream &os) const {
    auto attr = llvm::cast<DenseResourceElementsAttr>(baseAttr);
    auto handle = attr.getRawHandle();

    // Special testing path for elided attributes. We want this to be an
    // error in normal circumstances as the output will produce garbage
    // results if executed but it can be useful when building reproducers.
    if (handle.getKey() == "__elided__") {
      if (!clZeroFillElidedAttrs) {
        return mlir::emitError(UnknownLoc::get(baseAttr.getContext()))
               << "elided attributes cannot be serialized; provide non-elided "
                  "values or pass --iree-util-zero-fill-elided-attrs for "
                  "testing and expect invalid execution results";
      }
      os.write_zeros(attr.getNumElements() *
                     IREE::Util::getRoundedElementByteWidth(
                         attr.getType().getElementType()));
      return success();
    }

    return mlir::emitError(UnknownLoc::get(baseAttr.getContext()))
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

  LogicalResult serializeToVector(Attribute baseAttr,
                                  llvm::support::endianness endian,
                                  SmallVectorImpl<char> &buffer) const {
    buffer.resize(getStorageSize(baseAttr));
    return serializeToBuffer(baseAttr, endian, buffer);
  }

  LogicalResult serializeToBuffer(Attribute baseAttr,
                                  llvm::support::endianness endian,
                                  ArrayRef<char> buffer) const {
    raw_inplace_ostream os(buffer);
    return serializeToStream(baseAttr, endian, os);
  }

  LogicalResult serializeToStream(Attribute baseAttr,
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
#include "iree/compiler/Dialect/Util/IR/UtilAttrs.cpp.inc"  // IWYU pragma: keep
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

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
