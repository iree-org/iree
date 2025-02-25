// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"

#include <cassert>

namespace mlir::iree_compiler::IREE::Encoding {

EncodingAttr EncodingAttr::get(MLIRContext *ctx, int64_t operandIndex,
                               EncodingOpType opType, ArrayRef<Type> elemTypes,
                               ArrayRef<AffineMap> maps,
                               std::optional<AffineMap> bcastMap,
                               ArrayRef<int64_t> roundDimsTo,
                               ArrayRef<Attribute> layouts) {
  Builder b(ctx);
  auto opTypeAttr = EncodingOpTypeAttr::get(ctx, opType);
  auto roundDimsToAttr = roundDimsTo.empty()
                             ? DenseI64ArrayAttr()
                             : b.getDenseI64ArrayAttr(roundDimsTo);
  auto bcastMapAttr = bcastMap.has_value()
                          ? AffineMapAttr::get(bcastMap.value())
                          : AffineMapAttr();
  auto layoutsAttr = layouts.empty() ? ArrayAttr() : b.getArrayAttr(layouts);
  return get(ctx, b.getIndexAttr(operandIndex), opTypeAttr,
             b.getTypeArrayAttr(elemTypes), b.getAffineMapArrayAttr(maps),
             bcastMapAttr, roundDimsToAttr, layoutsAttr);
}

AffineMap EncodingAttr::getMapForOperandIndex() const {
  auto index = getOperandIndex().getValue().getZExtValue();
  switch (index) {
  case MATMUL_LHS:
  case MATMUL_RHS:
  case MATMUL_RESULT: {
    auto indexingMap =
        llvm::cast<AffineMapAttr>(getUserIndexingMaps()[index]).getAffineMap();
    if (auto bcastMap = getBcastMap()) {
      indexingMap = bcastMap.getAffineMap().compose(indexingMap);
    }
    return indexingMap;
  }
  default:
    return AffineMap();
  }
}

std::optional<unsigned>
EncodingAttr::mapDimToOperandIndex(int64_t dimPos) const {
  return getMapForOperandIndex().getResultPosition(
      getAffineDimExpr(dimPos, getContext()));
}

ArrayRef<int64_t> EncodingAttr::getRoundDimsToArray() const {
  auto roundDimsTo = getRoundDimsTo();
  if (!roundDimsTo) {
    return {};
  }
  return llvm::cast<DenseI64ArrayAttr>(roundDimsTo).asArrayRef();
}

SmallVector<Type> EncodingAttr::getElementTypesArray() {
  return llvm::map_to_vector(getElementTypes().getValue(), [](Attribute a) {
    return llvm::cast<TypeAttr>(a).getValue();
  });
}

EncodingAttr EncodingAttr::clone(AffineMap bcastMap) {
  return get(bcastMap.getContext(), getOperandIndex(), getOpType(),
             getElementTypes(), getUserIndexingMaps(),
             AffineMapAttr::get(bcastMap), getRoundDimsTo(), getLayouts());
}

bool EncodingAttr::isSerialized() const { return getLayouts() ? true : false; }

Attribute EncodingAttr::cloneWithLayouts(ArrayRef<Attribute> layouts) const {
  MLIRContext *ctx = getContext();
  return get(ctx, getOperandIndex(), getOpType(), getElementTypes(),
             /*user_indexing_maps=*/ArrayAttr(),
             /*bcast_map=*/AffineMapAttr(),
             /*round_dims_to=*/DenseI64ArrayAttr(),
             ArrayAttr::get(ctx, layouts));
}

/// Returns the bit-width of the scalar type. If the type is complex, it returns
/// the type of individual elements * 2 (1 for real and 1 for complex).
static unsigned getTypeBitWidth(Type type) {
  if (auto complexType = dyn_cast<ComplexType>(type)) {
    return 2 * complexType.getElementType().getIntOrFloatBitWidth();
  }
  return type.getIntOrFloatBitWidth();
}

/// Returns the number of bytes an element of the given type occupies in memory.
/// This is in the default dense conversion to machine words where sizes must be
/// powers of two aligned to bytes.
///
/// Examples:
///   getRoundedElementByteWidth(i1) = 1
///   getRoundedElementByteWidth(i23) = 4
///   getRoundedElementByteWidth(i32) = 4
///   getRoundedElementByteWidth(bf16) = 2
///   getRoundedElementByteWidth(i33) = 8
///   getRoundedElementByteWidth(complex<f32>) = 8
static int32_t getRoundedElementByteWidth(Type type) {
  unsigned bitsUnaligned = getTypeBitWidth(type);
  assert(bitsUnaligned > 0 && "0-width types unsupported");
  // Round up to 8-bit aligned bytes.
  unsigned byteAligned = (bitsUnaligned + 8 - 1) / 8;
  // Round up to the next power of two (unless already a power of two).
  return llvm::PowerOf2Ceil(byteAligned);
}

Value EncodingAttr::calculateStorageSizeInBytes(Location loc,
                                                OpBuilder &builder,
                                                RankedTensorType type,
                                                ValueRange dynamicDims) const {
  if (ArrayAttr layoutsAttr = getLayouts()) {
    if (!llvm::all_of(layoutsAttr.getValue(),
                      llvm::IsaPred<SerializableEncodingAttrInterface>)) {
      return nullptr;
    }

    Value res;
    for (auto attr :
         layoutsAttr.getAsRange<SerializableEncodingAttrInterface>()) {
      Value requestedSize =
          attr.calculateStorageSizeInBytes(loc, builder, type, dynamicDims);
      if (!res) {
        res = requestedSize;
        continue;
      }
      res = builder.create<arith::MaxUIOp>(loc, res, requestedSize);
    }
    return res;
  }

  // TODO(hanchung): Deprecate the below logic once EncodingSpecialization pass
  // is enabled by default. The layouts should be resolved and `roundDimsTo`
  // will be deprecated.
  SmallVector<int64_t> paddedShape(type.getShape());
  SmallVector<Value> paddedDynamicDims(dynamicDims.begin(), dynamicDims.end());
  ArrayRef<int64_t> roundDimsTo = getRoundDimsToArray();
  FailureOr<linalg::ContractionDimensions> cDims =
      getEncodingContractionDims(*this);
  auto pad = [&](int dim, int value) {
    std::optional<unsigned> maybeMappedDim = mapDimToOperandIndex(dim);
    if (!maybeMappedDim) {
      return;
    }
    unsigned mappedDim = maybeMappedDim.value();
    if (type.isDynamicDim(mappedDim)) {
      mappedDim = type.getDynamicDimIndex(mappedDim);
      auto alignment = builder.create<arith::ConstantIndexOp>(loc, value);
      paddedDynamicDims[mappedDim] = builder.create<arith::CeilDivUIOp>(
          loc, paddedDynamicDims[mappedDim], alignment);
      paddedDynamicDims[mappedDim] = builder.create<arith::MulIOp>(
          loc, paddedDynamicDims[mappedDim], alignment);
    } else {
      paddedShape[mappedDim] = llvm::alignTo(paddedShape[mappedDim], value);
    }
  };
  for (auto m : cDims->m) {
    pad(m, roundDimsTo[0]);
  }
  for (auto n : cDims->n) {
    pad(n, roundDimsTo[1]);
  }
  for (auto k : cDims->k) {
    pad(k, roundDimsTo[2]);
  }

  constexpr int64_t kNumBitsInByte = 8;
  unsigned elementBits = getTypeBitWidth(type.getElementType());
  int64_t numBytesPerElem = 1;
  if (elementBits > kNumBitsInByte) {
    numBytesPerElem *= getRoundedElementByteWidth(type.getElementType());
  }

  int64_t staticCount = numBytesPerElem;
  for (unsigned i = 0, e = type.getRank(); i < e; ++i) {
    if (!type.isDynamicDim(i)) {
      staticCount *= paddedShape[i];
    }
  }

  Value result =
      builder.create<arith::ConstantIndexOp>(loc, staticCount).getResult();
  for (auto dim : paddedDynamicDims) {
    result = builder.create<arith::MulIOp>(loc, result, dim);
  }

  // Always pack the elements back-to-back for subtypes.
  if (elementBits < kNumBitsInByte) {
    if (kNumBitsInByte % elementBits) {
      assert(false && "unsupported subtype");
      return Value();
    }
    Value divisor = builder.create<arith::ConstantIndexOp>(
        loc, kNumBitsInByte / elementBits);
    result = builder.create<arith::CeilDivUIOp>(loc, result, divisor);
  }

  return result;
}

Value PadEncodingLayoutAttr::calculateStorageSizeInBytes(
    Location loc, OpBuilder &builder, RankedTensorType type,
    ValueRange dynamicDims) const {
  ArrayRef<int32_t> padding = getPadding().asArrayRef();
  assert(padding.size() == type.getRank() && "Invalid padding");

  const int64_t elementSize = getRoundedElementByteWidth(type.getElementType());
  int64_t staticProduct = elementSize;
  Value dynamicProduct = builder.create<arith::ConstantIndexOp>(loc, 1);

  size_t dynamicDimIdx = 0;
  for (auto [dimSize, padValue] : llvm::zip_equal(type.getShape(), padding)) {
    if (!ShapedType::isDynamic(dimSize)) {
      staticProduct *= (dimSize + padValue);
      continue;
    }

    Value dynamicDimSize = dynamicDims[dynamicDimIdx];
    ++dynamicDimIdx;

    if (padValue != 0) {
      dynamicDimSize = builder.create<arith::AddIOp>(
          loc, dynamicDimSize,
          builder.create<arith::ConstantIndexOp>(loc, padValue),
          arith::IntegerOverflowFlags::nsw);
    }
    dynamicProduct = builder.createOrFold<arith::MulIOp>(
        loc, dynamicProduct, dynamicDimSize, arith::IntegerOverflowFlags::nsw);
  }

  return builder.createOrFold<arith::MulIOp>(
      loc, builder.create<arith::ConstantIndexOp>(loc, staticProduct),
      dynamicProduct, arith::IntegerOverflowFlags::nsw);
}

//===---------------------------------------------------------------------===//
// encoding.identity_encoding
//===---------------------------------------------------------------------===//

Attribute
IdentityEncodingAttr::cloneWithSimplifiedConfig(DictionaryAttr) const {
  return *this;
}

Attribute IdentityEncodingAttr::getLayout(RankedTensorType type) const {
  MLIRContext *ctx = getContext();
  SmallVector<int32_t> zeros(type.getRank(), 0);
  return Encoding::PadEncodingLayoutAttr::get(
      ctx, DenseI32ArrayAttr::get(ctx, zeros));
}

//===---------------------------------------------------------------------===//
// encoding.unsupported_encoding
//===---------------------------------------------------------------------===//

Attribute
UnsupportedEncodingAttr::cloneWithSimplifiedConfig(DictionaryAttr) const {
  return *this;
}

Attribute UnsupportedEncodingAttr::getLayout(RankedTensorType) const {
  return nullptr;
}

//===---------------------------------------------------------------------===//
// Encoding attributes that are mainly for testing purpose.
//===---------------------------------------------------------------------===//

Attribute TestingEncodingAttr::parse(AsmParser &p, Type type) {
  if (failed(p.parseLess())) {
    return {};
  }
  ArrayAttr layouts;
  OptionalParseResult parseResult = p.parseOptionalAttribute(layouts);
  if (parseResult.has_value() && parseResult.value().failed()) {
    p.emitError(p.getNameLoc()) << "expected array attribute";
    return {};
  }
  if (failed(p.parseGreater())) {
    return {};
  }
  return get(p.getContext(), layouts);
}

void TestingEncodingAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  if (auto layouts = getLayouts()) {
    p.printAttribute(layouts);
  }
  os << ">";
}

bool TestingEncodingAttr::isSerialized() const {
  return getLayouts() ? true : false;
}

Attribute
TestingEncodingAttr::cloneWithLayouts(ArrayRef<Attribute> layouts) const {
  MLIRContext *ctx = getContext();
  return TestingEncodingAttr::get(ctx, ArrayAttr::get(ctx, layouts));
}

Attribute UnspecializedEncodingAttr::parse(AsmParser &p, Type type) {
  if (failed(p.parseLess())) {
    return {};
  }
  IntegerAttr seed;
  if (failed(p.parseAttribute(seed))) {
    return {};
  }
  if (failed(p.parseGreater())) {
    return {};
  }
  return get(p.getContext(), seed);
}

void UnspecializedEncodingAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  p.printAttributeWithoutType(getSeed());
  os << ">";
}

Attribute
UnspecializedEncodingAttr::cloneWithSimplifiedConfig(DictionaryAttr) const {
  MLIRContext *ctx = getContext();
  return SpecializedEncodingAttr::get(ctx, getSeed(), /*type=*/{});
}

Attribute SpecializedEncodingAttr::parse(AsmParser &p, Type type) {
  if (failed(p.parseLess())) {
    return {};
  }

  IntegerAttr seed;
  if (failed(p.parseAttribute(seed))) {
    return {};
  }

  TypeAttr typeAttr;
  if (succeeded(p.parseOptionalComma()) && failed(p.parseAttribute(typeAttr))) {
    return {};
  }

  if (failed(p.parseGreater())) {
    return {};
  }
  return get(p.getContext(), seed, typeAttr);
}

void SpecializedEncodingAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  p.printAttributeWithoutType(getSeed());
  if (auto typeAttr = getType()) {
    os << ", ";
    p.printAttribute(typeAttr);
  }
  os << ">";
}

Attribute SpecializedEncodingAttr::getLayout(RankedTensorType type) const {
  MLIRContext *ctx = getContext();
  return get(ctx, getSeed(), TypeAttr::get(dropEncoding(type)));
}

} // namespace mlir::iree_compiler::IREE::Encoding
