// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"

#include <cassert>

#define DEBUG_TYPE "iree-encoding-attrs"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler::IREE::Encoding {

//===---------------------------------------------------------------------===//
// iree_encoding.layout
//===---------------------------------------------------------------------===//

LayoutAttr
LayoutAttr::getChecked(llvm::function_ref<InFlightDiagnostic()> emitError,
                       MLIRContext *context, ArrayAttr layoutsAttr) {
  if (failed(LayoutAttr::verify(emitError, layoutsAttr))) {
    return LayoutAttr();
  }
  return LayoutAttr::get(context, layoutsAttr);
}

LayoutAttr LayoutAttr::get(MLIRContext *context, ArrayAttr layoutsAttr) {
  auto emitError = mlir::detail::getDefaultDiagnosticEmitFn(context);
  if (failed(LayoutAttr::verify(emitError, layoutsAttr))) {
    return LayoutAttr();
  }
  return Base::get(context, layoutsAttr);
}

LogicalResult
LayoutAttr::verify(function_ref<mlir::InFlightDiagnostic()> emitError,
                   ArrayAttr layoutsAttr) {
  ArrayRef<Attribute> layouts = layoutsAttr.getValue();
  if (layouts.empty()) {
    return emitError() << "expected non-empty layouts";
  }
  if (!llvm::all_of(layouts, llvm::IsaPred<SerializableAttr>)) {
    return emitError() << "expected all the layout attributes to implement "
                          "SerializableAttr";
  }
  return success();
}

bool LayoutAttr::isSerialized() const { return true; }

bool LayoutAttr::isIdentityLayout() const {
  auto layouts = getLayouts().getAsRange<SerializableAttr>();
  return llvm::all_of(layouts,
                      [](auto attr) { return attr.isIdentityLayout(); });
}

Value LayoutAttr::calculateStorageSizeInBytes(Location loc, OpBuilder &builder,
                                              RankedTensorType type,
                                              ValueRange dynamicDims) const {
  ArrayAttr layoutsAttr = getLayouts();
  Value res;
  for (auto attr : layoutsAttr.getAsRange<SerializableAttr>()) {
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

//===---------------------------------------------------------------------===//
// iree_encoding.encoding
//===---------------------------------------------------------------------===//

/// Returns a composed affine map from the provided attribute. `attr` can be
/// either an `AffineMapAttr` or an `ArrayAttr` containing `AffineMapAttr`. In
/// case of an empty `attr`, an empty affine map is returned. In case of
/// unrecognized attribute types, a failure is returned.
static FailureOr<AffineMap> getComposedAffineMap(Attribute attr) {
  if (!attr) {
    return AffineMap();
  }
  if (auto mapAttr = llvm::dyn_cast<AffineMapAttr>(attr)) {
    return mapAttr.getAffineMap();
  }
  if (auto mapsAttr = llvm::dyn_cast<ArrayAttr>(attr)) {
    if (mapsAttr.empty()) {
      return AffineMap();
    }
    // All entries should have type `AffineMapAttr`.
    if (!llvm::all_of(mapsAttr, [](Attribute attr) {
          return isa<AffineMapAttr>(attr);
        })) {
      return failure();
    }
    AffineMap map =
        llvm::cast<AffineMapAttr>(mapsAttr[mapsAttr.size() - 1]).getAffineMap();
    for (ssize_t i = mapsAttr.size() - 2; i >= 0; i--) {
      map = map.compose(llvm::cast<AffineMapAttr>(mapsAttr[i]).getAffineMap());
    }
    return map;
  }
  // Return failure in case of an unrecognized attribute type.
  return failure();
}

EncodingAttr EncodingAttr::get(MLIRContext *ctx, int64_t operandIndex,
                               EncodingOpType opType, ArrayRef<Type> elemTypes,
                               ArrayRef<AffineMap> maps,
                               ArrayRef<int64_t> iterationSizes) {
  Builder b(ctx);
  auto opTypeAttr = EncodingOpTypeAttr::get(ctx, opType);
  auto mapsAttr = maps.empty() ? ArrayAttr() : b.getAffineMapArrayAttr(maps);
  auto iterationSizesAttr =
      iterationSizes.empty() ? ArrayAttr() : b.getI64ArrayAttr(iterationSizes);
  return get(ctx, b.getIndexAttr(operandIndex), opTypeAttr,
             b.getTypeArrayAttr(elemTypes), mapsAttr, iterationSizesAttr);
}

/// Parse a list of integer values and/or dynamic values ('?')
static FailureOr<SmallVector<int64_t>>
parseDynamicI64IntegerList(AsmParser &parser) {
  SmallVector<int64_t> integerVals;
  if (failed(parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&] {
        int64_t value = ShapedType::kDynamic;
        if (failed(parser.parseOptionalQuestion()) &&
            failed(parser.parseInteger(value))) {
          return failure();
        }
        integerVals.push_back(value);
        return success();
      }))) {
    return failure();
  }
  return integerVals;
}

/// Utility to parse an array of integer and/or dynamic values (`?`).
static ParseResult parseDynamicI64ArrayAttr(AsmParser &p, ArrayAttr &attr) {
  FailureOr<SmallVector<int64_t>> integerVals = parseDynamicI64IntegerList(p);
  if (failed(integerVals)) {
    return failure();
  }
  auto integerValsAttr =
      llvm::map_to_vector(integerVals.value(), [&](int64_t val) -> Attribute {
        return IntegerAttr::get(IntegerType::get(p.getContext(), 64), val);
      });
  attr = ArrayAttr::get(p.getContext(), integerValsAttr);
  return success();
}

/// Print a list of integer values and/or dynamic values ('?')
static void printDynamicI64IntegerList(AsmPrinter &printer,
                                       ArrayRef<int64_t> vals) {
  printer << "[";
  llvm::interleaveComma(vals, printer, [&](int64_t val) {
    if (ShapedType::isDynamic(val)) {
      printer << "?";
    } else {
      printer << val;
    }
  });
  printer << "]";
}

/// Utility to print an array of integer and/or dynamic values. Dynamic values
/// are printed as `?`.
static void printDynamicI64ArrayAttr(AsmPrinter &p, ArrayAttr attrs) {
  SmallVector<int64_t> intVals = llvm::map_to_vector(
      attrs, [&](Attribute attr) { return cast<IntegerAttr>(attr).getInt(); });
  return printDynamicI64IntegerList(p, intVals);
}

LogicalResult
EncodingAttr::verify(function_ref<mlir::InFlightDiagnostic()> emitError,
                     IntegerAttr operandIndexAttr,
                     EncodingOpTypeAttr opTypeAttr, ArrayAttr elementTypesAttr,
                     ArrayAttr userIndexingMapsAttr,
                     ArrayAttr iterationSizesAttr) {
  AffineMap indexingMap;
  if (userIndexingMapsAttr) {
    unsigned operandIndex = operandIndexAttr.getValue().getZExtValue();
    if (operandIndex >= userIndexingMapsAttr.size()) {
      return emitError()
             << "`operandIndex` exceeds the size of `user_indexing_maps`";
    }
    for (auto &&[idx, attr] : llvm::enumerate(userIndexingMapsAttr)) {
      FailureOr<AffineMap> composedMap = getComposedAffineMap(attr);
      if (failed(composedMap)) {
        return emitError() << "found a non-composable attribute in "
                              "`user_indexing_maps` at index: "
                           << idx;
      }
      if (idx == operandIndex) {
        // Keep track of the indexing map for later verification use.
        indexingMap = composedMap.value();
      }
    }
  }
  if (iterationSizesAttr) {
    if (!indexingMap) {
      return emitError() << "found `iteration_sizes` without any corresponding "
                            "`user_indexing_maps`";
    }
    if (iterationSizesAttr.size() != indexingMap.getNumDims()) {
      return emitError() << "found an encoding with "
                         << iterationSizesAttr.size()
                         << " iteration sizes, but expected "
                         << indexingMap.getNumDims()
                         << " based on the user indexing maps";
    }
  }
  return success();
}

AffineMap EncodingAttr::getMapForOperandIndex() const {
  unsigned index = getOperandIndex().getValue().getZExtValue();
  ArrayAttr userIndexingMaps = getUserIndexingMaps();
  if (!userIndexingMaps) {
    return AffineMap();
  }
  FailureOr<AffineMap> map = getComposedAffineMap(userIndexingMaps[index]);
  assert(!failed(map) &&
         "Expected a composable map. The verifier should ensure that all "
         "`user_indexing_maps` are composable.");
  return map.value();
}

SmallVector<AffineMap> EncodingAttr::getRootMaps() const {
  return llvm::map_to_vector(
      getUserIndexingMaps(), [](Attribute m) -> AffineMap {
        if (auto mapAttr = llvm::dyn_cast<AffineMapAttr>(m)) {
          return llvm::cast<AffineMapAttr>(m).getAffineMap();
        }
        if (auto mapsAttr = llvm::dyn_cast<ArrayAttr>(m)) {
          if (mapsAttr.empty())
            return AffineMap();
          return llvm::cast<AffineMapAttr>(mapsAttr[0]).getAffineMap();
        }
        return AffineMap();
      });
}

AffineMap EncodingAttr::getLastMapForOperandIndex() const {
  unsigned index = getOperandIndex().getValue().getZExtValue();
  ArrayAttr userIndexingMaps = getUserIndexingMaps();
  if (!userIndexingMaps) {
    return AffineMap();
  }
  Attribute indexingMap = userIndexingMaps[index];
  if (auto mapAttr = llvm::dyn_cast<AffineMapAttr>(indexingMap)) {
    return mapAttr.getAffineMap();
  }
  if (auto mapsAttr = llvm::dyn_cast<ArrayAttr>(indexingMap)) {
    if (mapsAttr.empty())
      return AffineMap();
    return llvm::cast<AffineMapAttr>(mapsAttr[mapsAttr.size() - 1])
        .getAffineMap();
  }
  return AffineMap();
}

std::optional<unsigned>
EncodingAttr::mapDimToOperandIndex(int64_t dimPos) const {
  return getMapForOperandIndex().getResultPosition(
      getAffineDimExpr(dimPos, getContext()));
}

SmallVector<int64_t> EncodingAttr::getIterationSizesArray() const {
  ArrayAttr iterationSizes = getIterationSizes();
  if (!iterationSizes) {
    return {};
  }
  return llvm::map_to_vector(iterationSizes, [](Attribute attr) {
    return llvm::cast<IntegerAttr>(attr).getInt();
  });
}

SmallVector<Type> EncodingAttr::getElementTypesArray() {
  return llvm::map_to_vector(getElementTypes().getValue(), [](Attribute a) {
    return llvm::cast<TypeAttr>(a).getValue();
  });
}

EncodingAttr
EncodingAttr::cloneWithNewOperandIndexingMap(AffineMap newIndexingMap) {
  if (!newIndexingMap) {
    return *this;
  }
  ArrayAttr userIndexingMaps = getUserIndexingMaps();
  SmallVector<Attribute> newMaps(userIndexingMaps.begin(),
                                 userIndexingMaps.end());
  unsigned operandIndex = getOperandIndex().getValue().getZExtValue();
  SmallVector<Attribute> maps;
  if (auto mapForIndex = llvm::dyn_cast<AffineMapAttr>(newMaps[operandIndex])) {
    maps.push_back(AffineMapAttr::get(mapForIndex.getAffineMap()));
  } else if (auto mapForIndex =
                 llvm::dyn_cast<ArrayAttr>(newMaps[operandIndex])) {
    maps.assign(mapForIndex.begin(), mapForIndex.end());
  }
  maps.push_back(AffineMapAttr::get(newIndexingMap));
  newMaps[operandIndex] = ArrayAttr::get(getContext(), maps);
  return get(getContext(), getOperandIndex(), getOpType(), getElementTypes(),
             ArrayAttr::get(getContext(), newMaps), getIterationSizes());
}

bool EncodingAttr::isSerialized() const { return false; }

Attribute EncodingAttr::cloneWithLayouts(ArrayRef<Attribute> layouts) const {
  MLIRContext *ctx = getContext();
  return LayoutAttr::get(ctx, ArrayAttr::get(ctx, layouts));
}

std::optional<SmallVector<int32_t>> EncodingAttr::getReductionDims() const {
  if (!getUserIndexingMaps()) {
    return std::nullopt;
  }
  SmallVector<AffineMap> indexingMaps = getRootMaps();
  FailureOr<linalg::ContractionDimensions> contractionDims =
      linalg::inferContractionDims(indexingMaps);
  if (failed(contractionDims)) {
    return std::nullopt;
  }
  SmallVector<int32_t> result;
  for (unsigned k : contractionDims->k) {
    if (std::optional<unsigned> dimIdx = mapDimToOperandIndex(k)) {
      result.push_back(dimIdx.value());
    }
  }
  return result;
}

//===---------------------------------------------------------------------===//
// iree_encoding.matmul_k
//===---------------------------------------------------------------------===//

std::optional<SmallVector<int32_t>> MatmulKAttr::getReductionDims() const {
  return llvm::to_vector(getKDims().asArrayRef());
}

//===---------------------------------------------------------------------===//
// iree_encoding.matmul_k
//===---------------------------------------------------------------------===//

MatmulKAttr MatmulKAttr::get(MLIRContext *ctx, ArrayRef<int32_t> kDims) {
  return get(ctx, DenseI32ArrayAttr::get(ctx, kDims));
}

bool MatmulKAttr::isSerialized() const { return false; }

Attribute MatmulKAttr::cloneWithLayouts(ArrayRef<Attribute> layouts) const {
  MLIRContext *ctx = getContext();
  return LayoutAttr::get(ctx, ArrayAttr::get(ctx, layouts));
}

//===---------------------------------------------------------------------===//
// iree_encoding.pad_encoding_layout
//===---------------------------------------------------------------------===//

/// Custom printer/parser methods to handle dynamic shapes.
ParseResult parsePadding(AsmParser &parser, DenseI64ArrayAttr &padding) {
  FailureOr<SmallVector<int64_t>> integerVals =
      parseDynamicI64IntegerList(parser);
  if (failed(integerVals)) {
    return failure();
  }
  padding = DenseI64ArrayAttr::get(parser.getContext(), integerVals.value());
  return success();
}
void printPadding(AsmPrinter &printer, DenseI64ArrayAttr padding) {
  return printDynamicI64IntegerList(printer, padding.asArrayRef());
}

/// Returns the bit-width of the scalar type. If the type is complex, it
/// returns the type of individual elements * 2 (1 for real and 1 for
/// complex).
static unsigned getTypeBitWidth(Type type) {
  if (auto complexType = dyn_cast<ComplexType>(type)) {
    return 2 * complexType.getElementType().getIntOrFloatBitWidth();
  }
  return type.getIntOrFloatBitWidth();
}

/// Returns the number of bytes an element of the given type occupies in
/// memory. This is in the default dense conversion to machine words where
/// sizes must be powers of two aligned to bytes.
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

PadEncodingLayoutAttr PadEncodingLayoutAttr::get(MLIRContext *ctx,
                                                 ArrayRef<int64_t> padding) {
  return get(ctx, DenseI64ArrayAttr::get(ctx, padding));
}

PadEncodingLayoutAttr PadEncodingLayoutAttr::getIdentityAttr(MLIRContext *ctx,
                                                             int rank) {
  SmallVector<int64_t> zeros(rank, 0);
  return get(ctx, zeros);
}

Attribute
PadEncodingLayoutAttr::cloneWithLayouts(ArrayRef<Attribute> layouts) const {
  MLIRContext *ctx = getContext();
  return LayoutAttr::get(ctx, ArrayAttr::get(ctx, layouts));
}

bool PadEncodingLayoutAttr::isSerialized() const {
  return !ShapedType::isDynamicShape(getPadding().asArrayRef());
}

bool PadEncodingLayoutAttr::isIdentityLayout() const {
  ArrayRef<int64_t> padding = getPadding().asArrayRef();
  return llvm::all_of(padding, [](int64_t val) { return val == 0; });
}

Value PadEncodingLayoutAttr::calculateStorageSizeInBytes(
    Location loc, OpBuilder &builder, RankedTensorType type,
    ValueRange dynamicDims) const {
  ArrayRef<int64_t> padding = getPadding().asArrayRef();
  assert(padding.size() == type.getRank() && "Invalid padding");
  LLVM_DEBUG(if (llvm::any_of(padding, [](int64_t x) { return x != 0; })) {
    llvm::dbgs() << "Non-zero padding: " << type << "\n";
  });

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

LogicalResult
PadEncodingLayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              DenseI64ArrayAttr padding) {
  // You can only verify that the value is non-negative or dynamic.
  if (!llvm::all_of(padding.asArrayRef(), [](int64_t val) {
        return val == ShapedType::kDynamic || val >= 0;
      })) {
    return emitError() << "expected all padding values need to be "
                          "non-negative or dynamic";
  }
  return success();
}

//===---------------------------------------------------------------------===//
// iree_encoding.identity_encoding
//===---------------------------------------------------------------------===//

Attribute
IdentityEncodingAttr::cloneWithSimplifiedConfig(DictionaryAttr) const {
  return *this;
}

Attribute IdentityEncodingAttr::getLayout(RankedTensorType type) const {
  MLIRContext *ctx = getContext();
  SmallVector<int64_t> zeros(type.getRank(), 0);
  return Encoding::PadEncodingLayoutAttr::get(
      ctx, DenseI64ArrayAttr::get(ctx, zeros));
}

//===---------------------------------------------------------------------===//
// iree_encoding.unsupported_encoding
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

Attribute
UnspecializedEncodingAttr::cloneWithSimplifiedConfig(DictionaryAttr) const {
  return *this;
}

Attribute UnspecializedEncodingAttr::getLayout(RankedTensorType type) const {
  MLIRContext *ctx = getContext();
  return SpecializedEncodingAttr::get(ctx, getSeed(),
                                      TypeAttr::get(type.dropEncoding()));
}

} // namespace mlir::iree_compiler::IREE::Encoding

using namespace mlir::iree_compiler::IREE::Encoding;

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/Encoding/IR/EncodingAttrs.cpp.inc"

void IREEEncodingDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Dialect/Encoding/IR/EncodingAttrs.cpp.inc"
      >();
}
