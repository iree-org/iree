// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "llvm/ADT/Repeated.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SMT/IR/SMTTypes.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Support/LLVM.h"

// Custom parse/print helper for the knobs dictionary in constraints op.
// Prints `knobs = { ... }` on its own line with newlines before and after.
static mlir::ParseResult parseKnobsDictionary(mlir::OpAsmParser &parser,
                                              mlir::DictionaryAttr &attr) {
  if (parser.parseKeyword("knobs") || parser.parseEqual()) {
    return mlir::failure();
  }
  return parser.parseAttribute(attr);
}
static void printKnobsDictionary(mlir::OpAsmPrinter &p, mlir::Operation *,
                                 mlir::DictionaryAttr attr) {
  p.printNewline();
  p << " knobs = ";
  p.printAttributeWithoutType(attr);
  p.printNewline();
}

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.cpp.inc" // IWYU pragma: keep
// clang-format on

using namespace mlir;
using namespace mlir::iree_compiler::IREE::Codegen;
namespace IREE = mlir::iree_compiler::IREE;

//===----------------------------------------------------------------------===//
// ExtractStridedMetadataOp
//===----------------------------------------------------------------------===//

/// The number and type of the results are inferred from the
/// shape of the source.
LogicalResult ExtractStridedMetadataOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ExtractStridedMetadataOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto sourceType = dyn_cast<MemRefType>(adaptor.getSource().getType());
  if (!sourceType) {
    return failure();
  }

  unsigned sourceRank = sourceType.getRank();
  IndexType indexType = IndexType::get(context);
  auto memrefType =
      MemRefType::get({}, sourceType.getElementType(),
                      MemRefLayoutAttrInterface{}, sourceType.getMemorySpace());
  // Base.
  inferredReturnTypes.push_back(memrefType);
  // Offset.
  inferredReturnTypes.push_back(indexType);
  // Sizes and strides.
  for (unsigned i = 0; i < sourceRank * 2; ++i) {
    inferredReturnTypes.push_back(indexType);
  }
  return success();
}

void ExtractStridedMetadataOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getBaseBuffer(), "base_buffer");
  setNameFn(getOffset(), "offset");
  // For multi-result to work properly with pretty names and packed syntax `x:3`
  // we can only give a pretty name to the first value in the pack.
  if (!getSizes().empty()) {
    setNameFn(getSizes().front(), "sizes");
    setNameFn(getStrides().front(), "strides");
  }
}

//===----------------------------------------------------------------------===//
// LoadFromBufferOp
//===----------------------------------------------------------------------===//

LogicalResult LoadFromBufferOp::verify() {
  RankedTensorType tensorType = getTensor().getType();
  MemRefType memrefType = getBuffer().getType();
  if (failed(verifyCompatibleShape(tensorType.getShape(),
                                   memrefType.getShape())) ||
      tensorType.getElementType() != memrefType.getElementType()) {
    return emitOpError("buffer and tensor shapes must be compatible and "
                       "element types must match");
  }
  return success();
}

void LoadFromBufferOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getBufferMutable(),
                       SideEffects::DefaultResource::get());
}

LogicalResult LoadFromBufferOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointAfterValue(getBuffer());
  reifiedReturnShapes.resize(1);
  reifiedReturnShapes[0] = memref::getMixedSizes(b, getLoc(), getBuffer());
  return success();
}

//===----------------------------------------------------------------------===//
// StoreToBufferOp
//===----------------------------------------------------------------------===//

LogicalResult StoreToBufferOp::verify() {
  RankedTensorType tensorType = getTensor().getType();
  MemRefType memrefType = getBuffer().getType();
  if (failed(verifyCompatibleShape(tensorType.getShape(),
                                   memrefType.getShape())) ||
      tensorType.getElementType() != memrefType.getElementType()) {
    return emitOpError("tensor and buffer shapes must be compatible and "
                       "element types must match");
  }
  return success();
}

void StoreToBufferOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getBufferMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// InnerTiledOp
//===----------------------------------------------------------------------===//

void InnerTiledOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<utils::IteratorType> iteratorTypes,
    InnerTileDescAttrInterface kind, InnerTiledSemanticsAttrInterface semantics,
    std::optional<SmallVector<SmallVector<int64_t>>> permutations) {
  ArrayAttr indexingMapsAttr = builder.getAffineMapArrayAttr(indexingMaps);
  ArrayAttr iteratorTypesAttr = builder.getArrayAttr(llvm::map_to_vector(
      iteratorTypes, [&](utils::IteratorType t) -> mlir::Attribute {
        return linalg::IteratorTypeAttr::get(builder.getContext(), t);
      }));
  std::optional<ArrayAttr> permutationsAttr;
  if (permutations) {
    permutationsAttr = builder.getArrayAttr(llvm::map_to_vector(
        *permutations, [&](const SmallVector<int64_t> &perm) -> Attribute {
          return builder.getDenseI64ArrayAttr(perm);
        }));
  }
  build(builder, result, inputs, outputs, indexingMapsAttr, iteratorTypesAttr,
        kind, semantics, permutationsAttr);
}

void InnerTiledOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    ValueRange outputs, ArrayRef<ArrayRef<AffineExpr>> indexingExprs,
    ArrayRef<utils::IteratorType> iteratorTypes,
    InnerTileDescAttrInterface kind, InnerTiledSemanticsAttrInterface semantics,
    std::optional<SmallVector<SmallVector<int64_t>>> permutations) {
  SmallVector<AffineMap> indexingMaps =
      AffineMap::inferFromExprList(indexingExprs, builder.getContext());
  build(builder, result, inputs, outputs, indexingMaps, iteratorTypes, kind,
        semantics, permutations);
}

void InnerTiledOp::build(OpBuilder &builder, OperationState &result,
                         ValueRange inputs, ValueRange outputs,
                         ArrayAttr indexingMaps, ArrayAttr iteratorTypes,
                         InnerTileDescAttrInterface kind,
                         InnerTiledSemanticsAttrInterface semantics,
                         std::optional<ArrayAttr> permutations) {
  result.addOperands(inputs);
  result.addOperands(outputs);
  result.addTypes(outputs.getTypes());
  Properties &inherentAttrs = result.getOrAddProperties<Properties>();
  inherentAttrs.setOperandSegmentSizes(
      {static_cast<int>(inputs.size()), static_cast<int>(outputs.size())});
  inherentAttrs.setIndexingMaps(indexingMaps);
  inherentAttrs.setIteratorTypes(iteratorTypes);
  inherentAttrs.setKind(kind);
  inherentAttrs.setSemantics(semantics);
  if (permutations) {
    inherentAttrs.setPermutations(*permutations);
  }
}

// Note: we can't use an "AllTypesMatch" constraint because it will cause an
// inferReturnTypes() method that doesn't understand variadic inputs to
// be generated.
LogicalResult
InnerTiledOp::inferReturnTypes(MLIRContext *, std::optional<Location>,
                               Adaptor adaptor,
                               SmallVectorImpl<Type> &inferredReturnTypes) {
  llvm::append_range(inferredReturnTypes, adaptor.getOutputs().getTypes());
  return success();
}

// Returns true when the operand's inner tile (`tensorTileType`) matches
// the MMA vector tile `vectorType`:
//   - element types must match;
//   - If `opaqueSemantics` and the tiles are static/non-scalable, the
//     element counts must match.
//   - Otherwise, the shapes must match after dropping non-scalable unit extents
//     from each side.
static bool matchTileTypes(RankedTensorType tensorTileType,
                           VectorType vectorType, bool opaqueSemantics) {
  if (tensorTileType.getElementType() != vectorType.getElementType()) {
    return false;
  }

  if (opaqueSemantics) {
    // Opaque semantics are only used on GPU. No scalable vectors.
    return tensorTileType.hasStaticShape() && !vectorType.isScalable() &&
           llvm::product_of(tensorTileType.getShape()) ==
               llvm::product_of(vectorType.getShape());
  }

  // Keep only non-unit extents on the tensor side, and non-unit-or-scalable
  // extents (paired with their scalable flag) on the MMA side.
  auto tensorDims = llvm::filter_to_vector(tensorTileType.getShape(),
                                           [](int64_t d) { return d != 1; });
  auto vectorDims = llvm::filter_to_vector(
      llvm::zip_equal(vectorType.getShape(), vectorType.getScalableDims()),
      [](auto dim) {
        auto [size, scalable] = dim;
        return scalable || size != 1;
      });

  if (tensorDims.size() != vectorDims.size()) {
    return false;
  }
  for (auto [tensorSize, vecDim] : llvm::zip_equal(tensorDims, vectorDims)) {
    auto [vecSize, scalable] = vecDim;
    if (scalable && ShapedType::isStatic(tensorSize)) {
      return false;
    }
    if (!scalable && tensorSize != vecSize) {
      return false;
    }
  }
  return true;
}

static LogicalResult verifyOperandTypes(InnerTiledOp tiledOp) {
  SmallVector<VectorType> mmaVectorTypes;
  tiledOp.getSemantics().getTileTypes(tiledOp.getKind(), mmaVectorTypes);
  const bool opaque = tiledOp.getSemantics().getOpaque();

  for (auto [index, tuple] : llvm::enumerate(llvm::zip_equal(
           tiledOp.getOperandTypes(),
           tiledOp.getIndexingMapsAttr().getAsValueRange<AffineMapAttr>(),
           mmaVectorTypes))) {
    auto [opType, map, mmaVectorType] = tuple;
    ShapedType operandShapedType = cast<ShapedType>(opType);
    Type operandElemType = operandShapedType.getElementType();

    ArrayRef<int64_t> operandShape = operandShapedType.getShape();
    ArrayRef<int64_t> operandTileShape(
        operandShape.drop_front(map.getNumResults()));
    auto tensorTileType =
        RankedTensorType::get(operandTileShape, operandElemType);
    SmallVector<int64_t> mmaShape(mmaVectorType.getShape());
    SmallVector<bool> mmaScalable(mmaVectorType.getScalableDims());
    std::optional<ArrayAttr> permutations = tiledOp.getPermutations();
    if (permutations && !opaque) {
      ArrayRef<int64_t> perm =
          cast<DenseI64ArrayAttr>((*permutations)[index]).asArrayRef();
      applyPermutationToVector(mmaShape, perm);
      applyPermutationToVector(mmaScalable, perm);
    }
    auto vectorType =
        VectorType::get(mmaShape, mmaVectorType.getElementType(), mmaScalable);

    if (!matchTileTypes(tensorTileType, vectorType, opaque)) {
      return tiledOp.emitOpError()
             << "operand #" << index << " inner tile " << tensorTileType
             << " is incompatible with expected MMA tile type " << vectorType;
    }
  }
  return success();
}

LogicalResult InnerTiledOp::verify() {
  int64_t expectedNumIns = getKind().getExpectedNumInputs();
  if (expectedNumIns != getNumInputs()) {
    return emitOpError("number of inputs (" + Twine(getNumInputs()) +
                       ") doesn't match expected number from kind (" +
                       Twine(expectedNumIns) + ")");
  }
  int64_t expectedNumOuts = getKind().getExpectedNumOutputs();
  if (expectedNumOuts != getNumOutputs()) {
    return emitOpError("number of outputs (" + Twine(getNumOutputs()) +
                       ") doesn't match expected number from kind (" +
                       Twine(expectedNumOuts) + ")");
  }

  StringRef kindNs = cast<Attribute>(getKind()).getDialect().getNamespace();
  StringRef semNs = cast<Attribute>(getSemantics()).getDialect().getNamespace();
  if (kindNs != semNs) {
    return emitOpError(
        llvm::formatv("kind attribute (dialect '{0}') and semantics attribute "
                      "(dialect '{1}') must use the same dialect namespace",
                      kindNs, semNs));
  }

  SmallVector<ShapedType> opTypes =
      llvm::map_to_vector(getOperandTypes(), llvm::CastTo<ShapedType>);
  SmallVector<AffineMap, 4> indexingMaps = getIndexingMapsArray();

  // Verify that an indexing map was specified for each operand.
  if (indexingMaps.size() != expectedNumIns + expectedNumOuts) {
    return emitOpError("expected an indexing map for each operand");
  }

  // Verify that each index map has 'numIterators' inputs, no symbols, and
  // that the number of map outputs equals the rank of its associated
  // vector operand.
  unsigned numIterators = getIteratorTypes().getValue().size();
  for (const auto &it : llvm::enumerate(indexingMaps)) {
    auto index = it.index();
    auto map = it.value();
    if (map.getNumSymbols() != 0) {
      return emitOpError("expected indexing map ")
             << index << " to have no symbols";
    }
    auto shapedType = opTypes[index];
    unsigned rank = shapedType.getRank();
    // Verify that the map has the right number of inputs, outputs, and indices.
    // This also correctly accounts for (..) -> () for rank-0 results.
    if (map.getNumDims() != numIterators) {
      return emitOpError("expected indexing map ")
             << index << " to have " << numIterators << " input dims";
    }
    if (map.getNumResults() >= rank) {
      return emitOpError("expected indexing map ")
             << index << " to have fewer than " << rank << " results";
    }
    if (!map.isProjectedPermutation()) {
      return emitOpError("expected indexing map ")
             << index << " to be a projected permutation";
    }
  }

  if (failed(getKind().verifyIndexingMaps(indexingMaps))) {
    return emitOpError("failed to verify indexing maps");
  }

  SmallVector<int64_t> bounds;
  getIterationBounds(bounds);
  for (auto [type, map] : llvm::zip_equal(opTypes, indexingMaps)) {
    // The truncation functionality of llvm::zip is intentional here to ignore
    // the inner dimensions.
    for (auto [dim, size] : llvm::zip(map.getResults(), type.getShape())) {
      int64_t dimIdx = cast<AffineDimExpr>(dim).getPosition();
      if (size != bounds[dimIdx]) {
        return emitOpError("shape does not match iteration bounds");
      }
    }
  }

  if (getPermutations()) {
    auto permRange = getPermutations()->getAsRange<DenseI64ArrayAttr>();
    int permCount = permRange.end() - permRange.begin();
    if (permCount != opTypes.size()) {
      return emitOpError(
          llvm::formatv("mismatch between the number of permutations ({}) and "
                        "the number of operands ({})",
                        permCount, opTypes.size()));
    }
    for (auto [index, permAttr] : llvm::enumerate(permRange)) {
      ArrayRef<int64_t> perm = permAttr.asArrayRef();
      ShapedType operandType = getOperandShapedTypes()[index];
      int64_t rank = operandType.getRank();
      int64_t outerRank = getOperandOuterRank(index);
      int64_t innerRank = rank - outerRank;
      if (perm.size() != innerRank) {
        return emitOpError(
            llvm::formatv("permutation #{} length {} does not match the inner "
                          "rank {} of the corresponding operand of type {}",
                          index, perm.size(), innerRank, operandType));
      }
      if (!isPermutationVector(perm)) {
        return emitOpError(llvm::formatv(
            "permutation #{} is not a permutation vector", index));
      }
    }
  }

  return verifyOperandTypes(*this);
}

static int64_t getResultIndex(AffineMap map, AffineExpr targetExpr) {
  for (int64_t i = 0, e = map.getNumResults(); i < e; ++i) {
    if (targetExpr == map.getResult(i)) {
      return i;
    }
  }
  return -1;
}

void InnerTiledOp::getIterationBounds(
    SmallVectorImpl<int64_t> &iterationBounds) {
  SmallVector<ShapedType> operandTypes = getOperandShapedTypes();
  SmallVector<AffineMap, 4> indexingMaps(getIndexingMapsArray());
  AffineMap combinedMap = concatAffineMaps(indexingMaps, getContext());
  SmallVector<int64_t> combinedOuterShapes;
  for (auto [opType, map] : llvm::zip_equal(operandTypes, indexingMaps)) {
    llvm::append_range(combinedOuterShapes,
                       opType.getShape().take_front(map.getNumResults()));
  }
  AffineMap inverseMap = inversePermutation(combinedMap);
  iterationBounds.append(inverseMap.compose(combinedOuterShapes));
}

std::optional<SmallVector<int64_t, 4>> InnerTiledOp::getShapeForUnroll() {
  SmallVector<int64_t, 4> shape;
  getIterationBounds(shape);
  return shape;
}

void InnerTiledOp::populateBoundsForShapedValueDim(
    Value value, int64_t dim, ValueBoundsConstraintSet &cstr) {
  // Result shapes equal the corresponding DPS init shapes.
  auto resultIdx = cast<OpResult>(value).getResultNumber();
  cstr.bound(value)[dim] == cstr.getExpr(getDpsInits()[resultIdx], dim);
}

//===----------------------------------------------------------------------===//
// WorkgroupCountHintOp
//===----------------------------------------------------------------------===//

ParseResult WorkgroupCountHintOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 3> dynamicSizes;
  DenseI64ArrayAttr staticSizesAttr;

  if (parseDynamicIndexList(parser, dynamicSizes, staticSizesAttr,
                            /*valueTypes=*/{},
                            /*delimiter=*/AsmParser::Delimiter::Paren)) {
    return failure();
  }

  // All sizes are of index type. `parseDynamicIndexList` does not set the sizes
  // correctly when used as a custom directive so manually infer it from the
  // number of parsed sizes.
  IndexType indexType = parser.getBuilder().getIndexType();
  llvm::Repeated<Type> dynamicSizeTypes(dynamicSizes.size(), indexType);

  if (parser.resolveOperands(dynamicSizes, dynamicSizeTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }

  result.addAttribute("static_sizes", staticSizesAttr);
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  return success();
}

void WorkgroupCountHintOp::print(OpAsmPrinter &printer) {
  printDynamicIndexList(printer, getOperation(), getSizes(), getStaticSizes(),
                        /*valueTypes=*/{},
                        /*delimiter=*/AsmParser::Delimiter::Paren);
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"static_sizes"});
}

//===----------------------------------------------------------------------===//
// DispatchConfigOp
//===----------------------------------------------------------------------===//

void DispatchConfigOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                             FlatSymbolRefAttr functionRef) {
  DispatchConfigOp::build(odsBuilder, odsState, functionRef,
                          /*workgroup_size=*/nullptr,
                          /*subgroup_size=*/nullptr);
}

LogicalResult DispatchConfigOp::verify() {
  if (auto wgSize = getWorkgroupSize()) {
    if (wgSize->empty() || wgSize->size() > 3) {
      return emitOpError("workgroup_size must have 1 to 3 entries, got ")
             << wgSize->size();
    }
  }

  return success();
}

LogicalResult DispatchConfigOp::verifyRegions() {
  Block &block = getBody().front();
  // The terminator must yield exactly 3 index values (workgroup count x, y, z).
  auto yieldOp = cast<YieldOp>(block.getTerminator());
  if (yieldOp.getNumOperands() != 3) {
    return emitOpError("expected terminator to yield exactly 3 operands "
                       "(workgroup count x, y, z), got ")
           << yieldOp.getNumOperands();
  }
  for (auto [i, type] : llvm::enumerate(yieldOp.getOperandTypes())) {
    if (!type.isIndex()) {
      return emitOpError("expected terminator operand #")
             << i << " to be index type, got " << type;
    }
  }

  return success();
}

SmallVector<int64_t> DispatchConfigOp::getStaticNumWorkgroups() {
  SmallVector<int64_t> result;
  auto yieldOp = cast<YieldOp>(getBody().front().getTerminator());
  for (Value v : yieldOp.getOperands()) {
    if (std::optional<int64_t> cst = getConstantIntValue(v)) {
      result.push_back(*cst);
    } else {
      result.push_back(ShapedType::kDynamic);
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// WorkgroupCountHintOp
//===----------------------------------------------------------------------===//

void WorkgroupCountHintOp::build(OpBuilder &builder, OperationState &state,
                                 ArrayRef<OpFoldResult> sizes) {
  SmallVector<int64_t> staticSizes;
  SmallVector<Value> dynamicSizes;
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  build(builder, state, dynamicSizes,
        builder.getDenseI64ArrayAttr(staticSizes));
}

//===----------------------------------------------------------------------===//
// ConstraintsOp
//===----------------------------------------------------------------------===//

/// Recursively check whether `name` appears as a knob name in `attr`.
/// Checks IntKnobAttr/OneOfKnobAttr names and recurses into
/// DictionaryAttr/ArrayAttr.
static bool hasKnobName(Attribute attr, StringRef name) {
  return TypeSwitch<Attribute, bool>(attr)
      .Case<IntKnobAttr, OneOfKnobAttr>(
          [&](auto knob) { return knob.getName().getValue() == name; })
      .Case([&](DictionaryAttr dict) {
        return llvm::any_of(dict, [&](NamedAttribute entry) {
          return hasKnobName(entry.getValue(), name);
        });
      })
      .Case([&](ArrayAttr array) {
        return llvm::any_of(array, [&](Attribute element) {
          return hasKnobName(element, name);
        });
      })
      .Default(false);
}

LogicalResult ConstraintsOp::verify() {
  Block &block = getBody().front();

  // Check block arg count matches problem_dims count.
  if (block.getNumArguments() != getProblemDims().size()) {
    return emitOpError("expected ")
           << getProblemDims().size() << " block arguments but got "
           << block.getNumArguments();
  }

  // Check all block args are !smt.int.
  smt::IntType smtIntType = smt::IntType::get(getContext());
  for (auto [i, arg] : llvm::enumerate(block.getArguments())) {
    if (arg.getType() != smtIntType) {
      return emitOpError("block argument #")
             << i << " must be !smt.int but got " << arg.getType();
    }
  }

  // Verify knob ops: check names exist in the dict and reject duplicates.
  // Note that we considered using SymbolTable for uniqueness, but the knobs
  // dictionary contains attributes (not ops), so we'd still need custom
  // verification for dictionary <--> KnobOp correspondence.
  // Rejecting duplicates is not just pedantic -- when this op is lowered to
  // SMT, each KnobOp becomes an `smt.declare_const`. The SMT dialect
  // creates a fresh symbolic constant per declaration regardless of the name
  // string, so two KnobOps with the same name would silently introduce two
  // independent
  // solver variables where one was intended, producing incorrect constraints.
  DictionaryAttr knobs = getKnobsAttr();
  llvm::StringMap<Location> seenKnobs;
  for (auto knobOp : block.getOps<KnobOp>()) {
    auto [it, inserted] =
        seenKnobs.try_emplace(knobOp.getName(), knobOp.getLoc());
    if (!inserted) {
      InFlightDiagnostic diag = knobOp.emitOpError("duplicate knob name '")
                                << knobOp.getName() << "'";
      diag.attachNote(it->second) << "first occurrence here";
      return diag;
    }
    if (!hasKnobName(knobs, knobOp.getName())) {
      return knobOp.emitOpError("knob name '")
             << knobOp.getName() << "' not found in knobs dict";
    }
  }

  return success();
}

LogicalResult LookupOp::verify() {
  if (getKeys().size() != getValues().size()) {
    return emitOpError("keys and values must have the same size, got ")
           << getKeys().size() << " keys and " << getValues().size()
           << " values";
  }
  if (getKeys().empty()) {
    return emitOpError("lookup table must be non-empty");
  }
  // Check for duplicate keys -- a duplicate would make the table ambiguous
  // and could produce different behavior between direct evaluation and
  // the chained smt.ite lowering.
  llvm::SmallDenseSet<int64_t> seen;
  for (int64_t key : getKeys()) {
    if (!seen.insert(key).second) {
      return emitOpError("duplicate key ") << key << " in lookup table";
    }
  }
  return success();
}

LogicalResult AssertOp::verify() {
  size_t placeholderCount = 0;
  StringRef fmt = getMsg();
  for (size_t pos = 0; (pos = fmt.find("{}", pos)) != StringRef::npos;
       pos += 2) {
    ++placeholderCount;
  }
  if (placeholderCount != getPrintArgs().size()) {
    return emitOpError("format string has ")
           << placeholderCount << " placeholder(s) but got "
           << getPrintArgs().size() << " arg(s)";
  }
  return success();
}
