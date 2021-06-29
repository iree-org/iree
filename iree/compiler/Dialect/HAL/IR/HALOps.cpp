// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

template <typename T>
static LogicalResult parseEnumAttr(OpAsmParser &parser, StringRef attrName,
                                   NamedAttrList &attrs) {
  Attribute genericAttr;
  NamedAttrList attrList;
  auto loc = parser.getCurrentLocation();
  if (failed(parser.parseAttribute(genericAttr,
                                   parser.getBuilder().getNoneType(), attrName,
                                   attrList))) {
    return parser.emitError(loc) << "failed to parse enum string value";
  }
  auto stringAttr = genericAttr.dyn_cast<StringAttr>();
  if (!stringAttr) {
    return parser.emitError(loc)
           << "expected " << attrName << " attribute specified as string";
  }
  auto symbolized = symbolizeEnum<T>(stringAttr.getValue());
  if (!symbolized.hasValue()) {
    return parser.emitError(loc) << "failed to parse enum value";
  }
  attrs.push_back(parser.getBuilder().getNamedAttr(
      attrName, parser.getBuilder().getI32IntegerAttr(
                    static_cast<int32_t>(symbolized.getValue()))));
  return success();
}

//===----------------------------------------------------------------------===//
// custom<SizeAwareType>
//===----------------------------------------------------------------------===//
// type{%size}

static ParseResult parseSizeAwareType(OpAsmParser &parser, Type &type,
                                      OpAsmParser::OperandType &size) {
  if (failed(parser.parseType(type)) || failed(parser.parseLBrace()) ||
      failed(parser.parseOperand(size)) || failed(parser.parseRBrace())) {
    return failure();
  }
  return success();
}

static void printSizeAwareType(OpAsmPrinter &p, Operation *op, Type type,
                               Value size) {
  p.printType(type);
  p << "{";
  p.printOperand(size);
  p << "}";
}

//===----------------------------------------------------------------------===//
// custom<SizeAwareTypeList>
//===----------------------------------------------------------------------===//
// (type{%size0}, type, type{%size1})

static ParseResult parseSizeAwareTypeList(
    OpAsmParser &parser, SmallVectorImpl<Type> &types,
    SmallVectorImpl<OpAsmParser::OperandType> &sizes) {
  do {
    Type type;
    if (failed(parser.parseType(type))) return failure();
    if (type.isa<SizeAwareTypeInterface>()) {
      OpAsmParser::OperandType size;
      if (failed(parser.parseLBrace()) || failed(parser.parseOperand(size)) ||
          failed(parser.parseRBrace())) {
        return failure();
      }
      sizes.push_back(size);
    }
    types.push_back(type);
  } while (succeeded(parser.parseOptionalComma()));
  return success();
}

static void printSizeAwareTypeList(OpAsmPrinter &p, Operation *op,
                                   TypeRange types, OperandRange sizes) {
  int sizeIndex = 0;
  llvm::interleaveComma(types, p, [&](Type type) {
    p.printType(type);
    if (type.isa<SizeAwareTypeInterface>()) {
      p << "{";
      p.printOperand(sizes[sizeIndex++]);
      p << "}";
    }
  });
}

//===----------------------------------------------------------------------===//
// custom<DescriptorSetBindings>($binding_ordinals,
//                               $binding_buffers,
//                               type($binding_buffers),
//                               $binding_offsets,
//                               $binding_lengths)
//===----------------------------------------------------------------------===//

static ParseResult parseDescriptorSetBindings(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::OperandType> &ordinals,
    SmallVectorImpl<OpAsmParser::OperandType> &buffers,
    SmallVectorImpl<Type> &bufferTypes,
    SmallVectorImpl<OpAsmParser::OperandType> &bufferOffsets,
    SmallVectorImpl<OpAsmParser::OperandType> &bufferLengths) {
  do {
    OpAsmParser::OperandType ordinal;
    OpAsmParser::OperandType buffer;
    Type bufferType;
    OpAsmParser::OperandType bufferOffset;
    OpAsmParser::OperandType bufferLength;
    if (failed(parser.parseOperand(ordinal)) || failed(parser.parseEqual()) ||
        failed(parser.parseLParen()) || failed(parser.parseOperand(buffer)) ||
        failed(parser.parseColonType(bufferType)) ||
        failed(parser.parseRParen()) || failed(parser.parseLSquare()) ||
        failed(parser.parseOperand(bufferOffset)) ||
        failed(parser.parseComma()) ||
        failed(parser.parseOperand(bufferLength)) ||
        failed(parser.parseRSquare())) {
      return failure();
    }
    ordinals.push_back(ordinal);
    buffers.push_back(buffer);
    bufferTypes.push_back(bufferType);
    bufferOffsets.push_back(bufferOffset);
    bufferLengths.push_back(bufferLength);
  } while (succeeded(parser.parseOptionalComma()));
  return success();
}

static void printDescriptorSetBindings(OpAsmPrinter &p, Operation *op,
                                       ValueRange ordinals, ValueRange buffers,
                                       TypeRange bufferTypes,
                                       ValueRange bufferOffsets,
                                       ValueRange bufferLengths) {
  llvm::interleaveComma(
      llvm::zip(ordinals, buffers, bufferTypes, bufferOffsets, bufferLengths),
      p, [&](std::tuple<Value, Value, Type, Value, Value> it) {
        p.printNewline();
        p << "  ";
        p.printOperand(std::get<0>(it));
        p << " = (";
        p.printOperand(std::get<1>(it));
        p << " : ";
        p.printType(std::get<2>(it));
        p << ")[";
        p.printOperand(std::get<3>(it));
        p << ", ";
        p.printOperand(std::get<4>(it));
        p << "]";
      });
  p.printNewline();
}

//===----------------------------------------------------------------------===//
// custom<PackSliceRanges>($lifetime_intervals,
//                         $dynamic_slice_sizes,
//                         type($packed_offsets))
//===----------------------------------------------------------------------===//

static ParseResult parsePackSliceRanges(
    OpAsmParser &parser, ArrayAttr &lifetimeIntervals,
    SmallVectorImpl<OpAsmParser::OperandType> &dynamicSliceSizes,
    SmallVectorImpl<Type> &packedOffsetTypes) {
  auto indexType = parser.getBuilder().getIndexType();
  SmallVector<Attribute> lifetimeRangeValues;
  do {
    if (failed(parser.parseOptionalLSquare())) break;
    IntegerAttr lifetimeStart;
    IntegerAttr lifetimeEnd;
    OpAsmParser::OperandType dynamicSliceSize;
    if (failed(parser.parseAttribute(lifetimeStart, indexType)) ||
        failed(parser.parseComma()) ||
        failed(parser.parseAttribute(lifetimeEnd, indexType)) ||
        failed(parser.parseRSquare()) || failed(parser.parseEqual()) ||
        failed(parser.parseOperand(dynamicSliceSize))) {
      return failure();
    }
    lifetimeRangeValues.push_back(lifetimeStart);
    lifetimeRangeValues.push_back(lifetimeEnd);
    dynamicSliceSizes.push_back(dynamicSliceSize);
    packedOffsetTypes.push_back(indexType);
  } while (succeeded(parser.parseOptionalComma()));
  lifetimeIntervals = parser.getBuilder().getArrayAttr(lifetimeRangeValues);
  return success();
}

static void printPackSliceRanges(OpAsmPrinter &p, Operation *op,
                                 ArrayAttr lifetimeIntervals,
                                 ValueRange dynamicSliceSizes,
                                 TypeRange packedOffsetTypes) {
  if (packedOffsetTypes.empty()) return;
  for (unsigned i = 0; i < packedOffsetTypes.size(); ++i) {
    auto lifetimeStart = lifetimeIntervals[i * 2];
    auto lifetimeEnd = lifetimeIntervals[i * 2 + 1];
    auto sliceSize = dynamicSliceSizes[i];
    p.printNewline();
    p << "  [";
    p.printAttributeWithoutType(lifetimeStart);
    p << ", ";
    p.printAttributeWithoutType(lifetimeEnd);
    p << "] = ";
    p.printOperand(sliceSize);
    if (i < packedOffsetTypes.size() - 1) p << ",";
  }
  p.printNewline();
}

//===----------------------------------------------------------------------===//
// hal.ex.shared_device
//===----------------------------------------------------------------------===//

void ExSharedDeviceOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "device");
}

//===----------------------------------------------------------------------===//
// hal.tensor.cast
//===----------------------------------------------------------------------===//

void TensorCastOp::build(OpBuilder &builder, OperationState &result,
                         Type resultType, Value source,
                         ArrayRef<NamedAttribute> attrs) {
  SmallVector<Value> dynamicDims;
  if (source.getType().isa<IREE::HAL::BufferViewType>()) {
    auto shapedType = resultType.cast<ShapedType>();
    for (int64_t i = 0; i < shapedType.getRank(); ++i) {
      if (!shapedType.isDynamicDim(i)) continue;
      dynamicDims.push_back(builder.createOrFold<IREE::HAL::BufferViewDimOp>(
          result.location, builder.getIndexType(), source,
          builder.getIndexAttr(i)));
    }
  } else {
    dynamicDims =
        Shape::buildOrFindDynamicDimsForValue(result.location, source, builder);
  }
  build(builder, result, resultType, source, dynamicDims, attrs);
}

void TensorCastOp::build(OpBuilder &builder, OperationState &result,
                         Type resultType, Value source, ValueRange dynamicDims,
                         ArrayRef<NamedAttribute> attrs) {
  result.addTypes({resultType});
  result.addOperands({source});
  result.addOperands({dynamicDims});
  result.addAttributes(attrs);
  result.addAttribute(
      "operand_segment_sizes",
      builder.getI32VectorAttr({
          static_cast<int32_t>(1),
          static_cast<int32_t>(
              source.getType().isa<TensorType>() ? dynamicDims.size() : 0),
          static_cast<int32_t>(resultType.isa<TensorType>() ? dynamicDims.size()
                                                            : 0),
      }));
}

Value TensorCastOp::buildOperandRankedShape(unsigned idx, OpBuilder &builder) {
  if (source().getType().isa<TensorType>()) {
    return Shape::buildRankedShapeForValue(getLoc(), source(), source_dims(),
                                           builder);
  } else {
    return buildResultRankedShape(idx, builder);
  }
}

Value TensorCastOp::buildResultRankedShape(unsigned idx, OpBuilder &builder) {
  if (target().getType().isa<TensorType>()) {
    return Shape::buildRankedShapeForValue(getLoc(), target(), target_dims(),
                                           builder);
  } else {
    return buildOperandRankedShape(idx, builder);
  }
}

Value TensorCastOp::getTiedResult(unsigned resultIndex) {
  return IREE::TiedOpInterface::findTiedBaseValue(source());
}

::llvm::Optional<unsigned> TensorCastOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // source
}

SmallVector<int64_t, 4> TensorCastOp::getTiedResultOperandIndices() {
  return {0};  // source
}

//===----------------------------------------------------------------------===//
// hal.variable
//===----------------------------------------------------------------------===//

// Returns true if the given |accessType| is compatible with the |variableType|.
// For example, this will return true if the variable type is a tensor<?xf32>
// and the access is tensor<4xf32>.
static bool isVariableTypeCompatible(Type variableType, Type accessType) {
  // If one is a shaped type, then they both must be and have compatible
  // shapes.
  if (variableType.isa<ShapedType>() || accessType.isa<ShapedType>()) {
    return succeeded(mlir::verifyCompatibleShape(variableType, accessType));
  }

  // Otherwise, the types must be the same.
  return variableType == accessType;
}

static ParseResult parseVariableOp(OpAsmParser &parser,
                                   OperationState *result) {
  StringAttr nameAttr;
  if (failed(parser.parseSymbolName(nameAttr,
                                    mlir::SymbolTable::getSymbolAttrName(),
                                    result->attributes))) {
    return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("mutable"))) {
    result->addAttribute("is_mutable", UnitAttr::get(result->getContext()));
  }

  if (succeeded(parser.parseOptionalKeyword("init"))) {
    FlatSymbolRefAttr initializerAttr;
    if (failed(parser.parseLParen()) ||
        failed(parser.parseAttribute(initializerAttr, "initializer",
                                     result->attributes)) ||
        failed(parser.parseRParen())) {
      return failure();
    }
  }

  if (failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }

  Type type;
  if (succeeded(parser.parseOptionalEqual())) {
    // @foo = 4 : i32
    Attribute initialValueAttr;
    if (failed(parser.parseAttribute(initialValueAttr, "initial_value",
                                     result->attributes))) {
      return failure();
    }
    type = initialValueAttr.getType();
  } else {
    // @foo : index = 4 : i32
    if (failed(parser.parseColonType(type)) ||
        failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
      return failure();
    }
    if (succeeded(parser.parseOptionalEqual())) {
      Attribute initialValueAttr;
      if (failed(parser.parseAttribute(initialValueAttr, "initial_value",
                                       result->attributes))) {
        return failure();
      }
    }
  }
  result->addAttribute("type", TypeAttr::get(type));

  return success();
}

static void printVariableOp(OpAsmPrinter &p, VariableOp op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.sym_name());
  if (op.is_mutable()) {
    p << " mutable";
  }
  if (op.initializer().hasValue()) {
    p << " init(";
    p.printSymbolName(op.initializer().getValue());
    p << ')';
  }
  if (op.initial_value().hasValue() &&
      op.type() == op.initial_value().getValue().getType()) {
    // @foo = 4 : i32
  } else {
    // @foo : index = 4 : i32
    p << " : ";
    p.printType(op.type());
  }
  p.printOptionalAttrDictWithKeyword(op->getAttrs(), /*elidedAttrs=*/{
                                         "sym_name",
                                         "type",
                                         "is_mutable",
                                         "initializer",
                                         "initial_value",
                                     });
  if (op.initial_value().hasValue()) {
    p << " = ";
    p.printAttribute(op.initial_value().getValue());
  }
}

static LogicalResult verifyVariableOp(VariableOp op) {
  if (op.initializer().hasValue() && op.initial_value().hasValue()) {
    return op.emitOpError()
           << "variables can have either an initializer or an initial value";
  } else if (op.initializer().hasValue()) {
    // Ensure initializer returns the same type as the variable.
    auto *symbolOp =
        SymbolTable::lookupNearestSymbolFrom(op, op.initializer().getValue());
    if (!symbolOp) {
      return op.emitOpError() << "initializer function "
                              << op.initializer().getValue() << " not found";
    }
    auto initializerOp = dyn_cast<FuncOp>(symbolOp);
    if (initializerOp.getNumArguments() != 0 ||
        initializerOp.getNumResults() != 1 ||
        initializerOp.getType().getResult(0) != op.type()) {
      return op.emitOpError()
             << "initializer type mismatch; variable " << op.sym_name()
             << " is " << op.type() << " but initializer function "
             << initializerOp.getName() << " is " << initializerOp.getType();
    }
  }
  return success();
}

void VariableOp::build(OpBuilder &builder, OperationState &result,
                       StringRef name, bool isMutable, Type type,
                       Optional<StringRef> initializer,
                       Optional<Attribute> initialValue,
                       ArrayRef<NamedAttribute> attrs) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  if (isMutable) {
    result.addAttribute("is_mutable", builder.getUnitAttr());
  }
  if (initializer.hasValue()) {
    result.addAttribute("initializer",
                        builder.getSymbolRefAttr(initializer.getValue()));
  } else if (initialValue.hasValue()) {
    result.addAttribute("initial_value", initialValue.getValue());
  }
  result.addAttribute("type", TypeAttr::get(type));
  result.attributes.append(attrs.begin(), attrs.end());
}

void VariableOp::build(OpBuilder &builder, OperationState &result,
                       StringRef name, bool isMutable, mlir::FuncOp initializer,
                       ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, initializer.getType().getResult(0),
        initializer.getName(), llvm::None, attrs);
}

void VariableOp::build(OpBuilder &builder, OperationState &result,
                       StringRef name, bool isMutable, Type type,
                       Attribute initialValue, ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, type, llvm::None, initialValue,
        attrs);
}

void VariableOp::build(OpBuilder &builder, OperationState &result,
                       StringRef name, bool isMutable, Type type,
                       ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, type, llvm::None, llvm::None, attrs);
}

//===----------------------------------------------------------------------===//
// hal.variable.load
//===----------------------------------------------------------------------===//

void VariableLoadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // HACK: works around the lack of symbol side effects in mlir by only saying
  // we have a side-effect if the variable we are loading is mutable.
  auto *symbolOp = SymbolTable::lookupNearestSymbolFrom(*this, variable());
  assert(symbolOp);
  auto variableOp = dyn_cast<VariableOp>(symbolOp);
  if (variableOp.is_mutable()) {
    effects.emplace_back(MemoryEffects::Read::get());
  }
}

static LogicalResult verifyVariableLoadOp(VariableLoadOp &op) {
  auto *symbolOp = SymbolTable::lookupNearestSymbolFrom(op, op.variable());
  if (!symbolOp) {
    return op.emitOpError() << "undefined variable: " << op.variable();
  }
  auto variableOp = dyn_cast<VariableOp>(symbolOp);
  auto loadType = op.result().getType();
  if (!isVariableTypeCompatible(variableOp.type(), loadType)) {
    return op.emitOpError()
           << "variable type mismatch; variable " << op.variable() << " is "
           << variableOp.type() << " but load is " << loadType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// hal.variable.load.indirect
//===----------------------------------------------------------------------===//

static LogicalResult verifyVariableLoadIndirectOp(VariableLoadIndirectOp &op) {
  auto variableType =
      op.variable().getType().cast<IREE::PtrType>().getTargetType();
  auto loadType = op.result().getType();
  if (!isVariableTypeCompatible(variableType, loadType)) {
    return op.emitOpError() << "variable type mismatch; variable pointer is "
                            << variableType << " but load is " << loadType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// hal.variable.store
//===----------------------------------------------------------------------===//

static LogicalResult verifyVariableStoreOp(VariableStoreOp &op) {
  auto *symbolOp = SymbolTable::lookupNearestSymbolFrom(op, op.variable());
  if (!symbolOp) {
    return op.emitOpError() << "undefined variable: " << op.variable();
  }
  auto variableOp = dyn_cast<VariableOp>(symbolOp);
  auto storeType = op.value().getType();
  if (!isVariableTypeCompatible(variableOp.type(), storeType)) {
    return op.emitOpError()
           << "variable type mismatch; variable " << op.variable() << " is "
           << variableOp.type() << " but store is " << storeType;
  }
  if (!variableOp.is_mutable()) {
    return op.emitOpError() << "variable " << op.variable()
                            << " is not mutable and cannot be stored to";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// hal.variable.store.indirect
//===----------------------------------------------------------------------===//

static LogicalResult verifyVariableStoreIndirectOp(
    VariableStoreIndirectOp &op) {
  auto variableType =
      op.variable().getType().cast<IREE::PtrType>().getTargetType();
  auto storeType = op.value().getType();
  if (!isVariableTypeCompatible(variableType, storeType)) {
    return op.emitOpError() << "variable type mismatch; variable pointer is "
                            << variableType << " but store is " << storeType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// hal.allocator.compute_size
//===----------------------------------------------------------------------===//

void AllocatorComputeSizeOp::build(OpBuilder &builder, OperationState &state,
                                   Value allocator, ValueRange shape,
                                   int32_t elementType) {
  build(builder, state, allocator, shape,
        builder.createOrFold<ConstantIntOp>(state.location, elementType, 32));
}

void AllocatorComputeSizeOp::build(OpBuilder &builder, OperationState &state,
                                   Value allocator, ValueRange shape,
                                   Value elementType) {
  state.addOperands({allocator});
  state.addOperands(shape);
  state.addOperands(elementType);
  state.addTypes({builder.getIndexType()});
}

void AllocatorComputeSizeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "sz");
}

//===----------------------------------------------------------------------===//
// hal.allocator.compute_offset
//===----------------------------------------------------------------------===//

void AllocatorComputeOffsetOp::build(OpBuilder &builder, OperationState &state,
                                     Value allocator, ValueRange shape,
                                     int32_t elementType, ValueRange indices) {
  build(builder, state, allocator, shape,
        builder.createOrFold<ConstantIntOp>(state.location, elementType, 32),
        indices);
}

void AllocatorComputeOffsetOp::build(OpBuilder &builder, OperationState &state,
                                     Value allocator, ValueRange shape,
                                     Value elementType, ValueRange indices) {
  state.addOperands({allocator});
  state.addOperands(shape);
  state.addOperands(elementType);
  state.addOperands(indices);
  state.addTypes({builder.getIndexType()});
}

void AllocatorComputeOffsetOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(offset(), "off");
}

//===----------------------------------------------------------------------===//
// hal.allocator.compute_range
//===----------------------------------------------------------------------===//

void AllocatorComputeRangeOp::build(OpBuilder &builder, OperationState &state,
                                    Value allocator, ValueRange shape,
                                    int32_t elementType, ValueRange indices,
                                    ValueRange lengths) {
  build(builder, state, allocator, shape,
        builder.createOrFold<ConstantIntOp>(state.location, elementType, 32),
        indices, lengths);
}

void AllocatorComputeRangeOp::build(OpBuilder &builder, OperationState &state,
                                    Value allocator, ValueRange shape,
                                    Value elementType, ValueRange indices,
                                    ValueRange lengths) {
  state.addOperands({allocator});
  state.addOperands(shape);
  state.addOperands(elementType);
  state.addOperands(indices);
  state.addOperands(lengths);
  state.addTypes({builder.getIndexType(), builder.getIndexType()});
}

void AllocatorComputeRangeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(offset(), "off");
  setNameFn(length(), "len");
}

//===----------------------------------------------------------------------===//
// hal.allocator.allocate
//===----------------------------------------------------------------------===//

void AllocatorAllocateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "buffer");
}

Value AllocatorAllocateOp::getOperandSize(unsigned idx) { return {}; }

Value AllocatorAllocateOp::getResultSize(unsigned idx) { return result_size(); }

//===----------------------------------------------------------------------===//
// hal.allocator.constant
//===----------------------------------------------------------------------===//

void AllocatorConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "cbuffer");
}

//===----------------------------------------------------------------------===//
// hal.allocator.map
//===----------------------------------------------------------------------===//

void AllocatorMapOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "mapped");
}

Value AllocatorMapOp::getOperandSize(unsigned idx) { return {}; }

Value AllocatorMapOp::getResultSize(unsigned idx) { return length(); }

//===----------------------------------------------------------------------===//
// hal.allocator.pack
//===----------------------------------------------------------------------===//

void AllocatorPackOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  // TODO(benvanik): figure out if we can get the names to coalesce when there
  // are multiple results. Ideally we'd have `%total_length, %offsets:123` but
  // unfortunately all get splatted out and create 10k+ char lines that are a
  // pain to read.
  // setNameFn(total_length(), "total_length");
  // for (auto packedOffset : llvm::enumerate(packed_offsets())) {
  // setNameFn(packedOffset.value(),
  //           "offset" + std::to_string(packedOffset.index()));
  // }
}

static LogicalResult verifyAllocatorPackOp(AllocatorPackOp op) {
  size_t sliceCount = op.packed_offsets().size();
  if (op.lifetime_intervals().size() != sliceCount * 2) {
    return op.emitOpError() << "requires a [start, end] range for each slice";
  }
  if (op.dynamic_slice_sizes().size() != sliceCount) {
    return op.emitOpError() << "requires a size for each slice";
  }
  return success();
}

SmallVector<AllocatorPackOp::Slice> AllocatorPackOp::getSlices() {
  auto intervalPairs = lifetime_intervals().getValue();
  auto sizes = dynamic_slice_sizes();
  auto offsets = packed_offsets();
  SmallVector<AllocatorPackOp::Slice> slices(offsets.size());
  for (size_t i = 0; i < offsets.size(); ++i) {
    int64_t start = intervalPairs[i * 2 + 0].cast<IntegerAttr>().getInt();
    int64_t end = intervalPairs[i * 2 + 1].cast<IntegerAttr>().getInt();
    slices[i] = {start, end, sizes[i], offsets[i]};
  }
  return slices;
}

//===----------------------------------------------------------------------===//
// hal.buffer.allocator
//===----------------------------------------------------------------------===//

void BufferAllocatorOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "allocator");
}

//===----------------------------------------------------------------------===//
// hal.buffer.subspan
//===----------------------------------------------------------------------===//

void BufferSubspanOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "buffer");
}

Value BufferSubspanOp::getOperandSize(unsigned idx) { return length(); }

Value BufferSubspanOp::getResultSize(unsigned idx) { return length(); }

//===----------------------------------------------------------------------===//
// hal.buffer.byte_length
//===----------------------------------------------------------------------===//

void BufferLengthOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "len");
}

//===----------------------------------------------------------------------===//
// hal.buffer_view.create
//===----------------------------------------------------------------------===//

void BufferViewCreateOp::build(OpBuilder &builder, OperationState &state,
                               Value buffer, int32_t elementType,
                               ValueRange shape) {
  build(builder, state, buffer,
        builder.createOrFold<ConstantIntOp>(state.location, elementType, 32),
        shape);
}

void BufferViewCreateOp::build(OpBuilder &builder, OperationState &state,
                               Value buffer, Value elementType,
                               ValueRange shape) {
  state.addOperands({buffer, elementType});
  state.addOperands(shape);
  state.addTypes({BufferViewType::get(builder.getContext())});
}

void BufferViewCreateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "view");
}

//===----------------------------------------------------------------------===//
// hal.buffer_view.subview
//===----------------------------------------------------------------------===//

void BufferViewSubviewOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "view");
}

//===----------------------------------------------------------------------===//
// hal.buffer_view.buffer
//===----------------------------------------------------------------------===//

void BufferViewBufferOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "buffer");
}

//===----------------------------------------------------------------------===//
// hal.buffer_view.byte_length
//===----------------------------------------------------------------------===//

void BufferViewByteLengthOp::build(OpBuilder &builder, OperationState &state,
                                   Value bufferView) {
  state.addOperands({bufferView});
  state.addTypes({builder.getIndexType()});
}

void BufferViewByteLengthOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "len");
}

//===----------------------------------------------------------------------===//
// hal.buffer_view.compute_offset
//===----------------------------------------------------------------------===//

void BufferViewComputeOffsetOp::build(OpBuilder &builder, OperationState &state,
                                      Value bufferView, ValueRange indices) {
  state.addOperands({bufferView});
  state.addOperands(indices);
  state.addTypes({builder.getIndexType()});
}

void BufferViewComputeOffsetOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(offset(), "off");
}

//===----------------------------------------------------------------------===//
// hal.buffer_view.compute_range
//===----------------------------------------------------------------------===//

void BufferViewComputeRangeOp::build(OpBuilder &builder, OperationState &state,
                                     Value bufferView, ValueRange indices,
                                     ValueRange lengths) {
  state.addOperands({bufferView});
  state.addOperands(indices);
  state.addOperands(lengths);
  state.addTypes({builder.getIndexType(), builder.getIndexType()});
}

void BufferViewComputeRangeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(offset(), "off");
  setNameFn(length(), "len");
}

//===----------------------------------------------------------------------===//
// hal.command_buffer.create
//===----------------------------------------------------------------------===//

void CommandBufferCreateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "cmd");
}

//===----------------------------------------------------------------------===//
// hal.command_buffer.execution_barrier
//===----------------------------------------------------------------------===//

static ParseResult parseCommandBufferExecutionBarrierOp(
    OpAsmParser &parser, OperationState *result) {
  OpAsmParser::OperandType commandBuffer;
  if (failed(parser.parseOperand(commandBuffer)) ||
      failed(parser.resolveOperand(commandBuffer,
                                   CommandBufferType::get(result->getContext()),
                                   result->operands)) ||
      failed(parser.parseComma()) ||
      failed(parseEnumAttr<ExecutionStageBitfield>(parser, "source_stage_mask",
                                                   result->attributes)) ||
      failed(parser.parseComma()) ||
      failed(parseEnumAttr<ExecutionStageBitfield>(parser, "target_stage_mask",
                                                   result->attributes)) ||
      failed(parser.parseComma()) ||
      failed(parseEnumAttr<ExecutionBarrierFlagBitfield>(parser, "flags",
                                                         result->attributes))) {
    return failure();
  }
  if (failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

static void printCommandBufferExecutionBarrierOp(
    OpAsmPrinter &p, CommandBufferExecutionBarrierOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.command_buffer());
  p << ", \"";
  p << stringifyExecutionStageBitfield(op.source_stage_mask());
  p << "\", \"";
  p << stringifyExecutionStageBitfield(op.target_stage_mask());
  p << "\", \"";
  p << stringifyExecutionBarrierFlagBitfield(op.flags());
  p << "\"";
  p.printOptionalAttrDictWithKeyword(op->getAttrs(),
                                     /*elidedAttrs=*/{
                                         "source_stage_mask",
                                         "target_stage_mask",
                                         "flags",
                                     });
}

//===----------------------------------------------------------------------===//
// hal.command_buffer.push_descriptor_set
//===----------------------------------------------------------------------===//

void CommandBufferPushDescriptorSetOp::build(
    OpBuilder &builder, OperationState &state, Value commandBuffer,
    Value executableLayout, int64_t set,
    ArrayRef<DescriptorSetBindingValue> bindings) {
  build(builder, state, commandBuffer, executableLayout,
        builder.createOrFold<ConstantIndexOp>(state.location, set), bindings);
}

void CommandBufferPushDescriptorSetOp::build(
    OpBuilder &builder, OperationState &state, Value commandBuffer,
    Value executableLayout, Value set,
    ArrayRef<DescriptorSetBindingValue> bindings) {
  state.addOperands({commandBuffer, executableLayout, set});
  SmallVector<Value, 4> bindingOrdinals;
  SmallVector<Value, 4> bindingBuffers;
  SmallVector<Value, 4> bindingOffsets;
  SmallVector<Value, 4> bindingLengths;
  for (auto binding : bindings) {
    bindingOrdinals.push_back(std::get<0>(binding));
    bindingBuffers.push_back(std::get<1>(binding));
    bindingOffsets.push_back(std::get<2>(binding));
    bindingLengths.push_back(std::get<3>(binding));
  }
  state.addOperands(bindingOrdinals);
  state.addOperands(bindingBuffers);
  state.addOperands(bindingOffsets);
  state.addOperands(bindingLengths);
}

//===----------------------------------------------------------------------===//
// hal.constant_pool
//===----------------------------------------------------------------------===//

void ConstantPoolOp::build(OpBuilder &builder, OperationState &state,
                           StringRef name,
                           BufferConstraintsAttr bufferConstraints) {
  ensureTerminator(*state.addRegion(), builder, state.location);
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute("buffer_constraints", bufferConstraints);
}

static ParseResult parseConstantPoolOp(OpAsmParser &parser,
                                       OperationState *result) {
  StringAttr nameAttr;
  if (failed(parser.parseSymbolName(nameAttr,
                                    mlir::SymbolTable::getSymbolAttrName(),
                                    result->attributes)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }

  // Parse the module body.
  auto *body = result->addRegion();
  if (failed(parser.parseRegion(*body, llvm::None, llvm::None))) {
    return failure();
  }

  // Ensure that this module has a valid terminator.
  ConstantPoolOp::ensureTerminator(*body, parser.getBuilder(),
                                   result->location);
  return success();
}

static void printConstantPoolOp(OpAsmPrinter &p, ConstantPoolOp op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.sym_name());
  p.printOptionalAttrDictWithKeyword(
      op->getAttrs(),
      /*elidedAttrs=*/{mlir::SymbolTable::getSymbolAttrName()});
  p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

//===----------------------------------------------------------------------===//
// hal.constant_pool.load
//===----------------------------------------------------------------------===//

void ConstantPoolLoadOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(result(), "const");
}

//===----------------------------------------------------------------------===//
// hal.constant_storage.lookup
//===----------------------------------------------------------------------===//

void ConstantStorageLookupOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(result(), "storage");
}

//===----------------------------------------------------------------------===//
// hal.constant.subspan
//===----------------------------------------------------------------------===//

void ConstantSubspanOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(result(), "const_span");
}

//===----------------------------------------------------------------------===//
// hal.descriptor_set.create
//===----------------------------------------------------------------------===//

void DescriptorSetCreateOp::build(
    OpBuilder &builder, OperationState &state, Value device, Value setLayout,
    ArrayRef<DescriptorSetBindingValue> bindings) {
  state.addOperands({device, setLayout});
  SmallVector<Value, 4> bindingOrdinals;
  SmallVector<Value, 4> bindingBuffers;
  SmallVector<Value, 4> bindingOffsets;
  SmallVector<Value, 4> bindingLengths;
  for (auto binding : bindings) {
    bindingOrdinals.push_back(std::get<0>(binding));
    bindingBuffers.push_back(std::get<1>(binding));
    bindingOffsets.push_back(std::get<2>(binding));
    bindingLengths.push_back(std::get<3>(binding));
  }
  state.addOperands(bindingOrdinals);
  state.addOperands(bindingBuffers);
  state.addOperands(bindingOffsets);
  state.addOperands(bindingLengths);
}

void DescriptorSetCreateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "descriptor_set");
}

//===----------------------------------------------------------------------===//
// hal.descriptor_set_layout.create
//===----------------------------------------------------------------------===//

void DescriptorSetLayoutCreateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "descriptor_set_layout");
}

//===----------------------------------------------------------------------===//
// hal.descriptor_set_layout.lookup
//===----------------------------------------------------------------------===//

void DescriptorSetLayoutLookupOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "descriptor_set_layout");
}

//===----------------------------------------------------------------------===//
// hal.device.allocator
//===----------------------------------------------------------------------===//

void DeviceAllocatorOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "allocator");
}

//===----------------------------------------------------------------------===//
// hal.device.switch
//===----------------------------------------------------------------------===//

void DeviceSwitchOp::build(OpBuilder &builder, OperationState &state,
                           TypeRange resultTypes, Value device,
                           ArrayRef<Attribute> conditions,
                           ArrayRef<SmallVector<Value, 4>> conditionArgs,
                           ArrayRef<NamedAttribute> attributes) {
  state.addOperands({device});
  state.addAttribute("conditions", builder.getArrayAttr(conditions));
  for (auto args : conditionArgs) {
    state.addOperands(args);
    state.addRegion();
  }
  state.addTypes(resultTypes);
  state.addAttributes(attributes);
}

static ParseResult parseDeviceSwitchOp(OpAsmParser &parser,
                                       OperationState *result) {
  OpAsmParser::OperandType device;
  Type deviceType;
  if (failed(parser.parseLess()) || failed(parser.parseOperand(device)) ||
      failed(parser.parseColonType(deviceType)) ||
      failed(parser.resolveOperand(device, deviceType, result->operands)) ||
      failed(parser.parseGreater()) ||
      failed(parser.parseOptionalArrowTypeList(result->types))) {
    return failure();
  }

  // Parses each switch condition attribute and region, like:
  // #hal.device.match.id<"vulkan-v1.?-*">(%c1a = %c1 : i32) {
  //   hal.return %c1a : i32
  // }, ...
  SmallVector<Attribute, 4> conditionAttrs;
  do {
    Attribute conditionAttr;
    NamedAttrList dummyAttrs;
    if (failed(parser.parseAttribute(conditionAttr, "condition", dummyAttrs)) ||
        failed(parser.parseLParen())) {
      return failure();
    }
    conditionAttrs.push_back(conditionAttr);
    SmallVector<OpAsmParser::OperandType, 16> regionArgs;
    SmallVector<Type, 16> regionArgTypes;
    if (failed(parser.parseOptionalRParen())) {
      SmallVector<OpAsmParser::OperandType, 16> regionOperands;
      auto argsLoc = parser.getCurrentLocation();
      do {
        // Reserve entries in the lists.
        regionArgs.emplace_back();
        regionOperands.emplace_back();
        regionArgTypes.emplace_back();
        if (failed(parser.parseRegionArgument(regionArgs.back())) ||
            failed(parser.parseEqual()) ||
            failed(parser.parseOperand(regionOperands.back())) ||
            failed(parser.parseColonType(regionArgTypes.back()))) {
          return failure();
        }
      } while (succeeded(parser.parseOptionalComma()));
      if (failed(parser.parseRParen()) ||
          failed(parser.resolveOperands(regionOperands, regionArgTypes, argsLoc,
                                        result->operands))) {
        return failure();
      }
    }
    auto *regionBody = result->addRegion();
    if (failed(parser.parseRegion(*regionBody, regionArgs, regionArgTypes))) {
      return failure();
    }
  } while (succeeded(parser.parseOptionalComma()));
  result->addAttribute("conditions",
                       ArrayAttr::get(result->getContext(), conditionAttrs));

  if (failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

static void printDeviceSwitchOp(OpAsmPrinter &p, DeviceSwitchOp op) {
  p << op.getOperationName() << "<";
  p.printOperand(op.device());
  p << " : ";
  p.printType(op.device().getType());
  p << ">";
  p.printOptionalArrowTypeList(op.getResultTypes());
  p << "\n";
  p.getStream().indent(4);
  int argOffset = 0;
  interleave(
      llvm::zip(op.conditions(), op.condition_regions()),
      [&](std::tuple<Attribute, Region &> it) {
        auto &conditionAttr = std::get<0>(it);
        auto &conditionRegion = std::get<1>(it);
        p.printAttribute(conditionAttr);
        p << "(";
        auto regionOperands = conditionRegion.getArguments();
        auto regionArgs = op.args().slice(argOffset, regionOperands.size());
        argOffset += regionOperands.size();
        // TODO(benvanik): figure out how to parse with shadowing.
        // p.shadowRegionArgs(conditionRegion, regionArgs);
        interleaveComma(llvm::zip(regionOperands, regionArgs), p,
                        [&](std::tuple<BlockArgument, Value> it) {
                          p << std::get<0>(it) << " = " << std::get<1>(it);
                          p << " : ";
                          p << std::get<1>(it).getType();
                        });
        p << ")";
        p.printRegion(conditionRegion,
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
      },
      [&]() {
        p << ",\n";
        p.getStream().indent(4);
      });
  p.printOptionalAttrDictWithKeyword(op->getAttrs(),
                                     /*elidedAttrs=*/{"conditions"});
}

static LogicalResult verifyDeviceSwitchOp(DeviceSwitchOp op) {
  if (op.conditions().size() != op.condition_regions().size()) {
    return op.emitOpError() << "requires conditions and regions be matched 1:1";
  } else if (op.condition_regions().empty()) {
    return op.emitOpError() << "requires at least one condition";
  }
  int argOffset = 0;
  for (auto &region : op.condition_regions()) {
    auto regionOperands = region.getArguments();
    auto regionArgs = op.args().slice(argOffset, regionOperands.size());
    argOffset += regionOperands.size();

    for (auto it : llvm::zip(regionArgs, regionOperands)) {
      auto regionArg = std::get<0>(it);
      auto regionOperand = std::get<1>(it);
      if (regionArg.getType() != regionOperand.getType()) {
        return op.emitOpError() << "requires that regions have their arguments "
                                   "represented in the op arg list in order ("
                                << regionArg.getType()
                                << " != " << regionOperand.getType() << ")";
      }
    }

    for (auto &block : region) {
      if (auto returnOp =
              dyn_cast_or_null<IREE::HAL::ReturnOp>(block.getTerminator())) {
        if (!std::equal(returnOp.getOperandTypes().begin(),
                        returnOp.getOperandTypes().end(),
                        op.getResultTypes().begin())) {
          return op.emitOpError()
                 << "requires all regions return the same types";
        }
      }
    }
  }
  if (argOffset != op.args().size()) {
    return op.emitOpError() << "requires that the total argument list matches "
                               "the sum of all region operands";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// hal.executable
//===----------------------------------------------------------------------===//

void ExecutableOp::build(OpBuilder &builder, OperationState &state,
                         StringRef name) {
  ensureTerminator(*state.addRegion(), builder, state.location);
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
}

static ParseResult parseExecutableOp(OpAsmParser &parser,
                                     OperationState *result) {
  StringAttr nameAttr;
  if (failed(parser.parseSymbolName(nameAttr,
                                    mlir::SymbolTable::getSymbolAttrName(),
                                    result->attributes)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }

  // Parse the module body.
  auto *body = result->addRegion();
  if (failed(parser.parseRegion(*body, llvm::None, llvm::None))) {
    return failure();
  }

  // Ensure that this module has a valid terminator.
  ExecutableOp::ensureTerminator(*body, parser.getBuilder(), result->location);
  return success();
}

static void printExecutableOp(OpAsmPrinter &p, ExecutableOp op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.sym_name());
  p.printOptionalAttrDictWithKeyword(
      op->getAttrs(),
      /*elidedAttrs=*/{mlir::SymbolTable::getSymbolAttrName()});
  p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

static LogicalResult verifyExecutableOp(ExecutableOp op) {
  // TODO(benvanik): check export name conflicts.
  return success();
}

//===----------------------------------------------------------------------===//
// hal.executable.entry_point
//===----------------------------------------------------------------------===//

static ParseResult parseExecutableEntryPointOp(OpAsmParser &parser,
                                               OperationState *result) {
  StringAttr nameAttr;
  if (failed(parser.parseSymbolName(nameAttr,
                                    mlir::SymbolTable::getSymbolAttrName(),
                                    result->attributes)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  // For now assume that the workload is at max 3D. So arguments to the region
  // are workload along x, y and z.
  std::unique_ptr<Region> region;
  SmallVector<OpAsmParser::OperandType, 4> regionOperands;
  SmallVector<Type, 4> regionTypes;
  OptionalParseResult parseResult =
      parser.parseOptionalRegion(region, regionOperands, regionTypes);
  if (!parseResult.hasValue()) return success();
  if (failed(*parseResult)) return failure();
  result->addRegion(std::move(region));
  return success();
}

static void printExecutableEntryPointOp(OpAsmPrinter &p,
                                        ExecutableEntryPointOp op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.sym_name());
  p.printOptionalAttrDictWithKeyword(op->getAttrs(),
                                     /*elidedAttrs=*/{"sym_name"});
  if (op.workgroup_count_region().empty()) return;
  p.printRegion(op.workgroup_count_region().front());
}

static LogicalResult verifyExecutableEntryPointOp(ExecutableEntryPointOp op) {
  Region *region = op.getBody();
  // When there is no region, nothing to verify.
  if (!region) return success();

  if (!llvm::hasSingleElement(*region)) {
    return op.emitOpError() << "expected a single region";
  }
  if (region->getNumArguments() != 3) {
    return op.emitOpError(
        "expected three arguments for workgroup_count_region for workload "
        "along "
        "x, y, and z");
  }
  for (BlockArgument &blockArg : region->getArguments()) {
    if (!blockArg.getType().isa<IndexType>()) {
      return op.emitOpError(
          "expected arguments to workgroup_count_region be index type");
    }
  }
  // Check that the last statement in the block is `hal.yield` operation.
  // TODO(ravishankarm): The SingleBlockImplicitTerminator<"HAL::ReturnOp">
  // should generate this check, but it doesnt.
  auto returnOp = dyn_cast<ReturnOp>(region->front().getTerminator());
  if (!returnOp || returnOp.operands().size() != 3) {
    return op.emitOpError("expected operation to yield 3 values");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// hal.executable.variant
//===----------------------------------------------------------------------===//

void ExecutableVariantOp::build(OpBuilder &builder, OperationState &state,
                                StringRef symName,
                                StringRef targetBackendFilter) {
  ensureTerminator(*state.addRegion(), builder, state.location);
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(symName));
  state.addAttribute("target_backend_filter",
                     builder.getStringAttr(targetBackendFilter));
}

static ParseResult parseExecutableVariantOp(OpAsmParser &parser,
                                            OperationState *result) {
  auto *body = result->addRegion();
  StringAttr nameAttr;
  StringAttr targetBackendFilterAttr;
  if (failed(parser.parseSymbolName(nameAttr,
                                    mlir::SymbolTable::getSymbolAttrName(),
                                    result->attributes)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("filter")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(targetBackendFilterAttr,
                                   "target_backend_filter",
                                   result->attributes)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }

  OptionalParseResult parseResult = parser.parseOptionalRegion(*body);
  if (parseResult.hasValue() && failed(*parseResult)) {
    return failure();
  }

  // Ensure that this module has a valid terminator.
  ExecutableVariantOp::ensureTerminator(*body, parser.getBuilder(),
                                        result->location);
  return success();
}

static void printExecutableVariantOp(OpAsmPrinter &p, ExecutableVariantOp op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.sym_name());
  p << ", filter=\"" << op.target_backend_filter() << "\"";
  p.printOptionalAttrDictWithKeyword(
      op->getAttrs(),
      /*elidedAttrs=*/{mlir::SymbolTable::getSymbolAttrName(),
                       "target_backend_filter"});
  if (!op.body().empty()) {
    p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
  }
}

//===----------------------------------------------------------------------===//
// hal.executable.binary
//===----------------------------------------------------------------------===//

void ExecutableBinaryOp::build(OpBuilder &builder, OperationState &state,
                               StringRef symName, StringRef format,
                               std::vector<uint8_t> data) {
  ensureTerminator(*state.addRegion(), builder, state.location);
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(symName));
  state.addAttribute("format", builder.getStringAttr(format));
  state.addAttribute("data",
                     DenseIntElementsAttr::get(
                         VectorType::get({static_cast<int64_t>(data.size())},
                                         builder.getIntegerType(8)),
                         data));
}

void ExecutableBinaryOp::build(OpBuilder &builder, OperationState &state,
                               StringRef symName, StringAttr format,
                               DenseIntElementsAttr data) {
  ensureTerminator(*state.addRegion(), builder, state.location);
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(symName));
  state.addAttribute("format", format);
  state.addAttribute("data", data);
}

static ParseResult parseExecutableBinaryOp(OpAsmParser &parser,
                                           OperationState *result) {
  auto *body = result->addRegion();
  StringAttr nameAttr;
  if (failed(parser.parseSymbolName(nameAttr,
                                    mlir::SymbolTable::getSymbolAttrName(),
                                    result->attributes)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  OptionalParseResult parseResult = parser.parseOptionalRegion(*body);
  if (parseResult.hasValue() && failed(*parseResult)) {
    return failure();
  }

  // Ensure that this module has a valid terminator.
  ExecutableBinaryOp::ensureTerminator(*body, parser.getBuilder(),
                                       result->location);
  return success();
}

static void printExecutableBinaryOp(OpAsmPrinter &p, ExecutableBinaryOp op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.sym_name());
  p.printOptionalAttrDictWithKeyword(
      op->getAttrs(),
      /*elidedAttrs=*/{mlir::SymbolTable::getSymbolAttrName()});
  if (!op.body().empty()) {
    p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
  }
}

static LogicalResult verifyExecutableBinaryOp(ExecutableBinaryOp op) {
  // Zero or one ModuleOps allowed.
  if (std::distance(op.getBlock().getOps<ModuleOp>().begin(),
                    op.getBlock().getOps<ModuleOp>().end()) > 1) {
    return op.emitOpError() << "expects zero or one nested std.module ops";
  }

  // TODO(benvanik): check export name conflicts.
  return success();
}

//===----------------------------------------------------------------------===//
// hal.executable.create
//===----------------------------------------------------------------------===//

void ExecutableCreateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), StringRef("exe"));
}

//===----------------------------------------------------------------------===//
// hal.executable.lookup
//===----------------------------------------------------------------------===//

void ExecutableLookupOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "exe");
}

//===----------------------------------------------------------------------===//
// hal.interface
//===----------------------------------------------------------------------===//

void InterfaceOp::build(OpBuilder &builder, OperationState &state,
                        StringRef name, IntegerAttr pushConstants) {
  ensureTerminator(*state.addRegion(), builder, state.location);
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  if (pushConstants) {
    state.addAttribute("push_constants", pushConstants);
  }
}

static ParseResult parseInterfaceOp(OpAsmParser &parser,
                                    OperationState *result) {
  StringAttr nameAttr;
  if (failed(parser.parseSymbolName(nameAttr,
                                    mlir::SymbolTable::getSymbolAttrName(),
                                    result->attributes)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }

  // Parse the module body.
  auto *body = result->addRegion();
  if (failed(parser.parseRegion(*body, llvm::None, llvm::None))) {
    return failure();
  }

  // Ensure that this module has a valid terminator.
  InterfaceOp::ensureTerminator(*body, parser.getBuilder(), result->location);
  return success();
}

static void printInterfaceOp(OpAsmPrinter &p, InterfaceOp op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.sym_name());
  p.printOptionalAttrDictWithKeyword(
      op->getAttrs(),
      /*elidedAttrs=*/{mlir::SymbolTable::getSymbolAttrName()});
  p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

ArrayAttr InterfaceOp::getExecutableSetLayoutsAttr() {
  Builder builder(getContext());
  SmallVector<SmallVector<Attribute, 4>, 4> setAttrs;
  for (auto bindingOp : getBlock().getOps<InterfaceBindingOp>()) {
    unsigned set = bindingOp.set().getZExtValue();
    unsigned binding = bindingOp.binding().getZExtValue();
    if (set >= setAttrs.size()) setAttrs.resize(set + 1);
    auto &bindingAttrs = setAttrs[set];
    if (binding >= bindingAttrs.size()) bindingAttrs.resize(binding + 1);
    bindingAttrs[binding] = DescriptorSetLayoutBindingAttr::get(
        bindingOp.bindingAttr(), bindingOp.typeAttr(), bindingOp.accessAttr());
  }
  return builder.getArrayAttr(llvm::to_vector<4>(
      llvm::map_range(setAttrs, [&](ArrayRef<Attribute> bindingsArray) {
        return builder.getArrayAttr(bindingsArray).cast<Attribute>();
      })));
}

bool InterfaceOp::isEquivalentTo(InterfaceOp other) {
  auto bindings = llvm::to_vector<4>(getBlock().getOps<InterfaceBindingOp>());
  auto otherBindings =
      llvm::to_vector<4>(other.getBlock().getOps<InterfaceBindingOp>());
  return push_constantsAttr() == other.push_constantsAttr() &&
         bindings.size() == otherBindings.size() &&
         llvm::all_of(llvm::zip(bindings, otherBindings), [](auto bindings) {
           return OperationEquivalence::isEquivalentTo(std::get<0>(bindings),
                                                       std::get<1>(bindings));
         });
}

//===----------------------------------------------------------------------===//
// hal.interface.binding
//===----------------------------------------------------------------------===//

static ParseResult parseInterfaceBindingOp(OpAsmParser &parser,
                                           OperationState *result) {
  StringAttr nameAttr;
  IntegerAttr setAttr;
  IntegerAttr bindingAttr;
  if (failed(parser.parseSymbolName(nameAttr,
                                    mlir::SymbolTable::getSymbolAttrName(),
                                    result->attributes)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("set")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(setAttr, parser.getBuilder().getIndexType(),
                                   "set", result->attributes)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("binding")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(bindingAttr,
                                   parser.getBuilder().getIndexType(),
                                   "binding", result->attributes)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("type")) ||
      failed(parser.parseEqual()) ||
      failed(
          parseEnumAttr<DescriptorType>(parser, "type", result->attributes)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("access")) ||
      failed(parser.parseEqual()) ||
      failed(parseEnumAttr<MemoryAccessBitfield>(parser, "access",
                                                 result->attributes)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

static void printInterfaceBindingOp(OpAsmPrinter &p, InterfaceBindingOp op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.sym_name());
  p << ", set=" << op.set();
  p << ", binding=" << op.binding();
  p << ", type=\"" << stringifyDescriptorType(op.type()) << "\"";
  p << ", access=\"" << stringifyMemoryAccessBitfield(op.access()) << "\"";
  p.printOptionalAttrDictWithKeyword(op->getAttrs(),
                                     /*elidedAttrs=*/{
                                         mlir::SymbolTable::getSymbolAttrName(),
                                         "set",
                                         "binding",
                                         "type",
                                         "access",
                                     });
}

//===----------------------------------------------------------------------===//
// hal.interface.binding.subspan
//===----------------------------------------------------------------------===//

InterfaceBindingOp InterfaceBindingSubspanOp::queryBindingOp() {
  return dyn_cast_or_null<InterfaceBindingOp>(
      SymbolTable::lookupNearestSymbolFrom(getOperation(), binding()));
}

//===----------------------------------------------------------------------===//
// hal.interface.workgroup.*
//===----------------------------------------------------------------------===//

static void getAsmResultNamesForInterfaceWorkgroupOp(
    StringRef prefix, const APInt &dimension, Value result,
    function_ref<void(Value, StringRef)> setNameFn) {
  switch (dimension.getZExtValue()) {
    case 0:
      setNameFn(result, (prefix + "x").str());
      return;
    case 1:
      setNameFn(result, (prefix + "y").str());
      return;
    case 2:
      setNameFn(result, (prefix + "z").str());
      return;
  }
}

void InterfaceWorkgroupIDOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  getAsmResultNamesForInterfaceWorkgroupOp("workgroup_id_", dimension(),
                                           result(), setNameFn);
}

void InterfaceWorkgroupCountOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  getAsmResultNamesForInterfaceWorkgroupOp("workgroup_count_", dimension(),
                                           result(), setNameFn);
}

void InterfaceWorkgroupSizeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  getAsmResultNamesForInterfaceWorkgroupOp("workgroup_size_", dimension(),
                                           result(), setNameFn);
}

//===----------------------------------------------------------------------===//
// hal.executable_layout.create
//===----------------------------------------------------------------------===//

void ExecutableLayoutCreateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "executable_layout");
}

//===----------------------------------------------------------------------===//
// hal.executable_layout.lookup
//===----------------------------------------------------------------------===//

void ExecutableLayoutLookupOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "executable_layout");
}

//===----------------------------------------------------------------------===//
// hal.semaphore.create
//===----------------------------------------------------------------------===//

void SemaphoreCreateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "semaphore");
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/HAL/IR/HALOps.cpp.inc"
