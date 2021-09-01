// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/Hashing.h"
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
  return IREE::Util::TiedOpInterface::findTiedBaseValue(source());
}

::llvm::Optional<unsigned> TensorCastOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // source
}

SmallVector<int64_t, 4> TensorCastOp::getTiedResultOperandIndices() {
  return {0};  // source
}

//===----------------------------------------------------------------------===//
// hal.allocator.compute_size
//===----------------------------------------------------------------------===//

void AllocatorComputeSizeOp::build(OpBuilder &builder, OperationState &state,
                                   Value allocator, ValueRange shape,
                                   int32_t elementType, int32_t encodingType) {
  build(builder, state, allocator, shape,
        builder.createOrFold<ConstantIntOp>(state.location, elementType, 32),
        builder.createOrFold<ConstantIntOp>(state.location, encodingType, 32));
}

void AllocatorComputeSizeOp::build(OpBuilder &builder, OperationState &state,
                                   Value allocator, ValueRange shape,
                                   Value elementType, Value encodingType) {
  state.addOperands({allocator});
  state.addOperands(shape);
  state.addOperands(elementType);
  state.addOperands(encodingType);
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
                                     int32_t elementType, int32_t encodingType,
                                     ValueRange indices) {
  build(builder, state, allocator, shape,
        builder.createOrFold<ConstantIntOp>(state.location, elementType, 32),
        builder.createOrFold<ConstantIntOp>(state.location, encodingType, 32),
        indices);
}

void AllocatorComputeOffsetOp::build(OpBuilder &builder, OperationState &state,
                                     Value allocator, ValueRange shape,
                                     Value elementType, Value encodingType,
                                     ValueRange indices) {
  state.addOperands({allocator});
  state.addOperands(shape);
  state.addOperands(elementType);
  state.addOperands(encodingType);
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
                                    int32_t elementType, int32_t encodingType,
                                    ValueRange indices, ValueRange lengths) {
  build(builder, state, allocator, shape,
        builder.createOrFold<ConstantIntOp>(state.location, elementType, 32),
        builder.createOrFold<ConstantIntOp>(state.location, encodingType, 32),
        indices, lengths);
}

void AllocatorComputeRangeOp::build(OpBuilder &builder, OperationState &state,
                                    Value allocator, ValueRange shape,
                                    Value elementType, Value encodingType,
                                    ValueRange indices, ValueRange lengths) {
  state.addOperands({allocator});
  state.addOperands(shape);
  state.addOperands(elementType);
  state.addOperands(encodingType);
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
                               int32_t encodingType, ValueRange shape) {
  build(builder, state, buffer,
        builder.createOrFold<ConstantIntOp>(state.location, elementType, 32),
        builder.createOrFold<ConstantIntOp>(state.location, encodingType, 32),
        shape);
}

void BufferViewCreateOp::build(OpBuilder &builder, OperationState &state,
                               Value buffer, Value elementType,
                               Value encodingType, ValueRange shape) {
  state.addOperands({buffer, elementType, encodingType});
  state.addOperands(shape);
  state.addTypes({BufferViewType::get(builder.getContext())});
}

void BufferViewCreateOp::getAsmResultNames(
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
  p << ' ';
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
  p << ' ';
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
// hal.device.query
//===----------------------------------------------------------------------===//

static LogicalResult verifyDeviceQueryOp(DeviceQueryOp op) {
  if (op.default_value().hasValue()) {
    if (op.default_value()->getType() != op.value().getType()) {
      return op.emitOpError()
             << "type mismatch between result and default value";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// hal.device.switch
//===----------------------------------------------------------------------===//

void DeviceSwitchOp::build(OpBuilder &builder, OperationState &state,
                           TypeRange resultTypes, Value device,
                           ArrayRef<Attribute> conditions,
                           ArrayRef<NamedAttribute> attributes) {
  state.addOperands({device});
  state.addAttribute("conditions", builder.getArrayAttr(conditions));
  for (size_t i = 0; i < conditions.size(); ++i) {
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
  // #hal.device.match.id<"vulkan-v1.?-*"> {
  //   hal.return %c1 : i32
  // }, ...
  SmallVector<Attribute, 4> conditionAttrs;
  do {
    Attribute conditionAttr;
    NamedAttrList dummyAttrs;
    if (failed(parser.parseAttribute(conditionAttr, "condition", dummyAttrs))) {
      return failure();
    }
    conditionAttrs.push_back(conditionAttr);
    SmallVector<OpAsmParser::OperandType> regionArgs;
    SmallVector<Type> regionArgTypes;
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
  p << "<";
  p.printOperand(op.device());
  p << " : ";
  p.printType(op.device().getType());
  p << ">";
  p.printOptionalArrowTypeList(op.getResultTypes());
  p << "\n";
  p.getStream().indent(4);
  interleave(
      llvm::zip(op.conditions(), op.condition_regions()),
      [&](std::tuple<Attribute, Region &> it) {
        auto &conditionAttr = std::get<0>(it);
        auto &conditionRegion = std::get<1>(it);
        p.printAttribute(conditionAttr);
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
  for (auto &region : op.condition_regions()) {
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
  p << ' ';
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
  p << ' ';
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
                                IREE::HAL::ExecutableTargetAttr target) {
  ensureTerminator(*state.addRegion(), builder, state.location);
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(symName));
  state.addAttribute("target", target);
}

static ParseResult parseExecutableVariantOp(OpAsmParser &parser,
                                            OperationState *result) {
  auto *body = result->addRegion();
  StringAttr nameAttr;
  IREE::HAL::ExecutableTargetAttr targetAttr;
  if (failed(parser.parseSymbolName(nameAttr,
                                    mlir::SymbolTable::getSymbolAttrName(),
                                    result->attributes)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("target")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(targetAttr, "target", result->attributes)) ||
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
  p << ' ';
  p.printSymbolName(op.sym_name());
  p << ", target = " << op.target();
  p.printOptionalAttrDictWithKeyword(
      op->getAttrs(),
      /*elidedAttrs=*/{mlir::SymbolTable::getSymbolAttrName(), "target"});
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
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(symName));
  state.addAttribute("format", format);
  state.addAttribute("data", data);
}

static ParseResult parseExecutableBinaryOp(OpAsmParser &parser,
                                           OperationState *result) {
  StringAttr nameAttr;
  if (failed(parser.parseSymbolName(nameAttr,
                                    mlir::SymbolTable::getSymbolAttrName(),
                                    result->attributes)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

static void printExecutableBinaryOp(OpAsmPrinter &p, ExecutableBinaryOp op) {
  p << ' ';
  p.printSymbolName(op.sym_name());
  p.printOptionalAttrDictWithKeyword(
      op->getAttrs(),
      /*elidedAttrs=*/{mlir::SymbolTable::getSymbolAttrName()});
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
  p << ' ';
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
           return OperationEquivalence::isEquivalentTo(
               std::get<0>(bindings), std::get<1>(bindings),
               OperationEquivalence::exactValueMatch,
               OperationEquivalence::exactValueMatch,
               OperationEquivalence::Flags::IgnoreLocations);
         });
}

llvm::hash_code InterfaceOp::getInterfaceHash() {
  auto range = llvm::map_range(getBlock().getOps<InterfaceBindingOp>(),
                               [](InterfaceBindingOp bindingOp) {
                                 return bindingOp.getDescriptorHash();
                               });
  return llvm::hash_combine(
      push_constants(), llvm::hash_combine_range(range.begin(), range.end()));
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
  p << ' ';
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

llvm::hash_code InterfaceBindingOp::getDescriptorHash() {
  // Use the unwrapped attribute accessors so that we can have determinstic
  // hashes. Hashing against the wrapped attributes are hashing against pointer
  // values, which change per run.
  return llvm::hash_combine(set(), binding(), type(), access());
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
