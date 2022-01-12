// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
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
// hal.tensor.import/export
//===----------------------------------------------------------------------===//

void TensorImportOp::build(OpBuilder &builder, OperationState &result,
                           Type resultType, Value source) {
  auto shapedType = resultType.cast<ShapedType>();
  assert((source.getType().isa<IREE::HAL::BufferViewType>() ||
          shapedType.hasStaticShape()) &&
         "can only use this constructor for buffer views when shape "
         "information is required");
  SmallVector<Value> dynamicDims;
  for (int64_t i = 0; i < shapedType.getRank(); ++i) {
    if (!shapedType.isDynamicDim(i)) continue;
    dynamicDims.push_back(builder.createOrFold<IREE::HAL::BufferViewDimOp>(
        result.location, builder.getIndexType(), source,
        builder.getIndexAttr(i)));
  }
  build(builder, result, resultType, source, TypeAttr::get(shapedType),
        dynamicDims);
}

Value TensorImportOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(source());
}

::llvm::Optional<unsigned> TensorImportOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // source
}

SmallVector<int64_t, 4> TensorImportOp::getTiedResultOperandIndices() {
  return {0};  // source
}

static LogicalResult verifyTypeStorageCompatibility(Operation *op,
                                                    Type encodingType,
                                                    Type storageType) {
  if (encodingType == storageType) return success();
  auto encodingShapedType = encodingType.dyn_cast<ShapedType>();
  auto storageShapedType = storageType.dyn_cast<ShapedType>();
  if (!encodingShapedType || !storageShapedType) return success();

  if (IREE::Util::getRoundedElementByteWidth(
          encodingShapedType.getElementType()) !=
      IREE::Util::getRoundedElementByteWidth(
          storageShapedType.getElementType())) {
    // TODO(benvanik): more sophisticated logic here. There are a lot of valid
    // cases that are difficult to account for here statically; for example,
    // packing 8xi1 into 1xi8 or complex<f32> into 2xf32. We could try to guess
    // the element count (at least the static part of it) and ensure the scaling
    // matches but that wouldn't account for user variance. Really with this op
    // we are letting the _user_ control the bitcasting and type reflection and
    // purposefully don't want to mess with it (users should be able to put
    // custom types here, etc).
    //
    // NOTE: we round to bytes first as the base type (such as i1) may not be
    // representable in an external form.
    // return op->emitOpError() << "encoding and storage types must be "
    //                             "bitcastable; adjusted encoding bit width "
    //                             "of "
    //                          << encodingShapedType.getElementTypeBitWidth()
    //                          << " != adjusted storage bit width of "
    //                          << storageShapedType.getElementTypeBitWidth();
  }

  if (encodingShapedType.getNumDynamicDims() !=
      storageShapedType.getNumDynamicDims()) {
    // NOTE: we implicitly require that the dimensions are equivalent but
    // dont actually care about their order. For example, tensor<?x1xf32> is
    // compatible with tensor<?xf32>.
    return op->emitOpError()
           << "encoding and storage types must have the same "
              "dynamic dimension values; encoding shape "
           << encodingShapedType << " incompatible with storage shape "
           << storageShapedType;
  }

  return success();
}

static LogicalResult verifyTensorImportOp(TensorImportOp op) {
  auto targetType = op.target().getType().cast<TensorType>();
  if (targetType.getNumDynamicDims() != op.target_dims().size()) {
    return op->emitOpError() << "number of target_dims must match number of "
                                "dynamic dims in target type";
  }
  return verifyTypeStorageCompatibility(op, op.target_encoding(), targetType);
}

void TensorExportOp::build(OpBuilder &builder, OperationState &result,
                           Type resultType, Value source) {
  auto dynamicDims =
      IREE::Util::buildDynamicDimsForValue(result.location, source, builder);
  build(builder, result, resultType, source, TypeAttr::get(source.getType()),
        dynamicDims, /*target_storage=*/nullptr);
}

Value TensorExportOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(source());
}

::llvm::Optional<unsigned> TensorExportOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // source
}

SmallVector<int64_t, 4> TensorExportOp::getTiedResultOperandIndices() {
  return {0};  // source
}

static LogicalResult verifyTensorExportOp(TensorExportOp op) {
  auto sourceType = op.source().getType().cast<TensorType>();
  if (sourceType.getNumDynamicDims() != op.source_dims().size()) {
    return op->emitOpError() << "number of source_dims must match number of "
                                "dynamic dims in source type";
  }
  return verifyTypeStorageCompatibility(op, op.source_encoding(),
                                        op.source().getType());
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
// hal.allocator.map
//===----------------------------------------------------------------------===//

void AllocatorMapOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "mapped");
}

Value AllocatorMapOp::getOperandSize(unsigned idx) { return {}; }

Value AllocatorMapOp::getResultSize(unsigned idx) { return length(); }

//===----------------------------------------------------------------------===//
// hal.allocator.try_map
//===----------------------------------------------------------------------===//

void AllocatorTryMapOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(did_map(), "did_map");
  setNameFn(result(), "mapped");
}

Value AllocatorTryMapOp::getOperandSize(unsigned idx) { return {}; }

Value AllocatorTryMapOp::getResultSize(unsigned idx) { return length(); }

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
        builder.createOrFold<arith::ConstantIntOp>(state.location, elementType,
                                                   32),
        builder.createOrFold<arith::ConstantIntOp>(state.location, encodingType,
                                                   32),
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
        builder.createOrFold<arith::ConstantIndexOp>(state.location, set),
        bindings);
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
    bindingOrdinals.push_back(binding.ordinal);
    bindingBuffers.push_back(binding.buffer);
    bindingOffsets.push_back(binding.byteOffset);
    bindingLengths.push_back(binding.byteLength);
  }
  state.addOperands(bindingOrdinals);
  state.addOperands(bindingBuffers);
  state.addOperands(bindingOffsets);
  state.addOperands(bindingLengths);
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
    bindingOrdinals.push_back(binding.ordinal);
    bindingBuffers.push_back(binding.buffer);
    bindingOffsets.push_back(binding.byteOffset);
    bindingLengths.push_back(binding.byteLength);
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

static LogicalResult verifyExecutableOp(ExecutableOp op) {
  // TODO(benvanik): check export name conflicts.
  return success();
}

//===----------------------------------------------------------------------===//
// hal.executable.entry_point
//===----------------------------------------------------------------------===//

static ParseResult parseExecutableEntryPointOp(OpAsmParser &parser,
                                               OperationState *result) {
  StringAttr visibilityAttr;
  if (failed(parseSymbolVisibility(parser, visibilityAttr))) {
    return failure();
  }

  StringAttr nameAttr;
  IREE::HAL::ExecutableLayoutAttr layoutAttr;
  if (failed(parser.parseSymbolName(nameAttr,
                                    mlir::SymbolTable::getSymbolAttrName(),
                                    result->attributes))) {
    return failure();
  }
  if (succeeded(parser.parseOptionalKeyword("ordinal"))) {
    IntegerAttr ordinalAttr;
    if (failed(parser.parseLParen()) ||
        failed(parser.parseAttribute(ordinalAttr,
                                     parser.getBuilder().getIndexType())) ||
        failed(parser.parseRParen())) {
      return failure();
    }
    result->addAttribute("ordinal", ordinalAttr);
  }
  if (failed(parser.parseKeyword("layout")) || failed(parser.parseLParen()) ||
      failed(parser.parseAttribute(layoutAttr)) ||
      failed(parser.parseRParen()) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  result->addAttribute("layout", layoutAttr);

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
  printSymbolVisibility(p, op, op->getAttrOfType<StringAttr>("sym_visibility"));
  p << ' ';
  p.printSymbolName(op.sym_name());
  if (op.ordinalAttr()) {
    p << " ordinal(";
    p.printAttributeWithoutType(op.ordinalAttr());
    p << ")";
  }
  p << " layout(";
  p.printAttribute(op.layout());
  p << ")";
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"sym_name", "layout", "ordinal"});
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
// hal.interface.binding.subspan
//===----------------------------------------------------------------------===//

static LogicalResult verifyInterfaceBindingSubspanOp(
    InterfaceBindingSubspanOp op) {
  if (ShapedType shapedType = op.getType().dyn_cast<ShapedType>()) {
    if (shapedType.getNumDynamicDims() != op.dynamic_dims().size()) {
      return op.emitOpError("result type ")
             << op.getType() << " has " << shapedType.getNumDynamicDims()
             << " dynamic dimensions but " << op.dynamic_dims().size()
             << " associated dimension SSA values";
    }
  }

  return success();
}

// TODO(benvanik): share with align op folder and analysis.
// May need an interface for querying the alignment from ops that can carry it.

// Tries to find the alignment of the given |value| based on either the IR
// structure or annotations.
static llvm::Optional<APInt> lookupValueOrAlignment(Value value) {
  APInt constantValue;
  if (matchPattern(value, m_ConstantInt(&constantValue))) {
    // Value is constant and we can just treat that as if it were an alignment.
    return constantValue;
  }

  auto op = value.getDefiningOp();
  if (auto loadOp = dyn_cast_or_null<IREE::HAL::InterfaceConstantLoadOp>(op)) {
    // Push constants have an optional value alignment.
    auto alignment = loadOp.alignment();
    if (alignment.hasValue()) return alignment;
  } else if (auto alignmentAttr =
                 op->getAttrOfType<IntegerAttr>("stream.alignment")) {
    // The op has an alignment tagged on it we can use directly.
    return alignmentAttr.getValue();
  }

  // TODO(benvanik): more searching.
  return llvm::None;
}

llvm::Align InterfaceBindingSubspanOp::calculateAlignment() {
  // If we can't calculate an alignment we fall back to the natural alignment of
  // the element type (for example, a memref<?xi32> is known to be at least
  // 4-byte aligned).
  llvm::Align naturalAlignment(1);
  auto resultType = getType();
  if (auto shapedType = resultType.dyn_cast<ShapedType>()) {
    naturalAlignment = llvm::Align(
        IREE::Util::getRoundedElementByteWidth(shapedType.getElementType()));
  }

  // If the binding has no assigned alignment we fall back to natural alignment.
  auto bindingAlignmentInt = alignment();
  if (!bindingAlignmentInt) return naturalAlignment;
  auto bindingAlignment =
      llvm::Align(bindingAlignmentInt.getValue().getZExtValue());

  // If there's no offset specified then we can use the binding alignment
  // directly.
  if (!byte_offset()) return bindingAlignment;

  // Try to get the alignment of the byte offset. If it's a constant then we can
  // find a common alignment between it and the base and otherwise we need to
  // try to infer the alignment from the IR - otherwise we fall back.
  auto offsetOrAlignment = lookupValueOrAlignment(byte_offset());
  if (!offsetOrAlignment.hasValue()) return naturalAlignment;

  // Compute the common alignment between that of the binding base and that of
  // the byte offset.
  return llvm::commonAlignment(bindingAlignment,
                               offsetOrAlignment->getZExtValue());
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
