// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SMLoc.h"
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
// hal.ex.shared_device
//===----------------------------------------------------------------------===//

void ExSharedDeviceOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "dev");
}

//===----------------------------------------------------------------------===//
// hal.make_memory_barrier
//===----------------------------------------------------------------------===//

void MakeMemoryBarrierOp::build(OpBuilder &builder, OperationState &state,
                                IREE::HAL::AccessScopeBitfield sourceScope,
                                IREE::HAL::AccessScopeBitfield targetScope) {
  state.addAttribute("source_scope", builder.getI32IntegerAttr(
                                         static_cast<int32_t>(sourceScope)));
  state.addAttribute("target_scope", builder.getI32IntegerAttr(
                                         static_cast<int32_t>(targetScope)));
  state.addTypes({MemoryBarrierType::get(builder.getContext())});
}

void MakeMemoryBarrierOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "memory_barrier");
}

//===----------------------------------------------------------------------===//
// hal.make_buffer_barrier
//===----------------------------------------------------------------------===//

void MakeBufferBarrierOp::build(OpBuilder &builder, OperationState &state,
                                IREE::HAL::AccessScopeBitfield sourceScope,
                                IREE::HAL::AccessScopeBitfield targetScope,
                                Value buffer, Value offset, Value length) {
  state.addAttribute("source_scope", builder.getI32IntegerAttr(
                                         static_cast<int32_t>(sourceScope)));
  state.addAttribute("target_scope", builder.getI32IntegerAttr(
                                         static_cast<int32_t>(targetScope)));
  state.addOperands({buffer, offset, length});
  state.addTypes({BufferBarrierType::get(builder.getContext())});
}

void MakeBufferBarrierOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "buffer_barrier");
}

//===----------------------------------------------------------------------===//
// hal.variable
//===----------------------------------------------------------------------===//

// Returns true if the given |accessType| is compatible with the |variableType|.
// For example, this will return true if the variable type is a tensor<?xf32>
// and the access is tensor<4xf32>.
static bool isVariableTypeCompatible(Type variableType, Type accessType) {
  return succeeded(mlir::verifyCompatibleShape(variableType, accessType));
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

  if (failed(parser.parseOptionalColon())) {
    Attribute initialValueAttr;
    if (failed(parser.parseAttribute(initialValueAttr, "initial_value",
                                     result->attributes))) {
      return failure();
    }
    result->addAttribute("type", TypeAttr::get(initialValueAttr.getType()));
  } else {
    Type type;
    if (failed(parser.parseType(type))) {
      return failure();
    }
    result->addAttribute("type", TypeAttr::get(type));
  }

  if (failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }

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
  if (op.initial_value().hasValue()) {
    p << ' ';
    p.printAttribute(op.initial_value().getValue());
  } else {
    p << " : ";
    p.printType(op.type());
  }
  p.printOptionalAttrDictWithKeyword(op.getAttrs(), /*elidedAttrs=*/{
                                         "sym_name",
                                         "type",
                                         "is_mutable",
                                         "initializer",
                                         "initial_value",
                                     });
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
  } else if (op.initial_value().hasValue()) {
    // Ensure the value is something we can store in the variable
    if (!isVariableTypeCompatible(op.type(), op.initial_value()->getType())) {
      return op.emitOpError()
             << "initial value type mismatch; variable " << op.sym_name()
             << " is " << op.type() << " but initial value provided is "
             << op.initial_value()->getType();
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
  state.addOperands({allocator});
  state.addOperands(shape);
  state.addAttribute("element_type", builder.getI32IntegerAttr(elementType));
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
  state.addOperands({allocator});
  state.addOperands(shape);
  state.addAttribute("element_type", builder.getI32IntegerAttr(elementType));
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
  state.addOperands({allocator});
  state.addOperands(shape);
  state.addAttribute("element_type", builder.getI32IntegerAttr(elementType));
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

void AllocatorAllocateOp::build(OpBuilder &builder, OperationState &state,
                                Value allocator,
                                IREE::HAL::MemoryTypeBitfield memoryTypes,
                                IREE::HAL::BufferUsageBitfield bufferUsage,
                                Value allocationSize) {
  state.addOperands({allocator, allocationSize});
  state.addAttribute("memory_types", builder.getI32IntegerAttr(
                                         static_cast<int32_t>(memoryTypes)));
  state.addAttribute("buffer_usage", builder.getI32IntegerAttr(
                                         static_cast<int32_t>(bufferUsage)));
  state.addTypes({BufferType::get(builder.getContext())});
}

void AllocatorAllocateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "buffer");
}

//===----------------------------------------------------------------------===//
// hal.allocator.allocate.const
//===----------------------------------------------------------------------===//

void AllocatorAllocateConstOp::build(OpBuilder &builder, OperationState &state,
                                     Value allocator,
                                     IREE::HAL::MemoryTypeBitfield memoryTypes,
                                     IREE::HAL::BufferUsageBitfield bufferUsage,
                                     ElementsAttr value) {
  state.addOperands({allocator});
  state.addAttribute("memory_types", builder.getI32IntegerAttr(
                                         static_cast<int32_t>(memoryTypes)));
  state.addAttribute("buffer_usage", builder.getI32IntegerAttr(
                                         static_cast<int32_t>(bufferUsage)));
  state.addAttribute("value", value);
  state.addTypes({BufferType::get(builder.getContext())});
}

void AllocatorAllocateConstOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "cbuffer");
}

//===----------------------------------------------------------------------===//
// hal.buffer.allocator
//===----------------------------------------------------------------------===//

void BufferAllocatorOp::build(OpBuilder &builder, OperationState &state,
                              Value buffer) {
  state.addOperands({buffer});
  state.addTypes({AllocatorType::get(builder.getContext())});
}

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

//===----------------------------------------------------------------------===//
// hal.buffer_view.const
//===----------------------------------------------------------------------===//

void BufferViewConstOp::build(OpBuilder &builder, OperationState &state,
                              Value allocator,
                              IREE::HAL::MemoryTypeBitfield memoryTypes,
                              IREE::HAL::BufferUsageBitfield bufferUsage,
                              ElementsAttr value) {
  state.addOperands({allocator});
  state.addAttribute("memory_types", builder.getI32IntegerAttr(
                                         static_cast<int32_t>(memoryTypes)));
  state.addAttribute("buffer_usage", builder.getI32IntegerAttr(
                                         static_cast<int32_t>(bufferUsage)));
  state.addAttribute("value", value);
  state.addTypes({BufferViewType::get(builder.getContext())});
}

void BufferViewConstOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "view");
}

//===----------------------------------------------------------------------===//
// hal.buffer_view.create
//===----------------------------------------------------------------------===//

void BufferViewCreateOp::build(OpBuilder &builder, OperationState &state,
                               Value buffer, ValueRange shape,
                               int32_t elementType) {
  state.addOperands({buffer});
  state.addOperands(shape);
  state.addAttribute("element_type", builder.getI32IntegerAttr(elementType));
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

void BufferViewBufferOp::build(OpBuilder &builder, OperationState &state,
                               Value bufferView) {
  state.addOperands({bufferView});
  state.addTypes({BufferType::get(builder.getContext())});
}

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

void CommandBufferCreateOp::build(
    OpBuilder &builder, OperationState &state, Value device,
    IREE::HAL::CommandBufferModeBitfield modes,
    IREE::HAL::CommandCategoryBitfield commandCategories) {
  state.addOperands({device});
  state.addAttribute("modes",
                     builder.getI32IntegerAttr(static_cast<int32_t>(modes)));
  state.addAttribute(
      "command_categories",
      builder.getI32IntegerAttr(static_cast<int32_t>(commandCategories)));
  state.addTypes({CommandBufferType::get(builder.getContext())});
}

void CommandBufferCreateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "cmd");
}

//===----------------------------------------------------------------------===//
// hal.command_buffer.execution_barrier
//===----------------------------------------------------------------------===//

void CommandBufferExecutionBarrierOp::build(
    OpBuilder &builder, OperationState &state, Value commandBuffer,
    IREE::HAL::ExecutionStageBitfield sourceStageMask,
    IREE::HAL::ExecutionStageBitfield targetStageMask,
    ValueRange memoryBarriers, ValueRange bufferBarriers) {
  state.addAttribute(
      "source_stage_mask",
      builder.getI32IntegerAttr(static_cast<int32_t>(sourceStageMask)));
  state.addAttribute(
      "target_stage_mask",
      builder.getI32IntegerAttr(static_cast<int32_t>(targetStageMask)));
  state.addOperands(commandBuffer);
  state.addOperands(memoryBarriers);
  state.addOperands(bufferBarriers);
  state.addAttribute("operand_segment_sizes",
                     DenseIntElementsAttr::get(
                         VectorType::get({3}, builder.getIntegerType(32)),
                         {1, static_cast<int>(memoryBarriers.size()),
                          static_cast<int>(bufferBarriers.size())}));
}

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
                                                   result->attributes))) {
    return failure();
  }
  SmallVector<OpAsmParser::OperandType, 4> memoryBarriers;
  bool expectMoreOperands = succeeded(parser.parseOptionalComma());
  if (expectMoreOperands &&
      succeeded(parser.parseOptionalKeyword("memory_barriers"))) {
    if (failed(parser.parseEqual()) || failed(parser.parseLSquare()) ||
        failed(parser.parseOperandList(memoryBarriers)) ||
        failed(parser.parseRSquare()) ||
        failed(parser.resolveOperands(
            memoryBarriers, MemoryBarrierType::get(result->getContext()),
            result->operands))) {
      return failure();
    }
    expectMoreOperands = succeeded(parser.parseOptionalComma());
  }
  SmallVector<OpAsmParser::OperandType, 4> bufferBarriers;
  if (expectMoreOperands &&
      succeeded(parser.parseOptionalKeyword("buffer_barriers"))) {
    if (failed(parser.parseEqual()) || failed(parser.parseLSquare()) ||
        failed(parser.parseOperandList(bufferBarriers)) ||
        failed(parser.parseRSquare()) ||
        failed(parser.resolveOperands(
            bufferBarriers, BufferBarrierType::get(result->getContext()),
            result->operands))) {
      return failure();
    }
    expectMoreOperands = succeeded(parser.parseOptionalComma());
  }
  result->addAttribute(
      "operand_segment_sizes",
      DenseIntElementsAttr::get(
          VectorType::get({3}, parser.getBuilder().getIntegerType(32)),
          {1, static_cast<int>(memoryBarriers.size()),
           static_cast<int>(bufferBarriers.size())}));
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
  p << "\"";
  if (!op.memory_barriers().empty()) {
    p << ", memory_barriers=[";
    p.printOperands(op.memory_barriers());
    p << "]";
  }
  if (!op.buffer_barriers().empty()) {
    p << ", buffer_barriers=[";
    p.printOperands(op.buffer_barriers());
    p << "]";
  }
  p.printOptionalAttrDictWithKeyword(op.getAttrs(),
                                     /*elidedAttrs=*/{
                                         "source_stage_mask",
                                         "target_stage_mask",
                                         "operand_segment_sizes",
                                     });
}

//===----------------------------------------------------------------------===//
// hal.command_buffer.push_descriptor_set
//===----------------------------------------------------------------------===//

void CommandBufferPushDescriptorSetOp::build(
    OpBuilder &builder, OperationState &state, Value commandBuffer,
    Value executableLayout, uint32_t set,
    ArrayRef<DescriptorSetBindingValue> bindings) {
  state.addOperands({commandBuffer, executableLayout});
  state.addAttribute("set", builder.getI32IntegerAttr(set));
  SmallVector<int32_t, 4> bindingOrdinals;
  SmallVector<Value, 4> bindingBuffers;
  SmallVector<Value, 4> bindingOffsets;
  SmallVector<Value, 4> bindingLengths;
  for (auto binding : bindings) {
    bindingOrdinals.push_back(std::get<0>(binding));
    bindingBuffers.push_back(std::get<1>(binding));
    bindingOffsets.push_back(std::get<2>(binding));
    bindingLengths.push_back(std::get<3>(binding));
  }
  state.addAttribute("bindings", builder.getI32ArrayAttr(bindingOrdinals));
  state.addOperands(bindingBuffers);
  state.addOperands(bindingOffsets);
  state.addOperands(bindingLengths);
}

static ParseResult parseDescriptorSetBindings(OpAsmParser &parser,
                                              OperationState *result) {
  auto i32Type = parser.getBuilder().getIntegerType(32);
  auto indexType = parser.getBuilder().getIndexType();
  SmallVector<Attribute, 4> bindingAttrs;
  do {
    IntegerAttr bindingAttr;
    NamedAttrList attrList;
    OpAsmParser::OperandType buffer;
    OpAsmParser::OperandType bufferOffset;
    OpAsmParser::OperandType bufferLength;
    if (failed(
            parser.parseAttribute(bindingAttr, i32Type, "binding", attrList)) ||
        failed(parser.parseEqual()) || failed(parser.parseLParen()) ||
        failed(parser.parseOperand(buffer)) ||
        failed(parser.resolveOperand(
            buffer, BufferType::get(result->getContext()), result->operands)) ||
        failed(parser.parseComma()) ||
        failed(parser.parseOperand(bufferOffset)) ||
        failed(
            parser.resolveOperand(bufferOffset, indexType, result->operands)) ||
        failed(parser.parseComma()) ||
        failed(parser.parseOperand(bufferLength)) ||
        failed(
            parser.resolveOperand(bufferLength, indexType, result->operands)) ||
        failed(parser.parseRParen())) {
      return failure();
    }
    bindingAttrs.push_back(bindingAttr);
  } while (succeeded(parser.parseOptionalComma()));
  result->addAttribute("bindings",
                       parser.getBuilder().getArrayAttr(bindingAttrs));
  return success();
}

static ParseResult parseCommandBufferPushDescriptorSetOp(
    OpAsmParser &parser, OperationState *result) {
  OpAsmParser::OperandType commandBuffer;
  OpAsmParser::OperandType executableLayout;
  IntegerAttr setAttr;
  auto operandsLoc = parser.getCurrentLocation();
  if (failed(parser.parseOperand(commandBuffer)) ||
      failed(parser.parseComma()) ||
      failed(parser.parseOperand(executableLayout)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("set")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(setAttr,
                                   parser.getBuilder().getIntegerType(32),
                                   "set", result->attributes)) ||
      failed(parser.parseComma()) ||
      failed(parser.resolveOperands(
          ArrayRef<OpAsmParser::OperandType>{
              commandBuffer,
              executableLayout,
          },
          ArrayRef<Type>{
              CommandBufferType::get(result->getContext()),
              ExecutableLayoutType::get(result->getContext()),
          },
          operandsLoc, result->operands)) ||
      failed(parser.parseKeyword("bindings")) || failed(parser.parseEqual()) ||
      failed(parser.parseLSquare()) ||
      failed(parseDescriptorSetBindings(parser, result)) ||
      failed(parser.parseRSquare()) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

template <typename T>
static void printDescriptorSetBindings(OpAsmPrinter &p, T op) {
  for (int i = 0; i < op.bindings().size(); ++i) {
    p << op.bindings()[i].template cast<IntegerAttr>().getValue();
    p << " = (";
    p.printOperand(op.binding_buffers()[i]);
    p << ", ";
    p.printOperand(op.binding_offsets()[i]);
    p << ", ";
    p.printOperand(op.binding_lengths()[i]);
    p << ")";
    if (i < op.bindings().size() - 1) p << ", ";
  }
}

static void printCommandBufferPushDescriptorSetOp(
    OpAsmPrinter &p, CommandBufferPushDescriptorSetOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.command_buffer());
  p << ", ";
  p.printOperand(op.executable_layout());
  p << ", set=" << op.set();
  p << ", bindings=[";
  printDescriptorSetBindings(p, op);
  p << "]";
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{
                              "set",
                              "bindings",
                          });
}

//===----------------------------------------------------------------------===//
// hal.command_buffer.bind_descriptor_set
//===----------------------------------------------------------------------===//

void CommandBufferBindDescriptorSetOp::build(OpBuilder &builder,
                                             OperationState &state,
                                             Value commandBuffer,
                                             Value executableLayout,
                                             uint32_t set, Value descriptorSet,
                                             ValueRange dynamicOffsets) {
  state.addOperands({commandBuffer, executableLayout, descriptorSet});
  state.addAttribute("set",
                     builder.getIntegerAttr(builder.getIntegerType(32), set));
  state.addOperands(dynamicOffsets);
}

//===----------------------------------------------------------------------===//
// hal.command_buffer.dispatch
//===----------------------------------------------------------------------===//

void CommandBufferDispatchOp::build(
    OpBuilder &builder, OperationState &state, Value commandBuffer,
    Value executable, IREE::HAL::ExecutableEntryPointOp entryPoint,
    Value workgroupX, Value workgroupY, Value workgroupZ) {
  build(builder, state, commandBuffer, executable,
        entryPoint.ordinal().getZExtValue(), workgroupX, workgroupY,
        workgroupZ);
}

void CommandBufferDispatchOp::build(Builder &builder, OperationState &state,
                                    Value commandBuffer, Value executable,
                                    unsigned entryPointOrdinal,
                                    Value workgroupX, Value workgroupY,
                                    Value workgroupZ) {
  state.addOperands(
      {commandBuffer, executable, workgroupX, workgroupY, workgroupZ});
  state.addAttribute("entry_point",
                     builder.getI32IntegerAttr(entryPointOrdinal));
}

//===----------------------------------------------------------------------===//
// hal.descriptor_set.create
//===----------------------------------------------------------------------===//

void DescriptorSetCreateOp::build(
    OpBuilder &builder, OperationState &state, Value device, Value setLayout,
    ArrayRef<DescriptorSetBindingValue> bindings) {
  state.addOperands({device, setLayout});
  SmallVector<int32_t, 4> bindingOrdinals;
  SmallVector<Value, 4> bindingBuffers;
  SmallVector<Value, 4> bindingOffsets;
  SmallVector<Value, 4> bindingLengths;
  for (auto binding : bindings) {
    bindingOrdinals.push_back(std::get<0>(binding));
    bindingBuffers.push_back(std::get<1>(binding));
    bindingOffsets.push_back(std::get<2>(binding));
    bindingLengths.push_back(std::get<3>(binding));
  }
  state.addAttribute("bindings", builder.getI32ArrayAttr(bindingOrdinals));
  state.addOperands(bindingBuffers);
  state.addOperands(bindingOffsets);
  state.addOperands(bindingLengths);
}

static ParseResult parseDescriptorSetCreateOp(OpAsmParser &parser,
                                              OperationState *result) {
  OpAsmParser::OperandType device;
  OpAsmParser::OperandType setLayout;
  auto operandsLoc = parser.getCurrentLocation();
  if (failed(parser.parseOperand(device)) || failed(parser.parseComma()) ||
      failed(parser.parseOperand(setLayout)) || failed(parser.parseComma()) ||
      failed(parser.resolveOperands(
          ArrayRef<OpAsmParser::OperandType>{
              device,
              setLayout,
          },
          ArrayRef<Type>{
              DeviceType::get(result->getContext()),
              DescriptorSetLayoutType::get(result->getContext()),
          },
          operandsLoc, result->operands)) ||
      failed(parser.parseKeyword("bindings")) || failed(parser.parseEqual()) ||
      failed(parser.parseLSquare()) ||
      failed(parseDescriptorSetBindings(parser, result)) ||
      failed(parser.parseRSquare()) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

static void printDescriptorSetCreateOp(OpAsmPrinter &p,
                                       DescriptorSetCreateOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.device());
  p << ", ";
  p.printOperand(op.set_layout());
  p << ", bindings=[";
  printDescriptorSetBindings(p, op);
  p << "]";
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{
                              "bindings",
                          });
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
                           ArrayRef<ValueRange> conditionArgs,
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
  if (failed(parser.parseLParen()) || failed(parser.parseOperand(device)) ||
      failed(parser.parseColonType(deviceType)) ||
      failed(parser.resolveOperand(device, deviceType, result->operands)) ||
      failed(parser.parseRParen()) ||
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
                       ArrayAttr::get(conditionAttrs, result->getContext()));

  if (failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

static void printDeviceSwitchOp(OpAsmPrinter &p, DeviceSwitchOp op) {
  p << op.getOperationName() << "(";
  p.printOperand(op.device());
  p << " : ";
  p.printType(op.device().getType());
  p << ")";
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
  p.printOptionalAttrDictWithKeyword(op.getAttrs(),
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

InterfaceOp ExecutableOp::getInterfaceOp() {
  auto interfaceOps = llvm::to_vector<1>(getBlock().getOps<InterfaceOp>());
  assert(interfaceOps.size() == 1 && "executable must have one interface");
  return interfaceOps.front();
}

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
      op.getAttrs(),
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
  return success();
}

static void printExecutableEntryPointOp(OpAsmPrinter &p,
                                        ExecutableEntryPointOp op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.sym_name());
  p.printOptionalAttrDictWithKeyword(op.getAttrs(),
                                     /*elidedAttrs=*/{"sym_name"});
}

//===----------------------------------------------------------------------===//
// hal.executable.target
//===----------------------------------------------------------------------===//

void ExecutableTargetOp::build(OpBuilder &builder, OperationState &state,
                               StringRef targetBackend) {
  ensureTerminator(*state.addRegion(), builder, state.location);
  state.addAttribute("target_backend", builder.getStringAttr(targetBackend));
}

static ParseResult parseExecutableTargetOp(OpAsmParser &parser,
                                           OperationState *result) {
  auto *body = result->addRegion();
  StringAttr targetBackendAttr;
  if (failed(parser.parseAttribute(targetBackendAttr, "target_backend",
                                   result->attributes)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseOptionalRegion(*body, llvm::None, llvm::None))) {
    return failure();
  }

  // Ensure that this module has a valid terminator.
  ExecutableTargetOp::ensureTerminator(*body, parser.getBuilder(),
                                       result->location);
  return success();
}

static void printExecutableTargetOp(OpAsmPrinter &p, ExecutableTargetOp op) {
  p << op.getOperationName();
  p << " \"" << op.target_backend() << "\"";
  p.printOptionalAttrDictWithKeyword(op.getAttrs(),
                                     /*elidedAttrs=*/{"target_backend"});
  if (!op.body().empty()) {
    p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
  }
}

//===----------------------------------------------------------------------===//
// hal.executable.binary
//===----------------------------------------------------------------------===//

void ExecutableBinaryOp::build(OpBuilder &builder, OperationState &state,
                               uint32_t format, std::vector<uint8_t> data) {
  ensureTerminator(*state.addRegion(), builder, state.location);
  state.addAttribute(
      "format", builder.getIntegerAttr(builder.getIntegerType(32), format));
  state.addAttribute("data",
                     DenseIntElementsAttr::get(
                         VectorType::get({static_cast<int64_t>(data.size())},
                                         builder.getIntegerType(8)),
                         data));
}

static ParseResult parseExecutableBinaryOp(OpAsmParser &parser,
                                           OperationState *result) {
  auto *body = result->addRegion();
  if (failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseOptionalRegion(*body, llvm::None, llvm::None))) {
    return failure();
  }

  // Ensure that this module has a valid terminator.
  ExecutableBinaryOp::ensureTerminator(*body, parser.getBuilder(),
                                       result->location);
  return success();
}

static void printExecutableBinaryOp(OpAsmPrinter &p, ExecutableBinaryOp op) {
  p << op.getOperationName();
  p.printOptionalAttrDictWithKeyword(
      op.getAttrs(),
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
      op.getAttrs(),
      /*elidedAttrs=*/{mlir::SymbolTable::getSymbolAttrName()});
  p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

ArrayAttr InterfaceOp::getExecutableSetLayoutsAttr() {
  Builder builder(getContext());
  SmallVector<SmallVector<Attribute, 4>, 4> setAttrs;
  for (auto bindingOp : getBlock().getOps<InterfaceBindingOp>()) {
    int set = bindingOp.set().getZExtValue();
    int binding = bindingOp.binding().getZExtValue();
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
      failed(parser.parseAttribute(setAttr,
                                   parser.getBuilder().getIntegerType(32),
                                   "set", result->attributes)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("binding")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(bindingAttr,
                                   parser.getBuilder().getIntegerType(32),
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
  p.printOptionalAttrDictWithKeyword(op.getAttrs(),
                                     /*elidedAttrs=*/{
                                         mlir::SymbolTable::getSymbolAttrName(),
                                         "set",
                                         "binding",
                                         "type",
                                         "access",
                                     });
}

//===----------------------------------------------------------------------===//
// hal.interface.load.tensor
//===----------------------------------------------------------------------===//

InterfaceBindingOp InterfaceLoadTensorOp::queryBindingOp() {
  return dyn_cast_or_null<InterfaceBindingOp>(
      SymbolTable::lookupNearestSymbolFrom(getOperation(), binding()));
}

//===----------------------------------------------------------------------===//
// hal.interface.store.tensor
//===----------------------------------------------------------------------===//

InterfaceBindingOp InterfaceStoreTensorOp::queryBindingOp() {
  return dyn_cast_or_null<InterfaceBindingOp>(
      SymbolTable::lookupNearestSymbolFrom(getOperation(), binding()));
}

//===----------------------------------------------------------------------===//
// hal.executable_cache.create
//===----------------------------------------------------------------------===//

void ExecutableCacheCreateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), (StringRef("executable_cache_") + identifier()).str());
}

//===----------------------------------------------------------------------===//
// hal.executable_cache.select_format
//===----------------------------------------------------------------------===//

void ExecutableCacheSelectFormatOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "preferred_format");
}

//===----------------------------------------------------------------------===//
// hal.executable_cache.prepare
//===----------------------------------------------------------------------===//

void ExecutableCachePrepareOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), (StringRef("executable_") + executable()).str());
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

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/HAL/IR/HALOps.cpp.inc"

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
