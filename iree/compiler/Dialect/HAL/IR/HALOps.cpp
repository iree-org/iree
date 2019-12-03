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
#include "iree/compiler/Dialect/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

static Type getDimType(OpAsmParser &parser) {
  return parser.getBuilder().getIntegerType(32);
}

static Type getDeviceSizeType(OpAsmParser &parser) {
  return parser.getBuilder().getIntegerType(32);
}

template <typename T>
using EnumSymbolizerFn = llvm::Optional<T> (*)(llvm::StringRef);
template <typename T, EnumSymbolizerFn<T> F>
static LogicalResult parseEnumAttr(OpAsmParser &parser, StringRef attrName,
                                   SmallVectorImpl<NamedAttribute> &attrs) {
  Attribute genericAttr;
  SmallVector<NamedAttribute, 1> attrList;
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
  // TODO(b/145167884): remove F and use symbolizeEnum<T> instead.
  auto symbolized = F(stringAttr.getValue());
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
    function_ref<void(Value *, StringRef)> setNameFn) {
  setNameFn(result(), "dev");
}

static ParseResult parseExSharedDeviceOp(OpAsmParser &parser,
                                         OperationState *result) {
  Type deviceType;
  if (failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseColonType(deviceType))) {
    return failure();
  }
  result->addTypes(deviceType);
  return success();
}

static void printExSharedDeviceOp(OpAsmPrinter &p, ExSharedDeviceOp op) {
  p << op.getOperationName();
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
  p << " : ";
  p.printType(op.result()->getType());
}

//===----------------------------------------------------------------------===//
// hal.ex.cache_executable
//===----------------------------------------------------------------------===//

void ExCacheExecutableOp::getAsmResultNames(
    function_ref<void(Value *, StringRef)> setNameFn) {
  setNameFn(result(), "exe");
}

static ParseResult parseExCacheExecutableOp(OpAsmParser &parser,
                                            OperationState *result) {
  OpAsmParser::OperandType device;
  FlatSymbolRefAttr executableAttr;
  Type executableType;
  if (failed(parser.parseOperand(device)) || failed(parser.parseComma()) ||
      failed(parser.parseAttribute(executableAttr, "executable",
                                   result->attributes)) ||
      failed(parser.resolveOperand(
          device, RefPtrType::get(DeviceType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseColonType(executableType))) {
    return failure();
  }
  result->addTypes(executableType);
  return success();
}

static void printExCacheExecutableOp(OpAsmPrinter &p, ExCacheExecutableOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.device());
  p << ", ";
  p.printSymbolName(op.executable());
  p.printOptionalAttrDictWithKeyword(op.getAttrs(),
                                     /*elidedAttrs=*/{"executable"});
  p << " : ";
  p.printType(op.result()->getType());
}

//===----------------------------------------------------------------------===//
// hal.ex.executable_descriptor_set_layout
//===----------------------------------------------------------------------===//

void ExExecutableDescriptorSetLayoutOp::getAsmResultNames(
    function_ref<void(Value *, StringRef)> setNameFn) {
  setNameFn(result(), "dsl");
}

static ParseResult parseExExecutableDescriptorSetLayoutOp(
    OpAsmParser &parser, OperationState *result) {
  OpAsmParser::OperandType executable;
  IntegerAttr setAttr;
  Type descriptorSetLayoutType;
  if (failed(parser.parseOperand(executable)) || failed(parser.parseComma()) ||
      failed(parser.resolveOperand(
          executable,
          RefPtrType::get(ExecutableType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseAttribute(setAttr,
                                   parser.getBuilder().getIntegerType(32),
                                   "set", result->attributes)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseColonType(descriptorSetLayoutType))) {
    return failure();
  }
  result->addTypes(descriptorSetLayoutType);
  return success();
}

static void printExExecutableDescriptorSetLayoutOp(
    OpAsmPrinter &p, ExExecutableDescriptorSetLayoutOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.executable());
  p << ", " << op.set();
  p.printOptionalAttrDictWithKeyword(op.getAttrs(),
                                     /*elidedAttrs=*/{"executable"});
  p << " : ";
  p.printType(op.result()->getType());
}

//===----------------------------------------------------------------------===//
// hal.ex.defer_release
//===----------------------------------------------------------------------===//

static ParseResult parseExDeferReleaseOp(OpAsmParser &parser,
                                         OperationState *result) {
  OpAsmParser::OperandType operand;
  Type operandType;
  if (failed(parser.parseOperand(operand)) ||
      failed(parser.parseColonType(operandType)) ||
      failed(parser.resolveOperand(operand, operandType, result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

static void printExDeferReleaseOp(OpAsmPrinter &p, ExDeferReleaseOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.operand());
  p << " : ";
  p.printType(op.operand()->getType());
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// hal.ex.submit_and_wait
//===----------------------------------------------------------------------===//

static ParseResult parseExSubmitAndWaitOp(OpAsmParser &parser,
                                          OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> operands;
  auto operandsLoc = parser.getCurrentLocation();
  if (failed(parser.parseOperandList(operands)) ||
      failed(parser.resolveOperands(
          operands,
          ArrayRef<Type>{
              RefPtrType::get(DeviceType::get(result->getContext())),
              RefPtrType::get(CommandBufferType::get(result->getContext()))},
          operandsLoc, result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

static void printExSubmitAndWaitOp(OpAsmPrinter &p, ExSubmitAndWaitOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.device());
  p << ", ";
  p.printOperand(op.command_buffer());
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// hal.make_memory_barrier
//===----------------------------------------------------------------------===//

void MakeMemoryBarrierOp::build(Builder *builder, OperationState &state,
                                IREE::HAL::AccessScopeBitfield sourceScope,
                                IREE::HAL::AccessScopeBitfield targetScope) {
  state.addAttribute("source_scope", builder->getI32IntegerAttr(
                                         static_cast<int32_t>(sourceScope)));
  state.addAttribute("target_scope", builder->getI32IntegerAttr(
                                         static_cast<int32_t>(targetScope)));
  state.addTypes({MemoryBarrierType::get(builder->getContext())});
}

void MakeMemoryBarrierOp::getAsmResultNames(
    function_ref<void(Value *, StringRef)> setNameFn) {
  setNameFn(result(), "memory_barrier");
}

static ParseResult parseMakeMemoryBarrierOp(OpAsmParser &parser,
                                            OperationState *result) {
  Type memoryBarrierType;
  if (failed(parseEnumAttr<AccessScopeBitfield, symbolizeAccessScopeBitfield>(
          parser, "source_scope", result->attributes)) ||
      failed(parser.parseComma()) ||
      failed(parseEnumAttr<AccessScopeBitfield, symbolizeAccessScopeBitfield>(
          parser, "target_scope", result->attributes)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseColonType(memoryBarrierType))) {
    return failure();
  }
  result->addTypes(memoryBarrierType);
  return success();
}

static void printMakeMemoryBarrierOp(OpAsmPrinter &p, MakeMemoryBarrierOp op) {
  p << op.getOperationName() << ' ';
  p << "\"" << stringifyAccessScopeBitfield(op.source_scope()) << "\"";
  p << ", ";
  p << "\"" << stringifyAccessScopeBitfield(op.target_scope()) << "\"";
  p.printOptionalAttrDictWithKeyword(
      op.getAttrs(),
      /*elidedAttrs=*/{"source_scope", "target_scope"});
  p << " : ";
  p.printType(op.result()->getType());
}

//===----------------------------------------------------------------------===//
// hal.make_buffer_barrier
//===----------------------------------------------------------------------===//

void MakeBufferBarrierOp::build(Builder *builder, OperationState &state,
                                IREE::HAL::AccessScopeBitfield sourceScope,
                                IREE::HAL::AccessScopeBitfield targetScope,
                                Value *buffer, Value *offset, Value *length) {
  state.addAttribute("source_scope", builder->getI32IntegerAttr(
                                         static_cast<int32_t>(sourceScope)));
  state.addAttribute("target_scope", builder->getI32IntegerAttr(
                                         static_cast<int32_t>(targetScope)));
  state.addOperands({buffer, offset, length});
  state.addTypes({BufferBarrierType::get(builder->getContext())});
}

void MakeBufferBarrierOp::getAsmResultNames(
    function_ref<void(Value *, StringRef)> setNameFn) {
  setNameFn(result(), "buffer_barrier");
}

static ParseResult parseMakeBufferBarrierOp(OpAsmParser &parser,
                                            OperationState *result) {
  llvm::SMLoc operandLoc;
  SmallVector<OpAsmParser::OperandType, 3> operands;
  Type bufferBarrierType;
  if (failed(parseEnumAttr<AccessScopeBitfield, symbolizeAccessScopeBitfield>(
          parser, "source_scope", result->attributes)) ||
      failed(parser.parseComma()) ||
      failed(parseEnumAttr<AccessScopeBitfield, symbolizeAccessScopeBitfield>(
          parser, "target_scope", result->attributes)) ||
      failed(parser.parseComma()) ||
      failed(parser.parseOperandList(operands, 3)) ||
      failed(parser.getCurrentLocation(&operandLoc)) ||
      failed(parser.resolveOperands(
          operands,
          ArrayRef<Type>{
              RefPtrType::get(BufferType::get(result->getContext())),
              getDeviceSizeType(parser),
              getDeviceSizeType(parser),
          },
          operandLoc, result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseColonType(bufferBarrierType))) {
    return failure();
  }
  result->addTypes(bufferBarrierType);
  return success();
}

static void printMakeBufferBarrierOp(OpAsmPrinter &p, MakeBufferBarrierOp op) {
  p << op.getOperationName() << ' ';
  p << "\"" << stringifyAccessScopeBitfield(op.source_scope()) << "\"";
  p << ", ";
  p << "\"" << stringifyAccessScopeBitfield(op.target_scope()) << "\"";
  p << ", ";
  p.printOperands(op.getOperation()->getOperands());
  p.printOptionalAttrDictWithKeyword(
      op.getAttrs(),
      /*elidedAttrs=*/{"source_scope", "target_scope"});
  p << " : ";
  p.printType(op.result()->getType());
}

//===----------------------------------------------------------------------===//
// hal.allocator.compute_size
//===----------------------------------------------------------------------===//

void AllocatorComputeSizeOp::build(Builder *builder, OperationState &state,
                                   Value *allocator,
                                   IREE::HAL::MemoryTypeBitfield memoryTypes,
                                   IREE::HAL::BufferUsageBitfield bufferUsage,
                                   ArrayRef<Value *> shape,
                                   int32_t elementSize) {
  state.addOperands({allocator});
  state.addOperands(shape);
  state.addAttribute("memory_types", builder->getI32IntegerAttr(
                                         static_cast<int32_t>(memoryTypes)));
  state.addAttribute("buffer_usage", builder->getI32IntegerAttr(
                                         static_cast<int32_t>(bufferUsage)));
  state.addAttribute("element_size", builder->getI32IntegerAttr(elementSize));
  state.addTypes({builder->getIntegerType(32)});
}

void AllocatorComputeSizeOp::getAsmResultNames(
    function_ref<void(Value *, StringRef)> setNameFn) {
  setNameFn(result(), "sz");
}

static ParseResult parseAllocatorComputeSizeOp(OpAsmParser &parser,
                                               OperationState *result) {
  OpAsmParser::OperandType allocator;
  SmallVector<OpAsmParser::OperandType, 4> shape;
  IntegerAttr elementSize;
  if (failed(parser.parseOperand(allocator)) ||
      failed(parser.resolveOperand(
          allocator, RefPtrType::get(AllocatorType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseComma()) ||
      failed(parseEnumAttr<MemoryTypeBitfield, symbolizeMemoryTypeBitfield>(
          parser, "memory_types", result->attributes)) ||
      failed(parser.parseComma()) ||
      failed(parseEnumAttr<BufferUsageBitfield, symbolizeBufferUsageBitfield>(
          parser, "buffer_usage", result->attributes)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("shape")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseOperandList(shape, OpAsmParser::Delimiter::Square)) ||
      failed(parser.resolveOperands(shape, getDimType(parser),
                                    result->operands)) ||
      failed(parser.parseComma()) ||
      failed(parser.parseKeyword("element_size")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(elementSize, getDeviceSizeType(parser),
                                   "element_size", result->attributes)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  result->addTypes(getDeviceSizeType(parser));
  return success();
}

static void printAllocatorComputeSizeOp(OpAsmPrinter &p,
                                        AllocatorComputeSizeOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.allocator());
  p << ", ";
  p << "\"" << stringifyMemoryTypeBitfield(op.memory_types()) << "\"";
  p << ", ";
  p << "\"" << stringifyBufferUsageBitfield(op.buffer_usage()) << "\"";
  p << ", shape=[";
  p.printOperands(op.shape());
  p << "], element_size=" << op.element_size();
  p.printOptionalAttrDictWithKeyword(
      op.getAttrs(),
      /*elidedAttrs=*/{"memory_types", "buffer_usage", "element_size"});
}

//===----------------------------------------------------------------------===//
// hal.allocator.allocate
//===----------------------------------------------------------------------===//

void AllocatorAllocateOp::build(Builder *builder, OperationState &state,
                                Value *allocator,
                                IREE::HAL::MemoryTypeBitfield memoryTypes,
                                IREE::HAL::BufferUsageBitfield bufferUsage,
                                Value *allocationSize) {
  state.addOperands({allocator, allocationSize});
  state.addAttribute("memory_types", builder->getI32IntegerAttr(
                                         static_cast<int32_t>(memoryTypes)));
  state.addAttribute("buffer_usage", builder->getI32IntegerAttr(
                                         static_cast<int32_t>(bufferUsage)));
  state.addTypes({RefPtrType::get(BufferType::get(builder->getContext()))});
}

void AllocatorAllocateOp::getAsmResultNames(
    function_ref<void(Value *, StringRef)> setNameFn) {
  setNameFn(result(), "buffer");
}

static ParseResult parseAllocatorAllocateOp(OpAsmParser &parser,
                                            OperationState *result) {
  OpAsmParser::OperandType allocator;
  OpAsmParser::OperandType allocationSize;
  Type resultType;
  if (failed(parser.parseOperand(allocator)) ||
      failed(parser.resolveOperand(
          allocator, RefPtrType::get(AllocatorType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseComma()) ||
      failed(parseEnumAttr<MemoryTypeBitfield, symbolizeMemoryTypeBitfield>(
          parser, "memory_types", result->attributes)) ||
      failed(parser.parseComma()) ||
      failed(parseEnumAttr<BufferUsageBitfield, symbolizeBufferUsageBitfield>(
          parser, "buffer_usage", result->attributes)) ||
      failed(parser.parseComma()) ||
      failed(parser.parseOperand(allocationSize)) ||
      failed(parser.resolveOperand(allocationSize, getDeviceSizeType(parser),
                                   result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseColonType(resultType))) {
    return failure();
  }
  result->addTypes(resultType);
  return success();
}

static void printAllocatorAllocateOp(OpAsmPrinter &p, AllocatorAllocateOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.allocator());
  p << ", ";
  p << "\"" << stringifyMemoryTypeBitfield(op.memory_types()) << "\"";
  p << ", ";
  p << "\"" << stringifyBufferUsageBitfield(op.buffer_usage()) << "\"";
  p << ", ";
  p.printOperand(op.allocation_size());
  p.printOptionalAttrDictWithKeyword(
      op.getAttrs(),
      /*elidedAttrs=*/{"memory_types", "buffer_usage"});
  p << " : ";
  p.printType(op.result()->getType());
}

//===----------------------------------------------------------------------===//
// hal.allocator.allocate_shaped
//===----------------------------------------------------------------------===//

void AllocatorAllocateShapedOp::build(
    Builder *builder, OperationState &state, Value *allocator,
    IREE::HAL::MemoryTypeBitfield memoryTypes,
    IREE::HAL::BufferUsageBitfield bufferUsage, ArrayRef<Value *> shape,
    int32_t elementSize) {
  state.addOperands({allocator});
  state.addOperands(shape);
  state.addAttribute("memory_types", builder->getI32IntegerAttr(
                                         static_cast<int32_t>(memoryTypes)));
  state.addAttribute("buffer_usage", builder->getI32IntegerAttr(
                                         static_cast<int32_t>(bufferUsage)));
  state.addAttribute("element_size", builder->getI32IntegerAttr(elementSize));
  state.addTypes({RefPtrType::get(BufferType::get(builder->getContext()))});
}

void AllocatorAllocateShapedOp::getAsmResultNames(
    function_ref<void(Value *, StringRef)> setNameFn) {
  setNameFn(result(), "buffer");
}

static ParseResult parseAllocatorAllocateShapedOp(OpAsmParser &parser,
                                                  OperationState *result) {
  OpAsmParser::OperandType allocator;
  SmallVector<OpAsmParser::OperandType, 4> shape;
  IntegerAttr elementSize;
  Type resultType;
  if (failed(parser.parseOperand(allocator)) ||
      failed(parser.resolveOperand(
          allocator, RefPtrType::get(AllocatorType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseComma()) ||
      failed(parseEnumAttr<MemoryTypeBitfield, symbolizeMemoryTypeBitfield>(
          parser, "memory_types", result->attributes)) ||
      failed(parser.parseComma()) ||
      failed(parseEnumAttr<BufferUsageBitfield, symbolizeBufferUsageBitfield>(
          parser, "buffer_usage", result->attributes)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("shape")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseOperandList(shape, OpAsmParser::Delimiter::Square)) ||
      failed(parser.resolveOperands(shape, getDimType(parser),
                                    result->operands)) ||
      failed(parser.parseComma()) ||
      failed(parser.parseKeyword("element_size")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(elementSize, getDeviceSizeType(parser),
                                   "element_size", result->attributes)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseColonType(resultType))) {
    return failure();
  }
  result->addTypes(resultType);
  return success();
}

static void printAllocatorAllocateShapedOp(OpAsmPrinter &p,
                                           AllocatorAllocateShapedOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.allocator());
  p << ", ";
  p << "\"" << stringifyMemoryTypeBitfield(op.memory_types()) << "\"";
  p << ", ";
  p << "\"" << stringifyBufferUsageBitfield(op.buffer_usage()) << "\"";
  p << ", shape=[";
  p.printOperands(op.shape());
  p << "], element_size=" << op.element_size();
  p.printOptionalAttrDictWithKeyword(
      op.getAttrs(),
      /*elidedAttrs=*/{"memory_types", "buffer_usage", "element_size"});
  p << " : ";
  p.printType(op.result()->getType());
}

//===----------------------------------------------------------------------===//
// hal.buffer.subspan
//===----------------------------------------------------------------------===//

void BufferSubspanOp::getAsmResultNames(
    function_ref<void(Value *, StringRef)> setNameFn) {
  setNameFn(result(), "buffer");
}

static ParseResult parseBufferSubspanOp(OpAsmParser &parser,
                                        OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 3> operands;
  Type resultType;
  auto loc = parser.getCurrentLocation();
  if (failed(parser.parseOperandList(operands)) ||
      failed(parser.resolveOperands(
          operands,
          ArrayRef<Type>{
              RefPtrType::get(BufferType::get(result->getContext())),
              getDeviceSizeType(parser),
              getDeviceSizeType(parser),
          },
          loc, result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseColonType(resultType))) {
    return failure();
  }
  result->addTypes(resultType);
  return success();
}

static void printBufferSubspanOp(OpAsmPrinter &p, BufferSubspanOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.source_buffer());
  p << ", ";
  p.printOperand(op.source_offset());
  p << ", ";
  p.printOperand(op.length());
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
  p << " : ";
  p.printType(op.result()->getType());
}

//===----------------------------------------------------------------------===//
// hal.buffer.fill
//===----------------------------------------------------------------------===//

static ParseResult parseBufferFillOp(OpAsmParser &parser,
                                     OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 4> operands;
  auto loc = parser.getCurrentLocation();
  if (failed(parser.parseOperandList(operands)) ||
      failed(parser.resolveOperands(
          operands,
          ArrayRef<Type>{
              RefPtrType::get(BufferType::get(result->getContext())),
              getDeviceSizeType(parser),
              getDeviceSizeType(parser),
              parser.getBuilder().getIntegerType(32),
          },
          loc, result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

static void printBufferFillOp(OpAsmPrinter &p, BufferFillOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.target_buffer());
  p << ", ";
  p.printOperand(op.target_offset());
  p << ", ";
  p.printOperand(op.length());
  p << ", ";
  p.printOperand(op.pattern());
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// hal.buffer.read_data
//===----------------------------------------------------------------------===//

static ParseResult parseBufferReadDataOp(OpAsmParser &parser,
                                         OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 5> operands;
  Type targetBufferType;
  auto loc = parser.getCurrentLocation();
  if (failed(parser.parseOperandList(operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseColonType(targetBufferType)) ||
      failed(parser.resolveOperands(
          operands,
          ArrayRef<Type>{
              RefPtrType::get(BufferType::get(result->getContext())),
              getDeviceSizeType(parser),
              targetBufferType,
              getDeviceSizeType(parser),
              getDeviceSizeType(parser),
          },
          loc, result->operands))) {
    return failure();
  }
  return success();
}

static void printBufferReadDataOp(OpAsmPrinter &p, BufferReadDataOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.source_buffer());
  p << ", ";
  p.printOperand(op.source_offset());
  p << ", ";
  p.printOperand(op.target_buffer());
  p << ", ";
  p.printOperand(op.target_offset());
  p << ", ";
  p.printOperand(op.length());
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
  p << " : ";
  p.printType(op.target_buffer()->getType());
}

//===----------------------------------------------------------------------===//
// hal.buffer.write_data
//===----------------------------------------------------------------------===//

static ParseResult parseBufferWriteDataOp(OpAsmParser &parser,
                                          OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 5> operands;
  Type sourceBufferType;
  auto loc = parser.getCurrentLocation();
  if (failed(parser.parseOperandList(operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseColonType(sourceBufferType)) ||
      failed(parser.resolveOperands(
          operands,
          ArrayRef<Type>{
              sourceBufferType,
              getDeviceSizeType(parser),
              RefPtrType::get(BufferType::get(result->getContext())),
              getDeviceSizeType(parser),
              getDeviceSizeType(parser),
          },
          loc, result->operands))) {
    return failure();
  }
  return success();
}

static void printBufferWriteDataOp(OpAsmPrinter &p, BufferWriteDataOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.source_buffer());
  p << ", ";
  p.printOperand(op.source_offset());
  p << ", ";
  p.printOperand(op.target_buffer());
  p << ", ";
  p.printOperand(op.target_offset());
  p << ", ";
  p.printOperand(op.length());
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
  p << " : ";
  p.printType(op.source_buffer()->getType());
}

//===----------------------------------------------------------------------===//
// hal.buffer.copy_data
//===----------------------------------------------------------------------===//

static ParseResult parseBufferCopyDataOp(OpAsmParser &parser,
                                         OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 5> operands;
  auto loc = parser.getCurrentLocation();
  if (failed(parser.parseOperandList(operands)) ||
      failed(parser.resolveOperands(
          operands,
          ArrayRef<Type>{
              RefPtrType::get(BufferType::get(result->getContext())),
              getDeviceSizeType(parser),
              RefPtrType::get(BufferType::get(result->getContext())),
              getDeviceSizeType(parser),
              getDeviceSizeType(parser),
          },
          loc, result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

static void printBufferCopyDataOp(OpAsmPrinter &p, BufferCopyDataOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.source_buffer());
  p << ", ";
  p.printOperand(op.source_offset());
  p << ", ";
  p.printOperand(op.target_buffer());
  p << ", ";
  p.printOperand(op.target_offset());
  p << ", ";
  p.printOperand(op.length());
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// hal.buffer_view.compute_offset
//===----------------------------------------------------------------------===//

void BufferViewComputeOffsetOp::getAsmResultNames(
    function_ref<void(Value *, StringRef)> setNameFn) {
  setNameFn(offset(), "off");
}

static ParseResult parseBufferViewComputeOffsetOp(OpAsmParser &parser,
                                                  OperationState *result) {
  OpAsmParser::OperandType buffer;
  SmallVector<OpAsmParser::OperandType, 4> shape;
  SmallVector<OpAsmParser::OperandType, 4> indices;
  if (failed(parser.parseOperand(buffer)) ||
      failed(parser.resolveOperand(
          buffer, RefPtrType::get(BufferType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("shape")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseOperandList(shape, OpAsmParser::Delimiter::Square)) ||
      failed(parser.resolveOperands(shape, getDimType(parser),
                                    result->operands)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("indices")) ||
      failed(parser.parseEqual()) ||
      failed(
          parser.parseOperandList(indices, OpAsmParser::Delimiter::Square)) ||
      failed(parser.resolveOperands(indices, getDimType(parser),
                                    result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  result->addTypes(getDeviceSizeType(parser));
  return success();
}

static void printBufferViewComputeOffsetOp(OpAsmPrinter &p,
                                           BufferViewComputeOffsetOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.buffer());
  p << ", shape=[";
  p.printOperands(op.shape());
  p << "], indices=[";
  p.printOperands(op.indices());
  p << "]";
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// hal.buffer_view.compute_range
//===----------------------------------------------------------------------===//

void BufferViewComputeRangeOp::getAsmResultNames(
    function_ref<void(Value *, StringRef)> setNameFn) {
  setNameFn(offset(), "off");
  setNameFn(length(), "len");
}

static ParseResult parseBufferViewComputeRangeOp(OpAsmParser &parser,
                                                 OperationState *result) {
  OpAsmParser::OperandType buffer;
  SmallVector<OpAsmParser::OperandType, 4> shape;
  SmallVector<OpAsmParser::OperandType, 4> indices;
  SmallVector<OpAsmParser::OperandType, 4> lengths;
  if (failed(parser.parseOperand(buffer)) ||
      failed(parser.resolveOperand(
          buffer, RefPtrType::get(BufferType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("shape")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseOperandList(shape, OpAsmParser::Delimiter::Square)) ||
      failed(parser.resolveOperands(shape, getDimType(parser),
                                    result->operands)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("indices")) ||
      failed(parser.parseEqual()) ||
      failed(
          parser.parseOperandList(indices, OpAsmParser::Delimiter::Square)) ||
      failed(parser.resolveOperands(indices, getDimType(parser),
                                    result->operands)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("lengths")) ||
      failed(parser.parseEqual()) ||
      failed(
          parser.parseOperandList(lengths, OpAsmParser::Delimiter::Square)) ||
      failed(parser.resolveOperands(lengths, getDimType(parser),
                                    result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  result->addTypes({getDeviceSizeType(parser), getDeviceSizeType(parser)});
  return success();
}

static void printBufferViewComputeRangeOp(OpAsmPrinter &p,
                                          BufferViewComputeRangeOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.buffer());
  p << ", shape=[";
  p.printOperands(op.shape());
  p << "], indices=[";
  p.printOperands(op.indices());
  p << "], lengths=[";
  p.printOperands(op.lengths());
  p << "]";
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// hal.buffer_view.slice
//===----------------------------------------------------------------------===//

void BufferViewSliceOp::getAsmResultNames(
    function_ref<void(Value *, StringRef)> setNameFn) {
  setNameFn(result(), "slice");
}

static ParseResult parseBufferViewSliceOp(OpAsmParser &parser,
                                          OperationState *result) {
  OpAsmParser::OperandType buffer;
  SmallVector<OpAsmParser::OperandType, 4> shape;
  SmallVector<OpAsmParser::OperandType, 4> indices;
  SmallVector<OpAsmParser::OperandType, 4> lengths;
  if (failed(parser.parseOperand(buffer)) ||
      failed(parser.resolveOperand(
          buffer, RefPtrType::get(BufferType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("shape")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseOperandList(shape, OpAsmParser::Delimiter::Square)) ||
      failed(parser.resolveOperands(shape, getDimType(parser),
                                    result->operands)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("indices")) ||
      failed(parser.parseEqual()) ||
      failed(
          parser.parseOperandList(indices, OpAsmParser::Delimiter::Square)) ||
      failed(parser.resolveOperands(indices, getDimType(parser),
                                    result->operands)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("lengths")) ||
      failed(parser.parseEqual()) ||
      failed(
          parser.parseOperandList(lengths, OpAsmParser::Delimiter::Square)) ||
      failed(parser.resolveOperands(lengths, getDimType(parser),
                                    result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  result->addTypes(RefPtrType::get(BufferType::get(result->getContext())));
  return success();
}

static void printBufferViewSliceOp(OpAsmPrinter &p, BufferViewSliceOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.buffer());
  p << ", shape=[";
  p.printOperands(op.shape());
  p << "], indices=[";
  p.printOperands(op.indices());
  p << "], lengths=[";
  p.printOperands(op.lengths());
  p << "]";
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// hal.command_buffer.create
//===----------------------------------------------------------------------===//

void CommandBufferCreateOp::build(
    Builder *builder, OperationState &state, Value *device,
    IREE::HAL::CommandBufferModeBitfield modes,
    IREE::HAL::CommandCategoryBitfield commandCategories) {
  state.addOperands({device});
  state.addAttribute("modes",
                     builder->getI32IntegerAttr(static_cast<int32_t>(modes)));
  state.addAttribute(
      "command_categories",
      builder->getI32IntegerAttr(static_cast<int32_t>(commandCategories)));
  state.addTypes(
      {RefPtrType::get(CommandBufferType::get(builder->getContext()))});
}

void CommandBufferCreateOp::getAsmResultNames(
    function_ref<void(Value *, StringRef)> setNameFn) {
  setNameFn(result(), "cmd");
}

static ParseResult parseCommandBufferCreateOp(OpAsmParser &parser,
                                              OperationState *result) {
  OpAsmParser::OperandType device;
  Type resultType;
  if (failed(parser.parseOperand(device)) ||
      failed(parser.resolveOperand(
          device, RefPtrType::get(DeviceType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseComma()) ||
      failed(parseEnumAttr<CommandBufferModeBitfield,
                           symbolizeCommandBufferModeBitfield>(
          parser, "modes", result->attributes)) ||
      failed(parser.parseComma()) ||
      failed(parseEnumAttr<CommandCategoryBitfield,
                           symbolizeCommandCategoryBitfield>(
          parser, "command_categories", result->attributes)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseColonType(resultType))) {
    return failure();
  }
  result->addTypes(resultType);
  return success();
}

static void printCommandBufferCreateOp(OpAsmPrinter &p,
                                       CommandBufferCreateOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.device());
  p << ", \"";
  p << stringifyCommandBufferModeBitfield(op.modes());
  p << "\", \"";
  p << stringifyCommandCategoryBitfield(op.command_categories());
  p << "\"";
  p.printOptionalAttrDictWithKeyword(
      op.getAttrs(),
      /*elidedAttrs=*/{"modes", "command_categories"});
  p << " : ";
  p.printType(op.result()->getType());
}

//===----------------------------------------------------------------------===//
// hal.command_buffer.begin
//===----------------------------------------------------------------------===//

static ParseResult parseCommandBufferBeginOp(OpAsmParser &parser,
                                             OperationState *result) {
  OpAsmParser::OperandType commandBuffer;
  if (failed(parser.parseOperand(commandBuffer)) ||
      failed(parser.resolveOperand(
          commandBuffer,
          RefPtrType::get(CommandBufferType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

static void printCommandBufferBeginOp(OpAsmPrinter &p,
                                      CommandBufferBeginOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.command_buffer());
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// hal.command_buffer.end
//===----------------------------------------------------------------------===//

static ParseResult parseCommandBufferEndOp(OpAsmParser &parser,
                                           OperationState *result) {
  OpAsmParser::OperandType commandBuffer;
  if (failed(parser.parseOperand(commandBuffer)) ||
      failed(parser.resolveOperand(
          commandBuffer,
          RefPtrType::get(CommandBufferType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

static void printCommandBufferEndOp(OpAsmPrinter &p, CommandBufferEndOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.command_buffer());
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// hal.command_buffer.execution_barrier
//===----------------------------------------------------------------------===//

void CommandBufferExecutionBarrierOp::build(
    Builder *builder, OperationState &state, Value *commandBuffer,
    IREE::HAL::ExecutionStageBitfield sourceStageMask,
    IREE::HAL::ExecutionStageBitfield targetStageMask,
    ArrayRef<Value *> memoryBarriers, ArrayRef<Value *> bufferBarriers) {
  state.addAttribute(
      "source_stage_mask",
      builder->getI32IntegerAttr(static_cast<int32_t>(sourceStageMask)));
  state.addAttribute(
      "target_stage_mask",
      builder->getI32IntegerAttr(static_cast<int32_t>(targetStageMask)));
  state.addOperands(commandBuffer);
  state.addOperands(memoryBarriers);
  state.addOperands(bufferBarriers);
  state.addAttribute("operand_segment_sizes",
                     DenseIntElementsAttr::get(
                         VectorType::get({3}, builder->getIntegerType(32)),
                         {1, static_cast<int>(memoryBarriers.size()),
                          static_cast<int>(bufferBarriers.size())}));
}

static ParseResult parseCommandBufferExecutionBarrierOp(
    OpAsmParser &parser, OperationState *result) {
  OpAsmParser::OperandType commandBuffer;
  if (failed(parser.parseOperand(commandBuffer)) ||
      failed(parser.resolveOperand(
          commandBuffer,
          RefPtrType::get(CommandBufferType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseComma()) ||
      failed(parseEnumAttr<ExecutionStageBitfield,
                           symbolizeExecutionStageBitfield>(
          parser, "source_stage_mask", result->attributes)) ||
      failed(parser.parseComma()) ||
      failed(parseEnumAttr<ExecutionStageBitfield,
                           symbolizeExecutionStageBitfield>(
          parser, "target_stage_mask", result->attributes))) {
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
// hal.command_buffer.fill_buffer
//===----------------------------------------------------------------------===//

static ParseResult parseCommandBufferFillBufferOp(OpAsmParser &parser,
                                                  OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 5> operands;
  auto operandsLoc = parser.getCurrentLocation();
  if (failed(parser.parseOperandList(operands)) ||
      failed(parser.resolveOperands(
          operands,
          ArrayRef<Type>{
              RefPtrType::get(CommandBufferType::get(result->getContext())),
              RefPtrType::get(BufferType::get(result->getContext())),
              getDeviceSizeType(parser),
              getDeviceSizeType(parser),
              parser.getBuilder().getIntegerType(32),
          },
          operandsLoc, result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

static void printCommandBufferFillBufferOp(OpAsmPrinter &p,
                                           CommandBufferFillBufferOp op) {
  p << op.getOperationName() << ' ';
  p.printOperands(op.getOperation()->getOperands());
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// hal.command_buffer.copy_buffer
//===----------------------------------------------------------------------===//

static ParseResult parseCommandBufferCopyBufferOp(OpAsmParser &parser,
                                                  OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 6> operands;
  auto operandsLoc = parser.getCurrentLocation();
  if (failed(parser.parseOperandList(operands)) ||
      failed(parser.resolveOperands(
          operands,
          ArrayRef<Type>{
              RefPtrType::get(CommandBufferType::get(result->getContext())),
              RefPtrType::get(BufferType::get(result->getContext())),
              getDeviceSizeType(parser),
              RefPtrType::get(BufferType::get(result->getContext())),
              getDeviceSizeType(parser),
              getDeviceSizeType(parser),
          },
          operandsLoc, result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

static void printCommandBufferCopyBufferOp(OpAsmPrinter &p,
                                           CommandBufferCopyBufferOp op) {
  p << op.getOperationName() << ' ';
  p.printOperands(op.getOperation()->getOperands());
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// hal.command_buffer.bind_descriptor_set
//===----------------------------------------------------------------------===//

void CommandBufferBindDescriptorSetOp::build(Builder *builder,
                                             OperationState &state,
                                             Value *commandBuffer,
                                             Value *executable, uint32_t set,
                                             Value *descriptorSet,
                                             ArrayRef<Value *> dynamicOffsets) {
  state.addOperands({commandBuffer, executable, descriptorSet});
  state.addAttribute("set",
                     builder->getIntegerAttr(builder->getIntegerType(32), set));
  state.addOperands(dynamicOffsets);
}

static ParseResult parseCommandBufferBindDescriptorSetOp(
    OpAsmParser &parser, OperationState *result) {
  OpAsmParser::OperandType commandBuffer;
  OpAsmParser::OperandType executable;
  IntegerAttr setAttr;
  OpAsmParser::OperandType descriptorSet;
  auto operandsLoc = parser.getCurrentLocation();
  if (failed(parser.parseOperand(commandBuffer)) ||
      failed(parser.parseComma()) || failed(parser.parseOperand(executable)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("set")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(setAttr,
                                   parser.getBuilder().getIntegerType(32),
                                   "set", result->attributes)) ||
      failed(parser.parseComma()) ||
      failed(parser.parseOperand(descriptorSet)) ||
      failed(parser.resolveOperands(
          ArrayRef<OpAsmParser::OperandType>{
              commandBuffer,
              executable,
              descriptorSet,
          },
          ArrayRef<Type>{
              RefPtrType::get(CommandBufferType::get(result->getContext())),
              RefPtrType::get(ExecutableType::get(result->getContext())),
              RefPtrType::get(DescriptorSetType::get(result->getContext())),
          },
          operandsLoc, result->operands))) {
    return failure();
  }
  SmallVector<OpAsmParser::OperandType, 4> dynamicOffsets;
  if (succeeded(parser.parseOptionalComma())) {
    if (failed(parser.parseKeyword("offsets")) || failed(parser.parseEqual()) ||
        failed(parser.parseOperandList(dynamicOffsets,
                                       OpAsmParser::Delimiter::Square)) ||
        failed(parser.resolveOperands(dynamicOffsets,
                                      parser.getBuilder().getIntegerType(32),
                                      result->operands))) {
      return failure();
    }
  }
  if (failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

static void printCommandBufferBindDescriptorSetOp(
    OpAsmPrinter &p, CommandBufferBindDescriptorSetOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.command_buffer());
  p << ", ";
  p.printOperand(op.executable());
  p << ", set=" << op.set() << ", ";
  p.printOperand(op.descriptor_set());
  if (!op.dynamic_offsets().empty()) {
    p << ", offsets=[";
    interleaveComma(op.dynamic_offsets(), p,
                    [&](Value *value) { p.printOperand(value); });
    p << "]";
  }
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{
                              "set",
                          });
}

//===----------------------------------------------------------------------===//
// hal.command_buffer.dispatch
//===----------------------------------------------------------------------===//

void CommandBufferDispatchOp::build(
    Builder *builder, OperationState &state, Value *commandBuffer,
    Value *executable, IREE::HAL::ExecutableEntryPointOp entryPoint,
    Value *workgroupX, Value *workgroupY, Value *workgroupZ) {
  state.addOperands(
      {commandBuffer, executable, workgroupX, workgroupY, workgroupZ});
  state.addAttribute("entry_point",
                     builder->getIntegerAttr(builder->getIntegerType(32),
                                             entryPoint.ordinal()));
}

static ParseResult parseCommandBufferDispatchOp(OpAsmParser &parser,
                                                OperationState *result) {
  OpAsmParser::OperandType commandBuffer;
  OpAsmParser::OperandType executable;
  IntegerAttr entryPointAttr;
  OpAsmParser::OperandType workgroupX, workgroupY, workgroupZ;
  SmallVector<OpAsmParser::OperandType, 4> bindings;
  auto operandsLoc = parser.getCurrentLocation();
  if (failed(parser.parseOperand(commandBuffer)) ||
      failed(parser.parseComma()) || failed(parser.parseOperand(executable)) ||
      failed(parser.parseComma()) ||
      failed(parser.parseKeyword("entry_point")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(entryPointAttr,
                                   parser.getBuilder().getIntegerType(32),
                                   "entry_point", result->attributes)) ||
      failed(parser.parseComma()) ||
      failed(parser.parseKeyword("workgroup_xyz")) ||
      failed(parser.parseEqual()) || failed(parser.parseLSquare()) ||
      failed(parser.parseOperand(workgroupX)) || failed(parser.parseComma()) ||
      failed(parser.parseOperand(workgroupY)) || failed(parser.parseComma()) ||
      failed(parser.parseOperand(workgroupZ)) ||
      failed(parser.parseRSquare()) ||
      failed(parser.resolveOperands(
          ArrayRef<OpAsmParser::OperandType>{
              commandBuffer,
              executable,
              workgroupX,
              workgroupY,
              workgroupZ,
          },
          ArrayRef<Type>{
              RefPtrType::get(CommandBufferType::get(result->getContext())),
              RefPtrType::get(ExecutableType::get(result->getContext())),
              parser.getBuilder().getIntegerType(32),
              parser.getBuilder().getIntegerType(32),
              parser.getBuilder().getIntegerType(32),
          },
          operandsLoc, result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

static void printCommandBufferDispatchOp(OpAsmPrinter &p,
                                         CommandBufferDispatchOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.command_buffer());
  p << ", ";
  p.printOperand(op.executable());
  p << ", entry_point=" << op.entry_point() << ", workgroup_xyz=[";
  interleaveComma(
      ArrayRef<Value *>{op.workgroup_x(), op.workgroup_y(), op.workgroup_z()},
      p, [&](Value *value) { p.printOperand(value); });
  p << "]";
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{
                              "entry_point",
                          });
}

//===----------------------------------------------------------------------===//
// hal.command_buffer.dispatch.indirect
//===----------------------------------------------------------------------===//

static ParseResult parseCommandBufferDispatchIndirectOp(
    OpAsmParser &parser, OperationState *result) {
  OpAsmParser::OperandType commandBuffer;
  OpAsmParser::OperandType executable;
  IntegerAttr entryPointAttr;
  OpAsmParser::OperandType workgroupsBuffer;
  OpAsmParser::OperandType workgroupsOffset;
  auto operandsLoc = parser.getCurrentLocation();
  if (failed(parser.parseOperand(commandBuffer)) ||
      failed(parser.parseComma()) || failed(parser.parseOperand(executable)) ||
      failed(parser.parseComma()) ||
      failed(parser.parseKeyword("entry_point")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(entryPointAttr,
                                   parser.getBuilder().getIntegerType(32),
                                   "entry_point", result->attributes)) ||
      failed(parser.parseComma()) ||
      failed(parser.parseKeyword("workgroups")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseOperand(workgroupsBuffer)) ||
      failed(parser.parseLSquare()) ||
      failed(parser.parseOperand(workgroupsOffset)) ||
      failed(parser.parseRSquare()) ||
      failed(parser.resolveOperands(
          ArrayRef<OpAsmParser::OperandType>{
              commandBuffer,
              executable,
              workgroupsBuffer,
              workgroupsOffset,
          },
          ArrayRef<Type>{
              RefPtrType::get(CommandBufferType::get(result->getContext())),
              RefPtrType::get(ExecutableType::get(result->getContext())),
              RefPtrType::get(BufferType::get(result->getContext())),
              getDeviceSizeType(parser),
          },
          operandsLoc, result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

static void printCommandBufferDispatchIndirectOp(
    OpAsmPrinter &p, CommandBufferDispatchIndirectOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.command_buffer());
  p << ", ";
  p.printOperand(op.executable());
  p << ", entry_point=" << op.entry_point() << ", workgroups=";
  p.printOperand(op.workgroups_buffer());
  p << '[';
  p.printOperand(op.workgroups_offset());
  p << ']';
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{
                              "entry_point",
                          });
}

//===----------------------------------------------------------------------===//
// hal.descriptor_set.allocate
//===----------------------------------------------------------------------===//

void DescriptorSetAllocateOp::getAsmResultNames(
    function_ref<void(Value *, StringRef)> setNameFn) {
  setNameFn(result(), "descriptor_set");
}

static ParseResult parseDescriptorSetAllocateOp(OpAsmParser &parser,
                                                OperationState *result) {
  OpAsmParser::OperandType device;
  OpAsmParser::OperandType setLayout;
  Type descriptorSetType;
  if (failed(parser.parseOperand(device)) ||
      failed(parser.resolveOperand(
          device, RefPtrType::get(DeviceType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseComma()) || failed(parser.parseOperand(setLayout)) ||
      failed(parser.resolveOperand(
          setLayout,
          RefPtrType::get(DescriptorSetLayoutType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseColonType(descriptorSetType))) {
    return failure();
  }
  result->addTypes(descriptorSetType);
  return success();
}

static void printDescriptorSetAllocateOp(OpAsmPrinter &p,
                                         DescriptorSetAllocateOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.device());
  p << ", ";
  p.printOperand(op.set_layout());
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
  p << " : ";
  p.printType(op.result()->getType());
}

//===----------------------------------------------------------------------===//
// hal.descriptor_set.make_binding
//===----------------------------------------------------------------------===//

void DescriptorSetMakeBindingOp::build(Builder *builder, OperationState &state,
                                       int32_t binding, Value *buffer,
                                       Value *offset, Value *length,
                                       IREE::HAL::MemoryAccessBitfield access) {
  state.addAttribute("binding", builder->getI32IntegerAttr(binding));
  state.addOperands({buffer, offset, length});
  state.addAttribute("access",
                     builder->getI32IntegerAttr(static_cast<int32_t>(access)));
  state.addTypes({DescriptorSetBindingType::get(builder->getContext())});
}

void DescriptorSetMakeBindingOp::getAsmResultNames(
    function_ref<void(Value *, StringRef)> setNameFn) {
  setNameFn(result(), "binding");
}

static ParseResult parseDescriptorSetMakeBindingOp(OpAsmParser &parser,
                                                   OperationState *result) {
  OpAsmParser::OperandType buffer;
  OpAsmParser::OperandType offset;
  OpAsmParser::OperandType length;
  llvm::SMLoc operandLoc;
  IntegerAttr bindingAttr;
  Type resultType;
  if (failed(parser.parseKeyword("binding")) || failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(bindingAttr,
                                   parser.getBuilder().getIntegerType(32),
                                   "binding", result->attributes)) ||
      failed(parser.parseComma()) || failed(parser.parseOperand(buffer)) ||
      failed(parser.parseComma()) || failed(parser.parseOperand(offset)) ||
      failed(parser.parseComma()) || failed(parser.parseOperand(length)) ||
      failed(parser.getCurrentLocation(&operandLoc)) ||
      failed(parser.resolveOperands(
          {
              buffer,
              offset,
              length,
          },
          ArrayRef<Type>{
              RefPtrType::get(BufferType::get(result->getContext())),
              getDeviceSizeType(parser),
              getDeviceSizeType(parser),
          },
          operandLoc, result->operands)) ||
      failed(parser.parseComma()) ||
      failed(parseEnumAttr<MemoryAccessBitfield, symbolizeMemoryAccessBitfield>(
          parser, "access", result->attributes)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseColonType(resultType))) {
    return failure();
  }
  result->addTypes(resultType);
  return success();
}

static void printDescriptorSetMakeBindingOp(OpAsmPrinter &p,
                                            DescriptorSetMakeBindingOp op) {
  p << op.getOperationName() << ' ';
  p << "binding=" << op.binding() << ", ";
  p.printOperands(op.getOperation()->getOperands());
  p << ", \"" << stringifyMemoryAccessBitfield(op.access()) << "\"";
  p.printOptionalAttrDictWithKeyword(op.getAttrs(),
                                     /*elidedAttrs=*/{"binding", "access"});
  p << " : ";
  p.printType(op.result()->getType());
}

//===----------------------------------------------------------------------===//
// hal.descriptor_set.update
//===----------------------------------------------------------------------===//

static ParseResult parseDescriptorSetUpdateOp(OpAsmParser &parser,
                                              OperationState *result) {
  OpAsmParser::OperandType device;
  OpAsmParser::OperandType set;
  SmallVector<OpAsmParser::OperandType, 4> bindings;
  if (failed(parser.parseOperand(device)) ||
      failed(parser.resolveOperand(
          device, RefPtrType::get(DeviceType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseComma()) || failed(parser.parseOperand(set)) ||
      failed(parser.resolveOperand(
          set, RefPtrType::get(DescriptorSetType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("bindings")) ||
      failed(parser.parseEqual()) ||
      failed(
          parser.parseOperandList(bindings, OpAsmParser::Delimiter::Square)) ||
      failed(parser.resolveOperands(
          bindings, DescriptorSetBindingType::get(result->getContext()),
          result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

static void printDescriptorSetUpdateOp(OpAsmPrinter &p,
                                       DescriptorSetUpdateOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.device());
  p << ", ";
  p.printOperand(op.set());
  p << ", bindings=[";
  p.printOperands(op.bindings());
  p << ']';
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// hal.device.allocator
//===----------------------------------------------------------------------===//

void DeviceAllocatorOp::getAsmResultNames(
    function_ref<void(Value *, StringRef)> setNameFn) {
  setNameFn(result(), "allocator");
}

static ParseResult parseDeviceAllocatorOp(OpAsmParser &parser,
                                          OperationState *result) {
  OpAsmParser::OperandType device;
  Type allocatorType;
  if (failed(parser.parseOperand(device)) ||
      failed(parser.resolveOperand(
          device, RefPtrType::get(DeviceType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseColonType(allocatorType))) {
    return failure();
  }
  result->addTypes(allocatorType);
  return success();
}

static void printDeviceAllocatorOp(OpAsmPrinter &p, DeviceAllocatorOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.device());
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
  p << " : ";
  p.printType(op.result()->getType());
}

//===----------------------------------------------------------------------===//
// hal.executable
//===----------------------------------------------------------------------===//

void ExecutableOp::build(Builder *builder, OperationState &state,
                         StringRef name) {
  ensureTerminator(*state.addRegion(), *builder, state.location);
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder->getStringAttr(name));
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

static ParseResult parseRegionEndOp(OpAsmParser &parser,
                                    OperationState *result) {
  return parser.parseOptionalAttrDict(result->attributes);
}

static void printRegionEndOp(OpAsmPrinter &p, Operation *op) {
  p << op->getName();
  p.printOptionalAttrDict(op->getAttrs());
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
// hal.executable.binary
//===----------------------------------------------------------------------===//

void ExecutableBinaryOp::build(Builder *builder, OperationState &state,
                               uint32_t format, std::vector<uint8_t> data) {
  ensureTerminator(*state.addRegion(), *builder, state.location);
  state.addAttribute(
      "format", builder->getIntegerAttr(builder->getIntegerType(32), format));
  state.addAttribute("data",
                     DenseIntElementsAttr::get(
                         VectorType::get({static_cast<int64_t>(data.size())},
                                         builder->getIntegerType(8)),
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
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/HAL/IR/HALOps.cpp.inc"

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
