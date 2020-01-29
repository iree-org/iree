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
    function_ref<void(Value, StringRef)> setNameFn) {
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
  p.printType(op.result().getType());
}

//===----------------------------------------------------------------------===//
// hal.ex.cache_executable
//===----------------------------------------------------------------------===//

void ExCacheExecutableOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
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
  p.printType(op.result().getType());
}

//===----------------------------------------------------------------------===//
// hal.ex.push_binding
//===----------------------------------------------------------------------===//

static ParseResult parseExPushBindingOp(OpAsmParser &parser,
                                        OperationState *result) {
  OpAsmParser::OperandType commandBuffer;
  OpAsmParser::OperandType buffer;
  SmallVector<OpAsmParser::OperandType, 4> shape;
  IntegerAttr ordinalAttr;
  IntegerAttr elementTypeAttr;
  if (failed(parser.parseOperand(commandBuffer)) ||
      failed(parser.parseComma()) ||
      failed(parser.resolveOperand(
          commandBuffer,
          RefPtrType::get(CommandBufferType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseAttribute(ordinalAttr,
                                   parser.getBuilder().getIntegerType(32),
                                   "ordinal", result->attributes)) ||
      failed(parser.parseComma()) || failed(parser.parseOperand(buffer)) ||
      failed(parser.parseComma()) ||
      failed(parser.resolveOperand(
          buffer, RefPtrType::get(BufferType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseKeyword("shape")) || failed(parser.parseEqual()) ||
      failed(parser.parseOperandList(shape, OpAsmParser::Delimiter::Square)) ||
      failed(parser.resolveOperands(shape, getDimType(parser),
                                    result->operands)) ||
      failed(parser.parseComma()) ||
      failed(parser.parseKeyword("element_type")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(elementTypeAttr,
                                   parser.getBuilder().getIntegerType(32),
                                   "element_type", result->attributes)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

static void printExPushBindingOp(OpAsmPrinter &p, ExPushBindingOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.command_buffer());
  p << ", " << op.ordinal() << ", ";
  p.printOperand(op.buffer());
  p << ", shape=[";
  interleaveComma(op.shape(), p, [&](Value value) { p.printOperand(value); });
  p << "], element_type=" << op.element_type();
  p.printOptionalAttrDictWithKeyword(
      op.getAttrs(),
      /*elidedAttrs=*/{"ordinal", "element_type"});
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
  p.printType(op.operand().getType());
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
    function_ref<void(Value, StringRef)> setNameFn) {
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
  p.printType(op.result().getType());
}

//===----------------------------------------------------------------------===//
// hal.make_buffer_barrier
//===----------------------------------------------------------------------===//

void MakeBufferBarrierOp::build(Builder *builder, OperationState &state,
                                IREE::HAL::AccessScopeBitfield sourceScope,
                                IREE::HAL::AccessScopeBitfield targetScope,
                                Value buffer, Value offset, Value length) {
  state.addAttribute("source_scope", builder->getI32IntegerAttr(
                                         static_cast<int32_t>(sourceScope)));
  state.addAttribute("target_scope", builder->getI32IntegerAttr(
                                         static_cast<int32_t>(targetScope)));
  state.addOperands({buffer, offset, length});
  state.addTypes({BufferBarrierType::get(builder->getContext())});
}

void MakeBufferBarrierOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
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
  p.printType(op.result().getType());
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

void VariableOp::build(Builder *builder, OperationState &result, StringRef name,
                       bool isMutable, Type type,
                       Optional<StringRef> initializer,
                       Optional<Attribute> initialValue,
                       ArrayRef<NamedAttribute> attrs) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder->getStringAttr(name));
  if (isMutable) {
    result.addAttribute("is_mutable", builder->getUnitAttr());
  }
  if (initializer.hasValue()) {
    result.addAttribute("initializer",
                        builder->getSymbolRefAttr(initializer.getValue()));
  } else if (initialValue.hasValue()) {
    result.addAttribute("initial_value", initialValue.getValue());
  }
  result.addAttribute("type", TypeAttr::get(type));
  result.attributes.append(attrs.begin(), attrs.end());
}

void VariableOp::build(Builder *builder, OperationState &result, StringRef name,
                       bool isMutable, mlir::FuncOp initializer,
                       ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, initializer.getType().getResult(0),
        initializer.getName(), llvm::None, attrs);
}

void VariableOp::build(Builder *builder, OperationState &result, StringRef name,
                       bool isMutable, Type type, Attribute initialValue,
                       ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, type, llvm::None, initialValue,
        attrs);
}

void VariableOp::build(Builder *builder, OperationState &result, StringRef name,
                       bool isMutable, Type type,
                       ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, type, llvm::None, llvm::None, attrs);
}

//===----------------------------------------------------------------------===//
// hal.variable.load
//===----------------------------------------------------------------------===//

static ParseResult parseVariableLoadOp(OpAsmParser &parser,
                                       OperationState *result) {
  FlatSymbolRefAttr variableAttr;
  Type valueType;
  if (failed(parser.parseAttribute(variableAttr, "variable",
                                   result->attributes)) ||
      failed(parser.parseOptionalAttrDict(result->attributes)) ||
      failed(parser.parseColonType(valueType))) {
    return failure();
  }
  result->addTypes({valueType});
  return success();
}

static void printVariableLoadOp(OpAsmPrinter &p, VariableLoadOp &op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.variable());
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"variable"});
  p << " : ";
  p.printType(op.result().getType());
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
// hal.variable.store
//===----------------------------------------------------------------------===//

static ParseResult parseVariableStoreOp(OpAsmParser &parser,
                                        OperationState *result) {
  OpAsmParser::OperandType value;
  FlatSymbolRefAttr variableAttr;
  Type valueType;
  if (failed(parser.parseOperand(value)) || failed(parser.parseComma()) ||
      failed(parser.parseAttribute(variableAttr, "variable",
                                   result->attributes)) ||
      failed(parser.parseOptionalAttrDict(result->attributes)) ||
      failed(parser.parseColonType(valueType)) ||
      failed(parser.resolveOperand(value, valueType, result->operands))) {
    return failure();
  }
  return success();
}

static void printVariableStoreOp(OpAsmPrinter &p, VariableStoreOp &op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.value());
  p << ", ";
  p.printSymbolName(op.variable());
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"variable"});
  p << " : ";
  p.printType(op.value().getType());
}

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
// hal.allocator.compute_size
//===----------------------------------------------------------------------===//

void AllocatorComputeSizeOp::build(Builder *builder, OperationState &state,
                                   Value allocator, ValueRange shape,
                                   int32_t elementType) {
  state.addOperands({allocator});
  state.addOperands(shape);
  state.addAttribute("element_type", builder->getI32IntegerAttr(elementType));
  state.addTypes({builder->getIntegerType(32)});
}

void AllocatorComputeSizeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "sz");
}

static ParseResult parseAllocatorComputeSizeOp(OpAsmParser &parser,
                                               OperationState *result) {
  OpAsmParser::OperandType allocator;
  SmallVector<OpAsmParser::OperandType, 4> shape;
  IntegerAttr elementType;
  if (failed(parser.parseOperand(allocator)) ||
      failed(parser.resolveOperand(
          allocator, RefPtrType::get(AllocatorType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("shape")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseOperandList(shape, OpAsmParser::Delimiter::Square)) ||
      failed(parser.resolveOperands(shape, getDimType(parser),
                                    result->operands)) ||
      failed(parser.parseComma()) ||
      failed(parser.parseKeyword("element_type")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(elementType, getDeviceSizeType(parser),
                                   "element_type", result->attributes)) ||
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
  p << ", shape=[";
  p.printOperands(op.shape());
  p << "], element_type=" << op.element_type();
  p.printOptionalAttrDictWithKeyword(
      op.getAttrs(),
      /*elidedAttrs=*/{"memory_types", "buffer_usage", "element_type"});
}

//===----------------------------------------------------------------------===//
// hal.allocator.compute_offset
//===----------------------------------------------------------------------===//

void AllocatorComputeOffsetOp::build(Builder *builder, OperationState &state,
                                     Value allocator, ValueRange shape,
                                     int32_t elementType, ValueRange indices) {
  state.addOperands({allocator});
  state.addOperands(shape);
  state.addAttribute("element_type", builder->getI32IntegerAttr(elementType));
  state.addOperands(indices);
  state.addTypes({builder->getIntegerType(32)});
}

void AllocatorComputeOffsetOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(offset(), "off");
}

static ParseResult parseAllocatorComputeOffsetOp(OpAsmParser &parser,
                                                 OperationState *result) {
  OpAsmParser::OperandType allocator;
  SmallVector<OpAsmParser::OperandType, 4> shape;
  SmallVector<OpAsmParser::OperandType, 4> indices;
  IntegerAttr elementType;
  if (failed(parser.parseOperand(allocator)) ||
      failed(parser.resolveOperand(
          allocator, RefPtrType::get(AllocatorType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("shape")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseOperandList(shape, OpAsmParser::Delimiter::Square)) ||
      failed(parser.resolveOperands(shape, getDimType(parser),
                                    result->operands)) ||
      failed(parser.parseComma()) ||
      failed(parser.parseKeyword("element_type")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(elementType,
                                   parser.getBuilder().getIntegerType(32),
                                   "element_type", result->attributes)) ||
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

static void printAllocatorComputeOffsetOp(OpAsmPrinter &p,
                                          AllocatorComputeOffsetOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.allocator());
  p << ", shape=[";
  p.printOperands(op.shape());
  p << "], element_type=" << op.element_type();
  p << ", indices=[";
  p.printOperands(op.indices());
  p << "]";
  p.printOptionalAttrDictWithKeyword(
      op.getAttrs(),
      /*elidedAttrs=*/{"memory_types", "buffer_usage", "element_type"});
}

//===----------------------------------------------------------------------===//
// hal.allocator.compute_range
//===----------------------------------------------------------------------===//

void AllocatorComputeRangeOp::build(Builder *builder, OperationState &state,
                                    Value allocator, ValueRange shape,
                                    int32_t elementType, ValueRange indices,
                                    ValueRange lengths) {
  state.addOperands({allocator});
  state.addOperands(shape);
  state.addAttribute("element_type", builder->getI32IntegerAttr(elementType));
  state.addOperands(indices);
  state.addOperands(lengths);
  state.addTypes({builder->getIntegerType(32), builder->getIntegerType(32)});
}

void AllocatorComputeRangeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(offset(), "off");
  setNameFn(length(), "len");
}

static ParseResult parseAllocatorComputeRangeOp(OpAsmParser &parser,
                                                OperationState *result) {
  OpAsmParser::OperandType allocator;
  SmallVector<OpAsmParser::OperandType, 4> shape;
  SmallVector<OpAsmParser::OperandType, 4> indices;
  SmallVector<OpAsmParser::OperandType, 4> lengths;
  IntegerAttr elementType;
  if (failed(parser.parseOperand(allocator)) ||
      failed(parser.resolveOperand(
          allocator, RefPtrType::get(AllocatorType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("shape")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseOperandList(shape, OpAsmParser::Delimiter::Square)) ||
      failed(parser.resolveOperands(shape, getDimType(parser),
                                    result->operands)) ||
      failed(parser.parseComma()) ||
      failed(parser.parseKeyword("element_type")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(elementType,
                                   parser.getBuilder().getIntegerType(32),
                                   "element_type", result->attributes)) ||
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

static void printAllocatorComputeRangeOp(OpAsmPrinter &p,
                                         AllocatorComputeRangeOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.allocator());
  p << ", shape=[";
  p.printOperands(op.shape());
  p << "], element_type=" << op.element_type();
  p << ", indices=[";
  p.printOperands(op.indices());
  p << "], lengths=[";
  p.printOperands(op.lengths());
  p << "]";
  p.printOptionalAttrDictWithKeyword(
      op.getAttrs(),
      /*elidedAttrs=*/{"memory_types", "buffer_usage", "element_type"});
}

//===----------------------------------------------------------------------===//
// hal.allocator.allocate
//===----------------------------------------------------------------------===//

void AllocatorAllocateOp::build(Builder *builder, OperationState &state,
                                Value allocator,
                                IREE::HAL::MemoryTypeBitfield memoryTypes,
                                IREE::HAL::BufferUsageBitfield bufferUsage,
                                Value allocationSize) {
  state.addOperands({allocator, allocationSize});
  state.addAttribute("memory_types", builder->getI32IntegerAttr(
                                         static_cast<int32_t>(memoryTypes)));
  state.addAttribute("buffer_usage", builder->getI32IntegerAttr(
                                         static_cast<int32_t>(bufferUsage)));
  state.addTypes({RefPtrType::get(BufferType::get(builder->getContext()))});
}

void AllocatorAllocateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
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
  p.printType(op.result().getType());
}

//===----------------------------------------------------------------------===//
// hal.allocator.allocate.const
//===----------------------------------------------------------------------===//

void AllocatorAllocateConstOp::build(Builder *builder, OperationState &state,
                                     Value allocator,
                                     IREE::HAL::MemoryTypeBitfield memoryTypes,
                                     IREE::HAL::BufferUsageBitfield bufferUsage,
                                     ElementsAttr value) {
  state.addOperands({allocator});
  state.addAttribute("memory_types", builder->getI32IntegerAttr(
                                         static_cast<int32_t>(memoryTypes)));
  state.addAttribute("buffer_usage", builder->getI32IntegerAttr(
                                         static_cast<int32_t>(bufferUsage)));
  state.addAttribute("value", value);
  state.addTypes({RefPtrType::get(BufferType::get(builder->getContext()))});
}

void AllocatorAllocateConstOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "cbuffer");
}

static ParseResult parseAllocatorAllocateConstOp(OpAsmParser &parser,
                                                 OperationState *result) {
  OpAsmParser::OperandType allocator;
  ElementsAttr valueAttr;
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
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseColonType(resultType)) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(valueAttr, "value", result->attributes))) {
    return failure();
  }
  result->addTypes(resultType);
  return success();
}

static void printAllocatorAllocateConstOp(OpAsmPrinter &p,
                                          AllocatorAllocateConstOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.allocator());
  p << ", ";
  p << "\"" << stringifyMemoryTypeBitfield(op.memory_types()) << "\"";
  p << ", ";
  p << "\"" << stringifyBufferUsageBitfield(op.buffer_usage()) << "\"";
  p.printOptionalAttrDictWithKeyword(
      op.getAttrs(),
      /*elidedAttrs=*/{"memory_types", "buffer_usage", "value"});
  p << " : ";
  p.printType(op.result().getType());
  p << " = ";
  p.printAttribute(op.value());
}

//===----------------------------------------------------------------------===//
// hal.buffer.allocator
//===----------------------------------------------------------------------===//

void BufferAllocatorOp::build(Builder *builder, OperationState &state,
                              Value buffer) {
  state.addOperands({buffer});
  state.addTypes({RefPtrType::get(AllocatorType::get(builder->getContext()))});
}

void BufferAllocatorOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "allocator");
}

static ParseResult parseBufferAllocatorOp(OpAsmParser &parser,
                                          OperationState *result) {
  OpAsmParser::OperandType buffer;
  Type resultType;
  if (failed(parser.parseOperand(buffer)) ||
      failed(parser.resolveOperand(
          buffer, RefPtrType::get(BufferType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseColonType(resultType)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  result->addTypes(resultType);
  return success();
}

static void printBufferAllocatorOp(OpAsmPrinter &p, BufferAllocatorOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.buffer());
  p << " : ";
  p.printType(op.result().getType());
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// hal.buffer.subspan
//===----------------------------------------------------------------------===//

void BufferSubspanOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
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
  p.printType(op.result().getType());
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
  p.printType(op.target_buffer().getType());
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
  p.printType(op.source_buffer().getType());
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
// hal.buffer.load
//===----------------------------------------------------------------------===//

static ParseResult parseBufferLoadOp(OpAsmParser &parser,
                                     OperationState *result) {
  OpAsmParser::OperandType sourceBuffer;
  OpAsmParser::OperandType sourceOffset;
  Type resultType;
  if (failed(parser.parseOperand(sourceBuffer)) ||
      failed(parser.resolveOperand(
          sourceBuffer, RefPtrType::get(BufferType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseLSquare()) ||
      failed(parser.parseOperand(sourceOffset)) ||
      failed(parser.parseRSquare()) ||
      failed(parser.resolveOperand(sourceOffset, getDeviceSizeType(parser),
                                   result->operands)) ||
      failed(parser.parseColonType(resultType)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  result->addTypes({resultType});
  return success();
}

static void printBufferLoadOp(OpAsmPrinter &p, BufferLoadOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.source_buffer());
  p << "[";
  p.printOperand(op.source_offset());
  p << "] : ";
  p.printType(op.result().getType());
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// hal.buffer.store
//===----------------------------------------------------------------------===//

static ParseResult parseBufferStoreOp(OpAsmParser &parser,
                                      OperationState *result) {
  OpAsmParser::OperandType value;
  OpAsmParser::OperandType targetBuffer;
  OpAsmParser::OperandType targetOffset;
  Type valueType;
  if (failed(parser.parseOperand(value)) || failed(parser.parseComma()) ||
      failed(parser.parseOperand(targetBuffer)) ||
      failed(parser.parseLSquare()) ||
      failed(parser.parseOperand(targetOffset)) ||
      failed(parser.parseRSquare()) ||
      failed(parser.parseColonType(valueType)) ||
      failed(parser.resolveOperand(value, valueType, result->operands)) ||
      failed(parser.resolveOperand(
          targetBuffer, RefPtrType::get(BufferType::get(result->getContext())),
          result->operands)) ||
      failed(parser.resolveOperand(targetOffset, getDeviceSizeType(parser),
                                   result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  return success();
}

static void printBufferStoreOp(OpAsmPrinter &p, BufferStoreOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.value());
  p << ", ";
  p.printOperand(op.target_buffer());
  p << "[";
  p.printOperand(op.target_offset());
  p << "] : ";
  p.printType(op.value().getType());
  p.printOptionalAttrDictWithKeyword(op.getAttrs(),
                                     /*elidedAttrs=*/{"element_type"});
}

//===----------------------------------------------------------------------===//
// hal.buffer_view.create
//===----------------------------------------------------------------------===//

void BufferViewCreateOp::build(Builder *builder, OperationState &state,
                               Value buffer, ValueRange shape,
                               int32_t elementType) {
  state.addOperands({buffer});
  state.addOperands(shape);
  state.addAttribute("element_type", builder->getI32IntegerAttr(elementType));
  state.addTypes({RefPtrType::get(BufferViewType::get(builder->getContext()))});
}

void BufferViewCreateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "view");
}

static ParseResult parseBufferViewCreateOp(OpAsmParser &parser,
                                           OperationState *result) {
  OpAsmParser::OperandType buffer;
  SmallVector<OpAsmParser::OperandType, 4> shape;
  IntegerAttr elementType;
  Type resultType;
  if (failed(parser.parseOperand(buffer)) ||
      failed(parser.resolveOperand(
          buffer, RefPtrType::get(BufferType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("shape")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseOperandList(shape, OpAsmParser::Delimiter::Square)) ||
      failed(parser.resolveOperands(shape, getDimType(parser),
                                    result->operands)) ||
      failed(parser.parseComma()) ||
      failed(parser.parseKeyword("element_type")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(elementType,
                                   parser.getBuilder().getIntegerType(32),
                                   "element_type", result->attributes)) ||
      failed(parser.parseColonType(resultType)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  result->addTypes(resultType);
  return success();
}

static void printBufferViewCreateOp(OpAsmPrinter &p, BufferViewCreateOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.buffer());
  p << ", shape=[";
  p.printOperands(op.shape());
  p << "], element_type=" << op.element_type();
  p << " : ";
  p.printType(op.result().getType());
  p.printOptionalAttrDictWithKeyword(op.getAttrs(),
                                     /*elidedAttrs=*/{"element_type"});
}

//===----------------------------------------------------------------------===//
// hal.buffer_view.subview
//===----------------------------------------------------------------------===//

void BufferViewSubviewOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "view");
}

static ParseResult parseBufferViewSubviewOp(OpAsmParser &parser,
                                            OperationState *result) {
  OpAsmParser::OperandType bufferView;
  SmallVector<OpAsmParser::OperandType, 4> indices;
  SmallVector<OpAsmParser::OperandType, 4> lengths;
  Type resultType;
  if (failed(parser.parseOperand(bufferView)) ||
      failed(parser.resolveOperand(
          bufferView,
          RefPtrType::get(BufferViewType::get(result->getContext())),
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
      failed(parser.parseColonType(resultType)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  result->addTypes(resultType);
  return success();
}

static void printBufferViewSubviewOp(OpAsmPrinter &p, BufferViewSubviewOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.buffer_view());
  p << ", indices=[";
  p.printOperands(op.indices());
  p << "], lengths=[";
  p.printOperands(op.lengths());
  p << "] : ";
  p.printType(op.result().getType());
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// hal.buffer_view.buffer
//===----------------------------------------------------------------------===//

void BufferViewBufferOp::build(Builder *builder, OperationState &state,
                               Value bufferView) {
  state.addOperands({bufferView});
  state.addTypes({RefPtrType::get(BufferType::get(builder->getContext()))});
}

void BufferViewBufferOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "buffer");
}

static ParseResult parseBufferViewBufferOp(OpAsmParser &parser,
                                           OperationState *result) {
  OpAsmParser::OperandType bufferView;
  Type bufferType;
  if (failed(parser.parseOperand(bufferView)) ||
      failed(parser.parseColonType(bufferType)) ||
      failed(parser.resolveOperand(
          bufferView,
          RefPtrType::get(BufferViewType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  result->addTypes(bufferType);
  return success();
}

static void printBufferViewBufferOp(OpAsmPrinter &p, BufferViewBufferOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.buffer_view());
  p << " : ";
  p.printType(op.result().getType());
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// hal.buffer_view.byte_length
//===----------------------------------------------------------------------===//

void BufferViewByteLengthOp::build(Builder *builder, OperationState &state,
                                   Value bufferView) {
  state.addOperands({bufferView});
  state.addTypes({builder->getIntegerType(32)});
}

void BufferViewByteLengthOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "len");
}

static ParseResult parseBufferViewByteLengthOp(OpAsmParser &parser,
                                               OperationState *result) {
  OpAsmParser::OperandType bufferView;
  SmallVector<OpAsmParser::OperandType, 4> indices;
  Type lengthType;
  if (failed(parser.parseOperand(bufferView)) ||
      failed(parser.resolveOperand(
          bufferView,
          RefPtrType::get(BufferViewType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseColonType(lengthType)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }
  result->addTypes(getDeviceSizeType(parser));
  return success();
}

static void printBufferViewByteLengthOp(OpAsmPrinter &p,
                                        BufferViewByteLengthOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.buffer_view());
  p << " : ";
  p.printType(op.result().getType());
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// hal.buffer_view.compute_offset
//===----------------------------------------------------------------------===//

void BufferViewComputeOffsetOp::build(Builder *builder, OperationState &state,
                                      Value bufferView, ValueRange indices) {
  state.addOperands({bufferView});
  state.addOperands(indices);
  state.addTypes({builder->getIntegerType(32)});
}

void BufferViewComputeOffsetOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(offset(), "off");
}

static ParseResult parseBufferViewComputeOffsetOp(OpAsmParser &parser,
                                                  OperationState *result) {
  OpAsmParser::OperandType bufferView;
  SmallVector<OpAsmParser::OperandType, 4> indices;
  if (failed(parser.parseOperand(bufferView)) ||
      failed(parser.resolveOperand(
          bufferView,
          RefPtrType::get(BufferViewType::get(result->getContext())),
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
  p.printOperand(op.buffer_view());
  p << ", indices=[";
  p.printOperands(op.indices());
  p << "]";
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// hal.buffer_view.compute_range
//===----------------------------------------------------------------------===//

void BufferViewComputeRangeOp::build(Builder *builder, OperationState &state,
                                     Value bufferView, ValueRange indices,
                                     ValueRange lengths) {
  state.addOperands({bufferView});
  state.addOperands(indices);
  state.addOperands(lengths);
  state.addTypes({builder->getIntegerType(32), builder->getIntegerType(32)});
}

void BufferViewComputeRangeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(offset(), "off");
  setNameFn(length(), "len");
}

static ParseResult parseBufferViewComputeRangeOp(OpAsmParser &parser,
                                                 OperationState *result) {
  OpAsmParser::OperandType bufferView;
  SmallVector<OpAsmParser::OperandType, 4> indices;
  SmallVector<OpAsmParser::OperandType, 4> lengths;
  if (failed(parser.parseOperand(bufferView)) ||
      failed(parser.resolveOperand(
          bufferView,
          RefPtrType::get(BufferViewType::get(result->getContext())),
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
  p.printOperand(op.buffer_view());
  p << ", indices=[";
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
    Builder *builder, OperationState &state, Value device,
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
    function_ref<void(Value, StringRef)> setNameFn) {
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
  p.printType(op.result().getType());
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
    Builder *builder, OperationState &state, Value commandBuffer,
    IREE::HAL::ExecutionStageBitfield sourceStageMask,
    IREE::HAL::ExecutionStageBitfield targetStageMask,
    ValueRange memoryBarriers, ValueRange bufferBarriers) {
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
                                             Value commandBuffer,
                                             Value executableLayout,
                                             uint32_t set, Value descriptorSet,
                                             ValueRange dynamicOffsets) {
  state.addOperands({commandBuffer, executableLayout, descriptorSet});
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
              RefPtrType::get(ExecutableLayoutType::get(result->getContext())),
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
  p.printOperand(op.executable_layout());
  p << ", set=" << op.set() << ", ";
  p.printOperand(op.descriptor_set());
  if (!op.dynamic_offsets().empty()) {
    p << ", offsets=[";
    interleaveComma(op.dynamic_offsets(), p,
                    [&](Value value) { p.printOperand(value); });
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
    Builder *builder, OperationState &state, Value commandBuffer,
    Value executable, IREE::HAL::ExecutableEntryPointOp entryPoint,
    Value workgroupX, Value workgroupY, Value workgroupZ) {
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
      ValueRange{op.workgroup_x(), op.workgroup_y(), op.workgroup_z()}, p,
      [&](Value value) { p.printOperand(value); });
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
// hal.descriptor_set.make_binding
//===----------------------------------------------------------------------===//

void DescriptorSetMakeBindingOp::build(Builder *builder, OperationState &state,
                                       int32_t binding, Value buffer,
                                       Value offset, Value length) {
  state.addAttribute("binding", builder->getI32IntegerAttr(binding));
  state.addOperands({buffer, offset, length});
  state.addTypes({DescriptorSetBindingType::get(builder->getContext())});
}

void DescriptorSetMakeBindingOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
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
  p.printOptionalAttrDictWithKeyword(op.getAttrs(),
                                     /*elidedAttrs=*/{"binding"});
  p << " : ";
  p.printType(op.result().getType());
}

//===----------------------------------------------------------------------===//
// hal.descriptor_set.create
//===----------------------------------------------------------------------===//

void DescriptorSetCreateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "descriptor_set");
}

static ParseResult parseDescriptorSetCreateOp(OpAsmParser &parser,
                                              OperationState *result) {
  OpAsmParser::OperandType device;
  OpAsmParser::OperandType setLayout;
  SmallVector<OpAsmParser::OperandType, 4> bindings;
  Type resultType;
  if (failed(parser.parseOperand(device)) ||
      failed(parser.resolveOperand(
          device, RefPtrType::get(DeviceType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseComma()) || failed(parser.parseOperand(setLayout)) ||
      failed(parser.resolveOperand(
          setLayout,
          RefPtrType::get(DescriptorSetLayoutType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("bindings")) ||
      failed(parser.parseEqual()) ||
      failed(
          parser.parseOperandList(bindings, OpAsmParser::Delimiter::Square)) ||
      failed(parser.resolveOperands(
          bindings, DescriptorSetBindingType::get(result->getContext()),
          result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseColonType(resultType))) {
    return failure();
  }
  result->addTypes(resultType);
  return success();
}

static void printDescriptorSetCreateOp(OpAsmPrinter &p,
                                       DescriptorSetCreateOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.device());
  p << ", ";
  p.printOperand(op.set_layout());
  p << ", bindings=[";
  p.printOperands(op.bindings());
  p << "]";
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
  p << " : ";
  p.printType(op.result().getType());
}

//===----------------------------------------------------------------------===//
// hal.descriptor_set_layout.make_binding
//===----------------------------------------------------------------------===//

void DescriptorSetLayoutMakeBindingOp::build(
    Builder *builder, OperationState &state, int32_t binding,
    IREE::HAL::DescriptorType type, IREE::HAL::MemoryAccessBitfield access) {
  state.addAttribute("binding", builder->getI32IntegerAttr(binding));
  state.addAttribute("type",
                     builder->getI32IntegerAttr(static_cast<int32_t>(type)));
  state.addAttribute("access",
                     builder->getI32IntegerAttr(static_cast<int32_t>(access)));
  state.addTypes({DescriptorSetLayoutBindingType::get(builder->getContext())});
}

void DescriptorSetLayoutMakeBindingOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "binding");
}

static ParseResult parseDescriptorSetLayoutMakeBindingOp(
    OpAsmParser &parser, OperationState *result) {
  IntegerAttr bindingAttr;
  OpAsmParser::OperandType buffer;
  OpAsmParser::OperandType offset;
  OpAsmParser::OperandType length;
  llvm::SMLoc operandLoc;
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
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseColonType(resultType))) {
    return failure();
  }
  result->addTypes(resultType);
  return success();
}

static void printDescriptorSetLayoutMakeBindingOp(
    OpAsmPrinter &p, DescriptorSetLayoutMakeBindingOp op) {
  p << op.getOperationName() << ' ';
  p << "binding=" << op.binding() << ", ";
  p.printOperands(op.getOperation()->getOperands());
  p.printOptionalAttrDictWithKeyword(op.getAttrs(),
                                     /*elidedAttrs=*/{"binding"});
  p << " : ";
  p.printType(op.result().getType());
}

//===----------------------------------------------------------------------===//
// hal.descriptor_set_layout.create
//===----------------------------------------------------------------------===//

void DescriptorSetLayoutCreateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "descriptor_set_layout");
}

static ParseResult parseDescriptorSetLayoutCreateOp(OpAsmParser &parser,
                                                    OperationState *result) {
  OpAsmParser::OperandType device;
  OpAsmParser::OperandType setLayout;
  SmallVector<OpAsmParser::OperandType, 4> bindings;
  Type resultType;
  if (failed(parser.parseOperand(device)) ||
      failed(parser.resolveOperand(
          device, RefPtrType::get(DeviceType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseComma()) || failed(parser.parseOperand(setLayout)) ||
      failed(parser.resolveOperand(
          setLayout,
          RefPtrType::get(DescriptorSetLayoutType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseComma()) || failed(parser.parseKeyword("bindings")) ||
      failed(parser.parseEqual()) ||
      failed(
          parser.parseOperandList(bindings, OpAsmParser::Delimiter::Square)) ||
      failed(parser.resolveOperands(
          bindings, DescriptorSetBindingType::get(result->getContext()),
          result->operands)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseColonType(resultType))) {
    return failure();
  }
  result->addTypes(resultType);
  return success();
}

static void printDescriptorSetLayoutCreateOp(OpAsmPrinter &p,
                                             DescriptorSetLayoutCreateOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.device());
  p << ", bindings=[";
  p.printOperands(op.bindings());
  p << "]";
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
  p << " : ";
  p.printType(op.result().getType());
}

//===----------------------------------------------------------------------===//
// hal.device.allocator
//===----------------------------------------------------------------------===//

void DeviceAllocatorOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
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
  p.printType(op.result().getType());
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
// hal.executable_layout.create
//===----------------------------------------------------------------------===//

void ExecutableLayoutCreateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "executable_layout");
}

static ParseResult parseExecutableLayoutCreateOp(OpAsmParser &parser,
                                                 OperationState *result) {
  OpAsmParser::OperandType device;
  SmallVector<OpAsmParser::OperandType, 4> setLayouts;
  IntegerAttr pushConstants;
  Type resultType;
  if (failed(parser.parseOperand(device)) ||
      failed(parser.resolveOperand(
          device, RefPtrType::get(DeviceType::get(result->getContext())),
          result->operands)) ||
      failed(parser.parseComma()) ||
      failed(parser.parseKeyword("set_layouts")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseOperandList(setLayouts,
                                     OpAsmParser::Delimiter::Square)) ||
      failed(parser.resolveOperands(
          setLayouts, DescriptorSetLayoutType::get(result->getContext()),
          result->operands))) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (failed(parser.parseKeyword("push_constants")) ||
        failed(parser.parseEqual()) ||
        failed(parser.parseAttribute(pushConstants, "push_constants",
                                     result->attributes))) {
      return failure();
    }
  }
  if (failed(parser.parseOptionalAttrDictWithKeyword(result->attributes)) ||
      failed(parser.parseColonType(resultType))) {
    return failure();
  }
  result->addTypes(resultType);
  return success();
}

static void printExecutableLayoutCreateOp(OpAsmPrinter &p,
                                          ExecutableLayoutCreateOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.device());
  p << ", set_layouts=[";
  p.printOperands(op.set_layouts());
  p << "]";
  if (op.push_constants()) {
    p << ", push_constants=";
    p.printAttribute(op.push_constantsAttr());
  }
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
  p << " : ";
  p.printType(op.result().getType());
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
