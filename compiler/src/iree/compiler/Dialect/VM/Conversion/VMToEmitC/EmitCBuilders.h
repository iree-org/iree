// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCBUILDERS_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCBUILDERS_H_

#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/EmitCTypeConverter.h"

namespace mlir::iree_compiler::emitc_builders {

struct StructField {
  std::string type;
  std::string name;
  std::optional<size_t> arraySize = std::nullopt;

  bool isArray() const { return arraySize.has_value(); }
};

enum PreprocessorDirective {
  DEFINE = 0,
  UNDEF,
  IFDEF,
  IFNDEF,
  IF,
  ENDIF,
  ELSE,
  ELIF,
  LINE,
  ERROR,
  INCLUDE,
  PRAGMA
};

TypedValue<emitc::LValueType> allocateVariable(OpBuilder builder,
                                               Location location, Type type,
                                               Attribute initializer);

TypedValue<emitc::LValueType>
allocateVariable(OpBuilder builder, Location location, Type type,
                 std::optional<StringRef> initializer = std::nullopt);

/// Allocate a new zero initialized variable. This is done through a call to
/// memset, as all variables are declared without initializer in the emitter.
std::pair<TypedValue<emitc::LValueType>, TypedValue<emitc::PointerType>>
allocZeroInitializedVar(OpBuilder builder, Location location, Type type);

/// Convert a value to an EmitC LValue by allocating a new variable and
/// assigning the operand to it. Note that the variable declaration and
/// assignment are split into two separate statements, so that padding bytes for
/// struct values are not copied.
TypedValue<emitc::LValueType> asLValue(OpBuilder builder, Location loc,
                                       Value value);
Value asRValue(OpBuilder builder, Location loc,
               TypedValue<emitc::LValueType> value);

/// Replace values of lvalue type with rvalues.
void asRValues(OpBuilder builder, Location location,
               SmallVector<Value> &values);

TypedValue<emitc::PointerType> addressOf(OpBuilder builder, Location location,
                                         TypedValue<emitc::LValueType> operand);

Value contentsOf(OpBuilder builder, Location location, Value operand);

Value sizeOf(OpBuilder builder, Location location, Attribute attr);

Value sizeOf(OpBuilder builder, Location location, Value value);

void memcpy(OpBuilder builder, Location location, Value dest, Value src,
            Value count);

void memset(OpBuilder builder, Location location, Value dest, int ch,
            Value count);

Value arrayElement(OpBuilder builder, Location location, size_t index,
                   TypedValue<emitc::PointerType> operand);

Value arrayElementAddress(OpBuilder builder, Location location, size_t index,
                          TypedValue<emitc::PointerType> operand);

Value arrayElementAddress(OpBuilder builder, Location location, Value index,
                          TypedValue<emitc::PointerType> operand);

void arrayElementAssign(OpBuilder builder, Location location,
                        TypedValue<emitc::PointerType> array, size_t index,
                        Value value);

void structDefinition(OpBuilder builder, Location location,
                      StringRef structName, ArrayRef<StructField> fields);

Value structMember(OpBuilder builder, Location location, Type type,
                   StringRef memberName, TypedValue<emitc::LValueType> operand);

TypedValue<emitc::PointerType>
structMemberAddress(OpBuilder builder, Location location,
                    emitc::PointerType type, StringRef memberName,
                    TypedValue<emitc::LValueType> operand);

void structMemberAssign(OpBuilder builder, Location location,
                        StringRef memberName,
                        TypedValue<emitc::LValueType> operand, Value data);

Value structPtrMember(OpBuilder builder, Location location, Type type,
                      StringRef memberName,
                      TypedValue<emitc::LValueType> operand);

TypedValue<emitc::PointerType>
structPtrMemberAddress(OpBuilder builder, Location location,
                       emitc::PointerType type, StringRef memberName,
                       TypedValue<emitc::LValueType> operand);

void structPtrMemberAssign(OpBuilder builder, Location location,
                           StringRef memberName,
                           TypedValue<emitc::LValueType> operand, Value data);

Value ireeMakeCstringView(OpBuilder builder, Location location,
                          std::string str);

Value ireeOkStatus(OpBuilder builder, Location location);

Value ireeVmInstanceLookupType(OpBuilder builder, Location location,
                               Value instance, Value stringView);

void ireeVmRefRelease(OpBuilder builder, Location location, Value operand);

emitc::VerbatimOp preprocessorDirective(OpBuilder builder, Location location,
                                        PreprocessorDirective directive,
                                        StringRef value);

} // namespace mlir::iree_compiler::emitc_builders

#endif // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCBUILDERS_H_
