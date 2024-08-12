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

Value allocateVariable(OpBuilder builder, Location location, Type type,
                       Attribute initializer);

Value allocateVariable(OpBuilder builder, Location location, Type type,
                       std::optional<StringRef> initializer = std::nullopt);

Value addressOf(OpBuilder builder, Location location, Value operand);

Value contentsOf(OpBuilder builder, Location location, Value operand);

Value sizeOf(OpBuilder builder, Location location, Attribute attr);

Value sizeOf(OpBuilder builder, Location location, Value value);

void memcpy(OpBuilder builder, Location location, Value dest, Value src,
            Value count);

void memset(OpBuilder builder, Location location, Value dest, int ch,
            Value count);

Value arrayElement(OpBuilder builder, Location location, Type type,
                   size_t index, Value operand);

Value arrayElementAddress(OpBuilder builder, Location location, Type type,
                          IntegerAttr index, Value operand);

Value arrayElementAddress(OpBuilder builder, Location location, Type type,
                          Value index, Value operand);

void arrayElementAssign(OpBuilder builder, Location location, Value array,
                        size_t index, Value value);

void structDefinition(OpBuilder builder, Location location,
                      StringRef structName, ArrayRef<StructField> fields);

Value structMember(OpBuilder builder, Location location, Type type,
                   StringRef memberName, Value operand);

Value structMemberAddress(OpBuilder builder, Location location,
                          emitc::PointerType type, StringRef memberName,
                          Value operand);

void structMemberAssign(OpBuilder builder, Location location,
                        StringRef memberName, Value operand, Value data);

void structMemberAssign(OpBuilder builder, Location location,
                        StringRef memberName, Value operand, StringRef data);

Value structPtrMember(OpBuilder builder, Location location, Type type,
                      StringRef memberName, Value operand);

Value structPtrMemberAddress(OpBuilder builder, Location location,
                             emitc::PointerType type, StringRef memberName,
                             Value operand);

void structPtrMemberAssign(OpBuilder builder, Location location,
                           StringRef memberName, Value operand, Value data);

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
