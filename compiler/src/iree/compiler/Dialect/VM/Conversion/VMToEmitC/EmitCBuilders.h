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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace iree_compiler {
namespace emitc_builders {

struct StructField {
  std::string type;
  std::string name;
};

enum UnaryOperator {
  // arithmetic
  PLUS = 0,
  MINUS,
  BITWISE_NOT,
  // logical
  LOGICAL_NOT,
};

enum BinaryOperator {
  // arithmetic
  ADDITION = 0,
  SUBTRACTION,
  PRODUCT,
  DIVISION,
  REMAINDER,
  BITWISE_AND,
  BITWISE_OR,
  BITWISE_XOR,
  BITWISE_LEFT_SHIFT,
  BITWISE_RIGHT_SHIFT,
  // logical
  LOGICAL_AND,
  LOGICAL_OR,
  // comparison
  EQUAL_TO,
  NOT_EQUAL_TO,
  LESS_THAN,
  GREATER_THAN,
  LESS_THAN_OR_EQUAL,
  GREATER_THAN_OR_EQUAL,
};

Value unaryOperator(OpBuilder builder, Location location, UnaryOperator op,
                    Value operand, Type resultType);

Value binaryOperator(OpBuilder builder, Location location, BinaryOperator op,
                     Value lhs, Value rhs, Type resultType);

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

void structMemberAssign(OpBuilder builder, Location location,
                        StringRef memberName, Value operand, Value data);

void structMemberAssign(OpBuilder builder, Location location,
                        StringRef memberName, Value operand, StringRef data);

Value structPtrMember(OpBuilder builder, Location location, Type type,
                      StringRef memberName, Value operand);

void structPtrMemberAssign(OpBuilder builder, Location location,
                           StringRef memberName, Value operand, Value data);

Value ireeMakeCstringView(OpBuilder builder, Location location,
                          std::string str);

Value ireeOkStatus(OpBuilder builder, Location location);

Value ireeVmInstanceLookupType(OpBuilder builder, Location location,
                               Value instance, Value stringView);

void ireeVmRefRelease(OpBuilder builder, Location location, Value operand);

}  // namespace emitc_builders
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCBUILDERS_H_
