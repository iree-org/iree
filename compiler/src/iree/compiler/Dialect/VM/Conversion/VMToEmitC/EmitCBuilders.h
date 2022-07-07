// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCBUILDERS_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCBUILDERS_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace iree_compiler {
namespace emitc_builders {

typedef struct StructField {
  std::string type;
  std::string name;
} StructField;

Value arrayElementAddress(OpBuilder builder, Location location, Type type,
                          IntegerAttr index, Value operand);

void structDefinition(OpBuilder builder, Location location,
                      StringRef structName, SmallVector<StructField> fields);

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

Value ireeOkStatus(OpBuilder builder, Location location);

void ireeVmRefRelease(OpBuilder builder, Location location, Value operand);

}  // namespace emitc_builders
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCBUILDERS_H_
