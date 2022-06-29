// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCBUILDERS_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCBUILDERS_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace iree_compiler {
namespace emitc_builders {

Value structPtrMember(OpBuilder builder, Location location, Type type,
                      StringRef memberName, Value operand);

}  // namespace emitc_builders
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCBUILDERS_H_
