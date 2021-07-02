// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_UTILS_CONSTANTENCODING_H_
#define IREE_COMPILER_DIALECT_VM_UTILS_CONSTANTENCODING_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

LogicalResult serializeConstantArray(Location loc, ElementsAttr elementsAttr,
                                     size_t alignment, uint8_t *dst);

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_UTILS_CONSTANTENCODING_H_
