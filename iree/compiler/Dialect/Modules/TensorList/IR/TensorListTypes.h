// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_MODULES_TENSORLIST_IR_TENSORLISTTYPES_H_
#define IREE_COMPILER_DIALECT_MODULES_TENSORLIST_IR_TENSORLISTTYPES_H_

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace TensorList {

class TensorListType
    : public Type::TypeBase<TensorListType, Type, TypeStorage> {
 public:
  using Base::Base;
};

}  // namespace TensorList
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_MODULES_TENSORLIST_IR_TENSORLISTTYPES_H_
