// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef THIRD_PARTY_TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TENSORLIST_TYPES_H_
#define THIRD_PARTY_TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TENSORLIST_TYPES_H_

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"

namespace mlir {
namespace iree_integrations {
namespace tf_tensorlist {

class TensorListType
    : public Type::TypeBase<TensorListType, Type, TypeStorage> {
 public:
  using Base::Base;
};

}  // namespace tf_tensorlist
}  // namespace iree_integrations
}  // namespace mlir

#endif  // THIRD_PARTY_TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TENSORLIST_TYPES_H_
