// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE ops for working with buffers and buffer views.
// These are used by common transforms between the sequencer and interpreter and
// allow us to share some of the common lowering passes from other dialects.

#ifndef INTEGRATIONS_TENSORFLOW_COMPILER_DIALECT_TFSTRINGS_IR_TYPES_H_
#define INTEGRATIONS_TENSORFLOW_COMPILER_DIALECT_TFSTRINGS_IR_TYPES_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace iree_integrations {
namespace tf_strings {

class TFStringsType : public Type {
 public:
  using Type::Type;

  static bool classof(Type type);
};

class StringType
    : public Type::TypeBase<StringType, TFStringsType, TypeStorage> {
 public:
  using Base::Base;
};

}  // namespace tf_strings
}  // namespace iree_integrations
}  // namespace mlir

#endif  // INTEGRATIONS_TENSORFLOW_COMPILER_DIALECT_TFSTRINGS_IR_TYPES_H_
