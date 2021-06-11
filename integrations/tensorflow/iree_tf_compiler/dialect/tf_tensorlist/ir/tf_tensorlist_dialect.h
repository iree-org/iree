// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_tf_tensorlist_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_tf_tensorlist_H_

#include "iree_tf_compiler/dialect/tf_tensorlist/ir/tf_tensorlist_ops.h"
#include "iree_tf_compiler/dialect/tf_tensorlist/ir/tf_tensorlist_types.h"
#include "mlir/IR/Dialect.h"

namespace mlir {
namespace iree_integrations {
namespace tf_tensorlist {

class TFTensorListDialect : public Dialect {
 public:
  static StringRef getDialectNamespace() { return "tf_tensorlist"; }
  explicit TFTensorListDialect(MLIRContext *context);
  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type type, DialectAsmPrinter &printer) const override;
};

}  // namespace tf_tensorlist
}  // namespace iree_integrations
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_tf_tensorlist_H_
