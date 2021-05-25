// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/dialect/tf_tensorlist/ir/tf_tensorlist_dialect.h"

#include "mlir/IR/DialectImplementation.h"

namespace mlir {
namespace iree_integrations {
namespace tf_tensorlist {

//===----------------------------------------------------------------------===//
// TFTensorListDialect Dialect
//===----------------------------------------------------------------------===//

TFTensorListDialect::TFTensorListDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context,
              TypeID::get<TFTensorListDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "iree_tf_compiler/dialect/tf_tensorlist/ir/tf_tensorlist_ops.cc.inc"
      >();
  addTypes<TensorListType>();
}

Type TFTensorListDialect::parseType(DialectAsmParser &parser) const {
  StringRef type_name;
  if (parser.parseKeyword(&type_name)) return nullptr;
  if (type_name == "list") {
    return TensorListType::get(getContext());
  }
  parser.emitError(parser.getCurrentLocation(),
                   "unknown type in `tf_tensorlist` dialect");
  return nullptr;
}

void TFTensorListDialect::printType(Type type,
                                    DialectAsmPrinter &printer) const {
  printer << "list";
}

}  // namespace tf_tensorlist
}  // namespace iree_integrations
}  // namespace mlir
