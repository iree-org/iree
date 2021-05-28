// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/dialect/tf_strings/ir/dialect.h"

#include "iree_tf_compiler/dialect/tf_strings/ir/ops.h"
#include "iree_tf_compiler/dialect/tf_strings/ir/types.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace iree_integrations {
namespace tf_strings {

#include "iree_tf_compiler/dialect/tf_strings/ir/op_interface.cpp.inc"

TFStringsDialect::TFStringsDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<TFStringsDialect>()) {
  addTypes<StringType>();

#define GET_OP_LIST
  addOperations<
#include "iree_tf_compiler/dialect/tf_strings/ir/ops.cpp.inc"
      >();
}

Type TFStringsDialect::parseType(DialectAsmParser& parser) const {
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  llvm::StringRef spec = parser.getFullSymbolSpec();
  if (spec == "string") {
    return StringType::get(getContext());
  }
  emitError(loc, "unknown TFStrings type: ") << spec;
  return Type();
}

void TFStringsDialect::printType(Type type, DialectAsmPrinter& os) const {
  if (type.isa<tf_strings::StringType>())
    os << "string";
  else
    llvm_unreachable("unhandled string type");
}

bool TFStringsType::classof(Type type) {
  return llvm::isa<TFStringsDialect>(type.getDialect());
}

}  // namespace tf_strings
}  // namespace iree_integrations
}  // namespace mlir
