// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_MODULES_IO_STREAM_IR_IOSTREAMDIALECT_H_
#define IREE_COMPILER_MODULES_IO_STREAM_IR_IOSTREAMDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::iree_compiler::IREE::IO::Stream {

class IOStreamDialect : public Dialect {
public:
  explicit IOStreamDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "io_stream"; }

  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type type, DialectAsmPrinter &os) const override;
};

} // namespace mlir::iree_compiler::IREE::IO::Stream

#endif // IREE_COMPILER_MODULES_IO_STREAM_IR_IOSTREAMDIALECT_H_
