// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_MODULES_IO_PARAMETERS_IR_IOPARAMETERSDIALECT_H_
#define IREE_COMPILER_MODULES_IO_PARAMETERS_IR_IOPARAMETERSDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::iree_compiler::IREE::IO::Parameters {

class IOParametersDialect : public Dialect {
public:
  explicit IOParametersDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "io_parameters"; }
};

} // namespace mlir::iree_compiler::IREE::IO::Parameters

#endif // IREE_COMPILER_MODULES_IO_PARAMETERS_IR_IOPARAMETERSDIALECT_H_
