// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_PCF_TRANSFORMS_CONVERSIONDIALECTINTERFACE_H_
#define IREE_COMPILER_CODEGEN_DIALECT_PCF_TRANSFORMS_CONVERSIONDIALECTINTERFACE_H_

#include "mlir/IR/DialectInterface.h"

namespace mlir::iree_compiler {

// An interface for dialects to expose conversion functionality out of PCF.
class PCFConversionDialectInterface
    : public DialectInterface::Base<PCFConversionDialectInterface> {
public:
  PCFConversionDialectInterface(Dialect *dialect) : Base(dialect) {}
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_DIALECT_PCF_TRANSFORMS_CONVERSIONDIALECTINTERFACE_H_
