// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_CONVERSION_HALTOVM_CONVERTHALTOVM_H_
#define IREE_COMPILER_DIALECT_HAL_CONVERSION_HALTOVM_CONVERTHALTOVM_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Populates conversion patterns from the HAL dialect to the VM dialect.
void populateHALToVMPatterns(MLIRContext *context, SymbolTable &importSymbols,
                             RewritePatternSet &patterns,
                             TypeConverter &typeConverter);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_CONVERSION_HALTOVM_CONVERTHALTOVM_H_
