// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_MODULES_CHECK_CONVERSION_CONVERSION_PATTERNS_H_
#define IREE_COMPILER_DIALECT_MODULES_CHECK_CONVERSION_CONVERSION_PATTERNS_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Check {

// Populates conversion patterns from the Check dialect to the VM dialect.
void populateCheckToVMPatterns(MLIRContext *context, SymbolTable &importSymbols,
                               OwningRewritePatternList &patterns,
                               TypeConverter &typeConverter);

// Populates conversion patterns from the Check dialect to the HAL dialect.
// Mostly lowers tensors to buffer views.
void populateCheckToHALPatterns(MLIRContext *context,
                                OwningRewritePatternList &patterns,
                                TypeConverter &typeConverter);

}  // namespace Check
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_MODULES_CHECK_CONVERSION_CONVERSION_PATTERNS_H_
