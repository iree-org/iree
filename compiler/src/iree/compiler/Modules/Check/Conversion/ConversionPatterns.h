// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_MODULES_CHECK_CONVERSION_CONVERSION_PATTERNS_H_
#define IREE_COMPILER_MODULES_CHECK_CONVERSION_CONVERSION_PATTERNS_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::IREE::Check {

// Populates conversion patterns from the Check dialect to the VM dialect.
void populateCheckToVMPatterns(MLIRContext *context, SymbolTable &importSymbols,
                               RewritePatternSet &patterns,
                               TypeConverter &typeConverter);

// Populates conversion patterns from the Check dialect to the HAL dialect.
// Mostly lowers tensors to buffer views.
void populateCheckToHALPatterns(MLIRContext *context,
                                RewritePatternSet &patterns,
                                TypeConverter &typeConverter);

} // namespace mlir::iree_compiler::IREE::Check

#endif // IREE_COMPILER_MODULES_CHECK_CONVERSION_CONVERSION_PATTERNS_H_
