// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_ARITHTOVM_PATTERNS_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_ARITHTOVM_PATTERNS_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

// Appends arith dialect to vm dialect patterns to the given pattern list.
void populateArithToVMPatterns(MLIRContext *context,
                               TypeConverter &typeConverter,
                               RewritePatternSet &patterns);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_VM_CONVERSION_ARITHTOVM_PATTERNS_H_
