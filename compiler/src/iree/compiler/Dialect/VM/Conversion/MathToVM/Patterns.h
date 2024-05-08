// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_MATHTOVM_CONVERTMATHTOVM_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_MATHTOVM_CONVERTMATHTOVM_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

// Appends math dialect to vm dialect patterns to the given pattern list.
void populateMathToVMPatterns(MLIRContext *context,
                              TypeConverter &typeConverter,
                              RewritePatternSet &patterns);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_VM_CONVERSION_MATHTOVM_CONVERTMATHTOVM_H_
