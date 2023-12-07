// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_STANDARDTOVM_CONVERTSTANDARDTOVM_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_STANDARDTOVM_CONVERTSTANDARDTOVM_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

// Appends standard dialect to vm dialect patterns to the given pattern list.
void populateStandardToVMPatterns(MLIRContext *context,
                                  TypeConverter &typeConverter,
                                  RewritePatternSet &patterns);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_VM_CONVERSION_STANDARDTOVM_CONVERTSTANDARDTOVM_H_
