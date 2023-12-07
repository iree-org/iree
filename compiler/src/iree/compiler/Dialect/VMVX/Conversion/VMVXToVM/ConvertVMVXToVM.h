// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VMVX_CONVERSION_VMVXTOVM_CONVERTVMVXTOVM_H_
#define IREE_COMPILER_DIALECT_VMVX_CONVERSION_VMVXTOVM_CONVERTVMVXTOVM_H_

#include "iree/compiler/Dialect/VMVX/IR/VMVXOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

// Populates conversion patterns from the VMVX dialect to the VM dialect.
void populateVMVXToVMPatterns(MLIRContext *context,
                              ConversionTarget &conversionTarget,
                              TypeConverter &typeConverter,
                              SymbolTable &importSymbols,
                              RewritePatternSet &patterns);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_VMVX_CONVERSION_VMVXTOVM_CONVERTVMVXTOVM_H_
