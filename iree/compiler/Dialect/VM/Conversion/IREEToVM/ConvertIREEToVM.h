// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_IREETOVM_CONVERTIREETOVM_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_IREETOVM_CONVERTIREETOVM_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Appends IREE special hint ops to VM dialect patterns.
void populateIREEToVMPatterns(MLIRContext *context,
                              TypeConverter &typeConverter,
                              OwningRewritePatternList &patterns);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_IREETOVM_CONVERTIREETOVM_H_
