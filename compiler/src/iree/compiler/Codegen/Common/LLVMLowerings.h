// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_LLVMLOWERINGS_H_
#define IREE_COMPILER_CODEGEN_COMMON_LLVMLOWERINGS_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler {

/// Populate patterns to lower iree_codegen ops to LLVM.
void populateConvertIREECodegenToLLVMPatterns(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_LLVMLOWERINGS_H_
