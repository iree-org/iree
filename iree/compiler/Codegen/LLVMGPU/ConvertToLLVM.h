// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_COMMON_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_COMMON_H_

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"

namespace mlir {
namespace iree_compiler {

void populateLLVMConversionPatterns(MLIRContext *context,
                                    OwningRewritePatternList &patterns,
                                    LLVMTypeConverter &converter, bool useROCM);

void populateScalarizeMathOps(RewritePatternSet &patterns);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_LLVMGPU_COMMON_H_
