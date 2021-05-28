// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CONVERSION_VECTORTOLLVM_PASSES_H_
#define IREE_COMPILER_CONVERSION_VECTORTOLLVM_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

class LLVMTypeConverter;

namespace iree_compiler {

/// A pass that converts vector dialect operations to inline assembly
std::unique_ptr<FunctionPass> createVectorToAArch64InlineAssemblyPass();

/// Populates `patterns` to convert vector.contract op to a sequence
/// of AArch64 inline assembly operations.
void populateVectorContractToAArch64InlineAsm(
    OwningRewritePatternList &patterns, MLIRContext *context);

}  // namespace iree_compiler
}  // namespace mlir

#endif
