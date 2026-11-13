//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_FP8LOWERING_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_FP8LOWERING_H_

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;

namespace iree_compiler {

// Lowers FP8-to-FP32 extensions from i8 storage.
void populateFP8ToNVVMConversionPatterns(LLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns);

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_FP8LOWERING_H_
