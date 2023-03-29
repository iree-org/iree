// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_COMMON_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_COMMON_H_

#include "mlir/Conversion/LLVMCommon/Pattern.h"

namespace mlir {
namespace gpu {
enum class AddressSpace : uint32_t;
}
namespace iree_compiler {

void populateLLVMConversionPatterns(MLIRContext *context,
                                    RewritePatternSet &patterns,
                                    LLVMTypeConverter &converter);

void populateScalarizeMathOps(RewritePatternSet &patterns);

/// Lower hal.interface ops to the equivalent gpu ops.
void populateLowerHALInterfaceOp(RewritePatternSet &patterns);

/// Add patterns to convert AllocOp of shared memory to a global variable.
void populateConvertSharedMemoryAllocOps(RewritePatternSet &patterns);

void ConvertToDynamicSharedMemory(ModuleOp moduleOp);

using MemorySpaceMapping =
    std::function<unsigned(gpu::AddressSpace gpuAddressSpace)>;
void populateGpuMemorySpaceAttributeConversions(
    TypeConverter &typeConverter, const MemorySpaceMapping &mapping);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_LLVMGPU_COMMON_H_
