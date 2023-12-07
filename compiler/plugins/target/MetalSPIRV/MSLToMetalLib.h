// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_METALSPIRV_MSLTOMETALLIB_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_METALSPIRV_MSLTOMETALLIB_H_

#include "./MetalTargetPlatform.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

namespace mlir::iree_compiler::IREE::HAL {

// Invokes system commands to compile the given |mslCode| into a Metal library
// and returns the library binary code. |fileName| will be used as a hint for
// creating intermediate files.
std::unique_ptr<llvm::MemoryBuffer>
compileMSLToMetalLib(MetalTargetPlatform targetPlatform,
                     llvm::StringRef mslCode, llvm::StringRef fileName);

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_METALSPIRV_MSLTOMETALLIB_H_
