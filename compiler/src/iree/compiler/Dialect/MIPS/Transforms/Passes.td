// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_MIPS_TRANSFORMS_PASSES
#define IREE_COMPILER_DIALECT_MIPS_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// LowerMIPSToFuncCallPass
//
// Converts memref-form mips.matmul operations (produced after one-shot
// bufferization in the LLVMCPU codegen pipeline) to:
//
//   func.func private @my_matmul_kernel(...) attributes {llvm.bareptr = true}
//   func.call @my_matmul_kernel(A_base, A_off, A_s0, A_s1,
//                               B_base, B_off, B_s0, B_s1,
//                               C_base, C_off, C_s0, C_s1,
//                               M, N, K)
//
// The pass uses memref.extract_strided_metadata to decompose each memref into
// a (base_pointer, offset, strides...) tuple matching the C ABI of the kernel.
//===----------------------------------------------------------------------===//

def LowerMIPSToFuncCallPass :
    Pass<"iree-mips-lower-to-func-call", "ModuleOp"> {
  let summary = "Lower mips.matmul (memref form) to func.call @my_matmul_kernel";
  let description = [{
    Walks all mips.matmul ops in the module and replaces each one with a call
    to the external C kernel `my_matmul_kernel`.  The memref operands are
    decomposed via `memref.extract_strided_metadata` into base-pointer + offset
    + stride arguments, matching the ABI declared in my_matmul_kernel.h.

    The pass creates a `func.func private @my_matmul_kernel` declaration with
    `{llvm.bareptr = true}` so that the LLVM backend passes bare float* pointers
    rather than MLIR memref descriptor structs.

    This pass runs after one-shot bufferization in the LLVMCPU codegen pipeline.
  }];
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

#endif // IREE_COMPILER_DIALECT_MIPS_TRANSFORMS_PASSES
