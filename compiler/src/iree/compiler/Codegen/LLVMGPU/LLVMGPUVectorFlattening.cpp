// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-llvmgpu-vector-flattening"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUVECTORFLATTENINGPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

struct LLVMGPUVectorFlatteningPass final
    : impl::LLVMGPUVectorFlatteningPassBase<LLVMGPUVectorFlatteningPass> {

  void runOnOperation() override {}
};

} // namespace mlir::iree_compiler
