// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-llvmcpu-forall-to-for"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUFORALLTOFORPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {

struct LLVMCPUForallToForPass
    : impl::LLVMCPUForallToForPassBase<LLVMCPUForallToForPass> {
  using impl::LLVMCPUForallToForPassBase<
      LLVMCPUForallToForPass>::LLVMCPUForallToForPassBase;
  void runOnOperation() override {
    auto funcOp = getOperation();

    LLVM_DEBUG({
      llvm::dbgs() << "TACO: Running LLVMCPUForallToForPass on function: "
                   << funcOp.getName() << "\n";
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler
