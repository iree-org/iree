// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-test-llvmgpu-query-mma"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_TESTLLVMGPUQUERYMMAPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

static void printMMAVector(SmallVector<IREE::GPU::MMAAttr> &mmaAttrs,
                           const std::string &extraMessage = {}) {
  llvm::outs() << "Printing MMA Collection" << extraMessage
               << ", size: " << mmaAttrs.size() << "\n";
  for (const auto &mma : mmaAttrs) {
    llvm::outs() << mma << " ";
  }
  llvm::outs() << "\n";
}

namespace {

struct TestLLVMGPUQueryMMAPass final
    : impl::TestLLVMGPUQueryMMAPassBase<TestLLVMGPUQueryMMAPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SmallVector<IREE::GPU::MMAAttr> mmaCollecton;
    // Print mma vector before collection.
    printMMAVector(mmaCollecton,
                   " Before querying supported mma instrinsic instructions");
    // Collect mma intrinsic instructions.
    QueryMMAIntrinsics(moduleOp, mmaCollecton);
    // Print mma vector after collection.
    printMMAVector(mmaCollecton,
                   " After querying supported mma instrinsic instructions");
  }
};
} // namespace
} // namespace mlir::iree_compiler
