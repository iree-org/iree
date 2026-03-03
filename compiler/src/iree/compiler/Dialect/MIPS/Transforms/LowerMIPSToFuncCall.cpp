// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// mips.matmul is a tensor-only op that is eliminated entirely during
// One-Shot Bufferize: the BufferizableOpInterface implementation in
// MIPSBufferizableOpInterface.cpp emits func.call @my_matmul_kernel directly.
//
// This pass is therefore a no-op and exists only for registration purposes
// (so that --iree-mips-lower-to-func-call can be specified on the command line
// without error, and so that any pipeline that references it still compiles).

#include "iree/compiler/Dialect/MIPS/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // IWYU pragma: keep
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::MIPS {

#define GEN_PASS_DEF_LOWERMIPSTOFUNCCALLPASS
#include "iree/compiler/Dialect/MIPS/Transforms/Passes.h.inc"

namespace {

struct LowerMIPSToFuncCallPass
    : impl::LowerMIPSToFuncCallPassBase<LowerMIPSToFuncCallPass> {
  void runOnOperation() override {
    // mips.matmul is eliminated during One-Shot Bufferize (see
    // MIPSBufferizableOpInterface.cpp). No work to do here.
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::MIPS
