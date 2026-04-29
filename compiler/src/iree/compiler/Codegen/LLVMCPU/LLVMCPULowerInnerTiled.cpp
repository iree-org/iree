// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPULOWERINNERTILEDPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {

struct LLVMCPULowerInnerTiledPass final
    : impl::LLVMCPULowerInnerTiledPassBase<LLVMCPULowerInnerTiledPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    // Expects vector-semantics `iree_codegen.inner_tiled` produced by the
    // generic vectorization pass via the `VectorizableOpInterface` external
    // model in `compiler/src/iree/compiler/Codegen/Interfaces/`.
    //
    // (1) Drop the (now-unit) iter domain so the lowered op's iter bounds are
    //     empty, satisfying the precondition of (2).
    IREE::GPU::populateIREEGPUDropUnitDimsPatterns(patterns);
    // (2) Replace the iter-free vector-semantics inner_tiled with the
    //     per-intrinsic ops emitted by `buildUnderlyingOperations`. The
    //     pattern is interface-driven and picks up CPU's DataTiledMMAAttr
    //     transparently.
    IREE::GPU::populateIREEGPULowerInnerTiledPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
