// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_HOISTSTATICALLYBOUNDALLOCATIONSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct HoistStaticallyBoundAllocationsPass
    : impl::HoistStaticallyBoundAllocationsPassBase<
          HoistStaticallyBoundAllocationsPass> {
  using impl::HoistStaticallyBoundAllocationsPassBase<
      HoistStaticallyBoundAllocationsPass>::
      HoistStaticallyBoundAllocationsPassBase;
  void runOnOperation() override;
};

} // namespace

void HoistStaticallyBoundAllocationsPass::runOnOperation() {
  auto funcOp = getOperation();
  IRRewriter rewriter(funcOp->getContext());

  std::optional<VscaleRange> vscaleRange;
  if (this->vscaleMax != 0 && this->vscaleMin <= this->vscaleMax)
    vscaleRange = {this->vscaleMin, this->vscaleMax};

  hoistStaticallyBoundAllocationsInFunc<memref::AllocaOp>(rewriter, funcOp,
                                                          vscaleRange);
  hoistStaticallyBoundAllocationsInFunc<memref::AllocOp>(rewriter, funcOp,
                                                         vscaleRange);
}

} // namespace mlir::iree_compiler
