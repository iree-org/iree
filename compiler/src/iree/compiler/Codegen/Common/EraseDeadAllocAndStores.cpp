// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ERASEDEADALLOCANDSTORESPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

class EraseDeadAllocAndStoresPass
    : public impl::EraseDeadAllocAndStoresPassBase<
          EraseDeadAllocAndStoresPass> {
public:
  using impl::EraseDeadAllocAndStoresPassBase<
      EraseDeadAllocAndStoresPass>::EraseDeadAllocAndStoresPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void EraseDeadAllocAndStoresPass::runOnOperation() {
  auto funcOp = getOperation();
  IRRewriter rewriter(&getContext());
  memref::eraseDeadAllocAndStores(rewriter, funcOp);
}

} // namespace
} // namespace mlir::iree_compiler
