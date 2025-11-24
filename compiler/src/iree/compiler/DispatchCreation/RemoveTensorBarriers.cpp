// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_REMOVETENSORBARRIERSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

struct RemoveTensorBarriersPass final
    : public impl::RemoveTensorBarriersPassBase<RemoveTensorBarriersPass> {
  using Base::Base;

  void runOnOperation() override {
    auto funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    funcOp.walk([&](IREE::TensorExt::ComputeBarrierOp op) {
      rewriter.replaceOp(op, op.getValue());
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
