// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {
#include "iree/compiler/Codegen/Common/FoldTensorExtractOp.cpp.inc"
} // namespace

namespace {
/// Upstream canonicalization passes fold
///
/// (load (tensor_to_memref $value), $indices) to
///
/// (tensor_extract $value, $indices)
///
/// In general this is ill-defined because it ignores potential writes to the
/// result of the tensor_to_memref before the load. The assumption is that there
/// shouldn't be any writes using the result of tensor_to_memref. This is almost
/// impossible to enforce/verify. Nevertheless, in IREE we use
/// `tensor_to_memref` during bufferization of `std.constant` assuming that
/// downstream passes can handle the lowering of the `std.constant`.
///
/// On LLVM side, the `std.constant` is handled by the
/// `TensorConstantBufferizePass`, which creates a global object of `memref`
/// type. To get the tensor back you get a to_tensor. If the above
/// canonicalization pattern didnt exist, then a to_tensor would not be
/// needed.
///
/// This pass is specifically undoing the canonicalization by folding
///
/// (tensor_extract (to_tensor (get_global_memref:$value), $indices) to
///
/// (load $value, $indices)
///
/// In theory this could live upstream, but given that there is disagreement
/// about the validity of `tensor_to_memref` usage/canonicalizations, keeping
/// this pattern here.
class FoldTensorExtractOpPass
    : public FoldTensorExtractOpBase<FoldTensorExtractOpPass> {
  void runOnOperation() override;
};
} // namespace

void FoldTensorExtractOpPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<>> createFoldTensorExtractOpPass() {
  return std::make_unique<FoldTensorExtractOpPass>();
}

} // namespace mlir::iree_compiler
