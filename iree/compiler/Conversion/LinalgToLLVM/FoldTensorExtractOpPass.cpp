// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {
#include "iree/compiler/Conversion/LinalgToLLVM/FoldTensorExtractOp.cpp.inc"
}

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
/// type. To get the tensor back you get a tensor.load. If the above
/// canonicalization pattern didnt exist, then a tensor.load would not be
/// needed.
///
/// This pass is specifically undoing the canonicalization by folding
///
/// (tensor_extract (tensor_load (get_global_memref:$value), $indices) to
///
/// (load $value, $indices)
///
/// In theory this could live upstream, but given that there is disagreement
/// about the validity of `tensor_to_memref` usage/canonicalizations, keeping
/// this pattern here.
class FoldTensorExtractOpPass
    : public PassWrapper<FoldTensorExtractOpPass, OperationPass<>> {
  void runOnOperation() override;
};
}  // namespace

void FoldTensorExtractOpPass::runOnOperation() {
  MLIRContext *context = &getContext();
  OwningRewritePatternList patterns;
  populateWithGenerated(context, patterns);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<>> createFoldTensorExtractOpPass() {
  return std::make_unique<FoldTensorExtractOpPass>();
}

static PassRegistration<FoldTensorExtractOpPass> pass(
    "iree-codegen-fold-tensor-extract-op",
    "Fold `tensor.extract` operations prior to lowering to LLVM",
    [] { return std::make_unique<FoldTensorExtractOpPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
