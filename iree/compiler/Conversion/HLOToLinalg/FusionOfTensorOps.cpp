// Copyright 2020 Google LLC
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

//===--- FusionOfTensorsOps.cpp - Pass to fuse operations on tensors-------===//
//
// Pass to fuse operations on tensors after conversion to Linalg. Uses the
// patterns from MLIR for fusion linalg operations on tensors, and a few
// patterns to fuse these with IREE specific operations.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/HLOToLinalg/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace iree_compiler {

namespace {
/// Pattern to fuse hal.interface.load.tensor -> linalg.tensor_reshape
struct FuseWithHALInterfaceLoadTensor
    : public OpRewritePattern<linalg::TensorReshapeOp> {
  using OpRewritePattern<linalg::TensorReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto loadOp =
        reshapeOp.src().getDefiningOp<IREE::HAL::InterfaceLoadTensorOp>();
    if (!loadOp) return failure();
    rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceLoadTensorOp>(
        reshapeOp, reshapeOp.getResultType(), loadOp.offset(),
        loadOp.getAttrs());
    return success();
  }
};

/// Pattern to fuse linalg.tensor_reshape -> hal.interface.store.tensor
struct FuseWithHALInterfaceStoreTensor
    : public OpRewritePattern<IREE::HAL::InterfaceStoreTensorOp> {
  using OpRewritePattern<IREE::HAL::InterfaceStoreTensorOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::HAL::InterfaceStoreTensorOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto reshapeOp = storeOp.operand().getDefiningOp<linalg::TensorReshapeOp>();
    if (!reshapeOp) return failure();
    SmallVector<Value, 2> operands = {reshapeOp.src(), storeOp.offset()};
    rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceStoreTensorOp>(
        storeOp, ArrayRef<Type>(), operands, storeOp.getAttrs());
    return success();
  }
};

/// Pass to fuse linalg on tensor operations as well as fusion of hal.interface*
/// operations with linalg.tensor_reshape operation.
struct FusionOfTensorOpsPass
    : public PassWrapper<FusionOfTensorOpsPass, OperationPass<>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, IREE::HAL::HALDialect>();
  }

  void runOnOperation() override {
    OwningRewritePatternList fusionPatterns, interfacePatterns;
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    interfacePatterns.insert<FuseWithHALInterfaceLoadTensor,
                             FuseWithHALInterfaceStoreTensor>(context);
    applyPatternsAndFoldGreedily(op->getRegions(), interfacePatterns);

    populateLinalgTensorOpsFusionPatterns(context, fusionPatterns);
    applyPatternsAndFoldGreedily(op->getRegions(), fusionPatterns);

    applyPatternsAndFoldGreedily(op->getRegions(), interfacePatterns);
  }
};
}  // namespace

std::unique_ptr<Pass> createFusionOfTensorOpsPass() {
  return std::make_unique<FusionOfTensorOpsPass>();
}

static PassRegistration<FusionOfTensorOpsPass> pass(
    "iree-codegen-fusion-of-tensor-ops", "Fuse operations on tensors");

}  // namespace iree_compiler
}  // namespace mlir
