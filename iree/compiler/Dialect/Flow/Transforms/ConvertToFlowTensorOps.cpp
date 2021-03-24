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

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-convert-to-flow-tensor-ops"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {
/// Convert subtensor operation to flow.tensor.slice if
/// - all offsets apart from the first one are 0
/// - all the sizes apart from the first match the sizes of the source
/// - all strides are 1.
struct SubTensorToTensorSlice : OpRewritePattern<SubTensorOp> {
  using OpRewritePattern<SubTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubTensorOp subTensorOp,
                                PatternRewriter &rewriter) const override {
    if (subTensorOp->getParentOfType<Flow::DispatchWorkgroupsOp>()) {
      return failure();
    }
    SmallVector<OpFoldResult, 4> offsets = subTensorOp.getMixedOffsets();
    SmallVector<OpFoldResult, 4> sizes = subTensorOp.getMixedSizes();
    SmallVector<OpFoldResult, 4> strides = subTensorOp.getMixedStrides();
    ArrayRef<int64_t> srcShape = subTensorOp.getSourceType().getShape();
    for (unsigned dim :
         llvm::seq<unsigned>(0, subTensorOp.getType().getRank())) {
      auto matchVal = [](OpFoldResult valueOrAttr, int64_t val) -> bool {
        auto attr = valueOrAttr.dyn_cast<Attribute>();
        return attr && attr.cast<IntegerAttr>().getInt() == val;
      };
      if ((dim != 0 && (!matchVal(offsets[dim], 0) ||
                        !matchVal(sizes[dim], srcShape[dim]))) ||
          !matchVal(strides[dim], 1)) {
        return failure();
      }
    }
    Location loc = subTensorOp.getLoc();
    auto getAsValues =
        [&](ArrayRef<OpFoldResult> valueOrAttrList) -> SmallVector<Value, 4> {
      return llvm::to_vector<4>(llvm::map_range(
          valueOrAttrList, [&](OpFoldResult valueOrAttr) -> Value {
            if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
              return rewriter.create<ConstantIndexOp>(
                  loc, attr.cast<IntegerAttr>().getInt());
            }
            return valueOrAttr.get<Value>();
          }));
    };
    auto offsetVals = getAsValues(offsets);
    auto sizesVals = getAsValues(sizes);

    Value source = subTensorOp.source();
    SmallVector<Value, 4> sourceSizesVals = sizesVals;
    sourceSizesVals[0] = rewriter.createOrFold<memref::DimOp>(loc, source, 0);
    rewriter.replaceOpWithNewOp<TensorSliceOp>(
        subTensorOp, subTensorOp.getType(), subTensorOp.source(),
        sourceSizesVals, offsetVals, sizesVals, sizesVals);
    return success();
  }
};

/// Converts operations that can map to flow.tensor.* operations.
struct ConvertToFlowTensorOpsPass
    : public PassWrapper<ConvertToFlowTensorOpsPass, OperationPass<FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::StandardOpsDialect, IREE::Flow::FlowDialect>();
  }
  ConvertToFlowTensorOpsPass() = default;
  ConvertToFlowTensorOpsPass(const ConvertToFlowTensorOpsPass &pass) {}
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp->getContext();
    context->allowUnregisteredDialects(true);
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<SubTensorToTensorSlice>(context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createConvertToFlowTensorOpsPass() {
  return std::make_unique<ConvertToFlowTensorOpsPass>();
}

static PassRegistration<ConvertToFlowTensorOpsPass> pass(
    "iree-flow-convert-to-flow-tensor-ops-pass",
    "Convert operations to equivalent flow.tensor.* operations",
    [] { return std::make_unique<ConvertToFlowTensorOpsPass>(); });

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
