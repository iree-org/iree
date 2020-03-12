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

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VMLA {

namespace {

// Unrolls a multi-dimensional xla_hlo.reduce op into one xla_hlo.reduce op per
// dimension. The XLA operation semantics state that this is a valid
// transformation.
void unrollReduceOp(xla_hlo::ReduceOp reduceOp) {
  // Create one op per dimension being reduced.
  // We'll do this by chaining the original input through with the temporary
  // reduction results. The results we end up with will be the originally
  // requested shape and we can just substitute them.
  SmallVector<int64_t, 4> sortedDimensions{
      reduceOp.dimensions().getValues<int64_t>()};
  llvm::sort(sortedDimensions,
             [](int64_t a, int64_t b) { return (a - b) > 0; });

  // Insert at the same place as the original op.
  OpBuilder builder(reduceOp);
  SmallVector<Value, 4> temps{reduceOp.operands()};
  for (int64_t dimension : sortedDimensions) {
    // Create the new reduction using the results of the previous operation.
    auto singleAttrType =
        RankedTensorType::get({1}, builder.getIntegerType(64));
    auto singleReduceOp = builder.create<xla_hlo::ReduceOp>(
        reduceOp.getLoc(), temps, reduceOp.init_values(),
        DenseIntElementsAttr::get(singleAttrType, {dimension}));
    BlockAndValueMapping mapping;
    reduceOp.body().cloneInto(&singleReduceOp.body(), mapping);
    temps = singleReduceOp.getResults();
  }

  // Replace uses of the existing results with the new results.
  reduceOp.replaceAllUsesWith(temps);

  // Erase original op.
  reduceOp.erase();
}

// Unrolls a multi-dimensional xla_hlo.reduce_window op into one
// xla_hlo.reduce_window op per dimension. The XLA operation semantics state
// that this is a valid transformation.
void unrollReduceWindowOp(xla_hlo::ReduceWindowOp reduceOp) {
  // Create one dispatch per dimension being reduced.
  // We'll do this by chaining the original input through with the temporary
  // reduction results. The results we end up with will be the originally
  // requested shape and we can just substitute them.
  using WindowTuple = std::tuple<int64_t, int64_t, int64_t, int64_t>;
  auto windowDimensions = reduceOp.window_dimensions();
  auto windowStrides = reduceOp.window_stridesAttr();
  auto baseDilations = reduceOp.base_dilationsAttr();
  auto windowDilations = reduceOp.window_dilationsAttr();
  SmallVector<WindowTuple, 4> sortedWindowAttrs;
  for (uint32_t i = 0; i < windowDimensions.getNumElements(); ++i) {
    int64_t windowDimension = windowDimensions.getValue<int64_t>({i});
    int64_t windowStride = windowStrides.getValue<int64_t>({i});
    int64_t baseDilation = baseDilations.getValue<int64_t>({i});
    int64_t windowDilation = windowDilations.getValue<int64_t>({i});
    sortedWindowAttrs.push_back(WindowTuple(windowDimension, windowStride,
                                            baseDilation, windowDilation));
  }
  llvm::sort(sortedWindowAttrs, [](WindowTuple a, WindowTuple b) {
    return (std::get<0>(a) - std::get<0>(b)) > 0;
  });

  // Insert at the same place as the original op.
  OpBuilder builder(reduceOp);
  Value temp = reduceOp.operand();
  for (auto windowAttrs : llvm::enumerate(sortedWindowAttrs)) {
    int64_t windowDimension = std::get<0>(windowAttrs.value());
    int64_t windowStride = std::get<1>(windowAttrs.value());
    int64_t baseDilation = std::get<2>(windowAttrs.value());
    int64_t windowDilation = std::get<3>(windowAttrs.value());

    // Compute the new result.
    // NOTE: this should not be required and instead be on the HLO op.
    auto tempType = temp.getType().cast<RankedTensorType>();
    SmallVector<int64_t, 4> newShape;
    for (int64_t i = 0; i < tempType.getRank(); ++i) {
      if (i != windowDimension) {
        newShape.push_back(tempType.getDimSize(i));
      }
    }
    auto resultType =
        RankedTensorType::get(newShape, tempType.getElementType());

    // Create the new reduction using the results of the previous operation.
    auto singleAttrType =
        RankedTensorType::get({1}, builder.getIntegerType(64));
    auto singleReduceOp = builder.create<xla_hlo::ReduceWindowOp>(
        reduceOp.getLoc(), resultType, temp, reduceOp.init_value(),
        DenseIntElementsAttr::get(
            singleAttrType,
            {builder.getI64IntegerAttr(windowDimension).cast<Attribute>()}),
        DenseIntElementsAttr::get(
            singleAttrType,
            {builder.getI64IntegerAttr(windowStride).cast<Attribute>()}),
        DenseIntElementsAttr::get(
            singleAttrType,
            {builder.getI64IntegerAttr(baseDilation).cast<Attribute>()}),
        DenseIntElementsAttr::get(
            singleAttrType,
            {builder.getI64IntegerAttr(windowDilation).cast<Attribute>()}),
        reduceOp.paddingAttr());
    BlockAndValueMapping mapping;
    reduceOp.body().cloneInto(&singleReduceOp.body(), mapping);
    temp = singleReduceOp.getResult();
  }

  // Replace uses of the existing results with the new results.
  reduceOp.getResult().replaceAllUsesWith(temp);

  // Erase original op.
  reduceOp.erase();
}

}  // namespace

class UnrollReductionsPass : public FunctionPass<UnrollReductionsPass> {
 public:
  void runOnFunction() override {
    for (auto &block : getFunction()) {
      auto reduceOps = llvm::to_vector<4>(block.getOps<xla_hlo::ReduceOp>());
      for (auto reduceOp : reduceOps) {
        if (reduceOp.dimensions().getNumElements() > 1) {
          unrollReduceOp(reduceOp);
        }
      }

      auto reduceWindowOps =
          llvm::to_vector<4>(block.getOps<xla_hlo::ReduceWindowOp>());
      for (auto reduceOp : reduceWindowOps) {
        if (reduceOp.window_dimensions().getNumElements() > 1) {
          unrollReduceWindowOp(reduceOp);
        }
      }
    }
  }
};

std::unique_ptr<OpPassBase<FuncOp>> createUnrollReductionsPass() {
  return std::make_unique<UnrollReductionsPass>();
}

static PassRegistration<UnrollReductionsPass> pass(
    "iree-vmla-unroll-reductions",
    "Unrolls multi-dimensional reductions to one reduction per dimension.");

}  // namespace VMLA
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
