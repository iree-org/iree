// Copyright 2019 Google LLC
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

//===- PrepareReductionDispatch.cpp ----------------------------*- C++//-*-===//
//
// Prepare dispatch regions that implement reductions before SPIR-V lowering.
//
//===----------------------------------------------------------------------===//
#include <numeric>

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

namespace {

// TODO(ravishankarm): remove the entire need for this, as it does not support
// dynamic shapes (where we do not know the workload, or know only partial
// dimensions of the workload, at compile-time).
void calculateWorkload(ArrayRef<int64_t> shape,
                       std::array<int32_t, 3> &workload) {
  // Drop the trailing ones from the shape.
  while (shape.size() > 1 && shape.back() == 1) {
    shape = shape.drop_back();
  }
  if (shape.size() <= 3) {
    // Maps to XYZ (possibly with 1's for unused dimensions).
    for (auto dim : enumerate(shape)) {
      workload[shape.size() - 1 - dim.index()] = dim.value();
    }
  } else {
    // Need to flatten the shape to fit XYZ. For now we just squash from
    // LHS.
    auto zRange = shape.drop_back(2);
    workload[2] = std::accumulate(zRange.begin(), zRange.end(), 1,
                                  std::multiplies<int32_t>());
    workload[1] = shape[shape.size() - 2];
    workload[0] = shape.back();
  }
}

/// The entry function of the reduction dispatches is empty. This pattern fills
/// the body with an iree.load_input of the reduction input followed by an
/// iree.store_reduce for the later lowering passes to generate the appropriate
/// SPIR-V code.
struct AddReductionEntryFnBody : public OpRewritePattern<FuncOp> {
  using OpRewritePattern<FuncOp>::OpRewritePattern;
  PatternMatchResult matchAndRewrite(FuncOp fn,
                                     PatternRewriter &rewriter) const override;
};

/// Pass to prepare the reduction dispatch entry functions.
struct PrepareReductionDispatchPass
    : public OperationPass<PrepareReductionDispatchPass> {
  void runOnOperation() override;
};

}  // namespace

PatternMatchResult AddReductionEntryFnBody::matchAndRewrite(
    FuncOp fn, PatternRewriter &rewriter) const {
  if (!fn.getAttr("iree.executable.reduction") || !fn.empty()) {
    return matchFailure();
  }
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(fn.addEntryBlock());
  if (fn.getNumArguments() != 3) {
    return matchFailure();
  }
  rewriter.startRootUpdate(fn);

  auto src = fn.getArgument(0);
  auto dst = fn.getArgument(2);
  fn.setArgAttr(2, "iree.executable.reduction.output",
                UnitAttr::get(fn.getContext()));
  auto applyFn =
      fn.getAttrOfType<FlatSymbolRefAttr>("iree.executable.reduction.apply");
  auto srcType = src.getType().cast<MemRefType>();
  auto loc = fn.getLoc();
  auto loadInputOp = rewriter.create<IREE::LoadInputOp>(
      loc, RankedTensorType::get(srcType.getShape(), srcType.getElementType()),
      src);
  rewriter.create<IREE::StoreReduceOp>(loc, loadInputOp.getResult(), dst,
                                       applyFn);
  rewriter.create<mlir::ReturnOp>(fn.getLoc());

  // Finally update the workload size to be determined by the size of the input.
  auto convertFn = [&](std::array<int32_t, 3> value,
                       SmallVectorImpl<APInt> &vector) {
    vector.reserve(3);
    for (auto val : value) {
      vector.emplace_back(32, static_cast<uint64_t>(val), true);
    }
  };

  // TODO(ravishankarm): The workgroup size should have been set already, but is
  // not. So setting this here.
  SmallVector<APInt, 3> workgroupSizeAPInt;
  convertFn({32, 1, 1}, workgroupSizeAPInt);
  fn.setAttr(
      "iree.executable.workgroup_size",
      DenseElementsAttr::get(
          RankedTensorType::get(3, IntegerType::get(32, rewriter.getContext())),
          workgroupSizeAPInt)
          .cast<DenseIntElementsAttr>());

  rewriter.finalizeRootUpdate(fn);
  return matchSuccess();
}

void PrepareReductionDispatchPass::runOnOperation() {
  OwningRewritePatternList patterns;
  patterns.insert<AddReductionEntryFnBody>(&getContext());
  Operation *op = getOperation();
  applyPatternsGreedily(op->getRegions(), patterns);
}

static PassRegistration<PrepareReductionDispatchPass> pass(
    "iree-spirv-prepare-reduction-dispatch",
    "Prepare the entry function for generation of reduction dispatches");

std::unique_ptr<Pass> createPrepareReductionDispatchPass() {
  return std::make_unique<PrepareReductionDispatchPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
