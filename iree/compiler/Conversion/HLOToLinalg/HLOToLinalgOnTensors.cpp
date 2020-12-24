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

//===- XLAToLinalgOnTensors.cpp - Pass to convert XLA to Linalg on tensors-===//
//
// Pass to convert from XLA to linalg on tensers. Uses the patterns from
// tensorflow/compiler/mlir/xla/transforms/legalize_to_linalg.cc along with
// some IREE specific patterns.
//
//===----------------------------------------------------------------------===//
#include <memory>

#include "iree/compiler/Conversion/HLOToLinalg/HLOToLinalgOnTensorPasses.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace {

//===----------------------------------------------------------------------===//
// mhlo.torch_index_select conversion patterns.
//===----------------------------------------------------------------------===//

static Value getOutputTensor(OpBuilder &builder, Location loc, Value opResult) {
  ShapedType outputType = opResult.getType().cast<ShapedType>();
  if (outputType.hasStaticShape()) {
    return builder.create<linalg::InitTensorOp>(loc, outputType.getShape(),
                                                outputType.getElementType());
  }
  // Check for tie-shape operations for the result to get the shape of the
  // output.
  SmallVector<Value, 4> dynamicSizes;
  for (Operation *user : opResult.getUsers()) {
    auto tieShapeOp = dyn_cast<Shape::TieShapeOp>(user);
    if (!tieShapeOp) continue;
    auto makeShapeOp =
        tieShapeOp.shape().getDefiningOp<Shape::MakeRankedShapeOp>();
    if (!makeShapeOp) continue;
    dynamicSizes = llvm::to_vector<4>(makeShapeOp.dynamic_dimensions());
    break;
  }
  if (outputType.getNumDynamicDims() != dynamicSizes.size()) return nullptr;
  return builder.create<linalg::InitTensorOp>(
      loc, dynamicSizes, outputType.getShape(), outputType.getElementType());
}

namespace {

/// Converts xla-hlo.torch_index_select op to a linalg.indexed_generic op.
struct TorchIndexSelectOpConversion
    : public OpConversionPattern<mhlo::TorchIndexSelectOp> {
  using OpConversionPattern<mhlo::TorchIndexSelectOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::TorchIndexSelectOp op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const final {
    mhlo::TorchIndexSelectOp::Adaptor adaptor(args);
    int axis = op.dim();
    int batch = op.batch_dims();
    auto indexShapeType = adaptor.index().getType().dyn_cast<ShapedType>();
    int nIndices = indexShapeType.getRank();
    auto inputShapeType = adaptor.input().getType().dyn_cast<ShapedType>();
    if (axis < 0) axis += inputShapeType.getRank();
    if (batch < 0) batch += nIndices;

    Location loc = op.getLoc();
    Value output = op.getResult();
    int rank = output.getType().cast<ShapedType>().getRank();
    SmallVector<AffineMap, 2> indexingMaps;
    SmallVector<AffineExpr, 4> exprs;
    for (int i = 0; i < batch; ++i)
      exprs.push_back(rewriter.getAffineDimExpr(i));
    for (int i = 0, e = nIndices - batch; i < e; ++i)
      exprs.push_back(rewriter.getAffineDimExpr(axis + i));
    indexingMaps.emplace_back(
        AffineMap::get(rank, /*symbolCount=*/0, exprs, rewriter.getContext()));
    indexingMaps.emplace_back(rewriter.getMultiDimIdentityMap(rank));
    SmallVector<StringRef, 3> loopTypes(rank, getParallelIteratorTypeName());
    ShapedType outputType = op.getResult().getType().cast<ShapedType>();
    Value initOp = getOutputTensor(rewriter, loc, op.getResult());
    if (!initOp) return failure();
    auto linalgOp = rewriter.create<linalg::IndexedGenericOp>(
        loc, /*resultTensors=*/ArrayRef<Type>{op.getResult().getType()},
        /*inputs=*/adaptor.index(),
        /*outputBuffers=*/initOp, indexingMaps, loopTypes);

    SmallVector<Type, 4> bodyArgTypes, opResultTypes;
    SmallVector<Value, 2> linalgOpArgs = {adaptor.index()};
    // Add a block to the region.
    auto *region = &linalgOp.region();
    auto *block = rewriter.createBlock(region, region->end());
    bodyArgTypes.append(rank, rewriter.getIndexType());
    for (auto blockArgs : linalgOpArgs) {
      bodyArgTypes.push_back(
          blockArgs.getType().cast<ShapedType>().getElementType());
    }
    block->addArguments(bodyArgTypes);
    block->addArguments(outputType.getElementType());
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(block);

    SmallVector<Value, 4> indices;
    Value castedValue = rewriter.create<IndexCastOp>(
        loc, block->getArgument(rank), rewriter.getIndexType());
    for (int i = 0; i < axis; ++i) {
      indices.push_back(block->getArgument(i));
    }
    indices.push_back(castedValue);
    for (int i = axis + nIndices - batch; i < rank; ++i) {
      indices.push_back(block->getArgument(i));
    }

    Value res =
        rewriter.create<tensor::ExtractOp>(loc, adaptor.input(), indices);
    rewriter.create<linalg::YieldOp>(loc, res);

    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};
}  // namespace

struct ConvertHLOToLinalgOnTensorsPass
    : public PassWrapper<ConvertHLOToLinalgOnTensorsPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, mhlo::MhloDialect, ShapeDialect>();
  }

  void runOnFunction() override {
    OwningRewritePatternList patterns;
    populateHLOToLinalgOnTensorsConversionPatterns(&getContext(), patterns);

    ConversionTarget target(getContext());
    // Don't convert the body of reduction ops.
    target.addDynamicallyLegalDialect<mhlo::MhloDialect>(
        Optional<ConversionTarget::DynamicLegalityCallbackFn>(
            [](Operation *op) {
              auto parentOp = op->getParentRegion()->getParentOp();
              return isa<mhlo::ReduceOp>(parentOp) ||
                     isa<mhlo::ReduceWindowOp>(parentOp);
            }));
    // Let the rest fall through.
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(applyPartialConversion(getFunction(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

void populateHLOToLinalgOnTensorsConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  mhlo::populateHLOToLinalgConversionPattern(context, &patterns);
  patterns.insert<TorchIndexSelectOpConversion>(context);
}

std::unique_ptr<OperationPass<FuncOp>> createHLOToLinalgOnTensorsPass() {
  return std::make_unique<ConvertHLOToLinalgOnTensorsPass>();
}

static PassRegistration<ConvertHLOToLinalgOnTensorsPass> legalize_pass(
    "iree-codegen-hlo-to-linalg-on-tensors",
    "Convert from XLA-HLO ops to Linalg ops on tensors");

}  // namespace iree_compiler
}  // namespace mlir
