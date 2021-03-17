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
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
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
    auto indexShapeType = adaptor.index().getType().cast<ShapedType>();
    int nIndices = indexShapeType.getRank();
    auto inputShapeType = adaptor.input().getType().cast<ShapedType>();
    if (axis < 0) axis += inputShapeType.getRank();
    if (batch < 0) batch += nIndices;

    Location loc = op.getLoc();
    ShapedType resultType = op.getResult().getType().cast<ShapedType>();
    int rank = resultType.getRank();

    SmallVector<AffineMap, 2> indexingMaps;
    SmallVector<AffineExpr, 4> exprs;
    for (int i = 0; i < batch; ++i) {
      exprs.push_back(rewriter.getAffineDimExpr(i));
    }
    for (int i = 0, e = nIndices - batch; i < e; ++i) {
      exprs.push_back(rewriter.getAffineDimExpr(axis + i));
    }
    indexingMaps.emplace_back(
        AffineMap::get(rank, /*symbolCount=*/0, exprs, rewriter.getContext()));
    indexingMaps.emplace_back(rewriter.getMultiDimIdentityMap(rank));

    SmallVector<StringRef, 3> loopTypes(rank, getParallelIteratorTypeName());

    // The output shape is
    //   `params[:axis] + indices[batch_dims:] + params[axis + 1:]`
    SmallVector<Value, 4> dynSizes;
    for (int i = 0; i < rank; ++i) {
      if (!resultType.isDynamicDim(i)) continue;
      if (i < axis) {
        dynSizes.push_back(
            rewriter.create<memref::DimOp>(loc, adaptor.input(), i));
      } else if (i < (axis + nIndices - batch)) {
        int idx = i - axis + batch;
        dynSizes.push_back(
            rewriter.create<memref::DimOp>(loc, adaptor.index(), idx));
      } else {
        int idx = i - (axis + nIndices - batch) + axis + 1;
        dynSizes.push_back(
            rewriter.create<memref::DimOp>(loc, adaptor.input(), idx));
      }
    }
    Value initOp = rewriter.create<linalg::InitTensorOp>(
        loc, dynSizes, resultType.getShape(), resultType.getElementType());
    auto linalgOp = rewriter.create<linalg::IndexedGenericOp>(
        loc, /*resultTensors=*/ArrayRef<Type>{resultType},
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
    block->addArguments(resultType.getElementType());
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

//===----------------------------------------------------------------------===//
// linalg.pad_tensor conversion patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Pattern to convert a linalg.pad_tensor operation into a fill + subtensor
/// insert. This is needed till pad_tensor op can be fused with its consumers.
struct PadTensorOpConversion : public OpConversionPattern<linalg::PadTensorOp> {
  using OpConversionPattern<linalg::PadTensorOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::PadTensorOp padTensorOp, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const override {
    linalg::PadTensorOpAdaptor padOpAdaptor(args,
                                            padTensorOp->getAttrDictionary());
    // Check that the region is just a yield operation which is returning a
    // scalar that is not one of the arguments of the linalg operation.
    Region &region = padTensorOp.region();
    Block &block = region.front();
    if (!llvm::hasSingleElement(block)) return failure();
    auto yieldOp = cast<linalg::YieldOp>(block.getTerminator());
    if (!llvm::hasSingleElement(yieldOp.values())) return failure();
    Value yieldVal = yieldOp.values().front();
    if (llvm::any_of(block.getArguments(),
                     [&](Value v) { return v == yieldVal; })) {
      return failure();
    }

    OpBuilder::InsertionGuard g(rewriter);
    Location loc = padTensorOp.getLoc();
    auto lowPad = padTensorOp.getMixedLowPad();
    auto highPad = padTensorOp.getMixedHighPad();
    Value source = padOpAdaptor.source();
    RankedTensorType sourceType = padTensorOp.getSourceType();
    int64_t rank = sourceType.getRank();

    // TODO(ravishankarm): Use shape inference interface to get this.
    SmallVector<OpFoldResult> sourceShape;
    SmallVector<Value> outputShape;
    for (int64_t dim : llvm::seq<int64_t>(0, rank)) {
      SmallVector<Value> mapValues;
      Value sourceDim = rewriter.createOrFold<memref::DimOp>(loc, source, dim);
      mapValues.push_back(sourceDim);
      sourceShape.push_back(sourceDim);
      AffineExpr expr = rewriter.getAffineDimExpr(0);
      unsigned numSymbols = 0;
      auto addValueOrAttr = [&](AffineExpr e, OpFoldResult valueOrAttr) {
        if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
          e = e + attr.cast<IntegerAttr>().getInt();
          return e;
        }
        e = e + rewriter.getAffineSymbolExpr(numSymbols++);
        mapValues.push_back(valueOrAttr.get<Value>());
        return e;
      };
      expr = addValueOrAttr(expr, lowPad[dim]);
      expr = addValueOrAttr(expr, highPad[dim]);
      outputShape.push_back(linalg::applyMapToValues(
          rewriter, loc, AffineMap::get(1, numSymbols, expr), mapValues)[0]);
    }
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, outputShape, sourceType.getElementType());
    Value fill =
        rewriter.create<linalg::FillOp>(loc, initTensor, yieldVal).getResult(0);
    SmallVector<OpFoldResult> strides(rank, rewriter.getI64IntegerAttr(1));
    Value replacement = rewriter.create<SubTensorInsertOp>(
        loc, source, fill, lowPad, sourceShape, strides);
    if (padTensorOp.getResultType() != replacement.getType()) {
      replacement = rewriter.create<tensor::CastOp>(
          loc, padTensorOp.getResultType(), replacement);
    }
    rewriter.replaceOp(padTensorOp, replacement);
    return success();
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// mhlo.concatenate conversion patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Converts mhlo.concatenate operation to subtensor ops + subtensor_insert ops.
struct ConcatenateOpConversion
    : public OpConversionPattern<mhlo::ConcatenateOp> {
  using OpConversionPattern<mhlo::ConcatenateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConcatenateOp op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected static shape for output");
    }

    Location loc = op.getLoc();
    int dim = op.dimension();
    int rank = resultType.getRank();
    SmallVector<Value, 3> offsets, sizes, strides;
    for (int i = 0; i < rank; ++i) {
      offsets.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
      sizes.push_back(rewriter.create<memref::DimOp>(loc, args[0], i));
      strides.push_back(rewriter.create<ConstantIndexOp>(loc, 1));
    }
    Value resultDimSize = rewriter.create<ConstantIndexOp>(loc, 0);
    for (auto arg : args) {
      auto size = rewriter.create<memref::DimOp>(loc, arg, dim);
      resultDimSize = rewriter.create<AddIOp>(loc, resultDimSize, size);
    }
    sizes[dim] = resultDimSize;
    auto initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultType.getShape(), resultType.getElementType());
    auto zeroAttr = rewriter.getZeroAttr(resultType.getElementType());
    Value zero = rewriter.create<ConstantOp>(loc, zeroAttr);
    Value result =
        rewriter.create<linalg::FillOp>(loc, initTensor, zero).getResult(0);

    Value accBound = rewriter.create<ConstantIndexOp>(loc, 0);
    for (auto arg : args) {
      offsets[dim] = accBound;
      sizes[dim] = rewriter.create<memref::DimOp>(loc, arg, dim);
      result = rewriter.create<SubTensorInsertOp>(loc, arg, result, offsets,
                                                  sizes, strides);
      accBound = rewriter.create<AddIOp>(loc, accBound, sizes[dim]);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};
}  // namespace

struct ConvertHLOToLinalgOnTensorsPass
    : public PassWrapper<ConvertHLOToLinalgOnTensorsPass, FunctionPass> {
  ConvertHLOToLinalgOnTensorsPass(bool useLinalgOnTensorsPath = false)
      : useLinalgOnTensorsPath(useLinalgOnTensorsPath){};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, linalg::LinalgDialect,
                    mhlo::MhloDialect, ShapeDialect, math::MathDialect,
                    memref::MemRefDialect>();
  }

  void runOnFunction() override {
    OwningRewritePatternList patterns;
    MLIRContext *context = &getContext();
    populateHLOToLinalgOnTensorsConversionPatterns(context, patterns);
    if (useLinalgOnTensorsPath) {
      patterns.insert<PadTensorOpConversion>(context);
    }

    ConversionTarget target(getContext());
    target.addIllegalDialect<mhlo::MhloDialect>();
    // Let the rest fall through.
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (useLinalgOnTensorsPath) {
      // Set linalg.pad_tensor illegal for now.
      target.addIllegalOp<linalg::PadTensorOp>();
    }

    if (failed(applyPartialConversion(getFunction(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }

 private:
  bool useLinalgOnTensorsPath;
};

/// This pass is just added for lit-testing when using the linalg on tensors
/// path. Remove when the linalg on tensors path becomes default.
struct ConvertHLOToLinalgOnTensorsPassExperimental
    : public ConvertHLOToLinalgOnTensorsPass {
  ConvertHLOToLinalgOnTensorsPassExperimental()
      : ConvertHLOToLinalgOnTensorsPass(true){};
};

/// Convert mhlo.constant op into std.const.
struct ConstOpConversion : public OpRewritePattern<mhlo::ConstOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ConstOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ConstantOp>(op, op.value());
    return success();
  }
};

}  // namespace

void populateHLOToLinalgOnTensorsConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  mhlo::populateHLOToLinalgConversionPattern(context, &patterns);
  patterns.insert<TorchIndexSelectOpConversion, ConstOpConversion,
                  ConcatenateOpConversion>(context);
}

static llvm::cl::opt<bool> clUseLinalgOnTensorsPath(
    "iree-linalg-on-tensors-path",
    llvm::cl::desc("Convert from MHLO to Linalg on tensors for linalg on "
                   "tensor codegen path"),
    llvm::cl::init(false));

std::unique_ptr<OperationPass<FuncOp>> createHLOToLinalgOnTensorsPass(
    bool useLinalgOnTensorsPath) {
  return std::make_unique<ConvertHLOToLinalgOnTensorsPass>(
      useLinalgOnTensorsPath);
}

static PassRegistration<ConvertHLOToLinalgOnTensorsPass> legalize_pass(
    "iree-codegen-hlo-to-linalg-on-tensors",
    "Convert from XLA-HLO ops to Linalg ops on tensors", []() {
      return std::make_unique<ConvertHLOToLinalgOnTensorsPass>(
          clUseLinalgOnTensorsPath);
    });

}  // namespace iree_compiler
}  // namespace mlir
