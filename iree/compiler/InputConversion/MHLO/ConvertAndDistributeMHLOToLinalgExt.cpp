// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/InputConversion/MHLO/PassDetail.h"
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#include "iree/compiler/InputConversion/MHLO/Rewriters.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

static bool isInBodyOfLinalgExtOps(Operation *op) {
  auto parent_op = op->getParentRegion()->getParentOp();
  return parent_op->getDialect() ==
         parent_op->getContext()
             ->getLoadedDialect<linalg_ext::LinalgExtDialect>();
}

static SmallVector<int64_t> extract1DVector(DenseIntElementsAttr elements) {
  SmallVector<int64_t> ret;
  for (const APInt &element : elements) {
    ret.push_back(element.getLimitedValue());
  }
  return ret;
}

namespace {

//===----------------------------------------------------------------------===//
// Base classes.
//===----------------------------------------------------------------------===//

template <typename Derived, typename OpTy>
struct ConvertToLinalgExtPattern : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpTy op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const final {
    Value one = rewriter.create<ConstantIndexOp>(op.getLoc(), 1);
    SmallVector<Value> workload(3, one);

    // Gather the dynamic dimensions for all operands.
    SmallVector<Value> operandDynamicDims;
    for (Value arg : args) {
      if (auto rt = arg.getType().dyn_cast<RankedTensorType>()) {
        for (unsigned i = 0; i < rt.getRank(); ++i) {
          if (!rt.isDynamicDim(i)) continue;
          auto dim = rewriter.createOrFold<tensor::DimOp>(op.getLoc(), arg, i);
          operandDynamicDims.push_back(dim);
        }
      }
    }

    auto dispatchOp = rewriter.create<IREE::Flow::DispatchWorkgroupsOp>(
        op.getLoc(), workload, op->getResultTypes(),
        /*result_dims=*/ValueRange{},
        /*operands=*/args,
        /*operand_dims=*/operandDynamicDims,
        /*tied_operands=*/Derived::getTiedResultOperandIndices(args));
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&dispatchOp.getRegion().front());
      if (failed(Derived::lowerMHLOOp(dispatchOp, op, args, rewriter))) {
        return failure();
      }
      rewriter.create<IREE::Flow::ReturnOp>(op.getLoc());
    }
    rewriter.replaceOp(op, dispatchOp.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Region operations lowering.
//===----------------------------------------------------------------------===//

template <typename OpTy>
struct LinalgExtRegionHLOOpConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      OpTy op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const final {
    if (!isInBodyOfLinalgExtOps(op)) return failure();
    if (!op.getResult().getType().template isa<TensorType>()) return failure();
    if (llvm::all_of(args, [](Value arg) {
          return arg.getType().template isa<TensorType>();
        })) {
      return failure();
    }
    Value result = lmhlo::HloOpToStdScalarOp::map<OpTy>(
        op, getElementTypeOrSelf(op.getType()), args, &rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LinalgExtRegionReturnOpConversion
    : public OpConversionPattern<mhlo::ReturnOp> {
  using OpConversionPattern<mhlo::ReturnOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ReturnOp op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const final {
    if (!isInBodyOfLinalgExtOps(op)) return failure();
    rewriter.replaceOpWithNewOp<linalg_ext::YieldOp>(op, args);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

struct SortOpConversion
    : public ConvertToLinalgExtPattern<SortOpConversion, mhlo::SortOp> {
  using ConvertToLinalgExtPattern<SortOpConversion,
                                  mhlo::SortOp>::ConvertToLinalgExtPattern;

  static SmallVector<int64_t> getTiedResultOperandIndices(
      ArrayRef<Value> args) {
    return llvm::to_vector<4>(llvm::seq<int64_t>(0, args.size()));
  }

  static LogicalResult lowerMHLOOp(IREE::Flow::DispatchWorkgroupsOp dispatchOp,
                                   mhlo::SortOp op, ArrayRef<Value> args,
                                   ConversionPatternRewriter &rewriter) {
    auto blockArgs = dispatchOp.getClosureBodyRegion().getArguments();
    SmallVector<Value> initValues;
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    for (auto it : llvm::zip(args, blockArgs)) {
      auto argTy = std::get<0>(it).getType().cast<RankedTensorType>();
      auto blockArg = std::get<1>(it);
      initValues.push_back(
          b.create<IREE::Flow::DispatchTensorLoadOp>(argTy, blockArg));
    }

    auto sortOp = b.create<linalg_ext::SortOp>(op.getResultTypes(),
                                               /*inputs=*/ValueRange{},
                                               initValues, op.dimensionAttr());
    rewriter.inlineRegionBefore(op.comparator(), sortOp.region(),
                                sortOp.region().begin());
    Region &region = sortOp.region();
    Block &block = region.front();
    TypeConverter::SignatureConversion signature_converter(
        block.getNumArguments());
    for (auto en : llvm::enumerate(block.getArguments())) {
      signature_converter.addInputs(en.index(),
                                    getElementTypeOrSelf(en.value().getType()));
    }
    rewriter.applySignatureConversion(&region, signature_converter);

    for (auto it : llvm::zip(sortOp.getResults(), blockArgs)) {
      auto value = std::get<0>(it);
      auto target = std::get<1>(it);
      b.create<IREE::Flow::DispatchTensorStoreOp>(value, target);
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

struct ScatterOpConversion
    : public ConvertToLinalgExtPattern<ScatterOpConversion, mhlo::ScatterOp> {
  using ConvertToLinalgExtPattern<ScatterOpConversion,
                                  mhlo::ScatterOp>::ConvertToLinalgExtPattern;

  /// Returns true if the `dimensionNumbers` from the mhlo.scatter op follows a
  /// canonical form:
  ///
  /// * The rank of indices is greater than or equal to two.
  /// * The index_vector_dim is the last dim of indices.
  /// * Scatter dims to operand dims order: (0, ... , n)
  /// * Inserted window dims order: (0, ... , d)
  /// * Update window dims order: (d + 1, ... , m)
  ///
  /// TODO(hanchung): Add a pattern for legalizing mhlo.scatter to canonical
  /// form to MHLOToMHLOPreprocessingPass.
  static bool hasCanonicalDimensionNumbers(mhlo::ScatterOp op) {
    auto dimNumbers = op.scatter_dimension_numbers();
    auto indicesType = op.scatter_indices().getType().cast<ShapedType>();
    auto indicesRank = indicesType.getRank();

    if (indicesRank < 2) return false;
    if (dimNumbers.index_vector_dim().getInt() != indicesRank - 1) return false;

    auto indexDepth = indicesType.getShape().back();
    auto scatterDimsToOperandDims =
        extract1DVector(dimNumbers.scatter_dims_to_operand_dims());
    if (scatterDimsToOperandDims.size() != indexDepth) return false;
    for (auto en : llvm::enumerate(scatterDimsToOperandDims)) {
      if (en.index() != en.value()) return false;
    }

    auto insertedWindowDims =
        extract1DVector(dimNumbers.inserted_window_dims());
    for (auto en : llvm::enumerate(insertedWindowDims)) {
      if (en.index() != en.value()) return false;
    }

    auto updateWindowDims = extract1DVector(dimNumbers.update_window_dims());
    for (auto en : llvm::enumerate(updateWindowDims)) {
      if (en.index() + insertedWindowDims.size() != en.value()) return false;
    }

    return true;
  }

  static SmallVector<int64_t> getTiedResultOperandIndices(
      ArrayRef<Value> args) {
    // Mark linalg_ext.scatter::orinigal as readwrite tensor.
    return {0};
  }

  static LogicalResult collapseBatchDimsIfNeeded(Value &indices, Value &updates,
                                                 ImplicitLocOpBuilder &b) {
    auto indicesType = indices.getType().cast<ShapedType>();
    auto updatesType = updates.getType().cast<ShapedType>();
    if (indicesType.getRank() == 2) return success();

    int64_t batchSize = 1;
    auto indicesRank = indicesType.getRank();
    auto shape = indicesType.getShape();
    for (auto i : shape.drop_back(1)) {  // drop index_detph_dim
      if (i == ShapedType::kDynamicSize) {
        batchSize = ShapedType::kDynamicSize;
      } else if (batchSize != ShapedType::kDynamicSize) {
        batchSize *= i;
      }
    }

    SmallVector<linalg::ReassociationIndices> map;
    map.emplace_back(
        llvm::to_vector<4>(llvm::seq<int64_t>(0, indicesRank - 1)));
    map.emplace_back(1, indicesRank - 1);
    auto resultType = RankedTensorType::get({batchSize, shape.back()},
                                            indicesType.getElementType());
    indices = b.create<linalg::TensorCollapseShapeOp>(resultType, indices, map);

    auto updateShape = updatesType.getShape().drop_front(shape.size() - 1);
    SmallVector<int64_t> collapsedUpdateShape = {batchSize};
    collapsedUpdateShape.append(updateShape.begin(), updateShape.end());
    resultType = RankedTensorType::get(collapsedUpdateShape,
                                       updatesType.getElementType());
    // The batching dims are identical.
    map.pop_back();
    for (auto i : llvm::seq<int64_t>(indicesRank - 1, updatesType.getRank())) {
      map.emplace_back(1, i);
    }
    updates = b.create<linalg::TensorCollapseShapeOp>(resultType, updates, map);

    return success();
  }

  static LogicalResult lowerMHLOOp(IREE::Flow::DispatchWorkgroupsOp dispatchOp,
                                   mhlo::ScatterOp op, ArrayRef<Value> args,
                                   ConversionPatternRewriter &rewriter) {
    if (!hasCanonicalDimensionNumbers(op)) return failure();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    mhlo::ScatterOpAdaptor adaptor(args);

    auto blockArgs = dispatchOp.getClosureBodyRegion().getArguments();
    Value original = b.create<IREE::Flow::DispatchTensorLoadOp>(
        adaptor.operand().getType().cast<RankedTensorType>(), blockArgs[0]);
    Value indices = b.create<IREE::Flow::DispatchTensorLoadOp>(
        adaptor.scatter_indices().getType().cast<RankedTensorType>(),
        blockArgs[1]);
    Value updates = b.create<IREE::Flow::DispatchTensorLoadOp>(
        adaptor.updates().getType().cast<RankedTensorType>(), blockArgs[2]);

    if (failed(collapseBatchDimsIfNeeded(indices, updates, b))) {
      return failure();
    }
    auto scatterOp = b.create<linalg_ext::ScatterOp>(
        op->getResultTypes(), ValueRange{updates, indices},
        ValueRange{original});

    rewriter.inlineRegionBefore(op.update_computation(), scatterOp.region(),
                                scatterOp.region().begin());
    Region &region = scatterOp.region();
    TypeConverter::SignatureConversion signatureConverter(2);
    Type argType = getElementTypeOrSelf(original.getType());
    // mhlo.scatter ops takes:
    //   output[O] = update_computation(output[O], updates[U])
    // where output[O] maps to block args #1 in linalg_ext.scatter ops.
    signatureConverter.addInputs(1, argType);
    signatureConverter.addInputs(0, argType);
    rewriter.applySignatureConversion(&region, signatureConverter);

    b.create<IREE::Flow::DispatchTensorStoreOp>(scatterOp.getResult(0),
                                                blockArgs[0]);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct ConvertAndDistributeMHLOToLinalgExtPass
    : public ConvertAndDistributeMHLOToLinalgExtBase<
          ConvertAndDistributeMHLOToLinalgExtPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg_ext::LinalgExtDialect, linalg::LinalgDialect,
                    IREE::Flow::FlowDialect, StandardOpsDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    OwningRewritePatternList patterns(&getContext());
    MLIRContext *context = &getContext();

    patterns.insert<SortOpConversion, ScatterOpConversion>(context);
    patterns.insert<LinalgExtRegionHLOOpConversion<mhlo::CompareOp>,
                    LinalgExtRegionHLOOpConversion<mhlo::AddOp>,
                    LinalgExtRegionReturnOpConversion>(context,
                                                       PatternBenefit(1000));

    ConversionTarget target(getContext());
    target.addLegalDialect<linalg_ext::LinalgExtDialect, linalg::LinalgDialect,
                           IREE::Flow::FlowDialect, StandardOpsDialect,
                           tensor::TensorDialect>();
    target.addIllegalOp<mhlo::SortOp, mhlo::ScatterOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
createConvertAndDistributeMHLOToLinalgExtPass() {
  return std::make_unique<ConvertAndDistributeMHLOToLinalgExtPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
