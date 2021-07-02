// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/InputConversion/MHLO/PassDetail.h"
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#include "iree/compiler/InputConversion/MHLO/Rewriters.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

static bool isInBodyOfLinalgExtOps(Operation *op) {
  auto parent_op = op->getParentRegion()->getParentOp();
  return parent_op->getDialect() ==
         parent_op->getContext()
             ->getLoadedDialect<linalg_ext::LinalgExtDialect>();
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
    auto dispatchOp = rewriter.create<IREE::Flow::DispatchWorkgroupsOp>(
        op.getLoc(), workload, op->getResultTypes(),
        /*result_dims=*/ValueRange{},
        /*operands=*/args,
        /*operand_dims=*/ValueRange{},
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

struct ConvertAndDistributeMHLOToLinalgExtPass
    : public ConvertAndDistributeMHLOToLinalgExtBase<
          ConvertAndDistributeMHLOToLinalgExtPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg_ext::LinalgExtDialect, IREE::Flow::FlowDialect,
                    StandardOpsDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    OwningRewritePatternList patterns(&getContext());
    MLIRContext *context = &getContext();

    patterns.insert<SortOpConversion>(context);
    patterns.insert<LinalgExtRegionHLOOpConversion<mhlo::CompareOp>,
                    LinalgExtRegionReturnOpConversion>(context,
                                                       PatternBenefit(1000));

    ConversionTarget target(getContext());
    target
        .addLegalDialect<linalg_ext::LinalgExtDialect, IREE::Flow::FlowDialect,
                         StandardOpsDialect, tensor::TensorDialect>();
    target.addIllegalOp<mhlo::SortOp>();

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
