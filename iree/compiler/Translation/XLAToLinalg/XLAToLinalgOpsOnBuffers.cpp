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
// See the License for the specific language governing permissions an
// limitations under the License.
#include <memory>

#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Translation/XLAToLinalg/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {

/// Remove IREE::LoadInputOp operations
struct RemoveLoadInputOpPattern : OpConversionPattern<IREE::LoadInputOp> {
  using OpConversionPattern<IREE::LoadInputOp>::OpConversionPattern;
  PatternMatchResult matchAndRewrite(
      IREE::LoadInputOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, op.getOperand());
    return matchSuccess();
  }
};

template <typename SrcOpTy, typename OpTy>
struct ConvertLinalgOp {
  static OpTy apply(SrcOpTy op, ConversionPatternRewriter& rewriter,
                    Location& loc, ArrayRef<Value> args, Value& result);
};

// Convert HLO ops to Linalg named ops.
template <typename OpTy, typename LinalgOpTy>
struct ConvertHLOToLinalgNamedOp : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      OpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    if (!op.getResult().hasOneUse()) return ConversionPattern::matchFailure();
    auto resultMemrefOp =
        dyn_cast<IREE::StoreOutputOp>(*op.getResult().user_begin());
    if (!resultMemrefOp) return ConversionPattern::matchFailure();
    auto result = resultMemrefOp.dst();

    OpBuilder::InsertionGuard linalgOpGuard(rewriter);
    auto loc = op.getLoc();
    LinalgOpTy linalgOp = ConvertLinalgOp<OpTy, LinalgOpTy>::apply(
        op, rewriter, loc, operands, result);

    rewriter.replaceOp(op, linalgOp.getOutputBuffers());
    rewriter.eraseOp(resultMemrefOp);
    return ConversionPattern::matchSuccess();
  }
};

template <>
linalg::MatmulOp ConvertLinalgOp<xla_hlo::DotOp, linalg::MatmulOp>::apply(
    xla_hlo::DotOp op, ConversionPatternRewriter& rewriter, Location& loc,
    ArrayRef<Value> args, Value& result) {
  auto matmulOp =
      rewriter.create<linalg::MatmulOp>(loc, args[0], args[1], result);
  return matmulOp;
}

template <>
linalg::ConvOp ConvertLinalgOp<xla_hlo::ConvOp, linalg::ConvOp>::apply(
    xla_hlo::ConvOp op, ConversionPatternRewriter& rewriter, Location& loc,
    ArrayRef<Value> args, Value& result) {
  llvm::SmallVector<Attribute, 4> strides;
  llvm::SmallVector<Attribute, 4> dilation;
  if (op.window_strides().hasValue()) {
    strides.insert(strides.begin(),
                   op.window_strides().getValue().getAttributeValues().begin(),
                   op.window_strides().getValue().getAttributeValues().end());
  }

  // TODO(ataei): Support dilated convolution only for now we need to add lhs
  // for deconvolution support
  if (op.rhs_dilation().hasValue()) {
    dilation.insert(dilation.begin(),
                    op.rhs_dilation().getValue().getAttributeValues().begin(),
                    op.rhs_dilation().getValue().getAttributeValues().end());
  }

  auto stridesArg = ArrayAttr::get(strides, op.getContext());
  auto dilationArg = ArrayAttr::get(dilation, op.getContext());

  auto convOp = rewriter.create<linalg::ConvOp>(loc, args[1], args[0], result,
                                                stridesArg, dilationArg);

  return convOp;
}

void populateXlaToLinalgNamedOpsConversionPattern(
    MLIRContext* context, OwningRewritePatternList& patterns) {
  patterns.insert<RemoveLoadInputOpPattern,
                  ConvertHLOToLinalgNamedOp<xla_hlo::DotOp, linalg::MatmulOp>,
                  ConvertHLOToLinalgNamedOp<xla_hlo::ConvOp, linalg::ConvOp>>(
      context);
}

struct XLAToLinalgOpsOnBufferConversionPass
    : public FunctionPass<XLAToLinalgOpsOnBufferConversionPass> {
  void runOnFunction() override {
    auto func = getFunction();
    OwningRewritePatternList patterns;
    MLIRContext* context = &getContext();
    ConversionTarget target(*context);
    target.addLegalOp<FuncOp>();
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect>();
    populateXlaToLinalgNamedOpsConversionPattern(context, patterns);
    if (failed(applyPartialConversion(func, target, patterns))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OpPassBase<FuncOp>> createXLAToLinalgOpsOnBufferPass() {
  return std::make_unique<XLAToLinalgOpsOnBufferConversionPass>();
}

static PassRegistration<XLAToLinalgOpsOnBufferConversionPass> legalize_pass(
    "iree-hlo-to-named-linalg",
    "Legalize some ops from HLO dialect to named ops in Linalg dialect");
}  // namespace iree_compiler
}  // namespace mlir
