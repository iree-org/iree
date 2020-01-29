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

#include "iree/compiler/Translation/XLAToLinalg/MapHloToScalarOp.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {
namespace {

// TODO(hanchung): Refactor the common part with XLA.
// Returns an ArrayAttr that contains `nParallelLoops` "parallel".
ArrayAttr GetNParallelLoopsAttrs(unsigned nParallelLoops, Builder builder) {
  auto parallelLoopTypeAttr = builder.getStringAttr("parallel");
  SmallVector<Attribute, 3> iteratorTypes(nParallelLoops, parallelLoopTypeAttr);
  return builder.getArrayAttr(iteratorTypes);
}

template <typename HloOp>
class PointwiseConverter : public OpConversionPattern<HloOp> {
 public:
  using OpConversionPattern<HloOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      HloOp hloOp, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = hloOp.getLoc();
    // Unary operation doesn't have getOperand(i) method, so use getOperation()
    // first and then invoke the method.
    ShapedType argType = hloOp.getOperation()
                             ->getOperand(0)
                             .getType()
                             .template dyn_cast<ShapedType>();
    if (!argType || !argType.getElementType().isIntOrFloat()) {
      return ConversionPattern::matchFailure();
    }

    // Construct the indexing maps needed for linalg.generic op.
    auto newArgType = args.front().getType().dyn_cast<RankedTensorType>();
    if (!newArgType) return ConversionPattern::matchFailure();
    int numLoops = newArgType.getRank();
    if (!llvm::all_of(llvm::drop_begin(args, 1),
                      [&](Value arg) { return arg.getType() == newArgType; })) {
      return ConversionPattern::matchFailure();
    }
    SmallVector<Attribute, 2> indexingMaps(
        args.size(),
        AffineMapAttr::get(rewriter.getMultiDimIdentityMap(numLoops)));

    SmallVector<Type, 1> resultTypes = {hloOp.getResult().getType()};
    SmallVector<Value, 2> linalgOpArgs(args.begin(), args.end());
    SmallVector<Type, 2> bodyArgTypes(args.size(), newArgType.getElementType());
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, resultTypes, linalgOpArgs,
        rewriter.getI64IntegerAttr(bodyArgTypes.size()),  // args_in
        rewriter.getI64IntegerAttr(resultTypes.size()),   // args_out
        rewriter.getArrayAttr(indexingMaps),
        GetNParallelLoopsAttrs(numLoops, rewriter),
        /*doc=*/nullptr, /*fun=*/nullptr, /*library_call=*/nullptr);

    // Add a block to the region.
    auto* region = &linalgOp.region();
    auto* block = rewriter.createBlock(region, region->end());
    block->addArguments(bodyArgTypes);

    SmallVector<Value, 4> bodyArgs;
    for (int i = 0, e = bodyArgTypes.size(); i < e; ++i) {
      bodyArgs.push_back(block->getArgument(i));
    }

    Type resultElemType = resultTypes[0].cast<ShapedType>().getElementType();
    Operation* op = mapToStdScalarOp(hloOp, resultElemType, bodyArgs, rewriter);
    if (!op) return ConversionPattern::matchFailure();
    rewriter.create<linalg::YieldOp>(loc, op->getResults());

    rewriter.replaceOp(hloOp, linalgOp.output_tensors());
    return ConversionPattern::matchSuccess();
  }
};

void populateXlaToLinalgConversionPattern(MLIRContext* context,
                                          OwningRewritePatternList* patterns) {
  patterns->insert<
      PointwiseConverter<xla_hlo::AddOp>, PointwiseConverter<xla_hlo::DivOp>,
      PointwiseConverter<xla_hlo::ExpOp>, PointwiseConverter<xla_hlo::MulOp>,
      PointwiseConverter<xla_hlo::SubOp> >(context);
}

struct XlaLegalizeToLinalg : public FunctionPass<XlaLegalizeToLinalg> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect>();

    auto func = getFunction();
    populateXlaToLinalgConversionPattern(func.getContext(), &patterns);
    if (failed(applyPartialConversion(func, target, patterns, nullptr))) {
      signalPassFailure();
    }
  }
};

}  // namespace

static PassRegistration<XlaLegalizeToLinalg> legalize_pass(
    "iree-hlo-to-linalg", "Legalize from HLO dialect to Linalg dialect");

}  // namespace iree_compiler
}  // namespace mlir
