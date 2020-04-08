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

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

static bool isAllZero(DenseIntElementsAttr attr) {
  if (!attr.isSplat()) return false;
  return attr.getSplatValue<IntegerAttr>().getInt() == 0;
}

static bool isSplatConst(Value val) {
  auto op = val.getDefiningOp();
  if (!dyn_cast_or_null<xla_hlo::ConstOp>(op)) return false;
  Attribute attr = cast<xla_hlo::ConstOp>(op).value();
  if (auto x = attr.dyn_cast<DenseElementsAttr>()) return x.isSplat();
  return true;
}

// Returns true if a >= b. If the method can not evaluate or a < b, returns
// false.
static bool isGreaterThanOrEqualTo(Value a, Value b) {
  auto constOpA = dyn_cast_or_null<xla_hlo::ConstOp>(a.getDefiningOp());
  auto constOpB = dyn_cast_or_null<xla_hlo::ConstOp>(b.getDefiningOp());
  if (!constOpA || !constOpB) return false;
  Attribute attrA = constOpA.value();
  Attribute attrB = constOpB.value();
  assert(attrA.getType() == attrB.getType());

  if (attrA.isa<DenseFPElementsAttr>()) {
    auto valA =
        attrA.cast<DenseFPElementsAttr>().getSplatValue<FloatAttr>().getValue();
    auto valB =
        attrB.cast<DenseFPElementsAttr>().getSplatValue<FloatAttr>().getValue();
    return valA >= valB;
  }

  if (attrA.isa<DenseIntElementsAttr>()) {
    auto valA = attrA.cast<DenseIntElementsAttr>()
                    .getSplatValue<IntegerAttr>()
                    .getInt();
    auto valB = attrB.cast<DenseIntElementsAttr>()
                    .getSplatValue<IntegerAttr>()
                    .getInt();
    return valA >= valB;
  }

  llvm_unreachable("unknown value type");
}

// Returns the value of the given index. If `attr` is a nullptr, returns 0.
static int64_t getAttrValue(DenseIntElementsAttr attr,
                            ArrayRef<uint64_t> index) {
  if (!attr) return 0;
  return attr.getValue<int64_t>(index);
}

static DenseIntElementsAttr getPaddingAttrs(xla_hlo::PadOp padOp,
                                            DenseIntElementsAttr paddingAttr,
                                            Builder *builder) {
  int rank = padOp.operand().getType().cast<RankedTensorType>().getRank();
  SmallVector<int64_t, 8> padding;
  for (unsigned i = 0; i < rank; ++i) {
    padding.push_back(getAttrValue(paddingAttr, {i, 0}) +
                      padOp.edge_padding_low().getValue<int64_t>(i));
    padding.push_back(getAttrValue(paddingAttr, {i, 1}) +
                      padOp.edge_padding_high().getValue<int64_t>(i));
  }
  // paddingAttr.getType() doesn't work because it can be a nullptr.
  auto type = RankedTensorType::get({rank, 2}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(type, padding);
}

class FoldPadIntoMaxPool : public OpRewritePattern<xla_hlo::ReduceWindowOp> {
 public:
  using OpRewritePattern<xla_hlo::ReduceWindowOp>::OpRewritePattern;

  // Returns true if the region has a single block, and the block is formed as:
  // ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):       // no predecessors
  //   %res = xla_hlo.maximum %lhs, %rhs : tensor<f32>
  //   "xla_hlo.return"(%res) : (tensor<f32>) -> ()
  bool isMaxPool(Region &region) const {
    if (region.getBlocks().size() != 1) return false;
    Block &block = region.front();
    if (block.getOperations().size() != 2) return false;
    auto op = block.begin();
    return isa<xla_hlo::MaxOp>(op);
  }

  // Returns true if there is a reduction window doesn't cover any elements.
  // ReudceWindow takes padding attributes. The leftmost pixel of the first
  // window is at "0 - pad_low[i]", and the rightmost pixel is at
  // "window_size - 1 - pad_low[i]". If there is a window doesn't cover any
  // elements, the result of the window is initial value.
  bool hasOutsideWindow(xla_hlo::ReduceWindowOp op) const {
    int rank = op.getType().cast<RankedTensorType>().getRank();
    for (unsigned i = 0; i < rank; ++i) {
      int rightmost = (getAttrValue(op.window_dimensions(), {i}) - 1) -
                      getAttrValue(op.paddingAttr(), {i, 0});
      if (rightmost >= 0) return false;
    }
    return true;
  }

  // Matches the following conditions, so we can fold the PadOp into the
  // following ReduceWindowOp:
  // 1) This is a max pooling operation.
  // 2) The operand of ReduceWindowOp is defined by a PadOp, and the operand of
  //    the PadOp is defined by a MaxOp.
  // 3) All the elements of the result of MaxOp are greater than or equal to
  //    padding value.
  // 4) There is no a window that just contains initial value of ReduceWindowOp.
  // 5) The initial value of ReduceWindowOp is less than or equal to the
  //    padding value.
  // These conditions imply that all the elements are greater than or equal to
  // the padding value, so we can fold the PadOp into the ReduceWindowOp, and
  // use the padding value as initial value.
  LogicalResult matchAndRewrite(xla_hlo::ReduceWindowOp op,
                                PatternRewriter &rewriter) const override {
    if (!isMaxPool(op.body())) return failure();
    if (!isSplatConst(op.init_value())) return failure();

    auto padOp = dyn_cast_or_null<xla_hlo::PadOp>(op.operand().getDefiningOp());
    if (!padOp) return failure();
    if (!isAllZero(padOp.interior_padding())) return failure();
    if (!isSplatConst(padOp.padding_value())) return failure();

    auto maxOp =
        dyn_cast_or_null<xla_hlo::MaxOp>(padOp.operand().getDefiningOp());
    if (!maxOp) return failure();
    Value maxConstOperand = maxOp.lhs();
    if (!isSplatConst(maxConstOperand)) maxConstOperand = maxOp.rhs();
    if (!isSplatConst(maxConstOperand)) return failure();

    // In case that max op isn't folded when it takes two constant attributes.
    if (isSplatConst(maxOp.rhs()) &&
        isGreaterThanOrEqualTo(maxOp.rhs(), maxConstOperand))
      maxConstOperand = maxOp.rhs();

    // If `maxConstOperand` is greater than or equal to the padding value, then
    // all the elements (in `op.operand()`) are greater than or equal to the
    // padding value.
    if (!isGreaterThanOrEqualTo(maxConstOperand, padOp.padding_value()))
      return failure();

    // If the padding value is greater than or equal to initial value and all
    // the windows cover at least one padding value, we can just make the
    // initial value as padding value.
    // There are two cases:
    //   1) There is no outside window, i.e., all the windows at least cover an
    //      element. Since all the elements are greater than or equal to the
    //      padding value, it implies that the result is always the same if
    //      "`op.init_value()` <= padding_value".
    //   2) There is a outside window, i.e., all the elements are
    //      `op.init_value()`. In this case, the result is `op.init_value()`. We
    //      can not make the initial value as padding value in this case.
    if (!isGreaterThanOrEqualTo(padOp.padding_value(), op.init_value()) ||
        hasOutsideWindow(op))
      return failure();

    auto resultType = op.getResult().getType();
    auto paddingAttr = getPaddingAttrs(padOp, op.paddingAttr(), &rewriter);
    auto newOp = rewriter.create<xla_hlo::ReduceWindowOp>(
        op.getLoc(), resultType, padOp.operand(), padOp.padding_value(),
        op.window_dimensionsAttr(), op.window_stridesAttr(),
        op.base_dilationsAttr(), op.window_dilationsAttr(), paddingAttr);
    newOp.body().takeBody(op.body());
    rewriter.replaceOp(op, newOp.getResult());

    return success();
  }
};

struct HLOToHLOPreprocessing
    : public PassWrapper<HLOToHLOPreprocessing, FunctionPass> {
  void runOnFunction() override {
    MLIRContext *context = &getContext();
    ConversionTarget conversionTarget(*context);
    OwningRewritePatternList conversionPatterns;

    conversionTarget
        .addLegalDialect<xla_hlo::XlaHloDialect, StandardOpsDialect>();
    conversionTarget.addIllegalOp<xla_hlo::BatchNormInferenceOp>();

    xla_hlo::PopulateUnfuseBatchNormPatterns(context, &conversionPatterns);
    if (failed(applyPartialConversion(getFunction(), conversionTarget,
                                      conversionPatterns))) {
      return signalPassFailure();
    }

    OwningRewritePatternList patterns;
    patterns.insert<FoldPadIntoMaxPool>(context);
    applyPatternsGreedily(getOperation(), patterns);
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createHLOPreprocessingPass() {
  return std::make_unique<HLOToHLOPreprocessing>();
}

static PassRegistration<HLOToHLOPreprocessing> legalize_pass(
    "iree-flow-hlo-to-hlo-preprocessing",
    "Apply hlo to hlo transformations for some hlo ops");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
