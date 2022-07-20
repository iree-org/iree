// License stuff here

// do we need these?
// #include "iree_tf_compiler/TFL/PassDetail.h"
// #include "iree_tf_compiler/TFL/Passes.h"
// #include "iree_tf_compiler/Utils/ConversionUtils.h"
// #include "llvm/ADT/StringExtras.h"
// #include "llvm/Support/FormatVariadic.h"
#include "mlir/Pass/Pass.h"

// imports from legalize_tf.cc
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_set>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"  // from @llvm-project
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_common.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_utils.h"
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"

// will ultimately be under mlir::tosa
namespace mlir {
namespace iree_integrations {
namespace TFL {
namespace {

#define DECL_CONVERT_OP(tfl_op)                                              \
  struct ConvertTFL##tfl_op##Op : public RewritePattern {                    \
    explicit ConvertTFL##tfl_op##Op(MLIRContext* context)                    \
        : RewritePattern(TFL::tfl_op##Op::getOperationName(), 1, context) {} \
    LogicalResult matchAndRewrite(Operation* op,                             \
                                  PatternRewriter& rewriter) const override; \
  }

DECL_CONVERT_OP(While);
DECL_CONVERT_OP(If);

#undef DECL_CONVERT_OP

template <typename TflOp, typename TosaOp>
static LogicalResult matchAndRewriteWhileIf(Operation* op,
                                           mlir::OperandRange operands,
                                           PatternRewriter& rewriter) {
  auto tfl_cond_op = cast<TflOp>(op);

  rewriter.replaceOpWithNewOp<TosaOp>(op, op->getResultTypes(),
                                             op->getOperands(),
                                             opt->getRegions());

  return success();
}

LogicalResult ConvertTFLWhileOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  return matchAndRewriteWhileIf<TFL::WhileOp, tosa::WhileOp>(op, op->getOperands(),
                                                        rewriter);
}

LogicalResult ConvertTFLIfOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  return matchAndRewriteWhileIf<TFL::IfOp, tosa::IfOp>(op, op->getOperands(),
                                                        rewriter);
}

struct ConvertConditionalsPass : public PassWrapper<ConvertConditionalsPass, OperationPass<>> {
  StringRef getArgument() const final {
    return "tfl-to-tosa-convert-conditionals";
  }
  StringRef getDescription() const final {
    return  "Lower tfl.while and tfl.if to tosa dialect";
  }

  void runOnOperation() override {
    Operation *op = getOperation();

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<ConvertTFLWhileOp>(context);
    patterns.insert<ConvertTFLIfOp>(context);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

} // anonymous namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(ConvertConditionalsPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(ConvertConditionalsPass)

std::unique_ptr<OperationPass<>> createConvertConditionalsPass() {
  return std::make_unique<ConvertConditionalsPass>();
}

} // namespace TFL
} // namespace iree_integrations
} // namespace mlir