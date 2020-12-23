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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace {

// Fast polynomial approximation of exp(x) using its reduced range exp(y)
// where y is in the range [0, ln(2)], let y = x - floor(x / ln(2)) * ln(2)
// = x - k * ln(2), exp(x) = exp(y) * 2^k. exp(y) is computed with 4th degree
// polyomial: exp(y) = c0 + c1 * y + c2 * y^2 + c3 * y^3 + c4 * y^4
struct FastExpConversionPattern : public OpRewritePattern<LLVM::ExpOp> {
  using OpRewritePattern<LLVM::ExpOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::ExpOp op,
                                PatternRewriter &rewriter) const override {
    constexpr float ln2Const = 0.693147181f;
    constexpr float ln2InvConst = 1.44269504f;

    // Least squares polynomial fit computed :
    // cValues = np.polyfit(np.linspace(0, math.log(2), 10000), np.exp(x), 4)
    constexpr float cValues[5] = {0.05924867f, 0.15514645f, 0.50308552f,
                                  0.99968939f, 1.00000721531f};
    auto loc = op.getLoc();
    Value x = op.getOperand();

    auto floatType = LLVM::LLVMFloatType::get(rewriter.getContext());
    auto i32Type = LLVM::LLVMIntegerType::get(rewriter.getContext(), 32);

    Value ln2 = rewriter.create<LLVM::ConstantOp>(
        loc, floatType, rewriter.getF32FloatAttr(ln2Const));
    Value ln2Inv = rewriter.create<LLVM::ConstantOp>(
        loc, floatType, rewriter.getF32FloatAttr(ln2InvConst));

    // Compute reduced range input y = x - floor(x / ln(2)) * ln(2)
    Value xL2Inv = rewriter.create<LLVM::FMulOp>(loc, floatType, x, ln2Inv);
    Value kF32 = rewriter.create<LLVM::FFloorOp>(loc, floatType, xL2Inv);
    Value kLn2 = rewriter.create<LLVM::FMulOp>(loc, floatType, kF32, ln2);
    Value y = rewriter.create<LLVM::FSubOp>(loc, floatType, x, kLn2);

    SmallVector<Value, 4> PConst(5);
    for (int i = 0; i < 5; ++i) {
      PConst[i] = rewriter.create<LLVM::ConstantOp>(
          loc, floatType, rewriter.getF32FloatAttr(cValues[i]));
    }
    // Evaluate exp(y) = sum(c[i] * y**i, i)
    Value expY = rewriter.create<LLVM::FMulOp>(loc, floatType, y, PConst[0]);
    expY = rewriter.create<LLVM::FAddOp>(loc, floatType, expY, PConst[1]);
    expY = rewriter.create<LLVM::FMulOp>(loc, floatType, expY, y);
    expY = rewriter.create<LLVM::FAddOp>(loc, floatType, expY, PConst[2]);
    expY = rewriter.create<LLVM::FMulOp>(loc, floatType, expY, y);
    expY = rewriter.create<LLVM::FAddOp>(loc, floatType, expY, PConst[3]);
    expY = rewriter.create<LLVM::FMulOp>(loc, floatType, expY, y);
    expY = rewriter.create<LLVM::FAddOp>(loc, floatType, expY, PConst[4]);

    // Compute exp2(k) with integer bitshift:
    // exp2(k) = f32_bitcast((127 + k) << 23)
    Value fPBias = rewriter.create<LLVM::ConstantOp>(
        loc, i32Type, rewriter.getI32IntegerAttr(127));
    Value k = rewriter.create<LLVM::FPToSIOp>(loc, i32Type, kF32);
    Value kPlusfPBias = rewriter.create<LLVM::AddOp>(loc, i32Type, k, fPBias);
    Value shiftConst = rewriter.create<LLVM::ConstantOp>(
        loc, i32Type, rewriter.getI32IntegerAttr(23));
    Value twoPowkI =
        rewriter.create<LLVM::ShlOp>(loc, i32Type, kPlusfPBias, shiftConst);
    Value twoPowk = rewriter.create<LLVM::BitcastOp>(loc, floatType, twoPowkI);
    expY = rewriter.create<LLVM::FMulOp>(loc, floatType, expY, twoPowk);
    rewriter.replaceOp(op, {expY});
    // TODO(ataei): Handle overflow and underflow cases (e.g |k| > 128).
    return success();
  }
};

struct FastExpConversionPass
    : public PassWrapper<FastExpConversionPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() override;
};

}  // namespace

void populateFastExpConversionPatterns(OwningRewritePatternList &patterns,
                                       MLIRContext *context) {
  patterns.insert<FastExpConversionPattern>(context);
}

void FastExpConversionPass::runOnOperation() {
  auto moduleOp = getOperation();
  auto context = moduleOp.getContext();
  OwningRewritePatternList patterns;
  populateFastExpConversionPatterns(patterns, context);
  applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));
}

std::unique_ptr<OperationPass<ModuleOp>>
createFastExpApproximationConversionPass() {
  return std::make_unique<FastExpConversionPass>();
}

static PassRegistration<OperationPass<ModuleOp>> pass(
    "iree-codegen-linalg-to-llvm-fast-exp-conversion-pass",
    "Convert llvm.intr.exp into its fast polynomial approximation version",
    [] { return std::make_unique<FastExpConversionPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
