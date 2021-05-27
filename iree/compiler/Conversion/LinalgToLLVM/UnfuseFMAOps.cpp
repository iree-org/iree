// Copyright 2021 Google LLC
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
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Rewrites llvm.intr.fma as its un-fuse version.
// TODO(ataei): Upstream this pattern if needed ?
class UnfusedFMAOpsPassConversion : public OpRewritePattern<LLVM::FMAOp> {
 public:
  using OpRewritePattern<LLVM::FMAOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::FMAOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto mulPart = rewriter.create<LLVM::FMulOp>(loc, op.getResult().getType(),
                                                 op.a(), op.b());
    auto fmaResult = rewriter.create<LLVM::FAddOp>(
        loc, mulPart.getResult().getType(), mulPart.getResult(), op.c());
    rewriter.replaceOp(op, fmaResult.getResult());
    return success();
  }
};
}  // namespace

namespace {
struct UnfusedFMAOpsPass : PassWrapper<UnfusedFMAOpsPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnFunction() override;
};
}  // namespace

void populateUnfusedFMAOpsPassPatterns(MLIRContext *context,
                                       OwningRewritePatternList &patterns) {
  patterns.insert<UnfusedFMAOpsPassConversion>(context);
}

void UnfusedFMAOpsPass::runOnFunction() {
  auto funcOp = getOperation();
  auto context = funcOp.getContext();
  OwningRewritePatternList patterns(&getContext());
  populateUnfusedFMAOpsPassPatterns(context, patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<FunctionPass> createUnfusedFMAOpsPass() {
  return std::make_unique<UnfusedFMAOpsPass>();
}

static PassRegistration<UnfusedFMAOpsPass> pass(
    "iree-codegen-linalg-to-llvm-unfuse-fma-pass",
    "Convert llvm.fma into unfused mulf and addf ops",
    [] { return std::make_unique<UnfusedFMAOpsPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
