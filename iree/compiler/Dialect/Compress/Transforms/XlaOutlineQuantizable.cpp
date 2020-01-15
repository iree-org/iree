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

#include "iree/compiler/Dialect/Compress/IR/CompressOps.h"
#include "iree/compiler/Dialect/Compress/Transforms/XlaPasses.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Compress {

namespace {

class OutlineQuantizableOpsPass
    : public FunctionPass<OutlineQuantizableOpsPass> {
  void runOnFunction() override;
};

// Outlines a single op into a `quantized` op with the original op placed
// into the body region.
// Multi-op fusions should be implemented as their own RewritePatterns, but
// the single op case is shared and implemented generically to reduce code
// duplication.
class OutlineSingleOpRewrite : public RewritePattern {
 public:
  using AuxPredicate = std::function<bool(Operation *op)>;

  OutlineSingleOpRewrite(StringRef rootName, QuantConfig config,
                         PatternBenefit benefit, AuxPredicate auxPredicate,
                         MLIRContext *context)
      : RewritePattern(rootName, {QuantRegionOp::getOperationName()}, benefit,
                       context),
        config(config),
        auxPredicate(std::move(auxPredicate)) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override;

 private:
  QuantConfig config;
  AuxPredicate auxPredicate;
};

}  // namespace

PatternMatchResult OutlineSingleOpRewrite::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  // Do not recursively match.
  if (op->getParentOfType<QuantRegionOp>()) {
    return matchFailure();
  }
  if (auxPredicate && !auxPredicate(op)) {
    return matchFailure();
  }

  llvm::SmallVector<Type, 4> resultTypes(op->getResultTypes());
  auto qOp = rewriter.create<QuantRegionOp>(op->getLoc(), resultTypes,
                                            op->getOperands(), config);
  auto *bodyBlock = new Block();
  qOp.body().push_back(bodyBlock);

  // Map inputs and clone.
  OpBuilder bodyBuilder(bodyBlock);
  BlockAndValueMapping mapping;
  for (auto operand : op->getOperands()) {
    auto blockArg = bodyBlock->addArgument(operand.getType());
    mapping.map(operand, blockArg);
  }
  bodyBuilder.clone(*op, mapping);

  // Add terminator.
  llvm::SmallVector<Value, 4> resultValues;
  for (auto oldResult : op->getResults()) {
    resultValues.push_back(mapping.lookupOrDefault(oldResult));
  }
  bodyBuilder.create<ReturnOp>(op->getLoc(), resultValues);

  rewriter.replaceOp(op, qOp.getResults());
  return matchSuccess();
}

// Inserts a rewrite to outline a single op.
template <typename OpTy>
static void insertSingleOpPattern(
    OwningRewritePatternList &patterns, MLIRContext &context,
    QuantConfig config, PatternBenefit benefit,
    OutlineSingleOpRewrite::AuxPredicate auxPredicate = nullptr) {
  patterns.insert<OutlineSingleOpRewrite>(OpTy::getOperationName(), config,
                                          benefit, auxPredicate, &context);
}

static ArrayAttr getI32Sequence(Builder &b, int count) {
  llvm::SmallVector<int32_t, 8> indices;
  for (int i = 0; i < count; ++i) {
    indices.push_back(i);
  }
  return b.getI32ArrayAttr(indices);
}

// Gets a QuantConfig for a quant_region based on a specific logical kernel.
static QuantConfig configForLogicalKernel(Builder &b, StringRef logicalKernel) {
  return QuantConfig::get(b.getStringAttr(logicalKernel), b.getArrayAttr({}),
                          b.getContext());
}

// Gets a QuantConfig for an opaque passthrough op where all operands/results
// must be the same type and can be transformed to either the expressed or
// storage type.
static QuantConfig configForOpaquePassthrough(Builder &b, int operandCount,
                                              int resultCount) {
  auto operandGroup = QuantOperandGroup::get(
      b.getStringAttr("EXPRESSED_OR_STORAGE"), getI32Sequence(b, operandCount),
      getI32Sequence(b, resultCount), b.getContext());
  return QuantConfig::get(b.getStringAttr("GENERIC"),
                          b.getArrayAttr({operandGroup}), b.getContext());
}

static QuantConfig configForSelect(Builder &b) {
  auto predOperandGroup = QuantOperandGroup::get(
      b.getStringAttr("PREDICATE"), b.getI32ArrayAttr({0}),
      b.getI32ArrayAttr({}), b.getContext());
  auto mainOperandGroup = QuantOperandGroup::get(
      b.getStringAttr("EXPRESSED_OR_STORAGE"), b.getI32ArrayAttr({1, 2}),
      b.getI32ArrayAttr({0}), b.getContext());
  return QuantConfig::get(b.getStringAttr("GENERIC"),
                          b.getArrayAttr({predOperandGroup, mainOperandGroup}),
                          b.getContext());
}

void OutlineQuantizableOpsPass::runOnFunction() {
  auto fn = getFunction();
  OwningRewritePatternList patterns;
  auto &context = getContext();

  // Binary arithmetic ops.
  // TODO(laurenzo): Expand this. Just enough now to prototype.
  Builder b(&getContext());
  const PatternBenefit DEFAULT = 0;
  auto noBroadcastDimsPred = [](Operation *op) {
    return !op->getAttr("broadcast_dimensions");
  };
  insertSingleOpPattern<xla_hlo::AddOp>(patterns, context,
                                        configForLogicalKernel(b, "BINARY_ADD"),
                                        DEFAULT, noBroadcastDimsPred);
  insertSingleOpPattern<xla_hlo::DivOp>(patterns, context,
                                        configForLogicalKernel(b, "BINARY_DIV"),
                                        DEFAULT, noBroadcastDimsPred);
  insertSingleOpPattern<xla_hlo::MulOp>(patterns, context,
                                        configForLogicalKernel(b, "BINARY_MUL"),
                                        DEFAULT, noBroadcastDimsPred);
  insertSingleOpPattern<xla_hlo::SubOp>(patterns, context,
                                        configForLogicalKernel(b, "BINARY_SUB"),
                                        DEFAULT, noBroadcastDimsPred);

  // Select op.
  insertSingleOpPattern<xla_hlo::SelectOp>(patterns, context,
                                           configForSelect(b), DEFAULT);

  // Passthrough ops.
  insertSingleOpPattern<xla_hlo::CopyOp>(
      patterns, context, configForOpaquePassthrough(b, 1, 1), DEFAULT);
  insertSingleOpPattern<xla_hlo::ReverseOp>(
      patterns, context, configForOpaquePassthrough(b, 1, 1), DEFAULT);
  // TODO(laurenzo): Transpose probably needs to not be strictly passthrough
  // since for some quantization schemes, the solver needs to know how
  // the tensor is permuted.
  insertSingleOpPattern<xla_hlo::TransposeOp>(
      patterns, context, configForOpaquePassthrough(b, 1, 1), DEFAULT);

  applyPatternsGreedily(fn, patterns);
}

std::unique_ptr<OpPassBase<FuncOp>> createXlaOutlineQuantizableOpsPass() {
  return std::make_unique<OutlineQuantizableOpsPass>();
}

static PassRegistration<OutlineQuantizableOpsPass> pass(
    "iree-compress-xla-outline-quantizable",
    "Outline quantizable XLA ops into quantized generic ops.");

}  // namespace Compress
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
