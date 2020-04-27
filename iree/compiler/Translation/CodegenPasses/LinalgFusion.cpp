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

//===- LinalgFusion.cpp - Fuse Linalg operations within a dispatch region--===//
//
// Fuses all Linalg operations with a dispatch region into a single linalg
// operation.
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Translation/CodegenPasses/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
namespace mlir {
namespace iree_compiler {

namespace {

// Pattern to fuse linalg generic op with its producer if the producer is
// another linalg.generic op or a std.constant op of return type beign a 0-D
// tensor.
// TODO(ravishankarm): Generalize this to handle more valid fusion cases.
struct IREEFuseGenericTensorOps : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override;
};

/// Fuses linalg operations on tensors in dispatch function. For now does only
/// producer consumer fusion.
struct IREELinalgFusionPass
    : public PassWrapper<IREELinalgFusionPass, FunctionPass> {
  void runOnFunction() override;
};
}  // namespace

/// Returns true if a std.constant can be fused with a linalg.generic op, when
/// the result of the constant is the operand of the generic op at `consumerIdx`
static bool isConstantFusibleWithLinalgOp(ConstantOp producer,
                                          linalg::LinalgOp consumer,
                                          unsigned consumerIdx) {
  // Verify that the consumer is op on tensors.
  if (!consumer.hasTensorSemantics()) return false;
  // For now this only fusing constants that produce a zero-dim tensor.
  auto producerResultType =
      producer.getResult().getType().dyn_cast<RankedTensorType>();
  if (!producerResultType || producerResultType.getRank() != 0) return false;
  // Only handle generic ops.
  auto consumerOp = dyn_cast<linalg::GenericOp>(consumer.getOperation());
  if (!consumerOp || producer.getResult() != consumerOp.getOperand(consumerIdx))
    return false;
  return true;
}

/// Fuses scalar constant op with linalg::generic op when the former is a
/// producer and the latter is a consumer. This fuses when the producer has a
/// tensor of rank-0 return type since they are not expressible as linalg
/// generic ops.
// TODO(ravishankarm): Move this into Linalg fusion.
static Optional<linalg::LinalgOp> fuseGenericOpWithConstantScalar(
    OpBuilder &b, ConstantOp producer, linalg::GenericOp consumer,
    unsigned consumerIdx, OperationFolder *folder = nullptr) {
  // Check conditions for fusion.
  if (!isConstantFusibleWithLinalgOp(
          producer, cast<linalg::LinalgOp>(consumer.getOperation()),
          consumerIdx))
    return {};

  // Generate the fused operation.
  SmallVector<Value, 2> fusedOperandsList(consumer.getOperands());
  fusedOperandsList.erase(fusedOperandsList.begin() + consumerIdx);
  SmallVector<Attribute, 2> fusedIndexingMapAttrs(
      consumer.indexing_maps().begin(), consumer.indexing_maps().end());
  fusedIndexingMapAttrs.erase(
      std::next(fusedIndexingMapAttrs.begin(), consumerIdx));
  auto loc = UnknownLoc::get(b.getContext());
  auto fusedLinalgOp = b.create<linalg::GenericOp>(
      loc, consumer.getResultTypes(), fusedOperandsList,
      b.getI64IntegerAttr(consumer.getNumInputs() - 1),
      b.getI64IntegerAttr(consumer.getNumOutputs()),
      b.getArrayAttr(fusedIndexingMapAttrs), consumer.iterator_types(),
      /*doc=*/nullptr,
      /*library_call=*/nullptr);

  // Build the body of the fused operation. All arguments are the same, except
  // for the one that is replaced.
  auto &fusedRegion = fusedLinalgOp.region();
  Block &consumerBlock = consumer.region().front();
  Block *fusedBlock = new Block();
  fusedRegion.push_back(fusedBlock);
  BlockAndValueMapping mapper;
  for (auto consumerArg : llvm::enumerate(consumerBlock.getArguments())) {
    if (consumerArg.index() == consumerIdx) continue;
    mapper.map(consumerArg.value(),
               fusedBlock->addArgument(consumerArg.value().getType()));
  }
  OpBuilder::InsertionGuard blockInsertionGuard(b);
  b.setInsertionPointToEnd(fusedBlock);
  auto constantOp = b.create<ConstantOp>(
      loc, producer.value().cast<DenseElementsAttr>().getSplatValue());
  mapper.map(consumerBlock.getArgument(consumerIdx), constantOp.getResult());
  // Add operations from the consumer into this block.
  for (auto &op : consumerBlock.getOperations())
    fusedBlock->push_back(op.clone(mapper));
  return cast<linalg::LinalgOp>(fusedLinalgOp.getOperation());
}

LogicalResult IREEFuseGenericTensorOps::matchAndRewrite(
    linalg::GenericOp op, PatternRewriter &rewriter) const {
  if (!op.hasTensorSemantics()) return failure();
  for (auto operand : llvm::enumerate(op.getOperation()->getOperands())) {
    auto producer = operand.value().getDefiningOp();
    if (!producer || producer->getNumResults() != 1) continue;
    bool hasSingleUse = producer->getResult(0).hasOneUse();
    Optional<linalg::LinalgOp> fusedOp;
    if (auto producerOp = dyn_cast<ConstantOp>(producer)) {
      fusedOp = fuseGenericOpWithConstantScalar(rewriter, producerOp, op,
                                                operand.index());
    }
    if (!fusedOp) continue;
    rewriter.replaceOp(op, fusedOp.getValue().getOperation()->getResults());
    if (hasSingleUse) rewriter.eraseOp(producer);
    return success();
  }
  return failure();
}

// TODO(ataei): We should instead use xla_hlo -> std legalization pass before
// fusion. But this requires chaining reduction lowering and many other steps.
class HLOConstantConverter : public OpRewritePattern<xla_hlo::ConstOp> {
 public:
  using OpRewritePattern<xla_hlo::ConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xla_hlo::ConstOp op,
                                PatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto stdConstOp = rewriter.create<ConstantOp>(loc, op.value());
    rewriter.replaceOp(op, stdConstOp.getResult());
    return success();
  }
};

void IREELinalgFusionPass::runOnFunction() {
  OwningRewritePatternList patterns;
  Operation *op = getOperation();
  populateLinalgTensorOpsFusionPatterns(op->getContext(), patterns);
  patterns.insert<IREEFuseGenericTensorOps, HLOConstantConverter>(
      op->getContext());
  applyPatternsAndFoldGreedily(op->getRegions(), patterns);
}

std::unique_ptr<OperationPass<FuncOp>> createLinalgOnTensorsFusionPass() {
  return std::make_unique<IREELinalgFusionPass>();
}

static PassRegistration<IREELinalgFusionPass> pass(
    "iree-linalg-fusion", "Fuse Linalg operations within a dispatch region");
}  // namespace iree_compiler
}  // namespace mlir
