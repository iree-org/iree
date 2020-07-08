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

//===---- VectorToGPUPass.cpp - Pass for the final SPIR-V conversion
//-------===//
//
// This file implement a pass to convert vector dialect operations to GPU
// operations distributed across a subgroup.
//
//===----------------------------------------------------------------------===//
#include <memory>

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/CooperativeMatrixAnalysis.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {
// TODO(thomasraoux): Fetch this value from device properties.
static const int subgroupSize = 32;

struct ConvertVectorToGPUPass
    : public PassWrapper<ConvertVectorToGPUPass, OperationPass<FuncOp>> {
  void runOnOperation() override;
};

// Common class for all vector to GPU patterns.
template <typename OpTy>
class VectorToGPUPattern : public OpConversionPattern<OpTy> {
 public:
  VectorToGPUPattern<OpTy>(
      MLIRContext *context,
      const CooperativeMatrixAnalysis &cooperativeMatrixAnalysis)
      : OpConversionPattern<OpTy>::OpConversionPattern(context),
        cooperativeMatrixAnalysis(cooperativeMatrixAnalysis) {}

 protected:
  const CooperativeMatrixAnalysis &cooperativeMatrixAnalysis;
};

/// Converts unary and binary standard operations using new type.
template <typename StdOp>
class UnaryAndBinaryOpPattern final : public VectorToGPUPattern<StdOp> {
 public:
  using VectorToGPUPattern<StdOp>::VectorToGPUPattern;

  LogicalResult matchAndRewrite(
      StdOp operation, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (VectorToGPUPattern<StdOp>::cooperativeMatrixAnalysis
            .usesCooperativeMatrixType(operation))
      return failure();
    Value newOp = rewriter.create<StdOp>(
        operation.getLoc(), ValueRange(operands), ArrayRef<NamedAttribute>{});
    rewriter.replaceOp(operation, ValueRange(newOp));
    return success();
  }
};

class VectorTransferReadConversion
    : public VectorToGPUPattern<vector::TransferReadOp> {
 public:
  using VectorToGPUPattern<vector::TransferReadOp>::VectorToGPUPattern;

  LogicalResult matchAndRewrite(
      vector::TransferReadOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (cooperativeMatrixAnalysis.usesCooperativeMatrixType(op))
      return failure();
    // Only support identity map for now.
    if (!op.permutation_map().isIdentity()) return failure();
    if (op.getVectorType().getNumElements() != subgroupSize) return failure();
    // Only works for the case where one workgroups has only one subgroup.
    auto wgSize = spirv::lookupLocalWorkGroupSize(op);
    if (wgSize.getValue<int32_t>(0) != subgroupSize ||
        wgSize.getValue<int32_t>(1) != 1 || wgSize.getValue<int32_t>(2) != 1)
      return failure();
    auto loc = op.getLoc();
    SmallVector<Value, 4> indices(op.indices());
    // Use threadId.x as the subgroupInvocationId.
    // TODO(thomasraoux): Replace it once subgroup Ids are working.
    auto threadIndex = rewriter.create<gpu::ThreadIdOp>(
        loc, rewriter.getIndexType(), rewriter.getStringAttr("x"));
    Value index = rewriter.create<AddIOp>(loc, threadIndex, indices.back());
    indices.back() = index;
    Value newOp = rewriter.create<LoadOp>(loc, op.memref(), indices);
    rewriter.replaceOp(op, ValueRange(newOp));
    return success();
  }
};

class VectorTransferWriteConversion
    : public VectorToGPUPattern<vector::TransferWriteOp> {
 public:
  using VectorToGPUPattern<vector::TransferWriteOp>::VectorToGPUPattern;

  LogicalResult matchAndRewrite(
      vector::TransferWriteOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (cooperativeMatrixAnalysis.usesCooperativeMatrixType(op))
      return failure();
    if (!op.permutation_map().isIdentity()) return failure();
    if (op.getVectorType().getNumElements() != subgroupSize) return failure();
    auto loc = op.getLoc();
    SmallVector<Value, 4> indices(op.indices());
    auto ThreadIndex = rewriter.create<gpu::ThreadIdOp>(
        loc, rewriter.getIndexType(), rewriter.getStringAttr("x"));
    Value index = rewriter.create<AddIOp>(loc, ThreadIndex, indices.back());
    indices.back() = index;
    rewriter.create<StoreOp>(op.getLoc(), operands[0], operands[1], indices);
    rewriter.eraseOp(op);
    return success();
  }
};

class VectorToGPUConversionTarget : public ConversionTarget {
 public:
  using ConversionTarget::ConversionTarget;

 protected:
  // Standard operation are legal if they operate on scalars. We need to
  // legalize operations on vectors.
  bool isDynamicallyLegal(Operation *op) const override {
    auto isVectorType = [](Type t) { return t.isa<VectorType>(); };
    if (llvm::any_of(op->getResultTypes(), isVectorType) ||
        llvm::any_of(op->getOperandTypes(), isVectorType))
      return false;
    return true;
  }
};

void ConvertVectorToGPUPass::runOnOperation() {
  MLIRContext *context = &getContext();
  FuncOp funcOp = getOperation();
  auto &cooperativeMatrixAnalysis = getAnalysis<CooperativeMatrixAnalysis>();
  OwningRewritePatternList patterns;
  patterns.insert<UnaryAndBinaryOpPattern<AddFOp>, VectorTransferReadConversion,
                  VectorTransferWriteConversion>(context,
                                                 cooperativeMatrixAnalysis);
  populateParallelLoopToWorkgroupPatterns(context, patterns);
  std::unique_ptr<VectorToGPUConversionTarget> target =
      std::make_unique<VectorToGPUConversionTarget>(*context);
  target->addDynamicallyLegalDialect<StandardOpsDialect>();
  target->addIllegalOp<scf::ParallelOp>();
  target->addLegalOp<scf::YieldOp>();
  target->addLegalOp<scf::ForOp>();
  target->addLegalDialect<gpu::GPUDialect>();
  if (failed(applyPartialConversion(funcOp, *target, patterns)))
    return signalPassFailure();
}
}  // namespace

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//
std::unique_ptr<OperationPass<FuncOp>> createVectorToGPUPass() {
  return std::make_unique<ConvertVectorToGPUPass>();
}

static PassRegistration<ConvertVectorToGPUPass> pass(
    "iree-codegen-vector-to-gpu",
    "Convert vector dialect to gpu subgroup level GPU instructions");
}  // namespace iree_compiler
}  // namespace mlir
