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
#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/MatmulCodegenStrategy.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/CooperativeMatrixAnalysis.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace {
// TODO(thomasraoux): Fetch this value from device properties.
static const int subgroupSize = 32;

struct ConvertVectorToGPUPass
    : public PassWrapper<ConvertVectorToGPUPass, OperationPass<FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<AffineDialect,
                    gpu::GPUDialect,
                    scf::SCFDialect,
                    vector::VectorDialect>();
    // clang-format on
  }

  void runOnOperation() override;

 private:
  void tileAndVectorizeLinalgCopy(FuncOp funcOp, MLIRContext *context);
  void lowerVectorOps(FuncOp funcOp, MLIRContext *context);
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
    Value newOp =
        rewriter.create<StdOp>(operation.getLoc(), ValueRange(operands));
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
    rewriter.replaceOpWithNewOp<StoreOp>(op, operands[0], operands[1], indices);
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

void ConvertVectorToGPUPass::tileAndVectorizeLinalgCopy(FuncOp funcOp,
                                                        MLIRContext *context) {
  // 1. Tile linalg and distribute it on invocations.
  std::unique_ptr<ConversionTarget> target =
      std::make_unique<ConversionTarget>(*context);
  target->addDynamicallyLegalOp<linalg::CopyOp>([&](linalg::CopyOp copy) {
    return !(hasMarker(copy, getCopyToWorkgroupMemoryMarker()));
  });
  target->markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  OwningRewritePatternList tileAndDistributePattern;
  populateLinalgTileAndDistributePatterns(context, tileAndDistributePattern);
  if (failed(applyPartialConversion(funcOp, *target,
                                    std::move(tileAndDistributePattern)))) {
    return signalPassFailure();
  }

  // 2. Canonicalize the IR generated by tiling.
  OwningRewritePatternList canonicalizePatterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  canonicalizePatterns.insert<AffineMinCanonicalizationPattern,
                              linalg::AffineMinSCFCanonicalizationPattern>(
      context);
  applyPatternsAndFoldGreedily(funcOp, std::move(canonicalizePatterns));

  // 3. Vectorize the tiled linalg to be able to map it to load/store vector.
  OwningRewritePatternList vectorizationPatterns;
  vectorizationPatterns
      .insert<linalg::LinalgVectorizationPattern<linalg::CopyOp>>(
          context, linalg::LinalgMarker(
                       Identifier::get(getVectorizeMarker(), context), {}));
  applyPatternsAndFoldGreedily(funcOp, std::move(vectorizationPatterns));
}

// Convert vector transfer_read to a load if possible. This is the case only if
// the element type of the memref matches the element type we want to load.
class VectorTransferReadToLoad
    : public OpRewritePattern<vector::TransferReadOp> {
 public:
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getVectorType().getNumElements() != 1 ||
        op.getMemRefType().getElementType() !=
            op.getVectorType().getElementType()) {
      return failure();
    }
    auto loc = op.getLoc();
    Value newOp = rewriter.create<LoadOp>(loc, op.memref(), op.indices());
    newOp =
        rewriter.create<vector::BroadcastOp>(loc, op.getVectorType(), newOp);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// Convert vector transfer_write to a store if possible. This is the case only
// if the element type of the memref matches the element type we want to store.
class VectorTransferWriteToStore
    : public OpRewritePattern<vector::TransferWriteOp> {
 public:
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getVectorType().getNumElements() != 1 ||
        op.getMemRefType().getElementType() !=
            op.getVectorType().getElementType()) {
      return failure();
    }
    auto loc = op.getLoc();
    SmallVector<int64_t, 2> zero(op.getVectorType().getRank(), 0);
    Value scalarValue =
        rewriter.create<vector::ExtractOp>(loc, op.vector(), zero);
    rewriter.create<StoreOp>(loc, scalarValue, op.memref(), op.indices());
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower vector contract to a single scalar or vector mulf+addf. Insert casts to
// convert from 2D vector to 1D vector or scalar.
class VectorContractLowering : public OpRewritePattern<vector::ContractionOp> {
 public:
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    auto iteratorTypes = op.iterator_types().getValue();
    if (iteratorTypes.size() != 3 || !isParallelIterator(iteratorTypes[0]) ||
        !isParallelIterator(iteratorTypes[1]) ||
        !isReductionIterator(iteratorTypes[2]) ||
        !isRowMajorMatmul(op.indexing_maps())) {
      return failure();
    }
    if (op.getLhsType().getNumElements() != 1) return failure();
    unsigned vecSize = op.getAccType().cast<VectorType>().getNumElements();
    if (!(vecSize >= 1 && vecSize <= 4)) return failure();
    auto loc = op.getLoc();
    VectorType vecType = VectorType::get(
        vecSize, op.getResultType().cast<VectorType>().getElementType());
    std::array<int64_t, 2> zero = {0, 0};
    Value lhs = rewriter.create<vector::ExtractOp>(loc, op.lhs(), zero);
    Value rhs, acc;
    if (vecSize == 1) {
      rhs = rewriter.create<vector::ExtractOp>(loc, op.rhs(), zero);
      acc = rewriter.create<vector::ExtractOp>(loc, op.acc(), zero);
    } else {
      lhs = rewriter.create<vector::BroadcastOp>(loc, vecType, lhs);
      rhs = rewriter.create<vector::ShapeCastOp>(loc, vecType, op.rhs());
      acc = rewriter.create<vector::ShapeCastOp>(loc, vecType, op.acc());
    }
    Value newOp = rewriter.create<MulFOp>(loc, lhs, rhs);
    newOp = rewriter.create<AddFOp>(loc, newOp, acc);
    if (vecSize == 1)
      newOp =
          rewriter.create<vector::BroadcastOp>(loc, op.getResultType(), newOp);
    else
      newOp =
          rewriter.create<vector::ShapeCastOp>(loc, op.getResultType(), newOp);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// Lower ExtractStridedSliceOp to an ExtractOp instruction that can be natively
// converted to SPIR-V. Add a BroadcastOp to keep the type consistent, we expect
// the Broadcast to be removed by canonicalization.
class ExtractStridedLowering
    : public OpRewritePattern<vector::ExtractStridedSliceOp> {
 public:
  using OpRewritePattern<vector::ExtractStridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    // Only handle cases extracting a degenerated vector so that we can generate
    // an extractOp with scalar destination.
    if (op.getResult().getType().cast<VectorType>().getNumElements() != 1)
      return failure();
    auto loc = op.getLoc();
    SmallVector<int64_t, 4> offsets = llvm::to_vector<4>(
        llvm::map_range(op.offsets().getAsRange<IntegerAttr>(),
                        [](IntegerAttr attr) { return attr.getInt(); }));
    offsets.resize(op.getVectorType().getRank(), 0);
    Value newOp = rewriter.create<vector::ExtractOp>(loc, op.vector(), offsets);
    newOp = rewriter.create<vector::BroadcastOp>(loc, op.getResult().getType(),
                                                 newOp);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// Lower vector ops to instructions that can be later converted to SPIR-V.
void ConvertVectorToGPUPass::lowerVectorOps(FuncOp funcOp,
                                            MLIRContext *context) {
  OwningRewritePatternList patterns;
  patterns.insert<VectorContractLowering, VectorTransferReadToLoad,
                  VectorTransferWriteToStore, ExtractStridedLowering>(context);
  applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

void ConvertVectorToGPUPass::runOnOperation() {
  MLIRContext *context = &getContext();
  FuncOp funcOp = getOperation();
  tileAndVectorizeLinalgCopy(funcOp, context);

  lowerVectorOps(funcOp, context);

  auto &cooperativeMatrixAnalysis = getAnalysis<CooperativeMatrixAnalysis>();
  OwningRewritePatternList patterns;
  patterns.insert<UnaryAndBinaryOpPattern<AddFOp>, VectorTransferReadConversion,
                  VectorTransferWriteConversion>(context,
                                                 cooperativeMatrixAnalysis);
  std::unique_ptr<VectorToGPUConversionTarget> target =
      std::make_unique<VectorToGPUConversionTarget>(*context);
  target->addDynamicallyLegalDialect<StandardOpsDialect>();
  target->addIllegalOp<scf::ParallelOp>();
  target->addLegalOp<scf::YieldOp>();
  target->addLegalOp<scf::ForOp>();
  target->addLegalDialect<gpu::GPUDialect>();
  if (failed(applyPartialConversion(funcOp, *target, std::move(patterns))))
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
