// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-linalg-to-llvm-tile-and-vectorize"

namespace mlir {
namespace iree_compiler {

namespace {
// Could just be linalg::TilingPattern with a ContractionOpInterface filter, but
// that is always templated on an op.
struct TileWorkgroups : public linalg::LinalgBaseTilingPattern {
  using Base = linalg::LinalgBaseTilingPattern;
  TileWorkgroups(MLIRContext *context, linalg::LinalgTilingOptions options,
                 linalg::LinalgTransformationFilter marker)
      : LinalgBaseTilingPattern(context, options, marker) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto contractionOp = dyn_cast<linalg::ContractionOpInterface>(op);
    if (!contractionOp) return failure();

    linalg::TiledLinalgOp tiledLinalgOp;
    if (failed(Base::matchAndRewriteBase(op, rewriter, tiledLinalgOp)) ||
        !tiledLinalgOp.tensorResults.empty()) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

namespace {
struct LLVMCPUVectorizationPass
    : public LLVMCPUVectorizationBase<LLVMCPUVectorizationPass> {
  LLVMCPUVectorizationPass(bool vectorize = true) : lowerToVectors(vectorize) {}
  LLVMCPUVectorizationPass(const LLVMCPUVectorizationPass &pass) {
    lowerToVectors = pass.lowerToVectors;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, AffineDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override;

 private:
  /// TODO(ravishankarm): Option to not generate any `vector.` instructions. The
  /// VMVX backend uses the same lowering as the CPU pass but there is no
  /// lowering of these `vector.` operations to scalar code. So as a WAR do the
  /// same tiling scheme but avoid generating vector instructions. When VMVX can
  /// handle vector instructions, drop this options.
  bool lowerToVectors;

  Option<bool> enableVectorContractToAarch64Asm{
      *this, "vector-contract-to-aarch64-asm",
      llvm::cl::desc("Enable promoting wokgroup memory to full tiles allocated "
                     "on the stack."),
      llvm::cl::init(false)};
};
}  // namespace

void LLVMCPUVectorizationPass::runOnOperation() {
  auto funcOp = getOperation();
  MLIRContext *context = &getContext();

  // Workgroup first level of tiling.
  {
    // First level of tiling patterns. (workgroups memory)
    RewritePatternSet l1patterns(context);
    l1patterns.insert<TileWorkgroups>(
        context,
        linalg::LinalgTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder &builder, Operation *op) -> SmallVector<Value, 4> {
              return getTileSizes(builder, op,
                                  static_cast<unsigned>(TilingLevel::L1Tiles));
            }),
        linalg::LinalgTransformationFilter(
            ArrayRef<Identifier>{},
            Identifier::get(getWorkgroupL1TileMarker(), context)));

    (void)applyPatternsAndFoldGreedily(funcOp, std::move(l1patterns));
  }

  // Second level of tiling. (workgroups memory -> vectors)
  {
    RewritePatternSet l2patterns(context);
    l2patterns.insert<TileWorkgroups>(
        context,
        linalg::LinalgTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder &builder, Operation *op) -> SmallVector<Value, 4> {
              return getTileSizes(
                  builder, op, static_cast<unsigned>(TilingLevel::VectorTiles));
            }),
        linalg::LinalgTransformationFilter(
            Identifier::get(getWorkgroupL1TileMarker(), context),
            Identifier::get(getVectorizeMarker(), context)));

    (void)applyPatternsAndFoldGreedily(funcOp, std::move(l2patterns));
  }

  // Apply canonicalization.
  {
    RewritePatternSet canonicalizationPatterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    populateAffineMinCanonicalizationPattern(canonicalizationPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(canonicalizationPatterns)))) {
      return signalPassFailure();
    }
  }

  if (!lowerToVectors) {
    return;
  }

  // Op specific conversion.
  {
    RewritePatternSet vectorizeOpsPattenrs(context);
    populateLinalgToVectorVectorizeMMT4dPatterns(context, vectorizeOpsPattenrs);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(vectorizeOpsPattenrs)))) {
      return signalPassFailure();
    }
  }

  // Apply vectorization patterns.
  {
    RewritePatternSet vectorizationPatterns(context);
    linalg::insertVectorizationPatterns<linalg::ContractionOpInterface,
                                        linalg::CopyOp, linalg::FillOp>(
        vectorizationPatterns, linalg::LinalgVectorizationOptions(),
        linalg::LinalgTransformationFilter(
            Identifier::get(getVectorizeMarker(), context)));
    vector::populateVectorTransferPermutationMapLoweringPatterns(
        vectorizationPatterns);
    vector::populateVectorReductionToContractPatterns(vectorizationPatterns);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(vectorizationPatterns));
  }

  {
    // Fold consumer add ops into the contraction op itself.
    RewritePatternSet canonicalizationPatterns(context);
    vector::ContractionOp::getCanonicalizationPatterns(canonicalizationPatterns,
                                                       context);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(canonicalizationPatterns));
  }

  if (enableVectorContractToAarch64Asm) {
    RewritePatternSet vectorToAArch64AsmPatterns(context);
    populateVectorContractToAArch64InlineAsm(vectorToAArch64AsmPatterns,
                                             context);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(vectorToAArch64AsmPatterns));
  }

  // Apply vector specific operation lowering.
  {
    vector::VectorTransformsOptions vectorTransformsOptions =
        vector::VectorTransformsOptions().setVectorTransformsOptions(
            vector::VectorContractLowering::OuterProduct);
    RewritePatternSet vectorContractLoweringPatterns(context);
    vectorContractLoweringPatterns.insert<
        vector::ContractionOpToOuterProductOpLowering,
        vector::ContractionOpToMatmulOpLowering, vector::ContractionOpLowering>(
        vectorTransformsOptions, context);
    vector::populateVectorTransferPermutationMapLoweringPatterns(
        vectorContractLoweringPatterns);
    (void)applyPatternsAndFoldGreedily(
        funcOp, std::move(vectorContractLoweringPatterns));
  }

  // Hosit hierarchical tiling indexing and other loop invariant transfer
  // ops computation.

  // Programmatic controlled lowering of vector.transfer only.
  {
    VectorTransferToSCFOptions vectorToSCFOptions =
        VectorTransferToSCFOptions().enableFullUnroll();
    RewritePatternSet vectorToLoopsPatterns(context);
    populateVectorToSCFConversionPatterns(vectorToLoopsPatterns,
                                          vectorToSCFOptions);
    // Hosit hierarchical tiling indexing and other loop invariant transfer
    // ops computation.
    linalg::hoistRedundantVectorTransfers(funcOp);

    memref::populateFoldSubViewOpPatterns(vectorToLoopsPatterns);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(vectorToLoopsPatterns));
  }
}

std::unique_ptr<OperationPass<FuncOp>> createLLVMCPUVectorizationPass(
    bool lowerToVectors) {
  return std::make_unique<LLVMCPUVectorizationPass>(lowerToVectors);
}

}  // namespace iree_compiler
}  // namespace mlir
