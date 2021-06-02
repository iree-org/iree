// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/TransformUtils.h"
#include "iree/compiler/Conversion/Common/Transforms.h"
#include "iree/compiler/Conversion/LinalgToLLVM/KernelDispatch.h"
#include "iree/compiler/Conversion/VectorToLLVM/Passes.h"
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRV.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-linalg-to-llvm-tile-and-vectorize"

namespace mlir {
namespace iree_compiler {

// TODO(ataei): Use pass options instead of global llvm flags.
static llvm::cl::opt<bool> clEnablePromoteWorkgroupToFullTiles(
    "iree-codegen-llvm-promote-workgroup-to-full-tiles",
    llvm::cl::desc("Enable promoting wokgroup memory to full tiles allocated "
                   "on the stack."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableVectorContractAarch64AsmTo(
    "iree-codegen-llvm-vector-contract-to-aarch64-asm",
    llvm::cl::desc("Enable promoting wokgroup memory to full tiles allocated "
                   "on the stack."),
    llvm::cl::init(false));

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
struct TileAndVectorizeWorkgroups
    : public PassWrapper<TileAndVectorizeWorkgroups, FunctionPass> {
  TileAndVectorizeWorkgroups(bool vectorize = true)
      : lowerToVectors(vectorize) {}
  TileAndVectorizeWorkgroups(const TileAndVectorizeWorkgroups &pass) {
    lowerToVectors = pass.lowerToVectors;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, AffineDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }
  void runOnFunction() override;

 private:
  /// TODO(ravishankarm): Option to not generate any `vector.` instructions. The
  /// VMVX backend uses the same lowering as the CPU pass but there is no
  /// lowering of these `vector.` operations to scalar code. So as a WAR do the
  /// same tiling scheme but avoid generating vector instructions. When VMVX can
  /// handle vector instructions, drop this options.
  bool lowerToVectors;
};
}  // namespace

namespace {
/// Pattern to promote all matmul operands to memory.
struct PromoteMatmulSubviewsPattern
    : public linalg::LinalgPromotionPattern<linalg::MatmulOp> {
  PromoteMatmulSubviewsPattern(MLIRContext *context,
                               linalg::LinalgPromotionOptions options,
                               linalg::LinalgTransformationFilter marker,
                               PatternBenefit benefit = 1)
      : linalg::LinalgPromotionPattern<linalg::MatmulOp>(
            context,
            options.setOperandsToPromote({0, 1, 2})
                .setUseFullTileBuffersByDefault(true),
            marker, benefit) {}
};
}  // namespace

namespace {

// TODO(ataei): Refactor this into a common utility with LinalgToSPIRV.
Optional<Value> allocateWorkgroupMemoryOnStack(
    OpBuilder &b, memref::SubViewOp subview,
    ArrayRef<Value> boundingSubViewSize, DataLayout &layout) {
  // Allocate the memory into the entry block of the parent FuncOp. This better
  // aligns with the semantics of this memory which is available at the entry of
  // the function.
  OpBuilder::InsertionGuard guard(b);
  FuncOp funcOp = subview->getParentOfType<FuncOp>();
  if (!funcOp) {
    subview.emitError("expected op to be within std.func");
    return llvm::None;
  }
  b.setInsertionPointToStart(&(*funcOp.getBody().begin()));
  // The bounding subview size is expected to be constant. This specified the
  // shape of the allocation.
  SmallVector<int64_t, 2> shape = llvm::to_vector<2>(
      llvm::map_range(boundingSubViewSize, [](Value v) -> int64_t {
        APInt value;
        if (matchPattern(v, m_ConstantInt(&value))) return value.getSExtValue();
        return -1;
      }));
  if (llvm::any_of(shape, [](int64_t v) { return v == -1; })) return {};
  MemRefType allocType =
      MemRefType::get(shape, subview.getType().getElementType(), {});
  Value buffer = b.create<memref::AllocaOp>(subview.getLoc(), allocType);
  return buffer;
}

LogicalResult deallocateWorkgroupMemory(OpBuilder &b, Value buffer) {
  MemRefType bufferType = buffer.getType().dyn_cast<MemRefType>();
  if (!bufferType) return failure();
  return success();
}

}  // namespace

void TileAndVectorizeWorkgroups::runOnFunction() {
  auto funcOp = getOperation();
  MLIRContext *context = &getContext();
  // Promotes workgroups subviews to a full-tile allocated on the stack.
  if (clEnablePromoteWorkgroupToFullTiles) {
    OwningRewritePatternList promotionPatterns(&getContext());
    promotionPatterns.insert<PromoteMatmulSubviewsPattern>(
        context,
        linalg::LinalgPromotionOptions().setAllocationDeallocationFns(
            allocateWorkgroupMemoryOnStack, deallocateWorkgroupMemory),
        linalg::LinalgTransformationFilter(
            Identifier::get(getWorkgroupMarker(), context),
            Identifier::get(getWorkgroupMemoryMarker(), context)));
    memref::ViewOp::getCanonicalizationPatterns(promotionPatterns, context);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(promotionPatterns));
  }

  // Workgroup first level of tiling.
  {
    // First level of tiling patterns. (workgroups memory)
    OwningRewritePatternList l1patterns(&getContext());
    l1patterns.insert<TileWorkgroups>(
        context,
        linalg::LinalgTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder &builder,
               Operation *operation) -> SmallVector<Value, 4> {
              return getTileSizes(
                  builder, operation,
                  static_cast<unsigned>(TilingLevel::Level1Tiles));
            }),
        linalg::LinalgTransformationFilter(
            Identifier::get(clEnablePromoteWorkgroupToFullTiles
                                ? getWorkgroupMemoryMarker()
                                : getWorkgroupMarker(),
                            context),
            Identifier::get(getWorkgroupL1TileMarker(), context)));

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(l1patterns)))) {
      return signalPassFailure();
    }
  }

  // Second level of tiling. (workgroups memory -> vectors)
  {
    OwningRewritePatternList l2patterns(&getContext());
    l2patterns.insert<TileWorkgroups>(
        context,
        linalg::LinalgTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder &builder,
               Operation *operation) -> SmallVector<Value, 4> {
              return getTileSizes(
                  builder, operation,
                  static_cast<unsigned>(TilingLevel::Level2Tiles));
            }),
        linalg::LinalgTransformationFilter(
            Identifier::get(getWorkgroupL1TileMarker(), context),
            Identifier::get(getVectorizeMarker(), context)));

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(l2patterns)))) {
      return signalPassFailure();
    }
  }

  // Apply canonicalization.
  {
    OwningRewritePatternList canonicalizationPatterns(&getContext());
    canonicalizationPatterns.insert<AffineMinCanonicalizationPattern>(context);
    AffineApplyOp::getCanonicalizationPatterns(canonicalizationPatterns,
                                               context);
    AffineMinOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
    memref::SubViewOp::getCanonicalizationPatterns(canonicalizationPatterns,
                                                   context);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(canonicalizationPatterns)))) {
      return signalPassFailure();
    }
  }

  if (!lowerToVectors) {
    return;
  }

  // Apply vectorization patterns.
  {
    OwningRewritePatternList vectorizationPatterns(&getContext());
    linalg::insertVectorizationPatterns<linalg::ContractionOpInterface,
                                        linalg::CopyOp, linalg::FillOp>(
        vectorizationPatterns, linalg::LinalgVectorizationOptions(),
        linalg::LinalgTransformationFilter(
            Identifier::get(getVectorizeMarker(), context)));
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(vectorizationPatterns)))) {
      return signalPassFailure();
    }
  }

  // TODO: This should be a folding of Add into Contract in core but while they
  // live in different dialects, it is not possible without unnatural
  // dependencies.
  funcOp.walk([&](Operation *op) {
    if (auto contract = canonicalizeContractionAdd(op))
      op->replaceAllUsesWith(contract);
  });

  if (clEnableVectorContractAarch64AsmTo) {
    OwningRewritePatternList vectorToAArch64AsmPatterns(context);
    populateVectorContractToAArch64InlineAsm(vectorToAArch64AsmPatterns,
                                             context);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(vectorToAArch64AsmPatterns)))) {
      return signalPassFailure();
    }
  }

  // Apply vector specific operation lowering.
  {
    vector::VectorTransformsOptions vectorTransformsOptions =
        vector::VectorTransformsOptions().setVectorTransformsOptions(
            vector::VectorContractLowering::OuterProduct);
    OwningRewritePatternList vectorContractLoweringPatterns(&getContext());
    vectorContractLoweringPatterns
        .insert<ContractionOpToOuterProductOpLowering,
                ContractionOpToMatmulOpLowering, ContractionOpLowering>(
            vectorTransformsOptions, context);
    vector::populateVectorTransferLoweringPatterns(
        vectorContractLoweringPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(vectorContractLoweringPatterns)))) {
      return signalPassFailure();
    }
  }

  // Hosit hierarchical tiling indexing and other loop invariant transfer
  // ops computation.

  // Programmatic controlled lowering of vector.transfer only.
  {
    VectorTransferToSCFOptions vectorToSCFOptions =
        VectorTransferToSCFOptions().setUnroll(true);
    OwningRewritePatternList vectorToLoopsPatterns(&getContext());
    populateVectorToSCFConversionPatterns(vectorToLoopsPatterns,
                                          vectorToSCFOptions);
    // Hosit hierarchical tiling indexing and other loop invariant transfer
    // ops computation.
    linalg::hoistRedundantVectorTransfers(funcOp);

    memref::populateFoldSubViewOpPatterns(vectorToLoopsPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(vectorToLoopsPatterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<FunctionPass> createLinalgTileAndVectorizeWorkgroupsPass(
    bool lowerToVectors) {
  return std::make_unique<TileAndVectorizeWorkgroups>(lowerToVectors);
}

static PassRegistration<TileAndVectorizeWorkgroups> pass(
    "iree-codegen-linalg-to-llvm-workgroups-vectorization-pass",
    "Tile and vectorize llvm workgroups",
    [] { return std::make_unique<TileAndVectorizeWorkgroups>(); });

}  // namespace iree_compiler
}  // namespace mlir
