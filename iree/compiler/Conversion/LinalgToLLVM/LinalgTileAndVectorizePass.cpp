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

#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/TransformUtils.h"
#include "iree/compiler/Conversion/LinalgToLLVM/KernelDispatch.h"
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRV.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

namespace {
// Could just be linalg::TilingPattern with a ContractionOpInterface filter, but
// that is always templated on an op.
struct TileWorkgroups : public linalg::LinalgBaseTilingPattern {
  using Base = linalg::LinalgBaseTilingPattern;
  TileWorkgroups(linalg::LinalgTilingOptions options,
                 linalg::LinalgTransformationFilter marker)
      : LinalgBaseTilingPattern(options, marker) {}
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
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, AffineDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }
  void runOnFunction() override;
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
    ArrayRef<Value> boundingSubViewSize, OperationFolder *folder) {
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
        linalg::LinalgTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder &builder,
               Operation *operation) -> SmallVector<Value, 4> {
              return TileSizeFn::get<TilingLevel::Level1Tiles>(builder,
                                                               operation);
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
        linalg::LinalgTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder &builder,
               Operation *operation) -> SmallVector<Value, 4> {
              return TileSizeFn::get<TilingLevel::Level2Tiles>(builder,
                                                               operation);
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
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(vectorContractLoweringPatterns)))) {
      return signalPassFailure();
    }
  }

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

    // TODO(ataei): Move this to common vector dialect patterns.
    populateStdLegalizationPatternsForSPIRVLowering(vectorToLoopsPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(vectorToLoopsPatterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<FunctionPass> createLinalgTileAndVectorizeWorkgroupsPass() {
  return std::make_unique<TileAndVectorizeWorkgroups>();
}

static PassRegistration<TileAndVectorizeWorkgroups> pass(
    "iree-codegen-linalg-to-llvm-workgroups-vectorization-pass",
    "Tile and vectorize llvm workgroups",
    [] { return std::make_unique<TileAndVectorizeWorkgroups>(); });

}  // namespace iree_compiler
}  // namespace mlir
