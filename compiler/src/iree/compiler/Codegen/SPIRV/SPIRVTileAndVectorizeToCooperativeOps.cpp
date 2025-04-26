// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVTileAndVectorizeToCooperativeOps.cpp --------------------------===//
//
// This pass tiles Linalg ops with buffer semantics to subgroups and vectorizes
// them into vector ops suitable for lowering to SPIR-V cooperative ops.
//
//===----------------------------------------------------------------------===//

#include <algorithm>

#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-tile-and-vectorize-to-cooperative-ops"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_SPIRVTILETOCOOPERATIVEOPSPASS
#define GEN_PASS_DEF_SPIRVVECTORIZETOCOOPERATIVEOPSPASS
#include "iree/compiler/Codegen/SPIRV/Passes.h.inc"

namespace {

void debugPrint(Operation *op, const char *message) {
  LLVM_DEBUG({
    llvm::dbgs() << "//--- " << message << " ---//\n";
    op->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
}

//===----------------------------------------------------------------------===//
// Cooperative matrix shape utilities
//===----------------------------------------------------------------------===//

/// Gets the chosen hardware cooperative op size attached to the given `op`
/// as CodeGen lowering configuration.
static SmallVector<int64_t> getTargetCooperativeOpSize(linalg::LinalgOp op) {
  return getTileSizes(op, 3); // For native vector sizes
}

constexpr char coopMatTypeAttrName[] = "iree.spirv.coopmatrix.type";
constexpr char coopMatShapeAttrName[] = "iree.spirv.coopmatrix.shape";

/// Sets the chosen cooperative matrix type/shape for CodeGen onto the
/// the given `funcOp`.
void setSPIRVCooperativeMatrixInfo(mlir::FunctionOpInterface funcOp,
                                   linalg::LinalgOp rootOp,
                                   ArrayRef<int64_t> shape) {
  Builder b(funcOp.getContext());
  funcOp->setAttr(coopMatShapeAttrName, b.getDenseI64ArrayAttr(shape));
  auto inputType = cast<ShapedType>(rootOp.getDpsInputs().front().getType());
  auto outputType = cast<ShapedType>(rootOp.getDpsInits().front().getType());
  auto elementTypes = b.getTypeArrayAttr(
      {inputType.getElementType(), outputType.getElementType()});
  funcOp->setAttr(coopMatTypeAttrName, elementTypes);
}

/// Returns the chosen cooperative matrix shape for CodeGen from the given
/// `funcOp`. Returns an empty ArrayRef if cannot query.
ArrayRef<int64_t>
getSPIRVCooperativeMatrixShape(mlir::FunctionOpInterface funcOp) {
  auto attr = funcOp->getAttrOfType<DenseI64ArrayAttr>(coopMatShapeAttrName);
  if (!attr)
    return {};
  return attr.asArrayRef();
}

//===----------------------------------------------------------------------===//
// Subgroup tiling patterns
//===----------------------------------------------------------------------===//

/// Deduces required subgroup counts along all workgroup tiled dimensions.
///
/// `op` should be an operation with a `lowering_config` attribute to specify
/// tiling sizes for the workgroup and subgroup.
static SmallVector<int64_t> deduceSubgroupCounts(linalg::LinalgOp op) {
  SmallVector<int64_t> workgroupTileSizes = getTileSizes(op, 0);
  SmallVector<int64_t> subgroupTileSizes = getTileSizes(op, 1);
  assert(workgroupTileSizes.size() == subgroupTileSizes.size());

  SmallVector<int64_t> subgroupCounts;
  for (int i = 0, e = workgroupTileSizes.size(); i < e; ++i) {
    if (subgroupTileSizes[i] == 0)
      continue;
    if (linalg::isReductionIterator(op.getIteratorTypesArray()[i]))
      continue;
    assert(workgroupTileSizes[i] % subgroupTileSizes[i] == 0);
    subgroupCounts.push_back(workgroupTileSizes[i] / subgroupTileSizes[i]);
  }
  LLVM_DEBUG(llvm::dbgs() << "deduced subgroup counts (X, Y, Z) = "
                          << llvm::interleaved_array(subgroupCounts) << "\n");
  return subgroupCounts;
}

/// Tiles Linalg ops with workgroup markers to subgroups.
static LogicalResult tileToSubgroup(mlir::FunctionOpInterface funcOp,
                                    ArrayRef<int64_t> subgroupCounts,
                                    const unsigned subgroupSize,
                                    ArrayRef<int64_t> subgroupTileSizes) {
  MLIRContext *context = funcOp.getContext();

  auto getSubgroupProcInfoFn =
      [subgroupCounts, subgroupSize](OpBuilder &builder, Location loc,
                                     ArrayRef<Range> parallelLoopRanges) {
        auto counts = llvm::to_vector<3>(subgroupCounts);
        // `getSubgroupIdsAndCounts` assumes we follow GPU (X, Y, Z) order.
        std::reverse(counts.begin(), counts.end());
        return getSubgroupIdsAndCounts(builder, loc, subgroupSize,
                                       parallelLoopRanges.size(), counts);
      };

  linalg::LinalgLoopDistributionOptions distributionOptions;
  distributionOptions.procInfo = getSubgroupProcInfoFn;

  auto setTileSizesFn = [subgroupTileSizes](OpBuilder &builder, Operation *op) {
    SmallVector<int64_t> tileSizes = llvm::to_vector(subgroupTileSizes);
    // Only consider parallel dimensions for tiling and distribution. Reduction
    // dimension distribution needs synchronization. We'll use vector unroll
    // later to "tile" along reduction dimensions.
    tileSizes.resize(
        std::min(cast<linalg::LinalgOp>(op).getNumParallelLoops(), 3u));
    return llvm::map_to_vector(tileSizes, [&](int64_t v) -> Value {
      return builder.create<arith::ConstantIndexOp>(op->getLoc(), v);
    });
  };
  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(setTileSizesFn)
                           .setDistributionOptions(distributionOptions);

  LinalgTransformationFilter filter(
      {StringAttr::get(context, getWorkgroupKTiledMarker()),
       StringAttr::get(context, getWorkgroupMemoryMarker())},
      StringAttr::get(context, getVectorizeMarker()));
  return distributeLinalgOpsWithFilter(funcOp, tilingOptions, filter);
}

//===----------------------------------------------------------------------===//
// Vectorization patterns
//===----------------------------------------------------------------------===//

template <typename ExtOpTy>
std::optional<SmallVector<int64_t>>
getExtOpVectorShape(ExtOpTy op, ArrayRef<int64_t> nativeShape) {
  auto insert =
      op.getOperand().template getDefiningOp<vector::InsertStridedSliceOp>();
  if (!insert)
    return std::nullopt;

  VectorType sliceType = insert.getSourceVectorType();
  for (Operation *users : op->getUsers()) {
    auto extract = dyn_cast<vector::ExtractStridedSliceOp>(users);
    if (!extract)
      return std::nullopt;
    auto vecType = llvm::cast<VectorType>(extract.getResult().getType());
    if (!llvm::equal(sliceType.getShape(), vecType.getShape()))
      return std::nullopt;
  }

  return llvm::to_vector(sliceType.getShape());
}

/// Returns vector shape matching native cooperative op sizes for unrolling
/// high-D vectors.
std::optional<SmallVector<int64_t>>
getCooperativeOpVectorShape(Operation *op, ArrayRef<int64_t> nativeShape) {
  // Unroll vector.contract ops according to native cooperative matrix size.
  if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
    return llvm::to_vector(nativeShape);
  }

  // Unroll elementwise ops according to native cooperative matrix size.
  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1) {
    if (auto vecType = llvm::dyn_cast<VectorType>(op->getResultTypes()[0]))
      return llvm::to_vector(nativeShape.drop_back()); // Drop K dim size
  }

  // Unrolling vector.contract generates vector.{insert|extract}_strided_slice
  // ops for the vector transfer ops associated with the original contract op.
  // We can use those to figure out how to unroll transfer ops accordingly
  // to match the native cooperative op sizes.
  //
  // A better way might be to inspect the SSA value chain to figure out how the
  // transfer ops are used (e.g., for cooperative matrix A/B/C matrix) and use
  // the corresponding cooperative matrix configuration.

  if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
    auto insert =
        writeOp.getVector().getDefiningOp<vector::InsertStridedSliceOp>();
    if (insert) {
      return llvm::to_vector(insert.getSourceVectorType().getShape());
    }

    // There can exist vector.transfer_write for initializing output. Unroll
    // them to native shape. Native shape is for ([B, ]M, N, K), here we only
    // need ([B, ]M, N).
    return llvm::to_vector(nativeShape.drop_back());
  }

  if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    auto sourceOp = op;
    if (op->hasOneUse()) {
      auto user = *op->user_begin();
      if (isa<arith::ExtUIOp>(user) || isa<arith::ExtSIOp>(user))
        sourceOp = user;
    }

    VectorType sliceType;
    for (Operation *users : sourceOp->getUsers()) {
      auto extract = dyn_cast<vector::ExtractStridedSliceOp>(users);
      if (!extract)
        return std::nullopt;
      auto vecType = llvm::cast<VectorType>(extract.getResult().getType());
      if (sliceType && sliceType != vecType)
        return std::nullopt;
      sliceType = vecType;
    }
    return llvm::to_vector(sliceType.getShape());
  }

  if (auto extOp = dyn_cast<arith::ExtSIOp>(op))
    return getExtOpVectorShape<arith::ExtSIOp>(extOp, nativeShape);
  if (auto extOp = dyn_cast<arith::ExtUIOp>(op))
    return getExtOpVectorShape<arith::ExtUIOp>(extOp, nativeShape);

  return std::nullopt;
}

/// Adds patterns to unroll vector ops to SPIR-V native vector size.
void populateVectorUnrollPatterns(ArrayRef<int64_t> cooperativeOpSize,
                                  RewritePatternSet &patterns) {
  auto getShapeFn = [cooperativeOpSize](Operation *op) {
    return getCooperativeOpVectorShape(op, cooperativeOpSize);
  };
  auto options = vector::UnrollVectorOptions().setNativeShapeFn(getShapeFn);
  vector::populateVectorUnrollPatterns(patterns, options);
}

/// Fuses vector.transpose into consumer vector.contract.
///
/// This is a workaround for SPIR-V backend limitations. SPIR-V vetorization
/// pass relies on unrolling to reduce instructions to a vector size we can
/// convert to SPIR-V. When vectorization creates transpose those block
/// unrolling and result in large vector we currently cannot lower. For now we
/// always merge the transpose into the contract op so that it can be unrolled.
//
// TODO(thomasraoux): Make transpose work with the current unrolling mechanism
// or replace unrolling.
class CombineContractTranspose final
    : public OpRewritePattern<vector::ContractionOp> {
public:
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    // Perform lhs + rhs transpositions to conform to matmul row-major
    // semantics. Bail out if the contraction cannot be put in this form.
    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();
    bool foundTranspose = false;
    std::array<Value, 3> sources = {op.getLhs(), op.getRhs(), op.getAcc()};
    SmallVector<AffineMap> newMaps;
    SmallVector<Value> newSources;
    for (auto [srcIdx, source] : llvm::enumerate(sources)) {
      auto map = op.getIndexingMapsArray()[srcIdx];
      auto transposeOp = source.getDefiningOp<vector::TransposeOp>();
      if (!transposeOp) {
        newSources.push_back(source);
        newMaps.push_back(map);
        continue;
      }
      ArrayRef<int64_t> perm = transposeOp.getPermutation();
      SmallVector<AffineExpr> exprs(perm.size());
      for (auto [remapIdx, remap] : llvm::enumerate(perm)) {
        exprs[remap] = map.getResult(remapIdx);
      }
      newMaps.push_back(
          AffineMap::get(map.getNumDims(), map.getNumSymbols(), exprs, ctx));
      newSources.push_back(transposeOp.getVector());
      foundTranspose = true;
    }
    if (!foundTranspose)
      return failure();

    Value res = rewriter.create<vector::ContractionOp>(
        loc, newSources[0], newSources[1], newSources[2],
        rewriter.getAffineMapArrayAttr(newMaps), op.getIteratorTypes());
    rewriter.replaceOp(op, res);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

class SPIRVTileToCooperativeOpsPass final
    : public impl::SPIRVTileToCooperativeOpsPassBase<
          SPIRVTileToCooperativeOpsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, IREE::GPU::IREEGPUDialect,
                    linalg::LinalgDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();

    // First we need to discover the CodeGen lowering configuration. It was
    // decided earlier and attached to a linalg op as an attribute.

    linalg::LinalgOp rootOp;
    funcOp.walk([&](linalg::LinalgOp linalgOp) {
      if (isMatmulOrBatchMatmul(linalgOp) && getLoweringConfig(linalgOp)) {
        rootOp = linalgOp;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!rootOp) {
      funcOp.emitError("expected lowering confg on a (batch) matmul op");
      return signalPassFailure();
    }

    // Transfer the cooperative matrix shape to an attribute on the export op,
    // given that after tiling and vectorization we won't have the root Linalg
    // op anymore.
    SmallVector<int64_t> cooperativeOpSize = getTargetCooperativeOpSize(rootOp);
    setSPIRVCooperativeMatrixInfo(funcOp, rootOp, cooperativeOpSize);

    SmallVector<int64_t> subgroupCounts = deduceSubgroupCounts(rootOp);

    // Then tile and distribute to subgroups.

    {
      std::optional<int> subgroupSize = getGPUSubgroupSize(funcOp);
      if (!subgroupSize) {
        funcOp.emitError("failed to query subgroup size");
        return signalPassFailure();
      }

      SmallVector<int64_t> subgroupTileSizes = getTileSizes(rootOp, 1);
      if (failed(tileToSubgroup(funcOp, subgroupCounts, *subgroupSize,
                                subgroupTileSizes))) {
        return signalPassFailure();
      }

      RewritePatternSet canonicalizationPatterns =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      SmallVector<int64_t> numWorkgroups = getStaticNumWorkgroups(funcOp);
      populateFoldAffineMinInDistributedLoopsPatterns(canonicalizationPatterns,
                                                      numWorkgroups);
      if (failed(applyPatternsGreedily(funcOp,
                                       std::move(canonicalizationPatterns)))) {
        return signalPassFailure();
      }
    }

    debugPrint(funcOp, "after tiling to subgroups");
  }
};

class SPIRVVectorizeToCooperativeOpsPass final
    : public impl::SPIRVVectorizeToCooperativeOpsPassBase<
          SPIRVVectorizeToCooperativeOpsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, linalg::LinalgDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();

    // First discover the chosen cooperative matrix shape. It was decided
    // earlier and attached to the export op as an attribute.
    ArrayRef<int64_t> cooperativeOpSize =
        getSPIRVCooperativeMatrixShape(funcOp);
    if (cooperativeOpSize.empty()) {
      funcOp->emitError(
          "expected attribute for chosen cooperative matrix shape");
      return signalPassFailure();
    }

    // Now prepare and unroll to native cooperative sizes.

    {
      RewritePatternSet patterns(context);
      vector::ContractionOp::getCanonicalizationPatterns(patterns, context);
      populateCombineVectorTransferReadBroadcastPatterns(patterns);
      populatePrepareVectorToMMAPatterns(patterns,
                                         /*useNvGPU=*/false);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    debugPrint(funcOp, "after preparing vector ops");

    {
      RewritePatternSet patterns(context);
      populateVectorUnrollPatterns(cooperativeOpSize, patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    debugPrint(funcOp, "after unrolling vector ops");

    // When using cooperative matrix we don't want to lower the contract,
    // instead we want to merge contract and transpose so that they can be
    // converted to cooperative matrix matmul op.
    {
      RewritePatternSet patterns(context);
      patterns.add<CombineContractTranspose>(context);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    debugPrint(funcOp, "after combining transpose ops");
  }
};

} // namespace
} // namespace mlir::iree_compiler
