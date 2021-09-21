// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVVectorize.cpp -------------------------------------------------===//
//
// This pass vectorizes Linalg ops with buffer semantics.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-vectorize"

namespace mlir {
namespace iree_compiler {
namespace {

/// Returns the cooperative matrix (M, N, K) sizes that are supported by the
/// target environment and match the given parameters.
static Optional<SmallVector<int64_t, 4>> getCooperativeMatmulSubgroupSize(
    spirv::ResourceLimitsAttr resourceLimits, Type lhsType, Type rhsType,
    Type initType, Type resultType) {
  auto range = resourceLimits.cooperative_matrix_properties_nv()
                   .getAsRange<spirv::CooperativeMatrixPropertiesNVAttr>();
  for (auto coopMatmulProperties : range) {
    if (coopMatmulProperties.a_type().getValue() == lhsType &&
        coopMatmulProperties.b_type().getValue() == rhsType &&
        coopMatmulProperties.c_type().getValue() == initType &&
        coopMatmulProperties.result_type().getValue() == resultType &&
        coopMatmulProperties.scope().getValue() == spirv::Scope::Subgroup) {
      return SmallVector<int64_t, 4>{
          coopMatmulProperties.m_size().getValue().getSExtValue(),
          coopMatmulProperties.n_size().getValue().getSExtValue(),
          coopMatmulProperties.k_size().getValue().getSExtValue()};
    }
  }
  return llvm::None;
}

/// Returns true if the target environment attached to `op`'s ancestor op
/// supports cooperative matrix.
bool useCooperativeMatrix(Operation *op) {
  auto attr = getSPIRVTargetEnvAttr(op);
  if (!attr) return false;

  auto targetEnv = spirv::TargetEnv(attr);
  return targetEnv.allows(spirv::Capability::CooperativeMatrixNV) &&
         targetEnv.allows(spirv::Extension::SPV_NV_cooperative_matrix);
}

Optional<SmallVector<int64_t, 4>> getSPIRVNativeVectorSize(Operation *op) {
  bool useCoopMatrix = useCooperativeMatrix(op);

  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1) {
    if (auto vecType = op->getResultTypes()[0].dyn_cast<VectorType>()) {
      // Use 4-element vectors for elementwise ops.
      SmallVector<int64_t, 4> nativeSize(vecType.getRank(), 1);
      nativeSize.back() = 4;
      return nativeSize;
    }
  } else if (auto vtOp = dyn_cast<VectorTransferOpInterface>(op)) {
    if (useCoopMatrix) {
      if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
        // Unroll cooperative martrix store based on the size of the contract.
        auto insert =
            writeOp.vector().getDefiningOp<vector::InsertStridedSliceOp>();
        if (!insert) return llvm::None;
        ArrayRef<int64_t> shape = insert.getSourceVectorType().getShape();
        return SmallVector<int64_t, 4>(shape.begin(), shape.end());
      } else if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
        // Unroll cooperative martrix load based on the size of the contract.
        VectorType dstVec;
        for (Operation *users : op->getUsers()) {
          auto extract = dyn_cast<vector::ExtractStridedSliceOp>(users);
          if (!extract) return llvm::None;
          auto vecType = extract.getResult().getType().cast<VectorType>();
          if (dstVec && dstVec != vecType) return llvm::None;
          dstVec = vecType;
        }
        return SmallVector<int64_t, 4>(dstVec.getShape().begin(),
                                       dstVec.getShape().end());
      }
    } else {
      auto rank = vtOp.getVectorType().getRank();
      SmallVector<int64_t, 4> nativeSize(rank, 1);
      for (auto dim : llvm::enumerate(vtOp.permutation_map().getResults())) {
        if (auto dimExpr = dim.value().dyn_cast<AffineDimExpr>()) {
          if (dimExpr.getPosition() == vtOp.permutation_map().getNumDims() - 1)
            nativeSize[dim.index()] = 4;
        }
      }
      return nativeSize;
    }
  } else if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
    if (useCoopMatrix) {
      auto targetEnv = spirv::TargetEnv(getSPIRVTargetEnvAttr(op));
      return getCooperativeMatmulSubgroupSize(
          targetEnv.getResourceLimits(),
          contractOp.getLhsType().getElementType(),
          contractOp.getRhsType().getElementType(),
          contractOp.getAccType().cast<VectorType>().getElementType(),
          contractOp.getResultType().cast<VectorType>().getElementType());
    } else {
      unsigned lastParalleldim = 0;
      for (auto it : llvm::enumerate(contractOp.iterator_types())) {
        if (isParallelIterator(it.value())) lastParalleldim = it.index();
      }
      SmallVector<int64_t, 4> nativeSize(contractOp.iterator_types().size(), 1);
      nativeSize[lastParalleldim] = 4;
      // Map to vec4 fma operations.
      return nativeSize;
    }
  }
  return llvm::None;
}

/// Add patterns to vectorize Linalg ops with vectorization marker.
void populateVectorizationPatterns(MLIRContext *context,
                                   RewritePatternSet &patterns) {
  linalg::insertVectorizationPatterns<linalg::FillOp, linalg::GenericOp,
                                      linalg::ContractionOpInterface>(
      patterns, linalg::LinalgVectorizationOptions(),
      linalg::LinalgTransformationFilter(
          Identifier::get(getVectorizeMarker(), context)));
}

/// Adds patterns to unroll vector ops to SPIR-V native vector size.
void populateVectorUnrollPatterns(MLIRContext *context,
                                  RewritePatternSet &patterns) {
  vector::populateVectorUnrollPatterns(
      patterns,
      vector::UnrollVectorOptions().setNativeShapeFn(getSPIRVNativeVectorSize));
}

/// Workaround SPIR-V backend limitations. SPIR-V vetorization pass relies on
/// unrolling to reduce instructions to a vector size we can convert to SPIR-V.
/// When vectorization creates transpose those block unrolling and result in
/// large vector we currently cannot lower. For now we always merge the
/// transpose into the contract op so that it can be unrolled.
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
    std::array<Value, 3> sources = {op.lhs(), op.rhs(), op.acc()};
    SmallVector<AffineMap> newMaps;
    SmallVector<Value> newSources;
    for (auto source : llvm::enumerate(sources)) {
      auto map =
          op.indexing_maps()[source.index()].cast<AffineMapAttr>().getValue();
      auto tranposeOp = source.value().getDefiningOp<vector::TransposeOp>();
      if (!tranposeOp) {
        newSources.push_back(source.value());
        newMaps.push_back(map);
        continue;
      }
      SmallVector<int64_t, 3> perm;
      tranposeOp.getTransp(perm);
      SmallVector<AffineExpr> exprs(perm.size());
      for (auto remap : llvm::enumerate(perm)) {
        exprs[remap.value()] = map.getResult(remap.index());
      }
      newMaps.push_back(
          AffineMap::get(map.getNumDims(), map.getNumSymbols(), exprs, ctx));
      newSources.push_back(tranposeOp.vector());
      foundTranspose = true;
    }
    if (!foundTranspose) return failure();

    Value res = rewriter.create<vector::ContractionOp>(
        loc, newSources[0], newSources[1], newSources[2],
        rewriter.getAffineMapArrayAttr(newMaps), op.iterator_types());
    rewriter.replaceOp(op, res);
    return success();
  }
};

/// Vectorizes Linalg ops on buffer semantics.
class SPIRVVectorizePass : public SPIRVVectorizeBase<SPIRVVectorizePass> {
 public:
  SPIRVVectorizePass() = default;
  SPIRVVectorizePass(const SPIRVVectorizePass &pass) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FuncOp funcOp = getOperation();

    auto entryPointOp = getEntryPoint(funcOp);
    if (!entryPointOp) return;

    {
      RewritePatternSet vectorizationPatterns(&getContext());
      populateVectorizationPatterns(context, vectorizationPatterns);
      populateLinalgToVectorVectorizeConvPatterns(context,
                                                  vectorizationPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(vectorizationPatterns));

      LLVM_DEBUG({
        llvm::dbgs() << "--- After vectorization ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }

    // TODO: This should be a folding of Add into Contract in core but while
    // they live in different dialects, it is not possible without unnatural
    // dependencies.
    funcOp.walk([&](Operation *op) {
      if (auto contract = canonicalizeContractionAdd(op))
        op->replaceAllUsesWith(contract);
    });

    {
      RewritePatternSet vectorUnrollPatterns(funcOp.getContext());
      populateVectorUnrollPatterns(funcOp.getContext(), vectorUnrollPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(vectorUnrollPatterns));
    }

    {
      linalg::hoistRedundantVectorTransfers(funcOp);

      LLVM_DEBUG({
        llvm::dbgs() << "--- After hoisting vector transfers ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }

    {
      RewritePatternSet canonicalizationPatterns(funcOp.getContext());
      vector::populateVectorTransferPermutationMapLoweringPatterns(
          canonicalizationPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(canonicalizationPatterns));

      if (useCooperativeMatrix(funcOp)) {
        // When using cooperative matrix we don't want to lower the contract,
        // instead we want to merge contract and transpose so that they can be
        // converted to cooperative matrix matmul op.
        // TODO(thomasraoux): remove that once we support cooperative matrix
        // lowering in MLIR core.
        RewritePatternSet combineTransposePatterns(funcOp.getContext());
        combineTransposePatterns.add<CombineContractTranspose>(
            funcOp.getContext());
        (void)applyPatternsAndFoldGreedily(funcOp,
                                           std::move(combineTransposePatterns));
      } else {
        RewritePatternSet contractLoweringPatterns(funcOp.getContext());
        vector::populateVectorContractLoweringPatterns(
            contractLoweringPatterns,
            vector::VectorTransformsOptions().setVectorTransformsOptions(
                vector::VectorContractLowering::OuterProduct));
        (void)applyPatternsAndFoldGreedily(funcOp,
                                           std::move(contractLoweringPatterns));
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After unrolling vector ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createSPIRVVectorizePass() {
  return std::make_unique<SPIRVVectorizePass>();
}

}  // namespace iree_compiler
}  // namespace mlir
