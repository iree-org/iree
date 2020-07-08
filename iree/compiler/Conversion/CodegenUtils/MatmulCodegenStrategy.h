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

#ifndef MLIR_EDGE_BENCHMARKS_STRATEGIES_MATMULCODEGENSTRATEGIES_H_
#define MLIR_EDGE_BENCHMARKS_STRATEGIES_MATMULCODEGENSTRATEGIES_H_

#include <functional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

class FuncOp;

/// Abstract Transformation class applied in a sequence that also handles state
/// through markers.
struct Transformation {
  virtual ~Transformation() = default;
  virtual OwningRewritePatternList buildRewritePatterns(
      MLIRContext *context, linalg::LinalgMarker m) = 0;
  linalg::LinalgMarker marker;
};

template <typename VectorOpType>
struct UnrollVector : public Transformation {
  explicit UnrollVector(ArrayRef<int64_t> targetShape)
      : targetShape(targetShape) {}

  OwningRewritePatternList buildRewritePatterns(
      MLIRContext *ctx, linalg::LinalgMarker m) override {
    OwningRewritePatternList vectorUnrollPatterns;
    vectorUnrollPatterns.insert<vector::UnrollVectorPattern<VectorOpType>>(
        targetShape, ctx);
    vector::populateVectorToVectorCanonicalizationPatterns(vectorUnrollPatterns,
                                                           ctx);
    vector::populateVectorToVectorTransformationPatterns(vectorUnrollPatterns,
                                                         ctx);
    return vectorUnrollPatterns;
  }

 private:
  ArrayRef<int64_t> targetShape;
};

/// Promotion transformation enqueues a particular stage-1 pattern for
/// `Tile<LinalgOpType>`with the appropriate `options`.
// TODO: variadic LinalgOpTypes.
template <typename LinalgOpType>
struct Tile : public Transformation {
  explicit Tile(linalg::LinalgTilingOptions options) : options(options) {}

  OwningRewritePatternList buildRewritePatterns(
      MLIRContext *context, linalg::LinalgMarker m) override {
    OwningRewritePatternList tilingPatterns;
    tilingPatterns.insert<linalg::LinalgTilingPattern<LinalgOpType>>(
        context, options, m);
    return tilingPatterns;
  }

 private:
  linalg::LinalgTilingOptions options;
};

/// Promotion transformation enqueues a particular stage-1 pattern for
/// `Promote<LinalgOpType>`with the appropriate `options`.
// TODO: variadic LinalgOpTypes.
template <typename LinalgOpType>
struct Promote : public Transformation {
  explicit Promote(linalg::LinalgPromotionOptions options) : options(options) {}

  OwningRewritePatternList buildRewritePatterns(
      MLIRContext *context, linalg::LinalgMarker m) override {
    OwningRewritePatternList promotionPatterns;
    promotionPatterns.insert<linalg::LinalgPromotionPattern<LinalgOpType>>(
        context, options, m);
    return promotionPatterns;
  }

 private:
  linalg::LinalgPromotionOptions options;
};

/// Vectorization transformation enqueues a particular stage-1 pattern for
/// `LinalgVectorizationPattern<LinalgOpType>` as well as copy to vector
/// transfer rewrite forwarding patterns.
// TODO: variadic LinalgOpTypes.
template <typename LinalgOpType>
struct Vectorize : public Transformation {
  OwningRewritePatternList buildRewritePatterns(
      MLIRContext *context, linalg::LinalgMarker m) override {
    OwningRewritePatternList vectorizationPatterns;
    // FillOp may interfere with forwarding patterns atm, so we bump up the
    // priority of LinalgCopyVTRForwardingPattern /
    // LinalgCopyVTWForwardingPattern.
    vectorizationPatterns
        .insert<linalg::LinalgVectorizationPattern<LinalgOpType>>(context, m);
    vectorizationPatterns.insert<linalg::LinalgCopyVTRForwardingPattern,
                                 linalg::LinalgCopyVTWForwardingPattern>(
        context, /*benefit=*/2);
    return vectorizationPatterns;
  }
};

/// Matmul-specific strategy object controls how a linalg.matmul is
/// progressively lowered.
/// The strategy uses a 3-level staged patterns strategy which allows ordering
/// transformations by using the Linalg `applyStagedPatterns` function, where:
///   1. The first stage consists of the successive `tile`, `promote` and
///   `vectorize` patterns, applied sequentially.
///   2. The second stage consists of common local canonicalization patterns
///   that are applied eagerly after each stage-1 pattern.
///   3. the third stage consists of more global transformation, also applied
///   eagerly, after all stage-2 patterns. Such more global transformations
struct MatmulCodegenStrategy {
  /// Append a pattern to add a level of tiling for `LinalgOpType` with tiling
  /// `options`.
  template <typename LinalgOpType>
  MatmulCodegenStrategy &tile(linalg::LinalgTilingOptions options) {
    transformationSequence.emplace_back(new Tile<LinalgOpType>(options));
    return *this;
  }
  /// Conditionally append a pattern to add a level of tiling for `LinalgOpType`
  /// with tiling `options`.
  template <typename LinalgOpType>
  MatmulCodegenStrategy &tileIf(bool b, linalg::LinalgTilingOptions options) {
    return b ? tile<LinalgOpType>(options) : *this;
  }
  /// Append a pattern to add a level of promotion for `LinalgOpType` with
  /// promotion `options`.
  template <typename LinalgOpType>
  MatmulCodegenStrategy &promote(linalg::LinalgPromotionOptions options) {
    transformationSequence.emplace_back(new Promote<LinalgOpType>(options));
    return *this;
  }
  /// Conditionally append a pattern to add a level of promotion for
  /// `LinalgOpType` with promotion `options`.
  template <typename LinalgOpType>
  MatmulCodegenStrategy &promoteIf(bool b,
                                   linalg::LinalgPromotionOptions options) {
    return b ? promote<LinalgOpType>(options) : *this;
    return *this;
  }
  /// Append a pattern to rewrite `LinalgOpType` as a vector operation.
  template <typename LinalgOpType>
  MatmulCodegenStrategy &vectorize() {
    transformationSequence.emplace_back(new Vectorize<LinalgOpType>());
    return *this;
  }
  /// Conditionally append a pattern to rewrite `LinalgOpType` as a vector
  /// operation.
  template <typename LinalgOpType>
  MatmulCodegenStrategy &vectorizeIf(bool b) {
    return b ? vectorize<LinalgOpType>() : *this;
    return *this;
  }
  /// Configure the post staged-patterns late vector transformations.
  MatmulCodegenStrategy &setVectorTransformsOptions(
      vector::VectorTransformsOptions options) {
    vectorTransformsOptions = options;
    return *this;
  }
  /// Configure the post staged-patterns late vector.transfer to scf conversion.
  MatmulCodegenStrategy &setVectorTransferToSCFOptions(
      VectorTransferToSCFOptions options) {
    vectorToSCFOptions = options;
    return *this;
  }
  /// Configure the post staged-patterns late vector.transfer to scf conversion.
  MatmulCodegenStrategy &setHoistInvariantCode(bool b) {
    hoistInvariantCode = b;
    return *this;
  }

  /// Apply the transformation patterns in sequence with cleanup transformations
  /// interleaved.
  void transform(FuncOp func) const;

  /// Set a function applying the lowering strategy. Different target need to
  /// use different lowering.
  MatmulCodegenStrategy &setLoweringFunction(std::function<void(FuncOp)> f) {
    lowering = f;
    return *this;
  }

  /// Append a pattern to unroll a `VectorOpType` to smaller vector operations.
  template <typename VectorOpType>
  MatmulCodegenStrategy &unrollVector(ArrayRef<int64_t> targetShape) {
    transformationSequence.emplace_back(
        new UnrollVector<VectorOpType>(targetShape));
    return *this;
  }
  /// Conditionally append a pattern to rewrite `LinalgOpType` as a vector
  /// operation.
  template <typename VectorOpType>
  MatmulCodegenStrategy &unrollVectorIf(bool b, ArrayRef<int64_t> targetShape) {
    return b ? unrollVector<VectorOpType>(targetShape) : *this;
    return *this;
  }

  // Enable default lowering strategy for CPU.
  MatmulCodegenStrategy &setDefaultCPULowering();

 private:
  LogicalResult postPatternTransforms(Operation *func) const;

  std::function<void(FuncOp)> lowering = nullptr;
  bool hoistInvariantCode = false;
  vector::VectorTransformsOptions vectorTransformsOptions;
  VectorTransferToSCFOptions vectorToSCFOptions;
  SmallVector<std::unique_ptr<Transformation>, 4> transformationSequence;
};

}  // namespace mlir

#endif  // MLIR_EDGE_BENCHMARKS_STRATEGIES_MATMULCODEGENSTRATEGIES_H_
