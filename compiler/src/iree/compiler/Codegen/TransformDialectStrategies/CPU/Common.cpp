// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformDialectStrategies/CPU/Common.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMCPU/TransformExtensions/LLVMCPUExtensions.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/CPU/ReductionStrategy.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/AbstractReductionStrategy.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/Common.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

// TODO: significantly better namespacing.
using iree_compiler::cpu::CPUModel;
using iree_compiler::cpu::ReductionConfig;
using iree_compiler::cpu::ReductionStrategy;
using iree_compiler::IREE::transform_dialect::ApplyPatternsOpPatterns;
using iree_compiler::IREE::transform_dialect::ForallToWorkgroupOp;
using transform::ApplyTransferPermutationPatternsOp;
using transform::LowerContractionOp;
using transform::LowerMultiReductionOp;
using transform::LowerShapeCastOp;
using transform::LowerTransferOp;
using transform::LowerTransposeOp;
using transform::MatchOp;
using transform::SplitHandleOp;
using transform::SplitTransferFullPartialOp;
using transform::TransferToScfOp;
using transform_ext::AllDims;
using transform_ext::m_StructuredOp;
using transform_ext::NumEqualsTo;
using transform_ext::RegisterMatchCallbacksOp;
using transform_ext::ShapeKind;
using transform_ext::StructuredOpMatcher;
using vector::VectorContractLoweringAttr;

//===----------------------------------------------------------------------===//
// Mid-level problem-specific strategy builder APIs, follow MLIR-style builders.
//===----------------------------------------------------------------------===//

// TODO: better builders.
static Value buildDefaultVectorLoweringStrategy(
    ImplicitLocOpBuilder &b, Value funcH,
    const vector::LowerVectorsOptions &lowerVectorsOpts) {
  auto pdlOperation = pdl::OperationType::get(b.getContext());
  funcH = b.create<LowerContractionOp>(
      pdlOperation, funcH,
      /*loweringStrategy*/ lowerVectorsOpts.vectorContractLowering);
  funcH = b.create<ApplyTransferPermutationPatternsOp>(pdlOperation, funcH);
  funcH = b.create<LowerMultiReductionOp>(
      pdlOperation, funcH,
      /*loweringStrategy=*/lowerVectorsOpts.vectorMultiReductionLowering);
  funcH = b.create<SplitTransferFullPartialOp>(
      pdlOperation, funcH,
      /*splitTransferStrategy=*/lowerVectorsOpts.vectorTransferSplit);
  funcH = b.create<TransferToScfOp>(
      pdlOperation, funcH,
      /*maxTransferRank=*/1,
      /*fullUnroll=*/lowerVectorsOpts.unrollVectorTransfers);
  funcH = b.create<LowerTransferOp>(pdlOperation, funcH, /*maxTransferRank=*/1);
  funcH = b.create<LowerShapeCastOp>(pdlOperation, funcH);
  funcH = b.create<LowerTransposeOp>(
      pdlOperation, funcH,
      /*loweringStrategy=*/lowerVectorsOpts.vectorTransposeLowering,
      /*avx2LoweringStrategy=*/lowerVectorsOpts.transposeAVX2Lowering);
  return funcH;
}

/// Take care of the last common steps in a CPU strategy (i.e. vectorize,
/// bufferize and map to blocks).
/// Return the handles to the updated variant and the func::FuncOp ops under
/// the variant op.
std::pair<Value, Value> mlir::iree_compiler::cpu::buildCommonTrailingStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const vector::LowerVectorsOptions &lowerVectorsOpts) {
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());

  // Step N-5. Fold tensor.empty to avoid large allocations.
  ApplyPatternsOpPatterns configuration;
  configuration.foldTensorEmptyExtract = true;

  // Step N-4. Perform a pass of canonicalization + enabling after tiling.
  funcH = mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(
      b, configuration, funcH);
  funcH = iree_compiler::buildVectorize(b, funcH);

  // Step N-3. Perform a pass of canonicalization + enabling after vectorization
  // as well as hoisting subset operations such as vector.transfer_read/write.
  funcH = mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(
      b, configuration, funcH);
  iree_compiler::buildHoisting(b, funcH);

  // Step N-2. Bufferize and drop HAL descriptor from memref ops.
  variantH = iree_compiler::buildBufferize(b, variantH);

  // Step N-1. Post-bufferization mapping to blocks only.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  b.create<ForallToWorkgroupOp>(funcH);

  // Step N. Lower vectors.
  funcH = buildDefaultVectorLoweringStrategy(b, funcH, lowerVectorsOpts);
  return std::make_pair(variantH, funcH);
}

//===----------------------------------------------------------------------===//
// Higher-level problem-specific strategy creation APIs, these should favor
// user-friendliness.
//===----------------------------------------------------------------------===//

/// Placeholder to encode fixed reductions that should take finer-grained
/// precedence over other heuristics. In the future, this could be lifted to
/// e.g. `cpuModel` or higher up in some transform dialect database summary of
/// "known good things".
static FailureOr<ReductionConfig> applyKnownGoodReductionConfigurations(
    const transform_ext::MatchedReductionCaptures &captures,
    const CPUModel &cpuModel) {
  int64_t reductionSize = captures.reductionOpSizes.back();
  if (cpuModel.model == CPUModel::kDefaultCPU) {
    if (captures.reductionOutputElementalTypeBitWidth == 32) {
      if (reductionSize == 32) return ReductionConfig{/*vectorSize=*/32};
    }
  }
  return failure();
}

static ReductionConfig getReductionConfig(
    const transform_ext::MatchedReductionCaptures &captures,
    const CPUModel &cpuModel) {
  return ReductionConfig{16};
}

LogicalResult iree_compiler::cpu::matchAndSetReductionStrategy(
    func::FuncOp entryPoint, linalg::LinalgOp op, const CPUModel &cpuModel) {
  // 1. Match a reduction and surrounding ops.
  StructuredOpMatcher *reduction;
  transform_ext::MatchedReductionCaptures captures;
  transform_ext::MatcherContext matcherContext;
  makeReductionMatcher(matcherContext, reduction, captures);
  if (!matchPattern(op, *reduction)) return failure();

  // 2. Construct the configuration and the strategy builder.
  // TODO: Generalize along the HW axis.
  auto strategyBuilder = [&](ImplicitLocOpBuilder &b, Value variant) {
    ReductionConfig reductionConfig = getReductionConfig(captures, cpuModel);
    auto strategy =
        ReductionStrategy::create(op->getContext(), captures, reductionConfig);
    return buildReductionStrategy(b, variant, strategy);
  };

  // 3. Build strategy embedded into the IR.
  createTransformRegion(entryPoint, strategyBuilder);

  return success();
}
