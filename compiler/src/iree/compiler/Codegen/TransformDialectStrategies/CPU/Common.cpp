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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

// TODO: significantly better namespacing.
using iree_compiler::cpu::ReductionStrategy;
using iree_compiler::IREE::transform_dialect::ForeachThreadToWorkgroupOp;
using transform::LowerVectorsOp;
using transform::MatchOp;
using transform::SplitHandlesOp;
using transform_ext::AllDims;
using transform_ext::m_StructuredOp;
using transform_ext::NumEqualsTo;
using transform_ext::RegisterMatchCallbacksOp;
using transform_ext::ShapeKind;
using transform_ext::StructuredOpMatcher;

//===----------------------------------------------------------------------===//
// Higher-level problem-specific strategy creation APIs, these should favor
// user-friendliness.
//===----------------------------------------------------------------------===//

/// Take care of the last common steps in a CPU strategy (i.e. vectorize,
/// bufferize and map to blocks).
/// Return the handles to the updated variant and the func::FuncOp ops under
/// the variant op.
std::pair<Value, Value> mlir::iree_compiler::cpu::buildCommonTrailingStrategy(
    ImplicitLocOpBuilder &b, Value variantH) {
  // Step N-2. Bufferize and drop HAL descriptor from memref ops.
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = iree_compiler::buildVectorize(b, funcH);
  variantH = iree_compiler::buildBufferize(b, variantH);

  // Step N-1. Post-bufferization mapping to blocks only.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = b.create<ForeachThreadToWorkgroupOp>(funcH);
  auto pdlOperation = pdl::OperationType::get(b.getContext());

  // Step N. Lower vectors.
  // TODO: Control the lowering to vectors.
  funcH = b.create<LowerVectorsOp>(pdlOperation, funcH);
  return std::make_pair(variantH, funcH);
}

LogicalResult iree_compiler::cpu::matchAndSetReductionStrategy(
    func::FuncOp entryPoint, linalg::LinalgOp op) {
  // 1. Match a reduction and surrounding ops.
  StructuredOpMatcher reduction, fill, leading, trailing;
  transform_ext::MatchedReductionCaptures captures;
  makeReductionMatcher(reduction, fill, leading, trailing, captures);
  if (!matchPattern(op, reduction)) return failure();

  // 2. Construct the configuration and the strategy builder.
  // TODO: Generalize along the HW axis.
  auto strategyBuilder = [&](ImplicitLocOpBuilder &b, Value variant) {
    // Otherwise, always fallback to the staged strategy.
    auto strategy = ReductionStrategy::create(op->getContext(), captures);
    return buildReductionStrategy(b, variant, strategy);
  };

  // 3. Build strategy embedded into the IR.
  createTransformRegion(entryPoint, strategyBuilder);

  return success();
}
