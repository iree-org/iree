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
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/Common.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

// TODO: significantly better namespacing.
using iree_compiler::IREE::transform_dialect::ForeachThreadToWorkgroupOp;
using transform::MatchOp;
using transform::SplitHandlesOp;
using transform_ext::AllDims;
using transform_ext::m_StructuredOp;
using transform_ext::NumEqualsTo;
using transform_ext::ShapeKind;

//===----------------------------------------------------------------------===//
// Higher-level problem-specific strategy creation APIs, these should favor
// user-friendliness.
//===----------------------------------------------------------------------===//

// TODO: consider lifting and exposing.

/// Structure to hold the parameters related to GPU reduction strategy.
struct CPUReductionStrategyInfos {
  int64_t workgroupSize;
  SmallVector<int64_t> tileSizes;
};

static bool matchCPUReduction(linalg::LinalgOp op,
                              CPUReductionStrategyInfos &infos) {
  // TODO: match the sequence the strategy supports.
  auto fill = m_StructuredOp<linalg::FillOp>();
  auto pattern = m_StructuredOp()
                     .dim(AllDims(), ShapeKind::Static)
                     .dim(-1, utils::IteratorType::reduction)
                     .output(NumEqualsTo(1))
                     .output(0, fill);

  // TODO: set the right config as expected by the strategy.
  infos.workgroupSize = 1;
  SmallVector<unsigned> partitionedLoops =
      cast<iree_compiler::PartitionableLoopsInterface>(op.getOperation())
          .getPartitionableLoops(iree_compiler::kNumMaxParallelDims);
  size_t numLoops = partitionedLoops.empty() ? 0 : partitionedLoops.back() + 1;
  // Tile all the parallel dimension to 1.
  infos.tileSizes.append(numLoops, 1);
  return true;
}

// TODO: generalize and automate over and over.
// TODO: significantly shrink this down.
static void createReductionCpuStrategy(ImplicitLocOpBuilder &b, Value variantH,
                                       const CPUReductionStrategyInfos &info) {
  // Step 0. Fetch transform information from the config and materialize it in
  // the payload IR.
  // TODO: this still requires specific knowledge of ops present in the IR
  // and is very brittle.
  Value originalFillH =
      b.create<MatchOp>(variantH, linalg::FillOp::getOperationName());
  Value originalGenericH =
      b.create<MatchOp>(variantH, linalg::GenericOp::getOperationName());

  // Step 1: Distribute to blocks using the current IREE lowering config.
  variantH = iree_compiler::buildReductionStrategyBlockDistributionPart(
      b, variantH, originalFillH, originalGenericH, Value(),
      getAsOpFoldResult(b.getI64ArrayAttr(info.tileSizes)));

  // Step 2. Rank-reduce and buildVectorize.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = iree_compiler::buildVectorize(b, funcH);

  // Step 3. Bufferize and drop HAL descriptor from memref ops.
  variantH = iree_compiler::buildBufferize(b, variantH, /*targetGpu=*/true);

  // Step 4. Post-bufferization mapping to blocks only.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = b.create<ForeachThreadToWorkgroupOp>(funcH);
}

LogicalResult iree_compiler::cpu::matchAndSetReductionStrategy(
    func::FuncOp entryPoint, linalg::LinalgOp op) {
  // 1. Match
  CPUReductionStrategyInfos infos;
  if (!matchCPUReduction(op, infos)) return failure();
  auto strategyBuilder = [&](ImplicitLocOpBuilder &b, Value variant) {
    return createReductionCpuStrategy(b, variant, infos);
  };
  // 2. Add the strategy.
  createTransformRegion(entryPoint, strategyBuilder);
  return success();
}
