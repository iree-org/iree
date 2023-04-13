// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformDialectStrategies/CPU/PackStrategy.h"

#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMCPU/TransformExtensions/LLVMCPUExtensions.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/CPU/Common.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/AbstractReductionStrategy.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/Common.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir {
namespace iree_compiler {
namespace cpu {

using IREE::transform_dialect::ApplyPatternsOpPatterns;
using IREE::transform_dialect::TileToForallAndWorkgroupCountRegionOp;
using transform::LowerPackOp;
using transform::MatchOp;

void buildPackStrategy(ImplicitLocOpBuilder &b, Value variantH,
                       const PackConfig &config) {
  MLIRContext *ctx = b.getContext();
  Value packOp = b.create<transform::MatchOp>(
      variantH, tensor::PackOp::getOperationName());

  // TODO(hanchung): Support multi-threaded.
  SmallVector<int64_t> numThreads = {1};
  auto tileToForAll = b.create<TileToForallAndWorkgroupCountRegionOp>(
      packOp, numThreads, transform::NumThreadsSpec(),
      b.getArrayAttr(
          mlir::gpu::GPUBlockMappingAttr::get(ctx, mlir::gpu::Blocks::DimX)));

  // Lower to pad + expand_reshape + transpose
  packOp = tileToForAll.getTiledOp();
  auto lowerPackOp = b.create<LowerPackOp>(
      transform::OperationType::get(ctx, tensor::PadOp::getOperationName()),
      transform::OperationType::get(ctx,
                                    tensor::ExpandShapeOp::getOperationName()),
      transform::OperationType::get(ctx,
                                    linalg::TransposeOp::getOperationName()),
      b.create<transform::CastOp>(transform::OperationType::get(
                                      ctx, tensor::PackOp::getOperationName()),
                                  packOp));
  Value transposeOp = lowerPackOp.getTransposeOp();
  SmallVector<OpFoldResult> tileSizes =
      getAsOpFoldResult(b.getIndexArrayAttr({1, 8, 8}));
  auto tiletoScfForOp = b.create<transform::TileOp>(transposeOp, tileSizes);

  // Generalize the transpose op and fold unit dims away.
  transposeOp = tiletoScfForOp.getTiledLinalgOp();
  auto pdlOpType = pdl::OperationType::get(ctx);
  b.create<transform::GeneralizeOp>(
      pdlOpType, b.create<transform::CastOp>(pdlOpType, transposeOp));

  ApplyPatternsOpPatterns configuration;
  configuration.rankReducingLinalgViaReshapes = true;
  Value funcH =
      b.create<transform::MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(
      b, configuration, funcH);

  vector::LowerVectorsOptions lowerVectorsOptions;
  lowerVectorsOptions
      .setVectorTransferSplit(vector::VectorTransferSplit::LinalgCopy)
      .setVectorTransposeLowering(vector::VectorTransposeLowering::Shuffle)
      .setTransposeAVX2Lowering(config.lowerToAVX2)
      .setUnrollVectorTransfers(true);
  buildCommonTrailingStrategy(b, variantH, lowerVectorsOptions);
}
}  // namespace cpu
}  // namespace iree_compiler
}  // namespace mlir
