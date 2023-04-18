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
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/Common.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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
using transform::MatchOp;

PackStrategy PackStrategy::create(MLIRContext *context, tensor::PackOp packOp,
                                  bool lowerToAVX2) {
  LLVM_DEBUG(DBGS() << "use CPU pack strategy\n");
  PackStrategy strategy;
  strategy.numThreads = 1;
  strategy.lowerToAVX2 = lowerToAVX2;

  auto destType = packOp.getDestType();
  int rank = destType.getRank();

  strategy.tileSizes.append(rank, 1);
  auto getMaxDivisor = [](int64_t val, int64_t tileSize) -> int64_t {
    if (ShapedType::isDynamic(val)) return tileSize;
    FailureOr<int64_t> res = maxDivisorOfValueBelowLimit(val, tileSize);
    if (failed(res)) return 1;
    return res.value();
  };

  for (int i = rank - 1, cnt = 0; i >= 0; --i) {
    int64_t size = destType.getDimSize(i);
    if (size == 1) continue;
    strategy.tileSizes[i] = getMaxDivisor(size, 8);
    cnt++;
    if (cnt >= 2) break;
  }

  return strategy;
}

void buildPackStrategy(ImplicitLocOpBuilder &b, Value variantH,
                       const PackStrategy &strategy) {
  MLIRContext *ctx = b.getContext();
  Value packOp = b.create<transform::MatchOp>(
      variantH, tensor::PackOp::getOperationName());

  TileToForallAndFuseAndDistributeResult distResult =
      buildTileFuseDistToForallAndWorgroupCountWithNumThreads(
          b, variantH, packOp, /*opsHToFuse=*/ValueRange(),
          /*numThreads=*/
          getAsOpFoldResult(b.getIndexArrayAttr({strategy.getNumThreads()})),
          b.getArrayAttr(mlir::gpu::GPUBlockMappingAttr::get(
              ctx, mlir::gpu::Blocks::DimX)));

  // Lower pack op, generalize the transpose op and fold unit dims away.
  Value transposeOp = std::get<2>(buildLowerPack(b, distResult.tiledOpH));
  SmallVector<OpFoldResult> tileSizes =
      getAsOpFoldResult(b.getIndexArrayAttr(strategy.getTransposeTileSizes()));
  TileToScfForAndFuseResult tileResult =
      buildTileFuseToScfFor(b, variantH, transposeOp, ValueRange(), tileSizes);
  transposeOp = tileResult.tiledOpH;
  auto pdlOpType = pdl::OperationType::get(ctx);
  b.create<transform::GeneralizeOp>(
      pdlOpType, b.create<transform::CastOp>(pdlOpType, transposeOp));

  ApplyPatternsOpPatterns configuration;
  configuration.rankReducingLinalg = true;
  buildCanonicalizationAndEnablingTransforms(b, configuration, variantH);

  vector::LowerVectorsOptions lowerVectorsOptions;
  lowerVectorsOptions
      .setVectorTransferSplit(vector::VectorTransferSplit::LinalgCopy)
      .setVectorTransposeLowering(vector::VectorTransposeLowering::Shuffle)
      .setTransposeAVX2Lowering(strategy.getLowerToAVX2())
      .setUnrollVectorTransfers(true);
  buildCommonTrailingStrategy(b, variantH, lowerVectorsOptions);
}
}  // namespace cpu
}  // namespace iree_compiler
}  // namespace mlir
