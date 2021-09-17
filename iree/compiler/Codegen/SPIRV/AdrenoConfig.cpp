// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- AdrenoConfig.h - Adreno CodeGen Configurations ---------------------===//
//
// This file contains CodeGen configurations for Adreno GPUs.
//
//===----------------------------------------------------------------------===//

#include <array>

#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

namespace mlir {
namespace iree_compiler {
namespace detail {

//===----------------------------------------------------------------------===//
// Matmul
//===----------------------------------------------------------------------===//

static LogicalResult setOpConfig(linalg::LinalgOp op) {
  ArrayRef<int64_t> lhsShape = getUntiledShape(op.inputs()[0]);
  ArrayRef<int64_t> rhsShape = getUntiledShape(op.inputs()[1]);
  if (llvm::any_of(lhsShape, ShapedType::isDynamic)) return success();
  if (llvm::any_of(rhsShape, ShapedType::isDynamic)) return success();

  bool isBM = isa<linalg::BatchMatmulOp>(op);

  int64_t dimM = lhsShape[0 + isBM];
  int64_t dimK = lhsShape[1 + isBM];
  int64_t dimN = rhsShape[1 + isBM];

  // The core idea is to distribute the matmul M/N dimension to the workgroup
  // Y/X dimension, with each thread in a workgroup handling multiple vector
  // elements. We start from the best (X, Y) and the tiling sizes for (M, N, K)
  // and try different configurations by scaling them down until we find a
  // configuration that can perfectly tile the input matmul.

  const int64_t bestX = 32, bestY = 2;
  const int64_t bestThreadM = 8, bestThreadN = 8, bestThreadK = 4;

  int64_t residualThreads = bestX * bestY;
  int64_t residualTilingFactor = (bestThreadM + bestThreadK) * bestThreadN;

  SmallVector<int64_t, 3> workgroupSize(3, 1);               // (X, Y, Z)
  SmallVector<int64_t, 4> workgroupTileSizes(3 + isBM, 0);   // (B, M, N, K)
  SmallVector<int64_t, 4> invocationTileSizes(3 + isBM, 0);  // (B, M, N, K)

  if (isBM) workgroupTileSizes[0] = invocationTileSizes[0] = 1;

  // Deduce the configuration for the N dimension. Start with the best workgroup
  // X size, and reduce by a factor of two each time.
  for (int64_t x = bestX; x >= 2; x >>= 1) {
    // Handle 4 elements per thread for the innermost dimension. We need this
    // for vectorized load.
    int64_t chosenTileSize = 4;
    if (dimN % (x * chosenTileSize) == 0) {
      workgroupSize[0] = x;
      workgroupTileSizes[1 + isBM] = x * chosenTileSize;
      invocationTileSizes[1 + isBM] = chosenTileSize;
      residualThreads /= x;
      assert(residualTilingFactor % chosenTileSize == 0);
      residualTilingFactor /= chosenTileSize;
      break;
    }
  }
  if (workgroupTileSizes[1 + isBM] == 0) return success();

  // Deduce the configuration for the M dimension. Start with the best workgroup
  // Y size, and reduce by a factor of two each time.
  for (int64_t y = residualThreads; y >= 1; y >>= 1) {
    int64_t chosenTileSize = 0;
    // Reduce the thread tiling size by one each time. We read one row each
    // time; so it's fine to not be some power of two here.
    for (int64_t t = bestThreadM; t >= 1; --t) {
      if (dimM % (y * t) == 0) {
        chosenTileSize = t;
        break;
      }
    }
    if (chosenTileSize) {
      workgroupSize[1] = y;
      workgroupTileSizes[0 + isBM] = y * chosenTileSize;
      invocationTileSizes[0 + isBM] = chosenTileSize;
      assert(residualTilingFactor > chosenTileSize);
      residualTilingFactor -= chosenTileSize;
      break;
    }
  }
  if (workgroupTileSizes[0 + isBM] == 0) return success();

  // Deduce the configuration for the K dimension. We need some power of two
  // here so that we can do vector load.
  for (int64_t t = llvm::PowerOf2Floor(residualTilingFactor); t >= 1; t >>= 1) {
    if (dimK % t == 0) {
      workgroupTileSizes[2 + isBM] = invocationTileSizes[2 + isBM] = t;
      break;
    }
  }

  auto pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize;
  TileSizesListType tileSizes;
  tileSizes.push_back(workgroupTileSizes);
  tileSizes.emplace_back();
  tileSizes.push_back(invocationTileSizes);
  return setOpConfigAndEntryPointFnTranslation(op->getParentOfType<FuncOp>(),
                                               op, tileSizes, {}, pipeline,
                                               workgroupSize);
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult setAdrenoCodeGenConfig(const spirv::TargetEnv &targetEnv,
                                     Operation *rootOp) {
  int64_t subgroupSize = targetEnv.getResourceLimits().subgroup_size().getInt();
  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::BatchMatmulOp, linalg::MatmulOp>(
          [](auto op) { return setOpConfig(op); })
      .Case<linalg::Conv2DNhwcHwcfOp>([subgroupSize](auto op) {
        return setConvOpConfig(op, subgroupSize,
                               /*bestTilingFactor=*/32);
      })
      .Case<linalg::DepthwiseConv2DNhwOp>([subgroupSize](auto op) {
        return setConvOpConfig(op, subgroupSize,
                               /*bestTilingFactor=*/16);
      })
      .Default([](Operation *) { return success(); });
}

}  // namespace detail
}  // namespace iree_compiler
}  // namespace mlir
