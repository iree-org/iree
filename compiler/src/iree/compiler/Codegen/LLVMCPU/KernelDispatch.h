// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_KERNELDISPATCH_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_KERNELDISPATCH_H_

#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/LLVMCPU/TargetMLTransformInfo.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::iree_compiler {

// Encodes the pre-processing strategy to be applied on a Linalg operation
// before vectorization.
enum class VectorPreProcStrategy {
  // Peel iterations from the vector dimensions so that they become multiple of
  // the vector length.
  Peeling,
  // Compute vector dimensions assuming vector masking support. Vector sizes may
  // be rounded up to the nearest power of two and out-of-bounds elements would
  // be masked-out.
  Masking,
  // Do not apply any vectorization pre-processing transformation.
  None
};

/// Struct that holds factors for heuristic distribution tile sizes selection.
/// The `minTileSizes`, `maxTileSizes` and `vectorSizeHints` can be empty or
/// as many as the number of loops.
struct DistributionHeuristicConfig {
  // TODO(hanchung): Remove `allowIncompleteTile` option after codegen can
  // vectorize all the shapes. Allowing incomplete tile is critical for odd
  // shapes (e.g., some dim sizes could be prime number).
  bool allowIncompleteTile = false;

  SmallVector<int64_t> minTileSizes;
  SmallVector<int64_t> maxTileSizes;

  // On the dimensions where the hints != 1, it will try to find the tile sizes
  // which are multipliers of the hints.
  SmallVector<int64_t> vectorSizeHints;
};

SmallVector<int64_t>
getMinTilingSizesForEachDim(mlir::FunctionOpInterface entryPointFn,
                            linalg::LinalgOp op,
                            const LinalgOpInfo &linalgOpInfo,
                            const TargetMLTransformInfo &targetMLTransInfo);

SmallVector<int64_t>
getDefaultDistributedLevelTileSizes(Operation *op,
                                    const DistributionHeuristicConfig &config);

int64_t getVectorSize(mlir::FunctionOpInterface entryPointFn,
                      ShapedType shapedType);

SmallVector<int64_t>
getMatmulCacheTileSizesForShape(ArrayRef<int64_t> inputTileSizes,
                                ArrayRef<int64_t> inputShape);

void setX86VectorTileSizes(linalg::GenericOp genericOp, unsigned numLoops,
                           ArrayRef<int64_t> distTileSizes,
                           ArrayRef<int64_t> minTileSizes,
                           ArrayRef<int64_t> maxTileSizes,
                           VectorPreProcStrategy vecPreProcStrategy,
                           SmallVectorImpl<int64_t> &vecTileSizes);

void getMatmulAArch64SMEVectorSizes(linalg::LinalgOp op,
                                    SmallVectorImpl<int64_t> &sizes,
                                    SmallVectorImpl<bool> &scalableSizeFlags);

LogicalResult initCPULaunchConfig(
    ModuleOp moduleOp,
    const SmallVector<std::unique_ptr<TileSizeSelectionPattern>> &tssPatterns);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_KERNELDISPATCH_H_
