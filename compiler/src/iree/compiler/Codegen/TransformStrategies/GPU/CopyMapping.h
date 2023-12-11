
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_COPY_MAPPING_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_COPY_MAPPING_H_

#include <numeric>

#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/MappingInfo.h"

namespace mlir::iree_compiler::gpu {

struct CopyMapping {
  /// Vector size to use for the copy.
  int64_t vectorSize;

  /// numThreads to use for the copy mapping, from most major to most minor dims
  /// (i.e. numThreads.back() should be mapped to contiguous threads for best
  /// coalescing).
  SmallVector<int64_t> numThreads;

  /// Determine the maximal vector size to use to copy a contiguous array of
  /// `numContiguousElements`, each of bitwidth `elementalBitWidth`.
  /// The `alignment` is the number of elements by which the most minor
  /// dimension of the copy is aligned. This is an approximation of actual
  /// memory alignment after bufferization, for each row of the copy. This is
  /// used to restrict the of the copied vector so that it is properly aligned
  /// with the requirements of cp.async. If the copy alignemnt does not match
  /// the required aligned for a cp.async, thae conversion to cp.async will be
  /// skipped.
  /// Asserts that `elementalBitWidth` divides `numContiguousElements`.
  static int64_t
  maxContiguousElementsToTransfer(int64_t alignment,
                                  int64_t numContiguousElements,
                                  int64_t elementalBitWidth = 32);

  /// Compute the number of threads to use to perform a copy of `sizes`
  /// elements of `elementalBitWidth`.
  /// The `alignment` is the number of elements by which the most minor
  /// dimension of the copy is aligned. This is an approximation of actual
  /// memory alignment after bufferization, for each row of the copy. This is
  /// used to restrict the of the copied vector so that it is properly aligned
  /// with the requirements of cp.async. If the copy alignemnt does not match
  /// the required aligned for a cp.async, thae conversion to cp.async will be
  /// skipped.
  /// When `favorPredication` is false, the implementation avoids predication in
  /// the copy, even if it means reducing the granularity of the transfer.
  /// Otherwise, the implementation will come up with a best-effort predicated
  /// mapping that respects the maximal vector transfer size.
  static FailureOr<CopyMapping>
  numThreadsForCopy(int totalNumThreads, int64_t alignment,
                    ArrayRef<int64_t> sizes, bool favorPredication,
                    int64_t elementalBitWidth = 32);

  /// Greedily compute the MappingInfo to use to perform a copy of `sizes`
  /// elements of bitwidth `elementalBitWidth`.
  /// The `alignment` is the number of elements by which the most minor
  /// dimension of the copy is aligned. This is an approximation of actual
  /// memory alignment after bufferization, for each row of the copy. This is
  /// used to restrict the of the copied vector so that it is properly aligned
  /// with the requirements of cp.async. If the copy alignemnt does not match
  /// the required aligned for a cp.async, thae conversion to cp.async will be
  /// skipped. When `favorPredication` if false, the mapping is computed to fill
  /// all threads with an equal amount of data to copy, so as to avoid
  /// predication. Predication often ends up breaking current pipelining
  /// implementations down the line and is generally discouraged. At the moment,
  /// asserts that sizes has exactly 2 entries.
  static MappingInfo getMappingInfo(MLIRContext *ctx, int totalNumThreads,
                                    int64_t alignment, ArrayRef<int64_t> sizes,
                                    bool favorPredication = false,
                                    int64_t elementalBitWidth = 32);
};

} // namespace mlir::iree_compiler::gpu

#endif // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_COPY_MAPPING_H_
