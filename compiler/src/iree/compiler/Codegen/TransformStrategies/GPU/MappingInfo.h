
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_MAPPING_INFO_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_MAPPING_INFO_H_

#include "mlir/IR/Attributes.h"

namespace mlir::iree_compiler::gpu {

/// Helper struct to hold the mapping information for a given operation.
struct MappingInfo {
  SmallVector<int64_t> numThreads;
  // Note: explicitly computing the tileSizes is only needed until masked
  // vectorization properly computes the bounds automatically.
  SmallVector<int64_t> tileSizes;
  SmallVector<Attribute> threadMapping;
  std::optional<int64_t> vectorSize;
  void print(llvm::raw_ostream &os) const;
  LLVM_DUMP_METHOD void dump() const;
};

} // namespace mlir::iree_compiler::gpu

#endif // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_MAPPING_INFO_H_
