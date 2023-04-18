// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_CPU_PACK_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_CPU_PACK_STRATEGY_H_

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace iree_compiler {
namespace cpu {

struct CPUModel;

class PackStrategy {
 public:
  static PackStrategy create(MLIRContext *context, tensor::PackOp packOp,
                             bool lowerToAVX2);

  PackStrategy() : numThreads(0), lowerToAVX2(false) {}
  PackStrategy(const PackStrategy &) = default;
  PackStrategy &operator=(const PackStrategy &) = default;

  int64_t getNumThreads() const { return numThreads; }
  bool getLowerToAVX2() const { return lowerToAVX2; }
  SmallVector<int64_t> getTransposeTileSizes() const { return tileSizes; }

 private:
  int64_t numThreads;
  bool lowerToAVX2;
  SmallVector<int64_t> tileSizes;
};

/// Entry point to build the transform IR corresponding to a pack op.
void buildPackStrategy(ImplicitLocOpBuilder &b, Value variantH,
                       const PackStrategy &strategy);

}  // namespace cpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_CPU_PACK_STRATEGY_H_
