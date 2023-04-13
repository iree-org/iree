// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_CPU_PACK_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_CPU_PACK_STRATEGY_H_

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace iree_compiler {
namespace cpu {

/// Structure to hold a summary of HW-derived properties to configure the pack
/// codegen strategy.
struct PackConfig {
  bool lowerToAVX2 = false;
};

void buildPackStrategy(ImplicitLocOpBuilder &b, Value variantH,
                       const PackConfig &config);

}  // namespace cpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_CPU_PACK_STRATEGY_H_
