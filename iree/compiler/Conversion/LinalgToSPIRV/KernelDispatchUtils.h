// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- KernelDispatchUtils.h - Utilities for generating dispatch info -----===//
//
// This file declares utility functions that can be used to create information
// the dispatch on the host side needs to execute an entry point function, like
// the number of workgroups to use for launch, etc.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CONVERSION_LINALGTOSPIRV_KERNELDISPATCHUTILS_H_
#define IREE_COMPILER_CONVERSION_LINALGTOSPIRV_KERNELDISPATCHUTILS_H_

#include <array>

#include "iree/compiler/Conversion/Common/LaunchConfig.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/CodeGenOptionUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {

Optional<LaunchConfig> initGPULaunchConfig(
    MLIRContext *context, const linalg::LinalgDependenceGraph &dependenceGraph,
    const SPIRVCodegenOptions &options, ArrayRef<linalg::LinalgOp> linalgOps);

/// Returns the size of instruction in `vector` dialect that maps directly to
/// the hardware.
Optional<SmallVector<int64_t, 4>> getNativeVectorSize(Operation *op);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_LINALGTOSPIRV_DISPATCHUTILS_H_
