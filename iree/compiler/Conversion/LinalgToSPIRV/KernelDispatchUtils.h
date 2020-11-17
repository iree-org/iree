// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class FuncOp;
class LogicalResult;
class Operation;
class PatternRewriter;
class ShapedType;
class Value;

namespace linalg {
class LinalgDependenceGraph;
class LinalgOp;
}  // namespace linalg
namespace iree_compiler {
struct SPIRVCodegenOptions;
}

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
