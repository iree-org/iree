
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

#include "iree/compiler/Conversion/LinalgToLLVM/KernelDispatch.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace iree_compiler {

llvm::SmallVector<int64_t, 4> getTileSizesImpl(linalg::MatmulOp op) {
  return {128, 128};
}

llvm::SmallVector<int64_t, 4> CPUKernelDispatch::getTileSizes(
    Operation* op) const {
  if (isa<linalg::MatmulOp>(op)) {
    return getTileSizesImpl(dyn_cast<linalg::MatmulOp>(op));
  }
  return {1, 1, 1};
}

}  // namespace iree_compiler
}  // namespace mlir
