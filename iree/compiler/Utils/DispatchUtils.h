// Copyright 2019 Google LLC
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

#ifndef IREE_COMPILER_UTILS_DISPATCHUTILS_H_
#define IREE_COMPILER_UTILS_DISPATCHUTILS_H_

#include <utility>

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace iree_compiler {

// Calculates the workload based on the shape of a tensor.
void calculateWorkload(ArrayRef<int64_t> shape,
                       std::array<int32_t, 3> &workload);

// Calculates the workload for |op| based on the op type.
Value calculateWorkload(Operation *op, Value baseOperand);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_DISPATCHUTILS_H_
