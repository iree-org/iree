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

#ifndef IREE_COMPILER_DIALECT_FLOW_UTILS_WORKLOADUTILS_H_
#define IREE_COMPILER_DIALECT_FLOW_UTILS_WORKLOADUTILS_H_

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

// Calculates the workload for |op| based on the given operation operand.
// Returns an index representing the total number of invocations required.
//
// The |baseOperand| is usually one of the results of a dispatch that signifies
// how many invocations are ideal for writing the result. Later on in the
// lowering process, once workgroup sizes are determined by target backends,
// the workload will be divided up. The returned value here represents the
// entirety of the computation.
Value calculateWorkload(Operation *op, Value baseOperand);

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_FLOW_UTILS_WORKLOADUTILS_H_
