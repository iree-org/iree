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

#ifndef IREE_COMPILER_DIALECT_FLOW_ANALYSIS_EXECUTABLEHASHANALYSIS_H_
#define IREE_COMPILER_DIALECT_FLOW_ANALYSIS_EXECUTABLEHASHANALYSIS_H_

#include "llvm/ADT/Hashing.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

// Analysis of an IREE::Flow::ExecutableOp that distills it down to a hash code.
class ExecutableHashAnalysis {
 public:
  explicit ExecutableHashAnalysis(Operation *op);

  llvm::hash_code hashCode;
};

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_FLOW_ANALYSIS_EXECUTABLEHASHANALYSIS_H_
