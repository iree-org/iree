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

//===- CooperativeMatrixAnalysis.h - Analysis to help lowering of matmul---===//
//
// Analysis class to help decide whether a chain of operations can use
// cooperative matrix extension.
// Since cooperative matrix is a separate type it needs to be used consistently
// across operations. This analyzes if a chain of operations can be fully
// converted to cooperative matrix operations. It then provides a query lowering
// passes can use to know whether an instruction should use cooperative matrix
// or not.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_CONVERSION_LINALGTOSPIRV_COOPERATIVEMATRIXANALYSIS_H_
#define IREE_COMPILER_CONVERSION_LINALGTOSPIRV_COOPERATIVEMATRIXANALYSIS_H_
#include "llvm/ADT/DenseSet.h"

namespace mlir {
class Operation;

namespace iree_compiler {

class CooperativeMatrixAnalysis {
 public:
  explicit CooperativeMatrixAnalysis(mlir::Operation *);

  // Return true if the operation should be lowered using operations on
  // cooperative matrix type.
  bool usesCooperativeMatrixType(mlir::Operation *op) const {
    return usesCooperativeMatrix.count(op);
  }

 private:
  llvm::DenseSet<mlir::Operation *> usesCooperativeMatrix;
};
}  // namespace iree_compiler
}  // namespace mlir
#endif  // IREE_COMPILER_CONVERSION_LINALGTOSPIRV_COOPERATIVEMATRIXANALYSIS_H_
