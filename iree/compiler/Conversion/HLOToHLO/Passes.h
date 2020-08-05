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

//===- Passes.h - Pass to convert from XLA-HLO to IREE supported XLA-HLO --===//
//
// IREE specific passes used to sanitize XLA-HLO to IREE compatible XLA-HLO.
// Some examples may be raising gather operations to torch index select or
// decomposing complex operations input real valued equivalents.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_CONVERSION_HLOTOHLO_PASSES_H_
#define IREE_COMPILER_CONVERSION_HLOTOHLO_PASSES_H_

#include <memory>

#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

/// Creates a pass to decompose XLA-HLO clamp ops into primitive ops.
std::unique_ptr<OperationPass<FuncOp>> createDecomposeHLOClampPass();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_HLOTOHLO_PASSES_H_
