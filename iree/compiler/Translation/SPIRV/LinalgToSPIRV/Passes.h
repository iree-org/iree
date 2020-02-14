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

//===- Passes.h - IREE specific passes used in Linalg To SPIRV conversion--===//
//
// IREE specific passes used in the XLA -> Linalg -> SPIRV Conversion.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_TRANSLATION_SPIRV_LINALGTOSPIRV_PASSES_H
#define IREE_COMPILER_TRANSLATION_SPIRV_LINALGTOSPIRV_PASSES_H

#include <memory>

namespace mlir {

class FuncOp;
template <typename OpTy>
class OpPassBase;

namespace iree_compiler {

/// Fuses linalg operations on tensors in dispatch function. For now does only
/// producer consumer fusion.
std::unique_ptr<OpPassBase<FuncOp>> createLinalgFusionPass();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_SPIRV_LINALGTOSPIRV_PASSES_H
