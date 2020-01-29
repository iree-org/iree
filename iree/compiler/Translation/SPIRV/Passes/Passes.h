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

//===- Passes.h ------------------------------------------------*- C++//-*-===//
//
// Utility passes used in SPIR-V lowering.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_TRANSLATION_SPIRV_PASSES_PASSES_H
#define IREE_COMPILER_TRANSLATION_SPIRV_PASSES_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class FuncOp;
namespace iree_compiler {

// Legalizes integer width from i1, i8 and i64 types to i32 type.
std::unique_ptr<Pass> createAdjustIntegerWidthPass();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_SPIRV_PASSES_PASSES_H
