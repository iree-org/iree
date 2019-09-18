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

#ifndef THIRD_PARTY_MLIR_EDGE_IREE_COMPILER_TRANSFORMS_INTERPRETER_PASSES_H_
#define THIRD_PARTY_MLIR_EDGE_IREE_COMPILER_TRANSFORMS_INTERPRETER_PASSES_H_

#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

// Expands reduction functions to their interpreter ops.
std::unique_ptr<OpPassBase<ModuleOp>> createExpandReductionsToOpsPass();

// Refactors entry points to match the IREE dispatch executable ABI.
std::unique_ptr<OpPassBase<ModuleOp>> createMakeExecutableABIPass();

// Lowers Standard dialect (std.*) ops to IREE Interpreter HL ops.
std::unique_ptr<OpPassBase<FuncOp>> createLowerStdToInterpreterDialectPass();

// Lowers XLA dialect (xla_hlo.*) ops to IREE Interpreter HL ops.
std::unique_ptr<OpPassBase<FuncOp>> createLowerXLAToInterpreterDialectPass();

// Lowers IREE HL ops (iree_hl_interp.*) to LL ops (iree_ll_interp.*).
std::unique_ptr<OpPassBase<FuncOp>> createLowerInterpreterDialectPass();

// Optimizes std.load and std.store to remove unnessisary copies.
std::unique_ptr<OpPassBase<FuncOp>> createInterpreterLoadStoreDataFlowOptPass();

// Legalizes all ops to canonical/supported forms.
std::unique_ptr<OpPassBase<FuncOp>> createLegalizeInterpreterOpsPass();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // THIRD_PARTY_MLIR_EDGE_IREE_COMPILER_TRANSFORMS_INTERPRETER_PASSES_H_
