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

#ifndef IREE_COMPILER_TRANSFORMS_SEQUENCER_PASSES_H_
#define IREE_COMPILER_TRANSFORMS_SEQUENCER_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Dispatches (iree.dispatch_region)
//===----------------------------------------------------------------------===//

// Identifies dispatchable regions of functions and wraps them in
// iree.dispatch_regions.
std::unique_ptr<OpPassBase<FuncOp>> createIdentifyDispatchRegionsPass();

// Folds multiple dispatch regions together that have compatible workloads.
std::unique_ptr<OpPassBase<FuncOp>> createFoldCompatibleDispatchRegionsPass();

// Rematerializes small previously-CSE'd constants into dispatch regions.
std::unique_ptr<OpPassBase<FuncOp>> createRematerializeDispatchConstantsPass();

// Outlines dispatch regions into executables.
std::unique_ptr<OpPassBase<ModuleOp>> createOutlineDispatchRegionsPass();

//===----------------------------------------------------------------------===//
// Reductions (iree.reduction_region)
//===----------------------------------------------------------------------===//

// Identifies reduction regions and wraps them in iree.reduction_regions.
std::unique_ptr<OpPassBase<ModuleOp>> createIdentifyReductionRegionsPass();

// Outlines dispatch regions into executables.
std::unique_ptr<OpPassBase<ModuleOp>> createOutlineReductionRegionsPass();

//===----------------------------------------------------------------------===//
// Lowering/Conversion
//===----------------------------------------------------------------------===//

// Lowers inputs to legal IREE inputs (a subset of XLA and Standard ops). Also
// serves to verify input is only made up of known ops.
std::unique_ptr<OpPassBase<FuncOp>> createLegalizeInputOpsPass();

// Lowers input dialect ops (e.g. std, xla_hlo) to IREE Sequencer HL dialect.
std::unique_ptr<OpPassBase<FuncOp>> createLowerToSequencerDialectPass();

// Lowers the HL sequencer dialect to the LL sequencer dialect.
std::unique_ptr<OpPassBase<FuncOp>> createLowerSequencerDialectPass();

//===----------------------------------------------------------------------===//
// Module Analysis and Assignment
//===----------------------------------------------------------------------===//

// Assigns module-unique ordinals to executables and their entry points.
std::unique_ptr<OpPassBase<ModuleOp>> createAssignExecutableOrdinalsPass();

// Assigns workload attributes to executable entry points based on dispatches.
std::unique_ptr<OpPassBase<ModuleOp>> createAssignExecutableWorkloadAttrsPass();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSFORMS_SEQUENCER_PASSES_H_
