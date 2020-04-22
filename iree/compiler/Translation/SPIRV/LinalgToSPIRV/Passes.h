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

#ifndef IREE_COMPILER_TRANSLATION_SPIRV_LINALGTOSPIRV_PASSES_H_
#define IREE_COMPILER_TRANSLATION_SPIRV_LINALGTOSPIRV_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

///--------------------------------------------------------------------------///
// Markers on Linalg operations that determine which processor heirarchy to use
// for partitioning
///--------------------------------------------------------------------------///

/// Marker to denote that a linalg operation is to be partitioned to workitems
inline StringRef getWorkItemMarker() { return "workitem"; }

///--------------------------------------------------------------------------///

/// Pass to tile and fuse linalg operations on buffers. The pass takes as
/// argument the `workgroupSize` that the tiling should use. Note that the
/// tile-sizes are the reverse of the workgroup size. So workgroup size along
/// "x" is used to tile the innermost loop, along "y" for the next innermost (if
/// it exists) and along "z" for the next loop (if it exists). The workgroup
/// size is expected to be of size at-most 3.
std::unique_ptr<OperationPass<FuncOp>> createLinalgTileAndFusePass(
    ArrayRef<int64_t> workGroupSize = {});

/// Pass to add the synchronizations and attributes needed to lower from PLoops
/// to GPU dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToGPUPass();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_SPIRV_LINALGTOSPIRV_PASSES_H_
