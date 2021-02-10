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

#include "iree/compiler/Conversion/Common/Passes.h"

#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

void addLinalgBufferizePasses(OpPassManager &passManager,
                              WorkgroupMemoryAllocationFn allocationFn) {
  passManager.addPass(createLinalgBufferizePass(allocationFn));
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addNestedPass<FuncOp>(createRemoveDeadMemAllocsPass());
  passManager.addPass(createCopyRemovalPass());
  // passManager.addPass(createBufferHoistingPass());
  // TODO(nicolasvasilache): bug in buffer loop hoisting with
  // dynamic_linalg_matmul_on_tensors_fuse_0.mlir
  // passManager.addPass(createBufferLoopHoistingPass());
}

}  // namespace iree_compiler
}  // namespace mlir
