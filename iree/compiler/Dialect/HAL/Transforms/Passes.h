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

#ifndef IREE_COMPILER_DIALECT_HAL_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_HAL_TRANSFORMS_PASSES_H_

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Adds a set of passes to the given pass manager that run the required HAL
// transforms in the canonical order.
//
// Most translation code should prefer to use this instead of manually adding
// the passes themselves to ensure that expected pass ordering is observed.
//
// The expected usage is:
//   <run conversion to flow/sequencer/etc>
//   buildHALTransformPassPipeline & run
//   <run conversion from HAL to vm/etc>
void buildHALTransformPassPipeline(OpPassManager &passManager,
                                   const TargetOptions &targetOptions);

void registerHALTransformPassPipeline();

//===----------------------------------------------------------------------===//
// Conversion
//===----------------------------------------------------------------------===//

// Convert input flow/std/etc dialects to the IREE HAL dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToHALPass();

//===----------------------------------------------------------------------===//
// Device management
//===----------------------------------------------------------------------===//

// Outlines hal.device.switch conditions into functions and inlines conditions.
std::unique_ptr<OperationPass<FuncOp>> createInlineDeviceSwitchesPass();

// Finds hal.device.query ops and creates variables initialized on startup.
std::unique_ptr<OperationPass<ModuleOp>> createMemoizeDeviceQueriesPass();

//===----------------------------------------------------------------------===//
// Executable translation and optimization
//===----------------------------------------------------------------------===//

// Defines hal.executables and hal.interfaces for flow.executable ops based on
// usage within the module. Target backends are queried to check for support and
// device placements are made.
std::unique_ptr<OperationPass<ModuleOp>> createMaterializeInterfacesPass(
    TargetOptions executableOptions);

// Translates hal.executable.target ops via a nested translation pipeline.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableOp>>
createTranslateExecutablesPass(TargetOptions executableOptions);

// Calls into each target backend to have it link multiple hal.executables
// together (if that makes sense). For example, the LLVM AOT backend may combine
// all executable targets for the same architecture into a single executable and
// link it as a shared library.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createLinkExecutablesPass(
    TargetOptions executableOptions);

// Resolves hal.executable.entry_point references to ordinals.
std::unique_ptr<OperationPass<ModuleOp>> createResolveEntryPointOrdinalsPass();

// Converts hal.executable.target ops to hal.executable.binary ops.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableOp>>
createSerializeExecutablesPass(TargetOptions executableOptions);

// For functions that contain reflection metadata in an
// iree.generateabi.reflection attribute, generate public ABI functions for
// typical clients to use.
std::unique_ptr<OperationPass<ModuleOp>> createPublicABIGenerationPass();

//===----------------------------------------------------------------------===//
// Resource initialization, caching, and optimization
//===----------------------------------------------------------------------===//

// Finds all resource lookups (such as hal.executable.lookup), materializes
// their cache storage and initialization, and rewrites the lookups to
// references.
std::unique_ptr<OperationPass<ModuleOp>> createMaterializeResourceCachesPass(
    TargetOptions executableOptions);

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

inline void registerHALPasses() {
  registerHALTransformPassPipeline();
  auto executableOptions = getTargetOptionsFromFlags();
  createConvertToHALPass();
  createInlineDeviceSwitchesPass();
  createMemoizeDeviceQueriesPass();
  createMaterializeInterfacesPass(executableOptions);
  createTranslateExecutablesPass(executableOptions);
  createLinkExecutablesPass(executableOptions);
  createResolveEntryPointOrdinalsPass();
  createSerializeExecutablesPass(executableOptions);
  createPublicABIGenerationPass();
  createMaterializeResourceCachesPass(executableOptions);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TRANSFORMS_PASSES_H_
