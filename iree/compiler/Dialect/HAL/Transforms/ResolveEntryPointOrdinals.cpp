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

#include <utility>

// TODO(scotttodd): trim includes
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// TODO(scotttodd): rewrite as conversion from CommandBufferDispatchOp to
//                  CommandBufferDispatchOrdinalOp?
class ResolveEntryPointOrdinalsPass
    : public PassWrapper<ResolveEntryPointOrdinalsPass,
                         OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override {
    auto moduleOp = getOperation();

    SmallVector<IREE::HAL::CommandBufferDispatchOp, 4> dispatchOps;
    moduleOp.walk([&](IREE::HAL::CommandBufferDispatchOp dispatchOp) {
      dispatchOps.push_back(dispatchOp);
    });
    for (auto dispatchOp : dispatchOps) {
      // Extract entry point ordinal from the nested symbol reference.
      auto entryPointOp = dyn_cast_or_null<IREE::HAL::ExecutableEntryPointOp>(
          SymbolTable::lookupNearestSymbolFrom(dispatchOp,
                                               dispatchOp.entry_point()));
      if (!entryPointOp) continue;
      OpBuilder builder(dispatchOp);
      builder.create<IREE::HAL::CommandBufferDispatchOrdinalOp>(
          dispatchOp.getLoc(), dispatchOp.command_buffer(),
          dispatchOp.executable(), entryPointOp.ordinalAttr(),
          dispatchOp.workgroup_x(), dispatchOp.workgroup_y(),
          dispatchOp.workgroup_z());
      dispatchOp.erase();
    }

    SmallVector<IREE::HAL::CommandBufferDispatchIndirectOp, 4>
        dispatchIndirectOps;
    moduleOp.walk(
        [&](IREE::HAL::CommandBufferDispatchIndirectOp dispatchIndirectOp) {
          dispatchIndirectOps.push_back(dispatchIndirectOp);
        });
    for (auto dispatchIndirectOp : dispatchIndirectOps) {
      // Extract entry point ordinal from the nested symbol reference.
      auto entryPointOp = dyn_cast_or_null<IREE::HAL::ExecutableEntryPointOp>(
          SymbolTable::lookupNearestSymbolFrom(
              dispatchIndirectOp, dispatchIndirectOp.entry_point()));
      if (!entryPointOp) continue;
      OpBuilder builder(dispatchIndirectOp);
      builder.create<IREE::HAL::CommandBufferDispatchIndirectOrdinalOp>(
          dispatchIndirectOp.getLoc(), dispatchIndirectOp.command_buffer(),
          dispatchIndirectOp.executable(), entryPointOp.ordinalAttr(),
          dispatchIndirectOp.workgroups_buffer(),
          dispatchIndirectOp.workgroups_offset());
      dispatchIndirectOp.erase();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createResolveEntryPointOrdinalsPass() {
  return std::make_unique<ResolveEntryPointOrdinalsPass>();
}

static PassRegistration<ResolveEntryPointOrdinalsPass> pass(
    "iree-hal-resolve-entry-point-ordinals",
    "Resolves hal.executable.entry_point references to ordinals");

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
