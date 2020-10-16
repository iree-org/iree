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

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

using AccessTypeArray =
    std::array<int, static_cast<size_t>(MemoryAccessBitfield::All)>;

struct InterfaceResources {
  AccessTypeArray count = {0};
  // Offset into binding ordinals of the given access type. The sum of all
  // counts of access bitfields prior to each entry.
  AccessTypeArray bindingOffsets = {0};
  unsigned int pushConstants = 0;
};

// Returns the counts for each access type bitmask.
// This could be made more sophisticated like allowing write/discard to be
// combined if it meant fewer total unique interfaces. For now it just counts
// each unique bitmask value.
static InterfaceResources sumInterfaceResources(
    ArrayRef<IREE::HAL::InterfaceOp> interfaceOps) {
  InterfaceResources result;
  for (auto interfaceOp : interfaceOps) {
    InterfaceResources interfaceCounts;
    for (auto bindingOp : interfaceOp.getOps<IREE::HAL::InterfaceBindingOp>()) {
      ++interfaceCounts.count[static_cast<int>(bindingOp.access())];
    }
    for (size_t i = 0; i < static_cast<size_t>(MemoryAccessBitfield::All);
         ++i) {
      result.count[i] = std::max(result.count[i], interfaceCounts.count[i]);
    }
    result.pushConstants = std::max(result.pushConstants,
                                    interfaceOp.push_constants().getValueOr(0));
  }

  int binding = 0;
  for (int accessBits = 0; accessBits < result.count.size(); ++accessBits) {
    result.bindingOffsets[accessBits] = binding;
    binding += result.count[accessBits];
  }

  return result;
}

class CanonicalizeInterfacesPass
    : public PassWrapper<CanonicalizeInterfacesPass, OperationPass<ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
  }

  void runOnOperation() override {
    // Get all interfaces in the module.
    SmallVector<IREE::HAL::InterfaceOp, 8> allInterfaceOps;
    for (auto executableOp : getOperation().getOps<IREE::HAL::ExecutableOp>()) {
      auto executableInterfaceOps =
          executableOp.getOps<IREE::HAL::InterfaceOp>();
      allInterfaceOps.append(executableInterfaceOps.begin(),
                             executableInterfaceOps.end());
    }

    // Find the counts of each binding type used in each interface (kind of like
    // a histogram).
    auto maxInterfaceResources = sumInterfaceResources(allInterfaceOps);

    // TODO(benvanik): actually group these by a meaningful relationship. For
    // example, if the same constant buffer is used in 90% of interfaces we
    // should dramatically favor assigning that to the same binding. That will
    // ensure we get the greatest savings of descriptor set update/pushes.

    // Create a location to use for the combined interface op based on all
    // interfaces we used.
    auto allInterfaceLocations = llvm::to_vector<8>(llvm::map_range(
        allInterfaceOps,
        [&](IREE::HAL::InterfaceOp op) { return op.getLoc(); }));
    auto combinedLoc = FusedLoc::get(allInterfaceLocations, &getContext());

    // Clone the new max-limit interface ops in each executable and replace
    // uses of existing interfaces.
    for (auto executableOp : getOperation().getOps<IREE::HAL::ExecutableOp>()) {
      SymbolTable symbolTable(executableOp);
      auto existingInterfaceOps =
          llvm::to_vector<8>(executableOp.getOps<IREE::HAL::InterfaceOp>());

      // Create new interface op given our max counts.
      auto interfaceBuilder = OpBuilder::atBlockBegin(executableOp.getBody());
      auto combinedInterfaceOp =
          interfaceBuilder.create<IREE::HAL::InterfaceOp>(
              combinedLoc, "_canonical_interface",
              interfaceBuilder.getI32IntegerAttr(
                  maxInterfaceResources.pushConstants));
      SymbolTable::setSymbolVisibility(combinedInterfaceOp,
                                       SymbolTable::Visibility::Nested);

      // Setup all bindings based on the max counts required for each access
      // type.
      int set = 0;
      int binding = 0;
      auto bindingBuilder =
          OpBuilder::atBlockBegin(combinedInterfaceOp.getBody());
      SmallVector<IREE::HAL::InterfaceBindingOp, 8> bindingOps;
      for (int accessBits = 0; accessBits < maxInterfaceResources.count.size();
           ++accessBits) {
        int count = maxInterfaceResources.count[accessBits];
        if (!count) continue;
        for (int i = 0; i < count; ++i) {
          auto bindingOp = bindingBuilder.create<IREE::HAL::InterfaceBindingOp>(
              combinedLoc,
              std::string("s") + std::to_string(set) + "b" +
                  std::to_string(binding),
              set, binding, IREE::HAL::DescriptorType::StorageBuffer,
              static_cast<IREE::HAL::MemoryAccessBitfield>(accessBits));
          bindingOps.push_back(bindingOp);
          ++binding;
        }
      }

      // Build a map of old bindings to new bindings.
      DenseMap<StringRef, DenseMap<StringRef, SymbolRefAttr>> bindingMap;
      for (auto interfaceOp : existingInterfaceOps) {
        AccessTypeArray counts = {0};
        for (auto bindingOp :
             interfaceOp.getOps<IREE::HAL::InterfaceBindingOp>()) {
          int accessBits = static_cast<int>(bindingOp.access());
          int binding = maxInterfaceResources.bindingOffsets[accessBits] +
                        counts[accessBits]++;
          auto &mapEntry = bindingMap[interfaceOp.getName()];
          mapEntry[bindingOp.getName()] = bindingBuilder.getSymbolRefAttr(
              combinedInterfaceOp.getName(),
              {bindingBuilder.getSymbolRefAttr(bindingOps[binding].getName())});
        }
      }
      auto getRemappedBinding = [&](SymbolRefAttr originalSymRef) {
        auto &mapEntry = bindingMap[originalSymRef.getRootReference()];
        return mapEntry[originalSymRef.getLeafReference()];
      };

      // Remap existing interfaces into new interface.
      for (auto targetOp :
           executableOp.getOps<IREE::HAL::ExecutableTargetOp>()) {
        for (auto entryPointOp :
             targetOp.getOps<IREE::HAL::ExecutableEntryPointOp>()) {
          entryPointOp.interfaceAttr(
              bindingBuilder.getSymbolRefAttr(combinedInterfaceOp.getName()));
        }
        auto moduleOp = targetOp.getInnerModule();
        for (auto funcOp : moduleOp.getOps<mlir::FuncOp>()) {
          for (auto &op : funcOp.getBody().getOps()) {
            if (auto loadOp = dyn_cast<IREE::HAL::InterfaceLoadTensorOp>(op)) {
              loadOp.bindingAttr(getRemappedBinding(loadOp.binding()));
            } else if (auto storeOp =
                           dyn_cast<IREE::HAL::InterfaceStoreTensorOp>(op)) {
              storeOp.bindingAttr(getRemappedBinding(storeOp.binding()));
            }
          }
        }

        // MaterializeInterfaces clones an interface into the module for symbol
        // lookup purposes; we need to replace that with our combined op.
        // TODO(benvanik): remove the nested/cloned interface.
        for (auto interfaceOp :
             llvm::to_vector<4>(moduleOp.getOps<IREE::HAL::InterfaceOp>())) {
          interfaceOp.erase();
        }
        OpBuilder::atBlockTerminator(moduleOp.getBody())
            .clone(*combinedInterfaceOp);
      }

      // Drop all the old interfaces.
      for (auto interfaceOp : existingInterfaceOps) {
        interfaceOp.erase();
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createCanonicalizeInterfacesPass() {
  return std::make_unique<CanonicalizeInterfacesPass>();
}

static PassRegistration<CanonicalizeInterfacesPass> pass(
    "iree-hal-canonicalize-interfaces",
    "Canonicalizes hal.interface ops across all hal.executables");

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
