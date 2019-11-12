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

#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

// Assigns per-category ordinals to module-level symbols in the module.
// Each ordinal is unique per-category and ordinals are contiguous starting from
// zero.
//
// NOTE: symbols are serialized in ordinal-order (hence the name!) and we have
// an opportunity here to set the layout of the final binaries, similar to how
// old-timey games would layout files on their spinning plastic discs to
// optimize the time spent moving a physical laser carridge around. Functions
// related to each other and global data accessed in proximity should be
// clustered together to make use of paging in memory mapped files.
class OrdinalAllocationPass
    : public OperationPass<OrdinalAllocationPass, ModuleOp> {
 public:
  void runOnOperation() override {
    Builder builder(&getContext());

    int nextFuncOrdinal = 0;
    int nextImportOrdinal = 0;
    int nextExportOrdinal = 0;
    int nextGlobalBytesOrdinal = 0;
    int nextGlobalRefOrdinal = 0;
    int nextRodataOrdinal = 0;
    for (auto &op : getOperation().getBlock().getOperations()) {
      Optional<int> ordinal = llvm::None;
      if (auto funcOp = dyn_cast<FuncOp>(op)) {
        if (funcOp.isExternal()) {
          ordinal = nextImportOrdinal++;
        } else {
          ordinal = nextFuncOrdinal++;
        }
      } else if (isa<ExportOp>(op)) {
        ordinal = nextExportOrdinal++;
      } else if (isa<GlobalI32Op>(op)) {
        ordinal = nextGlobalBytesOrdinal;
        nextGlobalBytesOrdinal += 4;
      } else if (isa<GlobalRefOp>(op)) {
        ordinal = nextGlobalRefOrdinal++;
      } else if (isa<RodataOp>(op)) {
        ordinal = nextRodataOrdinal++;
      }
      if (ordinal.hasValue()) {
        op.setAttr("ordinal", builder.getI32IntegerAttr(ordinal.getValue()));
      }
    }
  }
};

std::unique_ptr<OpPassBase<ModuleOp>> createOrdinalAllocationPass() {
  return std::make_unique<OrdinalAllocationPass>();
}

static PassRegistration<OrdinalAllocationPass> pass(
    "iree-vm-ordinal-allocation",
    "Assigns ordinals to function and global symbols");

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
