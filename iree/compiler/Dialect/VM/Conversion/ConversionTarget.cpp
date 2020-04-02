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

#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"

namespace mlir {
namespace iree_compiler {

// static
std::pair<mlir::ModuleOp, mlir::ModuleOp>
VMConversionTarget::nestModuleForConversion(mlir::ModuleOp outerModuleOp) {
  auto innerModuleOp = dyn_cast<ModuleOp>(outerModuleOp.getBody()->front());
  if (!innerModuleOp) {
    innerModuleOp =
        ModuleOp::create(outerModuleOp.getLoc(), outerModuleOp.getName());
    innerModuleOp.getBodyRegion().takeBody(outerModuleOp.getBodyRegion());
    outerModuleOp.getBodyRegion().getBlocks().push_back(new Block());
    OpBuilder builder = OpBuilder::atBlockEnd(outerModuleOp.getBody());
    builder.create<mlir::ModuleTerminatorOp>(outerModuleOp.getLoc());
    outerModuleOp.push_back(innerModuleOp);
  }
  return std::make_pair(outerModuleOp, innerModuleOp);
}

VMConversionTarget::VMConversionTarget(MLIRContext *context)
    : ConversionTarget(*context) {
  addLegalDialect<IREE::VM::VMDialect>();

  // NOTE: we need to allow the outermost std.module to be legal to support the
  // double-nesting (module { vm.module { ... } }).
  addDynamicallyLegalOp<mlir::ModuleOp>(+[](mlir::ModuleOp op) {
    return !op.getParentOp() || !isa<ModuleOp>(op.getParentOp());
  });
  addDynamicallyLegalOp<mlir::ModuleTerminatorOp>(
      +[](mlir::ModuleTerminatorOp op) {
        return !isa<IREE::VM::ModuleOp>(op.getParentOp());
      });
}

}  // namespace iree_compiler
}  // namespace mlir
