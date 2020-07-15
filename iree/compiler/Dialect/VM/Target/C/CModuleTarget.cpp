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

#include "iree/compiler/Dialect/VM/Target/C/CModuleTarget.h"

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/ConvertVMToEmitC.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

LogicalResult translateModuleToC(IREE::VM::ModuleOp moduleOp,
                                 llvm::raw_ostream &output) {
  // TODO: implement translation
  output << "// c module stub\n";

  // TODO(simon-camp) remove debug print and refactor this into a test in
  // ConvertVMToEmitC
  output << moduleOp << "\n";

  return success();
}

LogicalResult translateModuleToC(mlir::ModuleOp outerModuleOp,
                                 llvm::raw_ostream &output) {
  if (failed(convertVMtoEmitC(outerModuleOp))) {
    return failure();
  };

  auto moduleOps = outerModuleOp.getOps<IREE::VM::ModuleOp>();
  if (moduleOps.empty()) {
    return outerModuleOp.emitError()
           << "outer module does not contain a vm.module op";
  }
  return translateModuleToC(*moduleOps.begin(), output);
}

LogicalResult convertVMtoEmitC(mlir::ModuleOp &moduleOp) {
  PassManager pm(moduleOp.getContext());

  pm.addPass(std::make_unique<ConvertVMToEmitCPass>());

  return pm.run(moduleOp);
}
}  // namespace VM
}  // namespace IREE

}  // namespace iree_compiler
}  // namespace mlir
