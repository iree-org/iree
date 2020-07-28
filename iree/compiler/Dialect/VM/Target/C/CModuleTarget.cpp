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
#include "third_party/mlir-emitc/include/emitc/Target/Cpp.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

static LogicalResult translateReturnOpToC(mlir::emitc::CppEmitter &emitter,
                                          IREE::VM::ReturnOp returnOp,
                                          llvm::raw_ostream &output) {
  int i = 0;
  for (auto operand : returnOp.getOperands()) {
    output << "*out" << i++ << " = " << emitter.getOrCreateName(operand)
           << ";\n";
  }

  output << "return IREE_STATUS_OK;\n";

  return success();
}

static LogicalResult translateOpToC(mlir::emitc::CppEmitter &emitter,
                                    Operation &op, llvm::raw_ostream &output) {
  if (auto returnOp = dyn_cast<IREE::VM::ReturnOp>(op))
    return translateReturnOpToC(emitter, returnOp, output);
  if (succeeded(emitter.emitOperation(op))) {
    return success();
  }

  return failure();
}

static LogicalResult translateFunctionToC(mlir::emitc::CppEmitter &emitter,
                                          IREE::VM::FuncOp funcOp,
                                          llvm::raw_ostream &output) {
  emitc::CppEmitter::Scope scope(emitter);

  output << "iree_status_t " << funcOp.getName() << "(";

  // function arguments
  int i = 0;
  const int totalCount = funcOp.getNumArguments() + funcOp.getNumResults();
  for (auto arg : funcOp.getArguments()) {
    i++;
    emitter.emitType(arg.getType());
    output << " " << emitter.getOrCreateName(arg);
    if (i < totalCount) output << ", ";
  }

  int j = 0;
  for (auto result : funcOp.getType().getResults()) {
    i++;
    emitter.emitType(result);
    output << " *out_" << j++;
    if (i < totalCount) output << ", ";
  }

  output << ") {\n";

  // function body
  for (auto &op : funcOp.getOps()) {
    if (failed(translateOpToC(emitter, op, output))) {
      return failure();
    };
  }

  output << "}\n";

  return success();
}

LogicalResult translateModuleToC(IREE::VM::ModuleOp moduleOp,
                                 llvm::raw_ostream &output) {
  mlir::emitc::CppEmitter emitter(output);
  mlir::emitc::CppEmitter::Scope scope(emitter);

  output << "#include \"vm_c_funcs.h\"\n";
  output << "\n";

  for (auto funcOp : moduleOp.getOps<IREE::VM::FuncOp>()) {
    if (failed(translateFunctionToC(emitter, funcOp, output))) {
      return failure();
    };

    output << "\n";
  }

  return success();
}

LogicalResult translateModuleToC(mlir::ModuleOp outerModuleOp,
                                 llvm::raw_ostream &output) {
  PassManager pm(outerModuleOp.getContext());

  pm.addPass(createConvertVMToEmitCPass());

  if (failed(pm.run(outerModuleOp))) {
    return failure();
  }

  auto moduleOps = outerModuleOp.getOps<ModuleOp>();
  if (moduleOps.empty()) {
    return outerModuleOp.emitError()
           << "outer module does not contain a vm.module op";
  }
  return translateModuleToC(*moduleOps.begin(), output);
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
