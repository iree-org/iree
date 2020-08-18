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

static std::string buildFunctionName(IREE::VM::ModuleOp &moduleOp,
                                     IREE::VM::FuncOp &funcOp) {
  return std::string(moduleOp.getName()) + "_" + std::string(funcOp.getName());
}

static LogicalResult translateReturnOpToC(
    mlir::emitc::CppEmitter &emitter, IREE::VM::ReturnOp returnOp,
    llvm::raw_ostream &output, std::vector<std::string> resultNames) {
  for (std::tuple<Value, std::string> tuple :
       llvm::zip(returnOp.getOperands(), resultNames)) {
    Value operand = std::get<0>(tuple);
    std::string resultName = std::get<1>(tuple);
    output << "*" << resultName << " = " << emitter.getOrCreateName(operand)
           << ";\n";
  }

  output << "return iree_ok_status();\n";

  return success();
}

static LogicalResult translateOpToC(mlir::emitc::CppEmitter &emitter,
                                    Operation &op, llvm::raw_ostream &output,
                                    std::vector<std::string> resultNames) {
  if (auto returnOp = dyn_cast<IREE::VM::ReturnOp>(op))
    return translateReturnOpToC(emitter, returnOp, output, resultNames);
  if (succeeded(emitter.emitOperation(op))) {
    return success();
  }

  return failure();
}

static LogicalResult translateFunctionToC(mlir::emitc::CppEmitter &emitter,
                                          IREE::VM::ModuleOp &moduleOp,
                                          IREE::VM::FuncOp &funcOp,
                                          llvm::raw_ostream &output) {
  emitc::CppEmitter::Scope scope(emitter);

  output << "iree_status_t " << buildFunctionName(moduleOp, funcOp) << "(";

  mlir::emitc::interleaveCommaWithError(
      funcOp.getArguments(), output, [&](auto arg) -> LogicalResult {
        if (failed(emitter.emitType(arg.getType()))) {
          return failure();
        }
        output << " " << emitter.getOrCreateName(arg);
        return success();
      });

  if (funcOp.getNumResults() > 0) {
    output << ", ";
  }

  std::vector<std::string> resultNames;
  for (size_t idx = 0; idx < funcOp.getNumResults(); idx++) {
    std::string resultName("out");
    resultName.append(std::to_string(idx));
    resultNames.push_back(resultName);
  }

  mlir::emitc::interleaveCommaWithError(
      llvm::zip(funcOp.getType().getResults(), resultNames), output,
      [&](std::tuple<Type, std::string> tuple) -> LogicalResult {
        Type type = std::get<0>(tuple);
        std::string resultName = std::get<1>(tuple);

        if (failed(emitter.emitType(type))) {
          return failure();
        }
        output << " *" << resultName;
        return success();
      });

  output << ") {\n";

  for (auto &op : funcOp.getOps()) {
    if (failed(translateOpToC(emitter, op, output, resultNames))) {
      return failure();
    }
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
    if (failed(translateFunctionToC(emitter, moduleOp, funcOp, output))) {
      return failure();
    }

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
