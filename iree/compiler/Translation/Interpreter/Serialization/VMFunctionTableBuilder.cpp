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

#include "iree/compiler/Translation/Interpreter/Serialization/VMFunctionTableBuilder.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace iree_compiler {

VMFunctionTableBuilder::VMFunctionTableBuilder(
    ::flatbuffers::FlatBufferBuilder *fbb)
    : fbb_(fbb) {}

bool VMFunctionTableBuilder::IsFunctionDeclared(FuncOp funcOp) {
  return functionSet_.count(funcOp.getName()) != 0;
}

LogicalResult VMFunctionTableBuilder::DeclareFunction(FuncOp funcOp,
                                                      LinkageType linkageType) {
  if (functionSet_.count(funcOp.getName())) {
    return funcOp.emitError() << "Function has already been declared/defined";
  }
  auto functionOrdinal = funcOp.getAttrOfType<IntegerAttr>("iree.ordinal");
  if (!functionOrdinal) {
    return funcOp.emitError() << "Ordinal not assigned to function";
  }
  int ordinal = functionOrdinal.getInt();
  functionDefs_.resize(
      std::max(functionDefs_.size(), static_cast<size_t>(ordinal) + 1u));
  functionSet_.insert({funcOp.getName()});
  switch (linkageType) {
    case LinkageType::kInternal:
      break;
    case LinkageType::kImport:
      importIndices_.push_back(ordinal);
      break;
    case LinkageType::kExport:
      exportIndices_.push_back(ordinal);
      break;
  }
  return success();
}

LogicalResult VMFunctionTableBuilder::DefineFunction(
    FuncOp funcOp, ::flatbuffers::Offset<iree::FunctionDef> functionDef) {
  auto functionOrdinal = funcOp.getAttrOfType<IntegerAttr>("iree.ordinal");
  if (!functionOrdinal) {
    return funcOp.emitError() << "Ordinal not assigned to function";
  }
  int ordinal = functionOrdinal.getInt();
  if (!functionDefs_[ordinal].IsNull()) {
    return funcOp.emitOpError() << "Function has already been defined";
  }
  functionDefs_[ordinal] = functionDef;
  return success();
}

::flatbuffers::Offset<iree::FunctionTableDef> VMFunctionTableBuilder::Finish() {
  auto functionsOffset = fbb_->CreateVector(functionDefs_);
  auto importsOffset = fbb_->CreateVector(importIndices_);
  auto exportsOffset = fbb_->CreateVector(exportIndices_);
  iree::FunctionTableDefBuilder ftdb(*fbb_);
  ftdb.add_functions(functionsOffset);
  ftdb.add_imports(importsOffset);
  ftdb.add_exports(exportsOffset);
  return ftdb.Finish();
}

}  // namespace iree_compiler
}  // namespace mlir
