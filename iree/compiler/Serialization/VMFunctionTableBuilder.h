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

#ifndef IREE_COMPILER_SERIALIZATION_VM_FUNCTION_TABLE_BUILDER_H_
#define IREE_COMPILER_SERIALIZATION_VM_FUNCTION_TABLE_BUILDER_H_

#include <string>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "iree/compiler/Serialization/VMSourceMapBuilder.h"
#include "iree/schemas/function_def_generated.h"
#include "iree/schemas/function_table_def_generated.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"

namespace mlir {
namespace iree_compiler {

enum class LinkageType {
  kInternal,
  kImport,
  kExport,
};

class VMFunctionTableBuilder {
 public:
  explicit VMFunctionTableBuilder(::flatbuffers::FlatBufferBuilder *fbb);

  int max_function_ordinal() const { return functionDefs_.size(); }

  ArrayRef<VMFunctionSourceMap> function_source_maps() {
    return llvm::makeArrayRef(functionSourceMaps_);
  }

  // Returns true if |funcOp| has already been declared in the table.
  bool IsFunctionDeclared(FuncOp funcOp);

  // Declares |funcOp| with the given |linkageType|.
  // Fails if the function has already been declared or defined.
  LogicalResult DeclareFunction(FuncOp funcOp, LinkageType linkageType);

  // Defines |funcOp| using the given |functionDef|.
  LogicalResult DefineFunction(
      FuncOp funcOp, ::flatbuffers::Offset<iree::FunctionDef> functionDef,
      VMFunctionSourceMap functionSourceMap);

  ::flatbuffers::Offset<iree::FunctionTableDef> Finish();

 private:
  ::flatbuffers::FlatBufferBuilder *fbb_;
  llvm::StringSet<> functionSet_;
  std::vector<::flatbuffers::Offset<iree::FunctionDef>> functionDefs_;
  std::vector<VMFunctionSourceMap> functionSourceMaps_;
  std::vector<int> importIndices_;
  std::vector<int> exportIndices_;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_SERIALIZATION_VM_FUNCTION_TABLE_BUILDER_H_
