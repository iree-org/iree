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

#ifndef IREE_COMPILER_SERIALIZATION_VM_FUNCTION_BUILDER_H_
#define IREE_COMPILER_SERIALIZATION_VM_FUNCTION_BUILDER_H_

#include "iree/compiler/Translation/Interpreter/Serialization/BytecodeWriter.h"
#include "iree/compiler/Translation/Interpreter/Serialization/VMFunctionTableBuilder.h"
#include "iree/schemas/interpreter_module_def_generated.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace iree_compiler {

class VMFunctionBuilder {
 public:
  using CustomWriterFn =
      std::function<LogicalResult(Operation *, BytecodeWriter *writer)>;

  VMFunctionBuilder(FuncOp function, VMFunctionTableBuilder *functionTable,
                    ::flatbuffers::FlatBufferBuilder *fbb);
  ~VMFunctionBuilder() = default;

  void RegisterCustomWriter(StringRef operationName, CustomWriterFn writerFn);

  LogicalResult ConvertBytecode();

  ::flatbuffers::Offset<iree::FunctionDef> Finish();

  ::flatbuffers::Offset<iree::TypeDef> SerializeType(
      Type type, ::flatbuffers::FlatBufferBuilder *fbb);
  ::flatbuffers::Offset<iree::MemRefTypeDef> SerializeMemRefType(
      const MemRefType &genericType, ::flatbuffers::FlatBufferBuilder *fbb);
  ::flatbuffers::Offset<iree::ElementTypeDef> SerializeElementType(
      const Type &genericType, ::flatbuffers::FlatBufferBuilder *fbb);

 private:
  LogicalResult BeginFunction(FuncOp function, BytecodeWriter *writer);
  LogicalResult EndFunction(FuncOp function, BytecodeWriter *writer);
  LogicalResult BeginBlock(Block *block, BytecodeWriter *writer);
  LogicalResult EndBlock(Block *block, Operation *op, BytecodeWriter *writer);

  LogicalResult WriteOperation(Block *block, Operation *baseOp,
                               BytecodeWriter *writer);

  llvm::StringMap<CustomWriterFn> customWriters_;

  MLIRContext *context_;
  FuncOp function_;
  VMFunctionTableBuilder *functionTable_;
  ::flatbuffers::FlatBufferBuilder *fbb_;
  ::flatbuffers::Offset<iree::BytecodeDef> bytecodeDef_;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_SERIALIZATION_VM_FUNCTION_BUILDER_H_
