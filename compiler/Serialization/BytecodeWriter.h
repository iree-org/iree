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

#ifndef IREE_COMPILER_SERIALIZATION_BYTECODE_WRITER_H_
#define IREE_COMPILER_SERIALIZATION_BYTECODE_WRITER_H_

#include <cstddef>
#include <utility>
#include <vector>

#include "compiler/IR/StructureOps.h"
#include "llvm/ADT/Optional.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "schemas/bytecode/bytecode_v0.h"

namespace mlir {
namespace iree_compiler {

class BytecodeWriter {
 public:
  int offset() const { return bytecode_.size(); }

  int local_count() const { return localMap_.size(); }

  template <typename T>
  LogicalResult WriteOpcode(T value) {
    static_assert(sizeof(T) == sizeof(uint8_t), "Opcode enum size mismatch");
    return WriteUint8(static_cast<uint8_t>(value));
  }

  LogicalResult WriteCount(int count);

  LogicalResult WriteTypeIndex(Type type);

  LogicalResult WriteFunctionOrdinal(FuncOp function);
  LogicalResult WriteImportOrdinal(FuncOp function);

  LogicalResult WriteConstant(MemRefType memRefType, Attribute baseAttr);
  LogicalResult WriteAttributeData(Attribute baseAttr);

  llvm::Optional<int> LookupLocalOrdinal(Value *value);
  LogicalResult PrepareLocal(Value *value);
  LogicalResult WriteLocal(Value *value);
  LogicalResult WriteLocals(
      llvm::iterator_range<Operation::operand_iterator> values);
  LogicalResult WriteLocals(
      llvm::iterator_range<Operation::result_iterator> values);

  LogicalResult WriteBytes(const void *data, size_t dataLength);
  MutableArrayRef<uint8_t> ReserveBytes(size_t dataLength);
  LogicalResult WriteUint8(uint8_t value);
  LogicalResult WriteUint16(uint16_t value);
  LogicalResult WriteInt32(int32_t value);
  LogicalResult WriteUint32(uint32_t value);

  LogicalResult WriteElementsAttrInt32(ElementsAttr attr);

  LogicalResult WriteShapePieces(const ShapedType &type);
  LogicalResult WriteShapePieces(ElementsAttr pieces);

  LogicalResult MarkBlockOffset(Block *block);
  LogicalResult WriteBlockOffset(Block *targetBlock);
  LogicalResult FixupOffsets();

  std::vector<uint8_t> Finish();

 private:
  std::vector<uint8_t> bytecode_;

  llvm::DenseMap<Value *, int> localMap_;

  llvm::DenseMap<Block *, size_t> blockOffsets_;
  std::vector<std::pair<Block *, size_t>> blockOffsetFixups_;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_SERIALIZATION_BYTECODE_WRITER_H_
