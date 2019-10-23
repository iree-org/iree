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

#include "iree/compiler/Serialization/BytecodeWriter.h"

#include <algorithm>

#include "iree/compiler/Utils/Macros.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {

LogicalResult BytecodeWriter::WriteCount(int count) {
  if (count > UINT8_MAX) {
    // TODO(benvanik): varints?
    llvm::errs() << "Too many items: " << count
                 << "; only 0-UINT8_MAX are supported";
    return failure();
  }
  return WriteUint8(static_cast<uint8_t>(count));
}

LogicalResult BytecodeWriter::WriteTypeIndex(Type type) {
  iree::BuiltinType type_index;
  if (type.isInteger(8)) {
    type_index = iree::BuiltinType::kI8;
  } else if (type.isInteger(16)) {
    type_index = iree::BuiltinType::kI16;
  } else if (type.isInteger(32)) {
    type_index = iree::BuiltinType::kI32;
  } else if (type.isInteger(64)) {
    type_index = iree::BuiltinType::kI64;
  } else if (type.isF16()) {
    type_index = iree::BuiltinType::kF16;
  } else if (type.isF32()) {
    type_index = iree::BuiltinType::kF32;
  } else if (type.isF64()) {
    type_index = iree::BuiltinType::kF64;
  } else {
    // TODO(benvanik): support unknown types as BuiltinType::kOpaque?
    return emitError(UnknownLoc::get(type.getContext()))
           << "Type " << type << " cannot be represented by a builtin type";
  }
  return WriteUint8(static_cast<uint8_t>(type_index));
}

LogicalResult BytecodeWriter::WriteFunctionOrdinal(FuncOp function) {
  auto functionOrdinal = function.getAttrOfType<IntegerAttr>("iree.ordinal");
  if (!functionOrdinal) {
    return function.emitError() << "Ordinal not assigned to function";
  }
  RETURN_IF_FAILURE(WriteUint32(functionOrdinal.getInt()));
  return success();
}

LogicalResult BytecodeWriter::WriteImportOrdinal(FuncOp function) {
  // For now this is the same as internal function ordinals, though we could
  // probably shrink it.
  return WriteFunctionOrdinal(function);
}

LogicalResult BytecodeWriter::WriteConstant(MemRefType memRefType,
                                            Attribute baseAttr) {
  // All types are memrefs, so we only need the element type.
  RETURN_IF_FAILURE(WriteTypeIndex(memRefType.getElementType()));

  // Write shape (we could optimize this for cases of scalars and such).
  RETURN_IF_FAILURE(WriteCount(memRefType.getRank()));
  for (int i = 0; i < memRefType.getRank(); ++i) {
    RETURN_IF_FAILURE(WriteInt32(memRefType.getDimSize(i)));
  }

  if (auto attr = baseAttr.dyn_cast<SplatElementsAttr>()) {
    RETURN_IF_FAILURE(
        WriteUint8(static_cast<uint8_t>(iree::ConstantEncoding::kSplat)));
    return WriteAttributeData(attr.getSplatValue());
  }
  RETURN_IF_FAILURE(
      WriteUint8(static_cast<uint8_t>(iree::ConstantEncoding::kDense)));
  return WriteAttributeData(baseAttr);
}

LogicalResult BytecodeWriter::WriteAttributeData(Attribute baseAttr) {
  if (auto attr = baseAttr.dyn_cast<BoolAttr>()) {
    return WriteUint8(attr.getValue() ? 1 : 0);
  } else if (auto attr = baseAttr.dyn_cast<IntegerAttr>()) {
    if (attr.getType().isIndex()) {
      int32_t value = static_cast<int32_t>(attr.getInt());
      return WriteBytes(&value, 4);
    } else {
      int bitWidth = attr.getValue().getBitWidth();
      switch (bitWidth) {
        case 8:
        case 16:
        case 32:
        case 64:
          return WriteBytes(attr.getValue().getRawData(), bitWidth / 8);
        default:
          return emitError(UnknownLoc::get(baseAttr.getContext()))
                 << "Bit width for integers must be one of 8,16,32,64; others "
                    "not implemented: "
                 << bitWidth;
      }
    }
  } else if (auto attr = baseAttr.dyn_cast<FloatAttr>()) {
    int bitWidth = attr.getType().getIntOrFloatBitWidth();
    auto bitcastValue = attr.getValue().bitcastToAPInt();
    switch (bitWidth) {
      case 16:
      case 32:
      case 64:
        return WriteBytes(bitcastValue.getRawData(), bitWidth / 8);
      default:
        return emitError(UnknownLoc::get(baseAttr.getContext()))
               << "Bit width for floats must be one of 16,32,64; others "
                  "not implemented: "
               << bitWidth;
    }
  } else if (auto attr = baseAttr.dyn_cast<StringAttr>()) {
    // TODO(benvanik): other attribute encodings.
  } else if (auto attr = baseAttr.dyn_cast<ArrayAttr>()) {
    // TODO(benvanik): other attribute encodings.
  } else if (auto attr = baseAttr.dyn_cast<AffineMapAttr>()) {
    // TODO(benvanik): other attribute encodings.
  } else if (auto attr = baseAttr.dyn_cast<IntegerSetAttr>()) {
    // TODO(benvanik): other attribute encodings.
  } else if (auto attr = baseAttr.dyn_cast<TypeAttr>()) {
    // TODO(benvanik): other attribute encodings.
  } else if (auto attr = baseAttr.dyn_cast<SymbolRefAttr>()) {
    // TODO(benvanik): other attribute encodings.
  } else if (auto attr = baseAttr.dyn_cast<SplatElementsAttr>()) {
    return WriteAttributeData(attr.getSplatValue());
  } else if (auto attr = baseAttr.dyn_cast<DenseIntElementsAttr>()) {
    int elementCount = attr.getType().getNumElements();
    if (elementCount == 0) {
      return success();
    }
    int bitWidth = attr.getType().getElementTypeBitWidth();
    int byteWidth = bitWidth / 8;
    auto dst = ReserveBytes(elementCount * byteWidth);
    if (dst.empty()) return failure();
    uint8_t *dstPtr = dst.data();
    for (auto element : attr) {
      assert(element.getBitWidth() == bitWidth);
      std::memcpy(dstPtr, element.getRawData(), byteWidth);
      dstPtr += byteWidth;
    }
    return success();
  } else if (auto attr = baseAttr.dyn_cast<DenseFPElementsAttr>()) {
    int elementCount = attr.getType().getNumElements();
    if (elementCount == 0) {
      return success();
    }
    int bitWidth = attr.getType().getElementTypeBitWidth();
    auto dst = ReserveBytes(elementCount * bitWidth / 8);
    if (dst.empty()) return failure();
    uint8_t *dstPtr = dst.data();
    for (auto element : attr) {
      auto bitcastValue = element.bitcastToAPInt();
      std::memcpy(dstPtr, bitcastValue.getRawData(),
                  bitcastValue.getBitWidth() / 8);
      dstPtr += bitWidth / 8;
    }
    return success();
  } else if (auto attr = baseAttr.dyn_cast<DenseElementsAttr>()) {
    // TODO(benvanik): other attribute encodings.
  } else if (auto attr = baseAttr.dyn_cast<OpaqueElementsAttr>()) {
    // TODO(benvanik): other attribute encodings.
  } else if (auto attr = baseAttr.dyn_cast<SparseElementsAttr>()) {
    // TODO(benvanik): other attribute encodings.
  }
  return emitError(UnknownLoc::get(baseAttr.getContext()))
         << "Serializer for attribute kind "
         << static_cast<int>(baseAttr.getKind()) << " not implemented";
}

Optional<int> BytecodeWriter::LookupLocalOrdinal(Value *value) {
  int ordinal;
  auto it = localMap_.find(value);
  if (it != localMap_.end()) {
    ordinal = it->second;
  } else {
    ordinal = localMap_.size();
    localMap_.insert({value, ordinal});
  }
  if (ordinal > UINT16_MAX) {
    // TODO(benvanik): varints?
    emitError(UnknownLoc::get(value->getContext()))
        << "Too many ordinals: " << ordinal
        << "; only 0-UINT16_MAX are supported";
    return llvm::None;
  }
  return ordinal;
}

LogicalResult BytecodeWriter::PrepareLocal(Value *value) {
  if (!LookupLocalOrdinal(value).hasValue()) return failure();
  return success();
}

LogicalResult BytecodeWriter::WriteLocal(Value *value) {
  auto ordinal = LookupLocalOrdinal(value);
  if (!ordinal.hasValue()) {
    return failure();
  }
  if (ordinal.getValue() > UINT16_MAX) {
    // TODO(benvanik): varints?
    return emitError(UnknownLoc::get(value->getContext()))
           << "Too many locals: " << ordinal.getValue()
           << "; only 0-UINT16_MAX are supported";
  }
  return WriteUint16(static_cast<uint16_t>(ordinal.getValue()));
}

LogicalResult BytecodeWriter::WriteLocals(
    llvm::iterator_range<Operation::operand_iterator> values) {
  int count = std::distance(values.begin(), values.end());
  RETURN_IF_FAILURE(WriteCount(count));
  for (auto *value : values) {
    RETURN_IF_FAILURE(WriteLocal(value));
  }
  return success();
}

LogicalResult BytecodeWriter::WriteLocals(
    llvm::iterator_range<Operation::result_iterator> values) {
  int count = std::distance(values.begin(), values.end());
  RETURN_IF_FAILURE(WriteCount(count));
  for (auto *value : values) {
    RETURN_IF_FAILURE(WriteLocal(value));
  }
  return success();
}

MutableArrayRef<uint8_t> BytecodeWriter::ReserveBytes(size_t dataLength) {
  int offset = bytecode_.size();
  bytecode_.resize(offset + dataLength);
  return MutableArrayRef<uint8_t>(
      reinterpret_cast<uint8_t *>(bytecode_.data()) + offset, dataLength);
}

LogicalResult BytecodeWriter::WriteBytes(const void *data, size_t dataLength) {
  auto dst = ReserveBytes(dataLength);
  if (dataLength != dst.size()) {
    return failure();
  }
  std::memcpy(dst.data(), data, dst.size());
  return success();
}

LogicalResult BytecodeWriter::WriteUint8(uint8_t value) {
  return WriteBytes(&value, sizeof(value));
}

LogicalResult BytecodeWriter::WriteUint16(uint16_t value) {
  return WriteBytes(&value, sizeof(value));
}

LogicalResult BytecodeWriter::WriteInt32(int32_t value) {
  return WriteBytes(&value, sizeof(value));
}

LogicalResult BytecodeWriter::WriteUint32(uint32_t value) {
  return WriteBytes(&value, sizeof(value));
}

LogicalResult BytecodeWriter::WriteElementsAttrInt32(ElementsAttr attr) {
  int elementCount = attr.getType().getNumElements();
  RETURN_IF_FAILURE(WriteCount(elementCount));
  for (auto value : attr.getValues<int32_t>()) {
    RETURN_IF_FAILURE(WriteInt32(value));
  }
  return success();
}

LogicalResult BytecodeWriter::WriteShapePieces(const ShapedType &type) {
  RETURN_IF_FAILURE(WriteCount(type.getRank()));
  for (int64_t dim : type.getShape()) {
    RETURN_IF_FAILURE(WriteInt32(dim));
  }
  return success();
}

LogicalResult BytecodeWriter::WriteShapePieces(ElementsAttr pieces) {
  return WriteElementsAttrInt32(pieces);
}

LogicalResult BytecodeWriter::MarkBlockOffset(Block *block) {
  blockOffsets_[block] = bytecode_.size();
  return success();
}

LogicalResult BytecodeWriter::WriteBlockOffset(Block *targetBlock) {
  // Reserve space for the offset and stash for later fixup.
  blockOffsetFixups_.push_back({targetBlock, bytecode_.size()});
  bytecode_.resize(bytecode_.size() + sizeof(int32_t));
  return success();
}

LogicalResult BytecodeWriter::FixupOffsets() {
  for (const auto &fixup : blockOffsetFixups_) {
    auto it = blockOffsets_.find(fixup.first);
    if (it == blockOffsets_.end()) {
      llvm::errs() << "Block offset not found: " << fixup.first;
      return failure();
    }
    std::memcpy(bytecode_.data() + fixup.second, &it->second, sizeof(int32_t));
  }
  blockOffsetFixups_.clear();
  return success();
}

std::vector<uint8_t> BytecodeWriter::Finish() {
  localMap_.clear();
  return std::move(bytecode_);
}

}  // namespace iree_compiler
}  // namespace mlir
