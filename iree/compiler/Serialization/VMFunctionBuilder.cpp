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

#include "iree/compiler/Serialization/VMFunctionBuilder.h"

#include "flatbuffers/flatbuffers.h"
#include "iree/compiler/IR/Dialect.h"
#include "iree/compiler/IR/Types.h"
#include "iree/compiler/Serialization/BytecodeTables.h"
#include "iree/compiler/Utils/Macros.h"
#include "iree/schemas/bytecode/bytecode_v0.h"
#include "iree/schemas/type_def_generated.h"
#include "third_party/llvm/llvm/include/llvm/ADT/STLExtras.h"
#include "third_party/llvm/llvm/include/llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Module.h"

namespace mlir {
namespace iree_compiler {

namespace {

LogicalResult WriteGenericIreeOp(Block *block, Operation *op,
                                 BytecodeWriter *writer) {
  // Strip the dialect name from the op name and lookup the opcode.
  // TODO(benvanik): adjust for supporting sequencer opcodes.

  auto opName = op->getName().getStringRef();
  auto dialect = op->getDialect();
  if (!dialect) {
    return op->emitOpError() << "Op does not belong to a registered dialect";
  }

  auto dialectNamespace = dialect->getNamespace();
  std::unique_ptr<OpcodeInfo> operandInfo;
  auto strippedOpName = opName.substr(opName.find('.') + 1).str();
  if (dialectNamespace == "iree_ll_seq") {
    auto opcode = GetSequencerOpcodeByName(strippedOpName);
    if (!opcode.hasValue()) {
      return op->emitOpError()
             << "No sequencer opcode found for op; is it a pseudo op?";
    }
    RETURN_IF_FAILURE(writer->WriteOpcode(opcode.getValue()));
    operandInfo =
        std::make_unique<OpcodeInfo>(GetSequencerOpcodeInfo(opcode.getValue()));
  } else if (dialectNamespace == "iree_ll_interp" ||
             // TODO(gcmn) remove special case for IREE dialect?
             dialectNamespace == IREEDialect::getDialectNamespace()) {
    auto opcode = GetInterpreterOpcodeByName(strippedOpName);
    if (!opcode.hasValue()) {
      return op->emitOpError()
             << "No interpreter opcode found for op; is it a pseudo op?";
    }
    RETURN_IF_FAILURE(writer->WriteOpcode(opcode.getValue()));
    operandInfo = std::make_unique<OpcodeInfo>(
        GetInterpreterOpcodeInfo(opcode.getValue()));
  } else {
    return op->emitOpError()
           << "Op belongs to unknown dialect " << dialectNamespace.str();
  }
  // Write inputs and outputs based on the bytecode encoding.
  int operandIndex = 0;
  int resultIndex = 0;
  for (int i = 0; i < llvm::array_lengthof(operandInfo->operands); ++i) {
    auto op_encoding = operandInfo->operands[i];
    if (op_encoding == iree::OperandEncoding::kNone) break;
    switch (op_encoding) {
      case iree::OperandEncoding::kInputSlot:
      case iree::OperandEncoding::kOutputSlot: {
        auto *value = op->getOperand(operandIndex++);
        RETURN_IF_FAILURE(writer->WriteLocal(value));
        break;
      }
      case iree::OperandEncoding::kVariadicInputSlots:
      case iree::OperandEncoding::kVariadicOutputSlots: {
        int count = op->getNumOperands() - operandIndex;
        RETURN_IF_FAILURE(writer->WriteCount(count));
        for (; count; --count) {
          auto *value = op->getOperand(operandIndex++);
          RETURN_IF_FAILURE(writer->WriteLocal(value));
        }
        break;
      }
      case iree::OperandEncoding::kResultSlot: {
        auto *value = op->getResult(resultIndex++);
        RETURN_IF_FAILURE(writer->WriteLocal(value));
        break;
      }
      case iree::OperandEncoding::kVariadicResultSlots: {
        int count = op->getNumResults() - resultIndex;
        RETURN_IF_FAILURE(writer->WriteCount(count));
        for (; count; --count) {
          auto *value = op->getResult(resultIndex++);
          RETURN_IF_FAILURE(writer->WriteLocal(value));
        }
        break;
      }
      case iree::OperandEncoding::kConstant:
      case iree::OperandEncoding::kFunctionOrdinal:
      case iree::OperandEncoding::kBlockOffset:
      case iree::OperandEncoding::kTypeIndex:
      case iree::OperandEncoding::kIndex:
      case iree::OperandEncoding::kIndexList:
      case iree::OperandEncoding::kCmpIPredicate:
      case iree::OperandEncoding::kCmpFPredicate:
        return op->emitOpError()
               << "Operand encoding " << static_cast<char>(op_encoding)
               << " not supported by generic writer for " << opName.str();
        return failure();
      default:
        return op->emitOpError()
               << "Operand encoding " << static_cast<char>(op_encoding) << " ("
               << static_cast<int>(op_encoding) << ") not recognized (typo?)";
    }
  }

  return success();
}

}  // namespace

VMFunctionBuilder::VMFunctionBuilder(FuncOp function,
                                     VMFunctionTableBuilder *functionTable,
                                     ::flatbuffers::FlatBufferBuilder *fbb)
    : context_(function.getContext()),
      function_(function),
      functionTable_(functionTable),
      fbb_(fbb) {}

void VMFunctionBuilder::RegisterCustomWriter(StringRef operationName,
                                             CustomWriterFn writerFn) {
  customWriters_.insert({operationName, writerFn});
}

LogicalResult VMFunctionBuilder::ConvertBytecode() {
  BytecodeWriter writer;
  sourceMap_ = {};

  RETURN_IF_FAILURE(BeginFunction(function_, &writer));
  for (auto &block : function_.getBlocks()) {
    RETURN_IF_FAILURE(BeginBlock(&block, &writer));
    for (auto &op : block.getOperations()) {
      if (failed(WriteOperation(&block, &op, &writer))) {
        op.emitError() << "Unable to serialize operation";
        return failure();
      }
    }
    RETURN_IF_FAILURE(EndBlock(&block, block.getTerminator(), &writer));
  }
  RETURN_IF_FAILURE(EndFunction(function_, &writer));

  int localCount = writer.local_count();
  auto bodyBytes = writer.Finish();
  auto bodyOffset = fbb_->CreateVector(
      reinterpret_cast<const int8_t *>(bodyBytes.data()), bodyBytes.size());
  iree::BytecodeDefBuilder bdb(*fbb_);
  bdb.add_local_count(localCount);
  bdb.add_contents(bodyOffset);
  bytecodeDef_ = bdb.Finish();

  return success();
}

::flatbuffers::Offset<iree::FunctionDef> VMFunctionBuilder::Finish() {
  using TypeDefVector =
      ::flatbuffers::Vector<::flatbuffers::Offset<iree::TypeDef>>;

  const auto &functionType = function_.getType();
  std::vector<::flatbuffers::Offset<iree::TypeDef>> inputs;
  for (const auto &type : functionType.getInputs()) {
    auto typeOffset = SerializeType(type, fbb_);
    if (typeOffset.IsNull()) return {};
    inputs.push_back(typeOffset);
  }
  ::flatbuffers::Offset<TypeDefVector> inputsOffset;
  if (!inputs.empty()) {
    inputsOffset = fbb_->CreateVector(inputs);
  }

  std::vector<::flatbuffers::Offset<iree::TypeDef>> results;
  for (const auto &type : functionType.getResults()) {
    auto typeOffset = SerializeType(type, fbb_);
    if (typeOffset.IsNull()) return {};
    results.push_back(typeOffset);
  }
  ::flatbuffers::Offset<TypeDefVector> resultsOffset;
  if (!results.empty()) {
    resultsOffset = fbb_->CreateVector(results);
  }
  iree::FunctionTypeDefBuilder ftb(*fbb_);
  ftb.add_inputs(inputsOffset);
  ftb.add_results(resultsOffset);
  auto functionTypeOffset = ftb.Finish();

  // TODO(benvanik): strip names of internal functions.
  auto nameOffset = fbb_->CreateString(function_.getName().str());
  iree::FunctionDefBuilder fdb(*fbb_);
  fdb.add_name(nameOffset);
  fdb.add_type(functionTypeOffset);
  fdb.add_bytecode(bytecodeDef_);
  return fdb.Finish();
}

LogicalResult VMFunctionBuilder::BeginFunction(FuncOp function,
                                               BytecodeWriter *writer) {
  // Assign value slots for all arguments and results.
  // Keeping them at the front will make it easier to find during debugging
  // and makes spans easier to compute at runtime.
  for (auto argument : function.getArguments()) {
    RETURN_IF_FAILURE(writer->PrepareLocal(argument));
  }
  return success();
}

LogicalResult VMFunctionBuilder::EndFunction(FuncOp function,
                                             BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->FixupOffsets());
  return success();
}

LogicalResult VMFunctionBuilder::BeginBlock(Block *block,
                                            BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->MarkBlockOffset(block));
  return success();
}

LogicalResult VMFunctionBuilder::EndBlock(Block *block, Operation *op,
                                          BytecodeWriter *writer) {
  return success();
}

LogicalResult VMFunctionBuilder::WriteOperation(Block *block, Operation *baseOp,
                                                BytecodeWriter *writer) {
  if (!baseOp->getLoc().isa<UnknownLoc>()) {
    sourceMap_.locations.push_back({writer->offset(), baseOp->getLoc()});
  }

  // Check registered writers first to allow overrides.
  auto writerIt = customWriters_.find(baseOp->getName().getStringRef());
  if (writerIt != customWriters_.end()) {
    return writerIt->second(baseOp, writer);
  }

  // Fallback to using the generic writer.
  if (baseOp->getAbstractOperation()->dialect.getNamespace().startswith(
          "iree")) {
    RETURN_IF_FAILURE(WriteGenericIreeOp(block, baseOp, writer));
  } else {
    return baseOp->emitError()
           << "Unsupported op " << baseOp->getName().getStringRef().str()
           << "; incorrectly outlined or not yet implemented";
  }
  return success();
}

::flatbuffers::Offset<iree::TypeDef> VMFunctionBuilder::SerializeType(
    Type type, ::flatbuffers::FlatBufferBuilder *fbb) {
  ::flatbuffers::Offset<void> typeDefUnion;
  iree::TypeDefUnion typeUnionType;
  if (auto memRefType = type.dyn_cast<MemRefType>()) {
    auto memRefTypeOffset = SerializeMemRefType(memRefType, fbb_);
    if (memRefTypeOffset.IsNull()) return {};
    typeDefUnion = memRefTypeOffset.Union();
    typeUnionType = iree::TypeDefUnion::MemRefTypeDef;
  } else if (auto deviceType = type.dyn_cast<DeviceType>()) {
    typeDefUnion = iree::CreateDeviceTypeDef(*fbb).Union();
    typeUnionType = iree::TypeDefUnion::DeviceTypeDef;
  } else if (auto commandBufferType = type.dyn_cast<CommandBufferType>()) {
    typeDefUnion = iree::CreateCommandBufferTypeDef(*fbb).Union();
    typeUnionType = iree::TypeDefUnion::CommandBufferTypeDef;
  } else if (auto eventType = type.dyn_cast<EventType>()) {
    typeDefUnion = iree::CreateEventTypeDef(*fbb).Union();
    typeUnionType = iree::TypeDefUnion::EventTypeDef;
  } else if (auto semaphoreType = type.dyn_cast<SemaphoreType>()) {
    typeDefUnion = iree::CreateSemaphoreTypeDef(*fbb).Union();
    typeUnionType = iree::TypeDefUnion::SemaphoreTypeDef;
  } else if (auto fenceType = type.dyn_cast<FenceType>()) {
    typeDefUnion = iree::CreateFenceTypeDef(*fbb).Union();
    typeUnionType = iree::TypeDefUnion::FenceTypeDef;
  } else {
    function_.emitError() << "Function " << function_.getName().str()
                          << " has unsupported I/O with type " << type;
    return {};
  }

  iree::TypeDefBuilder tdb(*fbb);
  tdb.add_type_union_type(typeUnionType);
  tdb.add_type_union(typeDefUnion);
  return tdb.Finish();
}

::flatbuffers::Offset<iree::MemRefTypeDef>
VMFunctionBuilder::SerializeMemRefType(const MemRefType &type,
                                       ::flatbuffers::FlatBufferBuilder *fbb) {
  auto elementTypeOffset = SerializeElementType(type.getElementType(), fbb);
  if (elementTypeOffset.IsNull()) return {};
  std::vector<int> shape;
  for (int dim : type.getShape()) {
    shape.push_back(dim);
  }
  auto shapeOffset = fbb->CreateVector(shape);
  iree::MemRefTypeDefBuilder tb(*fbb);
  tb.add_element_type(elementTypeOffset);
  tb.add_shape(shapeOffset);
  tb.add_memory_space(type.getMemorySpace());
  return tb.Finish();
}

::flatbuffers::Offset<iree::ElementTypeDef>
VMFunctionBuilder::SerializeElementType(const Type &genericType,
                                        ::flatbuffers::FlatBufferBuilder *fbb) {
  ::flatbuffers::Offset<void> typeDefUnion;
  iree::ElementTypeDefUnion typeUnionType;
  if (auto type = genericType.dyn_cast<FloatType>()) {
    iree::FloatTypeDefBuilder tb(*fbb);
    tb.add_width(type.getWidth());
    typeDefUnion = tb.Finish().Union();
    typeUnionType = iree::ElementTypeDefUnion::FloatTypeDef;
  } else if (auto type = genericType.dyn_cast<IntegerType>()) {
    iree::IntegerTypeDefBuilder tb(*fbb);
    tb.add_width(type.getWidth());
    typeDefUnion = tb.Finish().Union();
    typeUnionType = iree::ElementTypeDefUnion::IntegerTypeDef;
  } else if (auto type = genericType.dyn_cast<OpaqueType>()) {
    auto dialectOffset = fbb->CreateString(type.getDialectNamespace().c_str());
    auto typeDataOffset = fbb->CreateString(type.getTypeData().data());
    iree::UnknownTypeDefBuilder tb(*fbb);
    tb.add_dialect(dialectOffset);
    tb.add_type_data(typeDataOffset);
    typeDefUnion = tb.Finish().Union();
    typeUnionType = iree::ElementTypeDefUnion::UnknownTypeDef;
  } else {
    function_.emitError()
        << "Unimplemented type encoding: " << genericType
        << "; ensure IREE lowering passes are converting types to the IREE "
           "set";
    return {};
  }

  iree::ElementTypeDefBuilder tdb(*fbb);
  tdb.add_type_union_type(typeUnionType);
  tdb.add_type_union(typeDefUnion);
  return tdb.Finish();
}

}  // namespace iree_compiler
}  // namespace mlir
