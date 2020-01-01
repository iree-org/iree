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

#include "iree/compiler/Translation/Interpreter/IR/OpWriters.h"

#include "iree/compiler/Translation/Interpreter/IR/LLOps.h"
#include "iree/compiler/Translation/Interpreter/Serialization/BytecodeWriter.h"
#include "iree/compiler/Translation/Interpreter/Utils/Macros.h"
#include "iree/schemas/bytecode/interpreter_bytecode_v0.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace iree_compiler {

namespace {

//===----------------------------------------------------------------------===//
// Sequencer ops
//===----------------------------------------------------------------------===//

LogicalResult writeOp(IREEInterp::LL::ConstantOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::InterpreterOpcode::kConstant));
  auto memrefType = op.getType().dyn_cast<MemRefType>();
  if (!memrefType) {
    return op.emitOpError()
           << "Constant has an unsupported type; must be a memref: "
           << op.getType();
  }
  RETURN_IF_FAILURE(writer->WriteConstant(memrefType, op.getAttr("value")));
  RETURN_IF_FAILURE(writer->WriteLocal(op.getResult()));
  return success();
}

LogicalResult writeOp(IREEInterp::LL::CallOp op, BytecodeWriter *writer) {
  auto module = op.getOperation()->getParentOfType<ModuleOp>();
  auto callee = module.lookupSymbol<FuncOp>(op.getCallee());
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::InterpreterOpcode::kCall));
  RETURN_IF_FAILURE(writer->WriteFunctionOrdinal(callee));
  RETURN_IF_FAILURE(writer->WriteLocals(op.getArgOperands()));
  RETURN_IF_FAILURE(writer->WriteLocals(op.getResults()));
  return success();
}

LogicalResult WriteConvertOperands(Operation *op, BytecodeWriter *writer) {
  auto src = op->getOperand(0);
  RETURN_IF_FAILURE(
      writer->WriteTypeIndex(getElementTypeOrSelf(src->getType())));
  RETURN_IF_FAILURE(writer->WriteLocal(src));
  auto dst = op->getOperand(1);
  RETURN_IF_FAILURE(
      writer->WriteTypeIndex(getElementTypeOrSelf(dst->getType())));
  RETURN_IF_FAILURE(writer->WriteLocal(dst));
  return success();
}

LogicalResult writeOp(IREEInterp::LL::ConvertSSOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::InterpreterOpcode::kConvertSS));
  return WriteConvertOperands(op, writer);
}

LogicalResult writeOp(IREEInterp::LL::ConvertUUOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::InterpreterOpcode::kConvertUU));
  return WriteConvertOperands(op, writer);
}

LogicalResult writeOp(IREEInterp::LL::ConvertSUOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::InterpreterOpcode::kConvertSU));
  return WriteConvertOperands(op, writer);
}

LogicalResult writeOp(IREEInterp::LL::ConvertUSOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::InterpreterOpcode::kConvertUS));
  return WriteConvertOperands(op, writer);
}

LogicalResult writeOp(IREEInterp::LL::BranchOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::InterpreterOpcode::kBranch));
  RETURN_IF_FAILURE(writer->WriteBlockOffset(op.getDest()));
  RETURN_IF_FAILURE(writer->WriteCount(op.getNumOperands()));
  for (int i = 0; i < op.getNumOperands(); ++i) {
    // Copy src->dst.
    RETURN_IF_FAILURE(writer->WriteLocal(op.getOperand(i)));
    RETURN_IF_FAILURE(writer->WriteLocal(op.getDest()->getArgument(i)));
  }
  return success();
}

LogicalResult writeOp(IREEInterp::LL::CondBranchOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::InterpreterOpcode::kCondBranch));
  RETURN_IF_FAILURE(writer->WriteLocal(op.getCondition()));
  RETURN_IF_FAILURE(writer->WriteBlockOffset(op.getTrueDest()));
  RETURN_IF_FAILURE(writer->WriteCount(op.getNumTrueOperands()));
  for (int i = 0; i < op.getNumTrueOperands(); ++i) {
    // Copy src->dst.
    RETURN_IF_FAILURE(writer->WriteLocal(op.getTrueOperand(i)));
    RETURN_IF_FAILURE(writer->WriteLocal(op.getTrueDest()->getArgument(i)));
  }
  RETURN_IF_FAILURE(writer->WriteBlockOffset(op.getFalseDest()));
  RETURN_IF_FAILURE(writer->WriteCount(op.getNumFalseOperands()));
  for (int i = 0; i < op.getNumFalseOperands(); ++i) {
    // Copy src->dst.
    RETURN_IF_FAILURE(writer->WriteLocal(op.getFalseOperand(i)));
    RETURN_IF_FAILURE(writer->WriteLocal(op.getFalseDest()->getArgument(i)));
  }
  return success();
}

LogicalResult writeOp(IREEInterp::LL::CmpIOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::InterpreterOpcode::kCmpI));
  RETURN_IF_FAILURE(
      writer->WriteUint8(static_cast<uint8_t>(op.predicate().getZExtValue())));
  RETURN_IF_FAILURE(writer->WriteLocal(op.getOperand(0)));
  RETURN_IF_FAILURE(writer->WriteLocal(op.getOperand(1)));
  RETURN_IF_FAILURE(writer->WriteLocal(op.getOperand(2)));
  return success();
}

LogicalResult writeOp(IREEInterp::LL::CmpFOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::InterpreterOpcode::kCmpF));
  RETURN_IF_FAILURE(
      writer->WriteUint8(static_cast<uint8_t>(op.predicate().getZExtValue())));
  RETURN_IF_FAILURE(writer->WriteLocal(op.getOperand(0)));
  RETURN_IF_FAILURE(writer->WriteLocal(op.getOperand(1)));
  RETURN_IF_FAILURE(writer->WriteLocal(op.getOperand(2)));
  return success();
}

LogicalResult writeOp(IREEInterp::LL::AllocHeapOp op, BytecodeWriter *writer) {
  auto memrefType = op.getType().cast<MemRefType>();
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::InterpreterOpcode::kAllocHeap));
  RETURN_IF_FAILURE(writer->WriteInt32(0));
  RETURN_IF_FAILURE(writer->WriteTypeIndex(memrefType.getElementType()));
  RETURN_IF_FAILURE(writer->WriteShapePieces(memrefType));
  RETURN_IF_FAILURE(writer->WriteLocals(op.getOperands()));
  RETURN_IF_FAILURE(writer->WriteLocal(op.getResult()));
  return success();
}

LogicalResult writeOp(IREEInterp::LL::StaticCopyOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::InterpreterOpcode::kStaticCopy));
  RETURN_IF_FAILURE(writer->WriteLocal(op.src()));
  RETURN_IF_FAILURE(writer->WriteShapePieces(op.srcIndices()));
  RETURN_IF_FAILURE(writer->WriteLocal(op.dst()));
  RETURN_IF_FAILURE(writer->WriteShapePieces(op.dstIndices()));
  RETURN_IF_FAILURE(writer->WriteShapePieces(op.lengths()));
  return success();
}

LogicalResult writeReduceOperands(Operation *op, BytecodeWriter *writer,
                                  APInt dimension) {
  RETURN_IF_FAILURE(writer->WriteLocal(op->getOperand(0)));
  RETURN_IF_FAILURE(writer->WriteLocal(op->getOperand(1)));
  RETURN_IF_FAILURE(writer->WriteInt32(dimension.getZExtValue()));
  RETURN_IF_FAILURE(writer->WriteLocal(op->getOperand(2)));
  return success();
}

LogicalResult writeOp(IREEInterp::LL::ReduceSumIOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::InterpreterOpcode::kReduceSumI));
  return writeReduceOperands(op, writer, op.dimension());
}

LogicalResult writeOp(IREEInterp::LL::ReduceSumFOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::InterpreterOpcode::kReduceSumF));
  return writeReduceOperands(op, writer, op.dimension());
}

LogicalResult writeOp(IREEInterp::LL::ReduceMinIOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::InterpreterOpcode::kReduceMinI));
  return writeReduceOperands(op, writer, op.dimension());
}

LogicalResult writeOp(IREEInterp::LL::ReduceMinFOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::InterpreterOpcode::kReduceMinF));
  return writeReduceOperands(op, writer, op.dimension());
}

LogicalResult writeOp(IREEInterp::LL::ReduceMaxIOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::InterpreterOpcode::kReduceMaxI));
  return writeReduceOperands(op, writer, op.dimension());
}

LogicalResult writeOp(IREEInterp::LL::ReduceMaxFOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::InterpreterOpcode::kReduceMaxF));
  return writeReduceOperands(op, writer, op.dimension());
}

}  // namespace

void registerInterpreterCustomWriters(VMFunctionBuilder *builder) {
#define REGISTER_CUSTOM_WRITER_IMPL(op_type)       \
  builder->RegisterCustomWriter(                   \
      op_type::getOperationName(),                 \
      +[](Operation *op, BytecodeWriter *writer) { \
        return writeOp(cast<op_type>(op), writer); \
      });
  REGISTER_CUSTOM_WRITER_IMPL(IREEInterp::LL::ConstantOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREEInterp::LL::CallOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREEInterp::LL::BranchOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREEInterp::LL::CondBranchOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREEInterp::LL::ConvertSSOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREEInterp::LL::ConvertUUOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREEInterp::LL::ConvertSUOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREEInterp::LL::ConvertUSOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREEInterp::LL::CmpIOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREEInterp::LL::CmpFOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREEInterp::LL::AllocHeapOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREEInterp::LL::StaticCopyOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREEInterp::LL::ReduceSumIOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREEInterp::LL::ReduceSumFOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREEInterp::LL::ReduceMinIOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREEInterp::LL::ReduceMinFOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREEInterp::LL::ReduceMaxIOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREEInterp::LL::ReduceMaxFOp);
}

}  // namespace iree_compiler
}  // namespace mlir
