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

#include "compiler/IR/Sequencer/OpWriters.h"

#include "compiler/IR/Sequencer/LLOps.h"
#include "compiler/IR/StructureOps.h"
#include "compiler/Serialization/BytecodeWriter.h"
#include "compiler/Utils/Macros.h"
#include "schemas/bytecode/sequencer_bytecode_v0.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace iree_compiler {

namespace {

//===----------------------------------------------------------------------===//
// Sequencer ops
//===----------------------------------------------------------------------===//

LogicalResult writeOp(IREESeq::LL::ConstantOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::SequencerOpcode::kConstant));
  auto memRefType = op.getType().dyn_cast<MemRefType>();
  if (!memRefType) {
    return op.emitError()
           << "Constant has an unsupported type; must be a memref: "
           << op.getType();
  }
  RETURN_IF_FAILURE(writer->WriteConstant(memRefType, op.getAttr("value")));
  RETURN_IF_FAILURE(writer->WriteLocal(op.getResult()));
  return success();
}

LogicalResult writeOp(IREESeq::LL::CallOp op, BytecodeWriter *writer) {
  auto module = op.getOperation()->getParentOfType<ModuleOp>();
  auto callee = module.lookupSymbol<FuncOp>(op.getCallee());
  // TODO(benvanik): switch with kCallTail if attr exists.
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::SequencerOpcode::kCall));
  RETURN_IF_FAILURE(writer->WriteFunctionOrdinal(callee));
  RETURN_IF_FAILURE(writer->WriteLocals(op.getArgOperands()));
  RETURN_IF_FAILURE(writer->WriteLocals(op.getResults()));
  return success();
}

LogicalResult writeOp(IREESeq::LL::CallImportOp op, BytecodeWriter *writer) {
  auto module = op.getOperation()->getParentOfType<ModuleOp>();
  auto callee = module.lookupSymbol<FuncOp>(op.getCallee());
  // TODO(benvanik): transforms to convert Call->CallImport.
  // TODO(benvanik): switch with kCallTail if attr exists.
  if (callee.isExternal()) {
    RETURN_IF_FAILURE(writer->WriteOpcode(iree::SequencerOpcode::kCallImport));
  } else {
    RETURN_IF_FAILURE(writer->WriteOpcode(iree::SequencerOpcode::kCall));
  }
  RETURN_IF_FAILURE(writer->WriteImportOrdinal(callee));
  RETURN_IF_FAILURE(writer->WriteLocals(op.getArgOperands()));
  RETURN_IF_FAILURE(writer->WriteLocals(op.getResults()));
  return success();
}

LogicalResult writeOp(IREESeq::LL::CallIndirectOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::SequencerOpcode::kCallIndirect));
  RETURN_IF_FAILURE(writer->WriteTypeIndex(op.getCallee()->getType()));
  RETURN_IF_FAILURE(writer->WriteLocal(op.getCallee()));
  RETURN_IF_FAILURE(writer->WriteLocals(op.getArgOperands()));
  RETURN_IF_FAILURE(writer->WriteLocals(op.getResults()));
  return success();
}

LogicalResult writeOp(IREESeq::LL::BranchOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::SequencerOpcode::kBranch));
  RETURN_IF_FAILURE(writer->WriteBlockOffset(op.getDest()));
  RETURN_IF_FAILURE(writer->WriteCount(op.getNumOperands()));
  for (int i = 0; i < op.getNumOperands(); ++i) {
    // Copy src->dst.
    RETURN_IF_FAILURE(writer->WriteLocal(op.getOperand(i)));
    RETURN_IF_FAILURE(writer->WriteLocal(op.getDest()->getArgument(i)));
  }
  return success();
}

LogicalResult writeOp(IREESeq::LL::CondBranchOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::SequencerOpcode::kCondBranch));
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

LogicalResult writeDispatchOpExecutableRef(Operation *op, StringRef executable,
                                           StringRef entryPoint,
                                           BytecodeWriter *writer) {
  auto module = op->getParentOfType<ModuleOp>();
  auto multiArchExecutableOp =
      module.lookupSymbol<IREE::MultiArchExecutableOp>(executable);
  if (!multiArchExecutableOp) {
    return op->emitError() << "Executable @" << executable.str()
                           << " not found in module";
  }

  auto executableOrdinalAttr = multiArchExecutableOp.getAttr("iree.ordinal")
                                   .dyn_cast_or_null<IntegerAttr>();
  if (!executableOrdinalAttr) {
    return op->emitError() << "No ordinal assigned to executable";
  }
  int executableOrdinal = executableOrdinalAttr.getInt();

  // TODO(benvanik): move an export table to the MAE to make this cleaner.
  auto executableOp =
      cast<IREE::ExecutableOp>(multiArchExecutableOp.getBlock().front());
  auto entryPointOp =
      executableOp.getInnerModule().lookupSymbol<FuncOp>(entryPoint);
  if (!entryPointOp) {
    return op->emitError() << "Entry point @" << entryPoint.str()
                           << " not found in executable @" << executable.str();
  }
  if (!entryPointOp.getAttr("iree.ordinal")) {
    return op->emitError() << "No ordinal assigned to entry point";
  }
  int entryPointOrdinal =
      entryPointOp.getAttr("iree.ordinal").cast<IntegerAttr>().getInt();

  RETURN_IF_FAILURE(writer->WriteUint32(executableOrdinal));
  RETURN_IF_FAILURE(writer->WriteUint16(entryPointOrdinal));

  return success();
}

LogicalResult writeOp(IREESeq::LL::DynamicDispatchOp op,
                      BytecodeWriter *writer) {
  RETURN_IF_FAILURE(
      writer->WriteOpcode(iree::SequencerOpcode::kDynamicDispatch));
  RETURN_IF_FAILURE(writeDispatchOpExecutableRef(op, op.getExecutable(),
                                                 op.getEntryPoint(), writer));
  RETURN_IF_FAILURE(writer->WriteLocal(op.getWorkload()));
  RETURN_IF_FAILURE(writer->WriteLocals(op.getArgOperands()));
  // TODO(benvanik): support output arg group (or change to tags).
  RETURN_IF_FAILURE(writer->WriteCount(/*output_arg_count*/ 0));
  RETURN_IF_FAILURE(writer->WriteLocals(op.getResults()));
  return success();
}

LogicalResult writeOp(IREESeq::LL::StaticDispatchOp op,
                      BytecodeWriter *writer) {
  RETURN_IF_FAILURE(
      writer->WriteOpcode(iree::SequencerOpcode::kStaticDispatch));
  RETURN_IF_FAILURE(writeDispatchOpExecutableRef(op, op.getExecutable(),
                                                 op.getEntryPoint(), writer));
  auto workloadAttr = op.getWorkload();
  RETURN_IF_FAILURE(
      writer->WriteInt32(workloadAttr.getValue<IntegerAttr>({0}).getInt()));
  RETURN_IF_FAILURE(
      writer->WriteInt32(workloadAttr.getValue<IntegerAttr>({1}).getInt()));
  RETURN_IF_FAILURE(
      writer->WriteInt32(workloadAttr.getValue<IntegerAttr>({2}).getInt()));
  RETURN_IF_FAILURE(writer->WriteLocals(op.getArgOperands()));
  // TODO(benvanik): support output arg group (or change to tags).
  RETURN_IF_FAILURE(writer->WriteCount(/*output_arg_count*/ 0));
  RETURN_IF_FAILURE(writer->WriteLocals(op.getResults()));
  return success();
}

LogicalResult writeOp(IREESeq::LL::AllocHeapOp op, BytecodeWriter *writer) {
  auto memRefType = op.getType().cast<MemRefType>();
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::SequencerOpcode::kAllocHeap));
  RETURN_IF_FAILURE(writer->WriteInt32(0));
  RETURN_IF_FAILURE(writer->WriteTypeIndex(memRefType.getElementType()));
  RETURN_IF_FAILURE(writer->WriteShapePieces(memRefType));
  RETURN_IF_FAILURE(writer->WriteLocals(op.getOperands()));
  RETURN_IF_FAILURE(writer->WriteLocal(op.getResult()));
  return success();
}

LogicalResult writeOp(IREESeq::LL::ComputeRangeOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::SequencerOpcode::kComputeRange));
  RETURN_IF_FAILURE(writer->WriteLocal(op.shape()));
  RETURN_IF_FAILURE(writer->WriteUint8(op.elementSize().getZExtValue()));
  RETURN_IF_FAILURE(writer->WriteLocal(op.indices()));
  RETURN_IF_FAILURE(writer->WriteLocal(op.lengths()));
  RETURN_IF_FAILURE(writer->WriteLocal(op.dstOffset()));
  RETURN_IF_FAILURE(writer->WriteLocal(op.dstLength()));
  return success();
}

LogicalResult writeOp(IREESeq::LL::StaticSliceOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::SequencerOpcode::kStaticSlice));
  RETURN_IF_FAILURE(writer->WriteLocal(op.src()));
  RETURN_IF_FAILURE(writer->WriteInt32(op.offset().getZExtValue()));
  RETURN_IF_FAILURE(writer->WriteInt32(op.length().getZExtValue()));
  RETURN_IF_FAILURE(writer->WriteTypeIndex(op.getResult()->getType()));
  RETURN_IF_FAILURE(
      writer->WriteShapePieces(op.getResult()->getType().cast<ShapedType>()));
  RETURN_IF_FAILURE(writer->WriteLocal(op.getResult()));
  return success();
}

LogicalResult writeOp(IREESeq::LL::StaticCopyOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::SequencerOpcode::kStaticCopy));
  RETURN_IF_FAILURE(writer->WriteLocal(op.src()));
  RETURN_IF_FAILURE(writer->WriteInt32(op.srcOffset().getZExtValue()));
  RETURN_IF_FAILURE(writer->WriteLocal(op.dst()));
  RETURN_IF_FAILURE(writer->WriteInt32(op.dstOffset().getZExtValue()));
  RETURN_IF_FAILURE(writer->WriteInt32(op.length().getZExtValue()));
  return success();
}

LogicalResult writeOp(IREESeq::LL::StaticFillOp op, BytecodeWriter *writer) {
  RETURN_IF_FAILURE(writer->WriteOpcode(iree::SequencerOpcode::kStaticFill));
  RETURN_IF_FAILURE(writer->WriteInt32(op.value().getZExtValue()));
  RETURN_IF_FAILURE(writer->WriteLocal(op.dst()));
  RETURN_IF_FAILURE(writer->WriteInt32(op.dstOffset().getZExtValue()));
  RETURN_IF_FAILURE(writer->WriteInt32(op.length().getZExtValue()));
  return success();
}

}  // namespace

void registerSequencerCustomWriters(VMFunctionBuilder *builder) {
#define REGISTER_CUSTOM_WRITER_IMPL(op_type)       \
  builder->RegisterCustomWriter(                   \
      op_type::getOperationName(),                 \
      +[](Operation *op, BytecodeWriter *writer) { \
        return writeOp(cast<op_type>(op), writer); \
      });
  REGISTER_CUSTOM_WRITER_IMPL(IREESeq::LL::ConstantOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREESeq::LL::CallOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREESeq::LL::CallImportOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREESeq::LL::CallIndirectOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREESeq::LL::BranchOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREESeq::LL::CondBranchOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREESeq::LL::DynamicDispatchOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREESeq::LL::StaticDispatchOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREESeq::LL::AllocHeapOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREESeq::LL::ComputeRangeOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREESeq::LL::StaticSliceOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREESeq::LL::StaticCopyOp);
  REGISTER_CUSTOM_WRITER_IMPL(IREESeq::LL::StaticFillOp);
}

}  // namespace iree_compiler
}  // namespace mlir
