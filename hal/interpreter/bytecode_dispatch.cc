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

// Implements a full bytecode dispatch system.
// Currently this is verbose and object oriented, but future revisions
// (once we have interesting benchmarks) will likely simplify and inline
// a lot of the checks to make things faster. Consider this to be as
// experimental an implementation as the entire rest of the project :)

#include "hal/interpreter/bytecode_dispatch.h"

#include <algorithm>

#include "absl/base/attributes.h"
#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "base/logging.h"
#include "base/memory.h"
#include "base/status.h"
#include "hal/buffer_view.h"
#include "hal/heap_buffer.h"
#include "hal/interpreter/bytecode_dispatch_conversion.h"
#include "hal/interpreter/bytecode_dispatch_util.h"
#include "hal/interpreter/bytecode_kernels.h"
#include "rt/function.h"
#include "schemas/bytecode/interpreter_bytecode_v0.h"
#include "vm/bytecode_module.h"
#include "vm/bytecode_reader.h"
#include "vm/bytecode_tables_interpreter.h"
#include "vm/bytecode_util.h"
#include "vm/opcode_info.h"

namespace iree {
namespace hal {

namespace {

using ::iree::rt::Stack;
using ::iree::rt::StackFrame;
using ::iree::vm::BytecodeReader;

}  // namespace

Status Dispatch(hal::Allocator* allocator,
                kernels::RuntimeState* kernel_runtime_state, Stack* stack,
                StackFrame* entry_stack_frame,
                absl::Span<BufferView> entry_results) {
  // Dispatch table mapping 1:1 with bytecode ops.
  // Each entry is a label within this function that can be used for computed
  // goto. You can find more information on computed goto here:
  // https://eli.thegreenplace.net/2012/07/12/computed-goto-for-efficient-dispatch-tables
  //
  // Note that we ensure the table is 256 elements long exactly to make sure
  // that unused opcodes are handled gracefully.
  static const void* kDispatchTable[256] = {
#define DECLARE_DISPATCH(ordinal, name, ...) &&_dispatch_##name,
#define DECLARE_DISPATCH_RESERVED(ordinal, name, ...) &&_dispatch_unhandled,
      IREE_INTERPRETER_OPCODE_LIST(DECLARE_DISPATCH, DECLARE_DISPATCH_RESERVED)
#undef DECLARE_DISPATCH
#undef DECLARE_DISPATCH_RESERVED
  };

  // Primary dispatch state. This is our 'native stack frame' and really just
  // enough to make dereferencing common addresses (like the current offset)
  // faster. You can think of this like CPU state (like PC).
  //
  // We hope that LLVM decides to keep these in registers (as they are touched
  // for every instruction executed). The stack_frame will change as we call
  // into different functions.
  BytecodeReader reader(stack);
  RETURN_IF_ERROR(reader.SwitchStackFrame(entry_stack_frame));

#define DISPATCH_NEXT()                                                    \
  {                                                                        \
    uint8_t opcode = *reader.AdvanceOffset().ValueOrDie();                 \
    DVLOG(1)                                                               \
        << "Interpreter dispatching op code: "                             \
        << GetOpcodeInfo(vm::interpreter_opcode_table(), opcode).mnemonic; \
    goto* kDispatchTable[opcode];                                          \
  }

#define DISPATCH_CORE_OPCODE(opcode, body) \
  _dispatch_##opcode : {body} DISPATCH_NEXT()
#if defined(IREE_SUPPORT_F32) || defined(IREE_SUPPORT_F64)
#define DISPATCH_FLOAT_OPCODE(opcode, body) \
  _dispatch_##opcode : {body} DISPATCH_NEXT()
#else
#define DISPATCH_FLOAT_OPCODE(...)
#endif  // IREE_SUPPORT_F32 || IREE_SUPPORT_F64

  DISPATCH_NEXT();

  DISPATCH_CORE_OPCODE(kConstant, {
    ASSIGN_OR_RETURN(auto value, reader.ReadConstant());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    *dst_local = std::move(value);
  });

  DISPATCH_CORE_OPCODE(kCall, {
    auto* old_stack_frame = stack->current_frame();
    ASSIGN_OR_RETURN(const auto& target_function, reader.ReadFunction());
    // TODO(benvanik): rework register storage interface.
    ASSIGN_OR_RETURN(
        const auto* function_def,
        static_cast<const vm::BytecodeModule*>(target_function.module())
            ->GetFunctionDef(target_function.linkage(),
                             target_function.ordinal()));
    ASSIGN_OR_RETURN(auto* new_stack_frame, stack->PushFrame(target_function));
    new_stack_frame->mutable_registers()->buffer_views.resize(
        function_def->bytecode()->local_count());
    RETURN_IF_ERROR(
        reader.CopyInputsAndSwitchStackFrame(old_stack_frame, new_stack_frame));
    DVLOG(1) << "Call; stack now: " << stack->DebugString();
  });

  DISPATCH_CORE_OPCODE(kCallImport, {
    return UnimplementedErrorBuilder(IREE_LOC)
           << "Non-module imports not supported";
  });

  DISPATCH_CORE_OPCODE(kCallIndirect, {
    return UnimplementedErrorBuilder(IREE_LOC) << "Unimplemented call_indirect";
  });

  DISPATCH_CORE_OPCODE(kReturn, {
    auto* old_stack_frame = stack->current_frame();
    auto* new_stack_frame = stack->caller_frame();
    if (old_stack_frame == entry_stack_frame) {
      // Returning from entry function. Marshal results from the return stmt.
      ASSIGN_OR_RETURN(int32_t src_count, reader.ReadCount());
      for (int i = 0; i < src_count; ++i) {
        ASSIGN_OR_RETURN(
            auto* src_local,
            reader.ReadLocal(old_stack_frame->mutable_registers()));
        entry_results[i] = std::move(*src_local);
      }
      DVLOG(1) << "Returning to entry";
      return OkStatus();
    } else if (!new_stack_frame) {
      return FailedPreconditionErrorBuilder(IREE_LOC) << "Stack underflow";
    }
    RETURN_IF_ERROR(reader.CopyResultsAndSwitchStackFrame(old_stack_frame,
                                                          new_stack_frame));
    RETURN_IF_ERROR(stack->PopFrame());
    DVLOG(1) << "Return; stack now: " << stack->DebugString();
  });

  DISPATCH_CORE_OPCODE(kBranch, {
    ASSIGN_OR_RETURN(int32_t offset, reader.ReadBlockOffset());
    RETURN_IF_ERROR(reader.CopySlots());
    RETURN_IF_ERROR(reader.BranchToOffset(offset));
  });

  DISPATCH_CORE_OPCODE(kCondBranch, {
    // Evaluate condition first so we can do the copies as we read them for
    // which side of the branch we take.
    ASSIGN_OR_RETURN(auto* cond_local, reader.ReadLocal());
    bool cond_value = BufferViewIsTrue(*cond_local);
    ASSIGN_OR_RETURN(int32_t true_offset, reader.ReadBlockOffset());
    if (cond_value) {
      RETURN_IF_ERROR(reader.CopySlots());
      RETURN_IF_ERROR(reader.BranchToOffset(true_offset));
    } else {
      ASSIGN_OR_RETURN(int32_t true_op_count, reader.ReadCount());
      RETURN_IF_ERROR(reader.SkipLocals(2 * true_op_count));
      ASSIGN_OR_RETURN(int32_t false_offset, reader.ReadBlockOffset());
      RETURN_IF_ERROR(reader.CopySlots());
      RETURN_IF_ERROR(reader.BranchToOffset(false_offset));
    }
  });

  DISPATCH_CORE_OPCODE(kCmpI, {
    ASSIGN_OR_RETURN(uint8_t predicate, reader.ReadUint8_t());
    ASSIGN_OR_RETURN(auto* lhs_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* rhs_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());

    switch (static_cast<CmpIPredicate>(predicate)) {
      case CmpIPredicate::kEq:
        RETURN_IF_ERROR(ApplyComparisonOpIS<kernels::CompareEQ>(
            lhs_local, rhs_local, dst_local));
        break;
      case CmpIPredicate::kNe:
        RETURN_IF_ERROR(ApplyComparisonOpIS<kernels::CompareNE>(
            lhs_local, rhs_local, dst_local));
        break;
      case CmpIPredicate::kSlt:
        RETURN_IF_ERROR(ApplyComparisonOpIS<kernels::CompareLT>(
            lhs_local, rhs_local, dst_local));
        break;
      case CmpIPredicate::kSle:
        RETURN_IF_ERROR(ApplyComparisonOpIS<kernels::CompareLE>(
            lhs_local, rhs_local, dst_local));
        break;
      case CmpIPredicate::kSgt:
        RETURN_IF_ERROR(ApplyComparisonOpIS<kernels::CompareGT>(
            lhs_local, rhs_local, dst_local));
        break;
      case CmpIPredicate::kSge:
        RETURN_IF_ERROR(ApplyComparisonOpIS<kernels::CompareGE>(
            lhs_local, rhs_local, dst_local));
        break;
      case CmpIPredicate::kUlt:
        RETURN_IF_ERROR(ApplyComparisonOpIU<kernels::CompareLT>(
            lhs_local, rhs_local, dst_local));
        break;
      case CmpIPredicate::kUle:
        RETURN_IF_ERROR(ApplyComparisonOpIU<kernels::CompareLE>(
            lhs_local, rhs_local, dst_local));
        break;
      case CmpIPredicate::kUgt:
        RETURN_IF_ERROR(ApplyComparisonOpIU<kernels::CompareGT>(
            lhs_local, rhs_local, dst_local));
        break;
      case CmpIPredicate::kUge:
        RETURN_IF_ERROR(ApplyComparisonOpIU<kernels::CompareGE>(
            lhs_local, rhs_local, dst_local));
        break;
    }
  });

  DISPATCH_FLOAT_OPCODE(kCmpF, {
    ASSIGN_OR_RETURN(uint8_t p, reader.ReadUint8_t());
    ASSIGN_OR_RETURN(auto* lhs_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* rhs_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());

    auto predicate = static_cast<CmpFPredicate>(p);
    switch (predicate) {
      case CmpFPredicate::kOeq:
        RETURN_IF_ERROR(ApplyComparisonOpF<kernels::CompareEQ>(
            lhs_local, rhs_local, dst_local));
        break;
      case CmpFPredicate::kUne:
        RETURN_IF_ERROR(ApplyComparisonOpF<kernels::CompareNE>(
            lhs_local, rhs_local, dst_local));
        break;
      case CmpFPredicate::kOlt:
        RETURN_IF_ERROR(ApplyComparisonOpF<kernels::CompareLT>(
            lhs_local, rhs_local, dst_local));
        break;
      case CmpFPredicate::kOle:
        RETURN_IF_ERROR(ApplyComparisonOpF<kernels::CompareLE>(
            lhs_local, rhs_local, dst_local));
        break;
      case CmpFPredicate::kOgt:
        RETURN_IF_ERROR(ApplyComparisonOpF<kernels::CompareGT>(
            lhs_local, rhs_local, dst_local));
        break;
      case CmpFPredicate::kOge:
        RETURN_IF_ERROR(ApplyComparisonOpF<kernels::CompareGE>(
            lhs_local, rhs_local, dst_local));
        break;
      case CmpFPredicate::kFalse:
      case CmpFPredicate::kOne:
      case CmpFPredicate::kOrd:
      case CmpFPredicate::kUeq:
      case CmpFPredicate::kUgt:
      case CmpFPredicate::kUge:
      case CmpFPredicate::kUlt:
      case CmpFPredicate::kUle:
      case CmpFPredicate::kUno:
      case CmpFPredicate::kTrue:
        // TODO(b/132183250) support these if we ever need them.
        return UnimplementedErrorBuilder(IREE_LOC)
               << "Unsupported comparison predicate value "
               << static_cast<int>(p) << " ("
               << vm::PredicateToString(predicate) << ")";
    }
  });

  DISPATCH_CORE_OPCODE(kAllocStatic, {
    return UnimplementedErrorBuilder(IREE_LOC) << "Unimplemented alloc_static";
  });

  DISPATCH_CORE_OPCODE(kAllocStack, {
    return UnimplementedErrorBuilder(IREE_LOC) << "Unimplemented alloc_stack";
  });

  DISPATCH_CORE_OPCODE(kAllocStackInit, {
    return UnimplementedErrorBuilder(IREE_LOC)
           << "Unimplemented alloc_stack_init";
  });

  DISPATCH_CORE_OPCODE(kAllocHeap, {
    ASSIGN_OR_RETURN(auto heap_type, reader.ReadInt32());
    ASSIGN_OR_RETURN(auto type, reader.ReadType());
    size_t element_size = type.element_size();

    // TODO(benvanik): more efficient reading and storage.
    size_t element_count = 0;
    ASSIGN_OR_RETURN(auto shape, reader.ReadShapePieces(&element_count));
    size_t allocation_size = element_size * element_count;

    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    dst_local->element_size = element_size;
    dst_local->shape = shape;

    // TODO(benvanik): properly allocate with attributes from op.
    CHECK_EQ(heap_type, 0);
    ASSIGN_OR_RETURN(
        dst_local->buffer,
        allocator->Allocate(MemoryType::kHostLocal | MemoryType::kDeviceVisible,
                            BufferUsage::kAll, allocation_size));
  });

  DISPATCH_CORE_OPCODE(kDiscard, {
    // NOTE: if we were an encoder we would actually discard the buffer.
    ASSIGN_OR_RETURN(auto* local, reader.ReadLocal());
    *local = {};
  });

  DISPATCH_CORE_OPCODE(kRank, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    int32_t rank = src_local->shape.size();
    RETURN_IF_ERROR(dst_local->buffer->WriteData(0, &rank, sizeof(int32_t)));
  });

  DISPATCH_CORE_OPCODE(kDim, {
    ASSIGN_OR_RETURN(int32_t axis, reader.ReadUint8_t());
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(int32_t dim, src_local->shape.ResolveAxis(axis));
    RETURN_IF_ERROR(dst_local->buffer->WriteData(0, &dim, sizeof(int32_t)));
  });

  DISPATCH_CORE_OPCODE(kShape, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    RETURN_IF_ERROR(dst_local->buffer->WriteData(
        0, src_local->shape.subspan().data(),
        src_local->shape.subspan().size() * sizeof(int32_t)));
  });

  DISPATCH_CORE_OPCODE(kLength, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    int32_t length = src_local->shape.element_count();
    RETURN_IF_ERROR(dst_local->buffer->WriteData(0, &length, sizeof(int32_t)));
  });

  DISPATCH_CORE_OPCODE(kDynamicSlice, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto indices, reader.ReadSlotElements<int32_t>());
    ASSIGN_OR_RETURN(auto lengths, reader.ReadSlotElements<int32_t>());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(*dst_local, src_local->Slice(indices, lengths));
  });

  DISPATCH_CORE_OPCODE(kStaticSlice, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto indices, reader.ReadIndexList());
    ASSIGN_OR_RETURN(auto lengths, reader.ReadIndexList());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(*dst_local, src_local->Slice(indices, lengths));
  });

  DISPATCH_CORE_OPCODE(kDynamicCopy, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto src_indices, reader.ReadSlotElements<int32_t>());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto dst_indices, reader.ReadSlotElements<int32_t>());
    ASSIGN_OR_RETURN(auto lengths, reader.ReadSlotElements<int32_t>());
    RETURN_IF_ERROR(
        ApplyCopy(src_local, src_indices, dst_local, dst_indices, lengths));
  });

  DISPATCH_CORE_OPCODE(kStaticCopy, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto src_indices, reader.ReadIndexList());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto dst_indices, reader.ReadIndexList());
    ASSIGN_OR_RETURN(auto lengths, reader.ReadIndexList());
    RETURN_IF_ERROR(
        ApplyCopy(src_local, src_indices, dst_local, dst_indices, lengths));
  });

  DISPATCH_CORE_OPCODE(kClone, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    dst_local->element_size = src_local->element_size;
    dst_local->shape = src_local->shape;
    dst_local->buffer = HeapBuffer::Allocate(src_local->buffer->usage(),
                                             src_local->buffer->byte_length());
    RETURN_IF_ERROR(dst_local->buffer->CopyData(0, src_local->buffer.get()));
  });

  DISPATCH_CORE_OPCODE(kSplit, {
    return UnimplementedErrorBuilder(IREE_LOC) << "Unimplemented split";
  });

  DISPATCH_CORE_OPCODE(kAssign, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    *dst_local = *src_local;
  });

  DISPATCH_CORE_OPCODE(kCondAssign, {
    ASSIGN_OR_RETURN(auto* cond_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* lhs_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* rhs_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    *dst_local = BufferViewIsTrue(*cond_local) ? *lhs_local : *rhs_local;
  });

  DISPATCH_CORE_OPCODE(kReshape, {
    // TODO(benvanik): more logic required if strides differ.
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto shape_data, reader.ReadSlotElements<int32_t>());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    Shape new_shape = Shape{shape_data};
    if (src_local->shape.element_count() != new_shape.element_count()) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "New element count " << new_shape.element_count()
             << " != source element count " << src_local->shape.element_count();
    }
    dst_local->shape = new_shape;
    dst_local->buffer = add_ref(src_local->buffer);
    dst_local->element_size = src_local->element_size;
  });

  DISPATCH_CORE_OPCODE(kSelect, {
    ASSIGN_OR_RETURN(auto* cond_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* lhs_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* rhs_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto cond_buffer, cond_local->buffer->MapMemory<uint8_t>(
                                           MemoryAccess::kRead));
    ASSIGN_OR_RETURN(auto lhs_buffer, lhs_local->buffer->MapMemory<uint8_t>(
                                          MemoryAccess::kRead));
    ASSIGN_OR_RETURN(auto rhs_buffer, rhs_local->buffer->MapMemory<uint8_t>(
                                          MemoryAccess::kRead));
    ASSIGN_OR_RETURN(auto dst_buffer, dst_local->buffer->MapMemory<uint8_t>(
                                          MemoryAccess::kDiscardWrite));
    if (cond_local->element_size != 1) {
      return InvalidArgumentErrorBuilder(IREE_LOC) << "Select cond must be i8";
    } else if (lhs_buffer.size() != rhs_buffer.size()) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "LHS " << lhs_buffer.size() << "b != RHS " << rhs_buffer.size()
             << "b; both arguments must match";
    } else if (lhs_buffer.size() != dst_buffer.size()) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Dest " << dst_buffer.size() << "b != LHS/RHS "
             << lhs_buffer.size() << "b; dest must match inputs";
    }
    switch (lhs_local->element_size) {
      case 1:
        RETURN_IF_ERROR(kernels::Select::Execute<uint8_t>(
            cond_buffer.contents(), lhs_buffer.contents(),
            rhs_buffer.contents(), dst_buffer.mutable_contents()));
        break;
      case 2:
        RETURN_IF_ERROR(kernels::Select::Execute<uint16_t>(
            cond_buffer.contents(),
            ReinterpretSpan<uint16_t>(lhs_buffer.contents()),
            ReinterpretSpan<uint16_t>(rhs_buffer.contents()),
            ReinterpretSpan<uint16_t>(dst_buffer.mutable_contents())));
        break;
      case 4:
        RETURN_IF_ERROR(kernels::Select::Execute<uint32_t>(
            cond_buffer.contents(),
            ReinterpretSpan<uint32_t>(lhs_buffer.contents()),
            ReinterpretSpan<uint32_t>(rhs_buffer.contents()),
            ReinterpretSpan<uint32_t>(dst_buffer.mutable_contents())));
        break;
      case 8:
        RETURN_IF_ERROR(kernels::Select::Execute<uint64_t>(
            cond_buffer.contents(),
            ReinterpretSpan<uint64_t>(lhs_buffer.contents()),
            ReinterpretSpan<uint64_t>(rhs_buffer.contents()),
            ReinterpretSpan<uint64_t>(dst_buffer.mutable_contents())));
        break;
      default:
        return UnimplementedErrorBuilder(IREE_LOC)
               << "Unimplemented element size: " << lhs_local->element_size;
    }
  });

  DISPATCH_CORE_OPCODE(kTranspose, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto perm_data, reader.ReadSlotElements<int32_t>());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    RETURN_IF_ERROR(ApplyUnaryOpIU<kernels::Transpose>(
        src_local, dst_local, src_local->shape,
        absl::MakeConstSpan(perm_data)));
  });

  DISPATCH_CORE_OPCODE(kReverse, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto perm_data, reader.ReadSlotElements<int32_t>());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    RETURN_IF_ERROR(
        ApplyUnaryOpIU<kernels::Reverse>(src_local, dst_local, src_local->shape,
                                         absl::MakeConstSpan(perm_data)));
  });

  DISPATCH_CORE_OPCODE(kPad, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* padding_value, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto edge_padding_low, reader.ReadSlotElements<int32_t>());
    ASSIGN_OR_RETURN(auto edge_padding_high,
                     reader.ReadSlotElements<int32_t>());
    ASSIGN_OR_RETURN(auto interior_padding, reader.ReadSlotElements<int32_t>());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());

    RETURN_IF_ERROR(ApplyBinaryOpIU<kernels::Pad>(
        src_local, padding_value, dst_local, src_local->shape, dst_local->shape,
        absl::MakeConstSpan(edge_padding_low),
        absl::MakeConstSpan(edge_padding_high),
        absl::MakeConstSpan(interior_padding)));
  });

  DISPATCH_CORE_OPCODE(kBroadcast, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto shape_data, reader.ReadSlotElements<int32_t>());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    dst_local->shape = Shape{shape_data};
    RETURN_IF_ERROR(ApplyUnaryOpIU<kernels::Broadcast>(src_local, dst_local));
  });

  DISPATCH_CORE_OPCODE(kTile, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto shape_data, reader.ReadSlotElements<int32_t>());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    dst_local->shape = Shape{shape_data};
    RETURN_IF_ERROR(ApplyUnaryOpIU<kernels::Tile>(
        src_local, dst_local, src_local->shape, dst_local->shape));
  });

  DISPATCH_CORE_OPCODE(kNot, {
    RETURN_IF_ERROR(DispatchElementwiseUnaryOpIU<kernels::Not>(&reader));
  });
  DISPATCH_CORE_OPCODE(kAnd, {
    RETURN_IF_ERROR(DispatchElementwiseBinaryOpIU<kernels::And>(&reader));
  });
  DISPATCH_CORE_OPCODE(kOr, {
    RETURN_IF_ERROR(DispatchElementwiseBinaryOpIU<kernels::Or>(&reader));
  });
  DISPATCH_CORE_OPCODE(kXor, {
    RETURN_IF_ERROR(DispatchElementwiseBinaryOpIU<kernels::Xor>(&reader));
  });
  DISPATCH_CORE_OPCODE(kShiftLeft, {
    RETURN_IF_ERROR(DispatchElementwiseBinaryOpIU<kernels::ShiftLeft>(&reader));
  });
  DISPATCH_CORE_OPCODE(kShiftRightLogical, {
    RETURN_IF_ERROR(
        DispatchElementwiseBinaryOpIU<kernels::ShiftRight>(&reader));
  });
  DISPATCH_CORE_OPCODE(kShiftRightArithmetic, {
    RETURN_IF_ERROR(
        DispatchElementwiseBinaryOpIS<kernels::ShiftRight>(&reader));
  });

  DISPATCH_CORE_OPCODE(kAddI, {
    RETURN_IF_ERROR(DispatchElementwiseBinaryOpIU<kernels::Add>(&reader));
  });
  DISPATCH_FLOAT_OPCODE(kAddF, {
    RETURN_IF_ERROR(DispatchElementwiseBinaryOpF<kernels::Add>(&reader));
  });

  DISPATCH_CORE_OPCODE(kSubI, {
    RETURN_IF_ERROR(DispatchElementwiseBinaryOpIU<kernels::Sub>(&reader));
  });
  DISPATCH_FLOAT_OPCODE(kSubF, {
    RETURN_IF_ERROR(DispatchElementwiseBinaryOpF<kernels::Sub>(&reader));
  });

  DISPATCH_CORE_OPCODE(kAbsI, {
    RETURN_IF_ERROR(DispatchElementwiseUnaryOpIS<kernels::Abs>(&reader));
  });
  DISPATCH_FLOAT_OPCODE(kAbsF, {
    RETURN_IF_ERROR(DispatchElementwiseUnaryOpF<kernels::Abs>(&reader));
  });

  DISPATCH_CORE_OPCODE(kMulI, {
    RETURN_IF_ERROR(DispatchElementwiseBinaryOpIU<kernels::Mul>(&reader));
  });
  DISPATCH_FLOAT_OPCODE(kMulF, {
    RETURN_IF_ERROR(DispatchElementwiseBinaryOpF<kernels::Mul>(&reader));
  });

  DISPATCH_CORE_OPCODE(kDivIS, {
    RETURN_IF_ERROR(DispatchElementwiseBinaryOpIS<kernels::Div>(&reader));
  });
  DISPATCH_CORE_OPCODE(kDivIU, {
    RETURN_IF_ERROR(DispatchElementwiseBinaryOpIU<kernels::Div>(&reader));
  });
  DISPATCH_FLOAT_OPCODE(kDivF, {
    RETURN_IF_ERROR(DispatchElementwiseBinaryOpF<kernels::Div>(&reader));
  });

  DISPATCH_CORE_OPCODE(kMulAddI, {
    RETURN_IF_ERROR(DispatchElementwiseTernaryOpIU<kernels::MulAdd>(&reader));
  });
  DISPATCH_FLOAT_OPCODE(kMulAddF, {
    RETURN_IF_ERROR(DispatchElementwiseTernaryOpF<kernels::MulAdd>(&reader));
  });
  DISPATCH_FLOAT_OPCODE(kExpF, {
    RETURN_IF_ERROR(DispatchElementwiseUnaryOpF<kernels::Exp>(&reader));
  });
  DISPATCH_FLOAT_OPCODE(kLogF, {
    RETURN_IF_ERROR(DispatchElementwiseUnaryOpF<kernels::Log>(&reader));
  });
  DISPATCH_FLOAT_OPCODE(kRsqrtF, {
    RETURN_IF_ERROR(DispatchElementwiseUnaryOpF<kernels::Rsqrt>(&reader));
  });
  DISPATCH_FLOAT_OPCODE(kCosF, {
    RETURN_IF_ERROR(DispatchElementwiseUnaryOpF<kernels::Cos>(&reader));
  });
  DISPATCH_FLOAT_OPCODE(kSinF, {
    RETURN_IF_ERROR(DispatchElementwiseUnaryOpF<kernels::Sin>(&reader));
  });
  DISPATCH_FLOAT_OPCODE(kTanhF, {
    RETURN_IF_ERROR(DispatchElementwiseUnaryOpF<kernels::Tanh>(&reader));
  });
  DISPATCH_FLOAT_OPCODE(kAtan2F, {
    RETURN_IF_ERROR(DispatchElementwiseBinaryOpF<kernels::Atan2>(&reader));
  });

  DISPATCH_CORE_OPCODE(kMinIS, {
    RETURN_IF_ERROR(DispatchElementwiseBinaryOpIS<kernels::Min>(&reader));
  });
  DISPATCH_CORE_OPCODE(kMinIU, {
    RETURN_IF_ERROR(DispatchElementwiseBinaryOpIU<kernels::Min>(&reader));
  });
  DISPATCH_FLOAT_OPCODE(kMinF, {
    RETURN_IF_ERROR(DispatchElementwiseBinaryOpF<kernels::Min>(&reader));
  });

  DISPATCH_CORE_OPCODE(kMaxIS, {
    RETURN_IF_ERROR(DispatchElementwiseBinaryOpIS<kernels::Max>(&reader));
  });
  DISPATCH_CORE_OPCODE(kMaxIU, {
    RETURN_IF_ERROR(DispatchElementwiseBinaryOpIU<kernels::Max>(&reader));
  });
  DISPATCH_FLOAT_OPCODE(kMaxF, {
    RETURN_IF_ERROR(DispatchElementwiseBinaryOpF<kernels::Max>(&reader));
  });

  DISPATCH_CORE_OPCODE(kClampIS, {
    RETURN_IF_ERROR(DispatchElementwiseTernaryOpIS<kernels::Clamp>(&reader));
  });
  DISPATCH_CORE_OPCODE(kClampIU, {
    RETURN_IF_ERROR(DispatchElementwiseTernaryOpIS<kernels::Clamp>(&reader));
  });
  DISPATCH_FLOAT_OPCODE(kClampF, {
    RETURN_IF_ERROR(DispatchElementwiseTernaryOpF<kernels::Clamp>(&reader));
  });

  DISPATCH_FLOAT_OPCODE(kFloorF, {
    RETURN_IF_ERROR(DispatchElementwiseUnaryOpF<kernels::Floor>(&reader));
  });
  DISPATCH_FLOAT_OPCODE(kCeilF, {
    RETURN_IF_ERROR(DispatchElementwiseUnaryOpF<kernels::Ceil>(&reader));
  });

  DISPATCH_CORE_OPCODE(kConvertSS, {
    ASSIGN_OR_RETURN(auto src_type, reader.ReadType());
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto dst_type, reader.ReadType());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    RETURN_IF_ERROR(
        ApplyConvertSS::Apply(src_type, src_local, dst_type, dst_local));
  });
  DISPATCH_CORE_OPCODE(kConvertUU, {
    ASSIGN_OR_RETURN(auto src_type, reader.ReadType());
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto dst_type, reader.ReadType());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    RETURN_IF_ERROR(
        ApplyConvertUU::Apply(src_type, src_local, dst_type, dst_local));
  });
  DISPATCH_CORE_OPCODE(kConvertSU, {
    ASSIGN_OR_RETURN(auto src_type, reader.ReadType());
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto dst_type, reader.ReadType());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    RETURN_IF_ERROR(
        ApplyConvertSU::Apply(src_type, src_local, dst_type, dst_local));
  });
  DISPATCH_CORE_OPCODE(kConvertUS, {
    ASSIGN_OR_RETURN(auto src_type, reader.ReadType());
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto dst_type, reader.ReadType());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    RETURN_IF_ERROR(
        ApplyConvertUS::Apply(src_type, src_local, dst_type, dst_local));
  });

  DISPATCH_CORE_OPCODE(kMatMulI, {
    ASSIGN_OR_RETURN(auto* lhs_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* rhs_local, reader.ReadLocal());
    // TODO(benvanik): add fused matmul-with-bias op in MLIR and lower to this.
    BufferView* bias_local = nullptr;
    ASSIGN_OR_RETURN(auto* multiplier_mantissa_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* multiplier_exponent_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    RETURN_IF_ERROR(ValidateMatMulOpI(lhs_local, rhs_local, bias_local,
                                      multiplier_mantissa_local,
                                      multiplier_exponent_local, dst_local));
    auto* mat_mul_state = kernel_runtime_state->mat_mul_state.get();
    // TODO(benvanik): define as a matrix of supported types to enable 8*8=16,
    // accumulator options, and other precision modes.
    switch (lhs_local->element_size) {
      case 1:
        RETURN_IF_ERROR(ApplyMatMulOpI<int8_t>(
            mat_mul_state, lhs_local, rhs_local, bias_local,
            multiplier_mantissa_local, multiplier_exponent_local, dst_local));
        break;
      case 2:
        RETURN_IF_ERROR(ApplyMatMulOpI<int16_t>(
            mat_mul_state, lhs_local, rhs_local, bias_local,
            multiplier_mantissa_local, multiplier_exponent_local, dst_local));
        break;
      case 4:
        RETURN_IF_ERROR(ApplyMatMulOpI<int32_t>(
            mat_mul_state, lhs_local, rhs_local, bias_local,
            multiplier_mantissa_local, multiplier_exponent_local, dst_local));
        break;
      case 8:
        RETURN_IF_ERROR(ApplyMatMulOpI<int64_t>(
            mat_mul_state, lhs_local, rhs_local, bias_local,
            multiplier_mantissa_local, multiplier_exponent_local, dst_local));
        break;
      default:
        return UnimplementedErrorBuilder(IREE_LOC)
               << "Unimplemented element size: " << lhs_local->element_size;
    }
  });

  DISPATCH_FLOAT_OPCODE(kMatMulF, {
    ASSIGN_OR_RETURN(auto* lhs_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* rhs_local, reader.ReadLocal());
    BufferView* bias_local = nullptr;
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    RETURN_IF_ERROR(
        ValidateMatMulOpF(lhs_local, rhs_local, bias_local, dst_local));
    auto* mat_mul_state = kernel_runtime_state->mat_mul_state.get();
    switch (lhs_local->element_size) {
      case 4:
        RETURN_IF_ERROR(ApplyMatMulOpF<float>(
            mat_mul_state, lhs_local, rhs_local, bias_local, dst_local));
        break;
      case 8:
        RETURN_IF_ERROR(ApplyMatMulOpF<double>(
            mat_mul_state, lhs_local, rhs_local, bias_local, dst_local));
        break;
      default:
        return UnimplementedErrorBuilder(IREE_LOC)
               << "Unimplemented element size: " << lhs_local->element_size;
    }
  });

  DISPATCH_CORE_OPCODE(kReduceSumI, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* init_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto dimension, reader.ReadInt32());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    // TODO(scotttodd): validate
    RETURN_IF_ERROR(ApplyBinaryOpIS<kernels::ReduceSum>(
        src_local, init_local, dst_local, dimension, src_local->shape,
        dst_local->shape));
  });

  DISPATCH_FLOAT_OPCODE(kReduceSumF, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* init_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto dimension, reader.ReadInt32());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    // TODO(scotttodd): validate
    RETURN_IF_ERROR(ApplyBinaryOpF<kernels::ReduceSum>(
        src_local, init_local, dst_local, dimension, src_local->shape,
        dst_local->shape));
  });

  DISPATCH_CORE_OPCODE(kReduceMinI, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* init_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto dimension, reader.ReadInt32());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    // TODO(scotttodd): validate
    RETURN_IF_ERROR(ApplyBinaryOpIS<kernels::ReduceMin>(
        src_local, init_local, dst_local, dimension, src_local->shape,
        dst_local->shape));
  });

  DISPATCH_FLOAT_OPCODE(kReduceMinF, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* init_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto dimension, reader.ReadInt32());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    // TODO(scotttodd): validate
    RETURN_IF_ERROR(ApplyBinaryOpF<kernels::ReduceMin>(
        src_local, init_local, dst_local, dimension, src_local->shape,
        dst_local->shape));
  });

  DISPATCH_CORE_OPCODE(kReduceMaxI, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* init_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto dimension, reader.ReadInt32());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    // TODO(scotttodd): validate
    RETURN_IF_ERROR(ApplyBinaryOpIS<kernels::ReduceMax>(
        src_local, init_local, dst_local, dimension, src_local->shape,
        dst_local->shape));
  });

  DISPATCH_FLOAT_OPCODE(kReduceMaxF, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* init_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto dimension, reader.ReadInt32());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    // TODO(scotttodd): validate
    RETURN_IF_ERROR(ApplyBinaryOpF<kernels::ReduceMax>(
        src_local, init_local, dst_local, dimension, src_local->shape,
        dst_local->shape));
  });

  DISPATCH_CORE_OPCODE(kTrace, {
    return UnimplementedErrorBuilder(IREE_LOC) << "Unimplemented trace";
  });

  DISPATCH_CORE_OPCODE(kBreak, {
    return UnimplementedErrorBuilder(IREE_LOC) << "Unimplemented break";
  });

  DISPATCH_CORE_OPCODE(kCondBreak, {
    return UnimplementedErrorBuilder(IREE_LOC) << "Unimplemented cond_break";
  });

_dispatch_unhandled:
  // TODO(benvanik): better tracing.
  return UnimplementedErrorBuilder(IREE_LOC) << "Unknown dispatch opcode";
}  // NOLINT(readability/fn_size)

}  // namespace hal
}  // namespace iree
