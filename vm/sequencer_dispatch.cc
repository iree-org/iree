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

// Implements a full bytecode dispatch system for sequencer ops.
// TODO(benvanik): rework to be async against CommandBuffers.

#include "vm/sequencer_dispatch.h"

#include <algorithm>

#include "absl/base/attributes.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "base/logging.h"
#include "base/memory.h"
#include "base/status.h"
#include "hal/buffer_view.h"
#include "hal/command_queue.h"
#include "hal/device.h"
#include "hal/heap_buffer.h"
#include "schemas/bytecode/sequencer_bytecode_v0.h"
#include "vm/bytecode_module.h"
#include "vm/bytecode_reader.h"
#include "vm/bytecode_tables_sequencer.h"
#include "vm/bytecode_util.h"
#include "vm/opcode_info.h"

namespace iree {
namespace vm {

namespace {

using ::iree::hal::Buffer;
using ::iree::hal::BufferView;

// TODO(benvanik): remove (this should happen via predication).
bool BufferViewIsTrue(const BufferView& buffer_view) {
  if (buffer_view.element_size == 0 || !buffer_view.buffer ||
      buffer_view.byte_length() == 0) {
    return false;
  }
  // TODO(benvanik): map more efficiently (based on element size?).
  auto mapping =
      buffer_view.buffer->MapMemory<uint8_t>(hal::MemoryAccess::kRead);
  if (!mapping.ok()) {
    return false;
  }
  for (uint8_t value : mapping.ValueOrDie().contents()) {
    if (value) return true;
  }
  return false;
}

// TODO(benvanik): insert fence callbacks and wait on fence.
Status CallExternalFunction(rt::Stack* stack, const rt::Function& function) {
  // Marshal inputs and outputs.
  const auto* stack_frame = stack->current_frame();
  auto buffer_views = absl::MakeSpan(stack_frame->registers().buffer_views);
  absl::InlinedVector<hal::BufferView, 8> arguments(
      buffer_views.begin(),
      buffer_views.begin() + function.signature().argument_count());
  absl::InlinedVector<hal::BufferView, 8> results(
      buffer_views.begin() + arguments.size(), buffer_views.end());
  return function.module()->Execute(stack, function, std::move(arguments),
                                    &results);
}

// Pretty prints an array, e.g. [1, 2, 3, 4]
inline std::string PrettyPrint(absl::Span<const int32_t> arr) {
  return "[" + absl::StrJoin(arr, ",") + "]";
}

// Calculates the byte offset into a buffer corresponding to the indices in the
// given shape.
StatusOr<device_size_t> CalculateOffset(absl::Span<const int32_t> indices,
                                        Shape shape, uint8_t element_size) {
  if (shape.empty() || indices.size() > shape.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Indices " << PrettyPrint(indices) << " out of bounds of shape "
           << PrettyPrint(shape.subspan());
  }
  device_size_t offset = 0;
  for (int i = 0; i < indices.size(); ++i) {
    if (indices[i] >= shape[i]) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Indices[" << i << "]=" << indices[i]
             << " out of bounds of shape " << PrettyPrint(shape.subspan());
    }
    device_size_t axis_offset = indices[i];
    for (int j = i + 1; j < shape.size(); ++j) {
      axis_offset *= shape[j];
    }
    offset += axis_offset;
  }
  offset *= element_size;
  return offset;
}

}  // namespace

Status DispatchSequence(const hal::DevicePlacement& placement, rt::Stack* stack,
                        rt::StackFrame* entry_stack_frame,
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
      IREE_SEQUENCER_OPCODE_LIST(DECLARE_DISPATCH, DECLARE_DISPATCH_RESERVED)
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

#define DISPATCH_NEXT()                                                   \
  {                                                                       \
    uint8_t opcode = *reader.AdvanceOffset().ValueOrDie();                \
    DVLOG(1) << "Sequencer dispatching op code: "                         \
             << GetOpcodeInfo(sequencer_opcode_table(), opcode).mnemonic; \
    goto* kDispatchTable[opcode];                                         \
  }

#define DISPATCH_CORE_OPCODE(opcode, body) \
  _dispatch_##opcode : {body} DISPATCH_NEXT()

  DISPATCH_NEXT();

  DISPATCH_CORE_OPCODE(kConstant, {
    ASSIGN_OR_RETURN(auto value, reader.ReadConstant());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    // TODO(b/139121143): until we have full command buffers we need to do this.
    ASSIGN_OR_RETURN(value.buffer,
                     placement.device->allocator()->AllocateConstant(
                         hal::BufferUsage::kConstant | hal::BufferUsage::kAll,
                         std::move(value.buffer)));
    *dst_local = std::move(value);
  });

  DISPATCH_CORE_OPCODE(kCall, {
    auto* old_stack_frame = stack->current_frame();
    ASSIGN_OR_RETURN(const auto& target_function, reader.ReadFunction());
    // TODO(benvanik): rework register storage interface.
    ASSIGN_OR_RETURN(
        const auto* function_def,
        static_cast<const BytecodeModule*>(target_function.module())
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
    auto* old_stack_frame = stack->current_frame();
    ASSIGN_OR_RETURN(const auto& target_function, reader.ReadImportFunction());
    ASSIGN_OR_RETURN(auto* new_stack_frame, stack->PushFrame(target_function));
    // TODO(benvanik): rework register storage interface.
    const auto& signature = target_function.signature();
    new_stack_frame->mutable_registers()->buffer_views.resize(
        signature.argument_count() + signature.result_count());
    RETURN_IF_ERROR(
        reader.CopyInputsAndSwitchStackFrame(old_stack_frame, new_stack_frame));
    DVLOG(1) << "Call native import; stack now: " << stack->DebugString();
    RETURN_IF_ERROR(CallExternalFunction(stack, target_function));
    RETURN_IF_ERROR(reader.CopyResultsAndSwitchStackFrame(old_stack_frame,
                                                          new_stack_frame));
    RETURN_IF_ERROR(stack->PopFrame());
    DVLOG(1) << "Return from native; stack now: " << stack->DebugString();
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

  DISPATCH_CORE_OPCODE(kDynamicDispatch, {
    return UnimplementedErrorBuilder(IREE_LOC)
           << "Unimplemented dynamic_dispatch";
  });

  DISPATCH_CORE_OPCODE(kStaticDispatch, {
    // TODO(benvanik): the real sequencer :)
    ASSIGN_OR_RETURN(auto dispatch_ordinal, reader.ReadInt32());
    ASSIGN_OR_RETURN(auto export_ordinal, reader.ReadUint16_t());
    ASSIGN_OR_RETURN(
        const auto* multi_arch_executable_def,
        static_cast<const BytecodeModule&>(stack->current_frame()->module())
            .LookupMultiArchExecutable(dispatch_ordinal));
    if (export_ordinal >= multi_arch_executable_def->entry_point_count()) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Invalid executable export ordinal " << export_ordinal;
    }
    auto* executable_def = multi_arch_executable_def->executables()->Get(0);
    hal::ExecutableSpec executable_spec;
    executable_spec.format = executable_def->format();
    executable_spec.executable_data = absl::Span<const uint8_t>(
        executable_def->contents()->data(), executable_def->contents()->size());
    auto executable_cache = placement.device->CreateExecutableCache();
    ref_ptr<hal::Executable> executable;
    for (auto* executable_def : *multi_arch_executable_def->executables()) {
      if (!executable_cache->CanPrepareFormat(executable_def->format())) {
        continue;
      }
      hal::ExecutableSpec executable_spec;
      executable_spec.format = executable_def->format();
      executable_spec.executable_data =
          absl::Span<const uint8_t>(executable_def->contents()->data(),
                                    executable_def->contents()->size());
      ASSIGN_OR_RETURN(executable,
                       executable_cache->PrepareExecutable(
                           hal::ExecutableCachingMode::kDefault |
                               hal::ExecutableCachingMode::kAliasProvidedData,
                           executable_spec),
                       _.LogError());
      break;
    }
    if (!executable) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "No executable found for the current driver";
    }

    ASSIGN_OR_RETURN(int workload_x, reader.ReadInt32());
    ASSIGN_OR_RETURN(int workload_y, reader.ReadInt32());
    ASSIGN_OR_RETURN(int workload_z, reader.ReadInt32());

    std::vector<hal::BufferBinding> bindings;
    ASSIGN_OR_RETURN(int input_count, reader.ReadCount());
    for (int i = 0; i < input_count; ++i) {
      ASSIGN_OR_RETURN(auto* input_local, reader.ReadLocal());
      bindings.push_back(hal::BufferBinding(
          input_local->buffer->allowed_access() & hal::MemoryAccess::kAll,
          *input_local));
    }
    ASSIGN_OR_RETURN(int output_count, reader.ReadCount());
    for (int i = 0; i < output_count; ++i) {
      ASSIGN_OR_RETURN(auto* output_local, reader.ReadLocal());
      bindings.push_back(
          hal::BufferBinding(hal::MemoryAccess::kWrite, *output_local));
    }
    ASSIGN_OR_RETURN(int result_count, reader.ReadCount());
    CHECK_EQ(0, result_count) << "Results not yet implemented";

    ASSIGN_OR_RETURN(
        auto cmd,
        placement.device->CreateCommandBuffer(
            hal::CommandBufferMode::kOneShot,
            hal::CommandCategory::kTransfer | hal::CommandCategory::kDispatch),
        _.LogError());
    RETURN_IF_ERROR(cmd->Begin());
    hal::DispatchRequest dispatch_request;
    dispatch_request.executable = executable.get();
    dispatch_request.entry_point = export_ordinal;
    dispatch_request.workload[0] = workload_x;
    dispatch_request.workload[1] = workload_y;
    dispatch_request.workload[2] = workload_z;
    dispatch_request.bindings = bindings;
    RETURN_IF_ERROR(cmd->Dispatch(dispatch_request));
    RETURN_IF_ERROR(cmd->End());
    auto* cmd_ptr = cmd.get();

    auto* queue = placement.device->dispatch_queues().front();
    hal::SubmissionBatch batch;
    batch.command_buffers = absl::MakeConstSpan(&cmd_ptr, 1);
    ASSIGN_OR_RETURN(auto fence, placement.device->CreateFence(0u));
    RETURN_IF_ERROR(queue->Submit(batch, {fence.get(), 1u}));
    RETURN_IF_ERROR(placement.device->WaitAllFences({{fence.get(), 1u}},
                                                    absl::InfiniteFuture()));
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

    // TODO(benvanik): pick an allocator and use that instead.
    CHECK_EQ(heap_type, 0);
    auto* allocator = placement.device->allocator();
    ASSIGN_OR_RETURN(
        dst_local->buffer,
        allocator->Allocate(
            hal::MemoryType::kHostLocal | hal::MemoryType::kDeviceVisible,
            hal::BufferUsage::kAll, allocation_size));
  });

  DISPATCH_CORE_OPCODE(kDiscard, {
    // NOTE: if we were an encoder we would actually discard the buffer.
    ASSIGN_OR_RETURN(auto* local, reader.ReadLocal());
    *local = {};
  });

  DISPATCH_CORE_OPCODE(kComputeRange, {
    ASSIGN_OR_RETURN(auto shape_data, reader.ReadSlotElements<int32_t>());
    ASSIGN_OR_RETURN(auto element_size, reader.ReadUint8_t());
    ASSIGN_OR_RETURN(auto indices, reader.ReadSlotElements<int32_t>());
    ASSIGN_OR_RETURN(auto lengths, reader.ReadSlotElements<int32_t>());
    ASSIGN_OR_RETURN(auto* dst_offset_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* dst_length_local, reader.ReadLocal());

    Shape shape(shape_data);
    ASSIGN_OR_RETURN(device_size_t dst_offset,
                     CalculateOffset(indices, shape, element_size));
    RETURN_IF_ERROR(
        dst_offset_local->buffer->WriteData(0, &dst_offset, sizeof(int32_t)));

    // A buffer range can only be computed for contiguous memory. To ensure that
    // this only requests such, we validate that the offset in the buffer
    // between the start and end indices is the same as the requested size.
    device_size_t dst_length = element_size;
    for (int i = 0; i < lengths.size(); ++i) {
      dst_length *= lengths[i];
      indices[i] += lengths[i] - 1;
    }
    ASSIGN_OR_RETURN(auto end_offset,
                     CalculateOffset(indices, shape, element_size));
    auto offset_based_length = end_offset - dst_offset + element_size;
    if (dst_length != offset_based_length) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Cannot compute range for non-contiguous region of memory;"
             << " shape: " << PrettyPrint(shape.subspan())
             << " indices: " << PrettyPrint(indices)
             << " lengths: " << PrettyPrint(lengths);
    }
    RETURN_IF_ERROR(
        dst_length_local->buffer->WriteData(0, &dst_length, sizeof(int32_t)));
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
    // TODO(b/139299169): implement indirect copies to avoid CPU readback.
    return UnimplementedErrorBuilder(IREE_LOC) << "Unimplemented dynamic_slice";
  });

  DISPATCH_CORE_OPCODE(kStaticSlice, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto offset, reader.ReadInt32());
    ASSIGN_OR_RETURN(auto length, reader.ReadInt32());
    ASSIGN_OR_RETURN(auto type, reader.ReadType());
    ASSIGN_OR_RETURN(auto shape_data, reader.ReadSlotElements<int32_t>());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    Shape new_shape = Shape{shape_data};
    if (new_shape.element_count() * type.element_size() != length) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "New element count " << new_shape.element_count()
             << " != length slice " << length;
    }
    ASSIGN_OR_RETURN(dst_local->buffer,
                     Buffer::Subspan(src_local->buffer, offset, length));
    dst_local->shape = new_shape;
    dst_local->element_size = type.element_size();
  });

  DISPATCH_CORE_OPCODE(kDynamicCopy, {
    // TODO(b/139299169): implement indirect copies to avoid CPU readback.
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto src_offset_span, reader.ReadSlotElements<int32_t>());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto dst_offset_span, reader.ReadSlotElements<int32_t>());
    ASSIGN_OR_RETURN(auto length_span, reader.ReadSlotElements<int32_t>());
    RETURN_IF_ERROR(dst_local->buffer->CopyData(
        dst_offset_span.front(), src_local->buffer.get(),
        src_offset_span.front(), length_span.front()));
  });

  DISPATCH_CORE_OPCODE(kStaticCopy, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto src_offset, reader.ReadInt32());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto dst_offset, reader.ReadInt32());
    ASSIGN_OR_RETURN(auto length, reader.ReadInt32());
    RETURN_IF_ERROR(dst_local->buffer->CopyData(
        dst_offset, src_local->buffer.get(), src_offset, length));
  });

  DISPATCH_CORE_OPCODE(kDynamicFill, {
    // TODO(b/139299169): implement indirect fills to avoid CPU readback.
    return UnimplementedErrorBuilder(IREE_LOC) << "Unimplemented dynamic_fill";
  });

  DISPATCH_CORE_OPCODE(kStaticFill, {
    ASSIGN_OR_RETURN(auto value, reader.ReadInt32());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto dst_offset, reader.ReadInt32());
    ASSIGN_OR_RETURN(auto length, reader.ReadInt32());
    RETURN_IF_ERROR(dst_local->buffer->Fill32(dst_offset, length, value));
  });

  DISPATCH_CORE_OPCODE(kClone, {
    ASSIGN_OR_RETURN(auto* src_local, reader.ReadLocal());
    ASSIGN_OR_RETURN(auto* dst_local, reader.ReadLocal());
    dst_local->element_size = src_local->element_size;
    dst_local->shape = src_local->shape;
    ASSIGN_OR_RETURN(dst_local->buffer, placement.device->allocator()->Allocate(
                                            src_local->buffer->memory_type(),
                                            src_local->buffer->usage(),
                                            src_local->buffer->byte_length()));
    RETURN_IF_ERROR(dst_local->buffer->CopyData(0, src_local->buffer.get()));
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
}

}  // namespace vm
}  // namespace iree
