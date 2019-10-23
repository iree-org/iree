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

#include "vm/bytecode_reader.h"

#include "base/shape.h"
#include "base/status.h"
#include "hal/heap_buffer.h"
#include "vm/bytecode_module.h"

namespace iree {
namespace vm {

namespace {

using ::iree::hal::BufferView;
using ::iree::rt::StackFrame;

}  // namespace

StatusOr<const uint8_t*> BytecodeReader::AdvanceOffset() {
  *stack_frame_->mutable_offset() = offset();
  // TODO(benvanik): make a flag and/or remove.
  DVLOG(1) << "dispatch(" << stack_frame_->function().name() << "@" << offset()
           << "): " << int(*bytecode_pc_);
  for (int i = 0; i < registers_->buffer_views.size(); ++i) {
    DVLOG(1) << "local[" << i << "] "
             << registers_->buffer_views[i].DebugStringShort();
  }
  return bytecode_pc_++;
}

Status BytecodeReader::SkipLocals(int count) {
  size_t stride = sizeof(uint16_t) * count;
  if (bytecode_pc_ + stride >= bytecode_limit_) {
    return OutOfRangeErrorBuilder(IREE_LOC) << "Bytecode underflow";
  }
  bytecode_pc_ += stride;
  return OkStatus();
}

Status BytecodeReader::ReadShape(Shape* out_shape) {
  ASSIGN_OR_RETURN(auto shape_dims, ReadIndexList());
  *out_shape = Shape(shape_dims);
  return OkStatus();
}

StatusOr<Shape> BytecodeReader::ReadShapePieces() {
  // TODO(benvanik): rewrite to be faster (multiple offsets to walk both lists).
  ASSIGN_OR_RETURN(auto shape_dims, ReadIndexList());
  if (shape_dims.size() >= kMaxRank) {
    return UnimplementedErrorBuilder(IREE_LOC)
           << "Shapes limited to rank " << kMaxRank << " right now";
  }
  int expected_dynamic_dims = 0;
  for (int i = 0; i < shape_dims.size(); ++i) {
    if (shape_dims[i] == -1) {
      ++expected_dynamic_dims;
    }
  }

  Shape shape(shape_dims);
  ASSIGN_OR_RETURN(int dynamic_dims, ReadCount());
  if (dynamic_dims != expected_dynamic_dims) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Expected " << expected_dynamic_dims << " dynamic dims but only "
           << dynamic_dims << " provided";
  } else if (dynamic_dims) {
    for (int i = 0; i < shape_dims.size(); ++i) {
      if (shape_dims[i] != -1) {
        continue;
      }
      // TODO(benvanik): kill this embarrassment.
      ASSIGN_OR_RETURN(auto dims_piece, ReadSlotElements<int32_t>());
      if (dims_piece.size() != 1) {
        return InvalidArgumentErrorBuilder(IREE_LOC)
               << "Dims piece has rank " << dims_piece.size() << "; must be 1";
      }
      shape[i] = dims_piece[0];
    }
  }
  return shape;
}

StatusOr<Shape> BytecodeReader::ReadShapePieces(size_t* out_element_count) {
  ASSIGN_OR_RETURN(auto shape, ReadShapePieces());
  *out_element_count = shape.element_count();
  return shape;
}

StatusOr<absl::Span<const int32_t>> BytecodeReader::ReadIndexList() {
  ASSIGN_OR_RETURN(int count, ReadCount());
  int stride = count * sizeof(int32_t);
  if (bytecode_pc_ + stride >= bytecode_limit_) {
    return OutOfRangeErrorBuilder(IREE_LOC) << "Bytecode underflow";
  }
  auto list = absl::Span<const int32_t>(
      reinterpret_cast<const int32_t*>(bytecode_pc_), count);
  bytecode_pc_ += stride;
  return list;
}

Status BytecodeReader::SwitchStackFrame(StackFrame* new_stack_frame) {
  // Flush old state.
  auto* old_stack_frame = stack_frame_;
  if (old_stack_frame) {
    *old_stack_frame->mutable_offset() = offset();
  }

  // Switch the frame. The FiberState holds the full stack, this is just the
  // current one for easy access.
  stack_frame_ = new_stack_frame;

  // Setup state pointers for faster dereferencing.
  const auto& function = new_stack_frame->function();
  ASSIGN_OR_RETURN(
      const auto* function_def,
      static_cast<const BytecodeModule*>(function.module())
          ->GetFunctionDef(function.linkage(), function.ordinal()));
  const auto& bytecode = *function_def->bytecode();
  bytecode_base_ = bytecode.contents()->Data();
  bytecode_limit_ = bytecode_base_ + bytecode.contents()->size();
  bytecode_pc_ = bytecode_base_ + new_stack_frame->offset();
  registers_ = new_stack_frame->mutable_registers();
  return OkStatus();
}

Status BytecodeReader::CopyInputsAndSwitchStackFrame(
    StackFrame* src_stack_frame, StackFrame* dst_stack_frame) {
  ASSIGN_OR_RETURN(size_t src_count, ReadCount());
  auto& dst_buffer_views = dst_stack_frame->mutable_registers()->buffer_views;
  for (int i = 0; i < std::min(src_count, dst_buffer_views.size()); ++i) {
    ASSIGN_OR_RETURN(auto* src_local,
                     ReadLocal(src_stack_frame->mutable_registers()));
    dst_buffer_views[i] = *src_local;
  }
  return SwitchStackFrame(dst_stack_frame);
}

Status BytecodeReader::CopyResultsAndSwitchStackFrame(
    StackFrame* src_stack_frame, StackFrame* dst_stack_frame) {
  ASSIGN_OR_RETURN(int32_t src_count, ReadCount());
  // TODO(benvanik): avoid vector.
  absl::InlinedVector<BufferView*, 8> src_locals(src_count);
  for (int i = 0; i < src_count; ++i) {
    ASSIGN_OR_RETURN(src_locals[i],
                     ReadLocal(src_stack_frame->mutable_registers()));
  }
  RETURN_IF_ERROR(SwitchStackFrame(dst_stack_frame));
  ASSIGN_OR_RETURN(int32_t dst_count, ReadCount());
  if (src_count != dst_count) {
    return OutOfRangeErrorBuilder(IREE_LOC)
           << "Src and dst value counts differ: " << src_count << " vs "
           << dst_count;
  }
  for (int i = 0; i < dst_count; ++i) {
    ASSIGN_OR_RETURN(auto* dst_local,
                     ReadLocal(dst_stack_frame->mutable_registers()));
    *dst_local = *src_locals[i];
  }
  return OkStatus();
}

Status BytecodeReader::CopySlots() {
  ASSIGN_OR_RETURN(int32_t count, ReadCount());
  for (int i = 0; i < count; ++i) {
    ASSIGN_OR_RETURN(auto* src_local,
                     ReadLocal(stack_frame_->mutable_registers()));
    ASSIGN_OR_RETURN(auto* dst_local,
                     ReadLocal(stack_frame_->mutable_registers()));
    *dst_local = *src_local;
  }
  return OkStatus();
}

Status BytecodeReader::BranchToOffset(int32_t offset) {
  const uint8_t* new_bytecode_pc = bytecode_base_ + offset;
  if (new_bytecode_pc < bytecode_base_ || new_bytecode_pc > bytecode_limit_) {
    return OutOfRangeErrorBuilder(IREE_LOC)
           << "Branch target " << offset
           << " is out of bounds of the function bytecode ("
           << static_cast<size_t>(bytecode_limit_ - bytecode_base_)
           << "b total)";
  }
  bytecode_pc_ = new_bytecode_pc;
  return OkStatus();
}

StatusOr<BufferView> BytecodeReader::ReadConstant() {
  BufferView buffer_view;

  // Element type defines the buffer_view size (but we don't really care about
  // the data format).
  ASSIGN_OR_RETURN(auto element_type, ReadType());
  buffer_view.element_size = element_type.element_size();

  // Parse shape - constants always define a full shape.
  RETURN_IF_ERROR(ReadShape(&buffer_view.shape));

  // Read encoding to determine how the constant data is stored in the file.
  ASSIGN_OR_RETURN(auto encoding, ReadValue<ConstantEncoding>());

  // Get buffer for the constant data.
  switch (encoding) {
    case ConstantEncoding::kDense: {
      // Validate we have all constant data present.
      device_size_t serialized_length = buffer_view.byte_length();
      if (bytecode_pc_ + serialized_length >= bytecode_limit_) {
        return OutOfRangeErrorBuilder(IREE_LOC)
               << "Constant data out of bounds";
      }

      buffer_view.buffer = hal::HeapBuffer::Wrap(
          hal::MemoryType::kHostLocal, hal::BufferUsage::kAll, bytecode_pc_,
          serialized_length);
      bytecode_pc_ += serialized_length;
      break;
    }
    case ConstantEncoding::kSplat: {
      // Validate we have at least one element worth of data in the buffer.
      if (bytecode_pc_ + buffer_view.element_size >= bytecode_limit_) {
        return OutOfRangeErrorBuilder(IREE_LOC)
               << "Constant data out of bounds";
      }

      // TODO(benvanik): replace with fancy constant pool and such.
      // NOTE: this is not much different than if a alloc_heap+broadcast pair
      // had been in the IR.
      buffer_view.buffer = hal::HeapBuffer::Allocate(
          hal::MemoryType::kHostLocal, hal::BufferUsage::kAll,
          buffer_view.byte_length());
      switch (buffer_view.element_size) {
        case 1: {
          uint8_t value = *reinterpret_cast<const uint8_t*>(bytecode_pc_);
          RETURN_IF_ERROR(buffer_view.buffer->Fill8(value));
          break;
        }
        case 2: {
          uint16_t value = *reinterpret_cast<const uint16_t*>(bytecode_pc_);
          RETURN_IF_ERROR(buffer_view.buffer->Fill16(value));
          break;
        }
        case 4: {
          uint32_t value = *reinterpret_cast<const uint32_t*>(bytecode_pc_);
          RETURN_IF_ERROR(buffer_view.buffer->Fill32(value));
          break;
        }
        case 8: {
          // TODO(benvanik): add Fill64.
          uint64_t value = *reinterpret_cast<const uint64_t*>(bytecode_pc_);
          ASSIGN_OR_RETURN(auto mapping,
                           buffer_view.buffer->MapMemory<uint64_t>(
                               hal::MemoryAccess::kDiscardWrite));
          auto mapped_data = mapping.mutable_contents();
          for (int i = 0; i < mapping.size(); ++i) {
            mapped_data[i] = value;
          }
          break;
        }
        default:
          return UnimplementedErrorBuilder(IREE_LOC)
                 << "Unimplemented splat element stride "
                 << buffer_view.element_size;
      }
      bytecode_pc_ += buffer_view.element_size;
      break;
    }
    default:
      return UnimplementedErrorBuilder(IREE_LOC)
             << "Unimplemented constant encoding "
             << static_cast<int>(encoding);
  }

  return buffer_view;
}

}  // namespace vm
}  // namespace iree
