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

#ifndef IREE_VM_BYTECODE_READER_H_
#define IREE_VM_BYTECODE_READER_H_

#include "absl/base/attributes.h"
#include "absl/container/inlined_vector.h"
#include "base/status.h"
#include "hal/buffer_view.h"
#include "rt/context.h"
#include "rt/stack.h"
#include "rt/stack_frame.h"
#include "schemas/bytecode/bytecode_v0.h"
#include "vm/type.h"

namespace iree {
namespace vm {

class BytecodeReader {
 public:
  explicit BytecodeReader(rt::Stack* stack) : stack_(stack) {}

  int offset() const { return static_cast<int>(bytecode_pc_ - bytecode_base_); }

  StatusOr<const uint8_t*> AdvanceOffset();

  Status SwitchStackFrame(rt::StackFrame* new_stack_frame);
  Status BranchToOffset(int32_t offset);

  Status CopyInputsAndSwitchStackFrame(rt::StackFrame* src_stack_frame,
                                       rt::StackFrame* dst_stack_frame);
  Status CopyResultsAndSwitchStackFrame(rt::StackFrame* src_stack_frame,
                                        rt::StackFrame* dst_stack_frame);
  Status CopySlots();

  StatusOr<hal::BufferView> ReadConstant();

  ABSL_ATTRIBUTE_ALWAYS_INLINE StatusOr<int> ReadCount() {
    return ReadValue<uint8_t>();
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE StatusOr<const Type> ReadType() {
    ASSIGN_OR_RETURN(uint8_t type_index, ReadValue<uint8_t>());
    return Type::FromTypeIndex(type_index);
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE StatusOr<const rt::Function> ReadFunction() {
    ASSIGN_OR_RETURN(auto value, ReadValue<uint32_t>());
    const auto& module = stack_frame_->module();
    return module.LookupFunctionByOrdinal(rt::Function::Linkage::kInternal,
                                          value);
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE StatusOr<const rt::Function>
  ReadImportFunction() {
    ASSIGN_OR_RETURN(auto value, ReadValue<uint32_t>());
    const auto& module = stack_frame_->module();
    return stack_->context()->ResolveImport(&module, value);
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE StatusOr<hal::BufferView*> ReadLocal(
      rt::Registers* registers) {
    ASSIGN_OR_RETURN(auto value, ReadValue<uint16_t>());
    if (value > registers->buffer_views.size()) {
      return OutOfRangeErrorBuilder(IREE_LOC)
             << "Out of bounds local access " << value << " of "
             << registers->buffer_views.size();
    }
    return &registers->buffer_views[value];
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE StatusOr<hal::BufferView*> ReadLocal() {
    return ReadLocal(registers_);
  }

  Status SkipLocals(int count);

  ABSL_ATTRIBUTE_ALWAYS_INLINE StatusOr<uint8_t> ReadUint8_t() {
    return ReadValue<uint8_t>();
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE StatusOr<uint16_t> ReadUint16_t() {
    return ReadValue<uint16_t>();
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE StatusOr<int32_t> ReadInt32() {
    return ReadValue<int32_t>();
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE StatusOr<uint32_t> ReadBlockOffset() {
    return ReadValue<uint32_t>();
  }

  template <typename T, size_t N = 8>
  ABSL_ATTRIBUTE_ALWAYS_INLINE StatusOr<absl::InlinedVector<T, N>>
  ReadSlotElements() {
    ASSIGN_OR_RETURN(auto* local, ReadLocal(registers_));
    absl::InlinedVector<T, N> result(local->shape.element_count());
    if (sizeof(T) == local->element_size) {
      // Fast(ish) path: requested element size matches the actual element size.
      RETURN_IF_ERROR(
          local->buffer->ReadData(0, result.data(), result.size() * sizeof(T)));
    } else {
      // Slow path: need to convert the data.
      switch (local->element_size) {
        case 4: {
          ASSIGN_OR_RETURN(auto mapping, local->buffer->MapMemory<int32_t>(
                                             hal::MemoryAccess::kRead));
          for (size_t i = 0; i < result.size(); ++i) {
            result[i] = static_cast<T>(mapping[i]);
          }
          break;
        }
        case 8: {
          ASSIGN_OR_RETURN(auto mapping, local->buffer->MapMemory<int64_t>(
                                             hal::MemoryAccess::kRead));
          for (size_t i = 0; i < result.size(); ++i) {
            result[i] = static_cast<T>(mapping[i]);
          }
          break;
        }
        default:
          return UnimplementedErrorBuilder(IREE_LOC)
                 << "Unsupported local element size: " << local->element_size;
      }
    }
    return result;
  }

  Status ReadShape(Shape* out_shape);

  StatusOr<Shape> ReadShapePieces();
  StatusOr<Shape> ReadShapePieces(size_t* out_element_count);

  StatusOr<absl::Span<const int32_t>> ReadIndexList();

 private:
  template <typename T>
  ABSL_ATTRIBUTE_ALWAYS_INLINE StatusOr<T> ReadValue() {
    // TODO(benvanik): validate bounds.
    T value = *reinterpret_cast<const T*>(bytecode_pc_);
    bytecode_pc_ += sizeof(T);
    return value;
  }

  rt::Stack* stack_ = nullptr;
  rt::StackFrame* stack_frame_ = nullptr;
  const uint8_t* bytecode_base_ = nullptr;
  const uint8_t* bytecode_limit_ = nullptr;
  const uint8_t* bytecode_pc_ = nullptr;
  rt::Registers* registers_ = nullptr;
};

}  // namespace vm
}  // namespace iree

#endif  // IREE_VM_BYTECODE_READER_H_
