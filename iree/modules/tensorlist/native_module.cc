// Copyright 2020 Google LLC
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

#include "iree/modules/tensorlist/native_module.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "iree/base/api.h"
#include "iree/base/status.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/vm/native_module_cc.h"
#include "iree/vm/ref_cc.h"

namespace iree {

//===----------------------------------------------------------------------===//
// TensorList runtime type.
// This is the type that backs the `tensorlist.list` VM type.
//===----------------------------------------------------------------------===//

namespace {
class TensorList final : public iree::vm::RefObject<TensorList> {
 public:
  TensorList(absl::Span<const int32_t> shape, iree_hal_element_type_t dtype)
      : shape_(shape.begin(), shape.end()), dtype_(dtype) {}

  TensorList(const vm::ref<TensorList>& other)
      : shape_(other->shape_), dtype_(other->dtype_) {
    CopyFrom(other);
  }

  void Resize(int32_t num_elements) { list_.resize(num_elements); }
  // Copy from another iree_tensorlist.
  // vm::ref has deleted copy operator=, so we can't use vector's operator=.
  void CopyFrom(const vm::ref<TensorList>& other) {
    list_.clear();
    for (auto& element : other->list_) {
      list_.push_back(vm::retain_ref(element));
    }
  }
  const vm::ref<iree_hal_buffer_view_t>& GetItem(int32_t index) const {
    // TODO(silvasean): Correct out-of-bounds behavior.
    return list_.at(index);
  }
  void SetItem(int32_t index, vm::ref<iree_hal_buffer_view_t> item) {
    // TODO(silvasean): Correct out-of-bounds behavior.
    list_.at(index) = std::move(item);
  }
  void Print() {
    fprintf(stderr, "tensorlist\n");
    for (auto& item : list_) {
      fprintf(stderr, "  item: %p\n", (void*)item.get());
    }
  }
  size_t Size() { return list_.size(); }
  absl::Span<int32_t> Shape() {
    return absl::Span<int32_t>(shape_.data(), shape_.size());
  }

  static StatusOr<vm::ref<TensorList>> FromTensor(
      vm::ref<iree_hal_buffer_view_t> tensor) {
    size_t rank = iree_hal_buffer_view_shape_rank(tensor.get());
    if (rank == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected rank > 0 buffer view");
    }
    absl::InlinedVector<int32_t, 6> shape(rank);
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_shape(tensor.get(), rank, shape.data(), nullptr));

    auto element_type = iree_hal_buffer_view_element_type(tensor.get());

    int32_t list_elements = shape[0];
    absl::Span<int32_t> element_shape(shape.data() + 1, shape.size() - 1);

    TensorList* list = new TensorList(element_shape, element_type);
    list->Resize(list_elements);

    // The python pseudocode for this is:
    // for i in range(t.shape[0]):
    //   list[i] = t[i,...]
    absl::InlinedVector<int32_t, 6> start_indices(shape.size());
    absl::InlinedVector<int32_t, 6> lengths = shape;
    lengths[0] = 1;
    for (int i = 0, e = list_elements; i < e; i++) {
      start_indices[0] = i;
      iree_device_size_t start_offset = 0;
      iree_device_size_t subview_length = 0;
      IREE_RETURN_IF_ERROR(iree_hal_buffer_view_compute_range(
          tensor.get(), start_indices.data(), start_indices.size(),
          lengths.data(), lengths.size(), &start_offset, &subview_length));
      vm::ref<iree_hal_buffer_t> subview_buffer;
      IREE_RETURN_IF_ERROR(iree_hal_buffer_subspan(
          iree_hal_buffer_view_buffer(tensor.get()), start_offset,
          subview_length, &subview_buffer));

      iree_hal_buffer_view_t* slice = nullptr;
      IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
          subview_buffer.get(), element_shape.data(), element_shape.size(),
          iree_hal_buffer_view_element_type(tensor.get()), &slice));
      list->SetItem(i, slice);
    }
    return list;
  }

  StatusOr<vm::ref<iree_hal_buffer_view_t>> Stack(
      vm::ref<iree_hal_allocator_t> hal_allocator) {
    size_t num_tensors = Size();
    if (num_tensors == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected non-empty list");
    }

    // Validate that all buffers are of the right shape/type.
    absl::Span<int32_t> shape(shape_);
    iree_hal_element_type_t type(dtype_);
    for (size_t i = 0; i < num_tensors; i++) {
      auto item = GetItem(i).get();
      if (!item) continue;
      size_t element_rank = iree_hal_buffer_view_shape_rank(item);
      absl::InlinedVector<int32_t, 6> element_shape(element_rank);
      IREE_RETURN_IF_ERROR(iree_hal_buffer_view_shape(
          item, element_rank, element_shape.data(), nullptr));
      if (absl::MakeSpan(shape) != absl::MakeSpan(element_shape) ||
          iree_hal_buffer_view_element_type(item) != type) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "stacking list with elements of different shapes or element types; "
            "mismatch between element 0 and element %zu",
            i);
        ;
      }
    }

    vm::ref<iree_hal_buffer_t> result_buffer;
    size_t num_elements_per_tensor = 1;
    for (int32_t dim : shape) {
      num_elements_per_tensor *= dim;
    }

    size_t element_size = iree_hal_element_byte_count(type);
    size_t num_result_elements = num_elements_per_tensor * num_tensors;
    size_t result_byte_size = num_result_elements * element_size;
    IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
        hal_allocator.get(),
        static_cast<iree_hal_memory_type_t>(
            IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE),
        IREE_HAL_BUFFER_USAGE_ALL, result_byte_size, &result_buffer));

    IREE_RETURN_IF_ERROR(CopyTensorBytes(result_buffer.get()));

    absl::InlinedVector<int32_t, 4> result_shape;
    result_shape.push_back(Size());
    for (int32_t dim : shape) {
      result_shape.push_back(dim);
    }
    vm::ref<iree_hal_buffer_view_t> result_view;
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_create(result_buffer.get(), result_shape.data(),
                                    result_shape.size(), type, &result_view));
    return std::move(result_view);
  }

  StatusOr<vm::ref<iree_hal_buffer_view_t>> Concat(
      vm::ref<iree_hal_allocator_t> hal_allocator) {
    size_t num_tensors = Size();
    if (num_tensors == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected non-empty list");
    }

    if (shape_.empty()) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "stacking rank must be greater than zero");
    }

    size_t rank = iree_hal_buffer_view_shape_rank(GetItem(0).get());
    iree_hal_element_type_t type = dtype_;
    absl::InlinedVector<int32_t, 6> shape(rank);
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_shape(GetItem(0).get(), rank,
                                                    shape.data(), nullptr));
    const size_t num_rows = num_tensors * shape[0];
    for (size_t i = 0; i < num_tensors; i++) {
      auto item = GetItem(i).get();
      if (!item) continue;
      size_t element_rank = iree_hal_buffer_view_shape_rank(item);
      if (element_rank < 1) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stacking rank %zu must be greater than zero",
                                i);
      }

      absl::InlinedVector<int32_t, 6> element_shape(element_rank);
      IREE_RETURN_IF_ERROR(iree_hal_buffer_view_shape(
          GetItem(i).get(), element_rank, element_shape.data(), nullptr));

      if (absl::MakeSpan(shape).subspan(1) !=
              absl::MakeSpan(element_shape).subspan(1) ||
          iree_hal_buffer_view_element_type(GetItem(i).get()) != type) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "stacking list with elements of different shapes or element types; "
            "mismatch between element 0 and element %zu",
            i);
      }
    }

    vm::ref<iree_hal_buffer_t> result_buffer;
    size_t num_elements_per_row = 1;
    for (int32_t dim : absl::MakeSpan(shape).subspan(1)) {
      num_elements_per_row *= dim;
    }
    size_t element_size = iree_hal_buffer_view_element_size(GetItem(0).get());
    size_t num_result_elements = num_elements_per_row * num_rows;
    size_t result_byte_size = num_result_elements * element_size;
    IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
        hal_allocator.get(),
        static_cast<iree_hal_memory_type_t>(
            IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE),
        IREE_HAL_BUFFER_USAGE_ALL, result_byte_size, &result_buffer));

    IREE_RETURN_IF_ERROR(CopyTensorBytes(result_buffer.get()));

    absl::InlinedVector<int32_t, 4> result_shape;
    result_shape.push_back(num_rows);
    for (int32_t dim : absl::MakeSpan(shape).subspan(1)) {
      result_shape.push_back(dim);
    }
    vm::ref<iree_hal_buffer_view_t> result_view;
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_create(result_buffer.get(), result_shape.data(),
                                    result_shape.size(), type, &result_view));

    return std::move(result_view);
  }

 private:
  iree_status_t CopyTensorBytes(iree_hal_buffer_t* buffer) {
    iree_hal_buffer_mapping_t result_mapping;
    iree_device_size_t dest_byte_size = iree_hal_buffer_byte_length(buffer);
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
        buffer, IREE_HAL_MEMORY_ACCESS_WRITE,
        /*byte_offset=*/0,
        /*byte_length=*/dest_byte_size, &result_mapping));

    // Copy each buffer into the result at the right offset.
    // This is just a naive map+memcpy.
    // If this is a bottleneck, simply optimizing this code here locally is
    // probably not the best answer. A better solution will use
    // iree_hal_command_buffer_copy_buffer to do the copies, but that will
    // require changing this op signature to take a command buffer and to make
    // sure that each of the contained tensors have
    // IREE_HAL_BUFFER_USAGE_TRANSFER. Both of these will probably require
    // compiler changes. In fact, we might want to expand this operation fully
    // in the compiler at which point there will be no "stack" function inside
    // this module at all.
    size_t num_tensors = Size();
    size_t tensor_byte_size = iree_hal_element_byte_count(dtype_);
    for (auto dim : shape_) tensor_byte_size *= dim;
    for (size_t i = 0; i < num_tensors; i++) {
      iree_hal_buffer_view_t* tensor = GetItem(i).get();

      auto block_begin = result_mapping.contents.data + i * tensor_byte_size;
      auto block_size = tensor_byte_size;

      if (!tensor) {
        memset(block_begin, 0, block_size);
        continue;
      }

      iree_hal_buffer_t* tensor_buffer = iree_hal_buffer_view_buffer(tensor);
      IREE_RETURN_IF_ERROR(
          iree_hal_buffer_read_data(tensor_buffer, 0, block_begin, block_size));
    }

    iree_hal_buffer_unmap_range(&result_mapping);
    return iree_ok_status();
  }

  std::vector<vm::ref<iree_hal_buffer_view_t>> list_;
  std::vector<int32_t> shape_;
  iree_hal_element_type_t dtype_;
};
}  // namespace

//===----------------------------------------------------------------------===//
// `tensorlist.list` VM type registration.
//===----------------------------------------------------------------------===//

static iree_vm_ref_type_descriptor_t iree_tensorlist_descriptor = {0};

// Register our type with the vm::ref<T> static machinery.
namespace vm {
template <>
struct ref_type_descriptor<TensorList> {
  static const iree_vm_ref_type_descriptor_t* get() {
    return &iree_tensorlist_descriptor;
  }
};
}  // namespace vm

extern "C" iree_status_t iree_tensorlist_module_register_types() {
  static bool has_registered = false;
  if (has_registered) return iree_ok_status();
  IREE_VM_REGISTER_CC_TYPE(TensorList, "tensorlist.list",
                           iree_tensorlist_descriptor);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

// Extremely low-performance helper for dealing with buffer views that
// contain scalar int32_t's.
// TODO(silvasean): Change relevant ops to just take a VM i32.
// That will require doing a bit more work in the compiler for conversion.
static StatusOr<int32_t> ReadInt32FromScalarBufferView(
    iree_hal_buffer_view_t* buffer_view) {
  if (iree_hal_buffer_view_element_type(buffer_view) !=
      IREE_HAL_ELEMENT_TYPE_SINT_32) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected i32 buffer view");
  }
  if (iree_hal_buffer_view_shape_rank(buffer_view) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected rank-0 buffer view");
  }
  iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(buffer_view);
  iree_hal_buffer_mapping_t mapped_memory;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MEMORY_ACCESS_READ, 0, 4, &mapped_memory));
  int32_t scalar = *reinterpret_cast<int32_t*>(mapped_memory.contents.data);
  iree_hal_buffer_unmap_range(&mapped_memory);
  return scalar;
}

static StatusOr<std::vector<int32_t>> ReadInt32VectorFromBufferView(
    iree_hal_buffer_view_t* buffer_view) {
  if (iree_hal_buffer_view_element_type(buffer_view) !=
      IREE_HAL_ELEMENT_TYPE_SINT_32) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected i32 buffer view");
  }
  if (iree_hal_buffer_view_shape_rank(buffer_view) != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected rank-1 buffer view");
  }

  int32_t length;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_shape(
      buffer_view, /*rank_capacity=*/1, &length, nullptr));

  iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(buffer_view);
  std::vector<int32_t> contents(length);
  IREE_RETURN_IF_ERROR(iree_hal_buffer_read_data(
      buffer, 0, contents.data(), contents.size() * sizeof(int32_t)));
  return contents;
}

namespace {
class TensorListModuleState final {
 public:
  TensorListModuleState() = default;
  ~TensorListModuleState() = default;

  // tensorlist.reserve(%element_shape, %num_elements) -> %list
  StatusOr<vm::ref<TensorList>> Reserve(
      vm::ref<iree_hal_buffer_view_t> element_shape,
      vm::ref<iree_hal_buffer_view_t> num_elements_buf,
      iree_hal_element_type_t element_type) {
    IREE_ASSIGN_OR_RETURN(std::vector<int32_t> shape,
                          ReadInt32VectorFromBufferView(element_shape.get()));
    TensorList* tensorlist = new TensorList(shape, element_type);
    IREE_ASSIGN_OR_RETURN(int32_t num_elements, ReadInt32FromScalarBufferView(
                                                    num_elements_buf.get()));
    tensorlist->Resize(num_elements);
    return tensorlist;
  }

  // tensorlist.get_item(%list, %index, %element_shape) -> %item
  StatusOr<vm::ref<iree_hal_buffer_view_t>> GetItem(
      vm::ref<TensorList> tensorlist,
      vm::ref<iree_hal_buffer_view_t> index_buf) {
    IREE_ASSIGN_OR_RETURN(int32_t index,
                          ReadInt32FromScalarBufferView(index_buf.get()));
    return vm::retain_ref(tensorlist->GetItem(index).get());
  }

  // tensorlist.set_item(%list, %index, %item) -> %new_list
  StatusOr<vm::ref<TensorList>> SetItem(
      vm::ref<TensorList> list, vm::ref<iree_hal_buffer_view_t> index_buf,
      vm::ref<iree_hal_buffer_view_t> item) {
    IREE_ASSIGN_OR_RETURN(int32_t index,
                          ReadInt32FromScalarBufferView(index_buf.get()));
    TensorList* new_list = new TensorList(list);
    new_list->SetItem(index, vm::retain_ref(item));
    return new_list;
  }

  // tensorlist.from_tensor(%tensor, %element_shape) -> %list
  StatusOr<vm::ref<TensorList>> FromTensor(
      vm::ref<iree_hal_buffer_view_t> tensor) {
    return TensorList::FromTensor(tensor);
  }

  // tensorlist.concat(%list) -> %list
  StatusOr<vm::ref<iree_hal_buffer_view_t>> Concat(
      vm::ref<iree_hal_allocator_t> allocator, vm::ref<TensorList> list) {
    return list->Concat(allocator);
  }

  // tensorlist.stack(%list, %element_shape, %num_elements) -> %list
  StatusOr<vm::ref<iree_hal_buffer_view_t>> Stack(
      vm::ref<iree_hal_allocator_t> allocator, vm::ref<TensorList> list,
      vm::ref<iree_hal_buffer_view_t> num_elements_buffer_view) {
    IREE_ASSIGN_OR_RETURN(
        int32_t num_elements,
        ReadInt32FromScalarBufferView(num_elements_buffer_view.get()));
    if (num_elements != -1 && list->Size() != num_elements) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "num_elements arg to tesorlist.stack doesn't match the list "
          "size");
    }
    return list->Stack(allocator);
  }
};
}  // namespace

static const vm::NativeFunction<TensorListModuleState>
    kTensorListModuleFunctions[] = {
        vm::MakeNativeFunction("reserve", &TensorListModuleState::Reserve),
        vm::MakeNativeFunction("get_item", &TensorListModuleState::GetItem),
        vm::MakeNativeFunction("set_item", &TensorListModuleState::SetItem),
        vm::MakeNativeFunction("from_tensor",
                               &TensorListModuleState::FromTensor),
        vm::MakeNativeFunction("concat", &TensorListModuleState::Concat),
        vm::MakeNativeFunction("stack", &TensorListModuleState::Stack),
};

namespace {
class TensorListModule final : public vm::NativeModule<TensorListModuleState> {
 public:
  using vm::NativeModule<TensorListModuleState>::NativeModule;

  // Creates per-context state when the module is added to a new context.
  // May be called from any thread.
  StatusOr<std::unique_ptr<TensorListModuleState>> CreateState(
      iree_allocator_t allocator) override {
    auto state = std::make_unique<TensorListModuleState>();
    return state;
  }
};
}  // namespace

extern "C" iree_status_t iree_tensorlist_module_create(
    iree_allocator_t allocator, iree_vm_module_t** out_module) {
  if (!out_module) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  *out_module = NULL;
  auto module = std::make_unique<TensorListModule>(
      "tensorlist", allocator, absl::MakeConstSpan(kTensorListModuleFunctions));
  *out_module = module.release()->interface();
  return iree_ok_status();
}

}  // namespace iree
