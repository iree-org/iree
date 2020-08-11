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
#include "iree/base/ref_ptr.h"
#include "iree/base/status.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/vm/module_abi_cc.h"
#include "iree/vm/module_abi_packing.h"

namespace iree {

//===----------------------------------------------------------------------===//
// TensorList runtime type.
// This is the type that backs the `tensorlist.list` VM type.
//===----------------------------------------------------------------------===//

namespace {
class TensorList final : public RefObject<TensorList> {
 public:
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

  static StatusOr<vm::ref<TensorList>> FromTensor(
      vm::ref<iree_hal_buffer_view_t> tensor) {
    size_t rank = iree_hal_buffer_view_shape_rank(tensor.get());
    if (rank == 0) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "expected rank > 0 buffer view";
    }
    absl::InlinedVector<int32_t, 6> shape(rank);
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_shape(tensor.get(), rank, shape.data(), nullptr));

    TensorList* list = new TensorList;
    list->Resize(shape[0]);
    // The python pseudocode for this is:
    // for i in range(t.shape[0]):
    //   list[i] = t[i,...]
    absl::InlinedVector<int32_t, 6> start_indices(shape.size());
    absl::InlinedVector<int32_t, 6> lengths = shape;
    lengths[0] = 1;
    for (int i = 0, e = shape[0]; i < e; i++) {
      start_indices[0] = i;
      iree_device_size_t start_offset = 0;
      iree_device_size_t subview_length = 0;
      IREE_RETURN_IF_ERROR(iree_hal_buffer_view_compute_range(
          tensor.get(), start_indices.data(), start_indices.size(),
          lengths.data(), lengths.size(), &start_offset, &subview_length));
      vm::ref<iree_hal_buffer_t> subview_buffer;
      IREE_RETURN_IF_ERROR(iree_hal_buffer_subspan(
          iree_hal_buffer_view_buffer(tensor.get()), start_offset,
          subview_length, iree_allocator_system(), &subview_buffer));

      iree_hal_buffer_view_t* slice = nullptr;
      IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
          subview_buffer.get(), shape.data() + 1, shape.size() - 1,
          iree_hal_buffer_view_element_type(tensor.get()),
          iree_allocator_system(), &slice));
      list->SetItem(i, slice);
    }
    return list;
  }

  StatusOr<vm::ref<iree_hal_buffer_view_t>> Stack() {
    size_t num_tensors = Size();
    if (num_tensors == 0) {
      return InvalidArgumentErrorBuilder(IREE_LOC) << "expected non-empty list";
    }
    for (size_t i = 0; i < num_tensors; i++) {
      if (!GetItem(i).get()) {
        return InvalidArgumentErrorBuilder(IREE_LOC)
               << "uninitialized element in list";
      }
    }

    size_t rank = iree_hal_buffer_view_shape_rank(GetItem(0).get());
    iree_hal_element_type_t type =
        iree_hal_buffer_view_element_type(GetItem(0).get());
    absl::InlinedVector<int32_t, 6> shape(rank);
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_shape(GetItem(0).get(), rank,
                                                    shape.data(), nullptr));
    for (size_t i = 0; i < num_tensors; i++) {
      size_t element_rank = iree_hal_buffer_view_shape_rank(GetItem(i).get());
      absl::InlinedVector<int32_t, 6> element_shape(element_rank);
      IREE_RETURN_IF_ERROR(iree_hal_buffer_view_shape(
          GetItem(i).get(), element_rank, element_shape.data(), nullptr));
      if (absl::MakeSpan(shape) != absl::MakeSpan(element_shape) ||
          iree_hal_buffer_view_element_type(GetItem(i).get()) != type) {
        return InvalidArgumentErrorBuilder(IREE_LOC)
               << "stacking list with elements of different shapes or element "
                  "types. Mismatch between element 0 and element "
               << i;
      }
    }

    vm::ref<iree_hal_buffer_t> result_buffer;
    size_t num_elements_per_tensor = 1;
    for (int32_t dim : shape) {
      num_elements_per_tensor *= dim;
    }
    size_t element_size = iree_hal_buffer_view_element_size(GetItem(0).get());
    size_t num_result_elements = num_elements_per_tensor * num_tensors;
    size_t result_byte_size = num_result_elements * element_size;
    iree_hal_allocator_t* hal_allocator = iree_hal_buffer_allocator(
        iree_hal_buffer_view_buffer(GetItem(0).get()));
    IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
        hal_allocator,
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
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
        result_buffer.get(), result_shape.data(), result_shape.size(), type,
        iree_allocator_system(), &result_view));
    return std::move(result_view);
  }

  StatusOr<vm::ref<iree_hal_buffer_view_t>> Concat() {
    size_t num_tensors = Size();
    if (num_tensors == 0) {
      return InvalidArgumentErrorBuilder(IREE_LOC) << "expected non-empty list";
    }
    for (size_t i = 0; i < num_tensors; i++) {
      if (!GetItem(i).get()) {
        return InvalidArgumentErrorBuilder(IREE_LOC)
               << "uninitialized element in list";
      }
    }

    size_t rank = iree_hal_buffer_view_shape_rank(GetItem(0).get());
    iree_hal_element_type_t type =
        iree_hal_buffer_view_element_type(GetItem(0).get());
    absl::InlinedVector<int32_t, 6> shape(rank);
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_shape(GetItem(0).get(), rank,
                                                    shape.data(), nullptr));
    size_t num_rows = 0;
    for (size_t i = 0; i < num_tensors; i++) {
      size_t element_rank = iree_hal_buffer_view_shape_rank(GetItem(i).get());
      if (element_rank < 1) {
        return InvalidArgumentErrorBuilder(IREE_LOC)
               << "stacking rank must be greater than zero." << i;
      }

      absl::InlinedVector<int32_t, 6> element_shape(element_rank);
      IREE_RETURN_IF_ERROR(iree_hal_buffer_view_shape(
          GetItem(i).get(), element_rank, element_shape.data(), nullptr));
      num_rows += element_shape.front();

      if (absl::MakeSpan(shape).subspan(1) !=
              absl::MakeSpan(element_shape).subspan(1) ||
          iree_hal_buffer_view_element_type(GetItem(i).get()) != type) {
        return InvalidArgumentErrorBuilder(IREE_LOC)
               << "stacking list with elements of different shapes or element "
                  "types. Mismatch between element 0 and element "
               << i;
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
    iree_hal_allocator_t* hal_allocator = iree_hal_buffer_allocator(
        iree_hal_buffer_view_buffer(GetItem(0).get()));
    IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
        hal_allocator,
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
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
        result_buffer.get(), result_shape.data(), result_shape.size(), type,
        iree_allocator_system(), &result_view));

    return std::move(result_view);
  }

 private:
  iree_status_t CopyTensorBytes(iree_hal_buffer_t* buffer) {
    iree_hal_mapped_memory_t result_mapping;
    iree_device_size_t dest_byte_size = iree_hal_buffer_byte_length(buffer);
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_map(buffer, IREE_HAL_MEMORY_ACCESS_WRITE,
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
    size_t offset = 0;
    for (size_t i = 0; i < num_tensors; i++) {
      iree_hal_buffer_view_t* tensor = GetItem(i).get();
      iree_hal_buffer_t* tensor_buffer = iree_hal_buffer_view_buffer(tensor);
      iree_hal_mapped_memory_t tensor_mapping;
      iree_device_size_t tensor_byte_size =
          iree_hal_buffer_byte_length(tensor_buffer);

      IREE_RETURN_IF_ERROR(
          iree_hal_buffer_map(tensor_buffer, IREE_HAL_MEMORY_ACCESS_READ, 0,
                              tensor_byte_size, &tensor_mapping));

      memcpy(result_mapping.contents.data + offset,
             tensor_mapping.contents.data, tensor_byte_size);
      offset += tensor_byte_size;

      IREE_RETURN_IF_ERROR(
          iree_hal_buffer_unmap(tensor_buffer, &tensor_mapping));
    }

    return iree_hal_buffer_unmap(buffer, &result_mapping);
  }

  std::vector<vm::ref<iree_hal_buffer_view_t>> list_;
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
    return InvalidArgumentErrorBuilder(IREE_LOC) << "expected i32 buffer view";
  }
  if (iree_hal_buffer_view_shape_rank(buffer_view) != 0) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "expected rank-0 buffer view";
  }
  iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(buffer_view);
  iree_hal_mapped_memory_t mapped_memory;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map(buffer, IREE_HAL_MEMORY_ACCESS_READ,
                                           0, 4, &mapped_memory));
  int32_t scalar = *reinterpret_cast<int32_t*>(mapped_memory.contents.data);
  IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap(buffer, &mapped_memory));
  return scalar;
}

namespace {
class TensorListModuleState final {
 public:
  TensorListModuleState() = default;
  ~TensorListModuleState() = default;

  // tensorlist.reserve(%element_shape, %num_elements) -> %list
  StatusOr<vm::ref<TensorList>> Reserve(
      vm::ref<iree_hal_buffer_view_t> element_shape,
      vm::ref<iree_hal_buffer_view_t> num_elements_buf) {
    // TODO(silvasean): Emulate element shape and dtype tracking in TensorList.
    (void)element_shape;
    TensorList* tensorlist = new TensorList;
    IREE_ASSIGN_OR_RETURN(int32_t num_elements, ReadInt32FromScalarBufferView(
                                                    num_elements_buf.get()));
    tensorlist->Resize(num_elements);
    return tensorlist;
  }

  // tensorlist.get_item(%list, %index, %element_shape) -> %item
  StatusOr<vm::ref<iree_hal_buffer_view_t>> GetItem(
      vm::ref<TensorList> tensorlist, vm::ref<iree_hal_buffer_view_t> index_buf,
      vm::ref<iree_hal_buffer_view_t> element_shape) {
    // TODO(silvasean): Emulate element shape and dtype tracking in TensorList.
    (void)element_shape;
    IREE_ASSIGN_OR_RETURN(int32_t index,
                          ReadInt32FromScalarBufferView(index_buf.get()));
    return vm::retain_ref(tensorlist->GetItem(index).get());
  }

  // tensorlist.set_item(%list, %index, %item) -> %new_list
  StatusOr<vm::ref<TensorList>> SetItem(
      vm::ref<TensorList> list, vm::ref<iree_hal_buffer_view_t> index_buf,
      vm::ref<iree_hal_buffer_view_t> item) {
    TensorList* new_list = new TensorList;
    IREE_ASSIGN_OR_RETURN(int32_t index,
                          ReadInt32FromScalarBufferView(index_buf.get()));
    new_list->CopyFrom(list);
    new_list->SetItem(index, vm::retain_ref(item));
    return new_list;
  }

  // tensorlist.from_tensor(%tensor, %element_shape) -> %list
  StatusOr<vm::ref<TensorList>> FromTensor(
      vm::ref<iree_hal_buffer_view_t> tensor,
      vm::ref<iree_hal_buffer_view_t> element_shape) {
    // TODO(silvasean): Emulate element shape and dtype tracking in TensorList.
    (void)element_shape;
    return TensorList::FromTensor(tensor);
  }

  // tensorlist.concat(%list) -> %list
  StatusOr<vm::ref<iree_hal_buffer_view_t>> Concat(vm::ref<TensorList> list) {
    return list->Concat();
  }

  // tensorlist.stack(%list, %element_shape, %num_elements) -> %list
  StatusOr<vm::ref<iree_hal_buffer_view_t>> Stack(
      vm::ref<TensorList> list,
      vm::ref<iree_hal_buffer_view_t> element_shape_buffer_view,
      vm::ref<iree_hal_buffer_view_t> num_elements_buffer_view) {
    // TODO(silvasean): Emulate element shape and dtype tracking in TensorList.
    (void)element_shape_buffer_view;
    IREE_ASSIGN_OR_RETURN(
        int32_t num_elements,
        ReadInt32FromScalarBufferView(num_elements_buffer_view.get()));
    if (num_elements != -1 && list->Size() != num_elements) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "num_elements arg to tesorlist.stack doesn't match the list "
                "size";
    }
    return list->Stack();
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
