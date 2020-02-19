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

#include "iree/base/api.h"
#include "iree/base/api_util.h"
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

 private:
  std::vector<vm::ref<iree_hal_buffer_view_t>> list_;
};
}  // namespace

//===----------------------------------------------------------------------===//
// `tensorlist.list` VM type registration.
//===----------------------------------------------------------------------===//

static iree_vm_ref_type_descriptor_t iree_tensorlist_descriptor = {0};

// Register our type with the vm::ref<T> static machinery.
template <>
struct ::iree::vm::ref_type_descriptor<TensorList> {
  static const iree_vm_ref_type_descriptor_t* get() {
    return &iree_tensorlist_descriptor;
  }
};

extern "C" iree_status_t iree_tensorlist_module_register_types() {
  static bool has_registered = false;
  if (has_registered) return IREE_STATUS_OK;
  IREE_VM_REGISTER_CC_TYPE(TensorList, "tensorlist.list",
                           iree_tensorlist_descriptor);
  return IREE_STATUS_OK;
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
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "expected i32 buffer view";
  }
  if (iree_hal_buffer_view_shape_rank(buffer_view) != 0) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "expected rank-0 buffer view";
  }
  iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(buffer_view);
  iree_hal_mapped_memory_t mapped_memory;
  RETURN_IF_ERROR(
      FromApiStatus(iree_hal_buffer_map(buffer, IREE_HAL_MEMORY_ACCESS_READ, 0,
                                        4, &mapped_memory),
                    IREE_LOC));
  int32_t scalar = *reinterpret_cast<int32_t*>(mapped_memory.contents.data);
  RETURN_IF_ERROR(
      FromApiStatus(iree_hal_buffer_unmap(buffer, &mapped_memory), IREE_LOC));
  return scalar;
}

namespace {
class TensorListModuleState final {
 public:
  TensorListModuleState() = default;
  ~TensorListModuleState() = default;

  // tensorlist.reserve(%element_shape, %num_elements) -> %list
  StatusOr<vm::ref<TensorList>> Reserve(
      vm::ref<iree_hal_buffer_view_t>& element_shape,
      vm::ref<iree_hal_buffer_view_t>& num_elements_buf) {
    // TODO(silvasean): Emulate element shape and dtype tracking in TensorList.
    (void)element_shape;
    TensorList* tensorlist = new TensorList;
    ASSIGN_OR_RETURN(int32_t num_elements,
                     ReadInt32FromScalarBufferView(num_elements_buf.get()));
    tensorlist->Resize(num_elements);
    return tensorlist;
  }

  // tensorlist.get_item(%list, %index, %element_shape) -> %item
  StatusOr<vm::ref<iree_hal_buffer_view_t>> GetItem(
      vm::ref<TensorList>& tensorlist,
      vm::ref<iree_hal_buffer_view_t>& index_buf,
      vm::ref<iree_hal_buffer_view_t>& element_shape) {
    // TODO(silvasean): Emulate element shape and dtype tracking in TensorList.
    (void)element_shape;
    ASSIGN_OR_RETURN(int32_t index,
                     ReadInt32FromScalarBufferView(index_buf.get()));
    return vm::retain_ref(tensorlist->GetItem(index).get());
  }

  // tensorlist.set_item(%list, %index, %item) -> %new_list
  StatusOr<vm::ref<TensorList>> SetItem(
      vm::ref<TensorList>& list, vm::ref<iree_hal_buffer_view_t>& index_buf,
      vm::ref<iree_hal_buffer_view_t>& item) {
    TensorList* new_list = new TensorList;
    ASSIGN_OR_RETURN(int32_t index,
                     ReadInt32FromScalarBufferView(index_buf.get()));
    new_list->CopyFrom(list);
    new_list->SetItem(index, vm::retain_ref(item));
    return new_list;
  }
};
}  // namespace

static const vm::NativeFunction<TensorListModuleState>
    kTensorListModuleFunctions[] = {
        vm::MakeNativeFunction("reserve", &TensorListModuleState::Reserve),
        vm::MakeNativeFunction("get_item", &TensorListModuleState::GetItem),
        vm::MakeNativeFunction("set_item", &TensorListModuleState::SetItem),
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
  if (!out_module) return IREE_STATUS_INVALID_ARGUMENT;
  *out_module = NULL;
  auto module = std::make_unique<TensorListModule>(
      "tensorlist", allocator, absl::MakeConstSpan(kTensorListModuleFunctions));
  *out_module = module.release()->interface();
  return IREE_STATUS_OK;
}

}  // namespace iree
