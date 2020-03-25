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

#include "iree/modules/strings/strings_module.h"

#include <sstream>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "iree/base/api.h"
#include "iree/base/logging.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/module.h"
#include "iree/vm/module_abi_cc.h"
#include "iree/vm/ref.h"
#include "iree/vm/stack.h"
#include "iree/vm/types.h"

static iree_vm_ref_type_descriptor_t string_descriptor = {0};
static iree_vm_ref_type_descriptor_t string_tensor_descriptor = {0};

typedef struct string {
  iree_vm_ref_object_t ref_object;
  iree_allocator_t allocator;
  iree_string_view_t value;
} string_t;

typedef struct string_tensor {
  iree_vm_ref_object_t ref_object;
  iree_allocator_t allocator;
  iree_string_view_t* values;
  size_t count;
  const int32_t* shape;
  size_t shape_rank;
} string_tensor_t;

IREE_VM_DEFINE_TYPE_ADAPTERS(string, string_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(string_tensor, string_tensor_t);

extern "C" iree_status_t string_create(iree_string_view_t value,
                                       iree_allocator_t allocator,
                                       string_t** out_message) {
  // Note that we allocate the message and the string value together.
  string_t* message = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, sizeof(string_t) + value.size, (void**)&message));
  message->ref_object.counter = IREE_ATOMIC_VAR_INIT(1);
  message->allocator = allocator;
  message->value.data = ((const char*)message) + sizeof(string_t);
  message->value.size = value.size;
  memcpy((void*)message->value.data, value.data, message->value.size);
  *out_message = message;
  return IREE_STATUS_OK;
}

extern "C" iree_status_t string_tensor_create(
    iree_allocator_t allocator, iree_string_view_t* value, int64_t value_count,
    const int32_t* shape, size_t shape_rank, string_tensor_t** out_message) {
  // TODO(suderman): Use separate allocation for each string. More ref counters
  // but prevents constantly copying.

  // Validate the count is correct.
  size_t count = 1;
  for (int i = 0; i < shape_rank; i++) {
    count *= shape[i];
  }

  if (count != value_count) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  // Compute our total memory requirements
  size_t string_bytes = 0;
  for (int i = 0; i < value_count; i++) {
    string_bytes += value[i].size;
  }

  const size_t shape_bytes = shape_rank * sizeof(int32_t);
  const size_t string_view_bytes = value_count * sizeof(iree_string_view_t);
  const size_t byte_count =
      sizeof(string_tensor_t) + shape_bytes + string_view_bytes + string_bytes;

  // Allocate and compute byte offsets.
  string_tensor_t* message = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, byte_count, (void**)&message));

  char* shape_ptr = ((char*)message) + sizeof(string_tensor_t);
  char* string_view_ptr = shape_ptr + shape_bytes;
  char* contents_ptr = string_view_ptr + string_view_bytes;

  // Setup the string tensor structure.
  message->shape = (int32_t*)shape_ptr;
  message->values = (iree_string_view_t*)string_view_ptr;
  message->ref_object.counter = IREE_ATOMIC_VAR_INIT(1);
  message->allocator = allocator;

  // Set string tensor values.
  message->shape_rank = shape_rank;
  message->count = count;

  // Copy the shape.
  memcpy((void*)message->shape, shape, shape_rank * sizeof(int32_t));

  // Copy and allocate each string.
  for (int i = 0; i < count; i++) {
    const auto& src = value[i];
    auto& dest = message->values[i];

    dest.data = (char*)contents_ptr;
    dest.size = src.size;
    memcpy((void*)dest.data, src.data, src.size);
    contents_ptr += src.size;
  }

  *out_message = message;
  return IREE_STATUS_OK;
}

namespace iree {
namespace {

class StringsModuleState final {
 public:
  explicit StringsModuleState(iree_allocator_t allocator)
      : allocator_(allocator) {}
  ~StringsModuleState() = default;

  Status Initialize() { return OkStatus(); }

  // strings.print(%str)
  Status Print(vm::ref<string_t> str) {
    fwrite(str->value.data, 1, str->value.size, stdout);
    fputc('\n', stdout);
    fflush(stdout);
    return OkStatus();
  }

  // strings.i32_to_string(%value) -> %str
  StatusOr<vm::ref<string_t>> I32ToString(int32_t value) {
    vm::ref<string_t> new_string;
    std::string str = std::to_string(value);
    RETURN_IF_ERROR(
        FromApiStatus(string_create(iree_make_cstring_view(str.c_str()),
                                    allocator_, &new_string),
                      IREE_LOC));
    return new_string;
  }

  const iree_string_view_t* PrintTensorHelper(const iree_string_view_t* strs,
                                              const int32_t* shape,
                                              int32_t shape_rank) {
    // Handle a scalar tensor value.
    if (shape_rank == 0) {
      const auto& str = strs[0];
      fwrite(str.data, 1, str.size, stdout);
      return strs + 1;
    }

    // The row for the final tensor dimension.
    if (shape_rank == 1) {
      fputc('[', stdout);
      for (int32_t i = 0, s = shape[0]; i < s; i++) {
        const auto& str = strs[i];
        fwrite(str.data, 1, str.size, stdout);
        if (i != s - 1) {
          fwrite(", ", 1, /*size=*/2, stdout);
        }
      }

      fputc(']', stdout);
      return strs + shape[0];
    }

    // Recurse to the lower dimension with the approrpiate brackets.
    fputc('[', stdout);
    for (int32_t i = 0, s = shape[0]; i < s; i++) {
      strs = PrintTensorHelper(strs, shape + 1, shape_rank - 1);
      if (i != s - 1) {
        fwrite(",\n", 1, /*size=*/2, stdout);
      }
    }
    fputc(']', stdout);
    return strs;
  }

  // strings.print_tensor(%str_tensor)
  Status PrintTensor(vm::ref<string_tensor_t> str_tensor) {
    if (!str_tensor) {
      return OkStatus();
    }

    PrintTensorHelper(str_tensor->values, str_tensor->shape,
                      str_tensor->shape_rank);
    fputc('\n', stdout);
    fflush(stdout);
    return OkStatus();
  }

  // strings.to_string(%hal_buffer) -> %str_tensor
  StatusOr<vm::ref<string_tensor_t>> ToString(
      vm::ref<iree_hal_buffer_view_t> hal_buffer) {
    return FromApiStatus(IREE_STATUS_UNIMPLEMENTED, IREE_LOC);
  }

 private:
  // Allocator that the caller requested we use for any allocations we need to
  // perform during operation.
  iree_allocator_t allocator_ = IREE_ALLOCATOR_SYSTEM;
};

static const vm::NativeFunction<StringsModuleState> kStringsModuleFunctions[] =
    {
        vm::MakeNativeFunction("print", &StringsModuleState::Print),
        vm::MakeNativeFunction("i32_to_string",
                               &StringsModuleState::I32ToString),
        vm::MakeNativeFunction("print_tensor",
                               &StringsModuleState::PrintTensor),
        vm::MakeNativeFunction("to_string", &StringsModuleState::ToString),

};

class StringsModule final : public vm::NativeModule<StringsModuleState> {
 public:
  using vm::NativeModule<StringsModuleState>::NativeModule;

  // Example of global initialization (shared across all contexts), such as
  // loading native libraries or creating shared pools.
  Status Initialize() { return OkStatus(); }

  // Creates per-context state when the module is added to a new context.
  // May be called from any thread.
  StatusOr<std::unique_ptr<StringsModuleState>> CreateState(
      iree_allocator_t allocator) override {
    auto state = std::make_unique<StringsModuleState>(allocator);
    RETURN_IF_ERROR(state->Initialize());
    return state;
  }
};

}  // namespace
}  // namespace iree

void string_destroy(void* ptr) {
  string_t* message = (string_t*)ptr;
  iree_allocator_free(message->allocator, ptr);
}

void string_tensor_destroy(void* ptr) {
  string_t* message = (string_t*)ptr;
  iree_allocator_free(message->allocator, ptr);
}

extern "C" iree_status_t strings_module_register_types() {
  if (string_descriptor.type) {
    return IREE_STATUS_OK;  // Already registered.
  }

  // Register strings.string
  string_descriptor.type_name = iree_make_cstring_view("strings.string");
  string_descriptor.offsetof_counter = offsetof(string_t, ref_object.counter);
  string_descriptor.destroy = string_destroy;
  IREE_RETURN_IF_ERROR(iree_vm_ref_register_type(&string_descriptor));

  // Register strings.string_tensor
  string_tensor_descriptor.type_name =
      iree_make_cstring_view("strings.string_tensor");
  string_tensor_descriptor.offsetof_counter =
      offsetof(string_tensor_t, ref_object.counter);
  string_tensor_descriptor.destroy = string_tensor_destroy;
  IREE_RETURN_IF_ERROR(iree_vm_ref_register_type(&string_tensor_descriptor));

  return IREE_STATUS_OK;
}

extern "C" iree_status_t strings_module_create(iree_allocator_t allocator,
                                               iree_vm_module_t** out_module) {
  if (!out_module) return IREE_STATUS_INVALID_ARGUMENT;
  *out_module = NULL;
  auto module = std::make_unique<iree::StringsModule>(
      "strings", allocator, absl::MakeConstSpan(iree::kStringsModuleFunctions));
  auto status = module->Initialize();
  if (!status.ok()) {
    return iree::ToApiStatus(status);
  }
  *out_module = module.release()->interface();
  return IREE_STATUS_OK;
}
