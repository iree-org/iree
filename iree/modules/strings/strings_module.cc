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
#include "iree/vm/bytecode_module.h"
#include "iree/vm/module.h"
#include "iree/vm/module_abi_cc.h"
#include "iree/vm/ref.h"
#include "iree/vm/stack.h"
#include "iree/vm/types.h"

static iree_vm_ref_type_descriptor_t iree_string_descriptor = {0};

typedef struct iree_string {
  iree_vm_ref_object_t ref_object;
  iree_allocator_t allocator;
  iree_string_view_t value;
} iree_string_t;

IREE_VM_DEFINE_TYPE_ADAPTERS(iree_string, iree_string_t);

iree_status_t iree_string_create(iree_string_view_t value,
                                 iree_allocator_t allocator,
                                 iree_string_t** out_message) {
  // Note that we allocate the message and the string value together.
  iree_string_t* message = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, sizeof(iree_string_t) + value.size, (void**)&message));
  message->ref_object.counter = 1;
  message->allocator = allocator;
  message->value.data = ((const char*)message) + sizeof(iree_string_t);
  message->value.size = value.size;
  memcpy((void*)message->value.data, value.data, message->value.size);
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
  Status Print(vm::ref<iree_string_t>& str) {
    fwrite(str->value.data, 1, str->value.size, stdout);
    fputc('\n', stdout);
    fflush(stdout);
    return OkStatus();
  }

  // strings.i32_to_string(%value) -> %str
  StatusOr<vm::ref<iree_string_t>> I32ToString(int32_t value) {
    vm::ref<iree_string_t> new_string;
    std::string str = std::to_string(value);
    RETURN_IF_ERROR(
        FromApiStatus(iree_string_create(iree_make_cstring_view(str.c_str()),
                                         allocator_, &new_string),
                      IREE_LOC));

    return std::move(new_string);
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

void iree_string_destroy(void* ptr) {
  iree_string_t* message = (iree_string_t*)ptr;
  iree_allocator_free(message->allocator, ptr);
}

extern "C" iree_status_t strings_module_register_types() {
  if (iree_string_descriptor.type) {
    return IREE_STATUS_OK;  // Already registered.
  }
  iree_string_descriptor.type_name = iree_make_cstring_view("strings.string");
  iree_string_descriptor.offsetof_counter =
      offsetof(iree_string_t, ref_object.counter);
  iree_string_descriptor.destroy = iree_string_destroy;
  return iree_vm_ref_register_type(&iree_string_descriptor);
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
