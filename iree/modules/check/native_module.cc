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

#include "iree/modules/check/native_module.h"

#include <cstdio>
#include <cstring>

#include "absl/strings/str_cat.h"
#include "iree/base/api.h"
#include "iree/base/api_util.h"
#include "iree/base/buffer_string_util.h"
#include "iree/base/status.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/testing/gtest.h"
#include "iree/vm/module_abi_cc.h"

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

namespace iree {
namespace {

using ::testing::Each;
using ::testing::Not;

template <typename T>
Status ExpectAllTrue(iree_byte_span_t bytes) {
  EXPECT_THAT(absl::Span<T>(reinterpret_cast<T*>(bytes.data),
                            bytes.data_length / sizeof(T)),
              Each(Not(0)));
  return OkStatus();
}

Status ExpectAllTrue(iree_byte_span_t bytes,
                     iree_hal_element_type_t element_type) {
  switch (element_type) {
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
      return ExpectAllTrue<int8_t>(bytes);
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
      return ExpectAllTrue<uint8_t>(bytes);
    case IREE_HAL_ELEMENT_TYPE_SINT_16:
      return ExpectAllTrue<int16_t>(bytes);
    case IREE_HAL_ELEMENT_TYPE_UINT_16:
      return ExpectAllTrue<uint16_t>(bytes);
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
      return ExpectAllTrue<int32_t>(bytes);
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
      return ExpectAllTrue<uint32_t>(bytes);
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
      return ExpectAllTrue<int64_t>(bytes);
    case IREE_HAL_ELEMENT_TYPE_UINT_64:
      return ExpectAllTrue<uint64_t>(bytes);
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      return ExpectAllTrue<float>(bytes);
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      return ExpectAllTrue<double>(bytes);
    case IREE_HAL_ELEMENT_TYPE_NONE:
    case IREE_HAL_ELEMENT_TYPE_OPAQUE_8:
    case IREE_HAL_ELEMENT_TYPE_OPAQUE_16:
    case IREE_HAL_ELEMENT_TYPE_OPAQUE_32:
    case IREE_HAL_ELEMENT_TYPE_OPAQUE_64:
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16: {
      // fall through
    }
  }
  char element_type_str[16];
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_format_element_type(element_type, sizeof(element_type_str),
                                   element_type_str, nullptr),
      IREE_LOC));
  return InvalidArgumentErrorBuilder(IREE_LOC)
         << "Unsupported element type " << element_type_str;
}

// Per-context module state.
// This can contain "globals" and other arbitrary state.
//
// Thread-compatible; the runtime will not issue multiple calls at the same
// time using the same state. If the implementation uses external threads then
// it must synchronize itself.
class CheckModuleState final {
 public:
  explicit CheckModuleState(iree_allocator_t allocator)
      : allocator_(allocator) {}
  ~CheckModuleState() = default;

  Status ExpectTrue(int32_t operand) {
    EXPECT_TRUE(operand) << "Expected " << operand << " to be nonzero.";
    return OkStatus();
  }

  Status ExpectFalse(int32_t operand) {
    EXPECT_FALSE(operand) << "Expected " << operand << " to be zero.";
    return OkStatus();
  }

  Status ExpectAllTrue(vm::ref<iree_hal_buffer_view_t> operand) {
    auto* view = operand.get();
    if (view == nullptr) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "View does not point to a buffer";
    }
    iree_hal_element_type_t element_type =
        iree_hal_buffer_view_element_type(view);
    iree_hal_buffer_t* buf = iree_hal_buffer_view_buffer(view);
    iree_hal_mapped_memory_t mapped_memory;
    RETURN_IF_ERROR(FromApiStatus(
        iree_hal_buffer_map(buf, IREE_HAL_MEMORY_ACCESS_READ, /*byte_offset=*/0,
                            IREE_WHOLE_BUFFER, &mapped_memory),
        IREE_LOC));
    RETURN_IF_ERROR(
        ::iree::ExpectAllTrue(mapped_memory.contents, element_type));
    iree_hal_buffer_unmap(buf, &mapped_memory);
    return OkStatus();
  }

 private:
  // Allocator that the caller requested we use for any allocations we need to
  // perform during operation.
  iree_allocator_t allocator_ = IREE_ALLOCATOR_SYSTEM;
};

// Function table mapping imported function names to their implementation.
// The signature of the target function is expected to match that in the
// check.imports.mlir file.
static const vm::NativeFunction<CheckModuleState> kCheckModuleFunctions[] = {
    vm::MakeNativeFunction("expect_true", &CheckModuleState::ExpectTrue),
    vm::MakeNativeFunction("expect_false", &CheckModuleState::ExpectFalse),
    vm::MakeNativeFunction("expect_all_true", &CheckModuleState::ExpectAllTrue),
};

// The module instance that will be allocated and reused across contexts.
// Any context-specific state must be stored in a state structure such as
// CheckModuleState below.
//
// Assumed thread-safe (by construction here, as it's immutable), though if more
// state is stored here it will need to be synchronized by the implementation.
class CheckModule final : public vm::NativeModule<CheckModuleState> {
 public:
  using vm::NativeModule<CheckModuleState>::NativeModule;

  // Creates per-context state when the module is added to a new context.
  // May be called from any thread.
  StatusOr<std::unique_ptr<CheckModuleState>> CreateState(
      iree_allocator_t allocator) override {
    auto state = std::make_unique<CheckModuleState>(allocator);
    return state;
  }
};

}  // namespace

// Note that while we are using C++ bindings internally we still expose the
// module as a C instance. This hides the details of our implementation.
extern "C" iree_status_t check_native_module_create(
    iree_allocator_t allocator, iree_vm_module_t** out_module) {
  if (!out_module) return IREE_STATUS_INVALID_ARGUMENT;
  *out_module = NULL;
  auto module = std::make_unique<CheckModule>(
      "check", allocator, absl::MakeConstSpan(kCheckModuleFunctions));
  *out_module = module.release()->interface();
  return IREE_STATUS_OK;
}

}  // namespace iree
