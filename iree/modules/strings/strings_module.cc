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

#include <cstdint>
#include <sstream>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "iree/base/api.h"
#include "iree/base/logging.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/modules/strings/api.h"
#include "iree/modules/strings/api_detail.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/module_abi_cc.h"

static iree_vm_ref_type_descriptor_t strings_string_descriptor = {0};
static iree_vm_ref_type_descriptor_t strings_string_tensor_descriptor = {0};

IREE_VM_DEFINE_TYPE_ADAPTERS(strings_string, strings_string_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(strings_string_tensor, strings_string_tensor_t);

namespace iree {
namespace {
class StringsModuleState final {
 public:
  explicit StringsModuleState(iree_allocator_t allocator)
      : allocator_(allocator) {}
  ~StringsModuleState() = default;

  Status Initialize() { return OkStatus(); }

  // strings.print(%str)
  Status Print(vm::ref<strings_string_t> str) {
    fwrite(str->value.data, 1, str->value.size, stdout);
    fputc('\n', stdout);
    fflush(stdout);
    return OkStatus();
  }

  // strings.i32_to_string(%value) -> %str
  StatusOr<vm::ref<strings_string_t>> I32ToString(int32_t value) {
    vm::ref<strings_string_t> new_string;
    std::string str = std::to_string(value);
    IREE_RETURN_IF_ERROR(strings_string_create(
        iree_make_cstring_view(str.c_str()), allocator_, &new_string));
    return new_string;
  }

  const iree_string_view_t* StringTensorToStringHelper(
      const iree_string_view_t* strs, const int32_t* shape, int32_t rank,
      std::string* output) {
    // Handle a scalar tensor value.
    if (rank == 0) {
      const auto& str = strs[0];
      output->append(str.data, str.size);
      return strs + 1;
    }

    // The row for the final tensor dimension.
    if (rank == 1) {
      output->append("[", 1);
      for (int32_t i = 0, s = shape[0]; i < s; i++) {
        const auto& str = strs[i];
        output->append(str.data, str.size);
        if (i != s - 1) {
          output->append(", ", 2);
        }
      }

      output->append("]", 1);
      return strs + shape[0];
    }

    // Recurse to the lower dimension with the approrpiate brackets.
    output->append("[", 1);
    for (int32_t i = 0, s = shape[0]; i < s; i++) {
      strs = StringTensorToStringHelper(strs, shape + 1, rank - 1, output);
      if (i != s - 1) {
        output->append(",\n", 2);
      }
    }
    output->append("]", 1);
    return strs;
  }

  // strings.print_tensor(%str_tensor)
  StatusOr<vm::ref<strings_string_t>> StringTensorToString(
      vm::ref<strings_string_tensor_t> str_tensor) {
    // Perform a rough estimation of the amount of space we need.
    size_t string_length = 0;
    for (int i = 0; i < str_tensor->count; i++) {
      string_length += str_tensor->values[i].size + 2;
    }

    vm::ref<strings_string_t> new_string;
    std::string str;
    str.reserve(string_length);
    StringTensorToStringHelper(str_tensor->values, str_tensor->shape,
                               str_tensor->rank, &str);

    IREE_RETURN_IF_ERROR(strings_string_create(
        iree_make_cstring_view(str.c_str()), allocator_, &new_string));
    return new_string;
  }

  // strings.to_string_tensor(%hal_buffer) -> %str_tensor
  StatusOr<vm::ref<strings_string_tensor_t>> ToStringTensor(
      vm::ref<iree_hal_buffer_view_t> hal_buffer_view) {
    const size_t rank = iree_hal_buffer_view_shape_rank(hal_buffer_view.get());
    absl::InlinedVector<int32_t, 6> shape(rank);
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_shape(hal_buffer_view.get(), rank,
                                                    shape.data(), nullptr));

    size_t num_elements = 1;
    for (auto val : shape) {
      num_elements *= val;
    }

    // Pull the values down.
    size_t element_size =
        iree_hal_buffer_view_element_size(hal_buffer_view.get());
    size_t tensor_size = element_size * num_elements;
    iree_hal_buffer_t* hal_buffer =
        iree_hal_buffer_view_buffer(hal_buffer_view.get());
    iree_hal_mapped_memory_t tensor_mapping;
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_map(hal_buffer, IREE_HAL_MEMORY_ACCESS_READ,
                            /*byte_offset=*/0, tensor_size, &tensor_mapping));

    iree_hal_element_type_t type =
        iree_hal_buffer_view_element_type(hal_buffer_view.get());

    std::vector<std::string> strings;
    strings.reserve(num_elements);

    switch (type) {
      case IREE_HAL_ELEMENT_TYPE_SINT_8:
        GenerateStringsByType<int8_t>(tensor_mapping, strings);
        break;

      case IREE_HAL_ELEMENT_TYPE_UINT_8:
        GenerateStringsByType<uint8_t>(tensor_mapping, strings);
        break;

      case IREE_HAL_ELEMENT_TYPE_SINT_16:
        GenerateStringsByType<int16_t>(tensor_mapping, strings);
        break;

      case IREE_HAL_ELEMENT_TYPE_UINT_16:
        GenerateStringsByType<uint16_t>(tensor_mapping, strings);
        break;

      case IREE_HAL_ELEMENT_TYPE_SINT_32:
        GenerateStringsByType<int32_t>(tensor_mapping, strings);
        break;

      case IREE_HAL_ELEMENT_TYPE_UINT_32:
        GenerateStringsByType<uint32_t>(tensor_mapping, strings);
        break;

      case IREE_HAL_ELEMENT_TYPE_SINT_64:
        GenerateStringsByType<int64_t>(tensor_mapping, strings);
        break;

      case IREE_HAL_ELEMENT_TYPE_UINT_64:
        GenerateStringsByType<uint64_t>(tensor_mapping, strings);
        break;

      case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
        GenerateStringsByType<float>(tensor_mapping, strings);
        break;

      case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
        GenerateStringsByType<double>(tensor_mapping, strings);
        break;

      default:
        return UnimplementedErrorBuilder(IREE_LOC);
    }

    // Unmap used buffer.
    IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap(hal_buffer, &tensor_mapping));

    // Place into iree_string_views.
    std::vector<iree_string_view_t> string_views;
    string_views.reserve(num_elements);

    for (const auto& str : strings) {
      string_views.push_back(iree_make_cstring_view(str.data()));
    }

    strings_string_tensor_t* string_tensor;
    IREE_RETURN_IF_ERROR(strings_string_tensor_create(
        allocator_, string_views.data(), string_views.size(), shape.data(),
        rank, &string_tensor));

    return string_tensor;
  }

  // strings.gather(%str_tensor, %hal_buffer) -> %str_tensor
  StatusOr<vm::ref<strings_string_tensor_t>> Gather(
      vm::ref<strings_string_tensor_t> dict,
      vm::ref<iree_hal_buffer_view_t> ids) {
    // The dict must be a simple list, and the indices must be integers.
    if (dict->rank != 1 || iree_hal_buffer_view_element_type(ids.get()) !=
                               IREE_HAL_ELEMENT_TYPE_SINT_32) {
      return InvalidArgumentErrorBuilder(IREE_LOC);
    }

    const size_t rank = iree_hal_buffer_view_shape_rank(ids.get());
    absl::InlinedVector<int32_t, 6> shape(rank);
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_shape(ids.get(), rank, shape.data(), nullptr));

    size_t num_elements = 1;
    for (auto val : shape) {
      num_elements *= val;
    }

    // Pull the values down.
    size_t element_size = iree_hal_buffer_view_element_size(ids.get());
    size_t tensor_size = element_size * num_elements;
    iree_hal_buffer_t* hal_buffer = iree_hal_buffer_view_buffer(ids.get());
    iree_hal_mapped_memory_t tensor_mapping;
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_map(hal_buffer, IREE_HAL_MEMORY_ACCESS_READ,
                            /*byte_offset=*/0, tensor_size, &tensor_mapping));
    iree_string_view_t str;
    const auto& contents = tensor_mapping.contents;
    std::vector<iree_string_view_t> string_views;
    string_views.reserve(num_elements);

    for (int32_t *p = (int32_t*)contents.data,
                 *s = (int32_t*)(contents.data + contents.data_length);
         p < s; p++) {
      IREE_RETURN_IF_ERROR(
          strings_string_tensor_get_element(dict.get(), p, 1, &str));
      string_views.push_back(str);
    }

    // Unmap used buffer.
    IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap(hal_buffer, &tensor_mapping));

    strings_string_tensor_t* string_tensor;
    IREE_RETURN_IF_ERROR(strings_string_tensor_create(
        allocator_, string_views.data(), string_views.size(), shape.data(),
        rank, &string_tensor));
    return string_tensor;
  }

  // strings.concat(%str_tensor) -> %str_tensor
  StatusOr<vm::ref<strings_string_tensor_t>> Concat(
      vm::ref<strings_string_tensor_t> str_tensor) {
    size_t new_rank = str_tensor->rank - 1;
    const int32_t* shape = str_tensor->shape;
    int32_t last_dim = shape[new_rank];

    int32_t rank_mul = 1;
    for (int32_t i = 0; i < new_rank; i++) {
      rank_mul *= shape[i];
    }

    // Place into iree_string_views.
    std::vector<iree_string_view_t> string_views;
    string_views.reserve(rank_mul);
    std::vector<std::string> strings;
    strings.reserve(rank_mul);

    for (int32_t i = 0; i < rank_mul; i++) {
      std::string str;
      for (int32_t j = 0; j < last_dim; j++) {
        int32_t curr_pos = i * last_dim + j;
        iree_string_view_t curr_str = str_tensor->values[curr_pos];
        str.append(curr_str.data, curr_str.size);
      }
      strings.push_back(str);
    }

    for (int i = 0; i < strings.size(); i++) {
      string_views.push_back(iree_make_cstring_view(strings[i].data()));
    }

    strings_string_tensor_t* string_tensor;
    IREE_RETURN_IF_ERROR(strings_string_tensor_create(
        allocator_, string_views.data(), string_views.size(), shape, new_rank,
        &string_tensor));
    return string_tensor;
  }

 private:
  // Allocator that the caller requested we use for any allocations we need to
  // perform during operation.
  iree_allocator_t allocator_ = iree_allocator_system();

  template <typename T>
  void GenerateStringsByType(iree_hal_mapped_memory_t tensor_mapping,
                             std::vector<std::string>& strings) {
    const auto& contents = tensor_mapping.contents;
    for (const T *p = (const T*)contents.data,
                 *s = (const T*)(contents.data + contents.data_length);
         p < s; p++) {
      std::string str = std::to_string(*p);
      strings.push_back(std::move(str));
    }
  }
};

static const vm::NativeFunction<StringsModuleState> kStringsModuleFunctions[] =
    {
        vm::MakeNativeFunction("print", &StringsModuleState::Print),
        vm::MakeNativeFunction("i32_to_string",
                               &StringsModuleState::I32ToString),
        vm::MakeNativeFunction("string_tensor_to_string",
                               &StringsModuleState::StringTensorToString),
        vm::MakeNativeFunction("to_string_tensor",
                               &StringsModuleState::ToStringTensor),
        vm::MakeNativeFunction("gather", &StringsModuleState::Gather),
        vm::MakeNativeFunction("concat", &StringsModuleState::Concat),
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
    IREE_RETURN_IF_ERROR(state->Initialize());
    return state;
  }
};

}  // namespace
}  // namespace iree

extern "C" iree_status_t iree_strings_module_register_types() {
  if (strings_string_descriptor.type) {
    return iree_ok_status();  // Already registered.
  }

  // Register strings.string
  strings_string_descriptor.type_name =
      iree_make_cstring_view("strings.string");
  strings_string_descriptor.offsetof_counter =
      offsetof(strings_string_t, ref_object.counter);
  strings_string_descriptor.destroy = strings_string_destroy;
  IREE_RETURN_IF_ERROR(iree_vm_ref_register_type(&strings_string_descriptor));

  // Register strings.string_tensor
  strings_string_tensor_descriptor.type_name =
      iree_make_cstring_view("strings.string_tensor");
  strings_string_tensor_descriptor.offsetof_counter =
      offsetof(strings_string_tensor_t, ref_object.counter);
  strings_string_tensor_descriptor.destroy = strings_string_tensor_destroy;
  IREE_RETURN_IF_ERROR(
      iree_vm_ref_register_type(&strings_string_tensor_descriptor));

  return iree_ok_status();
}

extern "C" iree_status_t iree_strings_module_create(
    iree_allocator_t allocator, iree_vm_module_t** out_module) {
  if (!out_module) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  *out_module = NULL;
  auto module = std::make_unique<iree::StringsModule>(
      "strings", allocator, absl::MakeConstSpan(iree::kStringsModuleFunctions));
  IREE_RETURN_IF_ERROR(module->Initialize());
  *out_module = module.release()->interface();
  return iree_ok_status();
}
