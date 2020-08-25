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

#include <math.h>

#include <cstdio>
#include <cstring>
#include <sstream>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "iree/base/api.h"
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
absl::Span<const T> AbslSpan(iree_byte_span_t bytes) {
  return absl::Span<T>(reinterpret_cast<T*>(bytes.data),
                       bytes.data_length / sizeof(T));
}

StatusOr<std::string> BufferViewToString(iree_hal_buffer_view_t* buffer_view) {
  std::string result_str(4096, '\0');
  iree_status_t status;
  do {
    iree_host_size_t actual_length = 0;
    status = iree_hal_buffer_view_format(
        buffer_view, /*max_element_count=*/1024, result_str.size() + 1,
        &result_str[0], &actual_length);
    result_str.resize(actual_length);
  } while (iree_status_is_out_of_range(status));
  IREE_RETURN_IF_ERROR(std::move(status));
  return std::move(result_str);
}

template <typename T>
Status ExpectAllTrue(iree_byte_span_t bytes) {
  EXPECT_THAT(AbslSpan<T>(bytes), Each(Not(0)));
  return OkStatus();
}

// TODO(b/146898896): Put this somewhere common. Operator overload?
bool EqByteSpan(iree_byte_span_t lhs_bytes, iree_byte_span_t rhs_bytes) {
  return AbslSpan<uint8_t>(lhs_bytes) == AbslSpan<uint8_t>(rhs_bytes);
}

template <typename T>
bool AlmostEqByteSpan(iree_byte_span_t lhs_bytes, iree_byte_span_t rhs_bytes) {
  auto lhs_span = AbslSpan<T>(lhs_bytes);
  auto rhs_span = AbslSpan<T>(rhs_bytes);
  assert(lhs_span.size() == rhs_span.size());
  for (int i = 0; i < lhs_span.size(); ++i) {
    if (fabs(lhs_span[i] - rhs_span[i]) > 0.0001) {
      return false;
    }
  }
  return true;
}

StatusOr<bool> AlmostEqByteSpan(iree_byte_span_t lhs_bytes,
                                iree_byte_span_t rhs_bytes,
                                iree_hal_element_type_t element_type) {
  switch (element_type) {
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      return AlmostEqByteSpan<float>(lhs_bytes, rhs_bytes);
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      return AlmostEqByteSpan<double>(lhs_bytes, rhs_bytes);
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
    case IREE_HAL_ELEMENT_TYPE_SINT_16:
    case IREE_HAL_ELEMENT_TYPE_UINT_16:
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
    case IREE_HAL_ELEMENT_TYPE_UINT_64:
      // TODO(gcmn): Consider supporting fuzzy matching for quantized integers.
    case IREE_HAL_ELEMENT_TYPE_OPAQUE_8:
    case IREE_HAL_ELEMENT_TYPE_OPAQUE_16:
    case IREE_HAL_ELEMENT_TYPE_OPAQUE_32:
    case IREE_HAL_ELEMENT_TYPE_OPAQUE_64:
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
    case IREE_HAL_ELEMENT_TYPE_NONE: {
      break;
    }
  }
  char element_type_str[16];
  IREE_RETURN_IF_ERROR(iree_hal_format_element_type(
      element_type, sizeof(element_type_str), element_type_str, nullptr));
  return InvalidArgumentErrorBuilder(IREE_LOC)
         << "Unsupported element type " << element_type_str;
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
      break;
    }
  }
  char element_type_str[16];
  IREE_RETURN_IF_ERROR(iree_hal_format_element_type(
      element_type, sizeof(element_type_str), element_type_str, nullptr));
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
    iree_hal_element_type_t element_type =
        iree_hal_buffer_view_element_type(view);
    iree_hal_buffer_t* buf = iree_hal_buffer_view_buffer(view);
    iree_hal_mapped_memory_t mapped_memory;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map(
        buf, IREE_HAL_MEMORY_ACCESS_READ,
        /*byte_offset=*/0, IREE_WHOLE_BUFFER, &mapped_memory));
    IREE_RETURN_IF_ERROR(
        ::iree::ExpectAllTrue(mapped_memory.contents, element_type));
    iree_hal_buffer_unmap(buf, &mapped_memory);
    return OkStatus();
  }

  Status ExpectEq(vm::ref<iree_hal_buffer_view_t> lhs_ref,
                  vm::ref<iree_hal_buffer_view_t> rhs_ref) {
    auto* lhs = lhs_ref.get();
    auto* rhs = rhs_ref.get();
    size_t lhs_rank = iree_hal_buffer_view_shape_rank(lhs);
    absl::InlinedVector<int32_t, 6> lhs_shape(lhs_rank);
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_shape(lhs, lhs_rank, lhs_shape.data(), nullptr));

    size_t rhs_rank = iree_hal_buffer_view_shape_rank(rhs);
    absl::InlinedVector<int32_t, 6> rhs_shape(rhs_rank);
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_shape(rhs, rhs_rank, rhs_shape.data(), nullptr));

    iree_hal_element_type_t lhs_element_type =
        iree_hal_buffer_view_element_type(lhs);
    iree_hal_element_type_t rhs_element_type =
        iree_hal_buffer_view_element_type(rhs);

    iree_hal_buffer_t* lhs_buf = iree_hal_buffer_view_buffer(lhs);
    iree_hal_mapped_memory_t lhs_mapped_memory;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map(
        lhs_buf, IREE_HAL_MEMORY_ACCESS_READ,
        /*byte_offset=*/0, IREE_WHOLE_BUFFER, &lhs_mapped_memory));
    iree_hal_buffer_t* rhs_buf = iree_hal_buffer_view_buffer(rhs);
    iree_hal_mapped_memory_t rhs_mapped_memory;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map(
        rhs_buf, IREE_HAL_MEMORY_ACCESS_READ,
        /*byte_offset=*/0, IREE_WHOLE_BUFFER, &rhs_mapped_memory));

    bool element_types_eq = lhs_element_type == rhs_element_type;
    bool shape_eq = lhs_shape == rhs_shape;
    bool contents_eq =
        EqByteSpan(lhs_mapped_memory.contents, rhs_mapped_memory.contents);
    iree_hal_buffer_unmap(lhs_buf, &lhs_mapped_memory);
    iree_hal_buffer_unmap(rhs_buf, &rhs_mapped_memory);

    if (!element_types_eq || !shape_eq || !contents_eq) {
      std::ostringstream os;
      os << "Expected equality of these values.";
      if (!element_types_eq) {
        os << " Element types do not match.";
      }
      if (!shape_eq) {
        os << " Shapes do not match.";
      }
      if (!contents_eq) {
        os << " Contents does not match.";
      }
      // TODO(b/146898896): Propagate original variable names.
      os << "\n"
            "  lhs:\n"
            "    ";
      IREE_ASSIGN_OR_RETURN(auto lhs_str, BufferViewToString(lhs));
      os << lhs_str;

      os << "\n"
            "  rhs:\n"
            "    ";
      IREE_ASSIGN_OR_RETURN(auto rhs_str, BufferViewToString(rhs));
      os << rhs_str;

      // TODO(b/146898896): Use ADD_FAILURE_AT to propagate source location.
      ADD_FAILURE() << os.str();
    }

    return OkStatus();
  }

  Status ExpectAlmostEq(vm::ref<iree_hal_buffer_view_t> lhs_ref,
                        vm::ref<iree_hal_buffer_view_t> rhs_ref) {
    auto* lhs = lhs_ref.get();
    auto* rhs = rhs_ref.get();
    size_t lhs_rank = iree_hal_buffer_view_shape_rank(lhs);
    absl::InlinedVector<int32_t, 6> lhs_shape(lhs_rank);
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_shape(lhs, lhs_rank, lhs_shape.data(), nullptr));

    size_t rhs_rank = iree_hal_buffer_view_shape_rank(rhs);
    absl::InlinedVector<int32_t, 6> rhs_shape(rhs_rank);
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_shape(rhs, rhs_rank, rhs_shape.data(), nullptr));

    iree_hal_element_type_t lhs_element_type =
        iree_hal_buffer_view_element_type(lhs);
    iree_hal_element_type_t rhs_element_type =
        iree_hal_buffer_view_element_type(rhs);

    iree_hal_buffer_t* lhs_buf = iree_hal_buffer_view_buffer(lhs);
    iree_hal_mapped_memory_t lhs_mapped_memory;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map(
        lhs_buf, IREE_HAL_MEMORY_ACCESS_READ,
        /*byte_offset=*/0, IREE_WHOLE_BUFFER, &lhs_mapped_memory));
    iree_hal_buffer_t* rhs_buf = iree_hal_buffer_view_buffer(rhs);
    iree_hal_mapped_memory_t rhs_mapped_memory;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map(
        rhs_buf, IREE_HAL_MEMORY_ACCESS_READ,
        /*byte_offset=*/0, IREE_WHOLE_BUFFER, &rhs_mapped_memory));

    bool element_types_eq = lhs_element_type == rhs_element_type;
    bool shape_eq = lhs_shape == rhs_shape;
    // Only check contents if shape and element type match. Otherwise we can't.
    bool contents_could_be_almost_eq = true;
    if (element_types_eq && shape_eq) {
      IREE_ASSIGN_OR_RETURN(
          contents_could_be_almost_eq,
          AlmostEqByteSpan(lhs_mapped_memory.contents,
                           rhs_mapped_memory.contents, lhs_element_type));
    }
    iree_hal_buffer_unmap(lhs_buf, &lhs_mapped_memory);
    iree_hal_buffer_unmap(rhs_buf, &rhs_mapped_memory);

    if (!element_types_eq || !shape_eq || !contents_could_be_almost_eq) {
      std::ostringstream os;
      os << "Expected near equality of these values.";
      if (!element_types_eq) {
        os << " Element types do not match.";
      }
      if (!shape_eq) {
        os << " Shapes do not match.";
      }
      if (!contents_could_be_almost_eq) {
        os << " Contents does not match.";
      }
      // TODO(b/146898896): Propagate original variable names.
      os << "\n"
            "  lhs:\n"
            "    ";
      IREE_ASSIGN_OR_RETURN(auto lhs_str, BufferViewToString(lhs));
      os << lhs_str;

      os << "\n"
            "  rhs:\n"
            "    ";
      IREE_ASSIGN_OR_RETURN(auto rhs_str, BufferViewToString(rhs));
      os << rhs_str;

      // TODO(b/146898896): Use ADD_FAILURE_AT to propagate source location.
      ADD_FAILURE() << os.str();
    }

    return OkStatus();
  }

 private:
  // Allocator that the caller requested we use for any allocations we need to
  // perform during operation.
  iree_allocator_t allocator_ = iree_allocator_system();
};

// Function table mapping imported function names to their implementation.
// The signature of the target function is expected to match that in the
// check.imports.mlir file.
static const vm::NativeFunction<CheckModuleState> kCheckModuleFunctions[] = {
    vm::MakeNativeFunction("expect_true", &CheckModuleState::ExpectTrue),
    vm::MakeNativeFunction("expect_false", &CheckModuleState::ExpectFalse),
    vm::MakeNativeFunction("expect_all_true", &CheckModuleState::ExpectAllTrue),
    vm::MakeNativeFunction("expect_eq", &CheckModuleState::ExpectEq),
    vm::MakeNativeFunction("expect_almost_eq",
                           &CheckModuleState::ExpectAlmostEq),
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
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  auto module = std::make_unique<CheckModule>(
      "check", allocator, absl::MakeConstSpan(kCheckModuleFunctions));
  *out_module = module.release()->interface();
  return iree_ok_status();
}

}  // namespace iree
