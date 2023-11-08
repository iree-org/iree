// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/check/module.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/testing/gtest.h"
#include "iree/vm/api.h"
#include "iree/vm/native_module_cc.h"

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

namespace iree {
namespace {

using ::testing::Each;
using ::testing::Not;

template <typename T>
iree::span<const T> ToSpan(iree_byte_span_t bytes) {
  return iree::span<const T>(reinterpret_cast<T*>(bytes.data),
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
  EXPECT_THAT(ToSpan<T>(bytes), Each(Not(T(0))));
  return OkStatus();
}

bool EqByteSpan(iree_byte_span_t lhs_bytes, iree_byte_span_t rhs_bytes) {
  return lhs_bytes.data_length == rhs_bytes.data_length &&
         memcmp(lhs_bytes.data, rhs_bytes.data, lhs_bytes.data_length) == 0;
}

static constexpr float kF32PrecisionThreshold = 0.0001f;

template <typename T>
bool AlmostEqByteSpan(iree_byte_span_t lhs_bytes, iree_byte_span_t rhs_bytes) {
  auto lhs_span = ToSpan<T>(lhs_bytes);
  auto rhs_span = ToSpan<T>(rhs_bytes);
  assert(lhs_span.size() == rhs_span.size());
  for (int i = 0; i < lhs_span.size(); ++i) {
    if (fabs(lhs_span[i] - rhs_span[i]) > kF32PrecisionThreshold) {
      return false;
    }
  }
  return true;
}

static constexpr float kF16PrecisionThreshold = 0.001f;

bool AlmostEqByteSpanF16(iree_byte_span_t lhs_bytes,
                         iree_byte_span_t rhs_bytes) {
  auto lhs_span = ToSpan<uint16_t>(lhs_bytes);
  auto rhs_span = ToSpan<uint16_t>(rhs_bytes);
  assert(lhs_span.size() == rhs_span.size());
  for (int i = 0; i < lhs_span.size(); ++i) {
    if (fabs(iree_math_f16_to_f32(lhs_span[i]) -
             iree_math_f16_to_f32(rhs_span[i])) > kF16PrecisionThreshold) {
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
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      return AlmostEqByteSpanF16(lhs_bytes, rhs_bytes);
    default:
      // TODO(gcmn): Consider supporting fuzzy matching for quantized integers.
      break;
  }
  char element_type_str[16];
  IREE_RETURN_IF_ERROR(iree_hal_format_element_type(
      element_type, sizeof(element_type_str), element_type_str, nullptr));
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported element type %s", element_type_str);
}

Status ExpectAllTrue(iree_byte_span_t bytes,
                     iree_hal_element_type_t element_type) {
  switch (element_type) {
    case IREE_HAL_ELEMENT_TYPE_INT_8:
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
      return ExpectAllTrue<int8_t>(bytes);
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
      return ExpectAllTrue<uint8_t>(bytes);
    case IREE_HAL_ELEMENT_TYPE_INT_16:
    case IREE_HAL_ELEMENT_TYPE_SINT_16:
      return ExpectAllTrue<int16_t>(bytes);
    case IREE_HAL_ELEMENT_TYPE_UINT_16:
      return ExpectAllTrue<uint16_t>(bytes);
    case IREE_HAL_ELEMENT_TYPE_INT_32:
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
      return ExpectAllTrue<int32_t>(bytes);
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
      return ExpectAllTrue<uint32_t>(bytes);
    case IREE_HAL_ELEMENT_TYPE_INT_64:
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
      return ExpectAllTrue<int64_t>(bytes);
    case IREE_HAL_ELEMENT_TYPE_UINT_64:
      return ExpectAllTrue<uint64_t>(bytes);
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      return ExpectAllTrue<float>(bytes);
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      return ExpectAllTrue<double>(bytes);
    default:
      break;
  }
  char element_type_str[16];
  IREE_RETURN_IF_ERROR(iree_hal_format_element_type(
      element_type, sizeof(element_type_str), element_type_str, nullptr));
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported element type %s", element_type_str);
}

static StatusOr<std::vector<vm::ref<iree_hal_buffer_view_t>>>
TransferBuffersToHost(
    iree_hal_device_t* device,
    const iree::span<const vm::ref<iree_hal_buffer_view_t>> source_views) {
  IREE_TRACE_SCOPE();

  // If all buffers are already host-accessible we can skip the transfer.
  std::vector<vm::ref<iree_hal_buffer_view_t>> target_views;
  bool requires_transfer = false;
  for (auto& source_view : source_views) {
    iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(source_view.get());
    if (!iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                           IREE_HAL_MEMORY_TYPE_HOST_VISIBLE) ||
        !iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer),
                           IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED)) {
      requires_transfer = true;
    }
  }
  if (!requires_transfer) {
    for (auto& source_view : source_views) target_views.push_back(source_view);
    return std::move(target_views);
  }

  vm::ref<iree_hal_command_buffer_t> command_buffer;
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
      device,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      &command_buffer));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer.get()));

  iree_hal_buffer_params_t target_params = {
      /*.usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
      /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
      /*.type=*/
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
      /*.min_alignment=*/0,
  };
  for (size_t i = 0; i < source_views.size(); ++i) {
    iree_hal_buffer_t* source_buffer =
        iree_hal_buffer_view_buffer(source_views[i].get());
    iree_device_size_t buffer_length =
        iree_hal_buffer_byte_length(source_buffer);
    vm::ref<iree_hal_buffer_t> target_buffer;
    IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), target_params, buffer_length,
        &target_buffer));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_copy_buffer(
        command_buffer.get(), source_buffer, 0, target_buffer.get(), 0,
        buffer_length));
    vm::ref<iree_hal_buffer_view_t> target_view;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create_like(
        target_buffer.get(), source_views[i].get(),
        iree_hal_device_host_allocator(device), &target_view));
    target_views.push_back(std::move(target_view));
  }

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_end(command_buffer.get()));
  vm::ref<iree_hal_semaphore_t> semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(device, 0ull, &semaphore));
  vm::ref<iree_hal_fence_t> fence;
  IREE_RETURN_IF_ERROR(iree_hal_fence_create_at(
      semaphore.get(), 1ull, iree_hal_device_host_allocator(device), &fence));
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_execute(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      iree_hal_fence_semaphore_list(fence.get()), 1, &command_buffer));
  IREE_RETURN_IF_ERROR(
      iree_hal_fence_wait(fence.get(), iree_infinite_timeout()));
  return std::move(target_views);
}

static Status TransferToHost(iree_hal_device_t* device,
                             vm::ref<iree_hal_buffer_view_t>& buffer_view) {
  IREE_TRACE_SCOPE();
  iree::span<const vm::ref<iree_hal_buffer_view_t>>
      source_views({buffer_view});
  IREE_ASSIGN_OR_RETURN(auto target_views,
                        TransferBuffersToHost(device, source_views));
  buffer_view = std::move(target_views[0]);
  return OkStatus();
}

static Status TransferToHost(iree_hal_device_t* device,
                             vm::ref<iree_hal_buffer_view_t>& buffer_view_a,
                             vm::ref<iree_hal_buffer_view_t>& buffer_view_b) {
  IREE_TRACE_SCOPE();
  iree::span<const vm::ref<iree_hal_buffer_view_t>>
    source_views({buffer_view_a, buffer_view_b});
  IREE_ASSIGN_OR_RETURN(
      auto target_views,
      TransferBuffersToHost(device, source_views));
  buffer_view_a = std::move(target_views[0]);
  buffer_view_b = std::move(target_views[1]);
  return OkStatus();
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

  Status ExpectAllTrue(vm::ref<iree_hal_device_t> device,
                       vm::ref<iree_hal_buffer_view_t> operand) {
    IREE_RETURN_IF_ERROR(TransferToHost(device.get(), operand));
    auto* view = operand.get();
    iree_hal_element_type_t element_type =
        iree_hal_buffer_view_element_type(view);
    iree_hal_buffer_t* buf = iree_hal_buffer_view_buffer(view);
    iree_device_size_t size = iree_hal_buffer_view_byte_length(view);
    iree_hal_buffer_mapping_t mapped_memory = {{0}};
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
        buf, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
        /*byte_offset=*/0, size, &mapped_memory));
    IREE_RETURN_IF_ERROR(
        ::iree::ExpectAllTrue(mapped_memory.contents, element_type));
    iree_status_ignore(iree_hal_buffer_unmap_range(&mapped_memory));
    return OkStatus();
  }

  Status ExpectEq(vm::ref<iree_hal_device_t> device,
                  vm::ref<iree_hal_buffer_view_t> lhs_ref,
                  vm::ref<iree_hal_buffer_view_t> rhs_ref) {
    IREE_RETURN_IF_ERROR(TransferToHost(device.get(), lhs_ref, rhs_ref));
    auto* lhs = lhs_ref.get();
    auto* rhs = rhs_ref.get();

    iree_device_size_t lhs_size = iree_hal_buffer_view_byte_length(lhs);
    size_t lhs_rank = iree_hal_buffer_view_shape_rank(lhs);
    std::vector<iree_hal_dim_t> lhs_shape(lhs_rank);
    if (lhs_rank > 0) {
      IREE_RETURN_IF_ERROR(
          iree_hal_buffer_view_shape(lhs, lhs_rank, lhs_shape.data(), nullptr));
    }

    iree_device_size_t rhs_size = iree_hal_buffer_view_byte_length(rhs);
    size_t rhs_rank = iree_hal_buffer_view_shape_rank(rhs);
    std::vector<iree_hal_dim_t> rhs_shape(rhs_rank);
    if (rhs_rank > 0) {
      IREE_RETURN_IF_ERROR(
          iree_hal_buffer_view_shape(rhs, rhs_rank, rhs_shape.data(), nullptr));
    }

    iree_hal_element_type_t lhs_element_type =
        iree_hal_buffer_view_element_type(lhs);
    iree_hal_element_type_t rhs_element_type =
        iree_hal_buffer_view_element_type(rhs);

    // HACK: this is all broken and will leak. Let's kill this entire module
    // please.

    iree_hal_buffer_t* lhs_buf = iree_hal_buffer_view_buffer(lhs);
    iree_hal_buffer_mapping_t lhs_mapped_memory = {{0}};
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
        lhs_buf, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
        /*byte_offset=*/0, lhs_size, &lhs_mapped_memory));
    iree_hal_buffer_t* rhs_buf = iree_hal_buffer_view_buffer(rhs);
    iree_hal_buffer_mapping_t rhs_mapped_memory = {{0}};
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
        rhs_buf, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
        /*byte_offset=*/0, rhs_size, &rhs_mapped_memory));

    bool element_types_eq = lhs_element_type == rhs_element_type;
    bool shape_eq = lhs_shape == rhs_shape;
    bool contents_eq =
        EqByteSpan(lhs_mapped_memory.contents, rhs_mapped_memory.contents);
    iree_status_ignore(iree_hal_buffer_unmap_range(&lhs_mapped_memory));
    iree_status_ignore(iree_hal_buffer_unmap_range(&rhs_mapped_memory));

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
      // TODO(gcmn): Propagate original variable names.
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

      // TODO(gcmn): Use ADD_FAILURE_AT to propagate source location.
      ADD_FAILURE() << os.str();
    }

    return OkStatus();
  }

  Status ExpectAlmostEq(vm::ref<iree_hal_device_t> device,
                        vm::ref<iree_hal_buffer_view_t> lhs_ref,
                        vm::ref<iree_hal_buffer_view_t> rhs_ref) {
    IREE_RETURN_IF_ERROR(TransferToHost(device.get(), lhs_ref, rhs_ref));
    auto* lhs = lhs_ref.get();
    auto* rhs = rhs_ref.get();

    iree_device_size_t lhs_size = iree_hal_buffer_view_byte_length(lhs);
    size_t lhs_rank = iree_hal_buffer_view_shape_rank(lhs);
    std::vector<iree_hal_dim_t> lhs_shape(lhs_rank);
    if (lhs_rank > 0) {
      IREE_RETURN_IF_ERROR(
          iree_hal_buffer_view_shape(lhs, lhs_rank, lhs_shape.data(), nullptr));
    }

    iree_device_size_t rhs_size = iree_hal_buffer_view_byte_length(rhs);
    size_t rhs_rank = iree_hal_buffer_view_shape_rank(rhs);
    std::vector<iree_hal_dim_t> rhs_shape(rhs_rank);
    if (rhs_rank > 0) {
      IREE_RETURN_IF_ERROR(
          iree_hal_buffer_view_shape(rhs, rhs_rank, rhs_shape.data(), nullptr));
    }

    iree_hal_element_type_t lhs_element_type =
        iree_hal_buffer_view_element_type(lhs);
    iree_hal_element_type_t rhs_element_type =
        iree_hal_buffer_view_element_type(rhs);

    iree_hal_buffer_t* lhs_buf = iree_hal_buffer_view_buffer(lhs);
    iree_hal_buffer_mapping_t lhs_mapped_memory = {{0}};
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
        lhs_buf, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
        /*byte_offset=*/0, lhs_size, &lhs_mapped_memory));
    iree_hal_buffer_t* rhs_buf = iree_hal_buffer_view_buffer(rhs);
    iree_hal_buffer_mapping_t rhs_mapped_memory = {{0}};
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
        rhs_buf, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
        /*byte_offset=*/0, rhs_size, &rhs_mapped_memory));

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
    iree_status_ignore(iree_hal_buffer_unmap_range(&lhs_mapped_memory));
    iree_status_ignore(iree_hal_buffer_unmap_range(&rhs_mapped_memory));

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
      // TODO(gcmn): Propagate original variable names.
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

      // TODO(gcmn): Use ADD_FAILURE_AT to propagate source location.
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
extern "C" iree_status_t iree_check_module_create(
    iree_vm_instance_t* instance, iree_allocator_t allocator,
    iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  auto module = std::make_unique<CheckModule>(
      "check", /*version=*/0, instance, allocator,
      iree::span<const vm::NativeFunction<CheckModuleState>>(
          kCheckModuleFunctions));
  *out_module = module.release()->interface();
  return iree_ok_status();
}

}  // namespace iree
