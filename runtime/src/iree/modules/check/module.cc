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

// Numpy-compatible fuzzy comparison of floating-point values lhs, rhs with
// respect to tolerance parameters atol, rtol.
//
// The meaning of the tolerance parameters atol and rtol is exactly as in NumPy
// isclose():
// https://github.com/numpy/numpy/blob/7297f3117d84745bfade1e2f9aec3531e5917500/numpy/_core/numeric.py#L2447-L2449
// The condition being verified on each lhs and rhs value is:
//   lhs == rhs || (isfinite(rhs) && abs(lhs - rhs) <= atol + rtol * abs(rhs)).
// Note that the `lhs == rhs` part is needed for the case (lhs=+inf, rhs+inf)
// to return true. Indeed, in that case, lhs-rhs is NaN.
// Finally, unlike the above NumPy code, we also tolerate the case where both
// lhs and rhs are NaN. That avoids nonsensical test failures whenever a NaN
// is the legitimate result.
template <typename T>
bool NumpyFuzzyCompare(T lhs, T rhs, float atol, float rtol) {
  return lhs == rhs ||
         (std::isfinite(rhs) &&
          std::abs(lhs - rhs) <= atol + rtol * std::abs(rhs)) ||
         (std::isnan(lhs) && std::isnan(rhs));
}

// Records information about some LHS/RHS scalars that failed a fuzzy comparison
// and their possible position within arrays that were being compared.
struct FuzzyCompareDiagnostic {
  // Position in the LHS/RHS arrays of the values for which comparison failed.
  // May be left as 0 if the fuzzy comparison wasn't comparing arrays.
  int index = 0;
  // LHS/RHS values whose difference failed the fuzzy comparison.
  double lhs_value = 0.;
  double rhs_value = 0.;
};

template <iree_hal_element_type_t type>
struct FloatTypeInfo {};

template <>
struct FloatTypeInfo<IREE_HAL_ELEMENT_TYPE_FLOAT_32> {
  using ArithmeticType = float;
  using StorageType = float;
  static ArithmeticType load(StorageType val) { return val; }
};

template <>
struct FloatTypeInfo<IREE_HAL_ELEMENT_TYPE_FLOAT_64> {
  using ArithmeticType = double;
  using StorageType = double;
  static ArithmeticType load(StorageType val) { return val; }
};

template <>
struct FloatTypeInfo<IREE_HAL_ELEMENT_TYPE_FLOAT_16> {
  using ArithmeticType = float;
  using StorageType = uint16_t;
  static ArithmeticType load(StorageType val) {
    return iree_math_f16_to_f32(val);
  }
};

template <>
struct FloatTypeInfo<IREE_HAL_ELEMENT_TYPE_BFLOAT_16> {
  using ArithmeticType = float;
  using StorageType = uint16_t;
  static ArithmeticType load(StorageType val) {
    return iree_math_bf16_to_f32(val);
  }
};

template <>
struct FloatTypeInfo<IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FN> {
  using ArithmeticType = float;
  using StorageType = uint8_t;
  static ArithmeticType load(StorageType val) {
    return iree_math_f8e4m3fn_to_f32(val);
  }
};

template <>
struct FloatTypeInfo<IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FNUZ> {
  using ArithmeticType = float;
  using StorageType = uint8_t;
  static ArithmeticType load(StorageType val) {
    return iree_math_f8e4m3fnuz_to_f32(val);
  }
};

template <>
struct FloatTypeInfo<IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2> {
  using ArithmeticType = float;
  using StorageType = uint8_t;
  static ArithmeticType load(StorageType val) {
    return iree_math_f8e5m2_to_f32(val);
  }
};

template <>
struct FloatTypeInfo<IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2_FNUZ> {
  using ArithmeticType = float;
  using StorageType = uint8_t;
  static ArithmeticType load(StorageType val) {
    return iree_math_f8e5m2fnuz_to_f32(val);
  }
};

template <>
struct FloatTypeInfo<IREE_HAL_ELEMENT_TYPE_FLOAT_8_E8M0_FNU> {
  using ArithmeticType = float;
  using StorageType = uint8_t;
  static ArithmeticType load(StorageType val) {
    return iree_math_f8e8m0fnu_to_f32(val);
  }
};

// Fuzzy comparison of spans.
// The meaning of atol, rtol is explained in the comment on NumpyFuzzyCompare.
// On failure, false is returned, and information about a specific failed
// comparison is written to *diagnostic.
template <iree_hal_element_type_t type>
bool AlmostEqByteSpan(iree_byte_span_t lhs_bytes, iree_byte_span_t rhs_bytes,
                      float atol, float rtol,
                      FuzzyCompareDiagnostic* diagnostic) {
  using Info = FloatTypeInfo<type>;
  using ArithmeticType = typename Info::ArithmeticType;
  using StorageType = typename Info::StorageType;
  auto lhs_span = ToSpan<StorageType>(lhs_bytes);
  auto rhs_span = ToSpan<StorageType>(rhs_bytes);
  assert(lhs_span.size() == rhs_span.size());
  for (int i = 0; i < lhs_span.size(); ++i) {
    ArithmeticType lhs_value = Info::load(lhs_span[i]);
    ArithmeticType rhs_value = Info::load(rhs_span[i]);
    if (!NumpyFuzzyCompare(lhs_value, rhs_value, atol, rtol)) {
      diagnostic->index = i;
      diagnostic->lhs_value = lhs_value;
      diagnostic->rhs_value = rhs_value;
      return false;
    }
  }
  return true;
}

// The meaning of atol, rtol is explained in the comment on NumpyFuzzyCompare.
StatusOr<bool> AlmostEqByteSpan(iree_byte_span_t lhs_bytes,
                                iree_byte_span_t rhs_bytes,
                                iree_hal_element_type_t element_type,
                                float atol, float rtol,
                                FuzzyCompareDiagnostic* diagnostic) {
  switch (element_type) {
#define IREE_ALMOSTEQBYTESPAN_CASE(T) \
  case T:                             \
    return AlmostEqByteSpan<T>(lhs_bytes, rhs_bytes, atol, rtol, diagnostic);
    IREE_ALMOSTEQBYTESPAN_CASE(IREE_HAL_ELEMENT_TYPE_FLOAT_64)
    IREE_ALMOSTEQBYTESPAN_CASE(IREE_HAL_ELEMENT_TYPE_FLOAT_32)
    IREE_ALMOSTEQBYTESPAN_CASE(IREE_HAL_ELEMENT_TYPE_FLOAT_16)
    IREE_ALMOSTEQBYTESPAN_CASE(IREE_HAL_ELEMENT_TYPE_BFLOAT_16)
    IREE_ALMOSTEQBYTESPAN_CASE(IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FN)
    IREE_ALMOSTEQBYTESPAN_CASE(IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FNUZ)
    IREE_ALMOSTEQBYTESPAN_CASE(IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2)
    IREE_ALMOSTEQBYTESPAN_CASE(IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2_FNUZ)
    IREE_ALMOSTEQBYTESPAN_CASE(IREE_HAL_ELEMENT_TYPE_FLOAT_8_E8M0_FNU)
#undef IREE_ALMOSTEQBYTESPAN_CASE
    default:
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
        command_buffer.get(),
        iree_hal_make_buffer_ref(source_buffer, 0, buffer_length),
        iree_hal_make_buffer_ref(target_buffer.get(), 0, buffer_length),
        IREE_HAL_COPY_FLAG_NONE));
    vm::ref<iree_hal_buffer_view_t> target_view;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create_like(
        target_buffer.get(), source_views[i].get(),
        iree_hal_device_host_allocator(device), &target_view));
    target_views.push_back(std::move(target_view));
  }

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_end(command_buffer.get()));
  vm::ref<iree_hal_semaphore_t> semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device, 0ull, IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
  vm::ref<iree_hal_fence_t> fence;
  IREE_RETURN_IF_ERROR(iree_hal_fence_create_at(
      semaphore.get(), 1ull, iree_hal_device_host_allocator(device), &fence));
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_execute(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      iree_hal_fence_semaphore_list(fence.get()), command_buffer.get(),
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_RETURN_IF_ERROR(
      iree_hal_fence_wait(fence.get(), iree_infinite_timeout()));
  return std::move(target_views);
}

static Status TransferToHost(iree_hal_device_t* device,
                             vm::ref<iree_hal_buffer_view_t>& buffer_view) {
  IREE_TRACE_SCOPE();
  IREE_ASSIGN_OR_RETURN(
      auto target_views,
      TransferBuffersToHost(
          device,
          iree::span<const vm::ref<iree_hal_buffer_view_t>>({buffer_view})));
  buffer_view = std::move(target_views[0]);
  return OkStatus();
}

static Status TransferToHost(iree_hal_device_t* device,
                             vm::ref<iree_hal_buffer_view_t>& buffer_view_a,
                             vm::ref<iree_hal_buffer_view_t>& buffer_view_b) {
  IREE_TRACE_SCOPE();
  IREE_ASSIGN_OR_RETURN(
      auto target_views,
      TransferBuffersToHost(device,
                            iree::span<const vm::ref<iree_hal_buffer_view_t>>(
                                {buffer_view_a, buffer_view_b})));
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
                        vm::ref<iree_hal_buffer_view_t> rhs_ref, float atol,
                        float rtol) {
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
    FuzzyCompareDiagnostic diagnostic;
    if (element_types_eq && shape_eq) {
      IREE_ASSIGN_OR_RETURN(
          contents_could_be_almost_eq,
          AlmostEqByteSpan(lhs_mapped_memory.contents,
                           rhs_mapped_memory.contents, lhs_element_type, atol,
                           rtol, &diagnostic));
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
        os << " Contents does not match to tolerance parameters atol=" << atol
           << ", rtol=" << rtol << ". The first failure occurs at index "
           << diagnostic.index << " as the lhs value " << diagnostic.lhs_value
           << " differs from the rhs value " << diagnostic.rhs_value << ".";
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

  StatusOr<std::unique_ptr<CheckModuleState>> ForkState(
      CheckModuleState* parent_state, iree_allocator_t allocator) override {
    // No state needs to be forked.
    return CreateState(allocator);
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
