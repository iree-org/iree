// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/buffer_view_matchers.h"

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/base/internal/span.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

using iree::testing::status::IsOk;
using iree::testing::status::StatusIs;
using ::testing::HasSubstr;

// TODO(benvanik): move this handle type to a base cc helper.
struct StringBuilder {
  static StringBuilder MakeSystem() {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_default(), &builder);
    return StringBuilder(builder);
  }
  static StringBuilder MakeEmpty() {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_null(), &builder);
    return StringBuilder(builder);
  }
  explicit StringBuilder(iree_string_builder_t builder)
      : builder(std::move(builder)) {}
  ~StringBuilder() { iree_string_builder_deinitialize(&builder); }
  operator iree_string_builder_t*() { return &builder; }
  std::string ToString() const {
    return std::string(builder.buffer, builder.size);
  }
  iree_string_builder_t builder;
};

// TODO(benvanik): move this handle type to a hal cc helper.

// C API iree_*_retain/iree_*_release function pointer.
template <typename T>
using HandleRefFn = void(IREE_API_PTR*)(T*);

// C++ RAII wrapper for an IREE C reference object.
// Behaves the same as a thread-safe intrusive pointer.
template <typename T, HandleRefFn<T> retain_fn, HandleRefFn<T> release_fn>
class Handle {
 public:
  using handle_type = Handle<T, retain_fn, release_fn>;

  static Handle Wrap(T* value) noexcept { return Handle(value, false); }

  Handle() noexcept = default;
  Handle(std::nullptr_t) noexcept {}
  Handle(T* value) noexcept : value_(value) { retain_fn(value_); }

  ~Handle() noexcept {
    if (value_) release_fn(value_);
  }

  Handle(const Handle& rhs) noexcept : value_(rhs.value_) {
    if (value_) retain_fn(value_);
  }
  Handle& operator=(const Handle& rhs) noexcept {
    if (value_ != rhs.value_) {
      if (value_) release_fn(value_);
      value_ = rhs.get();
      if (value_) retain_fn(value_);
    }
    return *this;
  }

  Handle(Handle&& rhs) noexcept : value_(rhs.release()) {}
  Handle& operator=(Handle&& rhs) noexcept {
    if (value_ != rhs.value_) {
      if (value_) release_fn(value_);
      value_ = rhs.release();
    }
    return *this;
  }

  // Gets the pointer referenced by this instance.
  constexpr T* get() const noexcept { return value_; }
  constexpr operator T*() const noexcept { return value_; }

  // Resets the object to nullptr and decrements the reference count, possibly
  // deleting it.
  void reset() noexcept {
    if (value_) {
      release_fn(value_);
      value_ = nullptr;
    }
  }

  // Returns the current pointer held by this object without having its
  // reference count decremented and resets the handle to empty. Returns
  // nullptr if the handle holds no value. To re-wrap in a handle use either
  // ctor(value) or assign().
  T* release() noexcept {
    auto* p = value_;
    value_ = nullptr;
    return p;
  }

  // Assigns a pointer.
  // The pointer will be accepted by the handle and its reference count will
  // not be incremented.
  void assign(T* value) noexcept {
    reset();
    value_ = value;
  }

  // Returns a pointer to the inner pointer storage.
  // This allows passing a pointer to the handle as an output argument to
  // C-style creation functions.
  constexpr T** operator&() noexcept { return &value_; }

  // Support boolean expression evaluation ala unique_ptr/shared_ptr:
  // https://en.cppreference.com/w/cpp/memory/shared_ptr/operator_bool
  typedef T* Handle::*unspecified_bool_type;
  constexpr operator unspecified_bool_type() const noexcept {
    return value_ ? &Handle::value_ : nullptr;
  }

  // Supports unary expression evaluation.
  constexpr bool operator!() const noexcept { return !value_; }

  // Swap support.
  void swap(Handle& rhs) noexcept { std::swap(value_, rhs.value_); }

 protected:
  Handle(T* value, bool) noexcept : value_(value) {}

 private:
  T* value_ = nullptr;
};

// C++ wrapper for iree_hal_buffer_t.
struct Buffer final : public Handle<iree_hal_buffer_t, iree_hal_buffer_retain,
                                    iree_hal_buffer_release> {
  using handle_type::handle_type;
};

// C++ wrapper for iree_hal_buffer_view_t.
struct BufferView final
    : public Handle<iree_hal_buffer_view_t, iree_hal_buffer_view_retain,
                    iree_hal_buffer_view_release> {
  using handle_type::handle_type;
};

static const iree_hal_buffer_equality_t kExactEquality = ([]() {
  iree_hal_buffer_equality_t equality;
  equality.mode = IREE_HAL_BUFFER_EQUALITY_EXACT;
  return equality;
})();

static const iree_hal_buffer_equality_t kApproximateEquality = ([]() {
  iree_hal_buffer_equality_t equality;
  equality.mode = IREE_HAL_BUFFER_EQUALITY_APPROXIMATE_ABSOLUTE;
  equality.f16_threshold = 0.001f;
  equality.f32_threshold = 0.0001f;
  equality.f64_threshold = 0.0001;
  return equality;
})();

class BufferViewMatchersTest : public ::testing::Test {
 protected:
  iree_hal_allocator_t* device_allocator_ = nullptr;
  virtual void SetUp() {
    IREE_CHECK_OK(iree_hal_allocator_create_heap(
        IREE_SV("heap"), iree_allocator_default(), iree_allocator_default(),
        &device_allocator_));
  }
  virtual void TearDown() { iree_hal_allocator_release(device_allocator_); }

  template <typename T>
  StatusOr<BufferView> CreateBufferView(iree::span<const iree_hal_dim_t> shape,
                                        iree_hal_element_type_t element_type,
                                        const T* contents) {
    iree_hal_dim_t num_elements = 1;
    for (iree_hal_dim_t dim : shape) num_elements *= dim;
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
    params.usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
    Buffer buffer;
    IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
        device_allocator_, params, num_elements * sizeof(T), &buffer));
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_write(buffer, 0, contents,
                                                   num_elements * sizeof(T)));
    BufferView buffer_view;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
        buffer, shape.size(), shape.data(), element_type,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, iree_allocator_default(),
        &buffer_view));
    return std::move(buffer_view);
  }
};

//===----------------------------------------------------------------------===//
// iree_hal_buffer_equality_t
//===----------------------------------------------------------------------===//

TEST_F(BufferViewMatchersTest, CompareBroadcastI8EQ) {
  const int8_t lhs = 1;
  const int8_t rhs[] = {1, 1, 1};
  iree_host_size_t index = 0;
  EXPECT_TRUE(iree_hal_compare_buffer_elements_broadcast(
      kApproximateEquality, iree_hal_make_buffer_element_i8(lhs),
      IREE_ARRAYSIZE(rhs), iree_make_const_byte_span(rhs, sizeof(rhs)),
      &index));
}

TEST_F(BufferViewMatchersTest, CompareBroadcastI8NE) {
  const int8_t lhs = 1;
  const int8_t rhs[] = {1, 2, 3};
  iree_host_size_t index = 0;
  EXPECT_FALSE(iree_hal_compare_buffer_elements_broadcast(
      kApproximateEquality, iree_hal_make_buffer_element_i8(lhs),
      IREE_ARRAYSIZE(rhs), iree_make_const_byte_span(rhs, sizeof(rhs)),
      &index));
  EXPECT_EQ(index, 1);
}

TEST_F(BufferViewMatchersTest, CompareBroadcastI64EQ) {
  const int64_t lhs = 1;
  const int64_t rhs[] = {1, 1, 1};
  iree_host_size_t index = 0;
  EXPECT_TRUE(iree_hal_compare_buffer_elements_broadcast(
      kApproximateEquality, iree_hal_make_buffer_element_i64(lhs),
      IREE_ARRAYSIZE(rhs), iree_make_const_byte_span(rhs, sizeof(rhs)),
      &index));
}

TEST_F(BufferViewMatchersTest, CompareBroadcastI64NE) {
  const int64_t lhs = 1;
  const int64_t rhs[] = {1, 2, 3};
  iree_host_size_t index = 0;
  EXPECT_FALSE(iree_hal_compare_buffer_elements_broadcast(
      kApproximateEquality, iree_hal_make_buffer_element_i64(lhs),
      IREE_ARRAYSIZE(rhs), iree_make_const_byte_span(rhs, sizeof(rhs)),
      &index));
  EXPECT_EQ(index, 1);
}

TEST_F(BufferViewMatchersTest, CompareBroadcastF16EQ) {
  const float lhs = 1.0f;
  const uint16_t rhs[] = {
      iree_math_f32_to_f16(1.0f),
      iree_math_f32_to_f16(1.0f),
      iree_math_f32_to_f16(1.0f),
  };
  iree_host_size_t index = 0;
  EXPECT_TRUE(iree_hal_compare_buffer_elements_broadcast(
      kApproximateEquality, iree_hal_make_buffer_element_f16(lhs),
      IREE_ARRAYSIZE(rhs), iree_make_const_byte_span(rhs, sizeof(rhs)),
      &index));
}

TEST_F(BufferViewMatchersTest, CompareBroadcastF16NE) {
  const float lhs = 1.0f;
  const uint16_t rhs[] = {
      iree_math_f32_to_f16(1.0f),
      iree_math_f32_to_f16(3.0f),
      iree_math_f32_to_f16(4.0f),
  };
  iree_host_size_t index = 0;
  EXPECT_FALSE(iree_hal_compare_buffer_elements_broadcast(
      kApproximateEquality, iree_hal_make_buffer_element_f16(lhs),
      IREE_ARRAYSIZE(rhs), iree_make_const_byte_span(rhs, sizeof(rhs)),
      &index));
  EXPECT_EQ(index, 1);
}

TEST_F(BufferViewMatchersTest, CompareBroadcastF32EQ) {
  const float lhs = 1.0f;
  const float rhs[] = {1.0f, 1.0f, 1.0f};
  iree_host_size_t index = 0;
  EXPECT_TRUE(iree_hal_compare_buffer_elements_broadcast(
      kApproximateEquality, iree_hal_make_buffer_element_f32(lhs),
      IREE_ARRAYSIZE(rhs), iree_make_const_byte_span(rhs, sizeof(rhs)),
      &index));
}

TEST_F(BufferViewMatchersTest, CompareBroadcastF32NE) {
  const float lhs = 1.0f;
  const float rhs[] = {1.0f, 3.0f, 4.0f};
  iree_host_size_t index = 0;
  EXPECT_FALSE(iree_hal_compare_buffer_elements_broadcast(
      kApproximateEquality, iree_hal_make_buffer_element_f32(lhs),
      IREE_ARRAYSIZE(rhs), iree_make_const_byte_span(rhs, sizeof(rhs)),
      &index));
  EXPECT_EQ(index, 1);
}

TEST_F(BufferViewMatchersTest, CompareBroadcastF64EQ) {
  const double lhs = 1.0;
  const double rhs[] = {1.0, 1.0, 1.0};
  iree_host_size_t index = 0;
  EXPECT_TRUE(iree_hal_compare_buffer_elements_broadcast(
      kApproximateEquality, iree_hal_make_buffer_element_f64(lhs),
      IREE_ARRAYSIZE(rhs), iree_make_const_byte_span(rhs, sizeof(rhs)),
      &index));
}

TEST_F(BufferViewMatchersTest, CompareBroadcastF64NE) {
  const double lhs = 1.0;
  const double rhs[] = {1.0, 3.0, 4.0};
  iree_host_size_t index = 0;
  EXPECT_FALSE(iree_hal_compare_buffer_elements_broadcast(
      kApproximateEquality, iree_hal_make_buffer_element_f64(lhs),
      IREE_ARRAYSIZE(rhs), iree_make_const_byte_span(rhs, sizeof(rhs)),
      &index));
  EXPECT_EQ(index, 1);
}

TEST_F(BufferViewMatchersTest, CompareElementwiseF16EQ) {
  const uint16_t lhs[] = {
      iree_math_f32_to_f16(1.0f),
      iree_math_f32_to_f16(2.0f),
      iree_math_f32_to_f16(3.0f),
  };
  const uint16_t rhs[] = {
      iree_math_f32_to_f16(1.0f),
      iree_math_f32_to_f16(2.0f),
      iree_math_f32_to_f16(3.0f),
  };
  iree_host_size_t index = 0;
  EXPECT_TRUE(iree_hal_compare_buffer_elements_elementwise(
      kApproximateEquality, IREE_HAL_ELEMENT_TYPE_FLOAT_16, IREE_ARRAYSIZE(lhs),
      iree_make_const_byte_span(lhs, sizeof(lhs)),
      iree_make_const_byte_span(rhs, sizeof(rhs)), &index));
}

TEST_F(BufferViewMatchersTest, CompareElementwiseF16NearEQ) {
  const uint16_t lhs[] = {
      iree_math_f32_to_f16(1.0f),
      iree_math_f32_to_f16(1.99999f),
      iree_math_f32_to_f16(0.00001f),
      iree_math_f32_to_f16(4.0f),
  };
  const uint16_t rhs[] = {
      iree_math_f32_to_f16(1.00001f),
      iree_math_f32_to_f16(2.0f),
      iree_math_f32_to_f16(0.0f),
      iree_math_f32_to_f16(4.0f),
  };
  iree_host_size_t index = 0;
  EXPECT_TRUE(iree_hal_compare_buffer_elements_elementwise(
      kApproximateEquality, IREE_HAL_ELEMENT_TYPE_FLOAT_16, IREE_ARRAYSIZE(lhs),
      iree_make_const_byte_span(lhs, sizeof(lhs)),
      iree_make_const_byte_span(rhs, sizeof(rhs)), &index));
}

TEST_F(BufferViewMatchersTest, CompareElementwiseF16NE) {
  const uint16_t lhs[] = {
      iree_math_f32_to_f16(1.0f),
      iree_math_f32_to_f16(2.0f),
      iree_math_f32_to_f16(4.0f),
  };
  const uint16_t rhs[] = {
      iree_math_f32_to_f16(1.0f),
      iree_math_f32_to_f16(3.0f),
      iree_math_f32_to_f16(4.0f),
  };
  iree_host_size_t index = 0;
  EXPECT_FALSE(iree_hal_compare_buffer_elements_elementwise(
      kApproximateEquality, IREE_HAL_ELEMENT_TYPE_FLOAT_16, IREE_ARRAYSIZE(lhs),
      iree_make_const_byte_span(lhs, sizeof(lhs)),
      iree_make_const_byte_span(rhs, sizeof(rhs)), &index));
  EXPECT_EQ(index, 1);
}

TEST_F(BufferViewMatchersTest, CompareElementwiseF32EQ) {
  const float lhs[] = {1.0f, 2.0f, 3.0f};
  const float rhs[] = {1.0f, 2.0f, 3.0f};
  iree_host_size_t index = 0;
  EXPECT_TRUE(iree_hal_compare_buffer_elements_elementwise(
      kApproximateEquality, IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_ARRAYSIZE(lhs),
      iree_make_const_byte_span(lhs, sizeof(lhs)),
      iree_make_const_byte_span(rhs, sizeof(rhs)), &index));
}

TEST_F(BufferViewMatchersTest, CompareElementwiseF32NE) {
  const float lhs[] = {1.0f, 2.0f, 4.0f};
  const float rhs[] = {1.0f, 3.0f, 4.0f};
  iree_host_size_t index = 0;
  EXPECT_FALSE(iree_hal_compare_buffer_elements_elementwise(
      kApproximateEquality, IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_ARRAYSIZE(lhs),
      iree_make_const_byte_span(lhs, sizeof(lhs)),
      iree_make_const_byte_span(rhs, sizeof(rhs)), &index));
  EXPECT_EQ(index, 1);
}

TEST_F(BufferViewMatchersTest, CompareElementwiseF64EQ) {
  const double lhs[] = {1.0, 2.0, 3.0};
  const double rhs[] = {1.0, 2.0, 3.0};
  iree_host_size_t index = 0;
  EXPECT_TRUE(iree_hal_compare_buffer_elements_elementwise(
      kApproximateEquality, IREE_HAL_ELEMENT_TYPE_FLOAT_64, IREE_ARRAYSIZE(lhs),
      iree_make_const_byte_span(lhs, sizeof(lhs)),
      iree_make_const_byte_span(rhs, sizeof(rhs)), &index));
}

TEST_F(BufferViewMatchersTest, CompareElementwiseF64NE) {
  const double lhs[] = {1.0, 2.0, 4.0};
  const double rhs[] = {1.0, 3.0, 4.0};
  iree_host_size_t index = 0;
  EXPECT_FALSE(iree_hal_compare_buffer_elements_elementwise(
      kApproximateEquality, IREE_HAL_ELEMENT_TYPE_FLOAT_64, IREE_ARRAYSIZE(lhs),
      iree_make_const_byte_span(lhs, sizeof(lhs)),
      iree_make_const_byte_span(rhs, sizeof(rhs)), &index));
  EXPECT_EQ(index, 1);
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_metadata_matcher_t
//===----------------------------------------------------------------------===//

TEST_F(BufferViewMatchersTest, MetadataEmpty) {
  const float contents[1] = {0};
  iree_hal_dim_t shape[] = {0};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto lhs,
      CreateBufferView(shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32, contents));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto rhs,
      CreateBufferView(shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32, contents));
  auto sb = StringBuilder::MakeSystem();
  bool match = false;
  IREE_ASSERT_OK(
      iree_hal_buffer_view_match_metadata_like(lhs, rhs, sb, &match));
  EXPECT_TRUE(match);
}

TEST_F(BufferViewMatchersTest, MetadataShapesDiffer) {
  const float lhs_contents[] = {1.0f};
  const float rhs_contents[] = {1.0f, 2.0f};
  iree_hal_dim_t lhs_shape[] = {1};
  iree_hal_dim_t rhs_shape[] = {1, 2};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto lhs, CreateBufferView(lhs_shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                 lhs_contents));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto rhs, CreateBufferView(rhs_shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                 rhs_contents));
  auto sb = StringBuilder::MakeSystem();
  bool match = false;
  IREE_ASSERT_OK(
      iree_hal_buffer_view_match_metadata_like(lhs, rhs, sb, &match));
  EXPECT_FALSE(match);
  EXPECT_THAT(sb.ToString(), HasSubstr("is 1x2xf32"));
  EXPECT_THAT(sb.ToString(), HasSubstr("matches 1xf32"));
}

TEST_F(BufferViewMatchersTest, MetadataElementTypesDiffer) {
  const float contents[] = {1.0f};
  iree_hal_dim_t shape[] = {1};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto lhs,
      CreateBufferView(shape, IREE_HAL_ELEMENT_TYPE_INT_32, contents));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto rhs,
      CreateBufferView(shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32, contents));
  auto sb = StringBuilder::MakeSystem();
  bool match = false;
  IREE_ASSERT_OK(
      iree_hal_buffer_view_match_metadata_like(lhs, rhs, sb, &match));
  EXPECT_FALSE(match);
  EXPECT_THAT(sb.ToString(), HasSubstr("is 1xf32"));
  EXPECT_THAT(sb.ToString(), HasSubstr("matches 1xi32"));
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_element_matcher_t
//===----------------------------------------------------------------------===//

TEST_F(BufferViewMatchersTest, ElementTypesDiffer) {
  const float lhs_value = 1;
  const int32_t rhs_contents[] = {1, 1, 1};
  const iree_hal_dim_t shape[] = {1, 3};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto rhs,
      CreateBufferView(shape, IREE_HAL_ELEMENT_TYPE_INT_32, rhs_contents));
  auto sb = StringBuilder::MakeSystem();
  bool match = false;
  IREE_ASSERT_OK(iree_hal_buffer_view_match_elements(
      kExactEquality, iree_hal_make_buffer_element_f32(lhs_value), rhs, sb,
      &match));
  EXPECT_FALSE(match);
  EXPECT_THAT(sb.ToString(), HasSubstr("type (i32)"));
  EXPECT_THAT(sb.ToString(), HasSubstr("expected (f32)"));
}

TEST_F(BufferViewMatchersTest, MatchElementContentsI32) {
  const int32_t lhs_value = 1;
  const int32_t rhs_contents[] = {1, 1, 1};
  const iree_hal_dim_t shape[] = {1, 3};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto rhs,
      CreateBufferView(shape, IREE_HAL_ELEMENT_TYPE_INT_32, rhs_contents));
  auto sb = StringBuilder::MakeSystem();
  bool match = false;
  IREE_ASSERT_OK(iree_hal_buffer_view_match_elements(
      kExactEquality, iree_hal_make_buffer_element_i32(lhs_value), rhs, sb,
      &match));
  EXPECT_TRUE(match);
  EXPECT_TRUE(sb.ToString().empty());
}

TEST_F(BufferViewMatchersTest, MismatchElementContentsI32) {
  const int32_t lhs_value = 1;
  const int32_t rhs_contents[] = {1, 2, 3};
  const iree_hal_dim_t shape[] = {1, 3};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto rhs,
      CreateBufferView(shape, IREE_HAL_ELEMENT_TYPE_INT_32, rhs_contents));
  auto sb = StringBuilder::MakeSystem();
  bool match = false;
  IREE_ASSERT_OK(iree_hal_buffer_view_match_elements(
      kExactEquality, iree_hal_make_buffer_element_i32(lhs_value), rhs, sb,
      &match));
  EXPECT_FALSE(match);
  EXPECT_THAT(sb.ToString(), HasSubstr("element at index 1"));
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_array_matcher_t
//===----------------------------------------------------------------------===//

TEST_F(BufferViewMatchersTest, MatchArrayTypesDiffer) {
  const float lhs_contents[] = {1, 1, 1};
  const int32_t rhs_contents[] = {1, 1, 1};
  const iree_hal_dim_t shape[] = {1, 3};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto rhs,
      CreateBufferView(shape, IREE_HAL_ELEMENT_TYPE_INT_32, rhs_contents));
  auto sb = StringBuilder::MakeSystem();
  bool match = false;
  IREE_ASSERT_OK(iree_hal_buffer_view_match_array(
      kExactEquality, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_ARRAYSIZE(lhs_contents),
      iree_make_const_byte_span(lhs_contents, sizeof(lhs_contents)), rhs, sb,
      &match));
  EXPECT_FALSE(match);
  EXPECT_THAT(sb.ToString(), HasSubstr("type (i32)"));
  EXPECT_THAT(sb.ToString(), HasSubstr("expected (f32)"));
}

TEST_F(BufferViewMatchersTest, MatchArrayCountsDiffer) {
  const int32_t lhs_contents[] = {1, 1};
  const int32_t rhs_contents[] = {1, 1, 1};
  const iree_hal_dim_t shape[] = {1, 3};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto rhs,
      CreateBufferView(shape, IREE_HAL_ELEMENT_TYPE_INT_32, rhs_contents));
  auto sb = StringBuilder::MakeSystem();
  bool match = false;
  IREE_ASSERT_OK(iree_hal_buffer_view_match_array(
      kExactEquality, IREE_HAL_ELEMENT_TYPE_INT_32,
      IREE_ARRAYSIZE(lhs_contents),
      iree_make_const_byte_span(lhs_contents, sizeof(lhs_contents)), rhs, sb,
      &match));
  EXPECT_FALSE(match);
  EXPECT_THAT(sb.ToString(), HasSubstr("count (3)"));
  EXPECT_THAT(sb.ToString(), HasSubstr("expected (2)"));
}

TEST_F(BufferViewMatchersTest, MatchArrayContentsI32) {
  const int32_t lhs_contents[] = {1, 1, 1};
  const int32_t rhs_contents[] = {1, 1, 1};
  const iree_hal_dim_t shape[] = {1, 3};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto rhs,
      CreateBufferView(shape, IREE_HAL_ELEMENT_TYPE_INT_32, rhs_contents));
  auto sb = StringBuilder::MakeSystem();
  bool match = false;
  IREE_ASSERT_OK(iree_hal_buffer_view_match_array(
      kExactEquality, IREE_HAL_ELEMENT_TYPE_INT_32,
      IREE_ARRAYSIZE(lhs_contents),
      iree_make_const_byte_span(lhs_contents, sizeof(lhs_contents)), rhs, sb,
      &match));
  EXPECT_TRUE(match);
  EXPECT_TRUE(sb.ToString().empty());
}

TEST_F(BufferViewMatchersTest, MismatchArrayContentsI32) {
  const int32_t lhs_contents[] = {1, 1, 1};
  const int32_t rhs_contents[] = {1, 2, 3};
  const iree_hal_dim_t shape[] = {1, 3};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto rhs,
      CreateBufferView(shape, IREE_HAL_ELEMENT_TYPE_INT_32, rhs_contents));
  auto sb = StringBuilder::MakeSystem();
  bool match = false;
  IREE_ASSERT_OK(iree_hal_buffer_view_match_array(
      kExactEquality, IREE_HAL_ELEMENT_TYPE_INT_32,
      IREE_ARRAYSIZE(lhs_contents),
      iree_make_const_byte_span(lhs_contents, sizeof(lhs_contents)), rhs, sb,
      &match));
  EXPECT_FALSE(match);
  EXPECT_THAT(sb.ToString(), HasSubstr("element at index 1"));
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_matcher_t
//===----------------------------------------------------------------------===//

TEST_F(BufferViewMatchersTest, MatchEmpty) {
  const float contents[1] = {0};
  iree_hal_dim_t shape[] = {0};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto lhs,
      CreateBufferView(shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32, contents));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto rhs,
      CreateBufferView(shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32, contents));
  auto sb = StringBuilder::MakeSystem();
  bool match = false;
  IREE_ASSERT_OK(
      iree_hal_buffer_view_match_equal(kExactEquality, lhs, rhs, sb, &match));
  EXPECT_TRUE(match);
  EXPECT_TRUE(sb.ToString().empty());
}

TEST_F(BufferViewMatchersTest, MatchShapesDiffer) {
  const float lhs_contents[] = {1.0f};
  const float rhs_contents[] = {1.0f, 2.0f};
  iree_hal_dim_t lhs_shape[] = {1};
  iree_hal_dim_t rhs_shape[] = {1, 2};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto lhs, CreateBufferView(lhs_shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                 lhs_contents));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto rhs, CreateBufferView(rhs_shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                 rhs_contents));
  auto sb = StringBuilder::MakeSystem();
  bool match = false;
  IREE_ASSERT_OK(
      iree_hal_buffer_view_match_equal(kExactEquality, lhs, rhs, sb, &match));
  EXPECT_FALSE(match);
  EXPECT_THAT(sb.ToString(), HasSubstr("is 1x2xf32"));
  EXPECT_THAT(sb.ToString(), HasSubstr("matches 1xf32"));
}

TEST_F(BufferViewMatchersTest, MatchElementTypesDiffer) {
  const float contents[] = {1.0f};
  iree_hal_dim_t shape[] = {1};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto lhs,
      CreateBufferView(shape, IREE_HAL_ELEMENT_TYPE_INT_32, contents));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto rhs,
      CreateBufferView(shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32, contents));
  auto sb = StringBuilder::MakeSystem();
  bool match = false;
  IREE_ASSERT_OK(
      iree_hal_buffer_view_match_equal(kExactEquality, lhs, rhs, sb, &match));
  EXPECT_FALSE(match);
  EXPECT_THAT(sb.ToString(), HasSubstr("is 1xf32"));
  EXPECT_THAT(sb.ToString(), HasSubstr("matches 1xi32"));
}

TEST_F(BufferViewMatchersTest, MatchContentsF16) {
  const uint16_t lhs_contents[] = {iree_math_f32_to_f16(2.0f)};
  const uint16_t rhs_contents[] = {iree_math_f32_to_f16(2.0f)};
  iree_hal_dim_t shape[] = {1};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto lhs,
      CreateBufferView(shape, IREE_HAL_ELEMENT_TYPE_FLOAT_16, lhs_contents));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto rhs,
      CreateBufferView(shape, IREE_HAL_ELEMENT_TYPE_FLOAT_16, rhs_contents));
  auto sb = StringBuilder::MakeSystem();
  bool match = false;
  IREE_ASSERT_OK(
      iree_hal_buffer_view_match_equal(kExactEquality, lhs, rhs, sb, &match));
  EXPECT_TRUE(match);
  EXPECT_TRUE(sb.ToString().empty());
}

TEST_F(BufferViewMatchersTest, MismatchContentsF16) {
  const uint16_t lhs_contents[] = {iree_math_f32_to_f16(1.0f)};
  const uint16_t rhs_contents[] = {iree_math_f32_to_f16(2.0f)};
  const iree_hal_dim_t shape[] = {1};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto lhs,
      CreateBufferView(shape, IREE_HAL_ELEMENT_TYPE_FLOAT_16, lhs_contents));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto rhs,
      CreateBufferView(shape, IREE_HAL_ELEMENT_TYPE_FLOAT_16, rhs_contents));
  auto sb = StringBuilder::MakeSystem();
  bool match = false;
  IREE_ASSERT_OK(
      iree_hal_buffer_view_match_equal(kExactEquality, lhs, rhs, sb, &match));
  EXPECT_FALSE(match);
  EXPECT_THAT(sb.ToString(), HasSubstr("element at index 0"));
}

}  // namespace
}  // namespace iree
