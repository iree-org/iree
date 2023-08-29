// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/span.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace {

using ::iree::testing::status::IsOkAndHolds;
using ::iree::testing::status::StatusIs;
using ::testing::ElementsAre;
using ::testing::Eq;

// TODO(benvanik): move these utils to C++ bindings.
using Shape = std::vector<iree_hal_dim_t>;

// Parses a serialized set of shape dimensions using the canonical shape format
// (the same as produced by FormatShape).
static StatusOr<Shape> ParseShape(const std::string& value) {
  Shape shape(6);
  iree_host_size_t actual_rank = 0;
  iree_status_t status = iree_ok_status();
  do {
    status = iree_hal_parse_shape(
        iree_string_view_t{value.data(), (iree_host_size_t)value.size()},
        shape.size(), &actual_rank, shape.data());
    shape.resize(actual_rank);
  } while (iree_status_is_out_of_range(status));
  IREE_RETURN_IF_ERROR(std::move(status));
  return std::move(shape);
}

// Converts shape dimensions into a `4x5x6` format.
static StatusOr<std::string> FormatShape(
    iree::span<const iree_hal_dim_t> value) {
  std::string buffer(16, '\0');
  iree_host_size_t actual_length = 0;
  iree_status_t status = iree_ok_status();
  do {
    status =
        iree_hal_format_shape(value.size(), value.data(), buffer.size() + 1,
                              &buffer[0], &actual_length);
    buffer.resize(actual_length);
  } while (iree_status_is_out_of_range(status));
  IREE_RETURN_IF_ERROR(std::move(status));
  return std::move(buffer);
}

// Parses a serialized iree_hal_element_type_t. The format is the same as
// produced by FormatElementType.
static StatusOr<iree_hal_element_type_t> ParseElementType(
    const std::string& value) {
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  iree_status_t status = iree_hal_parse_element_type(
      iree_string_view_t{value.data(), (iree_host_size_t)value.size()},
      &element_type);
  IREE_RETURN_IF_ERROR(status, "Failed to parse element type '%.*s'",
                       (int)value.size(), value.data());
  return element_type;
}

// Converts an iree_hal_element_type_t enum value to a canonical string
// representation, like `IREE_HAL_ELEMENT_TYPE_FLOAT_16` to `f16`.
static StatusOr<std::string> FormatElementType(iree_hal_element_type_t value) {
  std::string buffer(16, '\0');
  iree_host_size_t actual_length = 0;
  iree_status_t status = iree_ok_status();
  do {
    status = iree_hal_format_element_type(value, buffer.size() + 1, &buffer[0],
                                          &actual_length);
    buffer.resize(actual_length);
  } while (iree_status_is_out_of_range(status));
  IREE_RETURN_IF_ERROR(status);
  return std::move(buffer);
}

struct ShapeAndType {
  Shape shape;
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  ShapeAndType() = default;
  ShapeAndType(Shape shape, iree_hal_element_type_t element_type)
      : shape(std::move(shape)), element_type(element_type) {}
};
static bool operator==(const ShapeAndType& lhs,
                       const ShapeAndType& rhs) noexcept {
  return lhs.shape == rhs.shape && lhs.element_type == rhs.element_type;
}

// Parses a serialized set of shape dimensions and an element type.
static StatusOr<ShapeAndType> ParseShapeAndElementType(
    const std::string& value) {
  Shape shape(6);
  iree_host_size_t actual_rank = 0;
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  iree_status_t status = iree_ok_status();
  do {
    status = iree_hal_parse_shape_and_element_type(
        iree_string_view_t{value.data(), (iree_host_size_t)value.size()},
        shape.size(), &actual_rank, shape.data(), &element_type);
    shape.resize(actual_rank);
  } while (iree_status_is_out_of_range(status));
  IREE_RETURN_IF_ERROR(std::move(status));
  return ShapeAndType(std::move(shape), element_type);
}

// Parses a serialized element of |element_type| to its in-memory form.
// |buffer| be at least large enough to contain the bytes of the element.
// For example, "1.2" of type IREE_HAL_ELEMENT_TYPE_FLOAT32 will write the 4
// byte float value of 1.2 to |buffer|.
template <typename T>
static Status ParseElement(const std::string& value,
                           iree_hal_element_type_t element_type,
                           iree::span<T> buffer) {
  return iree_hal_parse_element(
      iree_string_view_t{value.data(), (iree_host_size_t)value.size()},
      element_type,
      iree_byte_span_t{
          reinterpret_cast<uint8_t*>(buffer.data()),
          static_cast<iree_host_size_t>(buffer.size() * sizeof(T))});
}

// Converts a single element of |element_type| to a string.
template <typename T>
static StatusOr<std::string> FormatElement(
    T value, iree_hal_element_type_t element_type) {
  std::string result(16, '\0');
  iree_status_t status;
  do {
    iree_host_size_t actual_length = 0;
    status = iree_hal_format_element(
        iree_const_byte_span_t{reinterpret_cast<const uint8_t*>(&value),
                               sizeof(T)},
        element_type, result.size() + 1, &result[0], &actual_length);
    result.resize(actual_length);
  } while (iree_status_is_out_of_range(status));
  IREE_RETURN_IF_ERROR(status, "failed to format buffer element");
  return std::move(result);
}

// Parses a serialized set of elements of the given |element_type|.
// The resulting parsed data is written to |buffer|, which must be at least
// large enough to contain the parsed elements. The format is the same as
// produced by FormatBufferElements. Supports additional inputs of
// empty to denote a 0 fill and a single element to denote a splat.
template <typename T>
static Status ParseBufferElements(const std::string& value,
                                  iree_hal_element_type_t element_type,
                                  iree::span<T> buffer) {
  IREE_RETURN_IF_ERROR(
      iree_hal_parse_buffer_elements(
          iree_string_view_t{value.data(), (iree_host_size_t)value.size()},
          element_type,
          iree_byte_span_t{
              reinterpret_cast<uint8_t*>(buffer.data()),
              static_cast<iree_host_size_t>(buffer.size() * sizeof(T))}),
      "failed to parse buffer elements '%.*s'",
      iree_min(256, (int)value.size()), value.data());
  return OkStatus();
}

// Converts a shaped buffer of |element_type| elements to a string.
// This will include []'s to denote each dimension, for example for a shape of
// 2x3 the elements will be formatted as `[1 2 3][4 5 6]`.
//
// |max_element_count| can be used to limit the total number of elements printed
// when the count may be large. Elided elements will be replaced with `...`.
template <typename T>
static StatusOr<std::string> FormatBufferElements(
    iree::span<const T> data, const Shape& shape,
    iree_hal_element_type_t element_type, size_t max_element_count) {
  std::string result(255, '\0');
  iree_status_t status;
  do {
    iree_host_size_t actual_length = 0;
    status = iree_hal_format_buffer_elements(
        iree_const_byte_span_t{reinterpret_cast<const uint8_t*>(data.data()),
                               (iree_host_size_t)(data.size() * sizeof(T))},
        shape.size(), shape.data(), element_type, max_element_count,
        result.size() + 1, &result[0], &actual_length);
    result.resize(actual_length);
  } while (iree_status_is_out_of_range(status));
  IREE_RETURN_IF_ERROR(std::move(status));
  return std::move(result);
}

// Maps a C type (eg float) to the HAL type (eg IREE_HAL_ELEMENT_TYPE_FLOAT32).
template <typename T>
struct ElementTypeFromCType;

template <>
struct ElementTypeFromCType<bool> {
  static constexpr iree_hal_element_type_t value = IREE_HAL_ELEMENT_TYPE_BOOL_8;
};
template <>
struct ElementTypeFromCType<int8_t> {
  static constexpr iree_hal_element_type_t value = IREE_HAL_ELEMENT_TYPE_SINT_8;
};
template <>
struct ElementTypeFromCType<uint8_t> {
  static constexpr iree_hal_element_type_t value = IREE_HAL_ELEMENT_TYPE_UINT_8;
};
template <>
struct ElementTypeFromCType<int16_t> {
  static constexpr iree_hal_element_type_t value =
      IREE_HAL_ELEMENT_TYPE_SINT_16;
};
template <>
struct ElementTypeFromCType<uint16_t> {
  static constexpr iree_hal_element_type_t value =
      IREE_HAL_ELEMENT_TYPE_UINT_16;
};
template <>
struct ElementTypeFromCType<int32_t> {
  static constexpr iree_hal_element_type_t value =
      IREE_HAL_ELEMENT_TYPE_SINT_32;
};
template <>
struct ElementTypeFromCType<uint32_t> {
  static constexpr iree_hal_element_type_t value =
      IREE_HAL_ELEMENT_TYPE_UINT_32;
};
template <>
struct ElementTypeFromCType<int64_t> {
  static constexpr iree_hal_element_type_t value =
      IREE_HAL_ELEMENT_TYPE_SINT_64;
};
template <>
struct ElementTypeFromCType<uint64_t> {
  static constexpr iree_hal_element_type_t value =
      IREE_HAL_ELEMENT_TYPE_UINT_64;
};
template <>
struct ElementTypeFromCType<float> {
  static constexpr iree_hal_element_type_t value =
      IREE_HAL_ELEMENT_TYPE_FLOAT_32;
};
template <>
struct ElementTypeFromCType<double> {
  static constexpr iree_hal_element_type_t value =
      IREE_HAL_ELEMENT_TYPE_FLOAT_64;
};

// Parses a serialized element of type T to its in-memory form.
// For example, "1.2" of type float (IREE_HAL_ELEMENT_TYPE_FLOAT32) will return
// 1.2f.
template <typename T>
inline StatusOr<T> ParseElement(const std::string& value) {
  T result = T();
  IREE_RETURN_IF_ERROR(ParseElement(value, ElementTypeFromCType<T>::value,
                                    iree::span<T>(&result, 1)));
  return result;
}
inline StatusOr<uint16_t> ParseElementBF16(const std::string& value) {
  uint16_t result = uint16_t();
  IREE_RETURN_IF_ERROR(ParseElement(value, IREE_HAL_ELEMENT_TYPE_BFLOAT_16,
                                    iree::span<uint16_t>(&result, 1)));
  return result;
}

// Converts a single element of to a string value.
template <typename T>
inline StatusOr<std::string> FormatElement(T value) {
  return FormatElement(value, ElementTypeFromCType<T>::value);
}
inline StatusOr<std::string> FormatElementBF16(uint16_t value) {
  return FormatElement(value, IREE_HAL_ELEMENT_TYPE_BFLOAT_16);
}

// Parses a serialized set of elements of type T.
// The resulting parsed data is written to |buffer|, which must be at least
// large enough to contain the parsed elements. The format is the same as
// produced by FormatBufferElements. Supports additional inputs of
// empty to denote a 0 fill and a single element to denote a splat.
template <typename T>
inline Status ParseBufferElements(const std::string& value,
                                  iree::span<T> buffer) {
  return ParseBufferElements(value, ElementTypeFromCType<T>::value, buffer);
}

// Parses a serialized set of elements of type T defined by |shape|.
// The format is the same as produced by FormatBufferElements. Supports
// additional inputs of empty to denote a 0 fill and a single element to denote
// a splat.
template <typename T>
inline StatusOr<std::vector<T>> ParseBufferElements(const std::string& value,
                                                    const Shape& shape) {
  iree_host_size_t element_count = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    element_count *= shape[i];
  }
  std::vector<T> result(element_count);
  IREE_RETURN_IF_ERROR(ParseBufferElements(value, iree::span<T>(result)));
  return std::move(result);
}

// Converts a shaped buffer of |element_type| elements to a string.
// This will include []'s to denote each dimension, for example for a shape of
// 2x3 the elements will be formatted as `[1 2 3][4 5 6]`.
//
// |max_element_count| can be used to limit the total number of elements printed
// when the count may be large. Elided elements will be replaced with `...`.
template <typename T>
StatusOr<std::string> FormatBufferElements(
    iree::span<const T> data, const Shape& shape,
    size_t max_element_count = SIZE_MAX) {
  return FormatBufferElements(data, shape, ElementTypeFromCType<T>::value,
                              max_element_count);
}

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

// C++ wrapper for iree_hal_allocator_t.
struct Allocator final
    : public Handle<iree_hal_allocator_t, iree_hal_allocator_retain,
                    iree_hal_allocator_release> {
  using handle_type::handle_type;

  // Creates a host-local heap allocator that can be used when buffers are
  // required that will not interact with a real hardware device (such as those
  // used in file IO or tests). Buffers allocated with this will not be
  // compatible with real device allocators and will likely incur a copy if
  // used.
  static StatusOr<Allocator> CreateHostLocal() {
    Allocator allocator;
    iree_status_t status = iree_hal_allocator_create_heap(
        iree_make_cstring_view("host_local"), iree_allocator_system(),
        iree_allocator_system(), &allocator);
    IREE_RETURN_IF_ERROR(std::move(status));
    return std::move(allocator);
  }
};

// C++ wrapper for iree_hal_buffer_t.
struct Buffer final : public Handle<iree_hal_buffer_t, iree_hal_buffer_retain,
                                    iree_hal_buffer_release> {
  using handle_type::handle_type;

  // Returns the size in bytes of the buffer.
  iree_device_size_t byte_length() const noexcept {
    return iree_hal_buffer_byte_length(get());
  }

  // Returns a copy of the buffer contents interpreted as the given type in
  // host-format.
  template <typename T>
  StatusOr<std::vector<T>> CloneData() noexcept {
    iree_device_size_t total_byte_length = byte_length();
    std::vector<T> result(total_byte_length / sizeof(T));
    iree_status_t status =
        iree_hal_buffer_map_read(get(), 0, result.data(), total_byte_length);
    IREE_RETURN_IF_ERROR(std::move(status));
    return std::move(result);
  }
};

// C++ wrapper for iree_hal_buffer_view_t.
struct BufferView final
    : public Handle<iree_hal_buffer_view_t, iree_hal_buffer_view_retain,
                    iree_hal_buffer_view_release> {
  using handle_type::handle_type;

  // Creates a buffer view with a reference to the given |buffer|.
  static StatusOr<BufferView> Create(Buffer buffer,
                                     iree::span<const iree_hal_dim_t> shape,
                                     iree_hal_element_type_t element_type) {
    iree_hal_encoding_type_t encoding_type =
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR;
    BufferView buffer_view;
    iree_status_t status = iree_hal_buffer_view_create(
        buffer, shape.size(), shape.data(), element_type, encoding_type,
        iree_allocator_system(), &buffer_view);
    IREE_RETURN_IF_ERROR(std::move(status));
    return std::move(buffer_view);
  }

  // TODO(benvanik): subview.

  // Returns the buffer underlying the buffer view.
  inline Buffer buffer() const noexcept {
    return Buffer(iree_hal_buffer_view_buffer(get()));
  }

  // Returns the dimensions of the shape.
  Shape shape() const noexcept {
    iree_status_t status;
    Shape shape(6);
    do {
      iree_host_size_t actual_rank = 0;
      status = iree_hal_buffer_view_shape(get(), shape.size(), shape.data(),
                                          &actual_rank);
      shape.resize(actual_rank);
    } while (iree_status_is_out_of_range(status));
    IREE_CHECK_OK(status);
    return shape;
  }

  // Returns the total number of elements stored in the view.
  inline iree_host_size_t element_count() const noexcept {
    return iree_hal_buffer_view_element_count(get());
  }

  // Returns the element type of the buffer.
  inline iree_hal_element_type_t element_type() const noexcept {
    return iree_hal_buffer_view_element_type(get());
  }

  // Returns the total size of the specified view in bytes.
  // Note that not all buffers are contiguous or densely packed.
  inline iree_device_size_t byte_length() const noexcept {
    return iree_hal_buffer_view_byte_length(get());
  }

  // TODO(benvanik): compute offset/range.

  // Parses a serialized set of buffer elements in the canonical tensor format
  // (the same as produced by Format).
  static StatusOr<BufferView> Parse(const std::string& value,
                                    Allocator allocator) {
    BufferView buffer_view;
    iree_status_t status = iree_hal_buffer_view_parse(
        iree_string_view_t{value.data(), (iree_host_size_t)value.size()},
        /*device=*/NULL, allocator, &buffer_view);
    IREE_RETURN_IF_ERROR(std::move(status));
    return std::move(buffer_view);
  }

  // Converts buffer view elements into a fully-specified string-form format
  // like `2x4xi16=[[1 2][3 4]]`.
  //
  // |max_element_count| can be used to limit the total number of elements
  // printed when the count may be large. Elided elements will be replaced with
  // `...`.
  StatusOr<std::string> ToString(size_t max_element_count = SIZE_MAX) const {
    std::string result(255, '\0');
    iree_status_t status;
    do {
      iree_host_size_t actual_length = 0;
      status = iree_hal_buffer_view_format(get(), max_element_count,
                                           result.size() + 1, &result[0],
                                           &actual_length);
      result.resize(actual_length);
    } while (iree_status_is_out_of_range(status));
    IREE_RETURN_IF_ERROR(std::move(status));
    return std::move(result);
  }
};

TEST(ShapeStringUtilTest, ParseShape) {
  EXPECT_THAT(ParseShape(""), IsOkAndHolds(Eq(Shape{})));
  EXPECT_THAT(ParseShape("0"), IsOkAndHolds(Eq(Shape{0})));
  EXPECT_THAT(ParseShape("1"), IsOkAndHolds(Eq(Shape{1})));
  EXPECT_THAT(ParseShape("1x2"), IsOkAndHolds(Eq(Shape{1, 2})));
  EXPECT_THAT(ParseShape(" 1 x 2 "), IsOkAndHolds(Eq(Shape{1, 2})));
  EXPECT_THAT(ParseShape("1x2x3x4x5"), IsOkAndHolds(Eq(Shape{1, 2, 3, 4, 5})));
  EXPECT_THAT(ParseShape("1x2x3x4x5x6x7x8x9"),
              IsOkAndHolds(Eq(Shape{1, 2, 3, 4, 5, 6, 7, 8, 9})));
}

TEST(ShapeStringUtilTest, ParseShapeInvalid) {
  EXPECT_THAT(ParseShape("abc"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("1xf"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("1xff23"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("1xf32"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("x"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("x1"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("1x"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("x1x2"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("1xx2"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("1x2x"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("0x-1"), StatusIs(StatusCode::kInvalidArgument));
}

TEST(ShapeStringUtilTest, FormatShape) {
  EXPECT_THAT(FormatShape(Shape{}), IsOkAndHolds(Eq("")));
  EXPECT_THAT(FormatShape(Shape{0}), IsOkAndHolds(Eq("0")));
  EXPECT_THAT(FormatShape(Shape{1}), IsOkAndHolds(Eq("1")));
  EXPECT_THAT(FormatShape(Shape{1, 2}), IsOkAndHolds(Eq("1x2")));
  EXPECT_THAT(FormatShape(Shape{1, 2, 3, 4, 5}), IsOkAndHolds(Eq("1x2x3x4x5")));
  EXPECT_THAT(
      FormatShape(Shape{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        17, 18, 19}),
      IsOkAndHolds(Eq("1x2x3x4x5x6x7x8x9x10x11x12x13x14x15x16x17x18x19")));
}

TEST(ElementTypeStringUtilTest, ParseElementType) {
  EXPECT_THAT(ParseElementType("i1"),
              IsOkAndHolds(Eq(IREE_HAL_ELEMENT_TYPE_BOOL_8)));
  EXPECT_THAT(ParseElementType("i8"),
              IsOkAndHolds(Eq(IREE_HAL_ELEMENT_TYPE_INT_8)));
  EXPECT_THAT(ParseElementType("si8"),
              IsOkAndHolds(Eq(IREE_HAL_ELEMENT_TYPE_SINT_8)));
  EXPECT_THAT(ParseElementType("ui16"),
              IsOkAndHolds(Eq(IREE_HAL_ELEMENT_TYPE_UINT_16)));
  EXPECT_THAT(ParseElementType("f32"),
              IsOkAndHolds(Eq(IREE_HAL_ELEMENT_TYPE_FLOAT_32)));
  EXPECT_THAT(ParseElementType("f16"),
              IsOkAndHolds(Eq(IREE_HAL_ELEMENT_TYPE_FLOAT_16)));
  EXPECT_THAT(ParseElementType("bf16"),
              IsOkAndHolds(Eq(IREE_HAL_ELEMENT_TYPE_BFLOAT_16)));
  EXPECT_THAT(ParseElementType("x64"),
              IsOkAndHolds(Eq(IREE_HAL_ELEMENT_TYPE_OPAQUE_64)));
  EXPECT_THAT(ParseElementType("*64"),
              IsOkAndHolds(Eq(IREE_HAL_ELEMENT_TYPE_OPAQUE_64)));
  EXPECT_THAT(ParseElementType("f4"),
              IsOkAndHolds(Eq(iree_hal_make_element_type(
                  IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE, 4))));
}

TEST(ElementTypeStringUtilTest, ParseElementTypeInvalid) {
  EXPECT_THAT(ParseElementType(""), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElementType("1"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElementType("*1234"),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST(ElementTypeStringUtilTest, FormatElementType) {
  EXPECT_THAT(FormatElementType(IREE_HAL_ELEMENT_TYPE_BOOL_8),
              IsOkAndHolds(Eq("i1")));
  EXPECT_THAT(FormatElementType(IREE_HAL_ELEMENT_TYPE_INT_8),
              IsOkAndHolds(Eq("i8")));
  EXPECT_THAT(FormatElementType(IREE_HAL_ELEMENT_TYPE_SINT_8),
              IsOkAndHolds(Eq("si8")));
  EXPECT_THAT(FormatElementType(IREE_HAL_ELEMENT_TYPE_UINT_16),
              IsOkAndHolds(Eq("ui16")));
  EXPECT_THAT(FormatElementType(IREE_HAL_ELEMENT_TYPE_FLOAT_32),
              IsOkAndHolds(Eq("f32")));
  EXPECT_THAT(FormatElementType(IREE_HAL_ELEMENT_TYPE_BFLOAT_16),
              IsOkAndHolds(Eq("bf16")));
  EXPECT_THAT(FormatElementType(IREE_HAL_ELEMENT_TYPE_OPAQUE_64),
              IsOkAndHolds(Eq("*64")));
  EXPECT_THAT(FormatElementType(iree_hal_make_element_type(
                  IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE, 4)),
              IsOkAndHolds(Eq("f4")));
}

TEST(StringUtilTest, ParseShapeAndElementType) {
  EXPECT_THAT(
      ParseShapeAndElementType("1xi8"),
      IsOkAndHolds(Eq(ShapeAndType(Shape{1}, IREE_HAL_ELEMENT_TYPE_INT_8))));
  EXPECT_THAT(ParseShapeAndElementType("1x2xi16"),
              IsOkAndHolds(
                  Eq(ShapeAndType(Shape{1, 2}, IREE_HAL_ELEMENT_TYPE_INT_16))));
  EXPECT_THAT(
      ParseShapeAndElementType("1x2x3x4x5x6x7x8x9xi32=invalid stuff here"),
      IsOkAndHolds(Eq(ShapeAndType(Shape{1, 2, 3, 4, 5, 6, 7, 8, 9},
                                   IREE_HAL_ELEMENT_TYPE_INT_32))));
}

TEST(StringUtilTest, ParseShapeAndElementTypeInvalid) {
  EXPECT_THAT(ParseShapeAndElementType(""),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShapeAndElementType("0"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShapeAndElementType("="),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShapeAndElementType("abc"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShapeAndElementType("1xf"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShapeAndElementType("1xff23"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShapeAndElementType("1xn3"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShapeAndElementType("x"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShapeAndElementType("x1"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShapeAndElementType("1x"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShapeAndElementType("x1x2="),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShapeAndElementType("1xx2"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShapeAndElementType("1x2x"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShapeAndElementType("0x-1"),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST(ElementStringUtilTest, ParseElement) {
  EXPECT_THAT(ParseElement<int8_t>("-128"), IsOkAndHolds(Eq(INT8_MIN)));
  EXPECT_THAT(ParseElement<int8_t>("127"), IsOkAndHolds(Eq(INT8_MAX)));
  EXPECT_THAT(ParseElement<uint8_t>("255"), IsOkAndHolds(Eq(UINT8_MAX)));
  EXPECT_THAT(ParseElement<int16_t>("-32768"), IsOkAndHolds(Eq(INT16_MIN)));
  EXPECT_THAT(ParseElement<int16_t>("32767"), IsOkAndHolds(Eq(INT16_MAX)));
  EXPECT_THAT(ParseElement<uint16_t>("65535"), IsOkAndHolds(Eq(UINT16_MAX)));
  EXPECT_THAT(ParseElement<int32_t>("-2147483648"),
              IsOkAndHolds(Eq(INT32_MIN)));
  EXPECT_THAT(ParseElement<int32_t>("2147483647"), IsOkAndHolds(Eq(INT32_MAX)));
  EXPECT_THAT(ParseElement<uint32_t>("4294967295"),
              IsOkAndHolds(Eq(UINT32_MAX)));
  EXPECT_THAT(ParseElement<int64_t>("-9223372036854775808"),
              IsOkAndHolds(Eq(INT64_MIN)));
  EXPECT_THAT(ParseElement<int64_t>("9223372036854775807"),
              IsOkAndHolds(Eq(INT64_MAX)));
  EXPECT_THAT(ParseElement<uint64_t>("18446744073709551615"),
              IsOkAndHolds(Eq(UINT64_MAX)));
  EXPECT_THAT(ParseElementBF16("1.5"), IsOkAndHolds(Eq(0x3FC0u)));
  EXPECT_THAT(ParseElement<float>("1.5"), IsOkAndHolds(Eq(1.5f)));
  EXPECT_THAT(ParseElement<double>("1.567890123456789"),
              IsOkAndHolds(Eq(1.567890123456789)));
  EXPECT_THAT(ParseElement<double>("-1.5e-10"), IsOkAndHolds(Eq(-1.5e-10)));
}

TEST(ElementStringUtilTest, ParseElementOutOfRange) {
  EXPECT_THAT(ParseElement<int8_t>("255"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<uint8_t>("-128"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<int16_t>("65535"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<uint16_t>("-32768"),
              StatusIs(StatusCode::kInvalidArgument));
  // TODO(benvanik): these don't seem to work the same across all stdlib
  // implementations. The current implementation works with MSVC but fails under
  // clang. The fact that these failed like they did at all may have just been
  // an artifact of abseil and I'm not too concerned about matching that
  // behavior exactly enough to spend any more time on it now.
  // EXPECT_THAT(ParseElement<int32_t>("4294967295"),
  //             StatusIs(StatusCode::kInvalidArgument));
  // EXPECT_THAT(ParseElement<uint32_t>("4294967296"),
  //             StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<int32_t>("18446744073709551615"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<uint32_t>("-9223372036854775808"),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST(ElementStringUtilTest, ParseElementInvalid) {
  EXPECT_THAT(ParseElement<int8_t>(""), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<uint8_t>(""),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<int16_t>(""),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<uint16_t>(""),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<int32_t>(""),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<uint32_t>(""),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<int32_t>(""),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<uint32_t>(""),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<float>(""), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<double>(""), StatusIs(StatusCode::kInvalidArgument));

  EXPECT_THAT(ParseElement<int8_t>("asdfasdf"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<uint8_t>("asdfasdf"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<int16_t>("asdfasdf"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<uint16_t>("asdfasdf"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<int32_t>("asdfasdf"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<uint32_t>("asdfasdf"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<int32_t>("asdfasdf"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<uint32_t>("asdfasdf"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElementBF16("asdfasdf"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<float>("asdfasdf"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement<double>("asdfasdf"),
              StatusIs(StatusCode::kInvalidArgument));

  EXPECT_THAT(ParseElement<int8_t>("🌮"),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST(ElementStringUtilTest, ParseOpaqueElement) {
  std::vector<uint8_t> buffer1(1);
  IREE_EXPECT_OK(ParseElement("FF", IREE_HAL_ELEMENT_TYPE_OPAQUE_8,
                              iree::span<uint8_t>(buffer1)));
  EXPECT_THAT(buffer1, Eq(std::vector<uint8_t>{0xFF}));

  std::vector<uint16_t> buffer2(1);
  IREE_EXPECT_OK(ParseElement("FFCD", IREE_HAL_ELEMENT_TYPE_OPAQUE_16,
                              iree::span<uint16_t>(buffer2)));
  EXPECT_THAT(buffer2, Eq(std::vector<uint16_t>{0xCDFFu}));

  std::vector<uint32_t> buffer4(1);
  IREE_EXPECT_OK(ParseElement("FFCDAABB", IREE_HAL_ELEMENT_TYPE_OPAQUE_32,
                              iree::span<uint32_t>(buffer4)));
  EXPECT_THAT(buffer4, Eq(std::vector<uint32_t>{0xBBAACDFFu}));

  std::vector<uint64_t> buffer8(1);
  IREE_EXPECT_OK(ParseElement("FFCDAABBCCDDEEFF",
                              IREE_HAL_ELEMENT_TYPE_OPAQUE_64,
                              iree::span<uint64_t>(buffer8)));
  EXPECT_THAT(buffer8, Eq(std::vector<uint64_t>{0xFFEEDDCCBBAACDFFull}));
}

TEST(ElementStringUtilTest, ParseOpaqueElementInvalid) {
  std::vector<uint8_t> buffer0(0);
  EXPECT_THAT(ParseElement("", IREE_HAL_ELEMENT_TYPE_OPAQUE_8,
                           iree::span<uint8_t>(buffer0)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement("FF", IREE_HAL_ELEMENT_TYPE_OPAQUE_8,
                           iree::span<uint8_t>(buffer0)),
              StatusIs(StatusCode::kInvalidArgument));

  std::vector<uint8_t> buffer1(1);
  EXPECT_THAT(ParseElement("", IREE_HAL_ELEMENT_TYPE_OPAQUE_8,
                           iree::span<uint8_t>(buffer1)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement("F", IREE_HAL_ELEMENT_TYPE_OPAQUE_8,
                           iree::span<uint8_t>(buffer1)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseElement("FFC", IREE_HAL_ELEMENT_TYPE_OPAQUE_8,
                           iree::span<uint8_t>(buffer1)),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST(ElementStringUtilTest, FormatElement) {
  EXPECT_THAT(FormatElement<int8_t>(INT8_MIN), IsOkAndHolds(Eq("-128")));
  EXPECT_THAT(FormatElement<int8_t>(INT8_MAX), IsOkAndHolds(Eq("127")));
  EXPECT_THAT(FormatElement<uint8_t>(UINT8_MAX), IsOkAndHolds(Eq("255")));
  EXPECT_THAT(FormatElement<int16_t>(INT16_MIN), IsOkAndHolds(Eq("-32768")));
  EXPECT_THAT(FormatElement<int16_t>(INT16_MAX), IsOkAndHolds(Eq("32767")));
  EXPECT_THAT(FormatElement<uint16_t>(UINT16_MAX), IsOkAndHolds(Eq("65535")));
  EXPECT_THAT(FormatElement<int32_t>(INT32_MIN),
              IsOkAndHolds(Eq("-2147483648")));
  EXPECT_THAT(FormatElement<int32_t>(INT32_MAX),
              IsOkAndHolds(Eq("2147483647")));
  EXPECT_THAT(FormatElement<uint32_t>(UINT32_MAX),
              IsOkAndHolds(Eq("4294967295")));
  EXPECT_THAT(FormatElement<int64_t>(INT64_MIN),
              IsOkAndHolds(Eq("-9223372036854775808")));
  EXPECT_THAT(FormatElement<int64_t>(INT64_MAX),
              IsOkAndHolds(Eq("9223372036854775807")));
  EXPECT_THAT(FormatElement<uint64_t>(UINT64_MAX),
              IsOkAndHolds(Eq("18446744073709551615")));
  EXPECT_THAT(FormatElementBF16(0x3FC0u), IsOkAndHolds(Eq("1.5")));
  EXPECT_THAT(FormatElement<float>(1.5f), IsOkAndHolds(Eq("1.5")));
  EXPECT_THAT(FormatElement<double>(1123.56789456789),
              IsOkAndHolds(Eq("1123.57")));
  EXPECT_THAT(FormatElement<double>(-1.5e-10), IsOkAndHolds(Eq("-1.5E-10")));
}

TEST(ElementStringUtilTest, FormatOpaqueElement) {
  EXPECT_THAT(FormatElement<uint8_t>(129, IREE_HAL_ELEMENT_TYPE_OPAQUE_8),
              IsOkAndHolds(Eq("81")));
  EXPECT_THAT(FormatElement<int16_t>(-12345, IREE_HAL_ELEMENT_TYPE_OPAQUE_16),
              IsOkAndHolds(Eq("C7CF")));
  EXPECT_THAT(FormatElement<int32_t>(0, IREE_HAL_ELEMENT_TYPE_OPAQUE_32),
              IsOkAndHolds(Eq("00000000")));
  EXPECT_THAT(FormatElement<uint64_t>(0x8899AABBCCDDEEFFull,
                                      IREE_HAL_ELEMENT_TYPE_OPAQUE_64),
              IsOkAndHolds(Eq("FFEEDDCCBBAA9988")));
}

TEST(BufferElementsStringUtilTest, ParseBufferElements) {
  // Empty:
  std::vector<int8_t> buffer0(0);
  IREE_EXPECT_OK(ParseBufferElements<int8_t>("", iree::span<int8_t>(buffer0)));
  EXPECT_THAT(buffer0, Eq(std::vector<int8_t>{}));
  std::vector<int8_t> buffer8(8, 123);
  IREE_EXPECT_OK(ParseBufferElements<int8_t>("", iree::span<int8_t>(buffer8)));
  EXPECT_THAT(buffer8, Eq(std::vector<int8_t>{0, 0, 0, 0, 0, 0, 0, 0}));
  // Scalar:
  std::vector<int8_t> buffer1(1);
  IREE_EXPECT_OK(ParseBufferElements<int8_t>("1", iree::span<int8_t>(buffer1)));
  EXPECT_THAT(buffer1, Eq(std::vector<int8_t>{1}));
  // Splat:
  IREE_EXPECT_OK(ParseBufferElements<int8_t>("3", iree::span<int8_t>(buffer8)));
  EXPECT_THAT(buffer8, Eq(std::vector<int8_t>{3, 3, 3, 3, 3, 3, 3, 3}));
  // 1:1:
  IREE_EXPECT_OK(ParseBufferElements<int8_t>("2", iree::span<int8_t>(buffer1)));
  EXPECT_THAT(buffer1, Eq(std::vector<int8_t>{2}));
  std::vector<int16_t> buffer8i16(8);
  IREE_EXPECT_OK(ParseBufferElements<int16_t>("0 1 2 3 4 5 6 7",
                                              iree::span<int16_t>(buffer8i16)));
  EXPECT_THAT(buffer8i16, Eq(std::vector<int16_t>{0, 1, 2, 3, 4, 5, 6, 7}));
  std::vector<int32_t> buffer8i32(8);
  IREE_EXPECT_OK(ParseBufferElements<int32_t>("[0 1 2 3] [4 5 6 7]",
                                              iree::span<int32_t>(buffer8i32)));
  EXPECT_THAT(buffer8i32, Eq(std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST(BufferElementsStringUtilTest, ParseBufferElementsOpaque) {
  std::vector<uint16_t> buffer3i16(3);
  IREE_EXPECT_OK(ParseBufferElements("0011 2233 4455",
                                     IREE_HAL_ELEMENT_TYPE_OPAQUE_16,
                                     iree::span<uint16_t>(buffer3i16)));
  EXPECT_THAT(buffer3i16, Eq(std::vector<uint16_t>{0x1100, 0x3322, 0x5544}));
}

TEST(BufferElementsStringUtilTest, ParseBufferElementsInvalid) {
  std::vector<int8_t> buffer0(0);
  EXPECT_THAT(ParseBufferElements("abc", iree::span<int8_t>(buffer0)),
              StatusIs(StatusCode::kOutOfRange));
  std::vector<int8_t> buffer1(1);
  EXPECT_THAT(ParseBufferElements("abc", iree::span<int8_t>(buffer1)),
              StatusIs(StatusCode::kInvalidArgument));
  std::vector<int8_t> buffer8(8);
  EXPECT_THAT(ParseBufferElements("1 2 3", iree::span<int8_t>(buffer8)),
              StatusIs(StatusCode::kOutOfRange));
  std::vector<int8_t> buffer4(4);
  EXPECT_THAT(ParseBufferElements("1 2 3 4 5", iree::span<int8_t>(buffer4)),
              StatusIs(StatusCode::kOutOfRange));
}

TEST(BufferElementsStringUtilTest, ParseBufferElementsShaped) {
  // Empty:
  EXPECT_THAT(ParseBufferElements<int8_t>("", Shape{2, 4}),
              IsOkAndHolds(Eq(std::vector<int8_t>{0, 0, 0, 0, 0, 0, 0, 0})));
  // Scalar:
  EXPECT_THAT(ParseBufferElements<int8_t>("", Shape{}),
              IsOkAndHolds(Eq(std::vector<int8_t>{0})));
  EXPECT_THAT(ParseBufferElements<int8_t>("1", Shape{}),
              IsOkAndHolds(Eq(std::vector<int8_t>{1})));
  // Splat:
  EXPECT_THAT(ParseBufferElements<int8_t>("3", Shape{2, 4}),
              IsOkAndHolds(Eq(std::vector<int8_t>{3, 3, 3, 3, 3, 3, 3, 3})));
  // 1:1:
  EXPECT_THAT(ParseBufferElements<int8_t>("2", Shape{1}),
              IsOkAndHolds(Eq(std::vector<int8_t>{2})));
  EXPECT_THAT(ParseBufferElements<int16_t>("0 1 2 3 4 5 6 7", Shape{2, 4}),
              IsOkAndHolds(Eq(std::vector<int16_t>{0, 1, 2, 3, 4, 5, 6, 7})));
  EXPECT_THAT(ParseBufferElements<int32_t>("[0 1 2 3] [4 5 6 7]", Shape{2, 4}),
              IsOkAndHolds(Eq(std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7})));
}

TEST(BufferElementsStringUtilTest, ParseBufferElementsShapedInvalid) {
  EXPECT_THAT(ParseBufferElements<int8_t>("abc", Shape{}),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBufferElements<int8_t>("1 2 3", Shape{2, 4}),
              StatusIs(StatusCode::kOutOfRange));
  EXPECT_THAT(ParseBufferElements<int8_t>("1 2 3 4 5", Shape{2, 2}),
              StatusIs(StatusCode::kOutOfRange));
}

TEST(BufferElementsStringUtilTest, FormatBufferElements) {
  std::vector<int8_t> buffer = {1};
  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{}), IsOkAndHolds("1"));
  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{1}),
              IsOkAndHolds("1"));
  buffer = {1, 2, 3, 4};
  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{4}),
              IsOkAndHolds("1 2 3 4"));
  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{2, 2}),
              IsOkAndHolds("[1 2][3 4]"));
  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{4, 1}),
              IsOkAndHolds("[1][2][3][4]"));
  EXPECT_THAT(
      FormatBufferElements<int32_t>(std::vector<int32_t>(300, -99),
                                    Shape{100, 3}),
      IsOkAndHolds(
          "[-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99]"));
}

TEST(BufferElementsStringUtilTest, FormatBufferElementsElided) {
  std::vector<int8_t> buffer = {1};
  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{}, 0),
              IsOkAndHolds("..."));
  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{}, 1),
              IsOkAndHolds("1"));
  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{}, 99123),
              IsOkAndHolds("1"));
  buffer = {1, 2, 3, 4};
  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{4}, 0),
              IsOkAndHolds("..."));
  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{4}, 1),
              IsOkAndHolds("1..."));
  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{4}, 3),
              IsOkAndHolds("1 2 3..."));
  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{4}, 99123),
              IsOkAndHolds("1 2 3 4"));

  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{2, 2}, 0),
              IsOkAndHolds("[...][...]"));
  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{2, 2}, 1),
              IsOkAndHolds("[1...][...]"));
  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{2, 2}, 3),
              IsOkAndHolds("[1 2][3...]"));
  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{2, 2}, 99123),
              IsOkAndHolds("[1 2][3 4]"));
}

TEST(BufferViewStringUtilTest, Parse) {
  IREE_ASSERT_OK_AND_ASSIGN(auto allocator, Allocator::CreateHostLocal());

  // Zero fill.
  IREE_ASSERT_OK_AND_ASSIGN(auto bv0, BufferView::Parse("i8", allocator));
  EXPECT_THAT(bv0.buffer().CloneData<int8_t>(),
              IsOkAndHolds(Eq(std::vector<int8_t>{0})));

  // Zero fill (empty value).
  IREE_ASSERT_OK_AND_ASSIGN(auto bv1, BufferView::Parse("2x2xi8=", allocator));
  EXPECT_THAT(bv1.buffer().CloneData<int8_t>(),
              IsOkAndHolds(Eq(std::vector<int8_t>{0, 0, 0, 0})));

  // Splat.
  IREE_ASSERT_OK_AND_ASSIGN(auto bv2, BufferView::Parse("2x2xi8=3", allocator));
  EXPECT_THAT(bv2.buffer().CloneData<int8_t>(),
              IsOkAndHolds(Eq(std::vector<int8_t>{3, 3, 3, 3})));

  // Flat list.
  IREE_ASSERT_OK_AND_ASSIGN(auto bv3,
                            BufferView::Parse("2x2xi8=1 2 3 4", allocator));
  EXPECT_THAT(bv3.buffer().CloneData<int8_t>(),
              IsOkAndHolds(Eq(std::vector<int8_t>{1, 2, 3, 4})));

  // Whitespace and separators shouldn't matter.
  IREE_ASSERT_OK_AND_ASSIGN(
      auto bv4, BufferView::Parse("  2x2xi8 =  1,\n2 3\t,4", allocator));
  EXPECT_THAT(bv4.buffer().CloneData<int8_t>(),
              IsOkAndHolds(Eq(std::vector<int8_t>{1, 2, 3, 4})));

  // Brackets are optional.
  IREE_ASSERT_OK_AND_ASSIGN(
      auto bv5, BufferView::Parse("4xi16=[[0][1][2]][3]", allocator));
  EXPECT_THAT(bv5.buffer().CloneData<int16_t>(),
              IsOkAndHolds(Eq(std::vector<int16_t>{0, 1, 2, 3})));
}

TEST(BufferViewStringUtilTest, ParseInvalid) {
  IREE_ASSERT_OK_AND_ASSIGN(auto allocator, Allocator::CreateHostLocal());

  // Incomplete.
  EXPECT_THAT(BufferView::Parse("", allocator),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(BufferView::Parse("asdf", allocator),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(BufferView::Parse("9x8=", allocator),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(BufferView::Parse("=4", allocator),
              StatusIs(StatusCode::kInvalidArgument));

  // Partial data.
  EXPECT_THAT(BufferView::Parse("2x4xi32=5 3", allocator),
              StatusIs(StatusCode::kOutOfRange));
}

TEST(BufferViewStringUtilTest, ToString) {
  std::vector<int8_t> buffer = {1};
  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{}), IsOkAndHolds("1"));
  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{1}),
              IsOkAndHolds("1"));
  buffer = {1, 2, 3, 4};
  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{4}),
              IsOkAndHolds("1 2 3 4"));
  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{2, 2}),
              IsOkAndHolds("[1 2][3 4]"));
  EXPECT_THAT(FormatBufferElements<int8_t>(buffer, Shape{4, 1}),
              IsOkAndHolds("[1][2][3][4]"));
  EXPECT_THAT(
      FormatBufferElements<int32_t>(std::vector<int32_t>(300, -99),
                                    Shape{100, 3}),
      IsOkAndHolds(
          "[-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
          "-99]"));
}

TEST(BufferViewStringUtilTest, RoundTrip) {
  IREE_ASSERT_OK_AND_ASSIGN(auto allocator, Allocator::CreateHostLocal());
  auto expect_round_trip = [&](std::string source_value) {
    IREE_ASSERT_OK_AND_ASSIGN(auto buffer_view,
                              BufferView::Parse(source_value, allocator));
    EXPECT_THAT(buffer_view.ToString(), IsOkAndHolds(source_value));
  };

  expect_round_trip("i8=-8");
  expect_round_trip("si8=-8");
  expect_round_trip("ui8=239");
  expect_round_trip("4xi8=0 -1 2 3");
  expect_round_trip("4xsi8=0 -1 2 3");
  expect_round_trip("4xi16=0 -1 2 3");
  expect_round_trip("4xui16=0 1 2 3");
  expect_round_trip("2x2xi32=[0 1][2 3]");
  expect_round_trip("4xf16=0 0.5 2 3");
  expect_round_trip("4xf32=0 1.1 2 3");
  expect_round_trip("4xf64=0 1.1 2 3");
  expect_round_trip("1x2x3xi8=[[0 1 2][3 4 5]]");
  expect_round_trip("2x*16=AABB CCDD");
  expect_round_trip(
      "100x3xi16=[-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
      "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
      "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
      "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
      "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
      "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
      "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
      "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
      "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
      "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
      "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
      "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
      "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
      "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
      "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
      "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
      "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
      "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
      "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 "
      "-99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99][-99 -99 -99]");
}

}  // namespace
}  // namespace hal
}  // namespace iree
