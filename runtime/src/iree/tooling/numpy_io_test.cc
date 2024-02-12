// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/numpy_io.h"

#include "iree/io/memory_stream.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/testdata/npy/npy_files.h"

namespace iree {
namespace {

using iree::testing::status::IsOk;
using iree::testing::status::StatusIs;
using ::testing::ElementsAreArray;

using StreamPtr =
    std::unique_ptr<iree_io_stream_t, void (*)(iree_io_stream_t*)>;

class NumpyIOTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    iree_status_t status = iree_hal_create_device(
        iree_hal_available_driver_registry(), IREE_SV("local-sync"),
        iree_allocator_system(), &device_);
    if (iree_status_is_not_found(status)) {
      fprintf(stderr, "Skipping test as 'local-sync' driver was not found:\n");
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP();
    }
    device_allocator_ = iree_hal_device_allocator(device_);
  }

  virtual void TearDown() { iree_hal_device_release(device_); }

  StreamPtr OpenInputFile(const char* name) {
    const struct iree_file_toc_t* file_toc = iree_numpy_npy_files_create();
    for (size_t i = 0; i < iree_numpy_npy_files_size(); ++i) {
      if (strcmp(file_toc[i].name, name) != 0) continue;
      iree_io_stream_t* stream = NULL;
      IREE_CHECK_OK(iree_io_memory_stream_wrap(
          IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_SEEKABLE,
          iree_make_byte_span((void*)file_toc[i].data, file_toc[i].size),
          iree_io_memory_stream_release_callback_null(),
          iree_allocator_system(), &stream));
      return StreamPtr(stream, iree_io_stream_release);
    }
    return StreamPtr{nullptr, iree_io_stream_release};
  }

  StreamPtr OpenOutputFile(const char* name) {
    iree_io_stream_t* stream = NULL;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
            IREE_IO_STREAM_MODE_SEEKABLE,
        // /*block_size=*/32 * 1024,
        /*block_size=*/64, iree_allocator_system(), &stream));
    return StreamPtr(stream, iree_io_stream_release);
  }

  iree_hal_device_t* device_ = nullptr;
  iree_hal_allocator_t* device_allocator_ = nullptr;
};

template <typename T>
static void AssertBufferViewContents(iree_hal_buffer_view_t* buffer_view,
                                     std::vector<iree_hal_dim_t> shape,
                                     iree_hal_element_type_t element_type,
                                     iree_hal_encoding_type_t encoding_type,
                                     std::vector<T> expected_contents) {
  ASSERT_EQ(iree_hal_buffer_view_shape_rank(buffer_view), shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(iree_hal_buffer_view_shape_dim(buffer_view, i), shape[i]);
  }
  ASSERT_EQ(iree_hal_buffer_view_element_type(buffer_view), element_type);
  ASSERT_EQ(iree_hal_buffer_view_encoding_type(buffer_view), encoding_type);

  iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(buffer_view);
  ASSERT_EQ(iree_hal_buffer_byte_length(buffer),
            expected_contents.size() * sizeof(T));

  std::vector<T> actual_contents;
  actual_contents.resize(expected_contents.size());
  IREE_ASSERT_OK(iree_hal_buffer_map_read(buffer, 0, actual_contents.data(),
                                          actual_contents.size() * sizeof(T)));

  ASSERT_THAT(actual_contents, ElementsAreArray(expected_contents));
}

template <typename T>
static void LoadArrayAndAssertContents(iree_io_stream_t* stream,
                                       iree_hal_device_t* device,
                                       iree_hal_allocator_t* device_allocator,
                                       std::vector<iree_hal_dim_t> shape,
                                       iree_hal_element_type_t element_type,
                                       iree_hal_encoding_type_t encoding_type,
                                       std::vector<T> contents) {
  iree_hal_buffer_params_t buffer_params = {};
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_READ;
  buffer_params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_ASSERT_OK(iree_numpy_npy_load_ndarray(
      stream, IREE_NUMPY_NPY_LOAD_OPTION_DEFAULT, buffer_params, device,
      device_allocator, &buffer_view));
  AssertBufferViewContents<T>(buffer_view, shape, element_type, encoding_type,
                              contents);
  iree_hal_buffer_view_release(buffer_view);
}

// Tests that an empty file returns EOF.
TEST_F(NumpyIOTest, LoadEmptyFile) {
  auto stream = OpenInputFile("empty.npy");

  // Should start at EOF - the file is empty.
  ASSERT_TRUE(iree_io_stream_is_eos(stream.get()));

  // Try (and fail) to parse something from the empty file.
  iree_hal_buffer_params_t buffer_params = {};
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_READ;
  buffer_params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  iree_hal_buffer_view_t* buffer_view = NULL;
  EXPECT_THAT(Status(iree_numpy_npy_load_ndarray(
                  stream.get(), IREE_NUMPY_NPY_LOAD_OPTION_DEFAULT,
                  buffer_params, device_, device_allocator_, &buffer_view)),
              StatusIs(StatusCode::kOutOfRange));

  // Should still be at EOF.
  ASSERT_TRUE(iree_io_stream_is_eos(stream.get()));
}

// Tests loading a single array from a file.
TEST_F(NumpyIOTest, LoadSingleArray) {
  auto stream = OpenInputFile("single.npy");

  // np.array([1.1, 2.2, 3.3], dtype=np.float32)
  LoadArrayAndAssertContents<float>(stream.get(), device_, device_allocator_,
                                    {3}, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                    IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                    {1.1f, 2.2f, 3.3f});

  // Should have hit EOF.
  ASSERT_TRUE(iree_io_stream_is_eos(stream.get()));
}

// Tests loading multiple arrays from a concatenated file.
TEST_F(NumpyIOTest, LoadMultipleArrays) {
  auto stream = OpenInputFile("multiple.npy");

  // np.array([1.1, 2.2, 3.3], dtype=np.float32)
  LoadArrayAndAssertContents<float>(stream.get(), device_, device_allocator_,
                                    {3}, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                    IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                    {1.1f, 2.2f, 3.3f});

  // np.array([[0, 1], [2, 3]], dtype=np.int32)
  LoadArrayAndAssertContents<int32_t>(stream.get(), device_, device_allocator_,
                                      {2, 2}, IREE_HAL_ELEMENT_TYPE_SINT_32,
                                      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                      {0, 1, 2, 3});

  // np.array(42, dtype=np.int32)
  LoadArrayAndAssertContents<int32_t>(stream.get(), device_, device_allocator_,
                                      {}, IREE_HAL_ELEMENT_TYPE_SINT_32,
                                      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                      {42});

  // Should have hit EOF.
  ASSERT_TRUE(iree_io_stream_is_eos(stream.get()));
}

// Tests loading arrays with various shapes.
TEST_F(NumpyIOTest, ArrayShapes) {
  auto stream = OpenInputFile("array_shapes.npy");

  // np.array(1, dtype=np.int8)
  LoadArrayAndAssertContents<int8_t>(stream.get(), device_, device_allocator_,
                                     {}, IREE_HAL_ELEMENT_TYPE_SINT_8,
                                     IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                     {1});

  // np.array([], dtype=np.int8)
  LoadArrayAndAssertContents<int8_t>(
      stream.get(), device_, device_allocator_, {0},
      IREE_HAL_ELEMENT_TYPE_SINT_8, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {});

  // np.array([1], dtype=np.int8)
  LoadArrayAndAssertContents<int8_t>(stream.get(), device_, device_allocator_,
                                     {1}, IREE_HAL_ELEMENT_TYPE_SINT_8,
                                     IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                     {1});

  // np.array([[1], [2]], dtype=np.int8)
  LoadArrayAndAssertContents<int8_t>(stream.get(), device_, device_allocator_,
                                     {2, 1}, IREE_HAL_ELEMENT_TYPE_SINT_8,
                                     IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                     {1, 2});

  // np.array([[0], [1], [2], [3], [4], [5], [6], [7]], dtype=np.int8)
  LoadArrayAndAssertContents<int8_t>(stream.get(), device_, device_allocator_,
                                     {8, 1}, IREE_HAL_ELEMENT_TYPE_SINT_8,
                                     IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                     {0, 1, 2, 3, 4, 5, 6, 7});

  // np.array([[1, 2], [3, 4]], dtype=np.int8)
  LoadArrayAndAssertContents<int8_t>(stream.get(), device_, device_allocator_,
                                     {2, 2}, IREE_HAL_ELEMENT_TYPE_SINT_8,
                                     IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                     {1, 2, 3, 4});

  // np.array([[[1], [2]], [[3], [4]]], dtype=np.int8)
  LoadArrayAndAssertContents<int8_t>(stream.get(), device_, device_allocator_,
                                     {2, 2, 1}, IREE_HAL_ELEMENT_TYPE_SINT_8,
                                     IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                     {1, 2, 3, 4});

  // Should have hit EOF.
  ASSERT_TRUE(iree_io_stream_is_eos(stream.get()));
}

// Tests loading arrays with various element types.
TEST_F(NumpyIOTest, ArrayTypes) {
  auto stream = OpenInputFile("array_types.npy");

  // np.array([True, False], dtype=np.bool_)
  LoadArrayAndAssertContents<int8_t>(stream.get(), device_, device_allocator_,
                                     {2}, IREE_HAL_ELEMENT_TYPE_BOOL_8,
                                     IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                     {1, 0});

  // np.array([-1, 1], dtype=np.int8)
  LoadArrayAndAssertContents<int8_t>(stream.get(), device_, device_allocator_,
                                     {2}, IREE_HAL_ELEMENT_TYPE_SINT_8,
                                     IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                     {-1, 1});

  // np.array([-20000, 20000], dtype=np.int16)
  LoadArrayAndAssertContents<int16_t>(stream.get(), device_, device_allocator_,
                                      {2}, IREE_HAL_ELEMENT_TYPE_SINT_16,
                                      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                      {-20000, 20000});

  // np.array([-2000000, 2000000], dtype=np.int32)
  LoadArrayAndAssertContents<int32_t>(stream.get(), device_, device_allocator_,
                                      {2}, IREE_HAL_ELEMENT_TYPE_SINT_32,
                                      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                      {-2000000, 2000000});

  // np.array([-20000000000, 20000000000], dtype=np.int64)
  LoadArrayAndAssertContents<int64_t>(stream.get(), device_, device_allocator_,
                                      {2}, IREE_HAL_ELEMENT_TYPE_SINT_64,
                                      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                      {-20000000000, 20000000000});

  // np.array([1, 255], dtype=np.uint8)
  LoadArrayAndAssertContents<uint8_t>(stream.get(), device_, device_allocator_,
                                      {2}, IREE_HAL_ELEMENT_TYPE_UINT_8,
                                      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                      {1, 255});

  // np.array([1, 65535], dtype=np.uint16)
  LoadArrayAndAssertContents<uint16_t>(stream.get(), device_, device_allocator_,
                                       {2}, IREE_HAL_ELEMENT_TYPE_UINT_16,
                                       IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                       {1, 65535});

  // np.array([1, 4294967295], dtype=np.uint32)
  LoadArrayAndAssertContents<uint32_t>(stream.get(), device_, device_allocator_,
                                       {2}, IREE_HAL_ELEMENT_TYPE_UINT_32,
                                       IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                       {1, 4294967295u});

  // np.array([1, 18446744073709551615], dtype=np.uint64)
  LoadArrayAndAssertContents<uint64_t>(stream.get(), device_, device_allocator_,
                                       {2}, IREE_HAL_ELEMENT_TYPE_UINT_64,
                                       IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                       {1, 18446744073709551615ull});

  // np.array([-1.1, 1.1], dtype=np.float16)
  LoadArrayAndAssertContents<uint16_t>(stream.get(), device_, device_allocator_,
                                       {2}, IREE_HAL_ELEMENT_TYPE_FLOAT_16,
                                       IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                       {0xBC66, 0x3C66});

  // np.array([-1.1, 1.1], dtype=np.float32)
  LoadArrayAndAssertContents<float>(stream.get(), device_, device_allocator_,
                                    {2}, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                    IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                    {-1.1f, 1.1f});

  // np.array([-1.1, 1.1], dtype=np.float64)
  LoadArrayAndAssertContents<double>(stream.get(), device_, device_allocator_,
                                     {2}, IREE_HAL_ELEMENT_TYPE_FLOAT_64,
                                     IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                     {-1.1, 1.1});

  // np.array([1 + 5j, 2 + 6j], dtype=np.complex64)
  LoadArrayAndAssertContents<float>(stream.get(), device_, device_allocator_,
                                    {2}, IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64,
                                    IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                    {1.0f, 5.0f, 2.0f, 6.0f});

  // np.array([-1.1, 1.1], dtype=np.float64)
  LoadArrayAndAssertContents<double>(
      stream.get(), device_, device_allocator_, {2},
      IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {1.0, 5.0, 2.0, 6.0});

  // Should have hit EOF.
  ASSERT_TRUE(iree_io_stream_is_eos(stream.get()));
}

static void RoundTripArrays(iree_io_stream_t* source_stream,
                            iree_io_stream_t* target_stream,
                            iree_hal_device_t* device,
                            iree_hal_allocator_t* device_allocator) {
  while (!iree_io_stream_is_eos(source_stream)) {
    iree_hal_buffer_params_t buffer_params = {};
    buffer_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
    buffer_params.access = IREE_HAL_MEMORY_ACCESS_READ;
    buffer_params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
    iree_hal_buffer_view_t* buffer_view = NULL;
    IREE_ASSERT_OK(iree_numpy_npy_load_ndarray(
        source_stream, IREE_NUMPY_NPY_LOAD_OPTION_DEFAULT, buffer_params,
        device, device_allocator, &buffer_view));
    IREE_ASSERT_OK(iree_numpy_npy_save_ndarray(
        target_stream, IREE_NUMPY_NPY_SAVE_OPTION_DEFAULT, buffer_view,
        iree_hal_allocator_host_allocator(device_allocator)));
    iree_hal_buffer_view_release(buffer_view);
  }
}

static void CompareStreams(iree_io_stream_t* source_stream,
                           iree_io_stream_t* target_stream) {
  iree_io_stream_pos_t source_size = iree_io_stream_length(source_stream);
  iree_io_stream_pos_t target_size = iree_io_stream_length(target_stream);
  ASSERT_EQ(source_size, target_size) << "streams should have the same length";

  IREE_ASSERT_OK(
      iree_io_stream_seek(source_stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(
      iree_io_stream_seek(target_stream, IREE_IO_STREAM_SEEK_SET, 0));

  std::vector<uint8_t> source_data;
  source_data.resize(source_size);
  std::vector<uint8_t> target_data;
  target_data.resize(target_size);
  IREE_ASSERT_OK(iree_io_stream_read(source_stream, source_data.size(),
                                     source_data.data(), NULL));
  IREE_ASSERT_OK(iree_io_stream_read(target_stream, target_data.size(),
                                     target_data.data(), NULL));
  ASSERT_THAT(target_data, ElementsAreArray(source_data));

  ASSERT_EQ(iree_io_stream_is_eos(source_stream),
            iree_io_stream_is_eos(target_stream))
      << "streams should have the same length";
}

// Tests round-tripping a single array.
TEST_F(NumpyIOTest, RoundTripSingleArray) {
  auto source_stream = OpenInputFile("single.npy");
  auto target_stream = OpenOutputFile("single_out.npy");
  RoundTripArrays(source_stream.get(), target_stream.get(), device_,
                  device_allocator_);
  CompareStreams(source_stream.get(), target_stream.get());
}

// Tests round-tripping multiple array.
TEST_F(NumpyIOTest, RoundTripMultipleArrays) {
  auto source_stream = OpenInputFile("multiple.npy");
  auto target_stream = OpenOutputFile("multiple_out.npy");
  RoundTripArrays(source_stream.get(), target_stream.get(), device_,
                  device_allocator_);
  CompareStreams(source_stream.get(), target_stream.get());
}

// Tests round-tripping arrays with various shapes.
TEST_F(NumpyIOTest, RoundTripArrayShapes) {
  auto source_stream = OpenInputFile("array_shapes.npy");
  auto target_stream = OpenOutputFile("array_shapes_out.npy");
  RoundTripArrays(source_stream.get(), target_stream.get(), device_,
                  device_allocator_);
  CompareStreams(source_stream.get(), target_stream.get());
}

// Tests round-tripping arrays with various types.
TEST_F(NumpyIOTest, RoundTripArrayTypes) {
  auto source_stream = OpenInputFile("array_types.npy");
  auto target_stream = OpenOutputFile("array_types_out.npy");
  RoundTripArrays(source_stream.get(), target_stream.get(), device_,
                  device_allocator_);
  CompareStreams(source_stream.get(), target_stream.get());
}

}  // namespace
}  // namespace iree
