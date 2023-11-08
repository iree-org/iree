// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/numpy_io.h"

#include "iree/base/internal/file_io.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/testdata/npy/npy_files.h"

namespace iree {
namespace {

using iree::testing::status::IsOk;
using iree::testing::status::StatusIs;
using ::testing::ElementsAreArray;

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.ToString();
  return os;
}

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

  static std::string GetTempFilename(const char* suffix) {
    static int unique_id = 0;
    char* test_tmpdir = getenv("TEST_TMPDIR");
    if (!test_tmpdir) {
      test_tmpdir = getenv("TMPDIR");
    }
    if (!test_tmpdir) {
      test_tmpdir = getenv("TEMP");
    }
    if (!test_tmpdir) {
      std::cerr << "TEST_TMPDIR/TMPDIR/TEMP not defined\n";
      exit(1);
    }
    return test_tmpdir + std::string("/iree_test_") +
           std::to_string(unique_id++) + '_' + suffix;
  }

  FILE* OpenInputFile(const char* name) {
    const struct iree_file_toc_t* file_toc = iree_numpy_npy_files_create();
    for (size_t i = 0; i < iree_numpy_npy_files_size(); ++i) {
      if (strcmp(file_toc[i].name, name) != 0) continue;
      auto file_path = GetTempFilename(name);
      IREE_CHECK_OK(iree_file_write_contents(
          file_path.c_str(),
          iree_make_const_byte_span(file_toc[i].data, file_toc[i].size)));
      return fopen(file_path.c_str(), "rb");
    }
    return NULL;
  }

  FILE* OpenOutputFile(const char* name) {
    auto file_path = GetTempFilename(name);
    return fopen(file_path.c_str(), "w+b");
  }

  iree_hal_device_t* device_ = nullptr;
  iree_hal_allocator_t* device_allocator_ = nullptr;
};

static bool IsEOF(FILE* stream) {
  long original_pos = ftell(stream);
  fseek(stream, 0, SEEK_END);
  long end_pos = ftell(stream);
  fseek(stream, original_pos, SEEK_SET);
  return original_pos == end_pos;
}

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
static void LoadArrayAndAssertContents(FILE* stream, iree_hal_device_t* device,
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
  FILE* stream = OpenInputFile("empty.npy");

  // Should start at EOF - the file is empty.
  ASSERT_TRUE(IsEOF(stream));

  // Try (and fail) to parse something from the empty file.
  iree_hal_buffer_params_t buffer_params = {};
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_READ;
  buffer_params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  iree_hal_buffer_view_t* buffer_view = NULL;
  EXPECT_THAT(Status(iree_numpy_npy_load_ndarray(
                  stream, IREE_NUMPY_NPY_LOAD_OPTION_DEFAULT, buffer_params,
                  device_, device_allocator_, &buffer_view)),
              StatusIs(StatusCode::kResourceExhausted));

  // Should still be at EOF.
  ASSERT_TRUE(IsEOF(stream));
  fclose(stream);
}

// Tests loading a single array from a file.
TEST_F(NumpyIOTest, LoadSingleArray) {
  FILE* stream = OpenInputFile("single.npy");

  // np.array([1.1, 2.2, 3.3], dtype=np.float32)
  LoadArrayAndAssertContents<float>(
      stream, device_, device_allocator_, {3}, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {1.1f, 2.2f, 3.3f});

  // Should have hit EOF.
  ASSERT_TRUE(IsEOF(stream));
  fclose(stream);
}

// Tests loading multiple arrays from a concatenated file.
TEST_F(NumpyIOTest, LoadMultipleArrays) {
  FILE* stream = OpenInputFile("multiple.npy");

  // np.array([1.1, 2.2, 3.3], dtype=np.float32)
  LoadArrayAndAssertContents<float>(
      stream, device_, device_allocator_, {3}, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {1.1f, 2.2f, 3.3f});

  // np.array([[0, 1], [2, 3]], dtype=np.int32)
  LoadArrayAndAssertContents<int32_t>(
      stream, device_, device_allocator_, {2, 2}, IREE_HAL_ELEMENT_TYPE_SINT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {0, 1, 2, 3});

  // np.array(42, dtype=np.int32)
  LoadArrayAndAssertContents<int32_t>(
      stream, device_, device_allocator_, {}, IREE_HAL_ELEMENT_TYPE_SINT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {42});

  // Should have hit EOF.
  ASSERT_TRUE(IsEOF(stream));
  fclose(stream);
}

// Tests loading arrays with various shapes.
TEST_F(NumpyIOTest, ArrayShapes) {
  FILE* stream = OpenInputFile("array_shapes.npy");

  // np.array(1, dtype=np.int8)
  LoadArrayAndAssertContents<int8_t>(
      stream, device_, device_allocator_, {}, IREE_HAL_ELEMENT_TYPE_SINT_8,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {1});

  // np.array([], dtype=np.int8)
  LoadArrayAndAssertContents<int8_t>(
      stream, device_, device_allocator_, {0}, IREE_HAL_ELEMENT_TYPE_SINT_8,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {});

  // np.array([1], dtype=np.int8)
  LoadArrayAndAssertContents<int8_t>(
      stream, device_, device_allocator_, {1}, IREE_HAL_ELEMENT_TYPE_SINT_8,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {1});

  // np.array([[1], [2]], dtype=np.int8)
  LoadArrayAndAssertContents<int8_t>(
      stream, device_, device_allocator_, {2, 1}, IREE_HAL_ELEMENT_TYPE_SINT_8,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {1, 2});

  // np.array([[0], [1], [2], [3], [4], [5], [6], [7]], dtype=np.int8)
  LoadArrayAndAssertContents<int8_t>(
      stream, device_, device_allocator_, {8, 1}, IREE_HAL_ELEMENT_TYPE_SINT_8,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {0, 1, 2, 3, 4, 5, 6, 7});

  // np.array([[1, 2], [3, 4]], dtype=np.int8)
  LoadArrayAndAssertContents<int8_t>(
      stream, device_, device_allocator_, {2, 2}, IREE_HAL_ELEMENT_TYPE_SINT_8,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {1, 2, 3, 4});

  // np.array([[[1], [2]], [[3], [4]]], dtype=np.int8)
  LoadArrayAndAssertContents<int8_t>(stream, device_, device_allocator_,
                                     {2, 2, 1}, IREE_HAL_ELEMENT_TYPE_SINT_8,
                                     IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                     {1, 2, 3, 4});

  // Should have hit EOF.
  ASSERT_TRUE(IsEOF(stream));
  fclose(stream);
}

// Tests loading arrays with various element types.
TEST_F(NumpyIOTest, ArrayTypes) {
  FILE* stream = OpenInputFile("array_types.npy");

  // np.array([True, False], dtype=np.bool_)
  LoadArrayAndAssertContents<int8_t>(
      stream, device_, device_allocator_, {2}, IREE_HAL_ELEMENT_TYPE_BOOL_8,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {1, 0});

  // np.array([-1, 1], dtype=np.int8)
  LoadArrayAndAssertContents<int8_t>(
      stream, device_, device_allocator_, {2}, IREE_HAL_ELEMENT_TYPE_SINT_8,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {-1, 1});

  // np.array([-20000, 20000], dtype=np.int16)
  LoadArrayAndAssertContents<int16_t>(
      stream, device_, device_allocator_, {2}, IREE_HAL_ELEMENT_TYPE_SINT_16,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {-20000, 20000});

  // np.array([-2000000, 2000000], dtype=np.int32)
  LoadArrayAndAssertContents<int32_t>(
      stream, device_, device_allocator_, {2}, IREE_HAL_ELEMENT_TYPE_SINT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {-2000000, 2000000});

  // np.array([-20000000000, 20000000000], dtype=np.int64)
  LoadArrayAndAssertContents<int64_t>(
      stream, device_, device_allocator_, {2}, IREE_HAL_ELEMENT_TYPE_SINT_64,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {-20000000000, 20000000000});

  // np.array([1, 255], dtype=np.uint8)
  LoadArrayAndAssertContents<uint8_t>(
      stream, device_, device_allocator_, {2}, IREE_HAL_ELEMENT_TYPE_UINT_8,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {1, 255});

  // np.array([1, 65535], dtype=np.uint16)
  LoadArrayAndAssertContents<uint16_t>(
      stream, device_, device_allocator_, {2}, IREE_HAL_ELEMENT_TYPE_UINT_16,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {1, 65535});

  // np.array([1, 4294967295], dtype=np.uint32)
  LoadArrayAndAssertContents<uint32_t>(
      stream, device_, device_allocator_, {2}, IREE_HAL_ELEMENT_TYPE_UINT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {1, 4294967295u});

  // np.array([1, 18446744073709551615], dtype=np.uint64)
  LoadArrayAndAssertContents<uint64_t>(
      stream, device_, device_allocator_, {2}, IREE_HAL_ELEMENT_TYPE_UINT_64,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {1, 18446744073709551615ull});

  // np.array([-1.1, 1.1], dtype=np.float16)
  LoadArrayAndAssertContents<uint16_t>(
      stream, device_, device_allocator_, {2}, IREE_HAL_ELEMENT_TYPE_FLOAT_16,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {0xBC66, 0x3C66});

  // np.array([-1.1, 1.1], dtype=np.float32)
  LoadArrayAndAssertContents<float>(
      stream, device_, device_allocator_, {2}, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {-1.1f, 1.1f});

  // np.array([-1.1, 1.1], dtype=np.float64)
  LoadArrayAndAssertContents<double>(
      stream, device_, device_allocator_, {2}, IREE_HAL_ELEMENT_TYPE_FLOAT_64,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, {-1.1, 1.1});

  // np.array([1 + 5j, 2 + 6j], dtype=np.complex64)
  LoadArrayAndAssertContents<float>(stream, device_, device_allocator_, {2},
                                    IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64,
                                    IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                    {1.0f, 5.0f, 2.0f, 6.0f});

  // np.array([-1.1, 1.1], dtype=np.float64)
  LoadArrayAndAssertContents<double>(stream, device_, device_allocator_, {2},
                                     IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128,
                                     IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                     {1.0, 5.0, 2.0, 6.0});

  // Should have hit EOF.
  ASSERT_TRUE(IsEOF(stream));
  fclose(stream);
}

static void RoundTripArrays(FILE* source_stream, FILE* target_stream,
                            iree_hal_device_t* device,
                            iree_hal_allocator_t* device_allocator) {
  while (!IsEOF(source_stream)) {
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
  fflush(target_stream);
}

static void CompareStreams(FILE* source_stream, FILE* target_stream) {
  fseek(source_stream, 0, SEEK_END);
  fseek(target_stream, 0, SEEK_END);
  size_t source_size = ftell(source_stream);
  size_t target_size = ftell(target_stream);
  ASSERT_EQ(source_size, target_size) << "streams should have the same length";
  fseek(source_stream, 0, SEEK_SET);
  fseek(target_stream, 0, SEEK_SET);

  std::vector<uint8_t> source_data;
  source_data.resize(source_size);
  std::vector<uint8_t> target_data;
  target_data.resize(target_size);

  ASSERT_EQ(source_data.size(),
            fread(source_data.data(), 1, source_data.size(), source_stream));
  ASSERT_EQ(target_data.size(),
            fread(target_data.data(), 1, target_data.size(), target_stream));
  ASSERT_THAT(target_data, ElementsAreArray(source_data));

  ASSERT_EQ(IsEOF(source_stream), IsEOF(target_stream))
      << "streams should have the same length";
}

// Tests round-tripping a single array.
TEST_F(NumpyIOTest, RoundTripSingleArray) {
  FILE* source_stream = OpenInputFile("single.npy");
  FILE* target_stream = OpenOutputFile("single_out.npy");
  RoundTripArrays(source_stream, target_stream, device_, device_allocator_);
  CompareStreams(source_stream, target_stream);
  fclose(source_stream);
  fclose(target_stream);
}

// Tests round-tripping multiple array.
TEST_F(NumpyIOTest, RoundTripMultipleArrays) {
  FILE* source_stream = OpenInputFile("multiple.npy");
  FILE* target_stream = OpenOutputFile("multiple_out.npy");
  RoundTripArrays(source_stream, target_stream, device_, device_allocator_);
  CompareStreams(source_stream, target_stream);
  fclose(source_stream);
  fclose(target_stream);
}

// Tests round-tripping arrays with various shapes.
TEST_F(NumpyIOTest, RoundTripArrayShapes) {
  FILE* source_stream = OpenInputFile("array_shapes.npy");
  FILE* target_stream = OpenOutputFile("array_shapes_out.npy");
  RoundTripArrays(source_stream, target_stream, device_, device_allocator_);
  CompareStreams(source_stream, target_stream);
  fclose(source_stream);
  fclose(target_stream);
}

// Tests round-tripping arrays with various types.
TEST_F(NumpyIOTest, RoundTripArrayTypes) {
  FILE* source_stream = OpenInputFile("array_types.npy");
  FILE* target_stream = OpenOutputFile("array_types_out.npy");
  RoundTripArrays(source_stream, target_stream, device_, device_allocator_);
  CompareStreams(source_stream, target_stream);
  fclose(source_stream);
  fclose(target_stream);
}

}  // namespace
}  // namespace iree
