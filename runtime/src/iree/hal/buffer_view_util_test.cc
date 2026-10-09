// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

typedef struct iree_hal_test_rounding_allocator_t {
  // Base HAL resource for allocator lifetime management.
  iree_hal_resource_t resource;
  // Host allocator used for wrapper metadata and staging allocations.
  iree_allocator_t host_allocator;
} iree_hal_test_rounding_allocator_t;

static void iree_hal_test_rounding_allocator_destroy(
    iree_hal_allocator_t* base_allocator) {
  iree_hal_test_rounding_allocator_t* allocator =
      (iree_hal_test_rounding_allocator_t*)base_allocator;
  iree_allocator_t host_allocator = allocator->host_allocator;
  iree_allocator_free(host_allocator, allocator);
}

static iree_allocator_t iree_hal_test_rounding_allocator_host_allocator(
    const iree_hal_allocator_t* base_allocator) {
  const iree_hal_test_rounding_allocator_t* allocator =
      (const iree_hal_test_rounding_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_hal_buffer_compatibility_t
iree_hal_test_rounding_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_params_t* params,
    iree_device_size_t* allocation_size) {
  (void)base_allocator;
  (void)params;
  *allocation_size = iree_device_align(*allocation_size, 64);
  return IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
         IREE_HAL_BUFFER_COMPATIBILITY_LOW_PERFORMANCE;
}

static const iree_hal_allocator_vtable_t
    iree_hal_test_rounding_allocator_vtable = {
        /*.destroy=*/iree_hal_test_rounding_allocator_destroy,
        /*.host_allocator=*/iree_hal_test_rounding_allocator_host_allocator,
        /*.trim=*/NULL,
        /*.query_statistics=*/NULL,
        /*.query_memory_heaps=*/NULL,
        /*.query_buffer_compatibility=*/
        iree_hal_test_rounding_allocator_query_buffer_compatibility,
};

static iree_status_t iree_hal_test_rounding_allocator_create(
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  iree_hal_test_rounding_allocator_t* allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                             (void**)&allocator));
  iree_hal_resource_initialize(&iree_hal_test_rounding_allocator_vtable,
                               &allocator->resource);
  allocator->host_allocator = host_allocator;
  *out_allocator = (iree_hal_allocator_t*)allocator;
  return iree_ok_status();
}

typedef struct iree_hal_test_generate_buffer_state_t {
  // Logical view byte length the generator must receive.
  iree_device_size_t expected_content_size;
  // Observed generator mapping byte length.
  iree_device_size_t actual_content_size;
} iree_hal_test_generate_buffer_state_t;

static iree_status_t iree_hal_test_generate_buffer_callback(
    iree_hal_buffer_mapping_t* mapping, void* user_data) {
  iree_hal_test_generate_buffer_state_t* state =
      (iree_hal_test_generate_buffer_state_t*)user_data;
  state->actual_content_size = mapping->contents.data_length;
  if (state->actual_content_size != state->expected_content_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "generator received padded contents");
  }
  return iree_make_status(IREE_STATUS_CANCELLED, "contract check complete");
}

TEST(BufferViewUtilTest, ComputeElementCount) {
  iree_device_size_t element_count = 0;
  IREE_ASSERT_OK(iree_hal_buffer_view_compute_element_count(
      /*shape_rank=*/0, /*shape=*/NULL, &element_count));
  EXPECT_EQ(1u, element_count);

  const iree_hal_dim_t shape[] = {2, 3};
  IREE_ASSERT_OK(iree_hal_buffer_view_compute_element_count(
      IREE_ARRAYSIZE(shape), shape, &element_count));
  EXPECT_EQ(6u, element_count);

  const iree_hal_dim_t max_shape[] = {IREE_DEVICE_SIZE_MAX};
  IREE_ASSERT_OK(iree_hal_buffer_view_compute_element_count(
      IREE_ARRAYSIZE(max_shape), max_shape, &element_count));
  EXPECT_EQ(IREE_DEVICE_SIZE_MAX, element_count);

  const iree_hal_dim_t zero_first_shape[] = {0, IREE_DEVICE_SIZE_MAX, 2};
  IREE_ASSERT_OK(iree_hal_buffer_view_compute_element_count(
      IREE_ARRAYSIZE(zero_first_shape), zero_first_shape, &element_count));
  EXPECT_EQ(0u, element_count);

  const iree_hal_dim_t zero_last_shape[] = {IREE_DEVICE_SIZE_MAX, 2, 0};
  IREE_ASSERT_OK(iree_hal_buffer_view_compute_element_count(
      IREE_ARRAYSIZE(zero_last_shape), zero_last_shape, &element_count));
  EXPECT_EQ(0u, element_count);

  const iree_hal_dim_t overflow_shape[] = {IREE_DEVICE_SIZE_MAX, 2};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_buffer_view_compute_element_count(
          IREE_ARRAYSIZE(overflow_shape), overflow_shape, &element_count));
  EXPECT_EQ(0u, element_count);
}

TEST(BufferViewUtilTest, ComputePackedByteCount) {
  iree_device_size_t byte_count = 0;
  IREE_ASSERT_OK(iree_hal_element_compute_packed_byte_count(
      IREE_HAL_ELEMENT_TYPE_INT_8, IREE_DEVICE_SIZE_MAX, &byte_count));
  EXPECT_EQ(IREE_DEVICE_SIZE_MAX, byte_count);

  IREE_ASSERT_OK(iree_hal_element_compute_packed_byte_count(
      IREE_HAL_ELEMENT_TYPE_INT_4, IREE_DEVICE_SIZE_MAX, &byte_count));
  EXPECT_EQ(IREE_DEVICE_SIZE_MAX / 2 + 1, byte_count);

  IREE_ASSERT_OK(iree_hal_element_compute_packed_byte_count(
      IREE_HAL_ELEMENT_TYPE_INT_4, 3, &byte_count));
  EXPECT_EQ(2u, byte_count);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_element_compute_packed_byte_count(
          IREE_HAL_ELEMENT_TYPE_INT_16, IREE_DEVICE_SIZE_MAX, &byte_count));
  EXPECT_EQ(0u, byte_count);

  const iree_hal_element_type_t int_24_type =
      iree_hal_make_element_type(IREE_HAL_NUMERICAL_TYPE_INTEGER, 24);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_element_compute_packed_byte_count(
          int_24_type, 8 * (IREE_DEVICE_SIZE_MAX / 24) + 7, &byte_count));
  EXPECT_EQ(0u, byte_count);
}

TEST(BufferViewUtilTest, ComputeViewSizeRejectsElementCountOverflow) {
  const iree_hal_dim_t zero_byte_shape[] = {IREE_DEVICE_SIZE_MAX / 2 + 1, 2};
  iree_device_size_t allocation_size = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_buffer_compute_view_size(
          IREE_ARRAYSIZE(zero_byte_shape), zero_byte_shape,
          IREE_HAL_ELEMENT_TYPE_INT_8, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
          &allocation_size));

  if (sizeof(iree_device_size_t) == sizeof(uint64_t)) {
    const iree_hal_dim_t one_byte_shape[] = {
        static_cast<iree_hal_dim_t>(UINT64_C(274177)),
        static_cast<iree_hal_dim_t>(UINT64_C(67280421310721))};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        iree_hal_buffer_compute_view_size(
            IREE_ARRAYSIZE(one_byte_shape), one_byte_shape,
            IREE_HAL_ELEMENT_TYPE_INT_8, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
            &allocation_size));
  }
}

TEST(BufferViewUtilTest, ComputeViewSizeChecksPackedByteCount) {
  const iree_hal_dim_t max_shape[] = {IREE_DEVICE_SIZE_MAX};
  iree_device_size_t allocation_size = 0;
  IREE_ASSERT_OK(iree_hal_buffer_compute_view_size(
      IREE_ARRAYSIZE(max_shape), max_shape, IREE_HAL_ELEMENT_TYPE_INT_8,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, &allocation_size));
  EXPECT_EQ(IREE_DEVICE_SIZE_MAX, allocation_size);

  IREE_ASSERT_OK(iree_hal_buffer_compute_view_size(
      IREE_ARRAYSIZE(max_shape), max_shape, IREE_HAL_ELEMENT_TYPE_INT_4,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, &allocation_size));
  EXPECT_EQ(IREE_DEVICE_SIZE_MAX / 2 + 1, allocation_size);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_buffer_compute_view_size(
          IREE_ARRAYSIZE(max_shape), max_shape, IREE_HAL_ELEMENT_TYPE_INT_16,
          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, &allocation_size));

  const iree_hal_dim_t addition_overflow_shape[] = {
      8 * (IREE_DEVICE_SIZE_MAX / 24) + 7};
  const iree_hal_element_type_t int_24_type =
      iree_hal_make_element_type(IREE_HAL_NUMERICAL_TYPE_INTEGER, 24);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_buffer_compute_view_size(IREE_ARRAYSIZE(addition_overflow_shape),
                                        addition_overflow_shape, int_24_type,
                                        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                        &allocation_size));
}

TEST(BufferViewUtilTest, ComputeViewSizeAllowsZeroDimensions) {
  const iree_hal_dim_t shape[] = {IREE_DEVICE_SIZE_MAX, 2, 0};
  iree_device_size_t allocation_size = IREE_DEVICE_SIZE_MAX;
  IREE_ASSERT_OK(iree_hal_buffer_compute_view_size(
      IREE_ARRAYSIZE(shape), shape, IREE_HAL_ELEMENT_TYPE_INT_8,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, &allocation_size));
  EXPECT_EQ(0u, allocation_size);
}

TEST(BufferViewUtilTest, CreateAndReshapeValidateShapes) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_create_heap(
      iree_make_cstring_view("host_local"), iree_allocator_system(),
      iree_allocator_system(), &allocator));

  iree_hal_buffer_params_t buffer_params = {
      /*.usage=*/IREE_HAL_BUFFER_USAGE_DEFAULT,
      /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
      /*.type=*/IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
      /*.queue_affinity=*/0,
  };
  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, buffer_params, /*allocation_size=*/1, &buffer));

  const iree_hal_dim_t shape[] = {IREE_DEVICE_SIZE_MAX / 2 + 1, 2};
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_buffer_view_create(buffer, IREE_ARRAYSIZE(shape), shape,
                                  IREE_HAL_ELEMENT_TYPE_INT_8,
                                  IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                  iree_allocator_system(), &buffer_view));
  EXPECT_EQ(NULL, buffer_view);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_buffer_view_create(buffer, IREE_ARRAYSIZE(shape), shape,
                                  IREE_HAL_ELEMENT_TYPE_INT_8,
                                  IREE_HAL_ENCODING_TYPE_OPAQUE,
                                  iree_allocator_system(), &buffer_view));
  EXPECT_EQ(NULL, buffer_view);

  const iree_hal_dim_t max_shape[] = {IREE_DEVICE_SIZE_MAX};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_buffer_view_create(buffer, IREE_ARRAYSIZE(max_shape), max_shape,
                                  IREE_HAL_ELEMENT_TYPE_INT_16,
                                  IREE_HAL_ENCODING_TYPE_OPAQUE,
                                  iree_allocator_system(), &buffer_view));
  EXPECT_EQ(NULL, buffer_view);

  const iree_hal_dim_t oversized_shape[] = {2};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_buffer_view_create(buffer, IREE_ARRAYSIZE(oversized_shape),
                                  oversized_shape, IREE_HAL_ELEMENT_TYPE_INT_8,
                                  IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                  iree_allocator_system(), &buffer_view));
  if (buffer_view) {
    iree_hal_buffer_view_release(buffer_view);
    buffer_view = NULL;
  }
  IREE_ASSERT_OK(iree_hal_buffer_view_create(
      buffer, IREE_ARRAYSIZE(oversized_shape), oversized_shape,
      IREE_HAL_ELEMENT_TYPE_INT_8, IREE_HAL_ENCODING_TYPE_OPAQUE,
      iree_allocator_system(), &buffer_view));
  EXPECT_EQ(2u, iree_hal_buffer_view_byte_length(buffer_view));
  iree_hal_buffer_view_release(buffer_view);
  buffer_view = NULL;

  const iree_hal_dim_t valid_shape[] = {1, 1};
  IREE_ASSERT_OK(iree_hal_buffer_view_create(
      buffer, IREE_ARRAYSIZE(valid_shape), valid_shape,
      IREE_HAL_ELEMENT_TYPE_INT_8, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      iree_allocator_system(), &buffer_view));
  iree_hal_dim_t reshape_shape[2];
  if (sizeof(iree_device_size_t) == sizeof(uint64_t)) {
    reshape_shape[0] = static_cast<iree_hal_dim_t>(UINT64_C(274177));
    reshape_shape[1] = static_cast<iree_hal_dim_t>(UINT64_C(67280421310721));
  } else {
    reshape_shape[0] = 641;
    reshape_shape[1] = 6700417;
  }
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_buffer_view_reshape(buffer_view, reshape_shape,
                                   IREE_ARRAYSIZE(reshape_shape)));
  EXPECT_EQ(1u, iree_hal_buffer_view_shape_dim(buffer_view, 0));
  EXPECT_EQ(1u, iree_hal_buffer_view_shape_dim(buffer_view, 1));

  iree_hal_buffer_view_release(buffer_view);

  const iree_hal_dim_t zero_shape[] = {IREE_DEVICE_SIZE_MAX, 2, 0};
  IREE_ASSERT_OK(iree_hal_buffer_view_create(
      buffer, IREE_ARRAYSIZE(zero_shape), zero_shape,
      IREE_HAL_ELEMENT_TYPE_INT_8, IREE_HAL_ENCODING_TYPE_OPAQUE,
      iree_allocator_system(), &buffer_view));
  EXPECT_EQ(0u, iree_hal_buffer_view_element_count(buffer_view));
  const iree_hal_dim_t reshaped_zero_shape[] = {0, IREE_DEVICE_SIZE_MAX, 2};
  IREE_ASSERT_OK(iree_hal_buffer_view_reshape(
      buffer_view, reshaped_zero_shape, IREE_ARRAYSIZE(reshaped_zero_shape)));
  EXPECT_EQ(0u, iree_hal_buffer_view_element_count(buffer_view));
  EXPECT_EQ(0u, iree_hal_buffer_view_shape_dim(buffer_view, 0));
  iree_hal_buffer_view_release(buffer_view);

  iree_hal_buffer_release(buffer);
  iree_hal_allocator_release(allocator);
}

TEST(BufferViewUtilTest, GenerateBufferUsesLogicalViewSizeForGenerator) {
  iree_allocator_t host_allocator = iree_allocator_system();

  iree_hal_allocator_t* allocator = NULL;
  IREE_ASSERT_OK(
      iree_hal_test_rounding_allocator_create(host_allocator, &allocator));

  iree_hal_test_generate_buffer_state_t state = {
      /*.expected_content_size=*/sizeof(float),
      /*.actual_content_size=*/0,
  };
  iree_hal_dim_t shape[] = {1};
  iree_hal_buffer_params_t buffer_params = {
      /*.usage=*/IREE_HAL_BUFFER_USAGE_DEFAULT,
      /*.access=*/0,
      /*.type=*/IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      /*.queue_affinity=*/0,
  };
  iree_hal_buffer_view_t* buffer_view = NULL;
  iree_status_t status = iree_hal_buffer_view_generate_buffer(
      /*device=*/NULL, allocator, IREE_ARRAYSIZE(shape), shape,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      buffer_params, iree_hal_test_generate_buffer_callback, &state,
      &buffer_view);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, status);
  EXPECT_EQ(sizeof(float), state.actual_content_size);

  iree_hal_buffer_view_release(buffer_view);
  iree_hal_allocator_release(allocator);
}

}  // namespace
}  // namespace iree
