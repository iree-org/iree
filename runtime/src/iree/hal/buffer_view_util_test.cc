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
