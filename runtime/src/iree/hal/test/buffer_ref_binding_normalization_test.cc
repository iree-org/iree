// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

class BufferRefBindingNormalizationBase : public ::testing::Test {
 public:
  ~BufferRefBindingNormalizationBase() override {
    for (auto buffer : auto_free) {
      iree_hal_buffer_release(buffer);
    }
    if (device_allocator) {
      iree_hal_allocator_destroy(device_allocator);
    }
  };

  iree_status_t lazy_init() {
    if (device_allocator) {
      return iree_ok_status();  // already initialized
    }
    IREE_RETURN_IF_ERROR(iree_hal_allocator_create_heap(
        iree_make_cstring_view("local"), host_allocator, host_allocator,
        &device_allocator));
    return iree_ok_status();
  }

  iree_status_t make_buffer(iree_device_size_t length,
                            iree_hal_buffer_t** out_buffer) {
    IREE_RETURN_IF_ERROR(lazy_init());
    iree_hal_buffer_params_t params = {};
    IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
        device_allocator, params, length, out_buffer));
    auto_free.push_back(*out_buffer);

    return iree_ok_status();
  }

  iree_status_t make_subspan(iree_hal_buffer_t* buffer,
                             iree_device_size_t offset,
                             iree_device_size_t length,
                             iree_hal_buffer_t** out_subspan_buffer) {
    IREE_RETURN_IF_ERROR(lazy_init());
    IREE_RETURN_IF_ERROR(iree_hal_buffer_subspan(
        buffer, /*byte_offset=*/128,
        /*byte_length=*/512, host_allocator, out_subspan_buffer));

    // sanity check of subspan buffer
    EXPECT_EQ(buffer, iree_hal_buffer_allocated_buffer(*out_subspan_buffer));
    EXPECT_EQ(offset, iree_hal_buffer_byte_offset(*out_subspan_buffer));
    EXPECT_EQ(length, iree_hal_buffer_byte_length(*out_subspan_buffer));

    auto_free.push_back(*out_subspan_buffer);

    return iree_ok_status();
  }

  static iree_hal_buffer_binding_t ref_to_binding(
      const iree_hal_buffer_ref_t& ref) {
    return iree_hal_buffer_binding_t{
        /*.buffer=*/ref.buffer, /*.offset=*/ref.offset, /*.length=*/ref.length};
  }

 private:
  static bool is_ok(iree_status_t status) {
    if (iree_status_is_ok(status)) {
      return true;
    }
    std::cerr << iree_status_code_string(iree_status_code(status)) << std::endl;
    iree_status_free(status);
    return false;
  }

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_hal_allocator_t* device_allocator = nullptr;
  std::vector<iree_hal_buffer_t*> auto_free;
};

namespace {

TEST_F(BufferRefBindingNormalizationBase,
       WrappedBufferRefBindingNormalizeAllocatedBuffer) {
  iree_hal_buffer_t* allocated_buffer = nullptr;
  IREE_ASSERT_OK(make_buffer(/*length=*/1024, &allocated_buffer));

  // test buffer reference

  iree_hal_buffer_ref_t ref =
      iree_hal_make_buffer_ref(allocated_buffer, /*offset=*/32, /*length=*/64);

  iree_hal_buffer_ref_t normalized_ref = ref;
  IREE_EXPECT_OK(iree_hal_buffer_ref_normalize(&normalized_ref));

  EXPECT_EQ(allocated_buffer, normalized_ref.buffer);
  EXPECT_EQ(32, normalized_ref.offset);
  EXPECT_EQ(64, normalized_ref.length);

  // test buffer binding

  iree_hal_buffer_binding_t normalized_binding = ref_to_binding(ref);
  IREE_EXPECT_OK(iree_hal_buffer_binding_normalize(&normalized_binding));

  EXPECT_EQ(allocated_buffer, normalized_binding.buffer);
  EXPECT_EQ(32, normalized_binding.offset);
  EXPECT_EQ(64, normalized_binding.length);
}

TEST_F(BufferRefBindingNormalizationBase,
       WrappedBufferRefBindingNormalizeSubspanBuffer) {
  iree_hal_buffer_t* allocated_buffer = nullptr;
  iree_hal_buffer_t* subspan_buffer = nullptr;
  IREE_ASSERT_OK(make_buffer(/*length=*/1024, &allocated_buffer));
  IREE_ASSERT_OK(make_subspan(allocated_buffer, /*offset=*/128, /*length=*/512,
                              &subspan_buffer));

  // test buffer reference

  iree_hal_buffer_ref_t ref =
      iree_hal_make_buffer_ref(subspan_buffer, /*offset=*/32, /*length=*/64);

  iree_hal_buffer_ref_t normalized_ref = ref;
  IREE_EXPECT_OK(iree_hal_buffer_ref_normalize(&normalized_ref));

  EXPECT_EQ(allocated_buffer, normalized_ref.buffer);
  EXPECT_EQ(160, normalized_ref.offset);
  EXPECT_EQ(64, normalized_ref.length);

  // test buffer binding

  iree_hal_buffer_binding_t normalized_binding = ref_to_binding(ref);
  IREE_EXPECT_OK(iree_hal_buffer_binding_normalize(&normalized_binding));

  EXPECT_EQ(allocated_buffer, normalized_binding.buffer);
  EXPECT_EQ(160, normalized_binding.offset);
  EXPECT_EQ(64, normalized_binding.length);
}

TEST_F(BufferRefBindingNormalizationBase,
       WrappedBufferRefBindingNormalizeWholeSubspanBuffer) {
  iree_hal_buffer_t* allocated_buffer = nullptr;
  iree_hal_buffer_t* subspan_buffer = nullptr;
  IREE_ASSERT_OK(make_buffer(/*length=*/1024, &allocated_buffer));
  IREE_ASSERT_OK(make_subspan(allocated_buffer, /*offset=*/128, /*length=*/512,
                              &subspan_buffer));

  // test buffer reference

  iree_hal_buffer_ref_t ref = iree_hal_make_buffer_ref(
      subspan_buffer, /*offset=*/32, /*length=*/IREE_HAL_WHOLE_BUFFER);

  iree_hal_buffer_ref_t normalized_ref = ref;
  IREE_EXPECT_OK(iree_hal_buffer_ref_normalize(&normalized_ref));

  EXPECT_EQ(allocated_buffer, normalized_ref.buffer);
  EXPECT_EQ(160, normalized_ref.offset);
  EXPECT_EQ(480, normalized_ref.length);

  // test buffer binding

  iree_hal_buffer_binding_t normalized_binding = ref_to_binding(ref);
  IREE_EXPECT_OK(iree_hal_buffer_binding_normalize(&normalized_binding));

  EXPECT_EQ(allocated_buffer, normalized_binding.buffer);
  EXPECT_EQ(160, normalized_binding.offset);
  EXPECT_EQ(480, normalized_binding.length);
}

TEST_F(BufferRefBindingNormalizationBase,
       WrappedBufferRefBindingNormalizeNOChangeToSlotRef) {
  iree_hal_buffer_ref_t ref = {/* reserved=*/0, /*.buffer_slot=*/42,
                               /*.buffer=*/NULL, /*.offset=*/23, /*.length=*/5};

  iree_hal_buffer_ref_t normalized_ref = ref;
  IREE_EXPECT_OK(iree_hal_buffer_ref_normalize(&normalized_ref));

  EXPECT_EQ(42, normalized_ref.buffer_slot);
  EXPECT_EQ(NULL, normalized_ref.buffer);
  EXPECT_EQ(23, normalized_ref.offset);
  EXPECT_EQ(5, normalized_ref.length);
}

}  // namespace
