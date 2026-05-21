// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/device_options.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

static iree_string_pair_list_t PairList(iree_host_size_t count,
                                        const iree_string_pair_t* pairs) {
  iree_string_pair_list_t list = {count, pairs};
  return list;
}

TEST(DeviceOptionsTest, InitializesDefaults) {
  iree_hal_vulkan_device_options_t options;
  iree_hal_vulkan_device_options_initialize(&options);

  EXPECT_EQ(IREE_HAL_VULKAN_DEVICE_FLAG_NONE, options.flags);
  EXPECT_EQ(IREE_HAL_VULKAN_DISPATCH_ABI_ALL_RECOGNIZED, options.dispatch_abis);
  EXPECT_EQ(16u, options.max_cached_bda_replay_instances);
  EXPECT_EQ(64ull * 1024ull * 1024ull,
            options.max_cached_bda_replay_publication_bytes);
  EXPECT_EQ(1u, options.retained_cached_bda_replay_instances);
}

TEST(DeviceOptionsTest, ParsesStringPairs) {
  iree_hal_vulkan_device_options_t options;
  iree_hal_vulkan_device_options_initialize(&options);
  iree_string_pair_t pairs[] = {
      iree_make_cstring_pair("dispatch_abi", "bda"),
      iree_make_cstring_pair("cached_bda_replay_instances", "4"),
      iree_make_cstring_pair("cached_bda_replay_publication_bytes", "1024"),
      iree_make_cstring_pair("retained_cached_bda_replay_instances", "3"),
  };

  IREE_ASSERT_OK(iree_hal_vulkan_device_options_parse(
      &options, PairList(IREE_ARRAYSIZE(pairs), pairs)));

  EXPECT_EQ(IREE_HAL_VULKAN_DISPATCH_ABI_BDA, options.dispatch_abis);
  EXPECT_EQ(4u, options.max_cached_bda_replay_instances);
  EXPECT_EQ(1024u, options.max_cached_bda_replay_publication_bytes);
  EXPECT_EQ(3u, options.retained_cached_bda_replay_instances);
}

TEST(DeviceOptionsTest, ParsesPrefixedAliases) {
  iree_hal_vulkan_device_options_t options;
  iree_hal_vulkan_device_options_initialize(&options);
  iree_string_pair_t pairs[] = {
      iree_make_cstring_pair("vulkan_dispatch_abi", "descriptors"),
      iree_make_cstring_pair("vulkan_cached_bda_replay_instances", "8"),
      iree_make_cstring_pair("vulkan_cached_bda_replay_publication_bytes",
                             "2048"),
      iree_make_cstring_pair("vulkan_retained_cached_bda_replay_instances",
                             "2"),
  };

  IREE_ASSERT_OK(iree_hal_vulkan_device_options_parse(
      &options, PairList(IREE_ARRAYSIZE(pairs), pairs)));

  EXPECT_EQ(IREE_HAL_VULKAN_DISPATCH_ABI_DESCRIPTOR, options.dispatch_abis);
  EXPECT_EQ(8u, options.max_cached_bda_replay_instances);
  EXPECT_EQ(2048u, options.max_cached_bda_replay_publication_bytes);
  EXPECT_EQ(2u, options.retained_cached_bda_replay_instances);
}

TEST(DeviceOptionsTest, RejectsUnknownOption) {
  iree_hal_vulkan_device_options_t options;
  iree_hal_vulkan_device_options_initialize(&options);
  iree_string_pair_t pairs[] = {
      iree_make_cstring_pair("mystery_option", "1"),
  };

  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        iree_hal_vulkan_device_options_parse(
                            &options, PairList(IREE_ARRAYSIZE(pairs), pairs)));
}

TEST(DeviceOptionsTest, RejectsMissingPairStorage) {
  iree_hal_vulkan_device_options_t options;
  iree_hal_vulkan_device_options_initialize(&options);

  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        iree_hal_vulkan_device_options_parse(
                            &options, PairList(/*count=*/1, /*pairs=*/NULL)));
}

TEST(DeviceOptionsTest, RejectsInvalidNumericOption) {
  iree_hal_vulkan_device_options_t options;
  iree_hal_vulkan_device_options_initialize(&options);
  iree_string_pair_t pairs[] = {
      iree_make_cstring_pair("cached_bda_replay_instances", "nope"),
  };

  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        iree_hal_vulkan_device_options_parse(
                            &options, PairList(IREE_ARRAYSIZE(pairs), pairs)));
}

TEST(DeviceOptionsTest, RejectsUnknownDeviceOptionFlags) {
  iree_hal_vulkan_device_options_t options;
  iree_hal_vulkan_device_options_initialize(&options);
  options.flags = 0x80000000u;

  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        iree_hal_vulkan_device_options_verify(&options));
}

TEST(DeviceOptionsTest, RejectsEmptyDispatchAbiMask) {
  iree_hal_vulkan_device_options_t options;
  iree_hal_vulkan_device_options_initialize(&options);
  options.dispatch_abis = IREE_HAL_VULKAN_DISPATCH_ABI_NONE;

  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        iree_hal_vulkan_device_options_verify(&options));
}

TEST(DeviceOptionsTest, RejectsExcessRetainedReplayInstances) {
  iree_hal_vulkan_device_options_t options;
  iree_hal_vulkan_device_options_initialize(&options);
  options.max_cached_bda_replay_instances = 2;
  options.retained_cached_bda_replay_instances = 3;

  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        iree_hal_vulkan_device_options_verify(&options));
}

TEST(DeviceOptionsTest, AllowsUnlimitedReplayInstanceRetentionLimit) {
  iree_hal_vulkan_device_options_t options;
  iree_hal_vulkan_device_options_initialize(&options);
  options.max_cached_bda_replay_instances = 0;
  options.retained_cached_bda_replay_instances = 3;

  IREE_ASSERT_OK(iree_hal_vulkan_device_options_verify(&options));
}

}  // namespace
}  // namespace iree::hal::vulkan
