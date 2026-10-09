// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/device_util.h"

#include "iree/hal/utils/statistics_sink.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

// Records the observable session lifecycle without starting worker threads or
// depending on a driver's profiling support.
struct ProfileDevice {
  iree_hal_resource_t resource;
  int begins = 0;
  int flushes = 0;
  int ends = 0;
  int captures = 0;
  int capture_ends = 0;
  bool fail_begin = false;
  bool fail_flush = false;
  bool fail_end = false;
  bool fail_capture = false;

  static ProfileDevice& Get(iree_hal_device_t* device) {
    return *reinterpret_cast<ProfileDevice*>(device);
  }
  iree_hal_device_t* get() {
    return reinterpret_cast<iree_hal_device_t*>(this);
  }

  ProfileDevice() {
    static const auto vtable = [] {
      iree_hal_device_vtable_t table = {};
      table.destroy = +[](iree_hal_device_t*) {};
      table.profiling_begin = +[](iree_hal_device_t* device,
                                  const iree_hal_device_profiling_options_t*) {
        auto& self = Get(device);
        ++self.begins;
        return self.fail_begin ? iree_status_from_code(IREE_STATUS_UNAVAILABLE)
                               : iree_ok_status();
      };
      table.profiling_flush = +[](iree_hal_device_t* device) {
        auto& self = Get(device);
        ++self.flushes;
        return self.fail_flush ? iree_status_from_code(IREE_STATUS_DATA_LOSS)
                               : iree_ok_status();
      };
      table.profiling_end = +[](iree_hal_device_t* device) {
        auto& self = Get(device);
        ++self.ends;
        return self.fail_end ? iree_status_from_code(IREE_STATUS_DATA_LOSS)
                             : iree_ok_status();
      };
      table.external_capture_begin =
          +[](iree_hal_device_t* device,
              const iree_hal_device_external_capture_options_t*) {
            auto& self = Get(device);
            ++self.captures;
            return self.fail_capture
                       ? iree_status_from_code(IREE_STATUS_UNAVAILABLE)
                       : iree_ok_status();
          };
      table.external_capture_end = +[](iree_hal_device_t* device) {
        ++Get(device).capture_ends;
        return iree_ok_status();
      };
      return table;
    }();
    iree_hal_resource_initialize(&vtable, &resource);
  }
  ~ProfileDevice() { iree_hal_device_release(get()); }
};

class ProfilingSessionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_profile_statistics_sink_create(
        iree_allocator_system(), &sink_));
    options_.flags = IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS;
    options_.sink = iree_hal_profile_statistics_sink_base(sink_);
  }
  void TearDown() override {
    IREE_EXPECT_OK(iree_hal_profiling_session_end(session_));
    iree_hal_profile_statistics_sink_release(sink_);
  }
  iree_status_t Begin(ProfileDevice& first, ProfileDevice& second,
                      bool external = false) {
    iree_hal_device_t* devices[] = {first.get(), second.get()};
    iree_hal_device_external_capture_options_t capture = {};
    capture.provider = IREE_SV("test");
    return iree_hal_profiling_session_begin(
        IREE_ARRAYSIZE(devices), devices, &options_,
        external ? &capture : nullptr, 0, iree_allocator_system(), &session_);
  }
  iree_status_t End() {
    auto* session = session_;
    session_ = nullptr;
    return iree_hal_profiling_session_end(session);
  }
  iree_hal_profile_statistics_sink_t* sink_ = nullptr;
  iree_hal_device_profiling_options_t options_ = {};
  iree_hal_profiling_session_t* session_ = nullptr;
};

TEST_F(ProfilingSessionTest, CoversAllDevices) {
  ProfileDevice first, second;
  IREE_ASSERT_OK(Begin(first, second, true));
  IREE_EXPECT_OK(iree_hal_profiling_session_flush(session_));
  IREE_EXPECT_OK(End());
  for (auto* device : {&first, &second}) {
    EXPECT_EQ(device->begins, 1);
    EXPECT_EQ(device->flushes, 1);
    EXPECT_EQ(device->ends, 1);
    EXPECT_EQ(device->captures, 1);
    EXPECT_EQ(device->capture_ends, 1);
  }
}

TEST_F(ProfilingSessionTest, UnwindsPartialNativeStartup) {
  ProfileDevice first, second;
  second.fail_begin = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, Begin(first, second));
  EXPECT_EQ(session_, nullptr);
  EXPECT_EQ(first.ends, 1);
  EXPECT_EQ(second.ends, 0);
}

TEST_F(ProfilingSessionTest, UnwindsPartialExternalStartup) {
  ProfileDevice first, second;
  second.fail_capture = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, Begin(first, second, true));
  EXPECT_EQ(session_, nullptr);
  EXPECT_EQ(first.ends, 1);
  EXPECT_EQ(second.ends, 1);
  EXPECT_EQ(first.capture_ends, 1);
  EXPECT_EQ(second.capture_ends, 0);
}

TEST_F(ProfilingSessionTest, FlushAndEndFailuresDoNotSkipOtherDevices) {
  ProfileDevice first, second;
  first.fail_flush = true;
  second.fail_end = true;
  IREE_ASSERT_OK(Begin(first, second));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_profiling_session_flush(session_));
  EXPECT_EQ(second.flushes, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, End());
  EXPECT_EQ(first.ends, 1);
  EXPECT_EQ(second.ends, 1);
}

}  // namespace
}  // namespace iree
