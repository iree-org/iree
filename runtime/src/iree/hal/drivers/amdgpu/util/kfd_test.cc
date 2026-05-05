// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/kfd.h"

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

#if defined(IREE_PLATFORM_LINUX)
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#endif  // IREE_PLATFORM_LINUX

namespace iree::hal::amdgpu {
namespace {

#if defined(IREE_PLATFORM_LINUX)
class StdinRestorer {
 public:
  StdinRestorer() : saved_stdin_(dup(STDIN_FILENO)) {}

  ~StdinRestorer() {
    if (saved_stdin_ >= 0) {
      dup2(saved_stdin_, STDIN_FILENO);
      close(saved_stdin_);
    }
  }

  bool ok() const { return saved_stdin_ >= 0; }

 private:
  int saved_stdin_ = -1;
};
#endif  // IREE_PLATFORM_LINUX

TEST(KFDStandaloneTest, GetClockCountersFailsForInvalidDescriptor) {
  iree_hal_amdgpu_kfd_clock_counters_t counters = {
      /*.gpu_clock_counter=*/1,
      /*.cpu_clock_counter=*/2,
      /*.system_clock_counter=*/3,
      /*.system_clock_freq=*/4,
  };

#if defined(IREE_PLATFORM_LINUX)
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_kfd_get_clock_counters(-1, 1234, &counters));
#else
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_amdgpu_kfd_get_clock_counters(-1, 1234, &counters));
#endif  // IREE_PLATFORM_LINUX

  EXPECT_EQ(counters.gpu_clock_counter, 0);
  EXPECT_EQ(counters.cpu_clock_counter, 0);
  EXPECT_EQ(counters.system_clock_counter, 0);
  EXPECT_EQ(counters.system_clock_freq, 0);
}

TEST(KFDStandaloneTest, CloseTreatsFdZeroAsValid) {
#if defined(IREE_PLATFORM_LINUX)
  StdinRestorer stdin_restorer;
  ASSERT_TRUE(stdin_restorer.ok());

  int pipe_fds[2] = {-1, -1};
  ASSERT_EQ(0, pipe(pipe_fds));
  ASSERT_EQ(STDIN_FILENO, dup2(pipe_fds[0], STDIN_FILENO));
  close(pipe_fds[0]);
  close(pipe_fds[1]);

  iree_hal_amdgpu_kfd_close(STDIN_FILENO);
  errno = 0;
  EXPECT_EQ(-1, fcntl(STDIN_FILENO, F_GETFD));
  EXPECT_EQ(EBADF, errno);
#else
  iree_hal_amdgpu_kfd_close(0);
#endif  // IREE_PLATFORM_LINUX
}

// NOTE: ROCR also opens the KFD - if it initializes then we're likely to
// succeed as well. We need information we can only get from HSA to make the
// ioctls so we have to setup a full topology here.
struct KFDTest : public ::testing::Test {
  static iree_allocator_t host_allocator;
  static iree_hal_amdgpu_libhsa_t libhsa;
  static iree_hal_amdgpu_topology_t topology;

  static void SetUpTestSuite() {
    IREE_TRACE_SCOPE();
    host_allocator = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator, &libhsa);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(
        iree_hal_amdgpu_topology_initialize_with_defaults(&libhsa, &topology));
    if (topology.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    IREE_TRACE_SCOPE();
    iree_hal_amdgpu_topology_deinitialize(&topology);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
  }

  std::optional<uint32_t> GetFirstGPUAgentUID() {
    uint32_t gpu_uid = 0;
    iree_status_t status = iree_hsa_agent_get_info(
        IREE_LIBHSA(&libhsa), topology.gpu_agents[0],
        (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DRIVER_UID, &gpu_uid);
    if (!iree_status_is_ok(status)) {
      iree_status_free(status);
      return std::nullopt;
    }
    return gpu_uid;
  }
};
iree_allocator_t KFDTest::host_allocator;
iree_hal_amdgpu_libhsa_t KFDTest::libhsa;
iree_hal_amdgpu_topology_t KFDTest::topology;

// Tests opening and closing the KFD. It should not crash/leak/etc.
TEST_F(KFDTest, Lifetime) {
  int kfd = -1;
  IREE_ASSERT_OK(iree_hal_amdgpu_kfd_open(&kfd));
#if defined(IREE_PLATFORM_LINUX)
  ASSERT_GE(kfd, 0);
#else
  ASSERT_EQ(kfd, -1);
#endif  // IREE_PLATFORM_LINUX
  iree_hal_amdgpu_kfd_close(kfd);
}

// Tests that we get non-zero counters from the clock.
TEST_F(KFDTest, GetClockCounters) {
  // Find a GPU ID we can use to make the ioctl. If we can't find one we skip
  // the test.
  auto gpu_uid = GetFirstGPUAgentUID();
  if (!gpu_uid) {
    GTEST_SKIP() << "no GPU agent UID available";
    return;
  }

  int kfd = -1;
  IREE_ASSERT_OK(iree_hal_amdgpu_kfd_open(&kfd));

  iree_hal_amdgpu_kfd_clock_counters_t counters = {0};
  IREE_ASSERT_OK(
      iree_hal_amdgpu_kfd_get_clock_counters(kfd, *gpu_uid, &counters));

  // Don't care about the values, just that they were populated.
  ASSERT_NE(counters.gpu_clock_counter, 0);
  ASSERT_NE(counters.cpu_clock_counter, 0);
  ASSERT_NE(counters.system_clock_counter, 0);
  ASSERT_NE(counters.system_clock_freq, 0);

  iree_hal_amdgpu_kfd_close(kfd);
}

}  // namespace
}  // namespace iree::hal::amdgpu
