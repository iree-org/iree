// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/host/host_fence.h"

#include <cstdint>
#include <thread>  // NOLINT

#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "iree/base/status.h"
#include "iree/base/status_matchers.h"

namespace iree {
namespace hal {
namespace {

// Tests that a fence that is unused properly cleans itself up.
TEST(HostFenceTest, NoOp) {
  HostFence fence(123u);
  EXPECT_TRUE(fence.status().ok());
  ASSERT_OK_AND_ASSIGN(uint64_t value, fence.QueryValue());
  EXPECT_EQ(123u, value);
}

// Tests that a fence will accept new values as it is signaled.
TEST(HostFenceTest, NormalSignaling) {
  HostFence fence(2u);
  EXPECT_EQ(2u, fence.QueryValue().ValueOrDie());
  EXPECT_OK(fence.Signal(3u));
  EXPECT_EQ(3u, fence.QueryValue().ValueOrDie());
  EXPECT_OK(fence.Signal(40u));
  EXPECT_EQ(40u, fence.QueryValue().ValueOrDie());
}

// Tests that a fence will fail to set non-increasing values.
TEST(HostFenceTest, RequireIncreasingValues) {
  HostFence fence(2u);
  EXPECT_EQ(2u, fence.QueryValue().ValueOrDie());
  // Same value.
  EXPECT_TRUE(IsInvalidArgument(fence.Signal(2u)));
  // Decreasing.
  EXPECT_TRUE(IsInvalidArgument(fence.Signal(1u)));
}

// Tests that a fence that has failed will remain in a failed state.
TEST(HostFenceTest, StickyFailure) {
  HostFence fence(2u);
  // Signal to 3.
  EXPECT_OK(fence.Signal(3u));
  EXPECT_TRUE(fence.status().ok());
  EXPECT_EQ(3u, fence.QueryValue().ValueOrDie());

  // Fail now.
  EXPECT_OK(fence.Fail(UnknownErrorBuilder(IREE_LOC)));
  EXPECT_TRUE(IsUnknown(fence.status()));
  EXPECT_EQ(UINT64_MAX, fence.QueryValue().ValueOrDie());

  // Unable to signal again (it'll return the sticky failure).
  EXPECT_TRUE(IsUnknown(fence.Signal(4u)));
  EXPECT_TRUE(IsUnknown(fence.status()));
  EXPECT_EQ(UINT64_MAX, fence.QueryValue().ValueOrDie());
}

// Tests waiting on no fences.
TEST(HostFenceTest, EmptyWait) {
  EXPECT_OK(
      HostFence::WaitForFences({}, /*wait_all=*/true, absl::InfiniteFuture()));
}

// Tests waiting on a fence that has already been signaled.
TEST(HostFenceTest, WaitAlreadySignaled) {
  HostFence fence(2u);
  // Test both previous and current values.
  EXPECT_OK(HostFence::WaitForFences({{&fence, 1u}}, /*wait_all=*/true,
                                     absl::InfiniteFuture()));
  EXPECT_OK(HostFence::WaitForFences({{&fence, 2u}}, /*wait_all=*/true,
                                     absl::InfiniteFuture()));
}

// Tests waiting on a fence that has not been signaled.
TEST(HostFenceTest, WaitUnsignaled) {
  HostFence fence(2u);
  // NOTE: we don't actually block here because otherwise we'd lock up.
  EXPECT_TRUE(IsDeadlineExceeded(HostFence::WaitForFences(
      {{&fence, 3u}}, /*wait_all=*/true, absl::InfinitePast())));
}

// Tests waiting on a failed fence (it should return the error on the fence).
TEST(HostFenceTest, WaitAlreadyFailed) {
  HostFence fence(2u);
  EXPECT_OK(fence.Fail(UnknownErrorBuilder(IREE_LOC)));
  EXPECT_TRUE(IsUnknown(HostFence::WaitForFences(
      {{&fence, 2u}}, /*wait_all=*/true, absl::InfinitePast())));
}

// Tests threading behavior by ping-ponging between the test main thread and
// a little thread.
TEST(HostFenceTest, PingPong) {
  HostFence a2b(0u);
  HostFence b2a(0u);
  std::thread thread([&]() {
    // Should advance right past this because the value is already set.
    ASSERT_OK(HostFence::WaitForFences({{&a2b, 0u}}, /*wait_all=*/true,
                                       absl::InfiniteFuture()));
    ASSERT_OK(b2a.Signal(1u));
    // Jump ahead.
    ASSERT_OK(HostFence::WaitForFences({{&a2b, 4u}}, /*wait_all=*/true,
                                       absl::InfiniteFuture()));
  });
  ASSERT_OK(HostFence::WaitForFences({{&b2a, 1u}}, /*wait_all=*/true,
                                     absl::InfiniteFuture()));
  ASSERT_OK(a2b.Signal(4u));
  thread.join();
}

// Tests that failure still wakes waiters and propagates the error.
TEST(HostFenceTest, FailNotifies) {
  HostFence a2b(0u);
  HostFence b2a(0u);
  bool got_failure = false;
  std::thread thread([&]() {
    ASSERT_OK(b2a.Signal(1u));
    got_failure = IsUnknown(HostFence::WaitForFences(
        {{&a2b, 1u}}, /*wait_all=*/true, absl::InfiniteFuture()));
  });
  ASSERT_OK(HostFence::WaitForFences({{&b2a, 1u}}, /*wait_all=*/true,
                                     absl::InfiniteFuture()));
  ASSERT_OK(a2b.Fail(UnknownErrorBuilder(IREE_LOC)));
  thread.join();
  ASSERT_TRUE(got_failure);
}

}  // namespace
}  // namespace hal
}  // namespace iree
