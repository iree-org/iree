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

#include "iree/hal/host/condvar_semaphore.h"

#include <cstdint>
#include <thread>  // NOLINT

#include "absl/time/time.h"
#include "iree/base/status.h"
#include "iree/base/status_matchers.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace host {
namespace {

// Tests that a semaphore that is unused properly cleans itself up.
TEST(CondVarSemaphoreTest, NoOp) {
  CondVarSemaphore semaphore(123u);
  ASSERT_OK_AND_ASSIGN(uint64_t value, semaphore.Query());
  EXPECT_EQ(123u, value);
}

// Tests that a semaphore will accept new values as it is signaled.
TEST(CondVarSemaphoreTest, NormalSignaling) {
  CondVarSemaphore semaphore(2u);
  EXPECT_EQ(2u, semaphore.Query().value());
  EXPECT_OK(semaphore.Signal(3u));
  EXPECT_EQ(3u, semaphore.Query().value());
  EXPECT_OK(semaphore.Signal(40u));
  EXPECT_EQ(40u, semaphore.Query().value());
}

// Tests that a semaphore will fail to set non-increasing values.
TEST(CondVarSemaphoreTest, RequireIncreasingValues) {
  CondVarSemaphore semaphore(2u);
  EXPECT_EQ(2u, semaphore.Query().value());
  // Same value.
  EXPECT_TRUE(IsInvalidArgument(semaphore.Signal(2u)));
  // Decreasing.
  EXPECT_TRUE(IsInvalidArgument(semaphore.Signal(1u)));
}

// Tests that a semaphore that has failed will remain in a failed state.
TEST(CondVarSemaphoreTest, StickyFailure) {
  CondVarSemaphore semaphore(2u);
  // Signal to 3.
  EXPECT_OK(semaphore.Signal(3u));
  EXPECT_EQ(3u, semaphore.Query().value());

  // Fail now.
  semaphore.Fail(UnknownErrorBuilder(IREE_LOC));
  EXPECT_TRUE(IsUnknown(semaphore.Query().status()));

  // Unable to signal again (it'll return the sticky failure).
  EXPECT_TRUE(IsUnknown(semaphore.Signal(4u)));
  EXPECT_TRUE(IsUnknown(semaphore.Query().status()));
}

// Tests waiting on no semaphores.
TEST(CondVarSemaphoreTest, EmptyWait) {
  EXPECT_OK(CondVarSemaphore::WaitForSemaphores({}, /*wait_all=*/true,
                                                absl::InfiniteFuture()));
}

// Tests waiting on a semaphore that has already been signaled.
TEST(CondVarSemaphoreTest, WaitAlreadySignaled) {
  CondVarSemaphore semaphore(2u);
  // Test both previous and current values.
  EXPECT_OK(CondVarSemaphore::WaitForSemaphores(
      {{&semaphore, 1u}}, /*wait_all=*/true, absl::InfiniteFuture()));
  EXPECT_OK(CondVarSemaphore::WaitForSemaphores(
      {{&semaphore, 2u}}, /*wait_all=*/true, absl::InfiniteFuture()));
}

// Tests waiting on a semaphore that has not been signaled.
TEST(CondVarSemaphoreTest, WaitUnsignaled) {
  CondVarSemaphore semaphore(2u);
  // NOTE: we don't actually block here because otherwise we'd lock up.
  EXPECT_TRUE(IsDeadlineExceeded(CondVarSemaphore::WaitForSemaphores(
      {{&semaphore, 3u}}, /*wait_all=*/true, absl::InfinitePast())));
}

// Tests waiting on a failed semaphore (it should return the error on the
// semaphore).
TEST(CondVarSemaphoreTest, WaitAlreadyFailed) {
  CondVarSemaphore semaphore(2u);
  semaphore.Fail(UnknownErrorBuilder(IREE_LOC));
  EXPECT_TRUE(IsUnknown(CondVarSemaphore::WaitForSemaphores(
      {{&semaphore, 2u}}, /*wait_all=*/true, absl::InfinitePast())));
}

// Tests threading behavior by ping-ponging between the test main thread and
// a little thread.
TEST(CondVarSemaphoreTest, PingPong) {
  CondVarSemaphore a2b(0u);
  CondVarSemaphore b2a(0u);
  std::thread thread([&]() {
    // Should advance right past this because the value is already set.
    ASSERT_OK(CondVarSemaphore::WaitForSemaphores(
        {{&a2b, 0u}}, /*wait_all=*/true, absl::InfiniteFuture()));
    ASSERT_OK(b2a.Signal(1u));
    // Jump ahead.
    ASSERT_OK(CondVarSemaphore::WaitForSemaphores(
        {{&a2b, 4u}}, /*wait_all=*/true, absl::InfiniteFuture()));
  });
  ASSERT_OK(CondVarSemaphore::WaitForSemaphores({{&b2a, 1u}}, /*wait_all=*/true,
                                                absl::InfiniteFuture()));
  ASSERT_OK(a2b.Signal(4u));
  thread.join();
}

// Tests that failure still wakes waiters and propagates the error.
TEST(CondVarSemaphoreTest, FailNotifies) {
  CondVarSemaphore a2b(0u);
  CondVarSemaphore b2a(0u);
  bool got_failure = false;
  std::thread thread([&]() {
    ASSERT_OK(b2a.Signal(1u));
    got_failure = IsUnknown(CondVarSemaphore::WaitForSemaphores(
        {{&a2b, 1u}}, /*wait_all=*/true, absl::InfiniteFuture()));
  });
  ASSERT_OK(CondVarSemaphore::WaitForSemaphores({{&b2a, 1u}}, /*wait_all=*/true,
                                                absl::InfiniteFuture()));
  a2b.Fail(UnknownErrorBuilder(IREE_LOC));
  thread.join();
  ASSERT_TRUE(got_failure);
}

}  // namespace
}  // namespace host
}  // namespace hal
}  // namespace iree
