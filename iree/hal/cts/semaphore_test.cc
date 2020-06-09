// Copyright 2020 Google LLC
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

#include "iree/hal/cts/cts_test_base.h"
#include "iree/hal/driver_registry.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace cts {

class SemaphoreTest : public CtsTestBase {};

// Tests that a semaphore that is unused properly cleans itself up.
TEST_P(SemaphoreTest, NoOp) {
  ASSERT_OK_AND_ASSIGN(auto semaphore, device_->CreateSemaphore(123u));
  ASSERT_OK_AND_ASSIGN(uint64_t value, semaphore->Query());
  EXPECT_EQ(123u, value);
}

// Tests that a semaphore will accept new values as it is signaled.
TEST_P(SemaphoreTest, NormalSignaling) {
  ASSERT_OK_AND_ASSIGN(auto semaphore, device_->CreateSemaphore(2u));
  EXPECT_EQ(2u, semaphore->Query().value());
  EXPECT_OK(semaphore->Signal(3u));
  EXPECT_EQ(3u, semaphore->Query().value());
  EXPECT_OK(semaphore->Signal(40u));
  EXPECT_EQ(40u, semaphore->Query().value());
}

// Tests that a semaphore will fail to set non-increasing values.
TEST_P(SemaphoreTest, RequireIncreasingValues) {
  ASSERT_OK_AND_ASSIGN(auto semaphore, device_->CreateSemaphore(2u));
  EXPECT_EQ(2u, semaphore->Query().value());
  // Same value.
  EXPECT_TRUE(IsInvalidArgument(semaphore->Signal(2u)));
  // Decreasing.
  EXPECT_TRUE(IsInvalidArgument(semaphore->Signal(1u)));
}

// Tests that a semaphore that has failed will remain in a failed state.
TEST_P(SemaphoreTest, StickyFailure) {
  ASSERT_OK_AND_ASSIGN(auto semaphore, device_->CreateSemaphore(2u));
  // Signal to 3.
  EXPECT_OK(semaphore->Signal(3u));
  EXPECT_EQ(3u, semaphore->Query().value());

  // Fail now.
  semaphore->Fail(UnknownErrorBuilder(IREE_LOC));
  EXPECT_TRUE(IsUnknown(semaphore->Query().status()));

  // Unable to signal again (it'll return the sticky failure).
  EXPECT_TRUE(IsUnknown(semaphore->Signal(4u)));
  EXPECT_TRUE(IsUnknown(semaphore->Query().status()));
}

// Tests waiting on no semaphores.
TEST_P(SemaphoreTest, EmptyWait) {
  EXPECT_OK(device_->WaitAllSemaphores({}, absl::InfiniteFuture()));
}

// Tests waiting on a semaphore that has already been signaled.
TEST_P(SemaphoreTest, WaitAlreadySignaled) {
  ASSERT_OK_AND_ASSIGN(auto semaphore, device_->CreateSemaphore(2u));
  // Test both previous and current values.
  EXPECT_OK(device_->WaitAllSemaphores({{semaphore.get(), 1u}},
                                       absl::InfiniteFuture()));
  EXPECT_OK(device_->WaitAllSemaphores({{semaphore.get(), 2u}},
                                       absl::InfiniteFuture()));
}

// Tests waiting on a semaphore that has not been signaled.
TEST_P(SemaphoreTest, WaitUnsignaled) {
  ASSERT_OK_AND_ASSIGN(auto semaphore, device_->CreateSemaphore(2u));
  // NOTE: we don't actually block here because otherwise we'd lock up.
  EXPECT_TRUE(IsDeadlineExceeded(device_->WaitAllSemaphores(
      {{semaphore.get(), 3u}}, absl::InfinitePast())));
}

// Tests waiting on a failed semaphore (it should return the error on the
// semaphore).
TEST_P(SemaphoreTest, WaitAlreadyFailed) {
  ASSERT_OK_AND_ASSIGN(auto semaphore, device_->CreateSemaphore(2u));
  semaphore->Fail(UnknownErrorBuilder(IREE_LOC));
  EXPECT_TRUE(IsUnknown(device_->WaitAllSemaphores({{semaphore.get(), 2u}},
                                                   absl::InfinitePast())));
}

// Tests threading behavior by ping-ponging between the test main thread and
// a little thread.
TEST_P(SemaphoreTest, PingPong) {
  ASSERT_OK_AND_ASSIGN(auto a2b, device_->CreateSemaphore(0u));
  ASSERT_OK_AND_ASSIGN(auto b2a, device_->CreateSemaphore(0u));
  std::thread thread([&]() {
    // Should advance right past this because the value is already set.
    ASSERT_OK(
        device_->WaitAllSemaphores({{a2b.get(), 0u}}, absl::InfiniteFuture()));
    ASSERT_OK(b2a->Signal(1u));
    // Jump ahead.
    ASSERT_OK(
        device_->WaitAllSemaphores({{a2b.get(), 4u}}, absl::InfiniteFuture()));
  });
  ASSERT_OK(
      device_->WaitAllSemaphores({{b2a.get(), 1u}}, absl::InfiniteFuture()));
  ASSERT_OK(a2b->Signal(4u));
  thread.join();
}

// Tests that failure still wakes waiters and propagates the error.
TEST_P(SemaphoreTest, FailNotifies) {
  ASSERT_OK_AND_ASSIGN(auto a2b, device_->CreateSemaphore(0u));
  ASSERT_OK_AND_ASSIGN(auto b2a, device_->CreateSemaphore(0u));
  bool got_failure = false;
  std::thread thread([&]() {
    ASSERT_OK(b2a->Signal(1u));
    got_failure = IsUnknown(
        device_->WaitAllSemaphores({{a2b.get(), 1u}}, absl::InfiniteFuture()));
  });
  ASSERT_OK(
      device_->WaitAllSemaphores({{b2a.get(), 1u}}, absl::InfiniteFuture()));
  a2b->Fail(UnknownErrorBuilder(IREE_LOC));
  thread.join();
  ASSERT_TRUE(got_failure);
}

INSTANTIATE_TEST_SUITE_P(AllDrivers, SemaphoreTest,
                         ::testing::ValuesIn(DriverRegistry::shared_registry()
                                                 ->EnumerateAvailableDrivers()),
                         GenerateTestName());

}  // namespace cts
}  // namespace hal
}  // namespace iree
