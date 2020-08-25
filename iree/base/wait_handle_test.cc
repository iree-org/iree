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

#include "iree/base/wait_handle.h"

#include <unistd.h>

#include <string>
#include <thread>  // NOLINT
#include <type_traits>

#include "absl/time/time.h"
#include "iree/base/status.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

// StatusOr<bool> will be true if the status is ok, which is bad.
#define ASSERT_STATUSOR_TRUE(x) ASSERT_TRUE(x.value())
#define ASSERT_STATUSOR_FALSE(x) ASSERT_FALSE(x.value())

namespace iree {
namespace {

using ::testing::_;
using ::testing::Return;

// Tests the AlwaysSignaling helper.
TEST(WaitHandleTest, AlwaysSignaling) {
  IREE_ASSERT_OK(WaitHandle::AlwaysSignaling().Wait());
  EXPECT_FALSE(WaitHandle::AlwaysSignaling().DebugString().empty());
}

// Tests the AlwaysFailing helper.
TEST(WaitHandleTest, AlwaysFailing) {
  ASSERT_FALSE(WaitHandle::AlwaysFailing().Wait().ok());
  EXPECT_FALSE(WaitHandle::AlwaysFailing().DebugString().empty());
}

// Tests the basic lifecycle of a permanently signaled wait handle.
TEST(WaitHandleTest, LifecyclePermanentSignaled) {
  // Just to be sure it's ok to safely no-op a WaitHandle value.
  WaitHandle wh_never_used;
  (void)wh_never_used;

  // Try waiting; should return immediately.
  WaitHandle wh0;
  IREE_ASSERT_OK(wh0.Wait());

  // Waits on multiple permanent handles should be ok.
  WaitHandle wh1;
  IREE_ASSERT_OK(WaitHandle::WaitAll({&wh0, &wh1}));
}

// Tests moving permanent WaitHandles around.
TEST(WaitHandleTest, MovePermanent) {
  WaitHandle wh0;
  WaitHandle wh1{std::move(wh0)};
  WaitHandle wh2 = std::move(wh1);
  wh1 = std::move(wh2);
}

// Tests moving around real handles (that may require closing).
TEST(WaitHandleTest, MoveRealHandle) {
  ManualResetEvent fence0;
  WaitHandle wh0 = fence0.OnSet();
  WaitHandle wh1{std::move(wh0)};
  WaitHandle wh2 = std::move(wh1);
  wh1 = std::move(wh2);

  // Now overwrite the handle value to force a close.
  ManualResetEvent fence1;
  WaitHandle wh3 = fence1.OnSet();
  wh1 = std::move(wh3);
  wh1 = WaitHandle();  // Ensure handle dies first.
}

// Tests the various forms of waiting on a single WaitHandle.
// Since these just call WaitAll we leave the involved testing to those.
TEST(WaitHandleTest, SingleWait) {
  WaitHandle wh;
  IREE_ASSERT_OK(wh.Wait());
  IREE_ASSERT_OK(wh.Wait(Now() + absl::Seconds(1)));
  IREE_ASSERT_OK(wh.Wait(absl::Seconds(1)));
  ASSERT_STATUSOR_TRUE(wh.TryWait());
}

// Tests using WaitAll with no valid handles. This should no-op.
TEST(WaitHandleTest, WaitAllNop) {
  IREE_ASSERT_OK(WaitHandle::WaitAll({}));
  IREE_ASSERT_OK(WaitHandle::WaitAll({nullptr}));
  IREE_ASSERT_OK(WaitHandle::WaitAll({nullptr, nullptr}));
}

// Tests polling with WaitAll with multiple wait handles.
TEST(WaitHandleTest, WaitAllPoll) {
  ManualResetEvent fence0;
  WaitHandle wh0 = fence0.OnSet();
  ManualResetEvent fence1;
  WaitHandle wh1 = fence1.OnSet();

  // Poll; should return immediately with timeout.
  ASSERT_TRUE(
      IsDeadlineExceeded(WaitHandle::WaitAll({&wh0, &wh1}, InfinitePast())));

  // Notify fence1.
  IREE_ASSERT_OK(fence1.Set());

  // Poll; should return immediately with timeout as fence1 is not signaled.
  ASSERT_TRUE(
      IsDeadlineExceeded(WaitHandle::WaitAll({&wh0, &wh1}, InfinitePast())));

  // Notify fence0.
  IREE_ASSERT_OK(fence0.Set());

  // Poll again; should return immediately with success.
  IREE_ASSERT_OK(WaitHandle::WaitAll({&wh0, &wh1}, InfinitePast()));
}

// Tests waiting when the first file handle is invalid. This is to verify a
// workaround for bad poll() behavior with fds[0] == -1.
TEST(WaitHandleTest, WaitAllWithInvalid0) {
  ManualResetEvent fence;
  WaitHandle wh = fence.OnSet();

  // Poll; should return immediately with timeout as fence is not signaled.
  ASSERT_TRUE(
      IsDeadlineExceeded(WaitHandle::WaitAll({nullptr, &wh}, InfinitePast())));

  // Notify fence.
  IREE_ASSERT_OK(fence.Set());

  // Poll again; should return immediately with success.
  IREE_ASSERT_OK(WaitHandle::WaitAll({nullptr, &wh}, InfinitePast()));
}

// Tests exceeding the timeout deadline with WaitAll.
TEST(WaitHandleTest, WaitAllTimeout) {
  ManualResetEvent fence;
  WaitHandle wh = fence.OnSet();

  // Wait with timeout on the unsignaled fence:
  // Via polling (should never block):
  ASSERT_TRUE(IsDeadlineExceeded(WaitHandle::WaitAll({&wh}, InfinitePast())));
  ASSERT_STATUSOR_FALSE(WaitHandle::TryWaitAll({&wh}));
  // Via time in the near future (should block):
  ASSERT_TRUE(
      IsDeadlineExceeded(WaitHandle::WaitAll({&wh}, Milliseconds(250))));
  // Via time in the past, should exceed deadline.
  ASSERT_TRUE(
      IsDeadlineExceeded(WaitHandle::WaitAll({&wh}, Milliseconds(-250))));

  // Notify and ensure no more timeouts.
  IREE_ASSERT_OK(fence.Set());
  IREE_ASSERT_OK(WaitHandle::WaitAll({&wh}, InfinitePast()));
  ASSERT_STATUSOR_TRUE(WaitHandle::TryWaitAll({&wh}));
  IREE_ASSERT_OK(WaitHandle::WaitAll({&wh}, Milliseconds(250)));

  // Via time in the past, should exceed deadline even if signaled.
  ASSERT_TRUE(
      IsDeadlineExceeded(WaitHandle::WaitAll({&wh}, Milliseconds(-250))));
}

// Tests using WaitAll to wait on other threads.
TEST(WaitHandleTest, WaitAllThreaded) {
  // Spin up two threads.
  ManualResetEvent fence0;
  std::thread t0{[&]() {
    ::usleep(absl::ToInt64Microseconds(Milliseconds(250)));
    IREE_ASSERT_OK(fence0.Set());
  }};
  ManualResetEvent fence1;
  std::thread t1{[&]() {
    ::usleep(absl::ToInt64Microseconds(Milliseconds(250)));
    IREE_ASSERT_OK(fence1.Set());
  }};

  // Wait on both threads to complete.
  WaitHandle wh0 = fence0.OnSet();
  WaitHandle wh1 = fence1.OnSet();
  IREE_ASSERT_OK(WaitHandle::WaitAll({&wh0, &wh1}));

  t0.join();
  t1.join();
}

// Tests using WaitAll with multiple wait handles from the same fence.
TEST(WaitHandleTest, WaitAllSameSource) {
  ManualResetEvent fence;
  WaitHandle wh0 = fence.OnSet();
  WaitHandle wh1 = fence.OnSet();
  ASSERT_TRUE(
      IsDeadlineExceeded(WaitHandle::WaitAll({&wh0, &wh1}, InfinitePast())));
  IREE_ASSERT_OK(fence.Set());
  IREE_ASSERT_OK(WaitHandle::WaitAll({&wh0, &wh1}));
}

// Tests using WaitAll with literally the same wait handles.
TEST(WaitHandleTest, WaitAllSameHandle) {
  ManualResetEvent fence;
  WaitHandle wh = fence.OnSet();
  ASSERT_TRUE(
      IsDeadlineExceeded(WaitHandle::WaitAll({&wh, &wh}, InfinitePast())));
  IREE_ASSERT_OK(fence.Set());
  IREE_ASSERT_OK(WaitHandle::WaitAll({&wh, &wh}));
}

// Tests WaitAll when a wait handle fails.
TEST(WaitHandleTest, WaitAllFailure) {
  WaitHandle good_wh;
  // Create a purposefully bad handle to induce an error.
  WaitHandle bad_wh = WaitHandle::AlwaysFailing();
  // Should fail with some posixy error.
  ASSERT_FALSE(WaitHandle::WaitAll({&good_wh, &bad_wh}).ok());
}

// Tests using WaitAny with no valid handles. This should no-op.
TEST(WaitHandleTest, WaitAnyNop) {
  ASSERT_TRUE(IsInvalidArgument(WaitHandle::WaitAny({}).status()));
  IREE_ASSERT_OK_AND_ASSIGN(int index, WaitHandle::WaitAny({nullptr}));
  ASSERT_EQ(0, index);
  IREE_ASSERT_OK_AND_ASSIGN(index, WaitHandle::WaitAny({nullptr, nullptr}));
  ASSERT_EQ(0, index);
}

// Tests polling with WaitAny with multiple wait handles.
TEST(WaitHandleTest, WaitAnyPoll) {
  ManualResetEvent fence0;
  WaitHandle wh0 = fence0.OnSet();
  ManualResetEvent fence1;
  WaitHandle wh1 = fence1.OnSet();

  // Poll; should return immediately with timeout.
  ASSERT_TRUE(IsDeadlineExceeded(
      WaitHandle::WaitAny({&wh0, &wh1}, InfinitePast()).status()));

  // Notify fence1.
  IREE_ASSERT_OK(fence1.Set());

  // Poll; should return immediately with fence1 signaled.
  IREE_ASSERT_OK_AND_ASSIGN(int index,
                            WaitHandle::WaitAny({&wh0, &wh1}, InfinitePast()));
  EXPECT_EQ(1, index);

  // Notify fence0.
  IREE_ASSERT_OK(fence0.Set());

  // Poll again; should return immediately; which one is signaled is undefined.
  IREE_ASSERT_OK_AND_ASSIGN(index,
                            WaitHandle::WaitAny({&wh0, &wh1}, InfinitePast()));
  ASSERT_TRUE(index == 0 || index == 1);
}

// Tests exceeding the timeout deadline with WaitAny.
TEST(WaitHandleTest, WaitAnyTimeout) {
  ManualResetEvent fence0;
  WaitHandle wh0 = fence0.OnSet();
  ManualResetEvent fence1;
  WaitHandle wh1 = fence1.OnSet();

  // Wait with timeout on the unsignaled fences:
  // Via polling (should never block):
  ASSERT_TRUE(IsDeadlineExceeded(
      WaitHandle::WaitAny({&wh0, &wh1}, InfinitePast()).status()));
  IREE_ASSERT_OK_AND_ASSIGN(int index, WaitHandle::TryWaitAny({&wh0, &wh1}));
  ASSERT_EQ(-1, index);
  // Via time in the near future (should block):
  ASSERT_TRUE(IsDeadlineExceeded(
      WaitHandle::WaitAny({&wh0, &wh1}, Milliseconds(250)).status()));

  // Notify one of the fences. Should return immediately.
  IREE_ASSERT_OK(fence1.Set());
  IREE_ASSERT_OK_AND_ASSIGN(index,
                            WaitHandle::WaitAny({&wh0, &wh1}, InfinitePast()));
  ASSERT_EQ(1, index);
  IREE_ASSERT_OK_AND_ASSIGN(index, WaitHandle::TryWaitAny({&wh0, &wh1}));
  ASSERT_EQ(1, index);
  IREE_ASSERT_OK_AND_ASSIGN(
      index, WaitHandle::WaitAny({&wh0, &wh1}, Milliseconds(250)));
  ASSERT_EQ(1, index);

  // The unnotified fence should still timeout.
  ASSERT_TRUE(
      IsDeadlineExceeded(WaitHandle::WaitAny({&wh0}, InfinitePast()).status()));
  IREE_ASSERT_OK_AND_ASSIGN(index, WaitHandle::TryWaitAny({&wh0}));
  ASSERT_EQ(-1, index);
  ASSERT_TRUE(IsDeadlineExceeded(
      WaitHandle::WaitAny({&wh0}, Milliseconds(250)).status()));

  // Notify last fence and ensure complete.
  IREE_ASSERT_OK(fence0.Set());
  IREE_ASSERT_OK_AND_ASSIGN(index, WaitHandle::WaitAny({&wh0}, InfinitePast()));
  ASSERT_EQ(0, index);
  IREE_ASSERT_OK_AND_ASSIGN(index, WaitHandle::TryWaitAny({&wh0}));
  ASSERT_EQ(0, index);
  IREE_ASSERT_OK_AND_ASSIGN(index,
                            WaitHandle::WaitAny({&wh0}, Milliseconds(250)));
  ASSERT_EQ(0, index);
}

// Tests using WaitAny to wait on other threads.
TEST(WaitHandleTest, WaitAnyThreaded) {
  // Spin up two threads.
  // t1 will wait on t0 such that they will act in sequence.
  ManualResetEvent fence0;
  std::thread t0{[&]() {
    ::usleep(absl::ToInt64Microseconds(Milliseconds(250)));
    IREE_ASSERT_OK(fence0.Set());
  }};
  ManualResetEvent fence1;
  std::thread t1{[&]() {
    IREE_ASSERT_OK(fence0.OnSet().Wait());
    ::usleep(absl::ToInt64Microseconds(Milliseconds(250)));
    IREE_ASSERT_OK(fence1.Set());
  }};

  // Wait on both threads. We expect 0 to complete first.
  WaitHandle wh0 = fence0.OnSet();
  WaitHandle wh1 = fence1.OnSet();
  IREE_ASSERT_OK_AND_ASSIGN(int index, WaitHandle::WaitAny({&wh0, &wh1}));
  ASSERT_EQ(0, index);

  // Now wait for thread 1.
  IREE_ASSERT_OK_AND_ASSIGN(index, WaitHandle::WaitAny({&wh1}));
  ASSERT_EQ(0, index);

  t0.join();
  t1.join();
}

// Tests using WaitAny with multiple wait handles from the same fence.
TEST(WaitHandleTest, WaitAnySameSource) {
  ManualResetEvent fence;
  WaitHandle wh0 = fence.OnSet();
  WaitHandle wh1 = fence.OnSet();
  ASSERT_TRUE(IsDeadlineExceeded(
      WaitHandle::WaitAny({&wh0, &wh1}, InfinitePast()).status()));
  IREE_ASSERT_OK(fence.Set());
  IREE_ASSERT_OK_AND_ASSIGN(int index, WaitHandle::WaitAny({&wh0, &wh1}));
  ASSERT_TRUE(index == 0 || index == 1);
}

// Tests using WaitAny with literally the same wait handles.
TEST(WaitHandleTest, WaitAnySameHandle) {
  ManualResetEvent fence;
  WaitHandle wh = fence.OnSet();
  ASSERT_TRUE(IsDeadlineExceeded(
      WaitHandle::WaitAny({&wh, &wh}, InfinitePast()).status()));
  IREE_ASSERT_OK(fence.Set());
  IREE_ASSERT_OK_AND_ASSIGN(int index, WaitHandle::WaitAny({&wh, &wh}));
  ASSERT_TRUE(index == 0 || index == 1);
}

// Tests WaitAny when a wait handle fails.
TEST(WaitHandleTest, WaitAnyFailure) {
  WaitHandle good_wh;
  // Create a purposefully bad handle to induce an error.
  WaitHandle bad_wh = WaitHandle::AlwaysFailing();
  // Should fail with some posixy error.
  ASSERT_FALSE(WaitHandle::WaitAny({&good_wh, &bad_wh}).ok());
}

// ManualResetEvent with innards exposed. Meh.
class ExposedManualResetEvent : public ManualResetEvent {
 public:
  using ManualResetEvent::AcquireFdForWait;
  using ManualResetEvent::TryResolveWakeOnFd;
};

// Mock type for the WaitableObject methods.
class MockWaitableObject : public ::testing::StrictMock<WaitableObject> {
 public:
  MockWaitableObject() : ::testing::StrictMock<WaitableObject>() {}

  MOCK_METHOD(std::string, DebugString, (), (const, override));
  MOCK_METHOD((StatusOr<std::pair<FdType, int>>), AcquireFdForWait,
              (Time deadline_ns), (override));
  MOCK_METHOD(StatusOr<bool>, TryResolveWakeOnFd, (int fd), (override));

  WaitHandle OnSomething() { return WaitHandle(add_ref(this)); }
};

// Tests normal AcquireFdForWait + TryResolveWakeOnFd use.
TEST(WaitableObjectTest, AcquireAndResolve) {
  MockWaitableObject mwo;
  WaitHandle wh = mwo.OnSomething();

  // Use a MRE for testing, as we can just use its fd.
  ExposedManualResetEvent mre;

  // Try waiting; we should see the AcquireFdForWait and then return because
  // the fd has not been resolved.
  EXPECT_CALL(mwo, AcquireFdForWait(_)).WillOnce([&](Time deadline_ns) {
    // Return the valid FD from the MRE.
    return mre.AcquireFdForWait(deadline);
  });
  ASSERT_STATUSOR_FALSE(wh.TryWait());

  // Signal the MRE.
  IREE_ASSERT_OK(mre.Set());

  // Try waiting again; we should get the AcquireFdForWait and then also get
  // the TryResolveWakeOnFd.
  EXPECT_CALL(mwo, AcquireFdForWait(_)).WillOnce([&](Time deadline_ns) {
    // Return the valid (and now signaled) FD from the MRE.
    return mre.AcquireFdForWait(deadline);
  });
  EXPECT_CALL(mwo, TryResolveWakeOnFd(_)).WillOnce(Return(true));
  ASSERT_STATUSOR_TRUE(wh.TryWait());
}

// Tests timing out in AcquireFdForWait.
TEST(WaitableObjectTest, AcquireFdForWaitTimeout) {
  ManualResetEvent mre;
  WaitHandle always_wait = mre.OnSet();
  WaitHandle always_signal = WaitHandle::AlwaysSignaling();
  MockWaitableObject mwo;
  WaitHandle wh = mwo.OnSomething();

  // Make the AcquireFdForWait take longer than the timeout. We should hit
  // deadline exceeded even though always_wait hasn't be signaled.
  EXPECT_CALL(mwo, AcquireFdForWait(_)).WillOnce([](Time deadline_ns) {
    ::usleep(absl::ToInt64Microseconds(Milliseconds(10)));
    return std::make_pair(WaitableObject::FdType::kPermanent,
                          WaitableObject::kInvalidFd);
  });
  ASSERT_TRUE(IsDeadlineExceeded(
      WaitHandle::WaitAll({&wh, &always_signal}, Now() - Milliseconds(250))));
}

// Tests TryResolveWakeOnFd when a handle is a permanent kSignaledFd.
TEST(WaitableObjectTest, SignaledFd) {
  MockWaitableObject mwo;
  WaitHandle wh = mwo.OnSomething();

  // Return the kSignaledFd handle and expect that we still get our notify call.
  // We can do this multiple times.
  for (int i = 0; i < 4; ++i) {
    EXPECT_CALL(mwo, AcquireFdForWait(_))
        .WillOnce(Return(std::make_pair(WaitableObject::FdType::kPermanent,
                                        WaitableObject::kSignaledFd)));
    EXPECT_CALL(mwo, TryResolveWakeOnFd(WaitableObject::kSignaledFd))
        .WillOnce(Return(true));
    ASSERT_STATUSOR_TRUE(wh.TryWait());
  }
}

// Tests that waiting will not resolve if TryResolveWakeOnFd returns false.
TEST(WaitableObjectTest, UnresolvedWake) {
  MockWaitableObject mwo;
  WaitHandle wh = mwo.OnSomething();

  // Fail to resolve the first time.
  // Since we are only trying to wait it should bail.
  EXPECT_CALL(mwo, AcquireFdForWait(_))
      .WillOnce(Return(std::make_pair(WaitableObject::FdType::kPermanent,
                                      WaitableObject::kSignaledFd)));
  EXPECT_CALL(mwo, TryResolveWakeOnFd(WaitableObject::kSignaledFd))
      .WillOnce(Return(false));
  ASSERT_STATUSOR_FALSE(wh.TryWait());

  // Resolve on the next try.
  EXPECT_CALL(mwo, AcquireFdForWait(_))
      .WillOnce(Return(std::make_pair(WaitableObject::FdType::kPermanent,
                                      WaitableObject::kSignaledFd)));
  EXPECT_CALL(mwo, TryResolveWakeOnFd(WaitableObject::kSignaledFd))
      .WillOnce(Return(true));
  ASSERT_STATUSOR_TRUE(wh.TryWait());
}

// Tests the normal lifecycle of a ManualResetEvent.
TEST(ManualResetEventTest, Lifecycle) {
  ManualResetEvent ev;
  EXPECT_FALSE(ev.DebugString().empty());
  WaitHandle wh0 = ev.OnSet();
  EXPECT_EQ(ev.DebugString(), wh0.DebugString());
  WaitHandle wh1 = ev.OnSet();
  EXPECT_EQ(ev.DebugString(), wh1.DebugString());
  // Should not be set.
  ASSERT_STATUSOR_FALSE(wh0.TryWait());
  ASSERT_STATUSOR_FALSE(wh1.TryWait());
  // Set should be sticky.
  IREE_ASSERT_OK(ev.Set());
  ASSERT_STATUSOR_TRUE(wh0.TryWait());
  ASSERT_STATUSOR_TRUE(wh1.TryWait());
  // Reset should clear.
  IREE_ASSERT_OK(ev.Reset());
  ASSERT_STATUSOR_FALSE(wh0.TryWait());
  ASSERT_STATUSOR_FALSE(wh1.TryWait());
  // Setting again should enable the previous WaitHandles to be signaled.
  IREE_ASSERT_OK(ev.Set());
  ASSERT_STATUSOR_TRUE(wh0.TryWait());
  ASSERT_STATUSOR_TRUE(wh1.TryWait());
}

// Tests moving ManualResetEvents around.
TEST(ManualResetEventTest, Move) {
  ManualResetEvent ev0;
  WaitHandle wh = ev0.OnSet();
  ManualResetEvent ev1{std::move(ev0)};
  ManualResetEvent ev2 = std::move(ev1);
  ev1 = std::move(ev2);
  IREE_ASSERT_OK(ev1.Set());
  ASSERT_STATUSOR_TRUE(wh.TryWait());
}

// Tests redundantly setting and resetting ManualResetEvents.
TEST(ManualResetEventTest, RedundantUse) {
  ManualResetEvent ev;
  IREE_ASSERT_OK(ev.Reset());
  IREE_ASSERT_OK(ev.Reset());
  ASSERT_FALSE(ev.OnSet().TryWait().value());
  IREE_ASSERT_OK(ev.Set());
  IREE_ASSERT_OK(ev.Set());
  ASSERT_TRUE(ev.OnSet().TryWait().value());
  IREE_ASSERT_OK(ev.Reset());
  ASSERT_FALSE(ev.OnSet().TryWait().value());
}

// Tests waiting on an initially-set ManualResetEvent;
TEST(ManualResetEventTest, SetThenWait) {
  ManualResetEvent ev;
  IREE_ASSERT_OK(ev.Set());
  ASSERT_TRUE(ev.OnSet().TryWait().value());
}

// Tests that dangling an event will not wake waiters.
// This is intentional (for now); we could with a bit of wrangling make it so
// that WaitableObjects tracked their waiters and ensured they were all cleaned
// up, but that seems hard. Don't drop your objects.
TEST(ManualResetEventTest, NeverSet) {
  ManualResetEvent ev;
  WaitHandle wh = ev.OnSet();
  ASSERT_STATUSOR_FALSE(wh.TryWait());
  // Kill event to unblock waiters.
  ev = ManualResetEvent();
  // Waiter should not have woken.
  ASSERT_STATUSOR_FALSE(wh.TryWait());
}

}  // namespace
}  // namespace iree
