// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/loop.h"

// NOTE: this file is meant to be included inside of a _test.cc source file.
// The file must define these functions to allocate/free the loop.
// |out_status| should receive the last global error encountered in the loop.
void AllocateLoop(iree_status_t* out_status, iree_allocator_t allocator,
                  iree_vm_loop_t* out_loop);
void FreeLoop(iree_allocator_t allocator, iree_vm_loop_t loop);

namespace iree {
namespace testing {

struct LoopTest : public ::testing::Test {
  iree_allocator_t allocator = iree_allocator_system();
  iree_vm_loop_t loop;
  iree_status_t loop_status = iree_ok_status();

  void SetUp() override {
    IREE_TRACE_SCOPE();
    AllocateLoop(&loop_status, allocator, &loop);
  }
  void TearDown() override {
    IREE_TRACE_SCOPE();
    FreeLoop(allocator, loop);
    iree_status_ignore(loop_status);
  }
};

//===----------------------------------------------------------------------===//
// iree_vm_loop_call
//===----------------------------------------------------------------------===//

// Tests the simple call interface for running work.
TEST_F(LoopTest, Call) {
  IREE_TRACE_SCOPE();
  struct UserData {
    iree_status_t call_status = iree_status_from_code(IREE_STATUS_DATA_LOSS);
  } user_data;
  IREE_ASSERT_OK(iree_vm_loop_call(
      loop, IREE_VM_LOOP_PRIORITY_DEFAULT,
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_OK(status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        user_data->call_status = status;
        return iree_ok_status();
      },
      &user_data));
  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));
  IREE_ASSERT_OK(loop_status);
  IREE_ASSERT_OK(user_data.call_status);
}

// Tests a call that forks into two other calls.
TEST_F(LoopTest, CallFork) {
  IREE_TRACE_SCOPE();
  struct UserData {
    bool called_a = false;
    bool called_b = false;
    bool called_c = false;
  } user_data;

  // A -> [B, C]
  IREE_ASSERT_OK(iree_vm_loop_call(
      loop, IREE_VM_LOOP_PRIORITY_DEFAULT,
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_OK(status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        user_data->called_a = true;

        // B
        IREE_EXPECT_OK(iree_vm_loop_call(
            loop, IREE_VM_LOOP_PRIORITY_DEFAULT,
            +[](void* user_data_ptr, iree_vm_loop_t loop,
                iree_status_t status) {
              IREE_TRACE_SCOPE();
              IREE_EXPECT_OK(status);
              auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
              user_data->called_b = true;
              return iree_ok_status();
            },
            user_data));

        // C
        IREE_EXPECT_OK(iree_vm_loop_call(
            loop, IREE_VM_LOOP_PRIORITY_DEFAULT,
            +[](void* user_data_ptr, iree_vm_loop_t loop,
                iree_status_t status) {
              IREE_TRACE_SCOPE();
              IREE_EXPECT_OK(status);
              auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
              user_data->called_c = true;
              return iree_ok_status();
            },
            user_data));

        return iree_ok_status();
      },
      &user_data));

  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));
  IREE_ASSERT_OK(loop_status);
  EXPECT_TRUE(user_data.called_a);
  EXPECT_TRUE(user_data.called_b);
  EXPECT_TRUE(user_data.called_c);
}

// Tests a repeating call - since the loops are intended to be stackless we
// should in theory be able to issue calls forever. This test ensures we can do
// a really large amount without blowing the native stack.
struct CallRepeatedData {
  int remaining = 2 * 1024;
};
static iree_status_t CallRepeatedFn(void* user_data_ptr, iree_vm_loop_t loop,
                                    iree_status_t status) {
  IREE_TRACE_SCOPE();
  IREE_EXPECT_OK(status);
  auto* user_data = reinterpret_cast<CallRepeatedData*>(user_data_ptr);
  if (--user_data->remaining) {
    IREE_RETURN_IF_ERROR(iree_vm_loop_call(loop, IREE_VM_LOOP_PRIORITY_DEFAULT,
                                           CallRepeatedFn, user_data));
  }
  return iree_ok_status();
}
TEST_F(LoopTest, CallRepeated) {
  IREE_TRACE_SCOPE();
  CallRepeatedData user_data;
  IREE_ASSERT_OK(iree_vm_loop_call(loop, IREE_VM_LOOP_PRIORITY_DEFAULT,
                                   CallRepeatedFn, &user_data));
  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));
  IREE_ASSERT_OK(loop_status);
  EXPECT_EQ(user_data.remaining, 0);
}

// Tests a call that results in failure.
TEST_F(LoopTest, CallFailure) {
  IREE_TRACE_SCOPE();
  struct UserData {
    bool completed = false;
  } user_data;
  IREE_ASSERT_OK(iree_vm_loop_call(
      loop, IREE_VM_LOOP_PRIORITY_DEFAULT,
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_OK(status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        EXPECT_FALSE(user_data->completed);
        user_data->completed = true;
        return iree_status_from_code(IREE_STATUS_DATA_LOSS);
      },
      &user_data));
  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, loop_status);
}

// Tests that a failure will abort other pending tasks.
TEST_F(LoopTest, CallFailureAborts) {
  IREE_TRACE_SCOPE();
  struct UserData {
    bool did_call_callback = false;
    bool did_wait_callback = false;
  } user_data;

  // Issue the call that will fail.
  IREE_ASSERT_OK(iree_vm_loop_call(
      loop, IREE_VM_LOOP_PRIORITY_DEFAULT,
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_OK(status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        EXPECT_FALSE(user_data->did_call_callback);
        user_data->did_call_callback = true;
        return iree_status_from_code(IREE_STATUS_DATA_LOSS);
      },
      &user_data));

  // Enqueue a wait that will never complete - if it runs it means we didn't
  // correctly abort it.
  IREE_ASSERT_OK(iree_vm_loop_wait_until(
      loop, iree_make_timeout_ms(1 * 60 * 1000),
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, status);
        iree_status_ignore(status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        EXPECT_FALSE(user_data->did_wait_callback);
        user_data->did_wait_callback = true;
        return iree_ok_status();
      },
      &user_data));

  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, loop_status);
  EXPECT_TRUE(user_data.did_call_callback);
  EXPECT_TRUE(user_data.did_wait_callback);
}

// Tests that a failure will abort other pending tasks, including those enqueued
// from within the failing call itself.
TEST_F(LoopTest, CallFailureAbortsNested) {
  IREE_TRACE_SCOPE();
  struct UserData {
    bool did_call_callback = false;
    bool did_wait_callback = false;
  } user_data;

  // Issue the call that will fail.
  IREE_ASSERT_OK(iree_vm_loop_call(
      loop, IREE_VM_LOOP_PRIORITY_DEFAULT,
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_OK(status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        EXPECT_FALSE(user_data->did_call_callback);
        user_data->did_call_callback = true;

        // Enqueue a wait that will never complete - if it runs it means we
        // didn't correctly abort it. We are enqueuing it reentrantly as a user
        // would before we encounter the error below.
        IREE_EXPECT_OK(iree_vm_loop_wait_until(
            loop, iree_make_timeout_ms(1 * 60 * 1000),
            +[](void* user_data_ptr, iree_vm_loop_t loop,
                iree_status_t status) {
              IREE_TRACE_SCOPE();
              IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, status);
              iree_status_ignore(status);
              auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
              EXPECT_FALSE(user_data->did_wait_callback);
              user_data->did_wait_callback = true;
              return iree_ok_status();
            },
            user_data));

        return iree_status_from_code(IREE_STATUS_DATA_LOSS);
      },
      &user_data));

  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, loop_status);
  EXPECT_TRUE(user_data.did_call_callback);
  EXPECT_TRUE(user_data.did_wait_callback);
}

//===----------------------------------------------------------------------===//
// iree_vm_loop_wait_until
//===----------------------------------------------------------------------===//

// Tests a wait-until delay with an immediate timeout.
TEST_F(LoopTest, WaitUntilImmediate) {
  IREE_TRACE_SCOPE();
  struct UserData {
    iree_status_t wait_status = iree_status_from_code(IREE_STATUS_DATA_LOSS);
  } user_data;
  IREE_ASSERT_OK(iree_vm_loop_wait_until(
      loop, iree_immediate_timeout(),
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_OK(status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        user_data->wait_status = status;
        return iree_ok_status();
      },
      &user_data));
  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));
  IREE_ASSERT_OK(loop_status);
  IREE_ASSERT_OK(user_data.wait_status);
}

// Tests a wait-until delay with an actual delay.
TEST_F(LoopTest, WaitUntil) {
  IREE_TRACE_SCOPE();
  struct UserData {
    iree_time_t start_ns = iree_time_now();
    iree_time_t end_ns = IREE_TIME_INFINITE_FUTURE;
    iree_status_t wait_status = iree_status_from_code(IREE_STATUS_DATA_LOSS);
  } user_data;
  IREE_ASSERT_OK(iree_vm_loop_wait_until(
      loop, iree_make_timeout_ms(50),
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_OK(status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        user_data->end_ns = iree_time_now();
        user_data->wait_status = status;
        return iree_ok_status();
      },
      &user_data));
  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));
  IREE_ASSERT_OK(loop_status);
  IREE_ASSERT_OK(user_data.wait_status);
  // Not checking exact timing as some devices may not have clocks.
  EXPECT_GE(user_data.end_ns, user_data.start_ns);
}

// Tests that multiple wait-until's can be active at once.
// NOTE: loops are not required to wake in any particular order.
TEST_F(LoopTest, MultiWaitUntil) {
  IREE_TRACE_SCOPE();
  struct UserData {
    bool woke_a = false;
    bool woke_b = false;
  } user_data;

  IREE_ASSERT_OK(iree_vm_loop_wait_until(
      loop, iree_make_timeout_ms(25),
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_OK(status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        user_data->woke_a = true;
        return iree_ok_status();
      },
      &user_data));

  IREE_ASSERT_OK(iree_vm_loop_wait_until(
      loop, iree_make_timeout_ms(50),
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_OK(status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        user_data->woke_b = true;
        return iree_ok_status();
      },
      &user_data));

  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));
  IREE_ASSERT_OK(loop_status);
  EXPECT_TRUE(user_data.woke_a);
  EXPECT_TRUE(user_data.woke_b);
}

//===----------------------------------------------------------------------===//
// iree_vm_loop_wait_one
//===----------------------------------------------------------------------===//

// Tests a wait-one with an immediate timeout.
// The wait source never resolves and if we didn't bail immediately we'd hang.
TEST_F(LoopTest, WaitOneImmediate) {
  IREE_TRACE_SCOPE();

  // A delay far in the future that will never resolve within the test.
  iree_wait_source_t wait_source =
      iree_wait_source_delay(IREE_TIME_INFINITE_FUTURE);

  struct UserData {
    bool did_wait_callback = false;
  } user_data;
  IREE_ASSERT_OK(iree_vm_loop_wait_one(
      loop, wait_source, iree_immediate_timeout(),
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED, status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        user_data->did_wait_callback = true;
        return iree_ok_status();
      },
      &user_data));
  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));

  IREE_ASSERT_OK(loop_status);
  EXPECT_TRUE(user_data.did_wait_callback);
}

// Tests a wait-one with a non-immediate timeout.
TEST_F(LoopTest, WaitOneTimeout) {
  IREE_TRACE_SCOPE();

  // A delay far in the future that will never resolve within the test.
  iree_wait_source_t wait_source =
      iree_wait_source_delay(IREE_TIME_INFINITE_FUTURE);

  struct UserData {
    bool did_wait_callback = false;
  } user_data;
  IREE_ASSERT_OK(iree_vm_loop_wait_one(
      loop, wait_source, iree_make_timeout_ms(10),
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED, status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        user_data->did_wait_callback = true;
        return iree_ok_status();
      },
      &user_data));
  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));

  IREE_ASSERT_OK(loop_status);
  EXPECT_TRUE(user_data.did_wait_callback);
}

// Tests a wait-one that times out does not abort other loop ops.
// The deadline exceeded status passed to the callback is sufficient.
TEST_F(LoopTest, WaitOneTimeoutNoAbort) {
  IREE_TRACE_SCOPE();

  // A delay far in the future that will never resolve within the test.
  iree_wait_source_t wait_source =
      iree_wait_source_delay(IREE_TIME_INFINITE_FUTURE);

  struct UserData {
    bool did_wait_callback = false;
    bool did_call_callback = false;
  } user_data;

  // Wait that will time out.
  IREE_ASSERT_OK(iree_vm_loop_wait_one(
      loop, wait_source, iree_make_timeout_ms(10),
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED, status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        user_data->did_wait_callback = true;

        // Call that should still be issued correctly.
        // Note that we queue it here as if we did it outside the wait we'd
        // immediately execute it on out-of-order implementations.
        IREE_EXPECT_OK(iree_vm_loop_call(
            loop, IREE_VM_LOOP_PRIORITY_DEFAULT,
            +[](void* user_data_ptr, iree_vm_loop_t loop,
                iree_status_t status) {
              IREE_TRACE_SCOPE();
              IREE_EXPECT_OK(status);
              auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
              EXPECT_FALSE(user_data->did_call_callback);
              user_data->did_call_callback = true;
              return iree_ok_status();
            },
            user_data));

        return iree_ok_status();
      },
      &user_data));

  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));

  IREE_ASSERT_OK(loop_status);
  EXPECT_TRUE(user_data.did_wait_callback);
  EXPECT_TRUE(user_data.did_call_callback);
}

// Tests a wait-one with an already signaled wait source.
TEST_F(LoopTest, WaitOneSignaled) {
  IREE_TRACE_SCOPE();

  // An immediately resolved wait source.
  iree_wait_source_t wait_source = iree_wait_source_immediate();

  struct UserData {
    bool did_wait_callback = false;
  } user_data;
  IREE_ASSERT_OK(iree_vm_loop_wait_one(
      loop, wait_source, iree_make_timeout_ms(10),
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_OK(status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        user_data->did_wait_callback = true;
        return iree_ok_status();
      },
      &user_data));
  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));

  IREE_ASSERT_OK(loop_status);
  EXPECT_TRUE(user_data.did_wait_callback);
}

// Tests a wait-one on a wait source that resolves after a short delay.
TEST_F(LoopTest, WaitOneBlocking) {
  IREE_TRACE_SCOPE();

  // A delay that resolves after 50ms.
  iree_wait_source_t wait_source =
      iree_wait_source_delay(iree_time_now() + 50 * 1000000LL);

  struct UserData {
    bool did_wait_callback = false;
  } user_data;
  IREE_ASSERT_OK(iree_vm_loop_wait_one(
      loop, wait_source, iree_make_timeout_ms(2000),
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_OK(status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        user_data->did_wait_callback = true;
        return iree_ok_status();
      },
      &user_data));
  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));

  IREE_ASSERT_OK(loop_status);
  EXPECT_TRUE(user_data.did_wait_callback);
}

//===----------------------------------------------------------------------===//
// iree_vm_loop_wait_any
//===----------------------------------------------------------------------===//

// Tests a wait-any with a immediate timeout (a poll).
TEST_F(LoopTest, WaitAnyImmediate) {
  IREE_TRACE_SCOPE();

  // Delays that never resolve within the test.
  iree_wait_source_t wait_sources[2] = {
      iree_wait_source_delay(IREE_TIME_INFINITE_FUTURE),
      iree_wait_source_delay(IREE_TIME_INFINITE_FUTURE),
  };
  struct UserData {
    bool did_wait_callback = false;
  } user_data;
  IREE_ASSERT_OK(iree_vm_loop_wait_any(
      loop, IREE_ARRAYSIZE(wait_sources), wait_sources,
      iree_immediate_timeout(),
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED, status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        user_data->did_wait_callback = true;
        return iree_ok_status();
      },
      &user_data));
  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));

  IREE_ASSERT_OK(loop_status);
  EXPECT_TRUE(user_data.did_wait_callback);
}

// Tests a wait-any with a non-immediate timeout.
TEST_F(LoopTest, WaitAnyTimeout) {
  IREE_TRACE_SCOPE();

  // Delays that never resolve within the test.
  iree_wait_source_t wait_sources[2] = {
      iree_wait_source_delay(IREE_TIME_INFINITE_FUTURE),
      iree_wait_source_delay(IREE_TIME_INFINITE_FUTURE),
  };
  struct UserData {
    bool did_wait_callback = false;
  } user_data;
  IREE_ASSERT_OK(iree_vm_loop_wait_any(
      loop, IREE_ARRAYSIZE(wait_sources), wait_sources,
      iree_make_timeout_ms(10),
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED, status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        user_data->did_wait_callback = true;
        return iree_ok_status();
      },
      &user_data));
  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));

  IREE_ASSERT_OK(loop_status);
  EXPECT_TRUE(user_data.did_wait_callback);
}

// Tests a wait-any with an already-resolved wait source.
TEST_F(LoopTest, WaitAnySignaled) {
  IREE_TRACE_SCOPE();

  // One immediately resolved, one never-resolving.
  iree_wait_source_t wait_sources[2] = {
      iree_wait_source_immediate(),
      iree_wait_source_delay(IREE_TIME_INFINITE_FUTURE),
  };
  struct UserData {
    bool did_wait_callback = false;
  } user_data;
  IREE_ASSERT_OK(iree_vm_loop_wait_any(
      loop, IREE_ARRAYSIZE(wait_sources), wait_sources,
      iree_make_timeout_ms(10),
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_OK(status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        user_data->did_wait_callback = true;
        return iree_ok_status();
      },
      &user_data));
  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));

  IREE_ASSERT_OK(loop_status);
  EXPECT_TRUE(user_data.did_wait_callback);
}

// Tests a wait-any with a wait source that resolves after a short delay.
TEST_F(LoopTest, WaitAnyBlocking) {
  IREE_TRACE_SCOPE();

  // One resolves after 50ms, one never resolves.
  iree_wait_source_t wait_sources[2] = {
      iree_wait_source_delay(iree_time_now() + 50 * 1000000LL),
      iree_wait_source_delay(IREE_TIME_INFINITE_FUTURE),
  };
  struct UserData {
    bool did_wait_callback = false;
  } user_data;
  IREE_ASSERT_OK(iree_vm_loop_wait_any(
      loop, IREE_ARRAYSIZE(wait_sources), wait_sources,
      iree_make_timeout_ms(2000),
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_OK(status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        user_data->did_wait_callback = true;
        return iree_ok_status();
      },
      &user_data));
  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));

  IREE_ASSERT_OK(loop_status);
  EXPECT_TRUE(user_data.did_wait_callback);
}

//===----------------------------------------------------------------------===//
// iree_vm_loop_wait_all
//===----------------------------------------------------------------------===//

// Tests a wait-all with a immediate timeout (a poll).
TEST_F(LoopTest, WaitAllImmediate) {
  IREE_TRACE_SCOPE();

  // One unresolved and one resolved (should fail the wait-all).
  iree_wait_source_t wait_sources[2] = {
      iree_wait_source_delay(IREE_TIME_INFINITE_FUTURE),
      iree_wait_source_immediate(),
  };
  struct UserData {
    bool did_wait_callback = false;
  } user_data;
  IREE_ASSERT_OK(iree_vm_loop_wait_all(
      loop, IREE_ARRAYSIZE(wait_sources), wait_sources,
      iree_immediate_timeout(),
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED, status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        user_data->did_wait_callback = true;
        return iree_ok_status();
      },
      &user_data));
  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));

  IREE_ASSERT_OK(loop_status);
  EXPECT_TRUE(user_data.did_wait_callback);
}

// Tests a wait-all with a non-immediate timeout.
TEST_F(LoopTest, WaitAllTimeout) {
  IREE_TRACE_SCOPE();

  // One unresolved and one resolved (should fail the wait-all).
  iree_wait_source_t wait_sources[2] = {
      iree_wait_source_delay(IREE_TIME_INFINITE_FUTURE),
      iree_wait_source_immediate(),
  };
  struct UserData {
    bool did_wait_callback = false;
  } user_data;
  IREE_ASSERT_OK(iree_vm_loop_wait_all(
      loop, IREE_ARRAYSIZE(wait_sources), wait_sources,
      iree_make_timeout_ms(10),
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED, status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        user_data->did_wait_callback = true;
        return iree_ok_status();
      },
      &user_data));
  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));

  IREE_ASSERT_OK(loop_status);
  EXPECT_TRUE(user_data.did_wait_callback);
}

// Tests a wait-all with already-resolved wait sources.
TEST_F(LoopTest, WaitAllSignaled) {
  IREE_TRACE_SCOPE();

  // Both immediately resolved.
  iree_wait_source_t wait_sources[2] = {
      iree_wait_source_immediate(),
      iree_wait_source_immediate(),
  };
  struct UserData {
    bool did_wait_callback = false;
  } user_data;
  IREE_ASSERT_OK(iree_vm_loop_wait_all(
      loop, IREE_ARRAYSIZE(wait_sources), wait_sources,
      iree_make_timeout_ms(10),
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_OK(status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        user_data->did_wait_callback = true;
        return iree_ok_status();
      },
      &user_data));
  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));

  IREE_ASSERT_OK(loop_status);
  EXPECT_TRUE(user_data.did_wait_callback);
}

// Tests a wait-all with wait sources that resolve after short delays.
TEST_F(LoopTest, WaitAllBlocking) {
  IREE_TRACE_SCOPE();

  // One resolves after 50ms, one is already resolved.
  iree_wait_source_t wait_sources[2] = {
      iree_wait_source_delay(iree_time_now() + 50 * 1000000LL),
      iree_wait_source_immediate(),
  };
  struct UserData {
    bool did_wait_callback = false;
  } user_data;
  IREE_ASSERT_OK(iree_vm_loop_wait_all(
      loop, IREE_ARRAYSIZE(wait_sources), wait_sources,
      iree_make_timeout_ms(2000),
      +[](void* user_data_ptr, iree_vm_loop_t loop, iree_status_t status) {
        IREE_TRACE_SCOPE();
        IREE_EXPECT_OK(status);
        auto* user_data = reinterpret_cast<UserData*>(user_data_ptr);
        user_data->did_wait_callback = true;
        return iree_ok_status();
      },
      &user_data));
  IREE_ASSERT_OK(iree_vm_loop_drain(loop, iree_infinite_timeout()));

  IREE_ASSERT_OK(loop_status);
  EXPECT_TRUE(user_data.did_wait_callback);
}

}  // namespace testing
}  // namespace iree
