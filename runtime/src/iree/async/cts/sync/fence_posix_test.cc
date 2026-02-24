// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// POSIX-specific fence CTS tests.
//
// Tests for device fence import/export using POSIX file descriptors.
// Uses eventfd (Linux) or pipe (other POSIX) to simulate device fences
// without requiring actual GPU hardware.
//
// Device fence bridging enables GPU↔proactor synchronization:
//   - import_fence: GPU completion → semaphore signal
//   - export_fence: semaphore signal → GPU wait
//
// This allows zero-copy pipelines where network I/O and GPU work are
// synchronized without host-side round-trips.

#include <poll.h>
#include <unistd.h>

#include <thread>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/primitive.h"
#include "iree/async/semaphore.h"

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
#include <sys/eventfd.h>
#define IREE_CTS_HAVE_EVENTFD 1
#endif

namespace iree::async::cts {

class FencePosixTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase<>::SetUp();
    if (!iree_any_bit_set(capabilities_,
                          IREE_ASYNC_PROACTOR_CAPABILITY_DEVICE_FENCE)) {
      GTEST_SKIP() << "backend lacks device fence capability";
    }
  }

  // Creates a fence fd for testing.
  // On Linux, uses eventfd. On other POSIX, uses the read end of a pipe.
  // Returns the fence fd and optionally the write end for signaling.
  // Caller owns both fds and must close them.
  void CreateTestFence(int* out_fence_fd, int* out_signal_fd) {
#if defined(IREE_CTS_HAVE_EVENTFD)
    // eventfd is ideal: single fd that can be both signaled and polled.
    int efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    ASSERT_GE(efd, 0) << "eventfd failed: " << strerror(errno);
    *out_fence_fd = efd;
    *out_signal_fd = efd;  // Same fd for eventfd.
#else
    // Fallback: pipe pair. Write end signals, read end is the fence.
    int pipefd[2];
    ASSERT_EQ(pipe(pipefd), 0) << "pipe failed: " << strerror(errno);
    *out_fence_fd = pipefd[0];   // Read end is the fence.
    *out_signal_fd = pipefd[1];  // Write end for signaling.
#endif
  }

  // Creates a test fence with a separate signaler fd.
  // The fence_fd is suitable for import_fence (proactor takes ownership and
  // closes it). The signaler_fd is a distinct fd that the caller can use to
  // signal the fence via SignalTestFence() and must close after use.
  //
  // On Linux (eventfd): dup()s the eventfd so proactor and signaler each own
  // a separate fd to the same underlying counter.
  // On other POSIX (pipe): fence_fd is the read end, signaler_fd is the
  // write end — already distinct, no dup needed.
  void CreateTestFenceWithSignaler(int* out_fence_fd, int* out_signaler_fd) {
    int fence_fd = -1;
    int signal_fd = -1;
    CreateTestFence(&fence_fd, &signal_fd);
#if defined(IREE_CTS_HAVE_EVENTFD)
    *out_fence_fd = fence_fd;
    *out_signaler_fd = dup(signal_fd);
    ASSERT_GE(*out_signaler_fd, 0) << "dup failed: " << strerror(errno);
#else
    *out_fence_fd = fence_fd;
    *out_signaler_fd = signal_fd;
#endif
  }

  // Signals the test fence (makes it readable/pollable).
  void SignalTestFence(int signal_fd) {
#if defined(IREE_CTS_HAVE_EVENTFD)
    uint64_t value = 1;
    ssize_t written = write(signal_fd, &value, sizeof(value));
    ASSERT_EQ(written, sizeof(value)) << "eventfd write failed";
#else
    char byte = 1;
    ssize_t written = write(signal_fd, &byte, 1);
    ASSERT_EQ(written, 1) << "pipe write failed";
#endif
  }

  // Checks if an fd is readable (fence signaled).
  bool IsFdReadable(int fd, int timeout_ms = 0) {
    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLIN;
    pfd.revents = 0;
    int result = poll(&pfd, 1, timeout_ms);
    return result > 0 && (pfd.revents & POLLIN);
  }
};

//===----------------------------------------------------------------------===//
// Import fence tests
//===----------------------------------------------------------------------===//

// Import a fence fd and verify semaphore advances when fence signals.
TEST_P(FencePosixTest, ImportFence_SignalAdvancesSemaphore) {
  // Create test fence with a separate signaler fd. Proactor takes ownership
  // of fence_fd; signaler_fd is distinct so the signaler thread can write
  // after proactor closes the original.
  int fence_fd = -1;
  int signaler_fd = -1;
  CreateTestFenceWithSignaler(&fence_fd, &signaler_fd);

  // Create software semaphore starting at 0.
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Import fence: when it signals, semaphore should advance to 1.
  iree_async_primitive_t fence_primitive =
      iree_async_primitive_from_fd(fence_fd);
  IREE_ASSERT_OK(iree_async_semaphore_import_fence(proactor_, fence_primitive,
                                                   semaphore, 1));

  // Semaphore should still be at 0.
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 0u);

  // Signal the fence from a background thread (simulates GPU completion).
  // The import is synchronous, so we can signal immediately.
  std::thread signaler([this, signaler_fd]() { SignalTestFence(signaler_fd); });

  // Poll until semaphore advances. Deferred import registration means the first
  // poll iteration may time out before the signaler thread writes.
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(1000);  // 1s
  while (iree_async_semaphore_query(semaphore) < 1 &&
         iree_time_now() < deadline) {
    iree_host_size_t completed = 0;
    iree_status_t status = iree_async_proactor_poll(
        proactor_, iree_make_deadline(iree_time_now() + 50000000LL),
        &completed);
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_ignore(status);
    } else {
      IREE_ASSERT_OK(status);
    }
  }

  signaler.join();

  // Semaphore should now be at 1.
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 1u);

  // Close the signaler's fd (either the dup for eventfd, or the pipe write end
  // that was distinct from fence_fd which was already closed by the proactor).
  close(signaler_fd);

  iree_async_semaphore_release(semaphore);
}

// Import a fence that's already signaled: semaphore should advance immediately.
TEST_P(FencePosixTest, ImportFence_AlreadySignaled) {
  // Create test fence and signal it before import.
  int fence_fd = -1;
  int signal_fd = -1;
  CreateTestFence(&fence_fd, &signal_fd);
  SignalTestFence(signal_fd);

  // Create semaphore starting at 0.
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Import fence.
  iree_async_primitive_t fence_primitive =
      iree_async_primitive_from_fd(fence_fd);
  IREE_ASSERT_OK(iree_async_semaphore_import_fence(proactor_, fence_primitive,
                                                   semaphore, 1));

  // Poll until semaphore advances. The first poll drains the pending import
  // and registers the already-readable fd; subsequent polls dispatch it.
  iree_host_size_t completed = 0;
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(500);
  while (iree_async_semaphore_query(semaphore) < 1 &&
         iree_time_now() < deadline) {
    iree_status_t status = iree_async_proactor_poll(
        proactor_, iree_make_deadline(iree_time_now() + 50000000LL),
        &completed);
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_ignore(status);
    } else {
      IREE_ASSERT_OK(status);
    }
  }

  EXPECT_EQ(iree_async_semaphore_query(semaphore), 1u);

#if !defined(IREE_CTS_HAVE_EVENTFD)
  close(signal_fd);
#endif

  iree_async_semaphore_release(semaphore);
}

// Import with invalid fd should fail.
TEST_P(FencePosixTest, ImportFence_InvalidFd) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  iree_async_primitive_t bad_fence = iree_async_primitive_from_fd(-1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_semaphore_import_fence(proactor_, bad_fence, semaphore, 1));

  iree_async_semaphore_release(semaphore);
}

// Import with NONE primitive type should fail.
TEST_P(FencePosixTest, ImportFence_NonePrimitive) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  iree_async_primitive_t none_fence = iree_async_primitive_none();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_semaphore_import_fence(proactor_, none_fence, semaphore, 1));

  iree_async_semaphore_release(semaphore);
}

//===----------------------------------------------------------------------===//
// Export fence tests
//===----------------------------------------------------------------------===//

// Export a fence and verify it becomes readable when semaphore advances.
TEST_P(FencePosixTest, ExportFence_SemaphoreAdvanceSignalsFence) {
  // Create semaphore starting at 0.
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Export fence that signals when semaphore reaches 1.
  iree_async_primitive_t exported_fence;
  IREE_ASSERT_OK(iree_async_semaphore_export_fence(proactor_, semaphore, 1,
                                                   &exported_fence));

  ASSERT_EQ(exported_fence.type, IREE_ASYNC_PRIMITIVE_TYPE_FD);
  ASSERT_GE(exported_fence.value.fd, 0);

  // Fence should not be readable yet.
  EXPECT_FALSE(IsFdReadable(exported_fence.value.fd, 0));

  // Advance semaphore to 1.
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, 1, /*frontier=*/NULL));

  // Poll to process internal semaphore watcher.
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(1000);
  while (!IsFdReadable(exported_fence.value.fd, 0) &&
         iree_time_now() < deadline) {
    iree_host_size_t completed = 0;
    IREE_ASSERT_OK(iree_async_proactor_poll(
        proactor_, iree_make_deadline(iree_time_now() + 50000000LL),
        &completed));
  }

  // Fence should now be readable.
  EXPECT_TRUE(IsFdReadable(exported_fence.value.fd, 0));

  // Caller owns the exported fence fd.
  close(exported_fence.value.fd);
  iree_async_semaphore_release(semaphore);
}

// Export fence when semaphore is already at target value: immediate signal.
TEST_P(FencePosixTest, ExportFence_SemaphoreAlreadyReached) {
  // Create semaphore already at 5.
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      5, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Export fence for value 3 (already reached).
  iree_async_primitive_t exported_fence;
  IREE_ASSERT_OK(iree_async_semaphore_export_fence(proactor_, semaphore, 3,
                                                   &exported_fence));

  ASSERT_EQ(exported_fence.type, IREE_ASYNC_PRIMITIVE_TYPE_FD);
  ASSERT_GE(exported_fence.value.fd, 0);

  // Poll to allow internal processing. When the semaphore already reached
  // the target, the callback fires synchronously during export_fence and no
  // io_uring SQE is submitted, so poll may find nothing and time out.
  PollOnce();

  // Fence should be readable immediately (or after minimal polling).
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(500);
  while (!IsFdReadable(exported_fence.value.fd, 0) &&
         iree_time_now() < deadline) {
    PollOnce();
  }

  EXPECT_TRUE(IsFdReadable(exported_fence.value.fd, 0));

  close(exported_fence.value.fd);
  iree_async_semaphore_release(semaphore);
}

//===----------------------------------------------------------------------===//
// Round-trip tests
//===----------------------------------------------------------------------===//

// Import fence → semaphore → export fence: end-to-end pipeline.
TEST_P(FencePosixTest, ImportExportRoundTrip) {
  // Create input fence (simulates GPU completion).
  int input_fence_fd = -1;
  int input_signal_fd = -1;
  CreateTestFence(&input_fence_fd, &input_signal_fd);

  // Create semaphore as the bridge.
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Import: input fence → semaphore value 1.
  iree_async_primitive_t input_primitive =
      iree_async_primitive_from_fd(input_fence_fd);
  IREE_ASSERT_OK(iree_async_semaphore_import_fence(proactor_, input_primitive,
                                                   semaphore, 1));

  // Export: semaphore value 1 → output fence.
  iree_async_primitive_t output_fence;
  IREE_ASSERT_OK(iree_async_semaphore_export_fence(proactor_, semaphore, 1,
                                                   &output_fence));

  ASSERT_EQ(output_fence.type, IREE_ASYNC_PRIMITIVE_TYPE_FD);
  ASSERT_GE(output_fence.value.fd, 0);

  // Output fence should not be readable yet.
  EXPECT_FALSE(IsFdReadable(output_fence.value.fd, 0));

  // Signal input fence (GPU completion).
  SignalTestFence(input_signal_fd);

  // Poll until output fence becomes readable. Deferred import registration
  // means the first poll iteration may time out before it fires.
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(1000);
  while (!IsFdReadable(output_fence.value.fd, 0) &&
         iree_time_now() < deadline) {
    iree_host_size_t completed = 0;
    iree_status_t status = iree_async_proactor_poll(
        proactor_, iree_make_deadline(iree_time_now() + 50000000LL),
        &completed);
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_ignore(status);
    } else {
      IREE_ASSERT_OK(status);
    }
  }

  // Output fence should now be readable.
  EXPECT_TRUE(IsFdReadable(output_fence.value.fd, 0));

  // Verify semaphore reached target.
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 1u);

#if !defined(IREE_CTS_HAVE_EVENTFD)
  close(input_signal_fd);
#endif
  close(output_fence.value.fd);
  iree_async_semaphore_release(semaphore);
}

//===----------------------------------------------------------------------===//
// Export fence error/edge-case tests
//===----------------------------------------------------------------------===//

// Export fence, then fail the semaphore: fd should stay unreadable.
// Exercises the failure branch in the export timepoint callback.
TEST_P(FencePosixTest, ExportFence_SemaphoreFailsAfterExport) {
  // Create semaphore starting at 0.
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Export fence for value 1 (not yet reached).
  iree_async_primitive_t exported_fence;
  IREE_ASSERT_OK(iree_async_semaphore_export_fence(proactor_, semaphore, 1,
                                                   &exported_fence));

  ASSERT_EQ(exported_fence.type, IREE_ASYNC_PRIMITIVE_TYPE_FD);
  ASSERT_GE(exported_fence.value.fd, 0);

  // Fence should not be readable yet.
  EXPECT_FALSE(IsFdReadable(exported_fence.value.fd, 0));

  // Fail the semaphore. The export callback fires synchronously (under the
  // semaphore's lock) and leaves the fd unreadable.
  iree_async_semaphore_fail(
      semaphore, iree_make_status(IREE_STATUS_ABORTED, "test failure"));

  // Drain any pending proactor work. The callback already fired synchronously,
  // but drain to be thorough.
  DrainPending(iree_make_duration_ms(100));

  // Fence should remain unreadable after semaphore failure.
  EXPECT_FALSE(IsFdReadable(exported_fence.value.fd, 0));

  close(exported_fence.value.fd);
  iree_async_semaphore_release(semaphore);
}

// Export fence on an already-failed semaphore: callback fires synchronously
// with the failure status, fd stays unreadable from the start.
TEST_P(FencePosixTest, ExportFence_SemaphoreAlreadyFailed) {
  // Create semaphore and fail it immediately.
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));
  iree_async_semaphore_fail(
      semaphore, iree_make_status(IREE_STATUS_ABORTED, "pre-failed"));

  // Export should succeed (it creates the fd and registers the timepoint).
  iree_async_primitive_t exported_fence;
  IREE_ASSERT_OK(iree_async_semaphore_export_fence(proactor_, semaphore, 1,
                                                   &exported_fence));

  ASSERT_EQ(exported_fence.type, IREE_ASYNC_PRIMITIVE_TYPE_FD);
  ASSERT_GE(exported_fence.value.fd, 0);

  // Poll to allow processing. When the semaphore is already failed, the
  // callback fires synchronously during export_fence and no io_uring SQE is
  // submitted, so poll may find nothing and time out.
  PollOnce();

  // Fence should be unreadable because the semaphore was already failed.
  EXPECT_FALSE(IsFdReadable(exported_fence.value.fd, 0));

  close(exported_fence.value.fd);
  iree_async_semaphore_release(semaphore);
}

// Multiple exports on the same semaphore with different wait values.
// Verifies that each export fires independently as its target is reached.
TEST_P(FencePosixTest, ExportFence_MultipleExportsDifferentValues) {
  // Create semaphore starting at 0.
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Export three fences at values 1, 2, and 3.
  iree_async_primitive_t fence_at_1, fence_at_2, fence_at_3;
  IREE_ASSERT_OK(
      iree_async_semaphore_export_fence(proactor_, semaphore, 1, &fence_at_1));
  IREE_ASSERT_OK(
      iree_async_semaphore_export_fence(proactor_, semaphore, 2, &fence_at_2));
  IREE_ASSERT_OK(
      iree_async_semaphore_export_fence(proactor_, semaphore, 3, &fence_at_3));

  ASSERT_GE(fence_at_1.value.fd, 0);
  ASSERT_GE(fence_at_2.value.fd, 0);
  ASSERT_GE(fence_at_3.value.fd, 0);

  // None should be readable yet.
  EXPECT_FALSE(IsFdReadable(fence_at_1.value.fd, 0));
  EXPECT_FALSE(IsFdReadable(fence_at_2.value.fd, 0));
  EXPECT_FALSE(IsFdReadable(fence_at_3.value.fd, 0));

  // Signal semaphore to 1. Only fence_at_1 should become readable.
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, 1, /*frontier=*/NULL));
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(500);
  while (!IsFdReadable(fence_at_1.value.fd, 0) && iree_time_now() < deadline) {
    iree_host_size_t completed = 0;
    IREE_ASSERT_OK(iree_async_proactor_poll(
        proactor_,
        iree_make_deadline(iree_time_now() + iree_make_duration_ms(10)),
        &completed));
  }
  EXPECT_TRUE(IsFdReadable(fence_at_1.value.fd, 0));
  EXPECT_FALSE(IsFdReadable(fence_at_2.value.fd, 0));
  EXPECT_FALSE(IsFdReadable(fence_at_3.value.fd, 0));

  // Signal semaphore to 3. Both fence_at_2 and fence_at_3 should fire.
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, 3, /*frontier=*/NULL));
  deadline = iree_time_now() + iree_make_duration_ms(500);
  while ((!IsFdReadable(fence_at_2.value.fd, 0) ||
          !IsFdReadable(fence_at_3.value.fd, 0)) &&
         iree_time_now() < deadline) {
    iree_host_size_t completed = 0;
    IREE_ASSERT_OK(iree_async_proactor_poll(
        proactor_,
        iree_make_deadline(iree_time_now() + iree_make_duration_ms(10)),
        &completed));
  }
  EXPECT_TRUE(IsFdReadable(fence_at_2.value.fd, 0));
  EXPECT_TRUE(IsFdReadable(fence_at_3.value.fd, 0));

  close(fence_at_1.value.fd);
  close(fence_at_2.value.fd);
  close(fence_at_3.value.fd);
  iree_async_semaphore_release(semaphore);
}

//===----------------------------------------------------------------------===//
// Cross-thread import fence tests
//===----------------------------------------------------------------------===//
//
// These tests exercise the thread-safety of import_fence by calling it from a
// background thread while poll() runs on the main thread. This is the real
// use case: a GPU driver completion callback imports a fence from a
// driver-internal thread.

// Import fence from a background thread while the main thread polls.
// Under TSAN, this catches data races in fd_map/event_set access (POSIX) or
// concurrent io_uring_enter calls (io_uring).
TEST_P(FencePosixTest, ImportFence_CrossThreadImportDuringPoll) {
  // Create semaphore starting at 0.
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Background thread: import fence + signal it.
  int signaler_fd = -1;
  std::thread importer([this, semaphore, &signaler_fd]() {
    // Brief delay to let the main thread enter poll().
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Create test fence with separate signaler fd. Proactor takes ownership
    // of fence_fd; signaler_fd survives for writing after proactor closes it.
    int fence_fd = -1;
    CreateTestFenceWithSignaler(&fence_fd, &signaler_fd);

    // Import fence from THIS thread (not the poll thread).
    iree_async_primitive_t fence_primitive =
        iree_async_primitive_from_fd(fence_fd);
    IREE_ASSERT_OK(iree_async_semaphore_import_fence(proactor_, fence_primitive,
                                                     semaphore, 1));

    // Signal the fence so the semaphore advances.
    SignalTestFence(signaler_fd);
  });

  // Main thread polls. The import_fence + signal happen concurrently.
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(2000);
  while (iree_async_semaphore_query(semaphore) < 1 &&
         iree_time_now() < deadline) {
    iree_host_size_t completed = 0;
    iree_status_t status = iree_async_proactor_poll(
        proactor_, iree_make_deadline(iree_time_now() + 50000000LL),
        &completed);
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_ignore(status);
    } else {
      IREE_ASSERT_OK(status);
    }
  }

  importer.join();

  EXPECT_EQ(iree_async_semaphore_query(semaphore), 1u);

  close(signaler_fd);
  iree_async_semaphore_release(semaphore);
}

// Multiple threads importing fences concurrently while poll runs.
// Exercises concurrent import_fence calls racing with each other AND with poll.
TEST_P(FencePosixTest, ImportFence_MultipleCrossThreadImports) {
  static constexpr int kThreadCount = 4;

  // Create semaphore starting at 0.
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Each thread gets its own signaler fd to close after join.
  int signaler_fds[kThreadCount];
  memset(signaler_fds, -1, sizeof(signaler_fds));

  std::thread threads[kThreadCount];
  for (int i = 0; i < kThreadCount; ++i) {
    threads[i] = std::thread([this, semaphore, &signaler_fds, i]() {
      // Stagger starts slightly so threads overlap with poll.
      std::this_thread::sleep_for(std::chrono::milliseconds(5 + i * 5));

      int fence_fd = -1;
      CreateTestFenceWithSignaler(&fence_fd, &signaler_fds[i]);

      // Import fence from this thread. Each thread signals a different
      // value so we can verify all imports completed.
      iree_async_primitive_t fence_primitive =
          iree_async_primitive_from_fd(fence_fd);
      IREE_ASSERT_OK(iree_async_semaphore_import_fence(
          proactor_, fence_primitive, semaphore, static_cast<uint64_t>(i + 1)));

      SignalTestFence(signaler_fds[i]);
    });
  }

  // Main thread polls until semaphore reaches the highest value.
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(2000);
  while (iree_async_semaphore_query(semaphore) < kThreadCount &&
         iree_time_now() < deadline) {
    iree_host_size_t completed = 0;
    iree_status_t status = iree_async_proactor_poll(
        proactor_, iree_make_deadline(iree_time_now() + 50000000LL),
        &completed);
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_ignore(status);
    } else {
      IREE_ASSERT_OK(status);
    }
  }

  for (int i = 0; i < kThreadCount; ++i) {
    threads[i].join();
  }

  // All fences should have signaled. The semaphore value is the maximum of
  // all signaled values (1, 2, 3, 4), so it should be kThreadCount.
  EXPECT_GE(iree_async_semaphore_query(semaphore),
            static_cast<uint64_t>(kThreadCount));

  for (int i = 0; i < kThreadCount; ++i) {
    if (signaler_fds[i] >= 0) close(signaler_fds[i]);
  }
  iree_async_semaphore_release(semaphore);
}

CTS_REGISTER_TEST_SUITE(FencePosixTest);

}  // namespace iree::async::cts
