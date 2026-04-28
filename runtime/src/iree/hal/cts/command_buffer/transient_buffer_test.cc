// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for command buffer operations (fill, copy, update) on transient
// buffers from queue_alloca.
//
// The existing fill/copy/update tests use only regular (synchronously
// allocated) buffers. Transient buffers have a different backing mechanism:
// they start uncommitted, get committed when the alloca signal fires, and
// may have different access flags. This exercises the transient buffer path
// through command buffer operations to catch access-flag propagation bugs
// and other transient-specific issues.

#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;
using ::testing::Each;

class DeferredSemaphoreSignal {
 public:
  DeferredSemaphoreSignal(iree_hal_semaphore_t* semaphore,
                          uint64_t payload_value)
      : semaphore_(semaphore), payload_value_(payload_value) {}

  ~DeferredSemaphoreSignal() {
    if (!semaphore_) return;
    IREE_EXPECT_OK(iree_hal_semaphore_signal(semaphore_, payload_value_,
                                             /*frontier=*/NULL));
  }

  iree_status_t SignalNow() {
    iree_hal_semaphore_t* semaphore = semaphore_;
    semaphore_ = nullptr;
    return iree_hal_semaphore_signal(semaphore, payload_value_,
                                     /*frontier=*/NULL);
  }

 private:
  // Semaphore to signal on destruction while armed.
  iree_hal_semaphore_t* semaphore_ = nullptr;

  // Payload value used when releasing the deferred wait.
  uint64_t payload_value_ = 0;
};

class TransientBufferTest : public CtsTestBase<> {
 protected:
  iree_hal_buffer_params_t MakeTransientBufferParams(
      iree_hal_memory_access_t access = IREE_HAL_MEMORY_ACCESS_ALL) {
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
    params.access = access;
    params.usage =
        IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
    return params;
  }

  iree_status_t QueueAllocaTransient(
      iree_hal_semaphore_list_t wait_semaphores,
      iree_hal_semaphore_list_t signal_semaphores, iree_device_size_t size,
      iree_hal_buffer_t** out_buffer) {
    *out_buffer = nullptr;
    iree_hal_buffer_params_t params = MakeTransientBufferParams();
    iree_hal_buffer_t* buffer = nullptr;
    iree_status_t status = iree_hal_device_queue_alloca(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphores,
        signal_semaphores,
        /*pool=*/NULL, params, size, IREE_HAL_ALLOCA_FLAG_NONE, &buffer);
    if (iree_status_is_ok(status)) {
      *out_buffer = buffer;
    } else {
      iree_hal_buffer_release(buffer);
    }
    return status;
  }

  // Allocates a transient buffer via queue_alloca and waits for commit.
  // The buffer is usable immediately after this returns.
  iree_status_t AllocateTransient(iree_device_size_t size,
                                  iree_hal_buffer_t** out_buffer) {
    *out_buffer = nullptr;
    SemaphoreList signal(device_, {0}, {1});
    SemaphoreList empty_wait;
    iree_hal_buffer_t* buffer = nullptr;
    iree_status_t status =
        QueueAllocaTransient(empty_wait, signal, size, &buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                            IREE_ASYNC_WAIT_FLAG_NONE);
    }
    if (iree_status_is_ok(status)) {
      *out_buffer = buffer;
    } else {
      iree_hal_buffer_release(buffer);
    }
    return status;
  }

  // Deallocates a transient buffer and waits for completion.
  iree_status_t DeallocateTransient(iree_hal_buffer_t* buffer) {
    SemaphoreList signal(device_, {0}, {1});
    SemaphoreList empty_wait;
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_dealloca(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, signal, buffer,
        IREE_HAL_DEALLOCA_FLAG_NONE));
    return iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                        IREE_ASYNC_WAIT_FLAG_NONE);
  }

  // Records and submits a one-shot transfer command buffer, waits for
  // completion.
  iree_status_t SubmitTransferAndWait(
      std::function<iree_status_t(iree_hal_command_buffer_t*)> record_fn) {
    Ref<iree_hal_command_buffer_t> command_buffer;
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/0, command_buffer.out()));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer));
    IREE_RETURN_IF_ERROR(record_fn(command_buffer));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_end(command_buffer));
    return SubmitCommandBufferAndWait(command_buffer);
  }
};

class AsyncTransientBufferTest : public TransientBufferTest {};

//===----------------------------------------------------------------------===//
// command_buffer_fill_buffer on transient buffers
//===----------------------------------------------------------------------===//

// Fills an entire transient buffer with a 4-byte pattern via command buffer.
// This exercises access flag propagation through transient buffer wrappers.
TEST_P(TransientBufferTest, FillTransientBuffer) {
  const iree_device_size_t buffer_size = 1024;
  iree_hal_buffer_t* raw = nullptr;
  IREE_ASSERT_OK(AllocateTransient(buffer_size, &raw));
  Ref<iree_hal_buffer_t> transient(raw);

  uint32_t pattern = 0xDEADCAFE;
  IREE_ASSERT_OK(SubmitTransferAndWait([&](iree_hal_command_buffer_t* cmd) {
    return iree_hal_command_buffer_fill_buffer(
        cmd, iree_hal_make_buffer_ref(transient, 0, buffer_size), &pattern,
        sizeof(pattern), IREE_HAL_FILL_FLAG_NONE);
  }));

  auto data = ReadBufferData<uint32_t>(transient);
  EXPECT_THAT(data, Each(0xDEADCAFEu));

  IREE_ASSERT_OK(DeallocateTransient(transient));
}

// Fills a subrange of a transient buffer, verifying boundaries are untouched.
TEST_P(TransientBufferTest, FillTransientBufferSubrange) {
  const iree_device_size_t buffer_size = 256;
  const iree_device_size_t fill_offset = 64;
  const iree_device_size_t fill_length = 128;
  iree_hal_buffer_t* raw = nullptr;
  IREE_ASSERT_OK(AllocateTransient(buffer_size, &raw));
  Ref<iree_hal_buffer_t> transient(raw);

  // Zero the whole buffer first, then fill a subrange.
  uint8_t zero = 0x00;
  uint32_t pattern = 0xCAFEF00D;
  IREE_ASSERT_OK(SubmitTransferAndWait([&](iree_hal_command_buffer_t* cmd) {
    return iree_hal_command_buffer_fill_buffer(
        cmd, iree_hal_make_buffer_ref(transient, 0, buffer_size), &zero,
        sizeof(zero), IREE_HAL_FILL_FLAG_NONE);
  }));
  IREE_ASSERT_OK(SubmitTransferAndWait([&](iree_hal_command_buffer_t* cmd) {
    return iree_hal_command_buffer_fill_buffer(
        cmd, iree_hal_make_buffer_ref(transient, fill_offset, fill_length),
        &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE);
  }));

  // Verify the filled region.
  auto filled = ReadBufferBytes(transient, fill_offset, fill_length);
  for (iree_device_size_t i = 0; i + 3 < fill_length; i += 4) {
    uint32_t value;
    memcpy(&value, filled.data() + i, sizeof(value));
    EXPECT_EQ(value, 0xCAFEF00Du) << "Mismatch at offset " << (fill_offset + i);
  }

  // Verify boundaries are still zero.
  auto before = ReadBufferBytes(transient, 0, fill_offset);
  EXPECT_THAT(before, Each(0x00)) << "Data before fill region was modified";
  auto after = ReadBufferBytes(transient, fill_offset + fill_length,
                               buffer_size - fill_offset - fill_length);
  EXPECT_THAT(after, Each(0x00)) << "Data after fill region was modified";

  IREE_ASSERT_OK(DeallocateTransient(transient));
}

// Fills a transient buffer with a 1-byte pattern. Tests the smallest
// pattern size through the transient path.
TEST_P(TransientBufferTest, FillTransientBuffer1Byte) {
  const iree_device_size_t buffer_size = 512;
  iree_hal_buffer_t* raw = nullptr;
  IREE_ASSERT_OK(AllocateTransient(buffer_size, &raw));
  Ref<iree_hal_buffer_t> transient(raw);

  uint8_t pattern = 0xAB;
  IREE_ASSERT_OK(SubmitTransferAndWait([&](iree_hal_command_buffer_t* cmd) {
    return iree_hal_command_buffer_fill_buffer(
        cmd, iree_hal_make_buffer_ref(transient, 0, buffer_size), &pattern,
        sizeof(pattern), IREE_HAL_FILL_FLAG_NONE);
  }));

  auto data = ReadBufferData<uint8_t>(transient);
  EXPECT_THAT(data, Each(0xAB));

  IREE_ASSERT_OK(DeallocateTransient(transient));
}

//===----------------------------------------------------------------------===//
// command_buffer_copy_buffer with transient buffers
//===----------------------------------------------------------------------===//

// Copies from a regular buffer to a transient buffer.
TEST_P(TransientBufferTest, CopyToTransientBuffer) {
  const iree_device_size_t buffer_size = 512;
  Ref<iree_hal_buffer_t> source;
  IREE_ASSERT_OK(CreateFilledDeviceBuffer<uint32_t>(buffer_size, 0xAAAAAAAAu,
                                                    source.out()));

  iree_hal_buffer_t* raw = nullptr;
  IREE_ASSERT_OK(AllocateTransient(buffer_size, &raw));
  Ref<iree_hal_buffer_t> transient(raw);

  IREE_ASSERT_OK(SubmitTransferAndWait([&](iree_hal_command_buffer_t* cmd) {
    return iree_hal_command_buffer_copy_buffer(
        cmd, iree_hal_make_buffer_ref(source, 0, buffer_size),
        iree_hal_make_buffer_ref(transient, 0, buffer_size),
        IREE_HAL_COPY_FLAG_NONE);
  }));

  auto data = ReadBufferData<uint32_t>(transient);
  EXPECT_THAT(data, Each(0xAAAAAAAAu));

  IREE_ASSERT_OK(DeallocateTransient(transient));
}

// Copies from a transient buffer to a regular buffer.
TEST_P(TransientBufferTest, CopyFromTransientBuffer) {
  const iree_device_size_t buffer_size = 512;

  // Fill the transient buffer first.
  iree_hal_buffer_t* raw = nullptr;
  IREE_ASSERT_OK(AllocateTransient(buffer_size, &raw));
  Ref<iree_hal_buffer_t> transient(raw);

  uint32_t pattern = 0xBBBBBBBB;
  IREE_ASSERT_OK(SubmitTransferAndWait([&](iree_hal_command_buffer_t* cmd) {
    return iree_hal_command_buffer_fill_buffer(
        cmd, iree_hal_make_buffer_ref(transient, 0, buffer_size), &pattern,
        sizeof(pattern), IREE_HAL_FILL_FLAG_NONE);
  }));

  // Copy from transient to a regular buffer.
  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(buffer_size, target.out()));

  IREE_ASSERT_OK(SubmitTransferAndWait([&](iree_hal_command_buffer_t* cmd) {
    return iree_hal_command_buffer_copy_buffer(
        cmd, iree_hal_make_buffer_ref(transient, 0, buffer_size),
        iree_hal_make_buffer_ref(target, 0, buffer_size),
        IREE_HAL_COPY_FLAG_NONE);
  }));

  // Verify the regular buffer received the data.
  auto data = ReadBufferData<uint32_t>(target);
  EXPECT_THAT(data, Each(0xBBBBBBBBu));

  IREE_ASSERT_OK(DeallocateTransient(transient));
}

//===----------------------------------------------------------------------===//
// Multi-operation command buffers with transient buffers
//===----------------------------------------------------------------------===//

// Fills a transient buffer then copies to a regular buffer in a single
// command buffer with a barrier between them. This is the pattern used
// when initializing scratch space and then extracting results.
TEST_P(TransientBufferTest, FillThenCopyInSingleCommandBuffer) {
  const iree_device_size_t buffer_size = 256;

  iree_hal_buffer_t* raw = nullptr;
  IREE_ASSERT_OK(AllocateTransient(buffer_size, &raw));
  Ref<iree_hal_buffer_t> transient(raw);

  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(buffer_size, output.out()));

  uint32_t pattern = 0x11223344;
  IREE_ASSERT_OK(SubmitTransferAndWait([&](iree_hal_command_buffer_t* cmd) {
    // Fill the transient buffer.
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_fill_buffer(
        cmd, iree_hal_make_buffer_ref(transient, 0, buffer_size), &pattern,
        sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));

    // Barrier: fill must complete before copy reads.
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_execution_barrier(
        cmd,
        IREE_HAL_EXECUTION_STAGE_TRANSFER |
            IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
            IREE_HAL_EXECUTION_STAGE_TRANSFER,
        IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 0, nullptr, 0, nullptr));

    // Copy transient → output.
    return iree_hal_command_buffer_copy_buffer(
        cmd, iree_hal_make_buffer_ref(transient, 0, buffer_size),
        iree_hal_make_buffer_ref(output, 0, buffer_size),
        IREE_HAL_COPY_FLAG_NONE);
  }));

  // Verify the output buffer (read from regular buffer, no transient issues).
  auto data = ReadBufferData<uint32_t>(output);
  EXPECT_THAT(data, Each(0x11223344u));

  IREE_ASSERT_OK(DeallocateTransient(transient));
}

//===----------------------------------------------------------------------===//
// command_buffer_update_buffer on transient buffers
//===----------------------------------------------------------------------===//

// Updates a transient buffer from host data via command_buffer_update_buffer.
TEST_P(TransientBufferTest, UpdateTransientBuffer) {
  const iree_device_size_t element_count = 16;
  const iree_device_size_t buffer_size = element_count * sizeof(uint32_t);

  iree_hal_buffer_t* raw = nullptr;
  IREE_ASSERT_OK(AllocateTransient(buffer_size, &raw));
  Ref<iree_hal_buffer_t> transient(raw);

  // Host data: sequential values.
  std::vector<uint32_t> host_data(element_count);
  std::iota(host_data.begin(), host_data.end(), 100u);

  IREE_ASSERT_OK(SubmitTransferAndWait([&](iree_hal_command_buffer_t* cmd) {
    return iree_hal_command_buffer_update_buffer(
        cmd, host_data.data(), /*source_offset=*/0,
        iree_hal_make_buffer_ref(transient, 0, buffer_size),
        IREE_HAL_UPDATE_FLAG_NONE);
  }));

  auto readback = ReadBufferData<uint32_t>(transient);
  EXPECT_THAT(readback, ContainerEq(host_data));

  IREE_ASSERT_OK(DeallocateTransient(transient));
}

//===----------------------------------------------------------------------===//
// Transient buffer with zero access flags (HAL module calling convention)
//===----------------------------------------------------------------------===//

// Allocates a transient buffer with zero access flags (as the HAL module does)
// and performs a command buffer fill on it. This specifically targets the
// access-flags propagation bug: queue_alloca canonicalizes access=0 to
// MEMORY_ACCESS_ALL, but the command buffer fill path may not handle
// transient buffers with the canonicalized flags correctly.
TEST_P(TransientBufferTest, FillTransientWithZeroAccessFlags) {
  const iree_device_size_t buffer_size = 256;

  SemaphoreList signal(device_, {0}, {1});
  SemaphoreList empty_wait;
  iree_hal_buffer_params_t params = MakeTransientBufferParams(
      /*access=*/IREE_HAL_MEMORY_ACCESS_NONE);
  // Deliberately use access = 0 to match HAL module behavior.

  iree_hal_buffer_t* raw = nullptr;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, signal,
      /*pool=*/NULL, params, buffer_size, IREE_HAL_ALLOCA_FLAG_NONE, &raw));
  Ref<iree_hal_buffer_t> transient(raw);
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  // Fill via command buffer — this is the path that was broken.
  uint32_t pattern = 0x55AA55AA;
  IREE_ASSERT_OK(SubmitTransferAndWait([&](iree_hal_command_buffer_t* cmd) {
    return iree_hal_command_buffer_fill_buffer(
        cmd, iree_hal_make_buffer_ref(transient, 0, buffer_size), &pattern,
        sizeof(pattern), IREE_HAL_FILL_FLAG_NONE);
  }));

  auto data = ReadBufferData<uint32_t>(transient);
  EXPECT_THAT(data, Each(0x55AA55AAu));

  IREE_ASSERT_OK(DeallocateTransient(transient));
}

//===----------------------------------------------------------------------===//
// Transient buffer resolution
//===----------------------------------------------------------------------===//

// Records a static transient buffer reference before the alloca has committed,
// then executes the command buffer after the alloca signal resolves.
TEST_P(AsyncTransientBufferTest, StaticRefRecordedBeforeAllocaCommitExecutes) {
  const iree_device_size_t buffer_size = 256;

  SemaphoreList alloca_wait(device_, {0}, {1});
  SemaphoreList alloca_signal(device_, {0}, {1});
  iree_hal_buffer_t* raw = nullptr;
  IREE_ASSERT_OK(
      QueueAllocaTransient(alloca_wait, alloca_signal, buffer_size, &raw));
  Ref<iree_hal_buffer_t> transient(raw);
  DeferredSemaphoreSignal release_alloca_wait(alloca_wait.semaphores[0], 1);

  uint64_t alloca_value = 0;
  IREE_ASSERT_OK(
      iree_hal_semaphore_query(alloca_signal.semaphores[0], &alloca_value));
  EXPECT_EQ(0u, alloca_value);

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  uint32_t pattern = 0xA55A1337u;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer, iree_hal_make_buffer_ref(transient, 0, buffer_size),
      &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  SemaphoreList execute_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, alloca_signal, execute_signal,
      command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));

  IREE_ASSERT_OK(release_alloca_wait.SignalNow());
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      execute_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  auto data = ReadBufferData<uint32_t>(transient);
  EXPECT_THAT(data, Each(pattern));

  IREE_ASSERT_OK(DeallocateTransient(transient));
}

// Records an indirect command-buffer reference with no concrete buffer, then
// binds a transient buffer whose alloca is still parked. This exercises the
// queue_execute binding-table path that resolves transient wrappers only after
// the alloca dependency has made their backing available.
TEST_P(AsyncTransientBufferTest, IndirectRefResolvedAfterAllocaCommit) {
  const iree_device_size_t buffer_size = 256;

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/1, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  uint32_t pattern = 0x5AA5C0DEu;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_indirect_buffer_ref(/*binding=*/0, 0, buffer_size),
      &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  SemaphoreList alloca_wait(device_, {0}, {1});
  SemaphoreList alloca_signal(device_, {0}, {1});
  iree_hal_buffer_t* raw = nullptr;
  IREE_ASSERT_OK(
      QueueAllocaTransient(alloca_wait, alloca_signal, buffer_size, &raw));
  Ref<iree_hal_buffer_t> transient(raw);
  DeferredSemaphoreSignal release_alloca_wait(alloca_wait.semaphores[0], 1);

  const iree_hal_buffer_binding_t bindings[] = {
      {transient, 0, IREE_HAL_WHOLE_BUFFER},
  };
  SemaphoreList execute_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, alloca_signal, execute_signal,
      command_buffer,
      iree_hal_buffer_binding_table_t{IREE_ARRAYSIZE(bindings), bindings},
      IREE_HAL_EXECUTE_FLAG_NONE));

  IREE_ASSERT_OK(release_alloca_wait.SignalNow());
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      execute_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  auto data = ReadBufferData<uint32_t>(transient);
  EXPECT_THAT(data, Each(pattern));

  IREE_ASSERT_OK(DeallocateTransient(transient));
}

CTS_REGISTER_TEST_SUITE(TransientBufferTest);
CTS_REGISTER_TEST_SUITE_WITH_TAGS(AsyncTransientBufferTest, {"async_queue"},
                                  {});

}  // namespace iree::hal::cts
