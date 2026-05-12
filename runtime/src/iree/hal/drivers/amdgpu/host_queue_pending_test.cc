// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_pending.h"

#include <cstdint>
#include <cstring>

#include "iree/async/frontier.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::hal::cts::Ref;

constexpr iree_hal_queue_affinity_t kQueueAffinity0 =
    ((iree_hal_queue_affinity_t)1ull) << 0;

class HostQueuePendingTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    host_allocator_ = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_with_defaults(
        &libhsa_, &topology_));
    if (topology_.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology_);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
  }

  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t HostQueuePendingTest::host_allocator_;
iree_hal_amdgpu_libhsa_t HostQueuePendingTest::libhsa_;
iree_hal_amdgpu_topology_t HostQueuePendingTest::topology_;

class TestLogicalDevice {
 public:
  ~TestLogicalDevice() {
    iree_hal_device_release(base_device_);
    iree_hal_device_group_release(device_group_);
  }

  iree_status_t Initialize(
      const iree_hal_amdgpu_logical_device_options_t* options,
      const iree_hal_amdgpu_libhsa_t* libhsa,
      const iree_hal_amdgpu_topology_t* topology,
      iree_allocator_t host_allocator) {
    IREE_RETURN_IF_ERROR(create_context_.Initialize(host_allocator));
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_create(
        IREE_SV("amdgpu"), options, libhsa, topology, create_context_.params(),
        host_allocator, &base_device_));
    return iree_hal_device_group_create_from_device(
        base_device_, create_context_.frontier_tracker(), host_allocator,
        &device_group_);
  }

  iree_hal_device_t* base_device() const { return base_device_; }

  iree_hal_allocator_t* allocator() const {
    return iree_hal_device_allocator(base_device_);
  }

  iree_hal_amdgpu_logical_device_t* logical_device() const {
    return (iree_hal_amdgpu_logical_device_t*)base_device_;
  }

  iree_hal_amdgpu_host_queue_t* first_host_queue() const {
    iree_hal_amdgpu_logical_device_t* logical_device = this->logical_device();
    if (logical_device->physical_device_count == 0) return NULL;
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[0];
    if (physical_device->host_queue_count == 0) return NULL;
    return &physical_device->host_queues[0];
  }

 private:
  // Creation context supplying the proactor pool and frontier tracker.
  iree::hal::cts::DeviceCreateContext create_context_;

  // Test-owned device reference released before the topology-owning group.
  iree_hal_device_t* base_device_ = NULL;

  // Device group that owns the topology assigned to |base_device_|.
  iree_hal_device_group_t* device_group_ = NULL;
};

static iree_hal_buffer_params_t MakeTransientBufferParams() {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  return params;
}

static iree_hal_buffer_params_t MakeHostLocalMappedTransientBufferParams(
    iree_hal_memory_type_t extra_memory_type) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE | extra_memory_type;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                 IREE_HAL_BUFFER_USAGE_MAPPING;
  return params;
}

static iree_status_t CreateHostVisibleTransferBuffer(
    iree_hal_allocator_t* allocator, iree_device_size_t buffer_size,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  return iree_hal_allocator_allocate_buffer(allocator, params, buffer_size,
                                            out_buffer);
}

static iree_status_t CreateSemaphore(iree_hal_device_t* device,
                                     iree_hal_semaphore_t** out_semaphore) {
  return iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, out_semaphore);
}

static iree_hal_semaphore_list_t MakeSemaphoreList(
    iree_hal_semaphore_t** semaphore, uint64_t* payload_value) {
  return iree_hal_semaphore_list_t{
      /*count=*/1,
      /*semaphores=*/semaphore,
      /*payload_values=*/payload_value,
  };
}

static void RunDefaultPoolServesHostLocalMappedAlloca(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_memory_type_t extra_memory_type) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, libhsa, topology, host_allocator));

  Ref<iree_hal_semaphore_t> alloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), alloca_signal.out()));
  uint64_t alloca_signal_value = 1;
  iree_hal_semaphore_t* alloca_signal_ptr = alloca_signal.get();
  const iree_hal_semaphore_list_t alloca_signal_list =
      MakeSemaphoreList(&alloca_signal_ptr, &alloca_signal_value);

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      test_device.base_device(), kQueueAffinity0,
      iree_hal_semaphore_list_empty(), alloca_signal_list, /*pool=*/NULL,
      MakeHostLocalMappedTransientBufferParams(extra_memory_type),
      /*allocation_size=*/8, IREE_HAL_ALLOCA_FLAG_NONE, &buffer));
  ASSERT_NE(buffer, nullptr);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(alloca_signal, alloca_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  EXPECT_TRUE(iree_all_bits_set(
      iree_hal_buffer_memory_type(buffer),
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  EXPECT_TRUE(iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer),
                                IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED));

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE,
      /*byte_offset=*/0, /*byte_length=*/8, &mapping));
  memset(mapping.contents.data, 0, 8);
  iree_hal_buffer_unmap_range(&mapping);

  Ref<iree_hal_semaphore_t> dealloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), dealloca_signal.out()));
  uint64_t dealloca_signal_value = 1;
  iree_hal_semaphore_t* dealloca_signal_ptr = dealloca_signal.get();
  const iree_hal_semaphore_list_t dealloca_signal_list =
      MakeSemaphoreList(&dealloca_signal_ptr, &dealloca_signal_value);
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      test_device.base_device(), kQueueAffinity0,
      iree_hal_semaphore_list_empty(), dealloca_signal_list, buffer,
      IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(dealloca_signal, dealloca_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_buffer_release(buffer);
}

TEST_F(HostQueuePendingTest, DefaultPoolServesHostLocalMappedAlloca) {
  RunDefaultPoolServesHostLocalMappedAlloca(
      &libhsa_, &topology_, host_allocator_, IREE_HAL_MEMORY_TYPE_NONE);
}

TEST_F(HostQueuePendingTest, DefaultPoolServesOptimalHostLocalMappedAlloca) {
  RunDefaultPoolServesHostLocalMappedAlloca(
      &libhsa_, &topology_, host_allocator_, IREE_HAL_MEMORY_TYPE_OPTIMAL);
}

static bool HostQueueHasPendingOps(iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  const bool has_pending_ops = queue->pending_head != NULL;
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  return has_pending_ops;
}

static bool HostQueueHasPostDrainAction(iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.post_drain_mutex);
  const bool has_action = queue->post_drain.head != NULL;
  iree_slim_mutex_unlock(&queue->locks.post_drain_mutex);
  return has_action;
}

static iree_status_t EnqueueRawBlockingBarrier(
    iree_hal_amdgpu_host_queue_t* queue, hsa_signal_t blocker_signal) {
  const uint64_t packet_id =
      iree_hal_amdgpu_aql_ring_reserve(&queue->aql_ring, /*count=*/1);
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  const hsa_signal_t dep_signals[1] = {blocker_signal};
  const uint16_t header = iree_hal_amdgpu_aql_emit_barrier_and(
      &packet->barrier_and, dep_signals, IREE_ARRAYSIZE(dep_signals),
      iree_hal_amdgpu_aql_packet_control_barrier_system(),
      iree_hsa_signal_null());
  iree_hal_amdgpu_aql_ring_commit(packet, header, /*setup=*/0);
  iree_hal_amdgpu_aql_ring_doorbell(&queue->aql_ring, packet_id);
  return iree_ok_status();
}

static iree_status_t CreateExplicitFixedBlockPool(iree_hal_device_t* device,
                                                  iree_device_size_t block_size,
                                                  iree_hal_pool_t** out_pool) {
  iree_hal_queue_pool_backend_t backend = {0};
  IREE_RETURN_IF_ERROR(iree_hal_device_query_queue_pool_backend(
      device, kQueueAffinity0, &backend));
  if (!backend.slab_provider || !backend.notification) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "queue pool backend query returned an incomplete backend bundle");
  }
  iree_hal_fixed_block_pool_options_t options = {};
  options.block_allocator_options.block_size = block_size;
  options.block_allocator_options.block_count = 1;
  options.block_allocator_options.frontier_capacity = 2;
  return iree_hal_fixed_block_pool_create(
      options, backend.slab_provider, backend.notification,
      iree_hal_pool_epoch_query_null(), iree_allocator_system(), out_pool);
}

static iree_status_t CreateExplicitTlsfPool(iree_hal_device_t* device,
                                            iree_device_size_t slab_size,
                                            iree_hal_pool_t** out_pool) {
  iree_hal_queue_pool_backend_t backend = {0};
  IREE_RETURN_IF_ERROR(iree_hal_device_query_queue_pool_backend(
      device, kQueueAffinity0, &backend));
  if (!backend.slab_provider || !backend.notification) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "queue pool backend query returned an incomplete backend bundle");
  }
  iree_hal_tlsf_pool_options_t options = {};
  options.tlsf_options.range_length = slab_size;
  options.tlsf_options.alignment = 16;
  options.tlsf_options.initial_block_capacity = 16;
  options.tlsf_options.frontier_capacity = 2;
  return iree_hal_tlsf_pool_create(
      options, backend.slab_provider, backend.notification,
      iree_hal_pool_epoch_query_null(), iree_allocator_system(), out_pool);
}

static iree_status_t SeedWaitableFixedBlockReservation(
    iree_hal_pool_t* pool, iree_device_size_t allocation_size,
    iree_async_axis_t death_axis) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t acquire_info;
  iree_hal_pool_acquire_result_t acquire_result;
  IREE_RETURN_IF_ERROR(iree_hal_pool_acquire_reservation(
      pool, allocation_size, /*alignment=*/1, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &acquire_info,
      &acquire_result));
  if (acquire_result != IREE_HAL_POOL_ACQUIRE_OK &&
      acquire_result != IREE_HAL_POOL_ACQUIRE_OK_FRESH) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "expected fresh fixed-block reservation");
  }

  iree_async_single_frontier_t death_frontier;
  iree_async_single_frontier_initialize(&death_frontier, death_axis, 1);
  iree_hal_pool_release_reservation(
      pool, &reservation,
      iree_async_single_frontier_as_const_frontier(&death_frontier));
  return iree_ok_status();
}

typedef struct HostActionState {
  // Notification posted after the action callback records its result.
  iree_notification_t notification;

  // Number of times the action callback has run.
  iree_atomic_int32_t call_count;

  // Last callback status code.
  iree_atomic_int32_t status_code;

  // Whether the callback received a reclaim entry.
  iree_atomic_int32_t had_entry;
} HostActionState;

static void HostActionStateInitialize(HostActionState* state) {
  iree_notification_initialize(&state->notification);
  iree_atomic_store(&state->call_count, 0, iree_memory_order_relaxed);
  iree_atomic_store(&state->status_code, IREE_STATUS_UNKNOWN,
                    iree_memory_order_relaxed);
  iree_atomic_store(&state->had_entry, 0, iree_memory_order_relaxed);
}

static void HostActionStateDeinitialize(HostActionState* state) {
  iree_notification_deinitialize(&state->notification);
}

static bool HostActionStateWasCalled(void* user_data) {
  HostActionState* state = (HostActionState*)user_data;
  return iree_atomic_load(&state->call_count, iree_memory_order_acquire) != 0;
}

static void RecordHostAction(iree_hal_amdgpu_reclaim_entry_t* entry,
                             void* user_data, iree_status_t status) {
  HostActionState* state = (HostActionState*)user_data;
  iree_atomic_store(&state->had_entry, entry ? 1 : 0,
                    iree_memory_order_release);
  iree_atomic_store(&state->status_code, iree_status_code(status),
                    iree_memory_order_release);
  iree_atomic_fetch_add(&state->call_count, 1, iree_memory_order_acq_rel);
  iree_notification_post(&state->notification, IREE_ALL_WAITERS);
}

static int32_t HostActionCallCount(HostActionState* state) {
  return iree_atomic_load(&state->call_count, iree_memory_order_acquire);
}

static iree_status_code_t HostActionStatusCode(HostActionState* state) {
  return (iree_status_code_t)iree_atomic_load(&state->status_code,
                                              iree_memory_order_acquire);
}

static bool HostActionHadEntry(HostActionState* state) {
  return iree_atomic_load(&state->had_entry, iree_memory_order_acquire) != 0;
}

TEST_F(HostQueuePendingTest,
       DeferredHostActionFailureRunsSynchronousCallbackOnce) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_semaphore_t> wait_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), wait_semaphore.out()));
  iree_hal_semaphore_fail(
      wait_semaphore,
      iree_make_status(IREE_STATUS_CANCELLED, "test wait failed"));
  uint64_t wait_value = 1;
  iree_hal_semaphore_t* wait_semaphore_ptr = wait_semaphore.get();
  const iree_hal_semaphore_list_t wait_list =
      MakeSemaphoreList(&wait_semaphore_ptr, &wait_value);

  HostActionState action_state;
  HostActionStateInitialize(&action_state);
  IREE_ASSERT_OK(iree_hal_amdgpu_host_queue_enqueue_host_action(
      queue, wait_list,
      iree_hal_amdgpu_reclaim_action_t{
          .fn = RecordHostAction,
          .user_data = &action_state,
      },
      /*operation_resources=*/NULL, /*operation_resource_count=*/0));

  EXPECT_EQ(HostActionCallCount(&action_state), 1);
  EXPECT_EQ(HostActionStatusCode(&action_state), IREE_STATUS_CANCELLED);
  EXPECT_FALSE(HostActionHadEntry(&action_state));
  EXPECT_FALSE(HostQueueHasPendingOps(queue));

  HostActionStateDeinitialize(&action_state);
}

TEST_F(HostQueuePendingTest, CancelPendingFillFailsSignalSemaphore) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), target_buffer.out()));

  Ref<iree_hal_semaphore_t> wait_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), wait_semaphore.out()));
  uint64_t wait_value = 1;
  iree_hal_semaphore_t* wait_semaphore_ptr = wait_semaphore.get();
  const iree_hal_semaphore_list_t wait_list =
      MakeSemaphoreList(&wait_semaphore_ptr, &wait_value);

  Ref<iree_hal_semaphore_t> signal_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), signal_semaphore.out()));
  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_semaphore_ptr = signal_semaphore.get();
  const iree_hal_semaphore_list_t signal_list =
      MakeSemaphoreList(&signal_semaphore_ptr, &signal_value);

  const uint32_t pattern = 0xCACE1100u;
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY, wait_list,
      signal_list, target_buffer, /*target_offset=*/0, sizeof(pattern),
      &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
  ASSERT_TRUE(HostQueueHasPendingOps(queue));

  iree_hal_amdgpu_host_queue_cancel_pending(queue, IREE_STATUS_CANCELLED,
                                            "test cancellation");
  EXPECT_FALSE(HostQueueHasPendingOps(queue));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        iree_hal_semaphore_wait(signal_semaphore, signal_value,
                                                iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_EXPECT_OK(
      iree_hal_semaphore_signal(wait_semaphore, wait_value, /*frontier=*/NULL));
}

TEST_F(HostQueuePendingTest, CapacityParkedHostActionRetriesAfterPostDrain) {
  static constexpr uint32_t kAqlCapacity = 64;
  static constexpr uint32_t kNotificationCapacity = 1;
  static constexpr uint32_t kKernargCapacity = 2 * kAqlCapacity;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.host_queues.aql_capacity = kAqlCapacity;
  options.host_queues.notification_capacity = kNotificationCapacity;
  options.host_queues.kernarg_capacity = kKernargCapacity;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_buffer_t> pressure_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), pressure_buffer.out()));

  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(&libhsa_), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/NULL, /*attributes=*/0, &blocker_signal));
  IREE_ASSERT_OK(EnqueueRawBlockingBarrier(queue, blocker_signal));

  Ref<iree_hal_semaphore_t> pressure_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), pressure_signal.out()));
  uint64_t pressure_signal_value = 1;
  iree_hal_semaphore_t* pressure_signal_ptr = pressure_signal.get();
  const iree_hal_semaphore_list_t pressure_signal_list =
      MakeSemaphoreList(&pressure_signal_ptr, &pressure_signal_value);
  const uint32_t pressure_pattern = 0xABCD1234u;
  iree_status_t status = iree_hal_device_queue_fill(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), pressure_signal_list, pressure_buffer,
      /*target_offset=*/0, sizeof(pressure_pattern), &pressure_pattern,
      sizeof(pressure_pattern), IREE_HAL_FILL_FLAG_NONE);

  HostActionState action_state;
  HostActionStateInitialize(&action_state);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_enqueue_host_action(
        queue, iree_hal_semaphore_list_empty(),
        iree_hal_amdgpu_reclaim_action_t{
            .fn = RecordHostAction,
            .user_data = &action_state,
        },
        /*operation_resources=*/NULL, /*operation_resource_count=*/0);
  }
  const bool retry_parked =
      iree_status_is_ok(status) && HostQueueHasPostDrainAction(queue);

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);

  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(pressure_signal, pressure_signal_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    ASSERT_TRUE(iree_notification_await(&action_state.notification,
                                        HostActionStateWasCalled, &action_state,
                                        iree_infinite_timeout()));
  }
  IREE_EXPECT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));

  IREE_ASSERT_OK(status);
  EXPECT_TRUE(retry_parked);
  EXPECT_EQ(HostActionCallCount(&action_state), 1);
  EXPECT_EQ(HostActionStatusCode(&action_state), IREE_STATUS_OK);
  EXPECT_TRUE(HostActionHadEntry(&action_state));

  HostActionStateDeinitialize(&action_state);
}

TEST_F(HostQueuePendingTest, QueueAllocaRejectsWaitableReservationWithoutFlag) {
  const iree_device_size_t allocation_size = 4096;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateExplicitFixedBlockPool(test_device.base_device(),
                                              allocation_size, pool.out()));
  const iree_async_axis_t death_axis =
      iree_async_axis_make_queue(0xFE, 0xFE, 0xFE, 0xFE);
  IREE_ASSERT_OK(
      SeedWaitableFixedBlockReservation(pool, allocation_size, death_axis));

  Ref<iree_hal_semaphore_t> alloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), alloca_signal.out()));
  uint64_t alloca_signal_value = 1;
  iree_hal_semaphore_t* alloca_signal_ptr = alloca_signal.get();
  const iree_hal_semaphore_list_t alloca_signal_list =
      MakeSemaphoreList(&alloca_signal_ptr, &alloca_signal_value);

  iree_hal_buffer_t* buffer = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_device_queue_alloca(
                            test_device.base_device(), kQueueAffinity0,
                            iree_hal_semaphore_list_empty(), alloca_signal_list,
                            pool, MakeTransientBufferParams(), allocation_size,
                            IREE_HAL_ALLOCA_FLAG_NONE, &buffer));
  EXPECT_EQ(buffer, nullptr);
  EXPECT_FALSE(HostQueueHasPendingOps(queue));
}

TEST_F(HostQueuePendingTest, QueueAllocaTlsfGrowthRetriesThroughColdPath) {
  const iree_device_size_t allocation_size = 4096;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateExplicitTlsfPool(test_device.base_device(),
                                        allocation_size, pool.out()));

  Ref<iree_hal_semaphore_t> alloca0_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), alloca0_signal.out()));
  uint64_t alloca0_signal_value = 1;
  iree_hal_semaphore_t* alloca0_signal_ptr = alloca0_signal.get();
  const iree_hal_semaphore_list_t alloca0_signal_list =
      MakeSemaphoreList(&alloca0_signal_ptr, &alloca0_signal_value);

  iree_hal_buffer_t* buffer0 = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      test_device.base_device(), kQueueAffinity0,
      iree_hal_semaphore_list_empty(), alloca0_signal_list, pool,
      MakeTransientBufferParams(), allocation_size, IREE_HAL_ALLOCA_FLAG_NONE,
      &buffer0));
  ASSERT_NE(buffer0, nullptr);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(alloca0_signal, alloca0_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  Ref<iree_hal_semaphore_t> alloca1_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), alloca1_signal.out()));
  uint64_t alloca1_signal_value = 1;
  iree_hal_semaphore_t* alloca1_signal_ptr = alloca1_signal.get();
  const iree_hal_semaphore_list_t alloca1_signal_list =
      MakeSemaphoreList(&alloca1_signal_ptr, &alloca1_signal_value);

  iree_hal_buffer_t* buffer1 = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      test_device.base_device(), kQueueAffinity0,
      iree_hal_semaphore_list_empty(), alloca1_signal_list, pool,
      MakeTransientBufferParams(), allocation_size, IREE_HAL_ALLOCA_FLAG_NONE,
      &buffer1));
  ASSERT_NE(buffer1, nullptr);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(alloca1_signal, alloca1_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_FALSE(HostQueueHasPendingOps(queue));

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_GE(stats.slab_count, 2u);
  EXPECT_GE(stats.exhausted_count, 1u);

  iree_hal_buffer_release(buffer1);
  iree_hal_buffer_release(buffer0);
}

TEST_F(HostQueuePendingTest, CancelPendingAllocaFrontierWait) {
  const iree_device_size_t allocation_size = 4096;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateExplicitFixedBlockPool(test_device.base_device(),
                                              allocation_size, pool.out()));
  const iree_async_axis_t death_axis = queue->axis;
  IREE_ASSERT_OK(
      SeedWaitableFixedBlockReservation(pool, allocation_size, death_axis));
  queue->wait_barrier_strategy = IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_DEFER;

  Ref<iree_hal_semaphore_t> alloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), alloca_signal.out()));
  uint64_t alloca_signal_value = 1;
  iree_hal_semaphore_t* alloca_signal_ptr = alloca_signal.get();
  const iree_hal_semaphore_list_t alloca_signal_list =
      MakeSemaphoreList(&alloca_signal_ptr, &alloca_signal_value);

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      test_device.base_device(), kQueueAffinity0,
      iree_hal_semaphore_list_empty(), alloca_signal_list, pool,
      MakeTransientBufferParams(), allocation_size,
      IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER, &buffer));
  ASSERT_NE(buffer, nullptr);
  EXPECT_FALSE(iree_hal_semaphore_list_poll(alloca_signal_list));
  ASSERT_TRUE(HostQueueHasPendingOps(queue));

  iree_hal_amdgpu_host_queue_cancel_pending(queue, IREE_STATUS_CANCELLED,
                                            "test cancellation");
  EXPECT_FALSE(HostQueueHasPendingOps(queue));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_hal_semaphore_wait(alloca_signal, alloca_signal_value,
                              iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  iree_hal_buffer_release(buffer);
}

TEST_F(HostQueuePendingTest, CancelPendingAllocaPoolNotificationWait) {
  const iree_device_size_t allocation_size = 4096;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateExplicitFixedBlockPool(test_device.base_device(),
                                              allocation_size, pool.out()));

  Ref<iree_hal_semaphore_t> alloca0_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), alloca0_signal.out()));
  uint64_t alloca0_signal_value = 1;
  iree_hal_semaphore_t* alloca0_signal_ptr = alloca0_signal.get();
  const iree_hal_semaphore_list_t alloca0_signal_list =
      MakeSemaphoreList(&alloca0_signal_ptr, &alloca0_signal_value);

  iree_hal_buffer_t* buffer0 = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      test_device.base_device(), kQueueAffinity0,
      iree_hal_semaphore_list_empty(), alloca0_signal_list, pool,
      MakeTransientBufferParams(), allocation_size, IREE_HAL_ALLOCA_FLAG_NONE,
      &buffer0));
  ASSERT_NE(buffer0, nullptr);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(alloca0_signal, alloca0_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  Ref<iree_hal_semaphore_t> alloca1_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), alloca1_signal.out()));
  uint64_t alloca1_signal_value = 1;
  iree_hal_semaphore_t* alloca1_signal_ptr = alloca1_signal.get();
  const iree_hal_semaphore_list_t alloca1_signal_list =
      MakeSemaphoreList(&alloca1_signal_ptr, &alloca1_signal_value);

  iree_hal_buffer_t* buffer1 = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      test_device.base_device(), kQueueAffinity0,
      iree_hal_semaphore_list_empty(), alloca1_signal_list, pool,
      MakeTransientBufferParams(), allocation_size, IREE_HAL_ALLOCA_FLAG_NONE,
      &buffer1));
  ASSERT_NE(buffer1, nullptr);
  EXPECT_FALSE(iree_hal_semaphore_list_poll(alloca1_signal_list));
  ASSERT_TRUE(HostQueueHasPendingOps(queue));

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_GE(stats.exhausted_count, 1u);

  iree_hal_amdgpu_host_queue_cancel_pending(queue, IREE_STATUS_CANCELLED,
                                            "test cancellation");
  EXPECT_FALSE(HostQueueHasPendingOps(queue));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_hal_semaphore_wait(alloca1_signal, alloca1_signal_value,
                              iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_release(buffer1);
  iree_hal_buffer_release(buffer0);
}

}  // namespace
}  // namespace iree::hal::amdgpu
