// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_service.h"

#include <thread>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/device/host_client.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::testing::status::StatusIs;

// Returns an error callback that will assign the
static iree_hal_amdgpu_error_callback_t MakeErrorCallback(
    iree_atomic_intptr_t* status_bind) {
  iree_hal_amdgpu_error_callback_t callback;
  callback.fn = +[](void* user_data, iree_status_t status) {
    IREE_TRACE_SCOPE();
    iree_atomic_store((iree_atomic_intptr_t*)user_data, (intptr_t)status,
                      iree_memory_order_seq_cst);
  };
  callback.user_data = (void*)status_bind;
  return callback;
}

// Returns the first fine-grained global region of the |host_agent|.
static hsa_region_t GetHostGlobalFineRegion(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t host_agent) {
  IREE_TRACE_SCOPE();
  typedef struct iree_hal_amdgpu_hsa_region_list_t {
    iree_host_size_t count;
    hsa_region_t values[32];
  } iree_hal_amdgpu_hsa_region_list_t;
  iree_hal_amdgpu_hsa_region_list_t all_regions = {
      .count = 0,
  };
  IREE_CHECK_OK(iree_hsa_agent_iterate_regions(
      IREE_LIBHSA(libhsa), host_agent,
      +[](hsa_region_t region, void* user_data) -> hsa_status_t {
        auto* pool_list = (iree_hal_amdgpu_hsa_region_list_t*)user_data;
        if (pool_list->count + 1 >= IREE_ARRAYSIZE(pool_list->values)) {
          return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
        }
        pool_list->values[pool_list->count++] = region;
        return HSA_STATUS_SUCCESS;
      },
      &all_regions));
  for (iree_host_size_t i = 0; i < all_regions.count; ++i) {
    hsa_region_t region = all_regions.values[i];
    hsa_region_segment_t segment = (hsa_region_segment_t)0;
    IREE_CHECK_OK(iree_hsa_region_get_info(IREE_LIBHSA(libhsa), region,
                                           HSA_REGION_INFO_SEGMENT, &segment));
    if (segment != HSA_REGION_SEGMENT_GLOBAL) continue;
    bool alloc_allowed = false;
    IREE_CHECK_OK(iree_hsa_region_get_info(
        IREE_LIBHSA(libhsa), region, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED,
        &alloc_allowed));
    if (!alloc_allowed) continue;
    hsa_region_global_flag_t global_flag = (hsa_region_global_flag_t)0;
    IREE_CHECK_OK(iree_hsa_region_get_info(IREE_LIBHSA(libhsa), region,
                                           HSA_REGION_INFO_GLOBAL_FLAGS,
                                           &global_flag));
    if (global_flag & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) {
      return region;
    }
  }
  return {0};
}

struct HsaSignal {
  HsaSignal(const iree_hal_amdgpu_libhsa_t* libhsa,
            hsa_signal_value_t initial_value = 0)
      : libhsa(libhsa) {
    IREE_CHECK_OK(
        iree_hsa_amd_signal_create(IREE_LIBHSA(libhsa), initial_value, 0, NULL,
                                   (hsa_amd_signal_attribute_t)0, &value));
  }
  ~HsaSignal() {
    if (value.handle) {
      iree_hsa_signal_destroy(IREE_LIBHSA(libhsa), value);
    }
  }
  operator hsa_signal_t() const noexcept { return value; }
  const iree_hal_amdgpu_libhsa_t* libhsa = nullptr;
  hsa_signal_t value = {0};
};

// Enqueues a HSA_PACKET_TYPE_BARRIER_AND or HSA_PACKET_TYPE_BARRIER_OR
// packet to the service queue.
void EnqueueBarrier(iree_hal_amdgpu_host_service_t* service,
                    hsa_packet_type_t packet_type,
                    std::array<hsa_signal_t, 5> dep_signals,
                    hsa_signal_t completion_signal) {
  IREE_TRACE_SCOPE();

  const uint64_t packet_id = iree_hsa_queue_add_write_index_relaxed(
      IREE_LIBHSA(service->libhsa), service->queue, 1u);
  while (packet_id - iree_hsa_queue_load_read_index_scacquire(
                         IREE_LIBHSA(service->libhsa), service->queue) >=
         service->queue->size) {
    iree_thread_yield();  // spinning
  }
  const uint64_t queue_mask = service->queue->size - 1;  // power of two
  hsa_barrier_or_packet_t* packet =
      (hsa_barrier_or_packet_t*)((uint8_t*)service->queue->base_address +
                                 (packet_id & queue_mask) * 64);

  packet->reserved1 = 0;
  memcpy(&packet->dep_signal[0], &dep_signals[0], sizeof(packet->dep_signal));
  packet->reserved2 = 0;
  packet->completion_signal = completion_signal;

  // NOTE: high uint16_t is reserved0.
  uint32_t header = packet_type << HSA_PACKET_HEADER_TYPE;
  header |= 1 << HSA_PACKET_HEADER_BARRIER;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
  iree_atomic_store((iree_atomic_uint32_t*)packet, header,
                    iree_memory_order_release);

  iree_hsa_signal_store_relaxed(IREE_LIBHSA(service->libhsa), service->doorbell,
                                packet_id);
}

// Enqueues a HSA_AMD_PACKET_TYPE_BARRIER_VALUE packet to the service queue.
void EnqueueBarrierValue(iree_hal_amdgpu_host_service_t* service,
                         hsa_signal_t dep_signal,
                         hsa_signal_condition32_t condition,
                         hsa_signal_value_t condition_value,
                         hsa_signal_value_t condition_mask,
                         hsa_signal_t completion_signal) {
  IREE_TRACE_SCOPE();

  const uint64_t packet_id = iree_hsa_queue_add_write_index_relaxed(
      IREE_LIBHSA(service->libhsa), service->queue, 1u);
  while (packet_id - iree_hsa_queue_load_read_index_scacquire(
                         IREE_LIBHSA(service->libhsa), service->queue) >=
         service->queue->size) {
    iree_thread_yield();  // spinning
  }
  const uint64_t queue_mask = service->queue->size - 1;  // power of two
  hsa_amd_barrier_value_packet_t* packet =
      (hsa_amd_barrier_value_packet_t*)((uint8_t*)service->queue->base_address +
                                        (packet_id & queue_mask) * 64);

  packet->reserved0 = 0;
  packet->signal = dep_signal;
  packet->value = condition_value;
  packet->mask = condition_mask;
  packet->cond = condition;
  packet->reserved1 = 0;
  packet->reserved2 = 0;
  packet->reserved3 = 0;
  packet->completion_signal = completion_signal;

  union {
    hsa_amd_vendor_packet_header_t vendor_header;
    uint32_t vendor_header_bits;
  };
  vendor_header.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC
                         << HSA_PACKET_HEADER_TYPE;
  vendor_header.header |= 1 << HSA_PACKET_HEADER_BARRIER;
  vendor_header.header |= HSA_FENCE_SCOPE_SYSTEM
                          << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
  vendor_header.header |= HSA_FENCE_SCOPE_SYSTEM
                          << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
  vendor_header.AmdFormat = HSA_AMD_PACKET_TYPE_BARRIER_VALUE;
  vendor_header.reserved = 0;
  iree_atomic_store((iree_atomic_uint32_t*)packet, vendor_header_bits,
                    iree_memory_order_release);

  iree_hsa_signal_store_relaxed(IREE_LIBHSA(service->libhsa), service->doorbell,
                                packet_id);
}

// Enqueues a unidirectional agent packet to the service queue.
void EnqueuePost(iree_hal_amdgpu_host_service_t* service, uint16_t type,
                 uint64_t return_address, uint64_t arg0, uint64_t arg1,
                 uint64_t arg2, uint64_t arg3, hsa_signal_t completion_signal) {
  IREE_TRACE_SCOPE();

  const uint64_t packet_id = iree_hsa_queue_add_write_index_relaxed(
      IREE_LIBHSA(service->libhsa), service->queue, 1u);
  while (packet_id - iree_hsa_queue_load_read_index_scacquire(
                         IREE_LIBHSA(service->libhsa), service->queue) >=
         service->queue->size) {
    iree_thread_yield();  // spinning
  }
  const uint64_t queue_mask = service->queue->size - 1;  // power of two
  hsa_agent_dispatch_packet_t* agent_packet =
      (hsa_agent_dispatch_packet_t*)((uint8_t*)service->queue->base_address +
                                     (packet_id & queue_mask) * 64);

  agent_packet->reserved0 = 0;
  agent_packet->return_address = (void*)return_address;
  agent_packet->arg[0] = arg0;
  agent_packet->arg[1] = arg1;
  agent_packet->arg[2] = arg2;
  agent_packet->arg[3] = arg3;
  agent_packet->reserved2 = 0;
  agent_packet->completion_signal = completion_signal;

  uint16_t header = HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE;
  header |= 1 << HSA_PACKET_HEADER_BARRIER;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
  const uint32_t header_type = header | (uint32_t)(type << 16);
  iree_atomic_store((iree_atomic_uint32_t*)agent_packet, header_type,
                    iree_memory_order_release);

  iree_hsa_signal_store_relaxed(IREE_LIBHSA(service->libhsa), service->doorbell,
                                packet_id);
}

struct HostServiceTest : public ::testing::Test {
  static iree_allocator_t host_allocator;
  static iree_hal_amdgpu_libhsa_t libhsa;
  static iree_hal_amdgpu_topology_t topology;
  static hsa_region_t host_fine_region;

  static void SetUpTestSuite() {
    IREE_TRACE_SCOPE();
    host_allocator = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator, &libhsa);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_ignore(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(
        iree_hal_amdgpu_topology_initialize_with_defaults(&libhsa, &topology));
    if (topology.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
    host_fine_region = GetHostGlobalFineRegion(&libhsa, topology.cpu_agents[0]);
  }

  static void TearDownTestSuite() {
    IREE_TRACE_SCOPE();
    iree_hal_amdgpu_topology_deinitialize(&topology);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
  }
};
iree_allocator_t HostServiceTest::host_allocator;
iree_hal_amdgpu_libhsa_t HostServiceTest::libhsa;
iree_hal_amdgpu_topology_t HostServiceTest::topology;
hsa_region_t HostServiceTest::host_fine_region;

// Tests that the host service can be initialized/deinitialized immediately.
TEST_F(HostServiceTest, Lifetime) {
  IREE_TRACE_SCOPE();

  iree_atomic_intptr_t service_status = IREE_ATOMIC_VAR_INIT(0);
  iree_hal_amdgpu_host_service_t service = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_host_service_initialize(
      &libhsa, /*host_ordinal=*/0, topology.cpu_agents[0], host_fine_region,
      /*device_ordinal=*/0, MakeErrorCallback(&service_status), host_allocator,
      &service));

  iree_hal_amdgpu_host_service_deinitialize(&service);
  IREE_EXPECT_OK((iree_status_t)iree_atomic_load(&service_status,
                                                 iree_memory_order_seq_cst));
}

// Tests handling of the HSA_PACKET_TYPE_BARRIER_AND packet type.
TEST_F(HostServiceTest, BarrierAnd) {
  IREE_TRACE_SCOPE();

  iree_atomic_intptr_t service_status = IREE_ATOMIC_VAR_INIT(0);
  iree_hal_amdgpu_host_service_t service = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_host_service_initialize(
      &libhsa, /*host_ordinal=*/0, topology.cpu_agents[0], host_fine_region,
      /*device_ordinal=*/0, MakeErrorCallback(&service_status), host_allocator,
      &service));

  HsaSignal signal0(&libhsa, /*initial_value=*/1);
  HsaSignal signal1(&libhsa, /*initial_value=*/1);

  HsaSignal completion_signal(&libhsa, /*initial_value=*/1);
  EnqueueBarrier(&service, HSA_PACKET_TYPE_BARRIER_AND, {signal0, signal1},
                 completion_signal);

  std::thread thread0([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    iree_hsa_signal_subtract_screlease(IREE_LIBHSA(&libhsa), signal0, 1);
  });
  std::thread thread1([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    iree_hsa_signal_subtract_screlease(IREE_LIBHSA(&libhsa), signal1, 1);
  });

  EXPECT_EQ(
      0, iree_hsa_signal_wait_scacquire(IREE_LIBHSA(&libhsa), completion_signal,
                                        HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                                        HSA_WAIT_STATE_BLOCKED));

  thread0.join();
  thread1.join();

  iree_hal_amdgpu_host_service_deinitialize(&service);
  IREE_EXPECT_OK((iree_status_t)iree_atomic_load(&service_status,
                                                 iree_memory_order_seq_cst));
}

// Tests handling of the HSA_PACKET_TYPE_BARRIER_OR packet type.
TEST_F(HostServiceTest, BarrierOr) {
  IREE_TRACE_SCOPE();

  iree_atomic_intptr_t service_status = IREE_ATOMIC_VAR_INIT(0);
  iree_hal_amdgpu_host_service_t service = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_host_service_initialize(
      &libhsa, /*host_ordinal=*/0, topology.cpu_agents[0], host_fine_region,
      /*device_ordinal=*/0, MakeErrorCallback(&service_status), host_allocator,
      &service));

  HsaSignal signal0(&libhsa, /*initial_value=*/1);
  HsaSignal signal1(&libhsa, /*initial_value=*/1);

  HsaSignal completion_signal(&libhsa, /*initial_value=*/1);
  EnqueueBarrier(&service, HSA_PACKET_TYPE_BARRIER_OR, {signal0, signal1},
                 completion_signal);

  // NOTE: we only resolve one signal; the other is never resolved so we can
  // test the OR behavior.
  std::thread thread0([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    iree_hsa_signal_subtract_screlease(IREE_LIBHSA(&libhsa), signal0, 1);
  });

  EXPECT_EQ(
      0, iree_hsa_signal_wait_scacquire(IREE_LIBHSA(&libhsa), completion_signal,
                                        HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                                        HSA_WAIT_STATE_BLOCKED));

  thread0.join();

  iree_hal_amdgpu_host_service_deinitialize(&service);
  IREE_EXPECT_OK((iree_status_t)iree_atomic_load(&service_status,
                                                 iree_memory_order_seq_cst));
}

// Tests handling of the HSA_AMD_PACKET_TYPE_BARRIER_VALUE packet type.
TEST_F(HostServiceTest, BarrierValue) {
  IREE_TRACE_SCOPE();

  iree_atomic_intptr_t service_status = IREE_ATOMIC_VAR_INIT(0);
  iree_hal_amdgpu_host_service_t service = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_host_service_initialize(
      &libhsa, /*host_ordinal=*/0, topology.cpu_agents[0], host_fine_region,
      /*device_ordinal=*/0, MakeErrorCallback(&service_status), host_allocator,
      &service));

  HsaSignal signal(&libhsa, /*initial_value=*/0);

  HsaSignal completion_signal(&libhsa, /*initial_value=*/1);
  EnqueueBarrierValue(&service, signal, HSA_SIGNAL_CONDITION_GTE,
                      /*condition_value=*/10, /*condition_mask=*/UINT64_MAX,
                      completion_signal);

  std::thread thread([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    iree_hsa_signal_add_screlease(IREE_LIBHSA(&libhsa), signal, 11);
  });

  EXPECT_EQ(
      0, iree_hsa_signal_wait_scacquire(IREE_LIBHSA(&libhsa), completion_signal,
                                        HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                                        HSA_WAIT_STATE_BLOCKED));

  thread.join();

  iree_hal_amdgpu_host_service_deinitialize(&service);
  IREE_EXPECT_OK((iree_status_t)iree_atomic_load(&service_status,
                                                 iree_memory_order_seq_cst));
}

// Tests that service worker errors are propagated to the error callback.
TEST_F(HostServiceTest, FailureState) {
  IREE_TRACE_SCOPE();

  iree_atomic_intptr_t service_status = IREE_ATOMIC_VAR_INIT(0);
  iree_hal_amdgpu_host_service_t service = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_host_service_initialize(
      &libhsa, /*host_ordinal=*/0, topology.cpu_agents[0], host_fine_region,
      /*device_ordinal=*/0, MakeErrorCallback(&service_status), host_allocator,
      &service));

  HsaSignal completion_signal(&libhsa, /*initial_value=*/1);
  EnqueuePost(&service, /*type=*/UINT16_MAX, 0, 0, 0, 0, 0, completion_signal);

  // NOTE: it's not required that the service ever signal completion - it is
  // allowed to immediately stop processing packets during launch if it desires.
  // We delay a bit to give the worker time to process the packet and hope that
  // it reaches its error state. Real usage of the service should never touch
  // the internal data structures.
  for (int i = 0; i < 1000; ++i) {
    if (iree_atomic_load(&service_status, iree_memory_order_seq_cst) != 0) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  iree_hal_amdgpu_host_service_deinitialize(&service);
  EXPECT_THAT((iree_status_t)iree_atomic_load(&service_status,
                                              iree_memory_order_seq_cst),
              StatusIs(StatusCode::kInvalidArgument));
}

typedef struct iree_hal_test_resource_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_atomic_uint32_t* live_count;
} iree_hal_test_resource_t;
typedef struct iree_hal_test_resource_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_test_resource_t* resource);
} iree_hal_test_resource_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_test_resource_vtable_t);
static const iree_hal_test_resource_vtable_t iree_hal_test_resource_vtable = {
    /*.destroy=*/+[](iree_hal_test_resource_t* resource) {
      iree_hal_test_resource_t* test_resource =
          (iree_hal_test_resource_t*)resource;
      iree_allocator_t host_allocator = test_resource->host_allocator;
      iree_atomic_fetch_sub(test_resource->live_count, 1u,
                            iree_memory_order_seq_cst);
      iree_allocator_free(host_allocator, test_resource);
    },
};
static iree_status_t iree_hal_test_resource_create(
    iree_atomic_uint32_t* live_count, iree_allocator_t host_allocator,
    iree_hal_resource_t** out_resource) {
  iree_hal_test_resource_t* test_resource = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*test_resource), (void**)&test_resource));
  iree_hal_resource_initialize(&iree_hal_test_resource_vtable,
                               &test_resource->resource);
  test_resource->host_allocator = host_allocator;
  test_resource->live_count = live_count;
  iree_atomic_fetch_add(test_resource->live_count, 1u,
                        iree_memory_order_seq_cst);
  *out_resource = (iree_hal_resource_t*)test_resource;
  return iree_ok_status();
}

// Tests IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_RELEASE agent dispatch.
// This is primarily a leak test as the only thing that will release the
// resources is the service.
TEST_F(HostServiceTest, PostRelease) {
  IREE_TRACE_SCOPE();

  iree_atomic_intptr_t service_status = IREE_ATOMIC_VAR_INIT(0);
  iree_hal_amdgpu_host_service_t service = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_host_service_initialize(
      &libhsa, /*host_ordinal=*/0, topology.cpu_agents[0], host_fine_region,
      /*device_ordinal=*/0, MakeErrorCallback(&service_status), host_allocator,
      &service));

  // Create some resources to test with. Each will +1 the live_count.
  iree_atomic_uint32_t live_count = IREE_ATOMIC_VAR_INIT(0);
  iree_hal_resource_t* resources[5] = {NULL};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(resources); ++i) {
    IREE_ASSERT_OK(iree_hal_test_resource_create(&live_count, host_allocator,
                                                 &resources[i]));
  }

  // Release all resources in two batches.
  // Batch 0: only resource[0] in arg2. This tests that NULL values are ignored.
  // Batch 1: remaining 4 resources[1]/[2]/[3]/[4].
  HsaSignal completion_signal(&libhsa, /*initial_value=*/2);
  EnqueuePost(&service, IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_RELEASE,
              /*return_address=*/0ull, /*arg0=*/0ull,
              /*arg1=*/0ull, /*arg2=*/(uint64_t)resources[0], /*arg3=*/0ull,
              completion_signal);
  EnqueuePost(&service, IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_RELEASE,
              /*return_address=*/0ull, /*arg0=*/(uint64_t)resources[1],
              /*arg1=*/(uint64_t)resources[2], /*arg2=*/(uint64_t)resources[3],
              /*arg3=*/(uint64_t)resources[4], completion_signal);

  // Await releases to complete.
  EXPECT_EQ(
      0, iree_hsa_signal_wait_scacquire(IREE_LIBHSA(&libhsa), completion_signal,
                                        HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                                        HSA_WAIT_STATE_BLOCKED));

  // All resources should have been released.
  EXPECT_EQ(0, iree_atomic_load(&live_count, iree_memory_order_seq_cst));

  iree_hal_amdgpu_host_service_deinitialize(&service);
  IREE_EXPECT_OK((iree_status_t)iree_atomic_load(&service_status,
                                                 iree_memory_order_seq_cst));
}

// TODO(benvanik): async iree_hal_amdgpu_host_service_notify_completion when
// there is a command using it. Today all are unidirectional post-only.

}  // namespace
}  // namespace iree::hal::amdgpu
