// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for SHM carrier buffer registration and direct read/write.
//
// Creates in-process carrier pairs and exercises the region table, buffer
// registration, direct_write (signaling and non-signaling), direct_read,
// and REFERENCE entry handling in drain_rx. These are SHM-specific features
// not covered by the generic carrier CTS.

#include "iree/net/carrier/shm/carrier.h"

#include <atomic>
#include <cstring>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/net/carrier/shm/carrier_pair.h"
#include "iree/net/carrier/shm/shared_wake.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

//===----------------------------------------------------------------------===//
// Recv capture utility
//===----------------------------------------------------------------------===//

struct RecvCapture {
  std::vector<uint8_t> buffer;
  std::atomic<iree_host_size_t> total_bytes{0};

  static iree_status_t Handler(void* user_data, iree_async_span_t data,
                               iree_async_buffer_lease_t* lease) {
    auto* capture = static_cast<RecvCapture*>(user_data);
    uint8_t* ptr = iree_async_span_ptr(data);
    capture->buffer.insert(capture->buffer.end(), ptr, ptr + data.length);
    capture->total_bytes.fetch_add(data.length, std::memory_order_relaxed);
    iree_async_buffer_lease_release(lease);
    return iree_ok_status();
  }

  iree_net_carrier_recv_handler_t AsHandler() { return {Handler, this}; }
};

//===----------------------------------------------------------------------===//
// Completion tracking
//===----------------------------------------------------------------------===//

struct CompletionTracker {
  std::atomic<int> call_count{0};
  std::atomic<iree_host_size_t> total_bytes{0};

  static void Callback(void* callback_user_data, uint64_t operation_user_data,
                       iree_status_t status, iree_host_size_t bytes_transferred,
                       iree_async_buffer_lease_t* recv_lease) {
    auto* tracker = static_cast<CompletionTracker*>(callback_user_data);
    tracker->call_count.fetch_add(1, std::memory_order_relaxed);
    tracker->total_bytes.fetch_add(bytes_transferred,
                                   std::memory_order_relaxed);
    iree_status_ignore(status);
  }

  iree_net_carrier_callback_t AsCallback() { return {Callback, this}; }
};

//===----------------------------------------------------------------------===//
// Null recv handler
//===----------------------------------------------------------------------===//

static iree_status_t NullRecvHandler(void* user_data, iree_async_span_t data,
                                     iree_async_buffer_lease_t* lease) {
  iree_async_buffer_lease_release(lease);
  return iree_ok_status();
}

static iree_net_carrier_recv_handler_t MakeNullRecvHandler() {
  return {NullRecvHandler, nullptr};
}

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class ShmCarrierTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_status_t status = iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_);
    if (iree_status_is_unavailable(status)) {
      iree_status_ignore(status);
      GTEST_SKIP() << "Platform proactor unavailable";
    }
    IREE_ASSERT_OK(status);

    IREE_ASSERT_OK(iree_net_shm_shared_wake_create(
        proactor_, iree_allocator_system(), &shared_wake_));

    IREE_ASSERT_OK(iree_net_shm_carrier_create_pair(
        shared_wake_, shared_wake_, iree_net_shm_carrier_options_default(),
        completions_.AsCallback(), iree_allocator_system(), &client_,
        &server_));
  }

  void TearDown() override {
    if (client_) {
      iree_net_carrier_set_recv_handler(client_, MakeNullRecvHandler());
    }
    if (server_) {
      iree_net_carrier_set_recv_handler(server_, MakeNullRecvHandler());
    }
    if (client_) {
      DeactivateAndDrain(client_);
      iree_net_carrier_release(client_);
      client_ = nullptr;
    }
    if (server_) {
      DeactivateAndDrain(server_);
      iree_net_carrier_release(server_);
      server_ = nullptr;
    }
    if (shared_wake_) {
      iree_net_shm_shared_wake_release(shared_wake_);
      shared_wake_ = nullptr;
    }
    if (proactor_) {
      iree_async_proactor_release(proactor_);
      proactor_ = nullptr;
    }
  }

  void ActivateBoth(iree_net_carrier_recv_handler_t client_handler,
                    iree_net_carrier_recv_handler_t server_handler) {
    iree_net_carrier_set_recv_handler(client_, client_handler);
    iree_net_carrier_set_recv_handler(server_, server_handler);
    IREE_ASSERT_OK(iree_net_carrier_activate(client_));
    IREE_ASSERT_OK(iree_net_carrier_activate(server_));
  }

  bool PollUntil(std::function<bool()> condition,
                 iree_duration_t budget = iree_make_duration_ms(5000)) {
    iree_time_t deadline_ns = iree_time_now() + budget;
    iree_timeout_t timeout = iree_make_deadline(deadline_ns);
    while (!condition()) {
      if (iree_time_now() >= deadline_ns) return false;
      iree_host_size_t completed = 0;
      iree_status_t status =
          iree_async_proactor_poll(proactor_, timeout, &completed);
      if (!iree_status_is_deadline_exceeded(status)) {
        if (!iree_status_is_ok(status)) {
          iree_status_ignore(status);
          return false;
        }
      } else {
        iree_status_ignore(status);
      }
    }
    return true;
  }

  void DeactivateAndDrain(iree_net_carrier_t* carrier) {
    iree_net_carrier_state_t state = iree_net_carrier_state(carrier);
    if (state == IREE_NET_CARRIER_STATE_CREATED ||
        state == IREE_NET_CARRIER_STATE_DEACTIVATED) {
      return;
    }
    if (state == IREE_NET_CARRIER_STATE_ACTIVE) {
      iree_status_t status =
          iree_net_carrier_deactivate(carrier, nullptr, nullptr);
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        return;
      }
    }
    iree_time_t deadline_ns = iree_time_now() + iree_make_duration_ms(5000);
    while (iree_net_carrier_state(carrier) !=
           IREE_NET_CARRIER_STATE_DEACTIVATED) {
      if (iree_time_now() >= deadline_ns) break;
      iree_host_size_t completed = 0;
      iree_status_t status = iree_async_proactor_poll(
          proactor_, iree_make_timeout_ms(100), &completed);
      if (iree_status_is_deadline_exceeded(status)) {
        iree_status_ignore(status);
      } else {
        iree_status_ignore(status);
      }
    }
  }

  iree_async_proactor_t* proactor_ = nullptr;
  iree_net_shm_shared_wake_t* shared_wake_ = nullptr;
  iree_net_carrier_t* client_ = nullptr;
  iree_net_carrier_t* server_ = nullptr;
  CompletionTracker completions_;
};

//===----------------------------------------------------------------------===//
// Capabilities
//===----------------------------------------------------------------------===//

TEST_F(ShmCarrierTest, CapabilitiesIncludeDirectAccess) {
  iree_net_carrier_capabilities_t caps = iree_net_carrier_capabilities(client_);
  EXPECT_TRUE(caps & IREE_NET_CARRIER_CAPABILITY_REGISTERED_REGIONS);
  EXPECT_TRUE(caps & IREE_NET_CARRIER_CAPABILITY_DIRECT_WRITE);
  EXPECT_TRUE(caps & IREE_NET_CARRIER_CAPABILITY_DIRECT_READ);
}

//===----------------------------------------------------------------------===//
// query_region
//===----------------------------------------------------------------------===//

TEST_F(ShmCarrierTest, QueryRegionValidIds) {
  // Pair carriers get two regions (creator + opener mappings).
  iree_net_shm_region_info_t region0 = {};
  IREE_ASSERT_OK(iree_net_shm_carrier_query_region(client_, 0, &region0));
  EXPECT_NE(region0.base_ptr, nullptr);
  EXPECT_GT(region0.size, 0u);

  iree_net_shm_region_info_t region1 = {};
  IREE_ASSERT_OK(iree_net_shm_carrier_query_region(client_, 1, &region1));
  EXPECT_NE(region1.base_ptr, nullptr);
  EXPECT_GT(region1.size, 0u);

  // The two regions should be distinct mappings.
  EXPECT_NE(region0.base_ptr, region1.base_ptr);
  // But the same size (same SHM region, different mappings).
  EXPECT_EQ(region0.size, region1.size);
}

TEST_F(ShmCarrierTest, QueryRegionOutOfRange) {
  iree_net_shm_region_info_t region = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_net_shm_carrier_query_region(client_, 99, &region));
}

//===----------------------------------------------------------------------===//
// register_buffer / unregister_buffer
//===----------------------------------------------------------------------===//

TEST_F(ShmCarrierTest, RegisterBufferInShmRegion) {
  iree_net_shm_region_info_t region_info = {};
  IREE_ASSERT_OK(iree_net_shm_carrier_query_region(client_, 0, &region_info));

  // Create a fake region referencing a sub-range of the SHM mapping.
  iree_async_region_t region = {};
  region.base_ptr = (uint8_t*)region_info.base_ptr + 256;
  region.length = 1024;

  iree_net_remote_handle_t handle = {};
  IREE_ASSERT_OK(iree_net_carrier_register_buffer(client_, &region, &handle));
  EXPECT_FALSE(iree_net_remote_handle_is_null(handle));
  EXPECT_EQ(handle.opaque[0], 0u);    // Region 0.
  EXPECT_EQ(handle.opaque[1], 256u);  // Offset within region.

  iree_net_carrier_unregister_buffer(client_, handle);
}

TEST_F(ShmCarrierTest, RegisterBufferOutsideShmFails) {
  // A region that doesn't fall within any SHM mapping.
  uint8_t stack_buffer[64];
  iree_async_region_t region = {};
  region.base_ptr = stack_buffer;
  region.length = sizeof(stack_buffer);

  iree_net_remote_handle_t handle = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_carrier_register_buffer(client_, &region, &handle));
  EXPECT_TRUE(iree_net_remote_handle_is_null(handle));
}

TEST_F(ShmCarrierTest, RegisterBufferSecondRegion) {
  iree_net_shm_region_info_t region_info = {};
  IREE_ASSERT_OK(iree_net_shm_carrier_query_region(client_, 1, &region_info));

  iree_async_region_t region = {};
  region.base_ptr = region_info.base_ptr;
  region.length = 512;

  iree_net_remote_handle_t handle = {};
  IREE_ASSERT_OK(iree_net_carrier_register_buffer(client_, &region, &handle));
  EXPECT_EQ(handle.opaque[0], 1u);  // Region 1.
  EXPECT_EQ(handle.opaque[1], 0u);  // Offset 0.

  iree_net_carrier_unregister_buffer(client_, handle);
}

//===----------------------------------------------------------------------===//
// iree_net_remote_handle_offset
//===----------------------------------------------------------------------===//

TEST_F(ShmCarrierTest, HandleOffsetAddsToOpaque1) {
  iree_net_remote_handle_t handle = {{42, 100}};
  iree_net_remote_handle_t offset_handle =
      iree_net_remote_handle_offset(handle, 50);
  EXPECT_EQ(offset_handle.opaque[0], 42u);
  EXPECT_EQ(offset_handle.opaque[1], 150u);
}

//===----------------------------------------------------------------------===//
// direct_write (non-signaling)
//===----------------------------------------------------------------------===//

TEST_F(ShmCarrierTest, DirectWriteNonSignaling) {
  // Register a sub-range of the SHM region on the server side.
  iree_net_shm_region_info_t server_region = {};
  IREE_ASSERT_OK(iree_net_shm_carrier_query_region(server_, 0, &server_region));

  // Pick an offset well past the ring headers to avoid clobbering live data.
  // The SHM region is large (header + 2 * ring_size), so byte 0x1000 is safe.
  iree_host_size_t write_offset = 0x1000;
  iree_host_size_t write_length = 64;

  // Zero the target area first.
  memset((uint8_t*)server_region.base_ptr + write_offset, 0, write_length);

  // Build a handle pointing to this area in region 0.
  iree_net_remote_handle_t handle = {{0, write_offset}};

  // Prepare local source data.
  uint8_t source_data[64];
  memset(source_data, 0xAB, sizeof(source_data));

  // Non-signaling direct_write: just memcpy, no ring entry.
  ActivateBoth(MakeNullRecvHandler(), MakeNullRecvHandler());

  iree_net_direct_write_params_t params = {};
  params.local = iree_async_span_from_ptr(source_data, sizeof(source_data));
  params.remote = handle;
  params.flags = IREE_NET_DIRECT_WRITE_FLAG_NONE;
  params.user_data = 0;
  IREE_ASSERT_OK(iree_net_carrier_direct_write(client_, &params));

  // The data should be immediately visible in SHM (no async involved).
  // Both carriers share region 0, so query from client or server — same memory.
  iree_net_shm_region_info_t client_region = {};
  IREE_ASSERT_OK(iree_net_shm_carrier_query_region(client_, 0, &client_region));
  EXPECT_EQ(memcmp((uint8_t*)client_region.base_ptr + write_offset, source_data,
                   write_length),
            0);
}

//===----------------------------------------------------------------------===//
// direct_write (signaling)
//===----------------------------------------------------------------------===//

TEST_F(ShmCarrierTest, DirectWriteSignalingDeliversToRecvHandler) {
  iree_net_shm_region_info_t region_info = {};
  IREE_ASSERT_OK(iree_net_shm_carrier_query_region(client_, 0, &region_info));

  iree_host_size_t write_offset = 0x2000;
  iree_host_size_t write_length = 32;

  // Write distinctive data at the target offset.
  uint8_t source_data[32];
  for (size_t i = 0; i < sizeof(source_data); ++i) {
    source_data[i] = (uint8_t)(i + 1);
  }

  // Set up recv capture on the server to observe the REFERENCE delivery.
  RecvCapture server_recv;
  ActivateBoth(MakeNullRecvHandler(), server_recv.AsHandler());

  iree_net_direct_write_params_t params = {};
  params.local = iree_async_span_from_ptr(source_data, sizeof(source_data));
  params.remote = iree_net_remote_handle_t{{0, write_offset}};
  params.flags = IREE_NET_DIRECT_WRITE_FLAG_SIGNAL_RECEIVER;
  params.immediate = 0xDEAD;
  params.user_data = 42;
  IREE_ASSERT_OK(iree_net_carrier_direct_write(client_, &params));

  // Poll until the server's recv handler receives the data.
  ASSERT_TRUE(PollUntil(
      [&] { return server_recv.total_bytes.load() >= write_length; }));

  // The recv handler should have received the resolved SHM data.
  ASSERT_EQ(server_recv.buffer.size(), write_length);
  EXPECT_EQ(memcmp(server_recv.buffer.data(), source_data, write_length), 0);

  // The sender's completion callback should have fired.
  ASSERT_TRUE(PollUntil([&] { return completions_.call_count.load() >= 1; }));
  EXPECT_EQ(completions_.total_bytes.load(), write_length);
}

//===----------------------------------------------------------------------===//
// direct_read
//===----------------------------------------------------------------------===//

TEST_F(ShmCarrierTest, DirectReadFromShmRegion) {
  iree_net_shm_region_info_t region_info = {};
  IREE_ASSERT_OK(iree_net_shm_carrier_query_region(client_, 0, &region_info));

  // Write known data directly into SHM.
  iree_host_size_t read_offset = 0x3000;
  iree_host_size_t read_length = 48;
  uint8_t expected_data[48];
  for (size_t i = 0; i < sizeof(expected_data); ++i) {
    expected_data[i] = (uint8_t)(0xFF - i);
  }
  memcpy((uint8_t*)region_info.base_ptr + read_offset, expected_data,
         sizeof(expected_data));

  ActivateBoth(MakeNullRecvHandler(), MakeNullRecvHandler());

  // Read it back via direct_read.
  uint8_t read_buffer[48];
  memset(read_buffer, 0, sizeof(read_buffer));

  iree_net_direct_read_params_t params = {};
  params.local = iree_async_span_from_ptr(read_buffer, sizeof(read_buffer));
  params.remote = iree_net_remote_handle_t{{0, read_offset}};
  params.user_data = 0;
  IREE_ASSERT_OK(iree_net_carrier_direct_read(server_, &params));

  EXPECT_EQ(memcmp(read_buffer, expected_data, read_length), 0);
}

//===----------------------------------------------------------------------===//
// Bounds checking
//===----------------------------------------------------------------------===//

TEST_F(ShmCarrierTest, DirectWriteOutOfBoundsRegionId) {
  ActivateBoth(MakeNullRecvHandler(), MakeNullRecvHandler());

  uint8_t data[16] = {};
  iree_net_direct_write_params_t params = {};
  params.local = iree_async_span_from_ptr(data, sizeof(data));
  params.remote = iree_net_remote_handle_t{{99, 0}};
  params.flags = IREE_NET_DIRECT_WRITE_FLAG_NONE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_net_carrier_direct_write(client_, &params));
}

TEST_F(ShmCarrierTest, DirectWriteOutOfBoundsOffset) {
  iree_net_shm_region_info_t region_info = {};
  IREE_ASSERT_OK(iree_net_shm_carrier_query_region(client_, 0, &region_info));

  ActivateBoth(MakeNullRecvHandler(), MakeNullRecvHandler());

  uint8_t data[16] = {};
  iree_net_direct_write_params_t params = {};
  params.local = iree_async_span_from_ptr(data, sizeof(data));
  // Offset that, combined with length, exceeds the region size.
  params.remote = iree_net_remote_handle_t{{0, region_info.size}};
  params.flags = IREE_NET_DIRECT_WRITE_FLAG_NONE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_net_carrier_direct_write(client_, &params));
}

TEST_F(ShmCarrierTest, DirectReadOutOfBoundsRegionId) {
  ActivateBoth(MakeNullRecvHandler(), MakeNullRecvHandler());

  uint8_t data[16] = {};
  iree_net_direct_read_params_t params = {};
  params.local = iree_async_span_from_ptr(data, sizeof(data));
  params.remote = iree_net_remote_handle_t{{99, 0}};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_net_carrier_direct_read(client_, &params));
}

TEST_F(ShmCarrierTest, DirectReadOutOfBoundsOffset) {
  iree_net_shm_region_info_t region_info = {};
  IREE_ASSERT_OK(iree_net_shm_carrier_query_region(client_, 0, &region_info));

  ActivateBoth(MakeNullRecvHandler(), MakeNullRecvHandler());

  uint8_t data[16] = {};
  iree_net_direct_read_params_t params = {};
  params.local = iree_async_span_from_ptr(data, sizeof(data));
  params.remote = iree_net_remote_handle_t{{0, region_info.size}};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_net_carrier_direct_read(client_, &params));
}

//===----------------------------------------------------------------------===//
// Inline send still works (regression test)
//===----------------------------------------------------------------------===//

TEST_F(ShmCarrierTest, InlineSendStillWorksWithRegions) {
  RecvCapture server_recv;
  ActivateBoth(MakeNullRecvHandler(), server_recv.AsHandler());

  const char* msg = "hello regions";
  iree_async_span_t span =
      iree_async_span_from_ptr(const_cast<char*>(msg), strlen(msg));
  iree_net_send_params_t params = {};
  params.data.values = &span;
  params.data.count = 1;
  params.flags = IREE_NET_SEND_FLAG_NONE;
  params.user_data = 0;
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

  ASSERT_TRUE(
      PollUntil([&] { return server_recv.total_bytes.load() >= strlen(msg); }));
  EXPECT_EQ(server_recv.buffer.size(), strlen(msg));
  EXPECT_EQ(memcmp(server_recv.buffer.data(), msg, strlen(msg)), 0);
}

}  // namespace
