// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_QUEUE_HOST_CALL_TEST_H_
#define IREE_HAL_CTS_QUEUE_HOST_CALL_TEST_H_

#include <cstdint>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

struct SemaphoreList {
  SemaphoreList() = default;
  SemaphoreList(iree_hal_device_t* device, std::vector<uint64_t> initial_values,
                std::vector<uint64_t> desired_values) {
    for (size_t i = 0; i < initial_values.size(); ++i) {
      iree_hal_semaphore_t* semaphore = NULL;
      IREE_EXPECT_OK(iree_hal_semaphore_create(
          device, initial_values[i], IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
      semaphores.push_back(semaphore);
    }
    payload_values = desired_values;
    assert(semaphores.size() == payload_values.size());
  }

  SemaphoreList(const SemaphoreList&) = delete;             // no copy
  SemaphoreList& operator=(const SemaphoreList&) = delete;  // no copy

  SemaphoreList(SemaphoreList&& other) noexcept
      : semaphores(std::move(other.semaphores)),
        payload_values(std::move(other.payload_values)) {
    other.semaphores.clear();
    other.payload_values.clear();
  }

  SemaphoreList& operator=(SemaphoreList&& other) noexcept {
    if (this != &other) {
      iree_hal_semaphore_list_release((iree_hal_semaphore_list_t)(*this));
      semaphores = std::move(other.semaphores);
      payload_values = std::move(other.payload_values);
      other.semaphores.clear();
      other.payload_values.clear();
    }
    return *this;
  }

  ~SemaphoreList() {
    iree_hal_semaphore_list_release((iree_hal_semaphore_list_t)(*this));
  }

  operator iree_hal_semaphore_list_t() {
    iree_hal_semaphore_list_t list;
    list.count = semaphores.size();
    list.semaphores = semaphores.data();
    list.payload_values = payload_values.data();
    return list;
  }

  std::vector<iree_hal_semaphore_t*> semaphores;
  std::vector<uint64_t> payload_values;
};

class QueueHostCallTest : public CTSTestBase<> {};

// DO NOT SUBMIT

// iree_hal_semaphore_t* semaphore = CreateSemaphore(); // @0
//
// CheckSemaphoreValue(semaphore, expected)
//
// iree_hal_semaphore_release(semaphore);

// call with no wait semas

// call with wait on thread-signaled
//   create thread
//   thread sleep
//   thread signal

// call non-blocking
//   wait on signal semaphores (sidechannel) to ensure signaled first

// TEST_F(QueueHostCallTest, XXX) {
//   //  iree_hal_device_queue_host_call(device_
//   // IREE_API_EXPORT iree_status_t iree_hal_device_queue_host_call(
//   //     iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
//   //     const iree_hal_semaphore_list_t wait_semaphore_list,
//   //     const iree_hal_semaphore_list_t signal_semaphore_list,
//   //     iree_hal_host_call_t call, const uint64_t args[4],
//   //     iree_hal_host_call_flags_t flags) {
//   struct {
//     int reserved;
//   } state;
//   auto call = iree_hal_make_host_call(
//       +[](void* user_data, const uint64_t args[4],
//           iree_hal_host_call_context_t* context) {
//         //
//         return
//         iree_hal_semaphore_list_signal(context->signal_semaphore_list);
//       },
//       &state);
//   auto wait_semaphore_list = iree_hal_semaphore_list_empty();
//   auto signal_semaphore_list = iree_hal_semaphore_list_empty();
//   signal_semaphore_list.count = 1;
//   iree_hal_semaphore_t* signal_semaphore = CreateSemaphore();
//   signal_semaphore_list.semaphores = &signal_semaphore;
//   uint64_t signal_value = 1;
//   signal_semaphore_list.payload_values = &signal_value;
//   uint64_t args[4] = {0, 0, 0, 0};
//   iree_status_t status = iree_hal_device_queue_host_call(
//       device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
//       signal_semaphore_list, call, args, IREE_HAL_HOST_CALL_FLAG_NONE);

//   status = iree_hal_semaphore_list_wait(signal_semaphore_list,
//                                         iree_infinite_timeout());
// }

// TEST_F(QueueHostCallTest, XXX) {
//   struct {
//     int did_call;
//   } state;
//   auto call = iree_hal_make_host_call(
//       +[](void* user_data, const uint64_t args[4],
//           iree_hal_host_call_context_t* context) {
//         //
//         return
//         iree_hal_semaphore_list_signal(context->signal_semaphore_list);
//       },
//       &state);

//   SemaphoreList wait_semaphore_list(device_, {0}, {1});
//   SemaphoreList signal_semaphore_list(device_, {0}, {1});

//   uint64_t args[4] = {0, 0, 0, 0};
//   iree_status_t status = iree_hal_device_queue_host_call(
//       device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
//       signal_semaphore_list, call, args, IREE_HAL_HOST_CALL_FLAG_NONE);

//   std::thread waker([&]() {
//     std::this_thread::sleep_for(std::chrono::milliseconds(150));
//     status = iree_hal_semaphore_list_signal(wait_semaphore_list);
//   });

//   status = iree_hal_semaphore_list_wait(signal_semaphore_list,
//                                         iree_infinite_timeout());

//   printf("hi");

//   waker.join();
// }

TEST_F(QueueHostCallTest, XXX) {
  // use random other semaphore manually

  struct {
    int did_call;
  } state;
  auto call = iree_hal_make_host_call(
      +[](void* user_data, const uint64_t args[4],
          iree_hal_host_call_context_t* context) {
        //
        return iree_ok_status();
      },
      &state);

  SemaphoreList wait_semaphore_list(device_, {0}, {1});
  SemaphoreList signal_semaphore_list(device_, {0}, {1});

  uint64_t args[4] = {0, 0, 0, 0};
  iree_status_t status = iree_hal_device_queue_host_call(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, call, args, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING);

  std::thread waker([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    status = iree_hal_semaphore_list_signal(wait_semaphore_list);
  });

  status = iree_hal_semaphore_list_wait(signal_semaphore_list,
                                        iree_infinite_timeout());

  printf("hi");

  waker.join();
}

// DO NOT SUBMIT
// async test (host call copies semaphores, launches a thread, and then signals)
// error propagation from host call

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS_QUEUE_HOST_CALL_TEST_H_
