// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/cts/util/test_base.h"

#include <memory>
#include <utility>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"

namespace iree::hal::cts {
namespace {

// Every CTS device gets a production-shaped frontier tracker. The capacity is
// generous for current CTS backends while still small enough to catch
// accidental steady-state axis registration.
static constexpr uint32_t kCtsFrontierAxisTableCapacity = 256;

}  // namespace

struct DeviceCreateContext::State {
  ~State() {
    iree_async_frontier_tracker_release(frontier_tracker);
    iree_async_proactor_pool_release(proactor_pool);
  }

  iree_async_proactor_pool_t* proactor_pool = nullptr;
  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  iree_hal_device_create_params_t params =
      iree_hal_device_create_params_default();
};

DeviceCreateContext::DeviceCreateContext() = default;

DeviceCreateContext::~DeviceCreateContext() = default;

iree_status_t DeviceCreateContext::Initialize(iree_allocator_t host_allocator) {
  if (state_) return iree_ok_status();

  auto state = std::make_unique<State>();
  iree_status_t status = iree_async_proactor_pool_create(
      iree_numa_node_count(), /*node_ids=*/NULL,
      iree_async_proactor_pool_options_default(), host_allocator,
      &state->proactor_pool);
  if (iree_status_is_ok(status)) {
    iree_async_frontier_tracker_options_t options =
        iree_async_frontier_tracker_options_default();
    options.axis_table_capacity = kCtsFrontierAxisTableCapacity;
    status = iree_async_frontier_tracker_create(options, host_allocator,
                                                &state->frontier_tracker);
  }

  if (iree_status_is_ok(status)) {
    state->params.proactor_pool = state->proactor_pool;
    state->params.frontier.tracker = state->frontier_tracker;
    // Source the base axis coordinates from the tracker itself so they stay
    // in sync with whatever tracker options the CTS harness was configured
    // with (rather than drifting if the tracker defaults ever change).
    state->params.frontier.base_axis = iree_async_axis_make_queue(
        iree_async_frontier_tracker_session_epoch(state->frontier_tracker),
        iree_async_frontier_tracker_machine_index(state->frontier_tracker),
        /*device_index=*/0, /*queue_index=*/0);
    state_ = std::move(state);
  }
  return status;
}

void DeviceCreateContext::Deinitialize() { state_.reset(); }

const iree_hal_device_create_params_t* DeviceCreateContext::params() const {
  IREE_ASSERT(state_, "DeviceCreateContext must be initialized");
  return &state_->params;
}

}  // namespace iree::hal::cts
