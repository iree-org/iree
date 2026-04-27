// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_UTIL_PROFILE_TEST_UTIL_H_
#define IREE_HAL_CTS_UTIL_PROFILE_TEST_UTIL_H_

#include <cstdint>
#include <vector>

#include "iree/hal/api.h"

namespace iree::hal::cts {

// RAII helper that ends an active device profiling session on destruction.
class DeviceProfilingScope {
 public:
  explicit DeviceProfilingScope(iree_hal_device_t* device) : device_(device) {}

  ~DeviceProfilingScope();

  iree_status_t Begin(iree_hal_device_profiling_data_families_t data_families,
                      iree_hal_profile_sink_t* sink = nullptr);

  iree_status_t Begin(const iree_hal_device_profiling_options_t* options);

  iree_status_t End();

 private:
  // Device whose profiling session is active.
  iree_hal_device_t* device_ = nullptr;

  // True when |device_| has an active profiling session owned by this scope.
  bool is_active_ = false;
};

// Profile sink implementation that copies native profile records into vectors
// so CTS tests can inspect them after profiling_end.
struct TestProfileSink {
  // HAL resource header for the profile sink.
  iree_hal_resource_t resource;

  // Number of session begin notifications observed.
  int begin_count = 0;

  // Number of session end notifications observed.
  int end_count = 0;

  // Number of device metadata chunks observed.
  int device_metadata_count = 0;

  // Number of queue metadata chunks observed.
  int queue_metadata_count = 0;

  // Number of executable metadata chunks observed.
  int executable_metadata_count = 0;

  // Number of executable export metadata chunks observed.
  int executable_export_metadata_count = 0;

  // Number of clock correlation chunks observed.
  int clock_correlation_count = 0;

  // Number of dispatch event chunks observed.
  int dispatch_event_count = 0;

  // Number of queue event chunks observed.
  int queue_event_count = 0;

  // Number of host execution event chunks observed.
  int host_execution_event_count = 0;

  // Number of queue device event chunks observed.
  int queue_device_event_count = 0;

  // Number of event relationship chunks observed.
  int event_relationship_count = 0;

  // Clock correlation records copied from CLOCK_CORRELATIONS chunks.
  std::vector<iree_hal_profile_clock_correlation_record_t> clock_correlations;

  // Dispatch event records copied from DISPATCH_EVENTS chunks.
  std::vector<iree_hal_profile_dispatch_event_t> dispatch_events;

  // Queue event records copied from QUEUE_EVENTS chunks.
  std::vector<iree_hal_profile_queue_event_t> queue_events;

  // Host execution event records copied from HOST_EXECUTION_EVENTS chunks.
  std::vector<iree_hal_profile_host_execution_event_t> host_execution_events;

  // Queue device event records copied from QUEUE_DEVICE_EVENTS chunks.
  std::vector<iree_hal_profile_queue_device_event_t> queue_device_events;

  // Event relationship records copied from EVENT_RELATIONSHIPS chunks.
  std::vector<iree_hal_profile_event_relationship_record_t> event_relationships;

  // Executable identifiers copied from EXECUTABLES chunks.
  std::vector<uint64_t> executable_ids;

  // Executable identifiers referenced by EXECUTABLE_EXPORTS chunks.
  std::vector<uint64_t> export_record_executable_ids;

  // Physical device ordinals for entries in |dispatch_events|.
  std::vector<uint32_t> dispatch_event_physical_device_ordinals;

  // Dispatch event flags expected for every event record.
  iree_hal_profile_dispatch_event_flags_t expected_dispatch_flags =
      IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_NONE;

  // True if dispatch event workgroup counts should be checked.
  bool validate_dispatch_workgroup_count = true;

  // Expected dispatch event workgroup counts when validated.
  uint32_t expected_workgroup_count[3] = {1, 1, 1};

  // True after the device metadata chunk has been observed.
  bool saw_device_metadata = false;

  // True after the queue metadata chunk has been observed.
  bool saw_queue_metadata = false;

  // True if the backend writes after ending the profiling session.
  bool write_after_end = false;

  // Session identifier observed at begin and expected on later callbacks.
  uint64_t session_id = 0;
};

// Initializes |sink| for use as an iree_hal_profile_sink_t.
void TestProfileSinkInitialize(TestProfileSink* sink);

// Casts |sink| to its HAL profile sink base pointer.
iree_hal_profile_sink_t* TestProfileSinkAsBase(TestProfileSink* sink);

// Returns true when |status| means a backend does not support the requested
// profiling mode in a cross-backend CTS test.
bool IsProfilingUnsupported(iree_status_t status);

// Verifies all dispatch events have device ticks bracketed by clock
// correlation records for their physical device.
void ExpectDispatchEventsWithinClockCorrelationRange(
    const TestProfileSink& sink);

// Verifies all queue device events have device ticks bracketed by clock
// correlation records for their physical device.
void ExpectQueueDeviceEventsWithinClockCorrelationRange(
    const TestProfileSink& sink);

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS_UTIL_PROFILE_TEST_UTIL_H_
