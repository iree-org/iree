// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for async relay operations using Windows primitives.
//
// These tests exercise the NOTIFICATION → SIGNAL_PRIMITIVE relay path on
// Windows using native Win32 Event objects. This is the primary relay use case:
// bridging a notification signal to a Windows primitive for external consumers
// (semaphore signaling, device wake, etc.).
//
// The portable notification-to-notification relay tests are in relay_test.cc.
// The POSIX primitive relay tests (eventfd/pipe) are in relay_posix_test.cc.
//
// Synchronization model: relay CQEs are processed synchronously during
// DrainPending. After DrainPending returns, all relay side effects (SetEvent
// calls, notification epoch increments) are complete.

#if defined(_WIN32)

// clang-format off
#include <windows.h>
// clang-format on

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/notification.h"
#include "iree/async/relay.h"

namespace iree::async::cts {

class RelayWindowsTest : public CtsTestBase<> {
 protected:
  // Creates a Windows Event for use as a relay sink primitive.
  // Auto-reset mode: event resets after a single wait is satisfied.
  HANDLE CreateTestEvent() {
    HANDLE event = CreateEventW(NULL, /*bManualReset=*/FALSE,
                                /*bInitialState=*/FALSE, NULL);
    EXPECT_NE(event, static_cast<HANDLE>(NULL));
    return event;
  }

  // Returns true if the event is currently signaled (non-blocking check).
  // Consumes the event signal for auto-reset events.
  bool IsEventSignaled(HANDLE event) {
    return WaitForSingleObject(event, 0) == WAIT_OBJECT_0;
  }
};

// One-shot NOTIFICATION source triggers SIGNAL_PRIMITIVE sink (Win32 Event).
TEST_P(RelayWindowsTest, NotificationToPrimitive) {
  iree_async_notification_t* source_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &source_notification));

  HANDLE sink_event = CreateTestEvent();
  ASSERT_NE(sink_event, static_cast<HANDLE>(NULL));

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_, iree_async_relay_source_from_notification(source_notification),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_win32_handle((uintptr_t)sink_event), 1),
      IREE_ASYNC_RELAY_FLAG_NONE, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  iree_async_notification_signal(source_notification, 1);
  DrainPending();

  EXPECT_TRUE(IsEventSignaled(sink_event));

  CloseHandle(sink_event);
  iree_async_notification_release(source_notification);
}

// Persistent NOTIFICATION→SIGNAL_PRIMITIVE relay fires multiple times.
TEST_P(RelayWindowsTest, PersistentNotificationToPrimitive) {
  iree_async_notification_t* source_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &source_notification));

  HANDLE sink_event = CreateTestEvent();
  ASSERT_NE(sink_event, static_cast<HANDLE>(NULL));

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_, iree_async_relay_source_from_notification(source_notification),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_win32_handle((uintptr_t)sink_event), 1),
      IREE_ASYNC_RELAY_FLAG_PERSISTENT, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  for (int i = 0; i < 3; ++i) {
    iree_async_notification_signal(source_notification, 1);
    DrainPending();

    EXPECT_TRUE(IsEventSignaled(sink_event))
        << "Relay failed to fire on iteration " << i;
    // Auto-reset event resets after WaitForSingleObject returns WAIT_OBJECT_0
    // (consumed by IsEventSignaled), ready for the next iteration.
  }

  iree_async_proactor_unregister_relay(proactor_, relay);
  DrainPending();

  CloseHandle(sink_event);
  iree_async_notification_release(source_notification);
}

// Multiple relays from one notification to different Win32 Event sinks.
TEST_P(RelayWindowsTest, MultipleRelaysToDifferentPrimitives) {
  iree_async_notification_t* source_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &source_notification));

  constexpr int kNumRelays = 3;
  HANDLE sink_events[kNumRelays] = {};
  iree_async_relay_t* relays[kNumRelays] = {};

  for (int i = 0; i < kNumRelays; ++i) {
    sink_events[i] = CreateTestEvent();
    ASSERT_NE(sink_events[i], static_cast<HANDLE>(NULL));

    IREE_ASSERT_OK(iree_async_proactor_register_relay(
        proactor_,
        iree_async_relay_source_from_notification(source_notification),
        iree_async_relay_sink_signal_primitive(
            iree_async_primitive_from_win32_handle((uintptr_t)sink_events[i]),
            1),
        IREE_ASYNC_RELAY_FLAG_NONE, iree_async_relay_error_callback_none(),
        &relays[i]));
    ASSERT_NE(relays[i], nullptr);
  }

  iree_async_notification_signal(source_notification, 1);
  DrainPending();

  for (int i = 0; i < kNumRelays; ++i) {
    EXPECT_TRUE(IsEventSignaled(sink_events[i]))
        << "Relay " << i << " failed to fire";
    CloseHandle(sink_events[i]);
  }

  iree_async_notification_release(source_notification);
}

// Unregister notification-to-primitive relay before any signal.
TEST_P(RelayWindowsTest, UnregisterWhilePending) {
  iree_async_notification_t* source_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &source_notification));

  HANDLE sink_event = CreateTestEvent();
  ASSERT_NE(sink_event, static_cast<HANDLE>(NULL));

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_, iree_async_relay_source_from_notification(source_notification),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_win32_handle((uintptr_t)sink_event), 1),
      IREE_ASYNC_RELAY_FLAG_PERSISTENT, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  iree_async_proactor_unregister_relay(proactor_, relay);
  DrainPending();

  // Signal after unregister — no relay SQE is watching.
  iree_async_notification_signal(source_notification, 1);
  DrainPending();

  EXPECT_FALSE(IsEventSignaled(sink_event))
      << "Relay should not fire after unregister";

  CloseHandle(sink_event);
  iree_async_notification_release(source_notification);
}

CTS_REGISTER_TEST_SUITE(RelayWindowsTest);

}  // namespace iree::async::cts

#endif  // defined(_WIN32)
