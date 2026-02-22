// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Internal header for io_uring notification creation and destruction.
//
// The notification struct and public APIs live in iree/async/notification.h and
// iree/async/notification.c respectively. This header declares only the
// io_uring-specific create/destroy functions called through the proactor
// vtable.

#ifndef IREE_ASYNC_PLATFORM_IO_URING_NOTIFICATION_H_
#define IREE_ASYNC_PLATFORM_IO_URING_NOTIFICATION_H_

#include "iree/async/notification.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_proactor_io_uring_t iree_async_proactor_io_uring_t;

// Creates an io_uring notification. Selects futex mode when the proactor has
// FUTEX_OPERATIONS capability, otherwise creates an eventfd for poll-based
// waits.
iree_status_t iree_async_io_uring_notification_create(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_notification_flags_t flags,
    iree_async_notification_t** out_notification);

// Destroys an io_uring notification, closing the eventfd if in event mode.
void iree_async_io_uring_notification_destroy(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_notification_t* notification);

// Platform-specific signal wakeup (futex_wake or eventfd write).
// Called from the shared notification_signal() after epoch increment.
void iree_async_io_uring_notification_signal(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_t* notification, int32_t wake_count);

// Platform-specific synchronous wait (futex_wait or poll on eventfd).
// Called from the shared notification_wait().
bool iree_async_io_uring_notification_wait(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_t* notification, iree_timeout_t timeout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_IO_URING_NOTIFICATION_H_
