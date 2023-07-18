// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_WAIT_HANDLE_IMPL_H_
#define IREE_BASE_INTERNAL_WAIT_HANDLE_IMPL_H_

//===----------------------------------------------------------------------===//
// Platform overrides
//===----------------------------------------------------------------------===//
// NOTE: this must come first prior to any local/system includes!

// Ensure that any posix header we include exposes GNU stuff. Ignored on
// platforms where we either don't have the GNU stuff or don't have posix
// headers at all.
//
// Note that this does not need to be the same for all compilation units, only
// those we want to access the non-portable features in. It *must* be defined
// prior to including any of the files, though, as otherwise header-guards will
// cause the setting at the time of first inclusion to win.
//
// https://stackoverflow.com/a/5583764
#define _GNU_SOURCE 1

//===----------------------------------------------------------------------===//
// Active wait API implementation selection (wait_handle_*.c)
//===----------------------------------------------------------------------===//

#include "iree/base/config.h"
#include "iree/base/target_platform.h"

// NOTE: order matters; priorities are (kqueue|epoll) > ppoll > poll.
// When overridden with NULL (no platform primitives) or on Win32 we always use
// those implementations (today).
#define IREE_WAIT_API_NULL 0
#define IREE_WAIT_API_INPROC 1
#define IREE_WAIT_API_WIN32 2
#define IREE_WAIT_API_POLL 3
#define IREE_WAIT_API_PPOLL 4
#define IREE_WAIT_API_EPOLL 5
#define IREE_WAIT_API_KQUEUE 6
#define IREE_WAIT_API_PROMISE 7

// We allow overriding the wait API via command line flags. If unspecified we
// try to guess based on the target platform.
#if !defined(IREE_WAIT_API)

// NOTE: we could be tighter here, but we today only have win32 or not-win32.
#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE
#define IREE_WAIT_API IREE_WAIT_API_NULL
#elif defined(IREE_PLATFORM_EMSCRIPTEN)
#define IREE_WAIT_API IREE_WAIT_API_PROMISE
#elif defined(IREE_PLATFORM_GENERIC)
#define IREE_WAIT_API IREE_WAIT_API_INPROC
#elif defined(IREE_PLATFORM_WINDOWS)
#define IREE_WAIT_API IREE_WAIT_API_WIN32  // WFMO used in wait_handle_win32.c
#else
// TODO(benvanik): EPOLL on android/linux/bsd/etc.
// TODO(benvanik): KQUEUE on mac/ios.
// KQUEUE is not implemented yet. Use POLL for mac/ios
// Android ppoll requires API version >= 21
#if !defined(IREE_PLATFORM_APPLE) && \
    (!defined(__ANDROID_API__) || __ANDROID_API__ >= 21)
#define IREE_WAIT_API IREE_WAIT_API_PPOLL
#else
#define IREE_WAIT_API IREE_WAIT_API_POLL
#endif  // insanity
#endif  // IREE_SYNCHRONIZATION_DISABLE_UNSAFE / IREE_PLATFORM_WINDOWS

#endif  // !IREE_WAIT_API

// Many implementations share the same posix-like nature (file descriptors/etc)
// and can share most of their code.
#if (IREE_WAIT_API == IREE_WAIT_API_POLL) ||  \
    (IREE_WAIT_API == IREE_WAIT_API_PPOLL) || \
    (IREE_WAIT_API == IREE_WAIT_API_EPOLL) || \
    (IREE_WAIT_API == IREE_WAIT_API_KQUEUE)
#define IREE_WAIT_API_POSIX_LIKE 1
#endif  // IREE_WAIT_API = posix-like

//===----------------------------------------------------------------------===//
// Wait handle included with options set
//===----------------------------------------------------------------------===//

#include "iree/base/internal/wait_handle.h"

#endif  // IREE_BASE_INTERNAL_WAIT_HANDLE_IMPL_H_
