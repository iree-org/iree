// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

#include "iree/base/target_platform.h"

// Priorities are (kqueue|epoll) > ppoll > poll
#define IREE_WAIT_API_POLL 1
#define IREE_WAIT_API_PPOLL 2
#define IREE_WAIT_API_EPOLL 3
#define IREE_WAIT_API_KQUEUE 4

// NOTE: we could be tighter here, but we today only have win32 or not-win32.
#if defined(IREE_PLATFORM_WINDOWS)
#define IREE_WAIT_API 0  // WFMO used in wait_handle_win32.c
#else

// TODO(benvanik): EPOLL on android/linux/bsd/etc.
// TODO(benvanik): KQUEUE on mac/ios.
// KQUEUE is not implemented yet. Use POLL for mac/ios
// Android ppoll requires API version >= 21
#if !defined(IREE_PLATFORM_APPLE) && !defined(__EMSCRIPTEN__) && \
    (!defined(__ANDROID_API__) || __ANDROID_API__ >= 21)
#define IREE_WAIT_API IREE_WAIT_API_PPOLL
#else
#define IREE_WAIT_API IREE_WAIT_API_POLL
#endif  // insanity

#endif  // IREE_PLATFORM_WINDOWS

//===----------------------------------------------------------------------===//
// Wait handle included with options set
//===----------------------------------------------------------------------===//

#include "iree/base/internal/wait_handle.h"

#endif  // IREE_BASE_INTERNAL_WAIT_HANDLE_IMPL_H_
