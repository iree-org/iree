// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_CTS_FILE_NATIVE_FILE_H_
#define IREE_ASYNC_CTS_FILE_NATIVE_FILE_H_

#include <cstdint>

#include "iree/async/file.h"
#include "iree/base/api.h"

namespace iree::async::cts {

typedef uint32_t NativeFileAccess;
enum NativeFileAccessBits : NativeFileAccess {
  kNativeFileAccessRead = 1u << 0,
  kNativeFileAccessWrite = 1u << 1,
};

// Opens a platform-native file handle and imports it into |proactor|.
//
// The returned file owns the native handle and must be released or closed
// through the async file API.
iree_status_t ImportNativeFile(iree_async_proactor_t* proactor,
                               iree_string_view_t path, NativeFileAccess access,
                               iree_async_file_t** out_file);

}  // namespace iree::async::cts

#endif  // IREE_ASYNC_CTS_FILE_NATIVE_FILE_H_
