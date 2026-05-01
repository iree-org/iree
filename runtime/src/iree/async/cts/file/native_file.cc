// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/cts/file/native_file.h"

#include <string>

#include "iree/async/primitive.h"

#if defined(IREE_PLATFORM_WINDOWS)
#include <windows.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#endif  // IREE_PLATFORM_WINDOWS

namespace iree::async::cts {

static constexpr NativeFileAccess kNativeFileAccessMask =
    kNativeFileAccessRead | kNativeFileAccessWrite;

iree_status_t ImportNativeFile(iree_async_proactor_t* proactor,
                               iree_string_view_t path, NativeFileAccess access,
                               iree_async_file_t** out_file) {
  if (!proactor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "proactor is required");
  }
  if (!out_file) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_file is required");
  }
  *out_file = nullptr;
  if (!iree_any_bit_set(access, kNativeFileAccessMask)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native file import requires read or write access");
  }
  if ((access & ~kNativeFileAccessMask) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native file import access contains unknown bits");
  }

  iree_async_primitive_t primitive = iree_async_primitive_none();
  const std::string path_string(path.data, path.size);
#if defined(IREE_PLATFORM_WINDOWS)
  DWORD desired_access = 0;
  if (iree_any_bit_set(access, kNativeFileAccessRead)) {
    desired_access |= GENERIC_READ;
  }
  if (iree_any_bit_set(access, kNativeFileAccessWrite)) {
    desired_access |= GENERIC_WRITE;
  }
  HANDLE handle =
      CreateFileA(path_string.c_str(), desired_access, FILE_SHARE_READ, NULL,
                  OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
  if (handle == INVALID_HANDLE_VALUE) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "CreateFileA failed");
  }
  primitive = iree_async_primitive_from_win32_handle((uintptr_t)handle);
#else
  int flags = 0;
  if (iree_all_bits_set(access,
                        kNativeFileAccessRead | kNativeFileAccessWrite)) {
    flags = O_RDWR;
  } else if (iree_any_bit_set(access, kNativeFileAccessRead)) {
    flags = O_RDONLY;
  } else {
    flags = O_WRONLY;
  }
  int fd = open(path_string.c_str(), flags);
  if (fd < 0) {
    return iree_make_status(iree_status_code_from_errno(errno), "open failed");
  }
  primitive = iree_async_primitive_from_fd(fd);
#endif  // IREE_PLATFORM_WINDOWS

  iree_status_t status = iree_async_file_import(proactor, primitive, out_file);
  if (!iree_status_is_ok(status)) {
    iree_async_primitive_close(&primitive);
  }
  return status;
}

}  // namespace iree::async::cts
