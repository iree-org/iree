// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/primitive.h"

#if defined(IREE_ASYNC_HAVE_FD)
#include <errno.h>
#include <unistd.h>
#endif  // IREE_ASYNC_HAVE_FD

#if defined(IREE_ASYNC_HAVE_WIN32_HANDLE)
#include <windows.h>
#endif  // IREE_ASYNC_HAVE_WIN32_HANDLE

#if defined(IREE_ASYNC_HAVE_MACH_PORT)
#include <mach/mach.h>
#endif  // IREE_ASYNC_HAVE_MACH_PORT

iree_status_t iree_async_primitive_dup(iree_async_primitive_t primitive,
                                       iree_async_primitive_t* out_primitive) {
  *out_primitive = iree_async_primitive_none();
  switch (primitive.type) {
    case IREE_ASYNC_PRIMITIVE_TYPE_NONE:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "cannot duplicate a NONE primitive");

#if defined(IREE_ASYNC_HAVE_FD)
    case IREE_ASYNC_PRIMITIVE_TYPE_FD: {
      int new_fd = dup(primitive.value.fd);
      if (IREE_UNLIKELY(new_fd == -1)) {
        return iree_make_status(iree_status_code_from_errno(errno),
                                "dup(%d) failed", primitive.value.fd);
      }
      *out_primitive = iree_async_primitive_from_fd(new_fd);
      return iree_ok_status();
    }
#endif  // IREE_ASYNC_HAVE_FD

#if defined(IREE_ASYNC_HAVE_WIN32_HANDLE)
    case IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE: {
      HANDLE new_handle = NULL;
      if (IREE_UNLIKELY(!DuplicateHandle(GetCurrentProcess(),
                                         (HANDLE)primitive.value.win32_handle,
                                         GetCurrentProcess(), &new_handle, 0,
                                         FALSE, DUPLICATE_SAME_ACCESS))) {
        return iree_make_status(
            iree_status_code_from_win32_error(GetLastError()),
            "DuplicateHandle failed");
      }
      *out_primitive =
          iree_async_primitive_from_win32_handle((uintptr_t)new_handle);
      return iree_ok_status();
    }
#endif  // IREE_ASYNC_HAVE_WIN32_HANDLE

#if defined(IREE_ASYNC_HAVE_MACH_PORT)
    case IREE_ASYNC_PRIMITIVE_TYPE_MACH_PORT: {
      // Increment the send right reference count. The caller gets an
      // independent send right that must be deallocated separately.
      kern_return_t kr = mach_port_mod_refs(
          mach_task_self(), (mach_port_t)primitive.value.mach_port,
          MACH_PORT_RIGHT_SEND, 1);
      if (IREE_UNLIKELY(kr != KERN_SUCCESS)) {
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "mach_port_mod_refs failed (%d)", kr);
      }
      *out_primitive =
          iree_async_primitive_from_mach_port(primitive.value.mach_port);
      return iree_ok_status();
    }
#endif  // IREE_ASYNC_HAVE_MACH_PORT

    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported primitive type %d for duplication",
                              (int)primitive.type);
  }
}

void iree_async_primitive_close(iree_async_primitive_t* primitive) {
  if (!primitive) return;
  switch (primitive->type) {
    case IREE_ASYNC_PRIMITIVE_TYPE_NONE:
      break;

#if defined(IREE_ASYNC_HAVE_FD)
    case IREE_ASYNC_PRIMITIVE_TYPE_FD:
      close(primitive->value.fd);
      break;
#endif  // IREE_ASYNC_HAVE_FD

#if defined(IREE_ASYNC_HAVE_WIN32_HANDLE)
    case IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE:
      CloseHandle((HANDLE)primitive->value.win32_handle);
      break;
#endif  // IREE_ASYNC_HAVE_WIN32_HANDLE

#if defined(IREE_ASYNC_HAVE_MACH_PORT)
    case IREE_ASYNC_PRIMITIVE_TYPE_MACH_PORT:
      mach_port_deallocate(mach_task_self(),
                           (mach_port_t)primitive->value.mach_port);
      break;
#endif  // IREE_ASYNC_HAVE_MACH_PORT

    default:
      break;
  }
  *primitive = iree_async_primitive_none();
}
