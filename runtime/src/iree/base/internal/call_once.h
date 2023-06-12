// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_CALL_ONCE_H_
#define IREE_BASE_INTERNAL_CALL_ONCE_H_

#include <stddef.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// iree_call_once
//==============================================================================
// Emulates the C11 call_once feature as few seem to have it.
// https://en.cppreference.com/w/c/thread/call_once

#if defined(__has_include)
#if __has_include(<thread.h>)
#define IREE_HAS_C11_THREAD_H 1
#endif
#endif

#if defined(IREE_HAS_C11_THREAD_H)

// Always prefer the C11 header if present.
#include <thread.h>
#define IREE_ONCE_FLAG_INIT ONCE_FLAG_INIT
#define iree_once_flag ONCE_FLAG
#define iree_call_once call_once

#elif defined(IREE_PLATFORM_WINDOWS)

// Windows fallback using the native InitOnceExecuteOnce:
// https://docs.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-initonceexecuteonce

// Expands to a value that can be used to initialize an object of type
// iree_once_flag.
#define IREE_ONCE_FLAG_INIT INIT_ONCE_STATIC_INIT

// Complete object type capable of holding a flag used by iree_call_once.
typedef INIT_ONCE iree_once_flag;

typedef struct {
  void (*func)(void);
} iree_call_once_impl_params_t;
static BOOL CALLBACK iree_call_once_callback_impl(PINIT_ONCE InitOnce,
                                                  PVOID Parameter,
                                                  PVOID* Context) {
  // https://docs.microsoft.com/en-us/windows/win32/api/synchapi/nc-synchapi-pinit_once_fn
  iree_call_once_impl_params_t* param =
      (iree_call_once_impl_params_t*)Parameter;
  (param->func)();
  ((void)InitOnce);
  ((void)Context);  // suppress warning
  return TRUE;
}

// Calls |func| exactly once, even if invoked from several threads.
// The completion of the function synchronizes with all previous or subsequent
// calls to call_once with the same flag variable.
static inline void iree_call_once(iree_once_flag* flag, void (*func)(void)) {
  iree_call_once_impl_params_t param;
  param.func = func;
  InitOnceExecuteOnce(flag, iree_call_once_callback_impl, (PVOID)&param, NULL);
}

#elif IREE_SYNCHRONIZATION_DISABLE_UNSAFE

// No-op when the thread control is disabled.
#define IREE_ONCE_FLAG_INIT 1
#define iree_once_flag uint32_t
static inline void iree_call_once(iree_once_flag* flag, void (*func)(void)) {}

#else

// Fallback using pthread_once:
// https://pubs.opengroup.org/onlinepubs/007908775/xsh/pthread_once.html

#include <pthread.h>

// Expands to a value that can be used to initialize an object of type
// iree_once_flag.
#define IREE_ONCE_FLAG_INIT PTHREAD_ONCE_INIT

// Complete object type capable of holding a flag used by iree_call_once.
typedef pthread_once_t iree_once_flag;

// Calls |func| exactly once, even if invoked from several threads.
// The completion of the function synchronizes with all previous or subsequent
// calls to call_once with the same flag variable.
static inline void iree_call_once(iree_once_flag* flag, void (*func)(void)) {
  pthread_once(flag, func);
}

#endif  // IREE_HAS_C11_THREAD_H / fallbacks

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_CALL_ONCE_H_
