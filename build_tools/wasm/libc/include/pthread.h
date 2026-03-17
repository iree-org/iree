// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// <pthread.h> for wasm32.
// Wasm threading uses SharedArrayBuffer + Atomics with web workers.
// The pthreads API is provided for source compatibility; implementations
// will map to wasm atomics and JS worker management.

#ifndef IREE_WASM_LIBC_PTHREAD_H_
#define IREE_WASM_LIBC_PTHREAD_H_

#include <stddef.h>
#include <time.h>

// Opaque thread types.
typedef unsigned long pthread_t;
typedef int pthread_once_t;

// Attribute types (opaque structs with enough storage).
typedef struct {
  int __detachstate;
  size_t __stacksize;
} pthread_attr_t;

typedef struct {
  int __type;
} pthread_mutexattr_t;

typedef struct {
  int __clock;
} pthread_condattr_t;

// Synchronization types.
typedef struct {
  int __lock;
} pthread_mutex_t;

typedef struct {
  int __lock;
} pthread_cond_t;

// Initializer macros.
#define PTHREAD_MUTEX_INITIALIZER {0}
#define PTHREAD_COND_INITIALIZER {0}
#define PTHREAD_ONCE_INIT 0

// Detach state.
#define PTHREAD_CREATE_JOINABLE 0
#define PTHREAD_CREATE_DETACHED 1

// Scheduling policy (minimal — wasm has no scheduler control).
#define SCHED_OTHER 0

struct sched_param {
  int sched_priority;
};

// Thread management.
int pthread_create(pthread_t* thread, const pthread_attr_t* attr,
                   void* (*start_routine)(void*), void* arg);
int pthread_join(pthread_t thread, void** retval);
pthread_t pthread_self(void);

// Thread attributes.
int pthread_attr_init(pthread_attr_t* attr);
int pthread_attr_destroy(pthread_attr_t* attr);
int pthread_attr_setdetachstate(pthread_attr_t* attr, int detachstate);
int pthread_attr_setstacksize(pthread_attr_t* attr, size_t stacksize);

// Thread naming.
int pthread_setname_np(pthread_t thread, const char* name);

// Thread scheduling (stubs — no scheduler control on wasm).
int pthread_setschedparam(pthread_t thread, int policy,
                          const struct sched_param* param);
int pthread_getschedparam(pthread_t thread, int* policy,
                          struct sched_param* param);

// Mutex.
int pthread_mutex_init(pthread_mutex_t* mutex, const pthread_mutexattr_t* attr);
int pthread_mutex_destroy(pthread_mutex_t* mutex);
int pthread_mutex_lock(pthread_mutex_t* mutex);
int pthread_mutex_trylock(pthread_mutex_t* mutex);
int pthread_mutex_unlock(pthread_mutex_t* mutex);

// Condition variable.
int pthread_cond_init(pthread_cond_t* cond, const pthread_condattr_t* attr);
int pthread_cond_destroy(pthread_cond_t* cond);
int pthread_cond_signal(pthread_cond_t* cond);
int pthread_cond_broadcast(pthread_cond_t* cond);
int pthread_cond_wait(pthread_cond_t* cond, pthread_mutex_t* mutex);
int pthread_cond_timedwait(pthread_cond_t* cond, pthread_mutex_t* mutex,
                           const struct timespec* abstime);

// Condition variable attributes.
int pthread_condattr_init(pthread_condattr_t* attr);
int pthread_condattr_destroy(pthread_condattr_t* attr);
int pthread_condattr_setclock(pthread_condattr_t* attr, int clock_id);

// Once.
int pthread_once(pthread_once_t* once_control, void (*init_routine)(void));

#endif  // IREE_WASM_LIBC_PTHREAD_H_
