// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// <stdatomic.h> for wasm32, delegating to clang builtins.

#ifndef IREE_WASM_LIBC_STDATOMIC_H_
#define IREE_WASM_LIBC_STDATOMIC_H_

#include <stddef.h>
#include <stdint.h>

// Memory ordering.
typedef enum {
  memory_order_relaxed = __ATOMIC_RELAXED,
  memory_order_consume = __ATOMIC_CONSUME,
  memory_order_acquire = __ATOMIC_ACQUIRE,
  memory_order_release = __ATOMIC_RELEASE,
  memory_order_acq_rel = __ATOMIC_ACQ_REL,
  memory_order_seq_cst = __ATOMIC_SEQ_CST,
} memory_order;

// Atomic flag type.
typedef struct {
  _Atomic(_Bool) value;
} atomic_flag;

#define ATOMIC_FLAG_INIT {0}

// Atomic types.
typedef _Atomic(_Bool) atomic_bool;
typedef _Atomic(char) atomic_char;
typedef _Atomic(signed char) atomic_schar;
typedef _Atomic(unsigned char) atomic_uchar;
typedef _Atomic(short) atomic_short;
typedef _Atomic(unsigned short) atomic_ushort;
typedef _Atomic(int) atomic_int;
typedef _Atomic(unsigned int) atomic_uint;
typedef _Atomic(long) atomic_long;
typedef _Atomic(unsigned long) atomic_ulong;
typedef _Atomic(long long) atomic_llong;
typedef _Atomic(unsigned long long) atomic_ullong;
typedef _Atomic(int_least8_t) atomic_int_least8_t;
typedef _Atomic(uint_least8_t) atomic_uint_least8_t;
typedef _Atomic(int_least16_t) atomic_int_least16_t;
typedef _Atomic(uint_least16_t) atomic_uint_least16_t;
typedef _Atomic(int_least32_t) atomic_int_least32_t;
typedef _Atomic(uint_least32_t) atomic_uint_least32_t;
typedef _Atomic(int_least64_t) atomic_int_least64_t;
typedef _Atomic(uint_least64_t) atomic_uint_least64_t;
typedef _Atomic(int_fast8_t) atomic_int_fast8_t;
typedef _Atomic(uint_fast8_t) atomic_uint_fast8_t;
typedef _Atomic(int_fast16_t) atomic_int_fast16_t;
typedef _Atomic(uint_fast16_t) atomic_uint_fast16_t;
typedef _Atomic(int_fast32_t) atomic_int_fast32_t;
typedef _Atomic(uint_fast32_t) atomic_uint_fast32_t;
typedef _Atomic(int_fast64_t) atomic_int_fast64_t;
typedef _Atomic(uint_fast64_t) atomic_uint_fast64_t;
typedef _Atomic(intptr_t) atomic_intptr_t;
typedef _Atomic(uintptr_t) atomic_uintptr_t;
typedef _Atomic(size_t) atomic_size_t;
typedef _Atomic(ptrdiff_t) atomic_ptrdiff_t;
typedef _Atomic(intmax_t) atomic_intmax_t;
typedef _Atomic(uintmax_t) atomic_uintmax_t;

// Lock-free queries.
#define ATOMIC_BOOL_LOCK_FREE __CLANG_ATOMIC_BOOL_LOCK_FREE
#define ATOMIC_CHAR_LOCK_FREE __CLANG_ATOMIC_CHAR_LOCK_FREE
#define ATOMIC_SHORT_LOCK_FREE __CLANG_ATOMIC_SHORT_LOCK_FREE
#define ATOMIC_INT_LOCK_FREE __CLANG_ATOMIC_INT_LOCK_FREE
#define ATOMIC_LONG_LOCK_FREE __CLANG_ATOMIC_LONG_LOCK_FREE
#define ATOMIC_LLONG_LOCK_FREE __CLANG_ATOMIC_LLONG_LOCK_FREE
#define ATOMIC_POINTER_LOCK_FREE __CLANG_ATOMIC_POINTER_LOCK_FREE

// Atomic variable initializer.
#define ATOMIC_VAR_INIT(value) (value)

// Generic atomic operations (C11 _Generic-based).
#define atomic_init(obj, value) __c11_atomic_init(obj, value)
#define atomic_store(obj, desired) \
  __c11_atomic_store(obj, desired, memory_order_seq_cst)
#define atomic_store_explicit(obj, desired, order) \
  __c11_atomic_store(obj, desired, order)
#define atomic_load(obj) __c11_atomic_load(obj, memory_order_seq_cst)
#define atomic_load_explicit(obj, order) __c11_atomic_load(obj, order)
#define atomic_exchange(obj, desired) \
  __c11_atomic_exchange(obj, desired, memory_order_seq_cst)
#define atomic_exchange_explicit(obj, desired, order) \
  __c11_atomic_exchange(obj, desired, order)
#define atomic_compare_exchange_strong(obj, expected, desired) \
  __c11_atomic_compare_exchange_strong(                        \
      obj, expected, desired, memory_order_seq_cst, memory_order_seq_cst)
#define atomic_compare_exchange_strong_explicit(obj, expected, desired, \
                                                success, failure)       \
  __c11_atomic_compare_exchange_strong(obj, expected, desired, success, failure)
#define atomic_compare_exchange_weak(obj, expected, desired) \
  __c11_atomic_compare_exchange_weak(                        \
      obj, expected, desired, memory_order_seq_cst, memory_order_seq_cst)
#define atomic_compare_exchange_weak_explicit(obj, expected, desired, success, \
                                              failure)                         \
  __c11_atomic_compare_exchange_weak(obj, expected, desired, success, failure)
#define atomic_fetch_add(obj, arg) \
  __c11_atomic_fetch_add(obj, arg, memory_order_seq_cst)
#define atomic_fetch_add_explicit(obj, arg, order) \
  __c11_atomic_fetch_add(obj, arg, order)
#define atomic_fetch_sub(obj, arg) \
  __c11_atomic_fetch_sub(obj, arg, memory_order_seq_cst)
#define atomic_fetch_sub_explicit(obj, arg, order) \
  __c11_atomic_fetch_sub(obj, arg, order)
#define atomic_fetch_or(obj, arg) \
  __c11_atomic_fetch_or(obj, arg, memory_order_seq_cst)
#define atomic_fetch_or_explicit(obj, arg, order) \
  __c11_atomic_fetch_or(obj, arg, order)
#define atomic_fetch_xor(obj, arg) \
  __c11_atomic_fetch_xor(obj, arg, memory_order_seq_cst)
#define atomic_fetch_xor_explicit(obj, arg, order) \
  __c11_atomic_fetch_xor(obj, arg, order)
#define atomic_fetch_and(obj, arg) \
  __c11_atomic_fetch_and(obj, arg, memory_order_seq_cst)
#define atomic_fetch_and_explicit(obj, arg, order) \
  __c11_atomic_fetch_and(obj, arg, order)

// Fence and signal fence.
#define atomic_thread_fence(order) __c11_atomic_thread_fence(order)
#define atomic_signal_fence(order) __c11_atomic_signal_fence(order)

// Atomic flag operations.
#define atomic_flag_test_and_set(flag) \
  __c11_atomic_exchange(&(flag)->value, 1, memory_order_seq_cst)
#define atomic_flag_test_and_set_explicit(flag, order) \
  __c11_atomic_exchange(&(flag)->value, 1, order)
#define atomic_flag_clear(flag) \
  __c11_atomic_store(&(flag)->value, 0, memory_order_seq_cst)
#define atomic_flag_clear_explicit(flag, order) \
  __c11_atomic_store(&(flag)->value, 0, order)

// Kill dependency.
#define kill_dependency(y) (y)

// Lock-free property query.
#define atomic_is_lock_free(obj) __c11_atomic_is_lock_free(sizeof(*(obj)))

#endif  // IREE_WASM_LIBC_STDATOMIC_H_
