// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CUDA_ATOMIC_H
#define _LIBCUDACXX___CUDA_ATOMIC_H

#ifndef __cuda_std__
#error "<__cuda/atomic> should only be included in from <cuda/std/atomic>"
#endif // __cuda_std__

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

using std::__detail::thread_scope;
using std::__detail::thread_scope_system;
using std::__detail::thread_scope_device;
using std::__detail::thread_scope_block;
using std::__detail::thread_scope_thread;

namespace __detail {
using std::__detail::__thread_scope_block_tag;
using std::__detail::__thread_scope_device_tag;
using std::__detail::__thread_scope_system_tag;
}

using memory_order = std::memory_order;

constexpr memory_order memory_order_relaxed = std::memory_order_relaxed;
constexpr memory_order memory_order_consume = std::memory_order_consume;
constexpr memory_order memory_order_acquire = std::memory_order_acquire;
constexpr memory_order memory_order_release = std::memory_order_release;
constexpr memory_order memory_order_acq_rel = std::memory_order_acq_rel;
constexpr memory_order memory_order_seq_cst = std::memory_order_seq_cst;

// atomic<T>

template <class _Tp, thread_scope _Sco = thread_scope::thread_scope_system>
struct atomic
    : public std::__atomic_base<_Tp, _Sco>
{
    typedef std::__atomic_base<_Tp, _Sco> __base;

    constexpr atomic() noexcept = default;
    __host__ __device__
    constexpr atomic(_Tp __d) noexcept : __base(__d) {}

    __host__ __device__
    _Tp operator=(_Tp __d) volatile noexcept
        {__base::store(__d); return __d;}
    __host__ __device__
    _Tp operator=(_Tp __d) noexcept
        {__base::store(__d); return __d;}

    __host__ __device__
    _Tp fetch_max(const _Tp & __op, memory_order __m = memory_order_seq_cst) volatile noexcept
    {
        return std::__detail::__cxx_atomic_fetch_max(&this->__a_, __op, __m);
    }

    __host__ __device__
    _Tp fetch_min(const _Tp & __op, memory_order __m = memory_order_seq_cst) volatile noexcept
    {
        return std::__detail::__cxx_atomic_fetch_min(&this->__a_, __op, __m);
    }
};

// atomic<T*>

template <class _Tp, thread_scope _Sco>
struct atomic<_Tp*, _Sco>
    : public std::__atomic_base<_Tp*, _Sco>
{
    typedef std::__atomic_base<_Tp*, _Sco> __base;

    constexpr atomic() noexcept = default;
    __host__ __device__
    constexpr atomic(_Tp* __d) noexcept : __base(__d) {}

    __host__ __device__
    _Tp* operator=(_Tp* __d) volatile noexcept
        {__base::store(__d); return __d;}
    __host__ __device__
    _Tp* operator=(_Tp* __d) noexcept
        {__base::store(__d); return __d;}

    __host__ __device__
    _Tp* fetch_add(ptrdiff_t __op, memory_order __m = memory_order_seq_cst)
                                                                        volatile noexcept
        {return __cxx_atomic_fetch_add(&this->__a_, __op, __m);}
    __host__ __device__
    _Tp* fetch_add(ptrdiff_t __op, memory_order __m = memory_order_seq_cst) noexcept
        {return __cxx_atomic_fetch_add(&this->__a_, __op, __m);}
    __host__ __device__
    _Tp* fetch_sub(ptrdiff_t __op, memory_order __m = memory_order_seq_cst)
                                                                        volatile noexcept
        {return __cxx_atomic_fetch_sub(&this->__a_, __op, __m);}
    __host__ __device__
    _Tp* fetch_sub(ptrdiff_t __op, memory_order __m = memory_order_seq_cst) noexcept
        {return __cxx_atomic_fetch_sub(&this->__a_, __op, __m);}

    __host__ __device__
    _Tp* operator++(int) volatile noexcept            {return fetch_add(1);}
    __host__ __device__
    _Tp* operator++(int) noexcept                     {return fetch_add(1);}
    __host__ __device__
    _Tp* operator--(int) volatile noexcept            {return fetch_sub(1);}
    __host__ __device__
    _Tp* operator--(int) noexcept                     {return fetch_sub(1);}
    __host__ __device__
    _Tp* operator++() volatile noexcept               {return fetch_add(1) + 1;}
    __host__ __device__
    _Tp* operator++() noexcept                        {return fetch_add(1) + 1;}
    __host__ __device__
    _Tp* operator--() volatile noexcept               {return fetch_sub(1) - 1;}
    __host__ __device__
    _Tp* operator--() noexcept                        {return fetch_sub(1) - 1;}
    __host__ __device__
    _Tp* operator+=(ptrdiff_t __op) volatile noexcept {return fetch_add(__op) + __op;}
    __host__ __device__
    _Tp* operator+=(ptrdiff_t __op) noexcept          {return fetch_add(__op) + __op;}
    __host__ __device__
    _Tp* operator-=(ptrdiff_t __op) volatile noexcept {return fetch_sub(__op) - __op;}
    __host__ __device__
    _Tp* operator-=(ptrdiff_t __op) noexcept          {return fetch_sub(__op) - __op;}
};

// atomic_ref<T>

template <class _Tp, thread_scope _Sco = thread_scope::thread_scope_system>
struct atomic_ref
    : public std::__atomic_base_ref<_Tp, _Sco>
{
    typedef std::__atomic_base_ref<_Tp, _Sco> __base;

    __host__ __device__
    constexpr atomic_ref(_Tp& __d) noexcept : __base(__d) {}

    __host__ __device__
    _Tp operator=(_Tp __d) const volatile noexcept
        {__base::store(__d); return __d;}
    __host__ __device__
    _Tp operator=(_Tp __d) const noexcept
        {__base::store(__d); return __d;}

    __host__ __device__
    _Tp fetch_max(const _Tp & __op, memory_order __m = memory_order_seq_cst) const volatile noexcept
    {
        return std::__detail::__cxx_atomic_fetch_max(&this->__a_, __op, __m);
    }

    __host__ __device__
    _Tp fetch_min(const _Tp & __op, memory_order __m = memory_order_seq_cst) const volatile noexcept
    {
        return std::__detail::__cxx_atomic_fetch_min(&this->__a_, __op, __m);
    }
};

// atomic_ref<T*>

template <class _Tp, thread_scope _Sco>
struct atomic_ref<_Tp*, _Sco>
    : public std::__atomic_base_ref<_Tp*, _Sco>
{
    typedef std::__atomic_base_ref<_Tp*, _Sco> __base;

    __host__ __device__
    constexpr atomic_ref(_Tp*& __d) noexcept : __base(__d) {}

    __host__ __device__
    _Tp* operator=(_Tp* __d) const volatile noexcept
        {__base::store(__d); return __d;}
    __host__ __device__
    _Tp* operator=(_Tp* __d) const noexcept
        {__base::store(__d); return __d;}

    __host__ __device__
    _Tp* fetch_add(ptrdiff_t __op,
                   memory_order __m = memory_order_seq_cst) const volatile noexcept
        {return __cxx_atomic_fetch_add(&this->__a_, __op, __m);}
    __host__ __device__
    _Tp* fetch_add(ptrdiff_t __op,
                   memory_order __m = memory_order_seq_cst) const noexcept
        {return __cxx_atomic_fetch_add(&this->__a_, __op, __m);}
    __host__ __device__
    _Tp* fetch_sub(ptrdiff_t __op,
                   memory_order __m = memory_order_seq_cst) const volatile noexcept
        {return __cxx_atomic_fetch_sub(&this->__a_, __op, __m);}
    __host__ __device__
    _Tp* fetch_sub(ptrdiff_t __op,
                   memory_order __m = memory_order_seq_cst) const noexcept
        {return __cxx_atomic_fetch_sub(&this->__a_, __op, __m);}

    __host__ __device__
    _Tp* operator++(int) const volatile noexcept            {return fetch_add(1);}
    __host__ __device__
    _Tp* operator++(int) const noexcept                     {return fetch_add(1);}
    __host__ __device__
    _Tp* operator--(int) const volatile noexcept            {return fetch_sub(1);}
    __host__ __device__
    _Tp* operator--(int) const noexcept                     {return fetch_sub(1);}
    __host__ __device__
    _Tp* operator++() const volatile noexcept               {return fetch_add(1) + 1;}
    __host__ __device__
    _Tp* operator++() const noexcept                        {return fetch_add(1) + 1;}
    __host__ __device__
    _Tp* operator--() const volatile noexcept               {return fetch_sub(1) - 1;}
    __host__ __device__
    _Tp* operator--() const noexcept                        {return fetch_sub(1) - 1;}
    __host__ __device__
    _Tp* operator+=(ptrdiff_t __op) const volatile noexcept {return fetch_add(__op) + __op;}
    __host__ __device__
    _Tp* operator+=(ptrdiff_t __op) const noexcept          {return fetch_add(__op) + __op;}
    __host__ __device__
    _Tp* operator-=(ptrdiff_t __op) const volatile noexcept {return fetch_sub(__op) - __op;}
    __host__ __device__
    _Tp* operator-=(ptrdiff_t __op) const noexcept          {return fetch_sub(__op) - __op;}
};

inline __host__ __device__ void atomic_thread_fence(memory_order __m, thread_scope _Scope = thread_scope::thread_scope_system) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            switch(_Scope) {
            case thread_scope::thread_scope_system:
                std::__detail::__atomic_thread_fence_cuda((int)__m, __detail::__thread_scope_system_tag());
                break;
            case thread_scope::thread_scope_device:
                std::__detail::__atomic_thread_fence_cuda((int)__m, __detail::__thread_scope_device_tag());
                break;
            case thread_scope::thread_scope_block:
                std::__detail::__atomic_thread_fence_cuda((int)__m, __detail::__thread_scope_block_tag());
                break;
            // Atomics scoped to themselves do not require fencing
            case thread_scope::thread_scope_thread:
                break;
            }
        ),
        NV_IS_HOST, (
            (void) _Scope;
            std::atomic_thread_fence(__m);
        )
    )
}

inline __host__ __device__ void atomic_signal_fence(memory_order __m) {
    std::atomic_signal_fence(__m);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _LIBCUDACXX___CUDA_ATOMIC_H
