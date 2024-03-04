// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CUDA_BARRIER_H
#define _LIBCUDACXX___CUDA_BARRIER_H

#ifndef __cuda_std__
#error "<__cuda/barrier> should only be included in from <cuda/std/barrier>"
#endif // __cuda_std__

#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 700
#  error "CUDA synchronization primitives are only supported for sm_70 and up."
#endif

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

// foward declaration required for memcpy_async, pipeline "sync" defined here
template<thread_scope _Scope>
class pipeline;

template<_CUDA_VSTD::size_t _Alignment>
struct aligned_size_t {
    static constexpr _CUDA_VSTD::size_t align = _Alignment;
    _CUDA_VSTD::size_t value;
    _LIBCUDACXX_INLINE_VISIBILITY
    explicit aligned_size_t(size_t __s) : value(__s) { }
    _LIBCUDACXX_INLINE_VISIBILITY
    operator size_t() const { return value; }
};

// Type only used for logging purpose
enum async_contract_fulfillment
{
    none,
    async
};

template<thread_scope _Sco, class _CompletionF = _CUDA_VSTD::__empty_completion>
class barrier : public _CUDA_VSTD::__barrier_base<_CompletionF, _Sco> {
public:
    barrier() = default;

    barrier(const barrier &) = delete;
    barrier & operator=(const barrier &) = delete;

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
    barrier(_CUDA_VSTD::ptrdiff_t __expected, _CompletionF __completion = _CompletionF())
        : _CUDA_VSTD::__barrier_base<_CompletionF, _Sco>(__expected, __completion) {
    }

    _LIBCUDACXX_INLINE_VISIBILITY
    friend void init(barrier * __b, _CUDA_VSTD::ptrdiff_t __expected) {
#if (_LIBCUDACXX_DEBUG_LEVEL >= 2)
        _LIBCUDACXX_DEBUG_ASSERT(__expected >= 0);
#endif

        new (__b) barrier(__expected);
    }

    _LIBCUDACXX_INLINE_VISIBILITY
    friend void init(barrier * __b, _CUDA_VSTD::ptrdiff_t __expected, _CompletionF __completion) {
#if (_LIBCUDACXX_DEBUG_LEVEL >= 2)
        _LIBCUDACXX_DEBUG_ASSERT(__expected >= 0);
#endif
        new (__b) barrier(__expected, __completion);
    }
};

struct __block_scope_barrier_base {};

_LIBCUDACXX_END_NAMESPACE_CUDA

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE

__device__
inline _CUDA_VSTD::uint64_t * barrier_native_handle(barrier<thread_scope_block> & b);

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template<>
class barrier<thread_scope_block, _CUDA_VSTD::__empty_completion> : public __block_scope_barrier_base {
    using __barrier_base = _CUDA_VSTD::__barrier_base<_CUDA_VSTD::__empty_completion, (int)thread_scope_block>;
    __barrier_base __barrier;

    __device__
    friend inline _CUDA_VSTD::uint64_t * device::_LIBCUDACXX_CUDA_ABI_NAMESPACE::barrier_native_handle(barrier<thread_scope_block> & b);

template<typename _Barrier>
friend class _CUDA_VSTD::__barrier_poll_tester_phase;
template<typename _Barrier>
friend class _CUDA_VSTD::__barrier_poll_tester_parity;

public:
    using arrival_token = typename __barrier_base::arrival_token;
    barrier() = default;

    barrier(const barrier &) = delete;
    barrier & operator=(const barrier &) = delete;

    _LIBCUDACXX_INLINE_VISIBILITY
    barrier(_CUDA_VSTD::ptrdiff_t __expected, _CUDA_VSTD::__empty_completion __completion = _CUDA_VSTD::__empty_completion()) {
        static_assert(_LIBCUDACXX_OFFSET_IS_ZERO(barrier<thread_scope_block>, __barrier), "fatal error: bad barrier layout");
        init(this, __expected, __completion);
    }

    _LIBCUDACXX_INLINE_VISIBILITY
    ~barrier() {
        NV_DISPATCH_TARGET(
            NV_PROVIDES_SM_90, (
                if (__isShared(&__barrier)) {
                    asm volatile ("mbarrier.inval.shared.b64 [%0];"
                        :: "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__barrier)))
                        : "memory");
                }
                else if (__isClusterShared(&__barrier)) {
                    __trap();
                }
            ), NV_PROVIDES_SM_80, (
                if (__isShared(&__barrier)) {
                    asm volatile ("mbarrier.inval.shared.b64 [%0];"
                        :: "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__barrier)))
                        : "memory");
                }
            )
        )
    }

    _LIBCUDACXX_INLINE_VISIBILITY
    friend void init(barrier * __b, _CUDA_VSTD::ptrdiff_t __expected, _CUDA_VSTD::__empty_completion __completion = _CUDA_VSTD::__empty_completion()) {
        NV_DISPATCH_TARGET(
            NV_PROVIDES_SM_90, (
                if (__isShared(&__b->__barrier)) {
                    asm volatile ("mbarrier.init.shared.b64 [%0], %1;"
                        :: "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__b->__barrier))),
                            "r"(static_cast<_CUDA_VSTD::uint32_t>(__expected))
                        : "memory");
                }
                else if (__isClusterShared(&__b->__barrier))
                {
                    __trap();
                }
                else
                {
                    new (&__b->__barrier) __barrier_base(__expected);
                }
            ),
            NV_PROVIDES_SM_80, (
                if (__isShared(&__b->__barrier)) {
                    asm volatile ("mbarrier.init.shared.b64 [%0], %1;"
                        :: "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__b->__barrier))),
                            "r"(static_cast<_CUDA_VSTD::uint32_t>(__expected))
                        : "memory");
                }
                else
                {
                    new (&__b->__barrier) __barrier_base(__expected);
                }
            ), NV_ANY_TARGET, (
                new (&__b->__barrier) __barrier_base(__expected);
            )
        )
    }

    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY
    arrival_token arrive(_CUDA_VSTD::ptrdiff_t __update = 1) {
#if (_LIBCUDACXX_DEBUG_LEVEL >= 2)
        _LIBCUDACXX_DEBUG_ASSERT(__update >= 0);
        _LIBCUDACXX_DEBUG_ASSERT(__expected_unit >=0);
#endif
        arrival_token __token = {};
        NV_DISPATCH_TARGET(
            NV_PROVIDES_SM_90, (
                if (!__isClusterShared(&__barrier)) {
                    return __barrier.arrive(__update);
                }
                else if (!__isShared(&__barrier)) {
                    __trap();
                }

                asm volatile ("mbarrier.arrive.shared.b64 %0, [%1], %2;"
                    : "=l"(__token)
                    : "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__barrier))),
                    "r"(static_cast<_CUDA_VSTD::uint32_t>(__update))
                    : "memory");
            ), NV_PROVIDES_SM_80, (
                if (!__isShared(&__barrier)) {
                    return __barrier.arrive(__update);
                }

                // Need 2 instructions, can't finish barrier with arrive > 1
                if (__update > 1) {
                    asm volatile ("mbarrier.arrive.noComplete.shared.b64 %0, [%1], %2;"
                        : "=l"(__token)
                        : "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__barrier))),
                            "r"(static_cast<_CUDA_VSTD::uint32_t>(__update - 1))
                        : "memory");
                }
                asm volatile ("mbarrier.arrive.shared.b64 %0, [%1];"
                    : "=l"(__token)
                    : "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__barrier)))
                    : "memory");
            ), NV_IS_DEVICE, (
                if (!__isShared(&__barrier)) {
                    return __barrier.arrive(__update);
                }

                unsigned int __mask = __activemask();
                unsigned int __activeA = __match_any_sync(__mask, __update);
                unsigned int __activeB = __match_any_sync(__mask, reinterpret_cast<_CUDA_VSTD::uintptr_t>(&__barrier));
                unsigned int __active = __activeA & __activeB;
                int __inc = __popc(__active) * __update;

                unsigned __laneid;
                asm ("mov.u32 %0, %laneid;" : "=r"(__laneid));
                int __leader = __ffs(__active) - 1;
                // All threads in mask synchronize here, establishing cummulativity to the __leader:
                __syncwarp(__mask);
                if(__leader == __laneid)
                {
                    __token = __barrier.arrive(__inc);
                }
                __token = __shfl_sync(__active, __token, __leader);
            ), NV_IS_HOST, (
                __token = __barrier.arrive(__update);
            )
        )
        return __token;
    }

private:

    _LIBCUDACXX_INLINE_VISIBILITY
    inline bool __test_wait_sm_80(arrival_token __token) const {
        int32_t __ready = 0;
        NV_DISPATCH_TARGET(
            NV_PROVIDES_SM_80, (
                asm volatile ("{\n\t"
                            ".reg .pred p;\n\t"
                            "mbarrier.test_wait.shared.b64 p, [%1], %2;\n\t"
                            "selp.b32 %0, 1, 0, p;\n\t"
                            "}"
                        : "=r"(__ready)
                        : "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__barrier))),
                          "l"(__token)
                        : "memory");
            )
        )
        return __ready;
    }

    // Document de drop > uint32_t for __nanosec on public for APIs
    _LIBCUDACXX_INLINE_VISIBILITY
    bool __try_wait(arrival_token __token) const {
        NV_DISPATCH_TARGET(
            NV_PROVIDES_SM_90, (
                int32_t __ready = 0;
                if (!__isClusterShared(&__barrier)) {
                    return _CUDA_VSTD::__call_try_wait(__barrier, _CUDA_VSTD::move(__token));
                }
                else if (!__isShared(&__barrier)) {
                    __trap();
                }
                asm volatile ("{\n\t"
                        ".reg .pred p;\n\t"
                        "mbarrier.try_wait.shared.b64 p, [%1], %2;\n\t"
                        "selp.b32 %0, 1, 0, p;\n\t"
                        "}"
                    : "=r"(__ready)
                    : "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__barrier))),
                    "l"(__token)
                    : "memory");
                return __ready;
            ), NV_PROVIDES_SM_80, (
                if (!__isShared(&__barrier)) {
                    return _CUDA_VSTD::__call_try_wait(__barrier, _CUDA_VSTD::move(__token));
                }
                return __test_wait_sm_80(__token);
            ), NV_ANY_TARGET, (
                    return _CUDA_VSTD::__call_try_wait(__barrier, _CUDA_VSTD::move(__token));
            )
        )
    }

    // Document de drop > uint32_t for __nanosec on public for APIs
    _LIBCUDACXX_INLINE_VISIBILITY
    bool __try_wait(arrival_token __token, _CUDA_VSTD::chrono::nanoseconds __nanosec) const {
        if (__nanosec.count() < 1) {
            return __try_wait(_CUDA_VSTD::move(__token));
        }

        NV_DISPATCH_TARGET(
            NV_PROVIDES_SM_90, (
                int32_t __ready = 0;
                if (!__isClusterShared(&__barrier)) {
                    return _CUDA_VSTD::__libcpp_thread_poll_with_backoff(
                        _CUDA_VSTD::__barrier_poll_tester_phase<barrier>(this, _CUDA_VSTD::move(__token)),
                        __nanosec);
                }
                else if (!__isShared(&__barrier)) {
                    __trap();
                }

                _CUDA_VSTD::chrono::high_resolution_clock::time_point const __start = _CUDA_VSTD::chrono::high_resolution_clock::now();
                _CUDA_VSTD::chrono::nanoseconds __elapsed;
                do {
                    const _CUDA_VSTD::uint32_t __wait_nsec = static_cast<_CUDA_VSTD::uint32_t>((__nanosec - __elapsed).count());
                    asm volatile ("{\n\t"
                            ".reg .pred p;\n\t"
                            "mbarrier.try_wait.shared.b64 p, [%1], %2, %3;\n\t"
                            "selp.b32 %0, 1, 0, p;\n\t"
                            "}"
                            : "=r"(__ready)
                            : "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__barrier))),
                            "l"(__token)
                            "r"(__wait_nsec)
                            : "memory");
                    __elapsed = _CUDA_VSTD::chrono::high_resolution_clock::now() - __start;
                } while (!__ready && (__nanosec > __elapsed));
                return __ready;
            ), NV_PROVIDES_SM_80, (
                bool __ready = 0;
                if (!__isShared(&__barrier)) {
                    return _CUDA_VSTD::__libcpp_thread_poll_with_backoff(
                        _CUDA_VSTD::__barrier_poll_tester_phase<barrier>(this, _CUDA_VSTD::move(__token)),
                        __nanosec);
                }

                _CUDA_VSTD::chrono::high_resolution_clock::time_point const __start = _CUDA_VSTD::chrono::high_resolution_clock::now();
                do {
                    __ready = __test_wait_sm_80(__token);
                } while (!__ready &&
                        __nanosec > (_CUDA_VSTD::chrono::high_resolution_clock::now() - __start));
                return __ready;
            ), NV_ANY_TARGET, (
                return _CUDA_VSTD::__libcpp_thread_poll_with_backoff(
                        _CUDA_VSTD::__barrier_poll_tester_phase<barrier>(this, _CUDA_VSTD::move(__token)),
                        _CUDA_VSTD::chrono::nanoseconds(__nanosec));
            )
        )
    }

    _LIBCUDACXX_INLINE_VISIBILITY
    inline bool __test_wait_parity_sm_80(bool __phase_parity) const {
        uint16_t __ready = 0;
        NV_DISPATCH_TARGET(
            NV_PROVIDES_SM_80, (
                asm volatile ("{"
                    ".reg .pred %p;"
                    "mbarrier.test_wait.parity.shared.b64 %p, [%1], %2;"
                    "selp.u16 %0, 1, 0, %p;"
                    "}"
                    : "=h"(__ready)
                    : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&__barrier))),
                        "r"(static_cast<uint32_t>(__phase_parity))
                    : "memory");
            )
        )
        return __ready;
    }

    _LIBCUDACXX_INLINE_VISIBILITY
    bool __try_wait_parity(bool __phase_parity)  const {
        NV_DISPATCH_TARGET(
            NV_PROVIDES_SM_90, (
                if (!__isClusterShared(&__barrier)) {
                    return _CUDA_VSTD::__call_try_wait_parity(__barrier, __phase_parity);
                }
                else if (!__isShared(&__barrier)) {
                    __trap();
                }
                int32_t __ready = 0;

                asm volatile ("{\n\t"
                        ".reg .pred p;\n\t"
                        "mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n\t"
                        "selp.b32 %0, 1, 0, p;\n\t"
                        "}"
                        : "=r"(__ready)
                        : "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__barrier))),
                          "r"(static_cast<_CUDA_VSTD::uint32_t>(__phase_parity))
                        :);

                return __ready;
            ), NV_PROVIDES_SM_80, (
                if (!__isShared(&__barrier)) {
                    return _CUDA_VSTD::__call_try_wait_parity(__barrier, __phase_parity);
                }

                return __test_wait_parity_sm_80(__phase_parity);
            ), NV_ANY_TARGET, (
                return _CUDA_VSTD::__call_try_wait_parity(__barrier, __phase_parity);
            )
        )
    }

    _LIBCUDACXX_INLINE_VISIBILITY
    bool __try_wait_parity(bool __phase_parity, _CUDA_VSTD::chrono::nanoseconds __nanosec) const {
        if (__nanosec.count() < 1) {
            return __try_wait_parity(__phase_parity);
        }

        NV_DISPATCH_TARGET(
            NV_PROVIDES_SM_90, (
                int32_t __ready = 0;
                if (!__isClusterShared(&__barrier)) {
                    return _CUDA_VSTD::__libcpp_thread_poll_with_backoff(
                            _CUDA_VSTD::__barrier_poll_tester_parity<barrier>(this, __phase_parity),
                            __nanosec);
                }
                else if (!__isShared(&__barrier)) {
                    __trap();
                }

                _CUDA_VSTD::chrono::high_resolution_clock::time_point const __start = _CUDA_VSTD::chrono::high_resolution_clock::now();
                _CUDA_VSTD::chrono::nanoseconds __elapsed;
                do {
                    const _CUDA_VSTD::uint32_t __wait_nsec = static_cast<_CUDA_VSTD::uint32_t>((__nanosec - __elapsed).count());
                    asm volatile ("{\n\t"
                            ".reg .pred p;\n\t"
                            "mbarrier.try_wait.parity.shared.b64 p, [%1], %2, %3;\n\t"
                            "selp.b32 %0, 1, 0, p;\n\t"
                            "}"
                            : "=r"(__ready)
                            : "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__barrier))),
                              "r"(static_cast<_CUDA_VSTD::uint32_t>(__phase_parity)),
                              "r"(__wait_nsec)
                            : "memory");
                    __elapsed = _CUDA_VSTD::chrono::high_resolution_clock::now() - __start;
                } while (!__ready && (__nanosec > __elapsed));

                return __ready;
            ), NV_PROVIDES_SM_80, (
                bool __ready = 0;
                if (!__isShared(&__barrier)) {
                    return _CUDA_VSTD::__libcpp_thread_poll_with_backoff(
                        _CUDA_VSTD::__barrier_poll_tester_parity<barrier>(this, __phase_parity),
                        __nanosec);
                }

                _CUDA_VSTD::chrono::high_resolution_clock::time_point const __start = _CUDA_VSTD::chrono::high_resolution_clock::now();
                do {
                    __ready = __test_wait_parity_sm_80(__phase_parity);
                } while (!__ready &&
                          __nanosec > (_CUDA_VSTD::chrono::high_resolution_clock::now() - __start));

                return __ready;
            ), NV_ANY_TARGET, (
                return _CUDA_VSTD::__libcpp_thread_poll_with_backoff(
                        _CUDA_VSTD::__barrier_poll_tester_parity<barrier>(this, __phase_parity),
                        __nanosec);
            )
        )
    }

public:
    _LIBCUDACXX_INLINE_VISIBILITY
    void wait(arrival_token && __phase) const {
        _CUDA_VSTD::__libcpp_thread_poll_with_backoff(_CUDA_VSTD::__barrier_poll_tester_phase<barrier>(this, _CUDA_VSTD::move(__phase)));
    }

    _LIBCUDACXX_INLINE_VISIBILITY
    void wait_parity(bool __phase_parity) const {
        _CUDA_VSTD::__libcpp_thread_poll_with_backoff(_CUDA_VSTD::__barrier_poll_tester_parity<barrier>(this, __phase_parity));
    }

    inline _LIBCUDACXX_INLINE_VISIBILITY
    void arrive_and_wait() {
        wait(arrive());
    }

    _LIBCUDACXX_INLINE_VISIBILITY
    void arrive_and_drop() {
        NV_DISPATCH_TARGET(
            NV_PROVIDES_SM_90, (
                if (!__isClusterShared(&__barrier)) {
                    return __barrier.arrive_and_drop();
                }
                else if (!__isShared(&__barrier)) {
                    __trap();
                }

                asm volatile ("mbarrier.arrive_drop.shared.b64 _, [%0];"
                    :: "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__barrier)))
                    : "memory");
            ), NV_PROVIDES_SM_80, (
                // Fallback to slowpath on device
                if (!__isShared(&__barrier)) {
                    __barrier.arrive_and_drop();
                    return;
                }

                asm volatile ("mbarrier.arrive_drop.shared.b64 _, [%0];"
                    :: "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__barrier)))
                    : "memory");
            ), NV_ANY_TARGET, (
                // Fallback to slowpath on device
                __barrier.arrive_and_drop();
            )
        )
    }

    _LIBCUDACXX_INLINE_VISIBILITY
    static constexpr ptrdiff_t max() noexcept {
        return (1 << 20) - 1;
    }

    template<class _Rep, class _Period>
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY
    bool try_wait_for(arrival_token && __token, const _CUDA_VSTD::chrono::duration<_Rep, _Period>& __dur) {
        auto __nanosec = _CUDA_VSTD::chrono::duration_cast<_CUDA_VSTD::chrono::nanoseconds>(__dur);

        return __try_wait(_CUDA_VSTD::move(__token), __nanosec);
    }

    template<class _Clock, class _Duration>
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY
    bool try_wait_until(arrival_token && __token, const _CUDA_VSTD::chrono::time_point<_Clock, _Duration>& __time) {
        return try_wait_for(_CUDA_VSTD::move(__token), (__time - _Clock::now()));
    }

    template<class _Rep, class _Period>
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY
    bool try_wait_parity_for(bool __phase_parity, const _CUDA_VSTD::chrono::duration<_Rep, _Period>& __dur) {
        auto __nanosec = _CUDA_VSTD::chrono::duration_cast<_CUDA_VSTD::chrono::nanoseconds>(__dur);

        return __try_wait_parity(__phase_parity, __nanosec);
    }

    template<class _Clock, class _Duration>
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY
    bool try_wait_parity_until(bool __phase_parity, const _CUDA_VSTD::chrono::time_point<_Clock, _Duration>& __time) {
        return try_wait_parity_for(__phase_parity, (__time - _Clock::now()));
    }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE

__device__
inline _CUDA_VSTD::uint64_t * barrier_native_handle(barrier<thread_scope_block> & b) {
    return reinterpret_cast<_CUDA_VSTD::uint64_t *>(&b.__barrier);
}

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template<>
class barrier<thread_scope_thread, _CUDA_VSTD::__empty_completion> : private barrier<thread_scope_block> {
    using __base = barrier<thread_scope_block>;

public:
    using __base::__base;

    _LIBCUDACXX_INLINE_VISIBILITY
    friend void init(barrier * __b, _CUDA_VSTD::ptrdiff_t __expected, _CUDA_VSTD::__empty_completion __completion = _CUDA_VSTD::__empty_completion()) {
        init(static_cast<__base *>(__b), __expected, __completion);
    }

    using __base::arrive;
    using __base::wait;
    using __base::arrive_and_wait;
    using __base::arrive_and_drop;
    using __base::max;
};

template <typename ... _Ty>
_LIBCUDACXX_INLINE_VISIBILITY constexpr bool __unused(_Ty...) {return true;}

template <typename _Ty>
_LIBCUDACXX_INLINE_VISIBILITY constexpr bool __unused(_Ty&) {return true;}

template<_CUDA_VSTD::size_t _Alignment>
_LIBCUDACXX_INLINE_VISIBILITY
inline void __strided_memcpy(char * __destination, char const * __source, _CUDA_VSTD::size_t __total_size, _CUDA_VSTD::size_t __rank, _CUDA_VSTD::size_t __stride = 1) {
    if (__stride == 1) {
        memcpy(__destination, __source, __total_size);
    }
    else {
        for (_CUDA_VSTD::size_t __offset = __rank * _Alignment; __offset < __total_size; __offset += __stride * _Alignment) {
            memcpy(__destination + __offset, __source + __offset, _Alignment);
        }
    }
}

template<_CUDA_VSTD::size_t _Alignment, bool _Large = (_Alignment > 16)>
struct __memcpy_async_impl {
    __device__ static inline async_contract_fulfillment __copy(char * __destination, char const * __source, _CUDA_VSTD::size_t __total_size, _CUDA_VSTD::size_t __rank, _CUDA_VSTD::size_t __stride) {
        __strided_memcpy<_Alignment>(__destination, __source, __total_size, __rank, __stride);
        return async_contract_fulfillment::none;
    }
};

template<>
struct __memcpy_async_impl<4, false> {
    __device__ static inline async_contract_fulfillment __copy(char * __destination, char const * __source, _CUDA_VSTD::size_t __total_size, _CUDA_VSTD::size_t __rank, _CUDA_VSTD::size_t __stride) {
        // Guard from host visibility when compiling in host only contexts
        NV_IF_TARGET(
            NV_IS_DEVICE, (
                for (_CUDA_VSTD::size_t __offset = __rank * 4; __offset < __total_size; __offset += __stride * 4) {
                    asm volatile ("cp.async.ca.shared.global [%0], [%1], 4, 4;"
                        :: "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(__destination + __offset))),
                            "l"(__source + __offset)
                        : "memory");
                }
            )
        )
        return async_contract_fulfillment::async;
    }
};

template<>
struct __memcpy_async_impl<8, false> {
    __device__ static inline async_contract_fulfillment __copy(char * __destination, char const * __source, _CUDA_VSTD::size_t __total_size, _CUDA_VSTD::size_t __rank, _CUDA_VSTD::size_t __stride) {
        // Guard from host visibility when compiling in host only contexts
        NV_IF_TARGET(
            NV_IS_DEVICE, (
                for (_CUDA_VSTD::size_t __offset = __rank * 8; __offset < __total_size; __offset += __stride * 8) {
                    asm volatile ("cp.async.ca.shared.global [%0], [%1], 8, 8;"
                        :: "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(__destination + __offset))),
                            "l"(__source + __offset)
                        : "memory");
                }
            )
        )
        return async_contract_fulfillment::async;
    }
};

template<>
struct __memcpy_async_impl<16, false> {
    __device__ static inline async_contract_fulfillment __copy(char * __destination, char const * __source, _CUDA_VSTD::size_t __total_size, _CUDA_VSTD::size_t __rank, _CUDA_VSTD::size_t __stride) {
        // Guard from host visibility when compiling in host only contexts
        NV_IF_TARGET(
            NV_IS_DEVICE, (
                for (_CUDA_VSTD::size_t __offset = __rank * 16; __offset < __total_size; __offset += __stride * 16) {
                    asm volatile ("cp.async.cg.shared.global [%0], [%1], 16, 16;"
                        :: "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(__destination + __offset))),
                            "l"(__source + __offset)
                        : "memory");
                }
            )
        )
        return async_contract_fulfillment::async;
    }
};

template<_CUDA_VSTD::size_t _Alignment>
struct __memcpy_async_impl<_Alignment, true> : public __memcpy_async_impl<16, false> { };

struct __memcpy_arrive_on_impl {
    template<thread_scope _Sco, typename _CompF, bool _Is_mbarrier = (_Sco >= thread_scope_block) && _CUDA_VSTD::is_same<_CompF, _CUDA_VSTD::__empty_completion>::value>
    _LIBCUDACXX_INLINE_VISIBILITY static inline void __arrive_on(barrier<_Sco, _CompF> & __barrier, async_contract_fulfillment __is_async) {
          NV_DISPATCH_TARGET(
              NV_PROVIDES_SM_90, (
                  if (_Is_mbarrier && __isClusterShared(&__barrier) && !__isShared(&__barrier)) {
                      __trap();
                  }
              )
          )

          NV_DISPATCH_TARGET(
              NV_PROVIDES_SM_80, (
                  if (__is_async == async_contract_fulfillment::async) {
                      if (_Is_mbarrier && __isShared(&__barrier)) {
                          asm volatile ("cp.async.mbarrier.arrive.shared.b64 [%0];"
                              :: "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__barrier)))
                              : "memory");
                      }
                      else {
                          asm volatile ("cp.async.wait_all;"
                              ::: "memory");
                      }
                  }
              )
          )
    }

    template<thread_scope _Sco>
    _LIBCUDACXX_INLINE_VISIBILITY static inline void __arrive_on(pipeline<_Sco> & __pipeline, async_contract_fulfillment __is_async) {
        // pipeline does not sync on memcpy_async, defeat pipeline purpose otherwise
        __unused(__pipeline);
        __unused(__is_async);
    }
};

template<_CUDA_VSTD::size_t _Native_alignment, typename _Group, typename _Sync>
_LIBCUDACXX_INLINE_VISIBILITY
void inline __memcpy_async_sm_dispatch(
        _Group const & __group, char * __destination, char const * __source,
        _CUDA_VSTD::size_t __size, _Sync & __sync, async_contract_fulfillment & __is_async) {
    // Broken out of __memcpy_async to avoid nesting dispatches
    NV_DISPATCH_TARGET(
        NV_PROVIDES_SM_80,
            __is_async = __memcpy_async_impl<16>::__copy(__destination, __source, __size, __group.thread_rank(), __group.size());
    )
}

template<_CUDA_VSTD::size_t _Native_alignment, typename _Group, typename _Sync>
_LIBCUDACXX_INLINE_VISIBILITY
async_contract_fulfillment inline __memcpy_async(
        _Group const & __group, char * __destination, char const * __source,
        _CUDA_VSTD::size_t __size, _Sync & __sync) {
    async_contract_fulfillment __is_async = async_contract_fulfillment::none;

    NV_DISPATCH_TARGET(
        NV_PROVIDES_SM_80,

        if (__isShared(__destination) && __isGlobal(__source)) {
            if (_Native_alignment < 4) {
                auto __source_address = reinterpret_cast<_CUDA_VSTD::uintptr_t>(__source);
                auto __destination_address = reinterpret_cast<_CUDA_VSTD::uintptr_t>(__destination);

                // Lowest bit set will tell us what the common alignment of the three values is.
                auto _Alignment = __ffs(__source_address | __destination_address | __size);

                switch (_Alignment) {
                    default:
                        __memcpy_async_sm_dispatch<_Native_alignment>(__group, __destination, __source, __size, __sync, __is_async);
                    case 4: __is_async = __memcpy_async_impl<8>::__copy(__destination, __source, __size, __group.thread_rank(), __group.size()); break;
                    case 3: __is_async = __memcpy_async_impl<4>::__copy(__destination, __source, __size, __group.thread_rank(), __group.size()); break;
                    case 2: // fallthrough
                    case 1: __is_async = __memcpy_async_impl<1>::__copy(__destination, __source, __size, __group.thread_rank(), __group.size()); break;
                }
            }
            else {
                __is_async = __memcpy_async_impl<_Native_alignment>::__copy(__destination, __source, __size, __group.thread_rank(), __group.size());
            }
        }
        else
        {
            __strided_memcpy<_Native_alignment>(__destination, __source, __size, __group.thread_rank(), __group.size());
        }

        __memcpy_arrive_on_impl::__arrive_on(__sync, __is_async);
        , NV_ANY_TARGET,
            __strided_memcpy<_Native_alignment>(__destination, __source, __size, __group.thread_rank(), __group.size());
    )

    return __is_async;
}

struct __single_thread_group {
    _LIBCUDACXX_INLINE_VISIBILITY
    void sync() const {}
    _LIBCUDACXX_INLINE_VISIBILITY
    constexpr _CUDA_VSTD::size_t size() const { return 1; };
    _LIBCUDACXX_INLINE_VISIBILITY
    constexpr _CUDA_VSTD::size_t thread_rank() const { return 0; };
};

template<typename _Group, class _Tp, thread_scope _Sco, typename _CompF>
_LIBCUDACXX_INLINE_VISIBILITY
async_contract_fulfillment memcpy_async(_Group const & __group, _Tp * __destination, _Tp const * __source, _CUDA_VSTD::size_t __size, barrier<_Sco, _CompF> & __barrier) {
    // When compiling with NVCC and GCC 4.8, certain user defined types that _are_ trivially copyable are
    // incorrectly classified as not trivially copyable. Remove this assertion to allow for their usage with
    // memcpy_async when compiling with GCC 4.8.
    // FIXME: remove the #if once GCC 4.8 is no longer supported.
#if !defined(_LIBCUDACXX_COMPILER_GCC) || _GNUC_VER > 408
    static_assert(_CUDA_VSTD::is_trivially_copyable<_Tp>::value, "memcpy_async requires a trivially copyable type");
#endif

    return __memcpy_async<alignof(_Tp)>(__group, reinterpret_cast<char *>(__destination), reinterpret_cast<char const *>(__source), __size, __barrier);
}

template<typename _Group, class _Tp, _CUDA_VSTD::size_t _Alignment, thread_scope _Sco, typename _CompF, _CUDA_VSTD::size_t _Larger_alignment = (alignof(_Tp) > _Alignment) ? alignof(_Tp) : _Alignment>
_LIBCUDACXX_INLINE_VISIBILITY
async_contract_fulfillment memcpy_async(_Group const & __group, _Tp * __destination, _Tp const * __source, aligned_size_t<_Alignment> __size, barrier<_Sco, _CompF> & __barrier) {
    // When compiling with NVCC and GCC 4.8, certain user defined types that _are_ trivially copyable are
    // incorrectly classified as not trivially copyable. Remove this assertion to allow for their usage with
    // memcpy_async when compiling with GCC 4.8.
    // FIXME: remove the #if once GCC 4.8 is no longer supported.
#if !defined(_LIBCUDACXX_COMPILER_GCC) || _GNUC_VER > 408
    static_assert(_CUDA_VSTD::is_trivially_copyable<_Tp>::value, "memcpy_async requires a trivially copyable type");
#endif

    return __memcpy_async<_Larger_alignment>(__group, reinterpret_cast<char *>(__destination), reinterpret_cast<char const *>(__source), __size, __barrier);
}

template<class _Tp, typename _Size, thread_scope _Sco, typename _CompF>
_LIBCUDACXX_INLINE_VISIBILITY
async_contract_fulfillment memcpy_async(_Tp * __destination, _Tp const * __source, _Size __size, barrier<_Sco, _CompF> & __barrier) {
    return memcpy_async(__single_thread_group{}, __destination, __source, __size, __barrier);
}

template<typename _Group, thread_scope _Sco, typename _CompF>
_LIBCUDACXX_INLINE_VISIBILITY
async_contract_fulfillment memcpy_async(_Group const & __group, void * __destination, void const * __source, _CUDA_VSTD::size_t __size, barrier<_Sco, _CompF> & __barrier) {
    return __memcpy_async<1>(__group, reinterpret_cast<char *>(__destination), reinterpret_cast<char const *>(__source), __size, __barrier);
}

template<typename _Group, _CUDA_VSTD::size_t _Alignment, thread_scope _Sco, typename _CompF>
_LIBCUDACXX_INLINE_VISIBILITY
async_contract_fulfillment memcpy_async(_Group const & __group, void * __destination, void const * __source, aligned_size_t<_Alignment> __size, barrier<_Sco, _CompF> & __barrier) {
    return __memcpy_async<_Alignment>(__group, reinterpret_cast<char *>(__destination), reinterpret_cast<char const *>(__source), __size, __barrier);
}

template<typename _Size, thread_scope _Sco, typename _CompF>
_LIBCUDACXX_INLINE_VISIBILITY
async_contract_fulfillment memcpy_async(void * __destination, void const * __source, _Size __size, barrier<_Sco, _CompF> & __barrier) {
    return memcpy_async(__single_thread_group{}, __destination, __source, __size, __barrier);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _LIBCUDACXX___CUDA_BARRIER_H
