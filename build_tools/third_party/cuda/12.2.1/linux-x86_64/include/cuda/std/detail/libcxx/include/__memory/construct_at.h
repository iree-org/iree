// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MEMORY_CONSTRUCT_AT_H
#define _LIBCUDACXX___MEMORY_CONSTRUCT_AT_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__assert"
#include "../__iterator/access.h"
#include "../__memory/addressof.h"
#include "../__memory/voidify.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_array.h"
#include "../__type_traits/is_constant_evaluated.h"
#include "../__utility/forward.h"
#include "../__utility/move.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#if defined(__cuda_std__) && _LIBCUDACXX_STD_VER > 17 // need to backfill ::std::construct_at
#ifndef _LIBCUDACXX_COMPILER_NVRTC
#include <memory>
#endif // _LIBCUDACXX_COMPILER_NVRTC

#ifndef __cpp_lib_constexpr_dynamic_alloc
namespace std {
template <class _Tp, class... _Args, class = decltype(::new(_CUDA_VSTD::declval<void*>()) _Tp(_CUDA_VSTD::declval<_Args>()...))>
_LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp* construct_at(_Tp* __location, _Args&&... __args) {
#if defined(_LIBCUDACXX_ADDRESSOF)
  return ::new (_CUDA_VSTD::__voidify(*__location)) _Tp(_CUDA_VSTD::forward<_Args>(__args)...);
#else
  return ::new (const_cast<void*>(static_cast<const volatile void*>(__location))) _Tp(_CUDA_VSTD::forward<_Args>(__args)...);
#endif
}
} // namespace std
#endif // __cpp_lib_constexpr_dynamic_alloc
#endif // __cuda_std__ && _LIBCUDACXX_STD_VER > 17

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// construct_at

#if _LIBCUDACXX_STD_VER > 17

#if defined(__CUDACC__)
#pragma nv_exec_check_disable
#endif
template <class _Tp, class... _Args, class = decltype(::new(_CUDA_VSTD::declval<void*>()) _Tp(_CUDA_VSTD::declval<_Args>()...))>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 _Tp* construct_at(_Tp* __location, _Args&&... __args) {
  _LIBCUDACXX_ASSERT(__location != nullptr, "null pointer given to construct_at");
#if defined(__cuda_std__)
  // Need to go through `std::construct_at` as that is the explicitly blessed function
  if (__libcpp_is_constant_evaluated()) {
    return ::std::construct_at(__location, _CUDA_VSTD::forward<_Args>(__args)...);
  }
#endif // __cuda_std__
  return ::new (_CUDA_VSTD::__voidify(*__location)) _Tp(_CUDA_VSTD::forward<_Args>(__args)...);
}

#endif // _LIBCUDACXX_STD_VER > 17

#if defined(__CUDACC__)
#pragma nv_exec_check_disable
#endif
template <class _Tp, class... _Args, class = decltype(::new(_CUDA_VSTD::declval<void*>()) _Tp(_CUDA_VSTD::declval<_Args>()...))>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 _Tp* __construct_at(_Tp* __location, _Args&&... __args) {
  _LIBCUDACXX_ASSERT(__location != nullptr, "null pointer given to construct_at");
#if defined(__cuda_std__) && _LIBCUDACXX_STD_VER > 17
  // Need to go through `std::construct_at` as that is the explicitly blessed function
  if (__libcpp_is_constant_evaluated()) {
    return ::std::construct_at(__location, _CUDA_VSTD::forward<_Args>(__args)...);
  }
#endif // __cuda_std__ && _LIBCUDACXX_STD_VER > 17
  return ::new (_CUDA_VSTD::__voidify(*__location)) _Tp(_CUDA_VSTD::forward<_Args>(__args)...);
}

// destroy_at

// The internal functions are available regardless of the language version (with the exception of the `__destroy_at`
// taking an array).
template <class _ForwardIterator>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
_ForwardIterator __destroy(_ForwardIterator, _ForwardIterator);

template <class _Tp, __enable_if_t<!is_array<_Tp>::value, int> = 0>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
void __destroy_at(_Tp* __loc) {
    _LIBCUDACXX_ASSERT(__loc != nullptr, "null pointer given to destroy_at");
    __loc->~_Tp();
}

#if _LIBCUDACXX_STD_VER > 17
template <class _Tp, __enable_if_t<is_array<_Tp>::value, int> = 0>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
void __destroy_at(_Tp* __loc) {
    _LIBCUDACXX_ASSERT(__loc != nullptr, "null pointer given to destroy_at");
    _CUDA_VSTD::__destroy(_CUDA_VSTD::begin(*__loc), _CUDA_VSTD::end(*__loc));
}
#endif

template <class _ForwardIterator>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
_ForwardIterator __destroy(_ForwardIterator __first, _ForwardIterator __last) {
    for (; __first != __last; ++__first)
        _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(*__first));
    return __first;
}

template <class _BidirectionalIterator>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
_BidirectionalIterator __reverse_destroy(_BidirectionalIterator __first, _BidirectionalIterator __last) {
    while (__last != __first) {
        --__last;
        _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(*__last));
    }
    return __last;
}

#if _LIBCUDACXX_STD_VER > 14

template <class _Tp, enable_if_t<!is_array_v<_Tp>, int> = 0>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
void destroy_at(_Tp* __loc) {
  _LIBCUDACXX_ASSERT(__loc != nullptr, "null pointer given to destroy_at");
  __loc->~_Tp();
}

#if _LIBCUDACXX_STD_VER > 17
template <class _Tp, enable_if_t<is_array_v<_Tp>, int> = 0>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
void destroy_at(_Tp* __loc) {
  _CUDA_VSTD::__destroy_at(__loc);
}
#endif // _LIBCUDACXX_STD_VER > 17

template <class _ForwardIterator>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
void destroy(_ForwardIterator __first, _ForwardIterator __last) {
  (void)_CUDA_VSTD::__destroy(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last));
}

template <class _ForwardIterator, class _Size>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
_ForwardIterator destroy_n(_ForwardIterator __first, _Size __n) {
    for (; __n > 0; (void)++__first, --__n)
        _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(*__first));
    return __first;
}

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MEMORY_CONSTRUCT_AT_H
