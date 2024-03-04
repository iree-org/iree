// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_DATA_H
#define _LIBCUDACXX___ITERATOR_DATA_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../cstddef"
#include "../initializer_list"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 11

template <class _Cont> constexpr
_LIBCUDACXX_INLINE_VISIBILITY
auto data(_Cont& __c)
_NOEXCEPT_(noexcept(__c.data()))
-> decltype        (__c.data())
{ return            __c.data(); }

template <class _Cont> constexpr
_LIBCUDACXX_INLINE_VISIBILITY
auto data(const _Cont& __c)
_NOEXCEPT_(noexcept(__c.data()))
-> decltype        (__c.data())
{ return            __c.data(); }

template <class _Tp, size_t _Sz>
_LIBCUDACXX_INLINE_VISIBILITY
constexpr _Tp* data(_Tp (&__array)[_Sz]) noexcept { return __array; }

template <class _Ep>
_LIBCUDACXX_INLINE_VISIBILITY
constexpr const _Ep* data(initializer_list<_Ep> __il) noexcept { return __il.begin(); }

#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_DATA_H
