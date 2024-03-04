// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_EMPTY_H
#define _LIBCUDACXX___ITERATOR_EMPTY_H

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

template <class _Cont>
_LIBCUDACXX_NODISCARD_AFTER_CXX17 _LIBCUDACXX_INLINE_VISIBILITY
constexpr auto empty(const _Cont& __c)
_NOEXCEPT_(noexcept(__c.empty()))
-> decltype        (__c.empty())
{ return            __c.empty(); }

template <class _Tp, size_t _Sz>
_LIBCUDACXX_NODISCARD_AFTER_CXX17 _LIBCUDACXX_INLINE_VISIBILITY
constexpr bool empty(const _Tp (&)[_Sz]) noexcept { return false; }

template <class _Ep>
_LIBCUDACXX_NODISCARD_AFTER_CXX17 _LIBCUDACXX_INLINE_VISIBILITY
constexpr bool empty(initializer_list<_Ep> __il) noexcept { return __il.size() == 0; }

#endif // _LIBCUDACXX_STD_VER > 17

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_EMPTY_H
